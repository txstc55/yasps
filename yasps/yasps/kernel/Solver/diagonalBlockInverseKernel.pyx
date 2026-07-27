from yasps.backend import gpuarray
import os
from yasps.backend import cuda
from typing import Set, List
import ctypes
import numpy as np
from yasps.helper import timed
from yasps.context import context
import time
from yasps.backend import is_metal
from pathlib import Path
import hashlib

# the diagonalBlockInverseKernel
# will be responsible for computing the explicit inverse of the diagonal blocks
class diagonalBlockInverseKernel:
  def __init__(
    self,
    unique_attribute_sizes: Set[int],
    diagonal_blocks_start: List[int],
    diagonal_blocks_count: List[int],
    diagonal_block_sizes: List[int],
    num_attributes: int
  ):
    self.__unique_attribute_sizes: Set[int] = unique_attribute_sizes
    self.__kernel = None
    self.__diagonal_blocks_start = np.array(diagonal_blocks_start, dtype=np.uint32)
    self.__diagonal_blocks_count = np.array(diagonal_blocks_count, dtype=np.uint32)
    self.__diagonal_block_sizes = np.array(diagonal_block_sizes, dtype=np.uint32)
    self.__metal_kernels = {}
    self.__generateKernel()
    # print("diagonal blocks start", self.__diagonal_blocks_start)
    # print("diagonal blocks count", self.__diagonal_blocks_count)
    # print("diagonal block sizes", self.__diagonal_block_sizes)
    self.__context = context()

  @timed("diagonalBlockInverseKernel.__generateKernel")
  def __generateKernel(self):
    # first we check if the kernel exists
    size_ordered_string = "_".join([str(size) for size in sorted(self.__unique_attribute_sizes)])
    file_name = f".yasps_constant/diagonal_inverse_for_size_{size_ordered_string}"
    if is_metal():
      self.__generateMetalKernel(file_name)
      return
    if not os.path.exists(f'{file_name}.so'):
      # generate and compile the kernel
      inverse_kernel_string = """
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <vector>
#include <Eigen/Eigenvalues>

template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output) {
  const int M = 4;
  // Initialize an M x M matrix with zeros
  Eigen::Matrix<double, M, M> symMtr = Eigen::Matrix<double, M, M>::Identity();

  // Copy the input N x N matrix into the top-left corner of the M x M matrix
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      symMtr(row, col) = A[row * N + col];
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, M, M>> eigenSolver(symMtr);
  const Eigen::Matrix<double, M, M>& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < M; i++) {
  eigenValues[i] = abs(eigenValues[i]) < 1e-6 ? abs(eigenValues[i]) : abs(1.0 / eigenValues[i]);
  }
  Eigen::Matrix<double, M, M> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();
  // Copy the top-left N x N submatrix back to A
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      output[row * N + col] = A_reconstructed(row, col);
    }
  }
  return;
}

template <unsigned int N>
__device__ void invert_by_evd(const double *A, double* output) {
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < N; i++) {
    eigenValues[i] = abs(eigenValues[i]) < 1e-6 ? abs(eigenValues[i]) : abs(1.0 / eigenValues[i]);
  }

  // Reconstruct the matrix directly without using an intermediate matrix
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += B(i, k) * eigenValues[k] * B(j, k);
      }
      output[i * N + j] = sum;
    }
  }
}

"""
      for attribute_size in self.__unique_attribute_sizes:
        inverse_kernel_string += f"""
__device__ void invert_diagonal_block_{attribute_size}_device(const double* input_block, double* output_block) {{
  if ({attribute_size} <= 4) {{
    spd_projection_small<{attribute_size}>(input_block, output_block);
    return;
  }}
  invert_by_evd<{attribute_size}>(input_block, output_block);
}}

__global__ void invert_diagonal_blocks_{attribute_size}_global(const double* diagonal_blocks, double* output_blocks, int num_blocks) {{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_blocks) {{
    const double* input_block = diagonal_blocks + idx * {attribute_size} * {attribute_size};
    double* output_block = output_blocks + idx * {attribute_size} * {attribute_size};
    invert_diagonal_block_{attribute_size}_device(input_block, output_block);
  }}
}}
"""

      inverse_kernel_string += """
extern "C" {
void invert_diagonal_blocks(
  const double* diagonal_blocks,
  double* output_blocks,
  unsigned int* diagonal_blocks_start,
  unsigned int* diagonal_blocks_count,
  unsigned int* diagonal_block_sizes,
  int num_attributes
) {
  std::vector<cudaStream_t> streams(num_attributes);
  for (int i = 0; i < num_attributes; ++i) {
    cudaStreamCreate(&streams[i]);
  }
  for (int i = 0; i < num_attributes; ++i) {
    int block_size = diagonal_block_sizes[i];
    int count = diagonal_blocks_count[i];
    const double* input_blocks = diagonal_blocks + diagonal_blocks_start[i];
    double* output_blocks_ptr = output_blocks + diagonal_blocks_start[i];
    int threads_per_block = 32;
    int num_blocks = (count + threads_per_block - 1) / threads_per_block;
    switch (block_size) {
"""
      for attribute_size in self.__unique_attribute_sizes:
        inverse_kernel_string += f"""
      case {attribute_size}: {{
        invert_diagonal_blocks_{attribute_size}_global<<<num_blocks, threads_per_block, 0, streams[i]>>>(
          input_blocks,
          output_blocks_ptr,
          count
        );
        break;
      }}
"""
      inverse_kernel_string += """
      default: {
        // Handle unsupported block size
        printf("Unsupported block size: %d\\n", block_size);
        break;
      }
    }
  }
  for (unsigned int i = 0; i < num_attributes; i++) {
    cudaStreamSynchronize(streams[i]);
  }
  for (int i = 0; i < num_attributes; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}
} // extern "C"
"""
      f = open(f"{file_name}.cu", "w")
      f.write(inverse_kernel_string)
      f.close()
      # compile the kernel
      os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_89 -cudart=shared -lcuda -I/usr/include/eigen3 --expt-relaxed-constexpr -std=c++17")
    # ok now we load the kernel
    self.__kernel = ctypes.CDLL(f"{file_name}.so").invert_diagonal_blocks
    self.__kernel.argtypes = [
      ctypes.c_void_p, # diagonal_blocks
      ctypes.c_void_p, # output_blocks
      ctypes.POINTER(ctypes.c_uint), # diagonal_blocks_start
      ctypes.POINTER(ctypes.c_uint), # diagonal_blocks_count
      ctypes.POINTER(ctypes.c_uint), # diagonal_block_sizes
      ctypes.c_int # num_attributes
    ]

  def __generateMetalKernel(self, file_name):
    from yasps.backend import metal_codegen

    source_parts = [
      '#include "metalMatrix.metal"',
      "",
    ]
    for attribute_size in sorted(self.__unique_attribute_sizes):
      source_parts.append(f'''
kernel void invert_diagonal_blocks_{attribute_size}_metal(
  device const float* diagonal_blocks [[buffer(0)]],
  device float* output_blocks [[buffer(1)]],
  constant uint& block_count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {{
  if (index >= block_count) {{
    return;
  }}
  constexpr uint block_size = {attribute_size};
  thread float input_block[block_size * block_size];
  thread float output_block[block_size * block_size];
  for (uint element = 0; element < block_size * block_size; ++element) {{
    input_block[element] =
      diagonal_blocks[index * block_size * block_size + element];
  }}
  yasps_symmetric_pseudoinverse<block_size>(
    input_block,
    output_block
  );
  for (uint element = 0; element < block_size * block_size; ++element) {{
    output_blocks[index * block_size * block_size + element] =
      output_block[element];
  }}
}}
''')
    source = "\n".join(source_parts)
    source_hash = hashlib.sha256(
      source.encode("utf-8")
      + (
        Path(metal_codegen.__file__).resolve().parents[1]
        / "kernel"
        / "Compute"
        / "metalMatrix.metal"
      ).read_bytes()
    ).hexdigest()[:16]
    source_path = Path(f"{file_name}.metal")
    library_path = Path(f"{file_name}_{source_hash}.metallib")
    matrix_dir = (
      Path(metal_codegen.__file__).resolve().parents[1]
      / "kernel"
      / "Compute"
    )
    if (
      not source_path.exists()
      or source_path.read_text(encoding="utf-8") != source
    ):
      source_path.write_text(source, encoding="utf-8")
    if not library_path.exists():
      gpuarray.compile_metal(
        [source_path],
        library_path,
        include_dirs=[matrix_dir],
      )
    for attribute_size in self.__unique_attribute_sizes:
      self.__metal_kernels[attribute_size] = gpuarray.MetalKernel(
        library_path,
        f"invert_diagonal_blocks_{attribute_size}_metal"
      )
    self.__kernel = self.__metal_kernels

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(int(x.gpudata))

  @timed("diagonalBlockInverseKernel.computeDiagonalBlockInverse")
  def computeDiagonalBlockInverse(
    self,
    diagonal_blocks: gpuarray.GPUArray,
    output_blocks: gpuarray.GPUArray,
  ):
    self.__context.useDefaultContext()
    assert self.__kernel is not None, "diagonalBlockInverseKernel.computeDiagonalBlockInverse: Kernel not linked"
    if is_metal():
      for attribute_index, block_size_value in enumerate(
        self.__diagonal_block_sizes
      ):
        block_size = int(block_size_value)
        block_count = int(
          self.__diagonal_blocks_count[attribute_index]
        )
        if block_count == 0:
          continue
        element_start = int(
          self.__diagonal_blocks_start[attribute_index]
        )
        element_count = block_count * block_size * block_size
        self.__metal_kernels[block_size].dispatch(
          [
            diagonal_blocks[
              element_start:element_start + element_count
            ],
            output_blocks[
              element_start:element_start + element_count
            ],
            np.uint32(block_count),
          ],
          block_count,
          32
        )
      return
    self.__kernel(
      self.__to_void_p(diagonal_blocks),
      self.__to_void_p(output_blocks),
      self.__diagonal_blocks_start.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      self.__diagonal_blocks_count.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      self.__diagonal_block_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      ctypes.c_int(len(self.__diagonal_block_sizes))
    )

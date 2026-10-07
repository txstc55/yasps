# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from typing import List
from yasps.helper import prune_duplicate_functions
import os
import ctypes
from yasps.helper import timed
import pycuda.gpuarray as gpuarray
from yasps.primitiveUnion import primitiveUnion
import subprocess
from yasps.context import context

class globalKernel:
  @timed("globalKernel.__init__")
  def __init__(self, att: attribute):
    self.__kernelString: str = ""
    self.__headerFileString: str = ""
    self.__att = att
    self.__kernel = None
    self.__additional_compile_flags = []  # --ptxas-options=-v,-warn-spills,-warn-lmem-usage  use this for memory checking
    self.__generateKernel()
    self.__context = context()

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @timed("globalKernel.__generateKernel")
  def __generateKernel(self) -> None:
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    file_name = f".yasps_tmp/compute_{self.__att.fullNameWithHash}_spd_upper_v1"
    if not os.path.exists(f'{file_name}.so'):
      print(f"File {file_name}.so does not exist, compiling")
      self.__headerFileString += '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// Fast FP64 LDLT-equivalent PSD test using an in-place packed Schur
// complement. It preserves A and uses half the explicit scratch of the
// dense lower-plus-diagonal implementation. No tolerance discards negative
// pivots or nonzero couplings. This is still floating-point, not an exact
// arithmetic certificate; failed checks defer to eigenvalue projection.
template <unsigned int N>
__device__ __forceinline__ bool is_positive_semidefinite(
    const double *__restrict__ A) {
  static_assert(N > 0);

  constexpr unsigned int PACKED_SIZE = N * (N + 1) / 2;
  double schur[PACKED_SIZE];

#pragma unroll 1
  for (unsigned int row = 0; row < N; ++row) {
    const unsigned int row_base = row * (row + 1) / 2;
#pragma unroll 4
    for (unsigned int column = 0; column <= row; ++column) {
      schur[row_base + column] = A[row * N + column];
      if (!isfinite(schur[row_base + column])) return false;
    }
  }

#pragma unroll 1
  for (unsigned int column = 0; column < N; ++column) {
    const unsigned int column_base = column * (column + 1) / 2;
    const double pivot = schur[column_base + column];

    if (!isfinite(pivot) || pivot < 0.0) return false;

    // A zero diagonal in a PSD Schur complement requires the corresponding
    // remaining column to be zero as well.
    if (pivot == 0.0) {
#pragma unroll 4
      for (unsigned int row = column + 1; row < N; ++row) {
        const unsigned int row_base = row * (row + 1) / 2;
        if (schur[row_base + column] != 0.0) return false;
      }
      continue;
    }

    // One correctly-rounded reciprocal per column instead of one division
    // for every remaining row.
    const double inverse_pivot = __drcp_rn(pivot);
    if (!isfinite(inverse_pivot)) return false;

#pragma unroll 1
    for (unsigned int row = column + 1; row < N; ++row) {
      const unsigned int row_base = row * (row + 1) / 2;
      const double factor = schur[row_base + column] * inverse_pivot;

#pragma unroll 4
      for (unsigned int trailing_column = column + 1;
           trailing_column <= row;
           ++trailing_column) {
        const unsigned int trailing_base =
            trailing_column * (trailing_column + 1) / 2;
        schur[row_base + trailing_column] = __fma_rn(
            -factor,
            schur[trailing_base + column],
            schur[row_base + trailing_column]);
      }
    }
  }
  return true;
}
// A must be symmetric. Preserve its lower triangle and diagonal, reuse the
// strict upper triangle for Schur storage, and restore it on every exit.
// Like the packed check, this is a floating-point test without a tolerance;
// failure falls back to eigenvalue projection.
template <unsigned int N>
__device__ __forceinline__ bool is_positive_semidefinite_reuse_upper(
    double *__restrict__ A) {
  static_assert(N > 0);
  bool ok = true;

#pragma unroll 1
  for (unsigned int row = 0; row < N; ++row) {
    if (!isfinite(A[row * N + row])) ok = false;
#pragma unroll 4
    for (unsigned int column = 0; column < row; ++column) {
      const double value = A[row * N + column];
      if (!isfinite(value)) ok = false;
      A[column * N + row] = value;
    }
  }

#pragma unroll 1
  for (unsigned int column = 0; column < N && ok; ++column) {
    const double pivot =
        (column == 0) ? A[0] : A[(column - 1) * N + column];
    if (!isfinite(pivot) || pivot < 0.0) {
      ok = false;
      break;
    }

    if (pivot == 0.0) {
#pragma unroll 4
      for (unsigned int row = column + 1; row < N; ++row) {
        if (A[column * N + row] != 0.0) {
          ok = false;
          break;
        }
      }
      if (!ok) break;
#pragma unroll 4
      for (unsigned int row = column + 1; row < N; ++row) {
        A[column * N + row] = (column == 0)
            ? A[row * N + row] : A[(column - 1) * N + row];
      }
      continue;
    }

    const double inverse_pivot = __drcp_rn(pivot);
    if (!isfinite(inverse_pivot)) {
      ok = false;
      break;
    }

    // Descend so current-column entries stay live until their last use.
#pragma unroll 1
    for (unsigned int row = N - 1; row > column; --row) {
      const double column_entry = A[column * N + row];
      const double factor = column_entry * inverse_pivot;
#pragma unroll 4
      for (unsigned int trailing_column = column + 1;
           trailing_column < row; ++trailing_column) {
        A[trailing_column * N + row] = __fma_rn(
            -factor, A[column * N + trailing_column],
            A[trailing_column * N + row]);
      }
      const double old_diagonal = (column == 0)
          ? A[row * N + row] : A[(column - 1) * N + row];
      A[column * N + row] = __fma_rn(-factor, column_entry, old_diagonal);
    }
  }

#pragma unroll 1
  for (unsigned int row = 1; row < N; ++row) {
#pragma unroll 4
    for (unsigned int column = 0; column < row; ++column) {
      A[column * N + row] = A[row * N + column];
    }
  }
  return ok;
}

// For small matrix < 4
template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output, int choice) {
  if (choice == 0){
    for (int i = 0; i < N * N; i++) {
      output[i] = A[i];
    }
    return;
  }
  if (N == 1){
    output[0] = choice == 1 ? abs(A[0]) : (A[0] < 1e-6 ? 1e-6: A[0]);
    return;
  }
  if (is_positive_semidefinite<N>(A)) {
    for (unsigned int i = 0; i < N * N; ++i) output[i] = A[i];
    return;
  }

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
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1e-6;
    }
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
__device__ void spd_projection(const double *A, double* output, int choice) {
  if (choice == 0){
    for (int i = 0; i < N * N; i++) {
      output[i] = A[i];
    }
    return;
  }
  // The output is writable scratch; the const input remains untouched.
  for (unsigned int i = 0; i < N * N; ++i) output[i] = A[i];
  if (is_positive_semidefinite_reuse_upper<N>(output)) return;
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(output);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  const auto& eigenValues = eigenSolver.eigenvalues();

  // Reconstruct one triangle directly into the output, then mirror it.
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = i; j < N; ++j) {
      double sum = 0.0;
      for (unsigned int k = 0; k < N; ++k) {
        double lambda = eigenValues[k];
        if (lambda < 0.0) {
          lambda = (choice == 1) ? -lambda : 0.0;
        }
        sum += B(i, k) * lambda * B(j, k);
      }
      output[i * N + j] = sum;
      output[j * N + i] = sum;
    }
  }
  return;
}

template <unsigned int N>
__device__ void spd_projection_inplace(double *A, int choice) {
  if (choice == 0){
    return;
  }
  if (is_positive_semidefinite_reuse_upper<N>(A)) return;
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  const auto& eigenValues = eigenSolver.eigenvalues();

  // Reconstruct one triangle directly into A, then mirror it.
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = i; j < N; ++j) {
      double sum = 0.0;
      for (unsigned int k = 0; k < N; ++k) {
        double lambda = eigenValues[k];
        if (lambda < 0.0) {
          lambda = (choice == 1) ? -lambda : 0.0;
        }
        sum += B(i, k) * lambda * B(j, k);
      }
      A[i * N + j] = sum;
      A[j * N + i] = sum;
    }
  }
  return;
}
'''
      # we first generate the header file
      for item in (sortedDependency+ [self.__att.deviceKernel]):
        self.__headerFileString += f'''
extern "C" {{
{item.kernelHeader};
}}'''
      with open(".yasps_tmp/allHeaders.cuh", 'w') as f:
        f.write(self.__headerFileString)
        f.close()

      compile_jobs = []
      obj_files = []
      seen_obj_files = set([])
      for item in (sortedDependency + [self.__att.deviceKernel]):
        # we check if the .o file exists
        cu_file = f".yasps_tmp/{item.attributeName}_spd_upper_v1.cu"
        obj_file = f".yasps_tmp/{item.attributeName}_spd_upper_v1.o"
        if not obj_file in seen_obj_files:
          obj_files.append(obj_file)
        if (not os.path.exists(obj_file)) and (not obj_file in seen_obj_files):
          with open(cu_file, 'w') as f:
            f.write(f'''
#include "allHeaders.cuh"
extern "C"{{
{item.kernelString}
}}
''')
            f.close()
          compile_cmd = [
            "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-arch=sm_89",
            "-c", cu_file, "-o", obj_file,
            "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
          ] + self.__additional_compile_flags
          print("Command is")
          print(" ".join(compile_cmd))
          job = subprocess.Popen(compile_cmd)
          compile_jobs.append(job)
        seen_obj_files.add(obj_file)

      # now actually generate the global kernel
      attributeName: str = ""
      if self.__att.name == "":
        attributeName = self.__att.fullName.replace("-", "_neg_")
      else:
        attributeName = self.__att.fullName.replace("-", "_neg_")

      kernelRawName = f'''
__global__ void {attributeName}_global_function({
  "".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
  double* result,
  unsigned int MAX_INDEX
)'''
      self.__kernelString += '''
#include "allHeaders.cuh"
'''
      self.__kernelString += f'''
extern "C" {{
{kernelRawName}{{
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  // now we call the device function
  {attributeName}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f'{x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
    index,
    result + index * {self.__att.size}
  );
}}
}}
'''
      self.__kernelString += f'''
extern "C"
int compute(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
  double* result,
  unsigned int MAX_INDEX
){{
  // cudaDeviceSynchronize();
  // cudaDeviceSetLimit(cudaLimitStackSize, 128);
  {attributeName}_global_function<<<(MAX_INDEX + 31) / 32, 32>>>(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
    result,
    MAX_INDEX
  );
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {{
    fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
    return -1;
  }}
  return 0;
}}
'''
      self.__kernelString = prune_duplicate_functions(self.__kernelString)
      f = open(f"{file_name}.cu", 'w')
      f.write(self.__kernelString)
      f.close()

      # Generate global kernel .o file
      kernel_cu_file = f"{file_name}.cu"
      kernel_obj_file = f"{file_name}.o"
      kernel_compile_cmd = [
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-arch=sm_89",
        "-c", kernel_cu_file, "-o", kernel_obj_file,
        "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
      ] + self.__additional_compile_flags
      print("Kernel compile command: ")
      print(" ".join(kernel_compile_cmd))
      job = subprocess.Popen(kernel_compile_cmd)
      compile_jobs.append(job)
      # Wait for all compilation jobs
      for job in compile_jobs:
        job.wait()


      obj_files = list(set(obj_files))
      # Device link step: critical for CUDA separable compilation
      device_link_obj = f"{file_name}_device_link.o"
      dlink_cmd = [
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89",
        *(obj_files + [kernel_obj_file]), "-o", device_link_obj,
      ] + self.__additional_compile_flags
      subprocess.run(dlink_cmd, check=True)
      print("Device link command: ")
      print(" ".join(dlink_cmd))

      # Final shared object linking
      final_link_cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89",
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", f"{file_name}.so",
        "-lcudart", "-lcuda",
      ] + self.__additional_compile_flags
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)

      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute
      self.__kernel.argtypes = [
        *[ctypes.c_void_p for _ in sortedDatas],
        *[ctypes.c_void_p for _ in sortedConnectivities],
        *[ctypes.c_void_p for x in sortedConnectivities if x.dimension == 0],
        *[ctypes.c_void_p for x in sortedPrimitiveUnions],
        ctypes.c_void_p,  # result
        ctypes.c_uint  # MAX_INDEX
      ]
      self.__kernel.restype = ctypes.c_int
    else:
      print(f"File {file_name}.so does exists, linking")
      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute
      self.__kernel.argtypes = [
        *[ctypes.c_void_p for _ in sortedDatas],
        *[ctypes.c_void_p for _ in sortedConnectivities],
        *[ctypes.c_void_p for x in sortedConnectivities if x.dimension == 0],
        *[ctypes.c_void_p for x in sortedPrimitiveUnions],
        ctypes.c_void_p,  # result
        ctypes.c_uint  # MAX_INDEX
      ]
      self.__kernel.restype = ctypes.c_int

  @timed("globalKernel.compute")
  def compute(self, output):
    assert self.__kernel is not None
    if self.__att.correspondance.numInstances == 0:
      return # there is nothing to compute
    counts_gpu = [x.children_primitive_counts_gpu for x in self.__att.deviceKernel.kernelPrimitiveUnions]
    args = [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelDatas]
    args += [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelConnectivity]
    args += [self.__to_void_p(x.compressedRows) for x in self.__att.deviceKernel.kernelConnectivity if x.dimension == 0]
    args += [self.__to_void_p(x) for x in counts_gpu]
    args += [self.__to_void_p(output)]
    args += [ctypes.c_uint32(self.__att.correspondance.numInstances)]
    self.__context.useDefaultContext()
    error_code = self.__kernel(*args)
    if error_code != 0:
      raise RuntimeError(f"globalKernel.compute: Kernel execution failed with error code {error_code}")





  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self):
    return self.__kernel

from __future__ import annotations

import ctypes
import hashlib
import os
import time

import pycuda.gpuarray as gpuarray

from yasps.context import context
from yasps.helper import timed


second_order_jacobian_kernel_string = r'''
#include <cuda_runtime.h>

__global__ void assemble_rectangular_chain_rule_blocks_kernel(
  const double* row_outer_jacobian,
  const double* column_outer_jacobian,
  const double* inner_hessian,
  const double* recursive_mixed_term,
  unsigned int inner_rows,
  unsigned int inner_cols,
  unsigned int combined_rows,
  unsigned int combined_cols,
  const unsigned int* row_indices,
  const unsigned short* row_sizes,
  const short* row_permutations,
  unsigned int row_stride,
  const unsigned int* column_indices,
  const unsigned short* column_sizes,
  const short* column_permutations,
  unsigned int column_stride,
  const unsigned int* coordinate_outer,
  const unsigned int* lookup,
  double* blocks,
  unsigned int num_instances
) {
  unsigned int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= num_instances) {
    return;
  }
  unsigned int occurrence = coordinate_outer[instance];
  for (unsigned int i = 0; i < row_stride; ++i) {
    unsigned int row_slot = instance * row_stride + i;
    if (row_permutations[row_slot] <= 0 || row_indices[row_slot] < 2) {
      continue;
    }
    unsigned int row_offset = (unsigned int)row_permutations[row_slot] - 1;
    unsigned int h = row_sizes[row_slot];
    for (unsigned int j = 0; j < column_stride; ++j) {
      unsigned int column_slot = instance * column_stride + j;
      if (column_permutations[column_slot] <= 0 ||
          column_indices[column_slot] < 2) {
        continue;
      }
      unsigned int column_offset =
        (unsigned int)column_permutations[column_slot] - 1;
      unsigned int w = column_sizes[column_slot];
      unsigned int output_offset = lookup[occurrence++];
      for (unsigned int r = 0; r < h; ++r) {
        for (unsigned int c = 0; c < w; ++c) {
          unsigned int combined_row = row_offset + r;
          unsigned int combined_col = column_offset + c;
          double value = recursive_mixed_term[
            instance * combined_rows * combined_cols
            + combined_row * combined_cols
            + combined_col
          ];
          // Strict second-order chain rule:
          //
          //   J_row^T * H_inner * J_column + H_recursive.
          //
          // row_outer and column_outer have the inner variable as their row
          // space and the combined differentiation variables as columns.
          for (unsigned int a = 0; a < inner_rows; ++a) {
            double left = row_outer_jacobian[
              instance * inner_rows * combined_rows
              + a * combined_rows
              + combined_row
            ];
            if (left == 0.0) {
              continue;
            }
            for (unsigned int b = 0; b < inner_cols; ++b) {
              value += left
                * inner_hessian[
                    instance * inner_rows * inner_cols
                    + a * inner_cols
                    + b
                  ]
                * column_outer_jacobian[
                    instance * inner_cols * combined_cols
                    + b * combined_cols
                    + combined_col
                  ];
            }
          }
          atomicAdd(blocks + output_offset + r * w + c, value);
        }
      }
    }
  }
}

__global__ void rectangular_spmv_kernel(
  const unsigned int* positions,
  const double* blocks,
  unsigned int count,
  unsigned int h,
  unsigned int w,
  const double* x,
  double* y,
  bool transpose
) {
  unsigned int block_index = blockIdx.x;
  unsigned int local = threadIdx.x;
  if (block_index >= count) {
    return;
  }
  unsigned int row = positions[2 * block_index];
  unsigned int col = positions[2 * block_index + 1];
  const double* block = blocks + block_index * h * w;
  if (!transpose && local < h) {
    double value = 0.0;
    for (unsigned int c = 0; c < w; ++c) {
      value += block[local * w + c] * x[col + c];
    }
    atomicAdd(y + row + local, value);
  } else if (transpose && local < w) {
    double value = 0.0;
    for (unsigned int r = 0; r < h; ++r) {
      value += block[r * w + local] * x[row + r];
    }
    atomicAdd(y + col + local, value);
  }
}

extern "C"
int assemble_rectangular_chain_rule_blocks(
  const double* row_outer_jacobian,
  const double* column_outer_jacobian,
  const double* inner_hessian,
  const double* recursive_mixed_term,
  unsigned int inner_rows,
  unsigned int inner_cols,
  unsigned int combined_rows,
  unsigned int combined_cols,
  const unsigned int* row_indices,
  const unsigned short* row_sizes,
  const short* row_permutations,
  unsigned int row_stride,
  const unsigned int* column_indices,
  const unsigned short* column_sizes,
  const short* column_permutations,
  unsigned int column_stride,
  const unsigned int* coordinate_outer,
  const unsigned int* lookup,
  double* blocks,
  unsigned int num_instances
) {
  assemble_rectangular_chain_rule_blocks_kernel<<<
    (num_instances + 255) / 256, 256
  >>>(
    row_outer_jacobian, column_outer_jacobian,
    inner_hessian, recursive_mixed_term,
    inner_rows, inner_cols, combined_rows, combined_cols,
    row_indices, row_sizes, row_permutations, row_stride,
    column_indices, column_sizes, column_permutations, column_stride,
    coordinate_outer, lookup, blocks, num_instances
  );
  cudaError_t err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : -1;
}

extern "C"
int rectangular_spmv(
  const unsigned int* positions,
  const double* blocks,
  unsigned int count,
  unsigned int h,
  unsigned int w,
  const double* x,
  double* y,
  bool transpose
) {
  unsigned int threads = transpose ? w : h;
  if (threads == 0 || count == 0) {
    return 0;
  }
  if (threads > 1024) {
    return -2;
  }
  rectangular_spmv_kernel<<<count, threads>>>(
    positions, blocks, count, h, w, x, y, transpose
  );
  cudaError_t err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : -1;
}
'''


class secondOrderJacobianKernel:
  def __init__(self):
    self.__assemble_kernel = None
    self.__spmv_kernel = None
    self.__context = context()

  def __to_void_p(self, value):
    if value is None or value.size == 0:
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(int(value.gpudata))

  def __loadKernels(self):
    if self.__assemble_kernel is not None:
      return
    kernel_hash = hashlib.sha256(
      second_order_jacobian_kernel_string.encode("utf-8")
    ).hexdigest()[:20]
    file_name = (
      f".yasps_constant/second_order_jacobian_kernel_{kernel_hash}"
    )
    if not os.path.exists(f"{file_name}.so"):
      time_start = time.time()
      with open(f"{file_name}.cu", "w") as output:
        output.write(second_order_jacobian_kernel_string)
      result = os.system(
        f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so "
        f"{file_name}.cu -O3 -arch=sm_89 -lcudart -lcuda"
      )
      if result != 0:
        raise RuntimeError(
          "secondOrderJacobianKernel: failed to compile CUDA kernels."
        )
      print(
        "Time taken to compile second-order Jacobian kernels: "
        f"{(time.time() - time_start) * 1000.0} ms"
      )
    library = ctypes.CDLL(f"{file_name}.so")
    self.__assemble_kernel = (
      library.assemble_rectangular_chain_rule_blocks
    )
    self.__assemble_kernel.restype = ctypes.c_int
    self.__assemble_kernel.argtypes = (
      [ctypes.c_void_p] * 4
      + [ctypes.c_uint32] * 4
      + [ctypes.c_void_p] * 3
      + [ctypes.c_uint32]
      + [ctypes.c_void_p] * 3
      + [ctypes.c_uint32]
      + [ctypes.c_void_p] * 3
      + [ctypes.c_uint32]
    )
    self.__spmv_kernel = library.rectangular_spmv
    self.__spmv_kernel.restype = ctypes.c_int
    self.__spmv_kernel.argtypes = (
      [ctypes.c_void_p] * 2
      + [ctypes.c_uint32] * 3
      + [ctypes.c_void_p] * 2
      + [ctypes.c_bool]
    )

  @timed("secondOrderJacobianKernel.assembleChainRule")
  def assembleChainRule(
    self,
    row_outer_jacobian,
    column_outer_jacobian,
    inner_hessian,
    recursive_mixed_term,
    indices_kernel,
    lookup,
    blocks
  ):
    self.__context.useDefaultContext()
    self.__loadKernels()
    if (
      row_outer_jacobian.rows != inner_hessian.rows
      or column_outer_jacobian.rows != inner_hessian.cols
      or row_outer_jacobian.cols != recursive_mixed_term.rows
      or column_outer_jacobian.cols != recursive_mixed_term.cols
    ):
      raise ValueError(
        "secondOrderJacobianKernel: incompatible outer Jacobian, "
        "inner Hessian, or recursive mixed-term dimensions."
      )
    row = indices_kernel.rowIndicesKernel
    column = indices_kernel.columnIndicesKernel
    error_code = self.__assemble_kernel(
      self.__to_void_p(row_outer_jacobian.value),
      self.__to_void_p(column_outer_jacobian.value),
      self.__to_void_p(inner_hessian.value),
      self.__to_void_p(recursive_mixed_term.value),
      ctypes.c_uint32(inner_hessian.rows),
      ctypes.c_uint32(inner_hessian.cols),
      ctypes.c_uint32(recursive_mixed_term.rows),
      ctypes.c_uint32(recursive_mixed_term.cols),
      self.__to_void_p(row.outputIndices),
      self.__to_void_p(row.outputSizes),
      self.__to_void_p(row.outputPermutations),
      ctypes.c_uint32(row.maxNumIndicesNeeded),
      self.__to_void_p(column.outputIndices),
      self.__to_void_p(column.outputSizes),
      self.__to_void_p(column.outputPermutations),
      ctypes.c_uint32(column.maxNumIndicesNeeded),
      self.__to_void_p(indices_kernel.coordinateCountsOuter),
      self.__to_void_p(lookup),
      self.__to_void_p(blocks),
      ctypes.c_uint32(indices_kernel.numInstances)
    )
    if error_code != 0:
      raise RuntimeError(
        "secondOrderJacobianKernel.assembleChainRule: CUDA kernel returned "
        f"{error_code}."
      )

  @timed("secondOrderJacobianKernel.spmvCategory")
  def spmvCategory(
    self,
    positions,
    blocks,
    count,
    h,
    w,
    x,
    output,
    transpose=False
  ):
    if count == 0:
      return
    self.__context.useDefaultContext()
    self.__loadKernels()
    error_code = self.__spmv_kernel(
      self.__to_void_p(positions),
      self.__to_void_p(blocks),
      ctypes.c_uint32(count),
      ctypes.c_uint32(h),
      ctypes.c_uint32(w),
      self.__to_void_p(x),
      self.__to_void_p(output),
      ctypes.c_bool(transpose)
    )
    if error_code != 0:
      raise RuntimeError(
        f"secondOrderJacobianKernel.spmvCategory: CUDA kernel returned {error_code}."
      )

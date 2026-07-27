# cython: language_level=3
from __future__ import annotations
from typing import List, Tuple, Set
from yasps.backend import gpuarray
import ctypes
import numpy as np
from yasps.backend import cuda
import os
import hashlib
import json
import time
from yasps.helper import timed
from yasps.context import context
from yasps.backend import is_metal
from pathlib import Path

class solverKernel:
  __cuda_cache_version = "relative_residual_v1"

  def __init__(self, blockDimensions: List[int]):
    self.__max_row_size = 0
    self.__cg_kernel = None
    self.__saved_block_dimensions = set([])
    self.__context = context()
    self.__metal_spmv_kernels = {}
    self.__metal_block_jacobi_kernel = None
    self.__metal_dot_kernel = None
    self.__metal_sum_kernel = None
    self.__metal_vec_add_kernel = None
    self.__metal_cg_direction_kernel = None
    self.__metal_cg_update_kernel = None
    self.__metal_cg_finalize_kernel = None
    self.__metal_fill_kernel = None
    self.__dot_scratch_a = gpuarray.empty(0, dtype=np.float32)
    self.__dot_scratch_b = gpuarray.empty(0, dtype=np.float32)
    self.__cg_state = gpuarray.empty(4, dtype=np.float32)
    self.__cg_status = gpuarray.empty(2, dtype=np.int32)

  def updateBlockDimensions(self, blockDimensions: List[int]):
    self.__init_kernel(blockDimensions)

  @timed("solverKernel.__init_kernel")
  def __init_kernel(self, blockDimensions: List[int]):
    # convert blockDimensions to a tuple of int, int
    blockDimensionsTuples = []
    for i in range(len(blockDimensions) // 2):
      blockDimensionsTuples.append((blockDimensions[i * 2], blockDimensions[i * 2 + 1]))
    blockDimensionsTuplesSet = set(blockDimensionsTuples)
    # print("Old block dimensions is", self.__saved_block_dimensions)
    # print("New block dimensions is", blockDimensionsTuplesSet)
    if blockDimensionsTuplesSet.issubset(self.__saved_block_dimensions):
      return
    else:
      # we may need to create a new kernel
      self.__saved_block_dimensions.update(blockDimensionsTuplesSet)
      max_modded_row_size = (max(blockDimensions[::2]) + 2) // 3 * 3
      self.__max_row_size = max_modded_row_size
      if is_metal():
        self.__generateMetalKernels()
        return
      dimension_to_text = [f'{dim[0]}_{dim[1]}' for dim in blockDimensionsTuplesSet]
      dimension_to_text = '__'.join(dimension_to_text)
      file_original_name = f".yasps_constant/cg_dims_{dimension_to_text}"
      cache_key = (
        f"{self.__cuda_cache_version}:{dimension_to_text}"
      )
      file_hashed_name = f".yasps_constant/cg_dims_{int(hashlib.sha256(cache_key.encode('utf-8')).hexdigest(), 16)}"
      # now we first record this information in a json file
      if not os.path.exists(".yasps_constant/cg_dimension_to_file.json"):
        file_to_dimensions = []
        with open(".yasps_constant/cg_dimension_to_file.json", "w", encoding="utf-8") as f:
          json.dump(file_to_dimensions, f, indent=2)

      # now open the json file and see if this dimension_to_text already exists
      with open(".yasps_constant/cg_dimension_to_file.json", "r", encoding="utf-8") as f:
        items = json.load(f)
        in_json_but_no_so = False # false means not in file, true means in file but so file not found
        for item in items:
          if (
            item.get("solver_version")
            != self.__cuda_cache_version
          ):
            continue
          # we check if the current dimensions has been compiled to a file before
          seen_dimensions = item["dimensions"]
          seen_dimensions = [tuple(dim) for dim in seen_dimensions]
          seen_dimensions = set(seen_dimensions)
          if self.__saved_block_dimensions.issubset(seen_dimensions):
            # now we check if the file exists
            file_hashed_name_existing = item["file_hashed_name"]
            if os.path.exists(f"{file_hashed_name_existing}.so"):
              self.__saved_block_dimensions = seen_dimensions
              self.__cg_kernel = ctypes.CDLL(f"{file_hashed_name_existing}.so").computeSolution
              self.__cg_kernel.argtypes = [
                ctypes.c_uint,   # maxIteration
                ctypes.c_double, # threshold
                ctypes.c_void_p, # block_values (device pointer to double)
                ctypes.c_void_p, # block_positions (device pointer to unsigned int)
                ctypes.POINTER(ctypes.c_uint), # block_values_start (unsigned int list from numpy array)
                ctypes.POINTER(ctypes.c_uint), # block_counts (unsigned int list from numpy)
                ctypes.POINTER(ctypes.c_uint), # block_dimensions (unsigned int list from numpy)
                ctypes.c_uint,   # NUM_BLOCK_DIMENSIONS
                ctypes.c_void_p, # block_values_dynamic (device pointer to double)
                ctypes.c_void_p, # block_positions_dynamic (device pointer to unsigned int)
                ctypes.POINTER(ctypes.c_uint), # block_values_start_dynamic (unsigned int list from numpy array)
                ctypes.POINTER(ctypes.c_uint), # block_counts_dynamic (unsigned int list from numpy)
                ctypes.POINTER(ctypes.c_uint), # block_dimensions_dynamic (unsigned int list from numpy)
                ctypes.c_uint,   # NUM_BLOCK_DIMENSIONS_DYNAMIC
                ctypes.c_void_p, # diagonal (device pointer to double)
                ctypes.c_void_p, # diagonalBlockInverse (device pointer to double)
                ctypes.POINTER(ctypes.c_uint), # diagonalBlocksStart
                ctypes.POINTER(ctypes.c_uint), # diagonalBlocksCount
                ctypes.POINTER(ctypes.c_uint), # diagonalBlocksSize
                ctypes.POINTER(ctypes.c_uint), # gradientSegmentsStart
                ctypes.c_int,   # numAttributes
                ctypes.c_void_p, # gradient (device pointer to double)
                ctypes.c_uint,   # MATRIX_SIZE
                ctypes.c_void_p, # d_p1_b (device pointer)
                ctypes.c_void_p, # d_r (device pointer)
                ctypes.c_void_p, # d_c (device pointer)
                ctypes.c_void_p, # d_q (device pointer)
                ctypes.c_void_p, # d_s (device pointer)
                ctypes.c_void_p, # solution (device pointer)
                ctypes.c_void_p  # initial_guess (device pointer)
              ]
              return
            else:
              in_json_but_no_so = True
      # if we reach here, we need to compile a new kernel
      # because either the dimension doesnt exist in the previous compiled files,
      # or the file is not found
      kernelString: str = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <vector>

// for checking cuda error
#define CUDA_CHECK_ERROR(ans)                                                  \\
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
                       if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}


template<unsigned int BLOCK_ROW_SIZE, unsigned int BLOCK_COL_SIZE>
__device__ __forceinline__ void blockMultiply(const double *blockValues,
                                              const double *x,
                                              double *y) {
  // multiply a row
  for (int i = 0; i < BLOCK_ROW_SIZE; i++) {
    for (int j = 0; j < BLOCK_COL_SIZE; j++) {
      y[i] += __ldg(&blockValues[i * BLOCK_COL_SIZE + j]) * __ldg(&x[j]);
    }
  }
}
template<unsigned int BLOCK_ROW_SIZE, unsigned int BLOCK_COL_SIZE>
__device__ __forceinline__ void
blockMultiplyTranspose(const double *blockValues,
                       const double *x,
                       double *y
                       ) {
  for (int i = 0; i < BLOCK_COL_SIZE; i++) {
    double temp = 0.0;
    for (int j = 0; j < BLOCK_ROW_SIZE; j++) {
      temp += __ldg(&blockValues[j * BLOCK_COL_SIZE + i]) * __ldg(&x[j]);
    }
    atomicAdd(y + i, temp);
  }
}

// computes Ax=y where A does not contain any diagonal blocks
template <unsigned int BLOCK_ROW_SIZE, unsigned int BLOCK_COL_SIZE>
__global__ void spmvOffDiagonalBlocks(const double *blockValues,
                                      const unsigned int VALUE_START, // where in the block values does this dimension's block start
                                      const unsigned int* positions, // the coordinate of this block
                                      const unsigned int POSITIONS_START, // where does the positions start for this dimension
                                      const unsigned int POSITIONS_END, // where does the positions end for this dimension
                                      const double *x, // the Ax = y
                                      double *y) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; // we first get the id of this thread
  int tid = threadIdx.x;
'''
      kernelString += f'''
  __shared__ double allResults[BLOCK_ROW_SIZE * 32]; // accumulate the multiplied result
  __shared__ unsigned int rows[32];
  __shared__ unsigned int cols[32];
  for (int i = tid; i < BLOCK_ROW_SIZE * 32; i += 32) {{
      allResults[i] = 0.0;
  }}
'''
      kernelString += '''
  if (id < POSITIONS_END - POSITIONS_START) {
    // do the multiplication, and put the result in allresults
    rows[tid] = positions[POSITIONS_START * 2 + id * 2]; // get the coordinate of the block
    cols[tid] = positions[POSITIONS_START * 2 + id * 2 + 1]; // get the coordinate of the block
    blockMultiply<BLOCK_ROW_SIZE, BLOCK_COL_SIZE>(blockValues + VALUE_START + id * BLOCK_ROW_SIZE * BLOCK_COL_SIZE, x + cols[tid], allResults + tid * BLOCK_ROW_SIZE);
  }else{
    rows[tid] = 1316134911; // TODO: REPLACE THIS WITH A BETTER VALUE
    cols[tid] = 1316134911;
  }
  __syncthreads();

  // here we do a almost reduction sum
  if (id < POSITIONS_END - POSITIONS_START) {
    if (tid == 0 || rows[tid] != rows[tid - 1]) {
      // this is usually where the start of a row
'''
      kernelString += f'''
      double sum[BLOCK_ROW_SIZE] = {{0}}; // initialize the sum
'''
      kernelString += '''
      for (int i = tid; i < 32 && rows[i] == rows[tid]; i++) {
        for (int j = 0; j < BLOCK_ROW_SIZE; j++) {
          sum[j] += allResults[i * BLOCK_ROW_SIZE + j];
        }
      }
      // we have accumulated the sum, now we need to add it to the final result
      for (int i = 0; i < BLOCK_ROW_SIZE; i++) {
        atomicAdd(y + rows[tid] + i, sum[i]);
      }
    }
  }

  if (id < POSITIONS_END - POSITIONS_START) {
    unsigned int row = rows[tid];
    unsigned int col = cols[tid];
    if (row != col) {
      blockMultiplyTranspose<BLOCK_ROW_SIZE, BLOCK_COL_SIZE>(blockValues + VALUE_START + id * BLOCK_ROW_SIZE * BLOCK_COL_SIZE, x + rows[tid], y + cols[tid]);
    }
  }
}

void spmvWithSystem(const double* block_values, // the value of the blocks in the hessian
                    const unsigned int* block_positions, // the coordinate of each block
                    const unsigned int* block_values_start, // for each different dimension of blocks, where in the values array does it start
                    const unsigned int* block_counts, // how many blocks in each dimension
                    const double* block_values_dynamic, // the value of the dynamic blocks in the hessian
                    const unsigned int* block_positions_dynamic, // the coordinate of each dynamic block
                    const unsigned int* block_values_start_dynamic, // for each different dimension of dynamic blocks, where in the values array does it start
                    const unsigned int* block_counts_dynamic, // how many dynamic blocks in each dimension
                    const double* x, // Ax = y
                    double* y,
                    const unsigned int* block_dimensions,
                    const unsigned int NUM_BLOCK_DIMENSIONS,
                    const unsigned int* block_dimensions_dynamic,
                    const unsigned int NUM_BLOCK_DIMENSIONS_DYNAMIC,
                    std::vector<cudaStream_t>& streams
                    ){
  unsigned int positions_start = 0;
  unsigned int positions_end = 0;
  for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS; i++){
    positions_end = positions_start + block_counts[i];
    switch(block_dimensions[i * 2]<< 16 | block_dimensions[i * 2 + 1]){
'''
      for dim in self.__saved_block_dimensions:
        kernelString += f'''
      case {dim[0]} << 16 | {dim[1]}:
        spmvOffDiagonalBlocks<{dim[0]}, {dim[1]}><<<(block_counts[i] + 31) / 32, 32, 0, streams[i]>>>(block_values, block_values_start[i], block_positions, positions_start, positions_end, x, y);
        break;
'''
      kernelString += '''
      default:
        printf("Unsupported block dimension %d x %d\\n", block_dimensions[i * 2], block_dimensions[i * 2 + 1]);
        break;
    }
    positions_start = positions_end;
  }
  positions_start = 0;
  positions_end = 0;
  for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS_DYNAMIC; i++){
    positions_end = positions_start + block_counts_dynamic[i];
    switch(block_dimensions_dynamic[i * 2]<< 16 | block_dimensions_dynamic[i * 2 + 1]){
'''
      for dim in self.__saved_block_dimensions:
        kernelString += f'''
      case {dim[0]} << 16 | {dim[1]}:
        spmvOffDiagonalBlocks<{dim[0]}, {dim[1]}><<<(block_counts_dynamic[i] + 31) / 32, 32, 0, streams[i + NUM_BLOCK_DIMENSIONS]>>>(block_values_dynamic, block_values_start_dynamic[i], block_positions_dynamic, positions_start, positions_end, x, y);
        break;
'''
      kernelString += '''
      default:
        printf("Unsupported dynamic block dimension %d x %d\\n", block_dimensions[i * 2], block_dimensions[i * 2 + 1]);
        break;
    }
    positions_start = positions_end;
  }
  // synchronize all streams
  for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; i++) {
    cudaStreamSynchronize(streams[i]);
  }
}

__global__ void jacobiPreconditioner(const double* diagonal, const double* x, double* y, unsigned int N){
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N){
    y[id] = x[id] / (abs(diagonal[id]) < 1e-6 ? 1.0 : diagonal[id]);
  }
}

__global__ void blockJacobiPreconditionerGlobal(const double* diagonalBlockInverse, const double* x, double* y, const unsigned int N, const unsigned int blockSize){
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N){
    for (int i = 0; i < blockSize; i++){
      double tmp = 0.0;
      for (int j = 0; j < blockSize; j++){
        tmp += diagonalBlockInverse[id * blockSize * blockSize + i * blockSize + j] * x[id * blockSize + j];
      }
      y[id * blockSize + i] = tmp;
    }
  }
}

void blockJacobiPreconditioner(
  const double* diagonalBlockInverse,
  const double* x,
  double* y,
  const unsigned int* diagonalBlocksStart,
  const unsigned int* diagonalBlocksCount,
  const unsigned int* diagonalBlockSize,
  const unsigned int* gradientSegmentsStart,
  const int numAttributes,
  std::vector<cudaStream_t>& streams
){
  for (unsigned int i = 0; i < numAttributes; i++){
    const unsigned int blockStart = diagonalBlocksStart[i];
    const unsigned int blockCount = diagonalBlocksCount[i];
    const unsigned int blockSize = diagonalBlockSize[i];
    const unsigned int segmentStart = gradientSegmentsStart[i];
    blockJacobiPreconditionerGlobal<<<(blockCount + 32 - 1) / 32, 32, 0, streams[i]>>>(
      diagonalBlockInverse + blockStart,
      x + segmentStart,
      y + segmentStart,
      blockCount,
      blockSize
    );
  }
  for (unsigned int i = 0; i < numAttributes; i++) {
    cudaStreamSynchronize(streams[i]);
  }
}

// this function computes c = a * b^T
__global__ void dotProduct(const double *a, const double *b, double *c, int n) {
  __shared__ double cache[256]; // Shared memory size per block
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;

  double temp = 0.0;
  while (tid < n) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  // Set the cache values
  cache[cacheIndex] = temp;

  // Synchronize threads in this block
  __syncthreads();

  // Reduction
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }

  // Only one thread writes the result for this block back to global memory
  if (cacheIndex == 0)
    atomicAdd(c, cache[0]);
}

// this function computes c = a + scalar * b
__global__ void vecAddWithScalar(const double *a, const double *b, double *c,
                                 double scalar, unsigned int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    c[i] = a[i] + b[i] * scalar;
  }
}
extern "C" {
int computeSolution(unsigned int maxIteration,
                            double threshold,
                            const double* block_values,
                            const unsigned int* block_positions,
                            const unsigned int* block_values_start,
                            const unsigned int* block_counts,
                            const unsigned int* block_dimensions,
                            const unsigned int NUM_BLOCK_DIMENSIONS,
                            const double* block_values_dynamic,
                            const unsigned int* block_positions_dynamic,
                            const unsigned int* block_values_start_dynamic,
                            const unsigned int* block_counts_dynamic,
                            const unsigned int* block_dimensions_dynamic,
                            const unsigned int NUM_BLOCK_DIMENSIONS_DYNAMIC,
                            const double* diagonal,
                            const double* diagonalBlockInverse,
                            const unsigned int* diagonalBlocksStart,
                            const unsigned int* diagonalBlocksCount,
                            const unsigned int* diagonalBlocksSize,
                            const unsigned int* gradientSegmentsStart,
                            const int numAttributes,
                            const double* gradient,
                            const unsigned int MATRIX_SIZE,
                            double* d_p1_b, // retained for generated-kernel ABI compatibility
                            double* d_r, // for residual
                            double* d_c,
                            double* d_q,
                            double* d_s,
                            double* solution,
                            double* initial_guess
                            ) {
  // Instead, retrieve the current context (if necessary)
  std::vector<cudaStream_t> streams;
  streams.resize(NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC);
  // initialize cuda streams for block multiplciations
  for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; i++) {
    cudaStreamCreate(&streams[i]);
  }

  std::vector<cudaStream_t> preconditioner_streams;
  preconditioner_streams.resize(numAttributes);
  for (unsigned int i = 0; i < numAttributes; i++) {
    cudaStreamCreate(&preconditioner_streams[i]);
  }
  // set the initial guess
  CUDA_CHECK_ERROR(cudaMemcpy(solution, initial_guess, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));

  // Compute the actual initial residual: r = gradient - A * initial_guess.
  spmvWithSystem(block_values,
                 block_positions,
                 block_values_start,
                 block_counts,
                 block_values_dynamic,
                 block_positions_dynamic,
                 block_values_start_dynamic,
                 block_counts_dynamic,
                 initial_guess,
                 d_r,
                 block_dimensions,
                 NUM_BLOCK_DIMENSIONS,
                 block_dimensions_dynamic,
                 NUM_BLOCK_DIMENSIONS_DYNAMIC,
                 streams);
  vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(gradient, d_r, d_r, -1.0, MATRIX_SIZE);



  // c = P^-1 * r
  cudaMemset(d_c, 0, MATRIX_SIZE * sizeof(double));
  // jacobiPreconditioner<<<(MATRIX_SIZE + 255) / 256, 256>>>(diagonal, d_r, d_c, MATRIX_SIZE);
  blockJacobiPreconditioner(
    diagonalBlockInverse,
    d_r,
    d_c,
    diagonalBlocksStart,
    diagonalBlocksCount,
    diagonalBlocksSize,
    gradientSegmentsStart,
    numAttributes,
    preconditioner_streams
  );

  // delta_new = r * c
  double* d_delta_new; // for device
  double h_delta_new, h_delta_old; // for host
  cudaMalloc(&d_delta_new, sizeof(double));
  cudaMemset(d_delta_new, 0, sizeof(double));
  dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_c, d_delta_new, MATRIX_SIZE);
  CUDA_CHECK_ERROR(cudaMemcpy(&h_delta_new, d_delta_new, sizeof(double), cudaMemcpyDeviceToHost));

  // Convergence is relative to the actual initial residual, not the RHS.
  // Use the raw residual only if the preconditioned metric degenerates.
  double* d_residual_norm;
  double h_residual_norm;
  cudaMalloc(&d_residual_norm, sizeof(double));

  bool useIdentityPreconditioner = !isfinite(h_delta_new) || h_delta_new <= 0.0;
  if (useIdentityPreconditioner){
    cudaMemset(d_residual_norm, 0, sizeof(double));
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_r, d_residual_norm, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaMemcpy(&h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost));
    if (!isfinite(h_residual_norm)){
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }
      cudaFree(d_delta_new);
      cudaFree(d_residual_norm);
      return -4;
    }
    if (h_residual_norm == 0.0){
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }
      cudaFree(d_delta_new);
      cudaFree(d_residual_norm);
      return 0;
    }
    CUDA_CHECK_ERROR(cudaMemcpy(d_c, d_r, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
    h_delta_new = h_residual_norm;
  }
  double relativeTolerance = threshold * h_delta_new;
  // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  // for setting alpha
  double* d_alpha; // for device
  double h_alpha;
  cudaMalloc(&d_alpha, sizeof(double));
  cudaMemset(d_alpha, 0, sizeof(double));

  for (unsigned int iteration = 1; iteration <= maxIteration; iteration++){
    // q = A * c
    CUDA_CHECK_ERROR(cudaMemset(d_q, 0, MATRIX_SIZE * sizeof(double)));
    spmvWithSystem(block_values,
                   block_positions,
                   block_values_start,
                   block_counts,
                   block_values_dynamic,
                   block_positions_dynamic,
                   block_values_start_dynamic,
                   block_counts_dynamic,
                   d_c,
                   d_q,
                   block_dimensions,
                   NUM_BLOCK_DIMENSIONS,
                   block_dimensions_dynamic,
                   NUM_BLOCK_DIMENSIONS_DYNAMIC,
                   streams);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemset(d_alpha, 0, sizeof(double));
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_c, d_q, d_alpha, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(&h_alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToHost);

    if (!isfinite(h_alpha)){
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }
      cudaFree(d_delta_new);
      cudaFree(d_residual_norm);
      cudaFree(d_alpha);
      return -4;
    }
    if (h_alpha <= 0.0){
      printf("Non SPD matrix detected in %d iterations with residual %lf and alpha %lf\\n", iteration, h_delta_new, h_alpha);
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }

      // free
      cudaFree(d_delta_new);
      cudaFree(d_residual_norm);
      cudaFree(d_alpha);
      return -iteration - 4;
    }
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    h_alpha = h_delta_new / h_alpha;

    // deltav = deltav + alpha * c
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(solution, d_c, solution, h_alpha, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // r = r - alpha * q
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_q, d_r, -h_alpha, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    if (useIdentityPreconditioner){
      CUDA_CHECK_ERROR(cudaMemcpy(d_s, d_r, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
    }else{
      blockJacobiPreconditioner(
        diagonalBlockInverse,
        d_r,
        d_s,
        diagonalBlocksStart,
        diagonalBlocksCount,
        diagonalBlocksSize,
        gradientSegmentsStart,
        numAttributes,
        preconditioner_streams
      );
    }

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    h_delta_old = h_delta_new;
    // delta_new = r * s
    cudaMemset(d_delta_new, 0, sizeof(double));
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_s, d_delta_new, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(&h_delta_new, d_delta_new, sizeof(double), cudaMemcpyDeviceToHost);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    bool converged = false;
    bool restartWithIdentity = false;
    if (useIdentityPreconditioner){
      if (!isfinite(h_delta_new)){
        for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
          cudaStreamDestroy(streams[i]);
        }
        for (unsigned int i = 0; i < numAttributes; ++i) {
          cudaStreamDestroy(preconditioner_streams[i]);
        }
        cudaFree(d_delta_new);
        cudaFree(d_residual_norm);
        cudaFree(d_alpha);
        return -4;
      }
      converged = h_delta_new <= relativeTolerance;
    }else if (!isfinite(h_delta_new) || h_delta_new <= 0.0){
      cudaMemset(d_residual_norm, 0, sizeof(double));
      dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_r, d_residual_norm, MATRIX_SIZE);
      CUDA_CHECK_ERROR(cudaMemcpy(&h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost));
      if (!isfinite(h_residual_norm)){
        for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
          cudaStreamDestroy(streams[i]);
        }
        for (unsigned int i = 0; i < numAttributes; ++i) {
          cudaStreamDestroy(preconditioner_streams[i]);
        }
        cudaFree(d_delta_new);
        cudaFree(d_residual_norm);
        cudaFree(d_alpha);
        return -4;
      }
      converged = h_residual_norm == 0.0;
      restartWithIdentity = h_residual_norm > 0.0;
    }else{
      converged = h_delta_new <= relativeTolerance;
    }
    if (converged){
      printf("Converged in %d iterations with residual %lf\\n", iteration, h_delta_new);
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }

      // free
      cudaFree(d_delta_new);
      cudaFree(d_residual_norm);
      cudaFree(d_alpha);
      return iteration;
    }
    if (restartWithIdentity){
      useIdentityPreconditioner = true;
      h_delta_new = h_residual_norm;
      relativeTolerance = threshold * h_residual_norm;
      CUDA_CHECK_ERROR(cudaMemcpy(d_c, d_r, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
      continue;
    }

    // c = s + (delta_new / delta_old) * c
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_s, d_c, d_c, h_delta_new / h_delta_old, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  }
  // after the two for‑loops that synchronize the streams
  for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
    cudaStreamDestroy(streams[i]);
  }

  for (unsigned int i = 0; i < numAttributes; ++i) {
    cudaStreamDestroy(preconditioner_streams[i]);
  }

  // free
  cudaFree(d_delta_new);
  cudaFree(d_residual_norm);
  cudaFree(d_alpha);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error during kernel execution: %s\\n", cudaGetErrorString(err));
    return -3;  // Return error to Python
  }
  printf("Converged in %d iterations with residual %lf\\n", maxIteration + 1, h_delta_new);
  return maxIteration + 1;
}

} // close the extern "C"
'''
      # ok now we compile the kernel by saving it to a file and then calling nvcc
      f = open(f"{file_hashed_name}.cu", 'w')
      f.write(kernelString)
      f.close()

      # now we compile the kernel
      os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_hashed_name}.so {file_hashed_name}.cu -O3 -arch=sm_89 -cudart=shared -lcuda --expt-relaxed-constexpr -std=c++17")
      self.__cg_kernel = ctypes.CDLL(f"{file_hashed_name}.so").computeSolution
      self.__cg_kernel.argtypes = [
        ctypes.c_uint,   # maxIteration
        ctypes.c_double, # threshold
        ctypes.c_void_p, # block_values (device pointer to double)
        ctypes.c_void_p, # block_positions (device pointer to unsigned int)
        ctypes.POINTER(ctypes.c_uint), # block_values_start (unsigned int list from numpy array)
        ctypes.POINTER(ctypes.c_uint), # block_counts (unsigned int list from numpy)
        ctypes.POINTER(ctypes.c_uint), # block_dimensions (unsigned int list from numpy)
        ctypes.c_uint,   # NUM_BLOCK_DIMENSIONS
        ctypes.c_void_p, # block_values_dynamic (device pointer to double)
        ctypes.c_void_p, # block_positions_dynamic (device pointer to unsigned int)
        ctypes.POINTER(ctypes.c_uint), # block_values_start_dynamic (unsigned int list from numpy array)
        ctypes.POINTER(ctypes.c_uint), # block_counts_dynamic (unsigned int list from numpy)
        ctypes.POINTER(ctypes.c_uint), # block_dimensions_dynamic (unsigned int list from numpy)
        ctypes.c_uint,   # NUM_BLOCK_DIMENSIONS_DYNAMIC
        ctypes.c_void_p, # diagonal (device pointer to double)
        ctypes.c_void_p, # diagonalBlockInverse (device pointer to double)
        ctypes.POINTER(ctypes.c_uint), # diagonalBlocksStart
        ctypes.POINTER(ctypes.c_uint), # diagonalBlocksCount
        ctypes.POINTER(ctypes.c_uint), # diagonalBlocksSize
        ctypes.POINTER(ctypes.c_uint), # gradientSegmentsStart
        ctypes.c_int,   # numAttributes
        ctypes.c_void_p, # gradient (device pointer to double)
        ctypes.c_uint,   # MATRIX_SIZE
        ctypes.c_void_p, # d_p1_b (device pointer)
        ctypes.c_void_p, # d_r (device pointer)
        ctypes.c_void_p, # d_c (device pointer)
        ctypes.c_void_p, # d_q (device pointer)
        ctypes.c_void_p, # d_s (device pointer)
        ctypes.c_void_p, # solution (device pointer)
        ctypes.c_void_p  # initial_guess (device pointer)
      ]
      data = []
      with open(".yasps_constant/cg_dimension_to_file.json", "r", encoding="utf-8") as f:
        data = json.load(f)
      for item in data:
        if item["file_hashed_name"] == file_hashed_name:
          # already exists
          return
      data.append({
        "dimensions": [dim for dim in self.__saved_block_dimensions],
        "file_hashed_name": file_hashed_name,
        "file_original_name": file_original_name,
        "solver_version": self.__cuda_cache_version,
      })
      with open(".yasps_constant/cg_dimension_to_file.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

  def __generateMetalKernels(self):
    source_parts = [
      "#include <metal_stdlib>",
      "using namespace metal;",
      "",
      (
        "inline void yasps_atomic_add("
        "device atomic_float* target, float value) {\n"
        "  atomic_fetch_add_explicit(\n"
        "    target,\n"
        "    value,\n"
        "    memory_order_relaxed\n"
        "  );\n"
        "}"
      ),
    ]
    for row_size, col_size in sorted(self.__saved_block_dimensions):
      source_parts.append(f'''
kernel void spmv_blocks_{row_size}_{col_size}_metal(
  device const float* block_values [[buffer(0)]],
  constant uint& value_start [[buffer(1)]],
  device const uint* positions [[buffer(2)]],
  constant uint& positions_start [[buffer(3)]],
  constant uint& block_count [[buffer(4)]],
  device const float* x [[buffer(5)]],
  device atomic_float* y [[buffer(6)]],
  device const int* cg_status [[buffer(7)]],
  uint index [[thread_position_in_grid]]
) {{
  if (cg_status[0] != 0 || index >= block_count) {{
    return;
  }}
  constexpr uint row_size = {row_size};
  constexpr uint col_size = {col_size};
  uint row = positions[(positions_start + index) * 2];
  uint col = positions[(positions_start + index) * 2 + 1];
  uint block_start = value_start + index * row_size * col_size;
  for (uint local_row = 0; local_row < row_size; ++local_row) {{
    float value = 0.0f;
    for (uint local_col = 0; local_col < col_size; ++local_col) {{
      value += block_values[
        block_start + local_row * col_size + local_col
      ] * x[col + local_col];
    }}
    yasps_atomic_add(&y[row + local_row], value);
  }}
  if (row == col) {{
    return;
  }}
  for (uint local_col = 0; local_col < col_size; ++local_col) {{
    float value = 0.0f;
    for (uint local_row = 0; local_row < row_size; ++local_row) {{
      value += block_values[
        block_start + local_row * col_size + local_col
      ] * x[row + local_row];
    }}
    yasps_atomic_add(&y[col + local_col], value);
  }}
}}
''')
    source_parts.append('''
kernel void block_jacobi_metal(
  device const float* diagonal_inverse [[buffer(0)]],
  device const float* x [[buffer(1)]],
  device float* y [[buffer(2)]],
  constant uint& block_count [[buffer(3)]],
  constant uint& block_size [[buffer(4)]],
  device const int* cg_status [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (cg_status[0] != 0 || index >= block_count) {
    return;
  }
  uint matrix_start = index * block_size * block_size;
  uint vector_start = index * block_size;
  for (uint row = 0; row < block_size; ++row) {
    float value = 0.0f;
    for (uint col = 0; col < block_size; ++col) {
      value += diagonal_inverse[
        matrix_start + row * block_size + col
      ] * x[vector_start + col];
    }
    y[vector_start + row] = value;
  }
}

kernel void dot_product_partial_metal(
  device const float* left [[buffer(0)]],
  device const float* right [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant uint& count [[buffer(3)]],
  device const int* cg_status [[buffer(4)]],
  uint index [[thread_position_in_grid]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  threadgroup float values[256];
  values[local_index] = (
    cg_status[0] == 0 && index < count
  ) ? left[index] * right[index] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride /= 2) {
    if (local_index < stride) {
      values[local_index] += values[local_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    output[group_index] = values[0];
  }
}

kernel void sum_partial_metal(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  device const int* cg_status [[buffer(3)]],
  uint index [[thread_position_in_grid]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  threadgroup float values[256];
  values[local_index] = (
    cg_status[0] == 0 && index < count
  ) ? input[index] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride /= 2) {
    if (local_index < stride) {
      values[local_index] += values[local_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    output[group_index] = values[0];
  }
}

kernel void vec_add_with_scalar_metal(
  device const float* left [[buffer(0)]],
  device const float* right [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant float& scalar [[buffer(3)]],
  constant uint& count [[buffer(4)]],
  device const int* cg_status [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (cg_status[0] == 0 && index < count) {
    output[index] = left[index] + right[index] * scalar;
  }
}

kernel void cg_direction_from_state_metal(
  device const float* preconditioned_residual [[buffer(0)]],
  device float* direction [[buffer(1)]],
  device const float* state [[buffer(2)]],
  device const int* status [[buffer(3)]],
  constant uint& update_direction [[buffer(4)]],
  constant uint& count [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (
    status[0] == 0 && update_direction != 0 && index < count
  ) {
    direction[index] =
      preconditioned_residual[index] + direction[index] * state[3];
  }
}

kernel void cg_solution_residual_from_denominator_metal(
  device float* solution [[buffer(0)]],
  device const float* direction [[buffer(1)]],
  device float* residual [[buffer(2)]],
  device const float* system_product [[buffer(3)]],
  device const float* denominator [[buffer(4)]],
  device float* state [[buffer(5)]],
  device int* status [[buffer(6)]],
  constant uint& iteration [[buffer(7)]],
  constant uint& count [[buffer(8)]],
  uint index [[thread_position_in_grid]]
) {
  if (status[0] != 0) {
    return;
  }
  float value = denominator[0];
  if (index == 0) {
    state[2] = value;
    if (!isfinite(value)) {
      status[0] = -4;
    } else if (value <= 0.0f) {
      status[0] = -int(iteration) - 4;
    }
  }
  if (index < count && isfinite(value) && value > 0.0f) {
    float alpha = state[0] / value;
    solution[index] += direction[index] * alpha;
    residual[index] -= system_product[index] * alpha;
  }
}

kernel void cg_finalize_iteration_metal(
  device const float* reduced_delta [[buffer(0)]],
  device float* state [[buffer(1)]],
  device int* status [[buffer(2)]],
  constant uint& identity_preconditioner [[buffer(3)]],
  constant uint& iteration [[buffer(4)]],
  uint index [[thread_position_in_grid]]
) {
  if (index != 0 || status[0] != 0) {
    return;
  }
  float delta = reduced_delta[0];
  if (!isfinite(delta)) {
    status[0] = identity_preconditioner != 0 ? -4 : -2147483647;
    status[1] = int(iteration);
    state[3] = delta;
    return;
  }
  if (identity_preconditioner == 0 && delta <= 0.0f) {
    status[0] = -2147483647;
    status[1] = int(iteration);
    state[3] = delta;
    return;
  }
  if (delta <= state[1]) {
    status[0] = int(iteration);
    return;
  }
  state[3] = delta / state[0];
  state[0] = delta;
}

kernel void fill_float_metal(
  device float* output [[buffer(0)]],
  constant float& value [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  device const int* cg_status [[buffer(3)]],
  uint index [[thread_position_in_grid]]
) {
  if (cg_status[0] == 0 && index < count) {
    output[index] = value;
  }
}
''')
    source = "\n".join(source_parts)
    source_hash = hashlib.sha256(
      source.encode("utf-8")
    ).hexdigest()[:16]
    source_path = Path(
      f".yasps_constant/cg_metal_{source_hash}.metal"
    )
    library_path = Path(
      f".yasps_constant/cg_metal_{source_hash}.metallib"
    )
    if not library_path.exists():
      source_path.write_text(source, encoding="utf-8")
      gpuarray.compile_metal([source_path], library_path)

    self.__metal_spmv_kernels = {}
    for row_size, col_size in self.__saved_block_dimensions:
      self.__metal_spmv_kernels[(row_size, col_size)] = (
        gpuarray.MetalKernel(
          library_path,
          f"spmv_blocks_{row_size}_{col_size}_metal"
        )
      )
    self.__metal_block_jacobi_kernel = gpuarray.MetalKernel(
      library_path,
      "block_jacobi_metal"
    )
    self.__metal_dot_kernel = gpuarray.MetalKernel(
      library_path,
      "dot_product_partial_metal"
    )
    self.__metal_sum_kernel = gpuarray.MetalKernel(
      library_path,
      "sum_partial_metal"
    )
    self.__metal_vec_add_kernel = gpuarray.MetalKernel(
      library_path,
      "vec_add_with_scalar_metal"
    )
    self.__metal_cg_direction_kernel = gpuarray.MetalKernel(
      library_path,
      "cg_direction_from_state_metal"
    )
    self.__metal_cg_update_kernel = gpuarray.MetalKernel(
      library_path,
      "cg_solution_residual_from_denominator_metal"
    )
    self.__metal_cg_finalize_kernel = gpuarray.MetalKernel(
      library_path,
      "cg_finalize_iteration_metal"
    )
    self.__metal_fill_kernel = gpuarray.MetalKernel(
      library_path,
      "fill_float_metal"
    )
    self.__cg_kernel = self.__metal_spmv_kernels

  def __metalDispatch(
    self,
    kernel,
    arguments,
    grid_size,
    threadgroup_size,
    dispatches=None,
  ):
    if dispatches is None:
      kernel.dispatch(arguments, grid_size, threadgroup_size)
    else:
      dispatches.append(
        (kernel, arguments, grid_size, threadgroup_size)
      )

  def __metalFill(self, output, value, dispatches=None):
    if output.size == 0:
      return
    self.__metalDispatch(
      self.__metal_fill_kernel,
      [
        output,
        np.float32(value),
        np.uint32(output.size),
        self.__cg_status,
      ],
      output.size,
      256,
      dispatches,
    )

  def __metalVecAdd(
    self,
    left,
    right,
    output,
    scalar,
    dispatches=None,
  ):
    self.__metalDispatch(
      self.__metal_vec_add_kernel,
      [
        left,
        right,
        output,
        np.float32(scalar),
        np.uint32(output.size),
        self.__cg_status,
      ],
      output.size,
      256,
      dispatches,
    )

  def __appendMetalDot(self, left, right, dispatches):
    count = left.size
    group_count = (count + 255) // 256
    if self.__dot_scratch_a.size < group_count:
      self.__dot_scratch_a = gpuarray.empty(
        group_count,
        dtype=np.float32
      )
      self.__dot_scratch_b = gpuarray.empty(
        group_count,
        dtype=np.float32
      )
    self.__metalDispatch(
      self.__metal_dot_kernel,
      [
        left,
        right,
        self.__dot_scratch_a,
        np.uint32(count),
        self.__cg_status,
      ],
      group_count * 256,
      256,
      dispatches,
    )
    current = self.__dot_scratch_a
    temporary = self.__dot_scratch_b
    current_count = group_count
    while current_count > 1:
      next_count = (current_count + 255) // 256
      self.__metalDispatch(
        self.__metal_sum_kernel,
        [
          current,
          temporary,
          np.uint32(current_count),
          self.__cg_status,
        ],
        next_count * 256,
        256,
        dispatches,
      )
      current, temporary = temporary, current
      current_count = next_count
    return current

  def __metalSpmv(
    self,
    block_values,
    block_positions,
    block_values_start,
    block_counts,
    block_dimensions,
    x,
    y,
    dispatches=None,
  ):
    positions_start = 0
    for dimension_index in range(len(block_dimensions) // 2):
      row_size = int(block_dimensions[dimension_index * 2])
      col_size = int(block_dimensions[dimension_index * 2 + 1])
      block_count = int(block_counts[dimension_index])
      if block_count > 0:
        self.__metalDispatch(
          self.__metal_spmv_kernels[(row_size, col_size)],
          [
            block_values,
            np.uint32(block_values_start[dimension_index]),
            block_positions,
            np.uint32(positions_start),
            np.uint32(block_count),
            x,
            y,
            self.__cg_status,
          ],
          block_count,
          32,
          dispatches,
        )
      positions_start += block_count

  def __metalSpmvWithSystem(
    self,
    block_values,
    block_positions,
    block_values_start,
    block_counts,
    block_dimensions,
    block_values_dynamic,
    block_positions_dynamic,
    block_values_start_dynamic,
    block_counts_dynamic,
    block_dimensions_dynamic,
    x,
    y,
    dispatches=None,
  ):
    self.__metalSpmv(
      block_values,
      block_positions,
      block_values_start,
      block_counts,
      block_dimensions,
      x,
      y,
      dispatches,
    )
    self.__metalSpmv(
      block_values_dynamic,
      block_positions_dynamic,
      block_values_start_dynamic,
      block_counts_dynamic,
      block_dimensions_dynamic,
      x,
      y,
      dispatches,
    )

  def __metalBlockJacobi(
    self,
    diagonal_inverse,
    x,
    y,
    block_starts,
    block_counts,
    block_sizes,
    segment_starts,
    dispatches=None,
  ):
    for attribute_index in range(len(block_sizes)):
      block_count = int(block_counts[attribute_index])
      block_size = int(block_sizes[attribute_index])
      if block_count == 0:
        continue
      block_start = int(block_starts[attribute_index])
      segment_start = int(segment_starts[attribute_index])
      matrix_count = block_count * block_size * block_size
      vector_count = block_count * block_size
      self.__metalDispatch(
        self.__metal_block_jacobi_kernel,
        [
          diagonal_inverse[
            block_start:block_start + matrix_count
          ],
          x[segment_start:segment_start + vector_count],
          y[segment_start:segment_start + vector_count],
          np.uint32(block_count),
          np.uint32(block_size),
          self.__cg_status,
        ],
        block_count,
        32,
        dispatches,
      )

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(int(x.gpudata))

  @timed("solverKernel.computeSolution")
  def computeSolution(self,
    maxIteration: int,
    threshold: float,
    block_values: gpuarray.GPUArray,
    block_positions: gpuarray.GPUArray,
    block_values_start: List[int],
    block_counts: List[int],
    block_dimensions: List[int],
    block_values_dynamic: gpuarray.GPUArray,
    block_positions_dynamic: gpuarray.GPUArray,
    block_values_start_dynamic: List[int],
    block_counts_dynamic: List[int],
    block_dimensions_dynamic: List[int],
    diagonal: gpuarray.GPUArray,
    diagonal_blocks_inverse: gpuarray.GPUArray,
    diagonal_blocks_start: List[int],
    diagonal_blocks_count: List[int],
    diagonal_blocks_size: List[int],
    gradient_segments_start: List[int],
    num_attributes: int,
    gradient: gpuarray.GPUArray,
    d_p1_b: gpuarray.GPUArray,
    d_r: gpuarray.GPUArray,
    d_c: gpuarray.GPUArray,
    d_q: gpuarray.GPUArray,
    d_s: gpuarray.GPUArray,
    solution: gpuarray.GPUArray,
    initial_guess: gpuarray.GPUArray
  ):
    self.__context.useDefaultContext()
    # self.__context.useNamedContext("solver") # 14624

    if is_metal():
      start_time = time.perf_counter()
      self.__cg_status.set(np.zeros(2, dtype=np.int32))
      cuda.memcpy_dtod(
        solution.gpudata,
        initial_guess.gpudata,
        solution.nbytes
      )
      initial_residual_dispatches = []
      self.__metalFill(d_r, 0.0, initial_residual_dispatches)
      self.__metalSpmvWithSystem(
        block_values,
        block_positions,
        block_values_start,
        block_counts,
        block_dimensions,
        block_values_dynamic,
        block_positions_dynamic,
        block_values_start_dynamic,
        block_counts_dynamic,
        block_dimensions_dynamic,
        initial_guess,
        d_r,
        initial_residual_dispatches,
      )
      self.__metalVecAdd(
        gradient,
        d_r,
        d_r,
        -1.0,
        initial_residual_dispatches,
      )
      self.__metalBlockJacobi(
        diagonal_blocks_inverse,
        d_r,
        d_c,
        diagonal_blocks_start,
        diagonal_blocks_count,
        diagonal_blocks_size,
        gradient_segments_start,
        initial_residual_dispatches,
      )
      delta_new_result = self.__appendMetalDot(
        d_r,
        d_c,
        initial_residual_dispatches,
      )
      gpuarray.dispatch_batch(
        initial_residual_dispatches,
        "cg_initial_residual",
      )
      delta_new = float(delta_new_result.get()[0])
      use_identity_preconditioner = (
        not np.isfinite(delta_new)
        or delta_new <= 0.0
      )
      result = 0
      if use_identity_preconditioner:
        residual_norm_dispatches = []
        residual_norm_result = self.__appendMetalDot(
          d_r,
          d_r,
          residual_norm_dispatches,
        )
        gpuarray.dispatch_batch(
          residual_norm_dispatches,
          "cg_fallback_residual_norm",
        )
        residual_norm = float(residual_norm_result.get()[0])
        if os.environ.get("YASPS_SOLVER_DIAGNOSTICS") == "1":
          print(
            "CG preconditioner fallback:"
            f" preconditioned_residual={delta_new:.9e},"
            f" residual_norm_squared={residual_norm:.9e}"
          )
        if not np.isfinite(residual_norm):
          if os.environ.get("YASPS_SOLVER_DIAGNOSTICS") == "1":
            for label, buffer in (
              ("gradient", gradient),
              ("initial_guess", initial_guess),
              ("residual", d_r),
              ("preconditioned_residual", d_c),
              ("diagonal", diagonal),
              ("diagonal_blocks_inverse", diagonal_blocks_inverse),
              ("block_values", block_values),
              ("block_values_dynamic", block_values_dynamic),
            ):
              values = buffer.get().reshape(-1)
              indices = np.flatnonzero(~np.isfinite(values))
              if indices.size > 0:
                print(
                  f"CG non-finite buffer: {label},"
                  f" count={indices.size},"
                  f" first_indices={indices[:8].tolist()}"
                )
            print(
              "CG system layout:"
              f" static_starts={block_values_start},"
              f" static_counts={block_counts},"
              f" static_dimensions={block_dimensions},"
              f" dynamic_starts={block_values_start_dynamic},"
              f" dynamic_counts={block_counts_dynamic},"
              f" dynamic_dimensions={block_dimensions_dynamic},"
              f" diagonal_starts={diagonal_blocks_start},"
              f" diagonal_counts={diagonal_blocks_count},"
              f" diagonal_sizes={diagonal_blocks_size},"
              f" gradient_starts={gradient_segments_start}"
            )
          result = -4
        elif residual_norm > 0.0:
          cuda.memcpy_dtod(
            d_c.gpudata,
            d_r.gpudata,
            d_c.nbytes,
          )
          delta_new = residual_norm

      delta_zero = delta_new
      relative_tolerance = threshold * delta_zero
      if result == 0 and delta_new > relative_tolerance:
        result = maxIteration + 1
        self.__cg_state.set(np.array(
          [
            delta_new,
            np.float32(relative_tolerance),
            0.0,
            0.0,
          ],
          dtype=np.float32,
        ))
        iteration = 1
        update_direction = False
        while iteration <= maxIteration:
          chunk_end = min(iteration + 3, maxIteration)
          iteration_dispatches = []
          for current_iteration in range(
            iteration,
            chunk_end + 1,
          ):
            if update_direction:
              self.__metalDispatch(
                self.__metal_cg_direction_kernel,
                [
                  d_s,
                  d_c,
                  self.__cg_state,
                  self.__cg_status,
                  np.uint32(1),
                  np.uint32(solution.size),
                ],
                solution.size,
                256,
                iteration_dispatches,
              )
            update_direction = True
            self.__metalFill(d_q, 0.0, iteration_dispatches)
            self.__metalSpmvWithSystem(
              block_values,
              block_positions,
              block_values_start,
              block_counts,
              block_dimensions,
              block_values_dynamic,
              block_positions_dynamic,
              block_values_start_dynamic,
              block_counts_dynamic,
              block_dimensions_dynamic,
              d_c,
              d_q,
              iteration_dispatches,
            )
            denominator_result = self.__appendMetalDot(
              d_c,
              d_q,
              iteration_dispatches,
            )
            self.__metalDispatch(
              self.__metal_cg_update_kernel,
              [
                solution,
                d_c,
                d_r,
                d_q,
                denominator_result,
                self.__cg_state,
                self.__cg_status,
                np.uint32(current_iteration),
                np.uint32(solution.size),
              ],
              solution.size,
              256,
              iteration_dispatches,
            )
            if use_identity_preconditioner:
              self.__metalVecAdd(
                d_r,
                d_r,
                d_s,
                0.0,
                iteration_dispatches,
              )
            else:
              self.__metalBlockJacobi(
                diagonal_blocks_inverse,
                d_r,
                d_s,
                diagonal_blocks_start,
                diagonal_blocks_count,
                diagonal_blocks_size,
                gradient_segments_start,
                iteration_dispatches,
              )
            delta_new_result = self.__appendMetalDot(
              d_r,
              d_s,
              iteration_dispatches,
            )
            self.__metalDispatch(
              self.__metal_cg_finalize_kernel,
              [
                delta_new_result,
                self.__cg_state,
                self.__cg_status,
                np.uint32(use_identity_preconditioner),
                np.uint32(current_iteration),
              ],
              1,
              1,
              iteration_dispatches,
            )
          gpuarray.dispatch_batch(
            iteration_dispatches,
            "cg_iterations",
          )
          status_values = self.__cg_status.get()
          status_code = int(status_values[0])
          if status_code == 0:
            iteration = chunk_end + 1
            update_direction = True
            continue
          if status_code != -2147483647:
            result = status_code
            break

          fallback_iteration = int(status_values[1])
          preconditioned_residual = float(
            self.__cg_state.get()[3]
          )
          self.__cg_status.set(np.zeros(2, dtype=np.int32))
          residual_norm_dispatches = []
          residual_norm_result = self.__appendMetalDot(
            d_r,
            d_r,
            residual_norm_dispatches,
          )
          gpuarray.dispatch_batch(
            residual_norm_dispatches,
            "cg_convergence_residual_norm",
          )
          residual_norm = float(residual_norm_result.get()[0])
          if not np.isfinite(residual_norm):
            result = -4
            break
          if residual_norm == 0.0:
            result = fallback_iteration
            break
          if os.environ.get("YASPS_SOLVER_DIAGNOSTICS") == "1":
            print(
              "CG preconditioner fallback:"
              f" iteration={fallback_iteration},"
              f" preconditioned_residual="
              f"{preconditioned_residual:.9e},"
              f" residual_norm_squared={residual_norm:.9e}"
            )
          use_identity_preconditioner = True
          relative_tolerance = threshold * residual_norm
          self.__cg_state.set(np.array(
            [
              residual_norm,
              np.float32(relative_tolerance),
              0.0,
              0.0,
            ],
            dtype=np.float32,
          ))
          cuda.memcpy_dtod(
            d_c.gpudata,
            d_r.gpudata,
            d_c.nbytes,
          )
          iteration = fallback_iteration + 1
          update_direction = False
          if iteration > maxIteration:
            break

      if result == -5:
        self.__cg_status.set(np.zeros(2, dtype=np.int32))
        self.__metalFill(solution, 0.0)
        self.__metalVecAdd(
          solution,
          gradient,
          solution,
          0.5
        )
        return result
      if result < 0:
        return result
      elapsed_time_ms = (
        time.perf_counter() - start_time
      ) * 1000.0
      print(f"Solver converged in {result} iterations")
      print(f"Solver time: {elapsed_time_ms:.5f} ms")
      return 0

    start_call = cuda.Event()

    end_call = cuda.Event()
    start_call.record()
    assert self.__cg_kernel is not None, "Kernel not initialized. Call __init_kernel first."
    result = self.__cg_kernel(
      maxIteration,
      threshold,
      self.__to_void_p(block_values),
      self.__to_void_p(block_positions),
      np.array(block_values_start, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(block_counts, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(block_dimensions, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      len(block_dimensions) // 2,
      self.__to_void_p(block_values_dynamic),
      self.__to_void_p(block_positions_dynamic),
      np.array(block_values_start_dynamic, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(block_counts_dynamic, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(block_dimensions_dynamic, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      len(block_dimensions_dynamic) // 2,
      self.__to_void_p(diagonal),
      self.__to_void_p(diagonal_blocks_inverse),
      np.array(diagonal_blocks_start, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(diagonal_blocks_count, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(diagonal_blocks_size, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(gradient_segments_start, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      num_attributes,
      self.__to_void_p(gradient),
      gradient.shape[0],
      self.__to_void_p(d_p1_b),
      self.__to_void_p(d_r),
      self.__to_void_p(d_c),
      self.__to_void_p(d_q),
      self.__to_void_p(d_s),
      self.__to_void_p(solution),
      self.__to_void_p(initial_guess)
    )
    # Record the end event
    end_call.record()
    # Wait for the end event to complete
    end_call.synchronize()
    if result == -1:
      # the kernel failed in the first iteration
      # we set the solution to gradient instead
      solution.set(gradient)
      result = -result
      print("Kernel failed in the first iteration")
      return -1
    elif result == -2:
      raise RuntimeError("solverKernel.computeSolution: CUDA context mismatch or not set")
      return -2
    elif result == -3:
      # the kernel failed to set the context
      raise RuntimeError("solverKernel.computeSolution: CUDA error during kernel execution")
      return -3
      # exit()
    elif result == -5:
      solution.set(0.5 * gradient)
      return result
    elif result < 0:
      return result
    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_call.time_till(end_call)
    print(f"Solver converged in {result} iterations")
    print(f"Solver time: {elapsed_time_ms:.5f} ms")
    return 0

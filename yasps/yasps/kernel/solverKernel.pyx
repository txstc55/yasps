# cython: language_level=3
from __future__ import annotations
from typing import List, Tuple, Set
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
import pycuda.driver as cuda
import os
import hashlib
import json
from yasps.helper import timed
from yasps.context import context

class solverKernel:
  def __init__(self, blockDimensions: List[int]):
    self.__max_row_size = 0
    self.__cg_kernel = None
    self.__saved_block_dimensions = set([])
    self.__context = context()

  def updateBlockDimensions(self, blockDimensions: List[int]):
    self.__init_kernel(blockDimensions)


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
      dimension_to_text = [f'{dim[0]}_{dim[1]}' for dim in blockDimensionsTuplesSet]
      dimension_to_text = '__'.join(dimension_to_text)
      file_original_name = f".yasps_constant/cg_dims_{dimension_to_text}"
      file_hashed_name = f".yasps_constant/cg_dims_{int(hashlib.sha256(dimension_to_text.encode('utf-8')).hexdigest(), 16)}"
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
                ctypes.c_void_p  # last 5 solution (device pointer)
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
                            double* d_p1_b, // for the computation of P^-1 * b
                            double* d_r, // for residual
                            double* d_c,
                            double* d_q,
                            double* d_s,
                            double* solution,
                            double* last_5_solution
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
  CUDA_CHECK_ERROR(cudaMemcpy(solution, last_5_solution, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));

  // now we compute P^-1 * b where P is the preconditioner
  // jacobiPreconditioner<<<MATRIX_SIZE / 32 + 1, 32>>>(diagonal, gradient, d_p1_b, MATRIX_SIZE);
  blockJacobiPreconditioner(
    diagonalBlockInverse,
    gradient,
    d_p1_b,
    diagonalBlocksStart,
    diagonalBlocksCount,
    diagonalBlocksSize,
    gradientSegmentsStart,
    numAttributes,
    preconditioner_streams
  );
  // delta0 = b * A^-1 b
  double* d_delta0; // for device
  double h_delta_0; // for host
  cudaMalloc(&d_delta0, sizeof(double));
  cudaMemset(d_delta0, 0, sizeof(double));
  // compute delta0
  dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_p1_b, gradient, d_delta0, MATRIX_SIZE);
  CUDA_CHECK_ERROR(cudaMemcpy(&h_delta_0, d_delta0, sizeof(double), cudaMemcpyDeviceToHost));

  // set residual equal to the gradient since our initial guess is 0
  // CUDA_CHECK_ERROR(cudaMemcpy(d_r, gradient, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));


  // we need to compute d_r = gradient - A * last_5_solution
  spmvWithSystem(block_values,
                 block_positions,
                 block_values_start,
                 block_counts,
                 block_values_dynamic,
                 block_positions_dynamic,
                 block_values_start_dynamic,
                 block_counts_dynamic,
                 last_5_solution,
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

  // check tolerance
  double relativeTolerance = threshold * h_delta_0;
  // printf("Initial residual %lf, relative tolerance: %lf\\n", h_delta_new, relativeTolerance);
  if (h_delta_new <= relativeTolerance){
    // printf("Converged in 0 iterations with residual %lf\\n", h_delta_new);
    for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
      cudaStreamDestroy(streams[i]);
    }
    for (unsigned int i = 0; i < numAttributes; ++i) {
      cudaStreamDestroy(preconditioner_streams[i]);
    }

    // free
    cudaFree(d_delta0);
    cudaFree(d_delta_new);
    return 0;
  }
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

    if (h_alpha < 0){
      printf("Non SPD matrix detected in %d iterations with residual %lf and alpha %lf\\n", iteration, h_delta_new, h_alpha);
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }

      // free
      cudaFree(d_delta0);
      cudaFree(d_delta_new);
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

    // s = P^-1 * r
    // jacobiPreconditioner<<<(MATRIX_SIZE + 255) / 256, 256>>>(diagonal, d_r, d_s, MATRIX_SIZE);
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

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    h_delta_old = h_delta_new;
    // delta_new = r * s
    cudaMemset(d_delta_new, 0, sizeof(double));
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_s, d_delta_new, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(&h_delta_new, d_delta_new, sizeof(double), cudaMemcpyDeviceToHost);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // c = s + (delta_new / delta_old) * c
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_s, d_c, d_c, h_delta_new / h_delta_old, MATRIX_SIZE);
    // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    if (h_delta_new <= relativeTolerance){
      printf("Converged in %d iterations with residual %lf\\n", iteration, h_delta_new);
      for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
        cudaStreamDestroy(streams[i]);
      }
      for (unsigned int i = 0; i < numAttributes; ++i) {
        cudaStreamDestroy(preconditioner_streams[i]);
      }

      // free
      cudaFree(d_delta0);
      cudaFree(d_delta_new);
      cudaFree(d_alpha);
      return iteration;
    }
  }
  // after the two for‑loops that synchronize the streams
  for (unsigned int i = 0; i < NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC; ++i) {
    cudaStreamDestroy(streams[i]);
  }

  for (unsigned int i = 0; i < numAttributes; ++i) {
    cudaStreamDestroy(preconditioner_streams[i]);
  }

  // free
  cudaFree(d_delta0);
  cudaFree(d_delta_new);
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
        ctypes.c_void_p  # last 5 solution (device pointer)
      ]
      data = []
      with open(".yasps_constant/cg_dimension_to_file.json", "r", encoding="utf-8") as f:
        data = json.load(f)
      for item in data:
        if item["file_hashed_name"] == file_hashed_name:
          # already exists
          return
      data.append({"dimensions": [dim for dim in self.__saved_block_dimensions], "file_hashed_name": file_hashed_name, "file_original_name": file_original_name})
      with open(".yasps_constant/cg_dimension_to_file.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

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
    last_5_solution: gpuarray.GPUArray
  ):
    self.__context.useDefaultContext()
    # self.__context.useNamedContext("solver") # 14624

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
      self.__to_void_p(last_5_solution)
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

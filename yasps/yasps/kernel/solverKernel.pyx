# cython: language_level=3
from __future__ import annotations
from typing import List, Tuple
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
import pycuda.driver as cuda


class solverKernel:
  def __init__(self, blockDimensions: List[Tuple[int, int]]):
    self.__kernelString: str = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

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
                                              const double *x, double *y) {
  // multiply a row
  for (int i = 0; i < BLOCK_ROW_SIZE; i++) {
    for (int j = 0; j < BLOCK_COL_SIZE; j++) {
      y[i] += blockValues[i * BLOCK_COL_SIZE + j] * x[j];
    }
  }
}

template<unsigned int BLOCK_ROW_SIZE, unsigned int BLOCK_COL_SIZE>
__device__ __forceinline__ void
blockMultiplyTranspose(const double *blockValues, const double *x, double *y) {
  double temp[BLOCK_COL_SIZE] = {.0};
  for (int i = 0; i < BLOCK_COL_SIZE; i++) {
  for (int j = 0; j < BLOCK_ROW_SIZE; j++) {
      temp[i] += blockValues[j * BLOCK_COL_SIZE + i] * x[j];
    }
  }

#pragma unroll
  for (int i = 0; i < BLOCK_COL_SIZE; i++) {
    atomicAdd(y + i, temp[i]);
  }
}

// computes Ax=y where A does not contain any diagonal blocks
template<unsigned int BLOCK_ROW_SIZE, unsigned int BLOCK_COL_SIZE>
__global__ void spmvOffDiagonalBlocks(const double *blockValues,
                                      const unsigned int VALUE_START, // where in the block values does this dimension's block start
                                      const unsigned int* positions, // the coordinate of this block
                                      const unsigned int POSITIONS_START, // where does the positions start for this dimension
                                      const unsigned int POSITIONS_END, // where does the positions end for this dimension
                                      const double *x, // the Ax = y
                                      double *y) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; // we first get the id of this thread
  int tid = threadIdx.x;
  __shared__ double allResults[BLOCK_ROW_SIZE * 32]; // accumulate the multiplied result
  __shared__ unsigned int rows[32];
  __shared__ unsigned int cols[32];
  if (tid == 0){
    // initialize allResults to 0
    for (unsigned int i = 0; i < BLOCK_ROW_SIZE * 32; i++){
      allResults[i] = 0.0;
    }
  }
  __syncthreads(); // synchronize all threads after initialization
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
      double sum[BLOCK_ROW_SIZE] = {}; // initialize the sum
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
                    const double* x, // Ax = y
                    double* y){
  unsigned int positions_start = 0;
  unsigned int positions_end = 0;
'''
    index: int = 0
    for dimension in blockDimensions:
      blockRowSize = dimension[0]
      blockColSize = dimension[1]
      self.__kernelString += f'''
  positions_end = positions_start + block_counts[{index}];
  spmvOffDiagonalBlocks<{blockRowSize}, {blockColSize}><<<(block_counts[{index}] + 32) / 32, 32>>>(block_values, block_values_start[{index}], block_positions, positions_start, positions_end, x, y);
  positions_start = positions_end;
'''
      index += 1
    self.__kernelString += '''
}

__global__ void jacobiPreconditioner(const double* diagonal, const double* x, double* y, unsigned int N){
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < N){
    y[id] = x[id] / (abs(diagonal[id]) < 1e-6 ? 1.0 : diagonal[id]);
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
int computeSolution(CUcontext ctx,
                             unsigned int maxIteration,
                             double threshold,
                             const double* block_values,
                             const unsigned int* block_positions,
                             const unsigned int* block_values_start,
                             const unsigned int* block_counts,
                             const double* diagonal,
                             const double* gradient,
                             const unsigned int MATRIX_SIZE,
                             double* d_p1_b, // for the computation of P^-1 * b
                             double* d_r, // for residual
                             double* d_c,
                             double* d_q,
                             double* d_s,
                             double* solution) {
  // Instead, retrieve the current context (if necessary)
  CUcontext current_ctx;
  cuCtxGetCurrent(&current_ctx);

  // Optionally, compare with the passed context
  if (current_ctx != ctx) {
    printf("Context mismatch\\n");
    return -2;
  }
  // Set the provided context as the current context
  CUresult res = cuCtxSetCurrent(ctx);
  if (res != CUDA_SUCCESS) {
    // Handle error
    printf("Failed to set CUDA context\\n");
    return -1;
  }

  // first resetting the solution
  cudaMemset(solution, 0, MATRIX_SIZE * sizeof(double));

  // now we compute P^-1 * b where P is the preconditioner
  jacobiPreconditioner<<<MATRIX_SIZE / 32 + 1, 32>>>(diagonal, gradient, d_p1_b, MATRIX_SIZE);
  // delta0 = b * A^-1 b
  double* d_delta0; // for device
  double h_delta_0; // for host
  cudaMalloc(&d_delta0, sizeof(double));
  cudaMemset(d_delta0, 0, sizeof(double));
  // compute delta0
  dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_p1_b, gradient, d_delta0, MATRIX_SIZE);
  CUDA_CHECK_ERROR(cudaMemcpy(&h_delta_0, d_delta0, sizeof(double), cudaMemcpyDeviceToHost));

  // set residual equal to the gradient since our initial guess is 0
  CUDA_CHECK_ERROR(cudaMemcpy(d_r, gradient, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));

  // c = P^-1 * r
  cudaMemset(d_c, 0, MATRIX_SIZE * sizeof(double));
  jacobiPreconditioner<<<(MATRIX_SIZE + 255) / 256, 256>>>(diagonal, d_r, d_c, MATRIX_SIZE);

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
    return 0;
  }
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
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
                   d_c,
                   d_q);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemset(d_alpha, 0, sizeof(double));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_c, d_q, d_alpha, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(&h_alpha, d_alpha, sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    h_alpha = h_delta_new / h_alpha;

    // deltav = deltav + alpha * c
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(solution, d_c, solution, h_alpha, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // r = r - alpha * q
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_q, d_r, -h_alpha, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // s = P^-1 * r
    jacobiPreconditioner<<<(MATRIX_SIZE + 255) / 256, 256>>>(diagonal, d_r, d_s, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    h_delta_old = h_delta_new;
    // delta_new = r * s
    cudaMemset(d_delta_new, 0, sizeof(double));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_s, d_delta_new, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    cudaMemcpy(&h_delta_new, d_delta_new, sizeof(double), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // c = s + (delta_new / delta_old) * c
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_s, d_c, d_c, h_delta_new / h_delta_old, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    if (h_delta_new <= relativeTolerance){
      // printf("Converged in %d iterations with residual %lf\\n", iteration, h_delta_new);
      return iteration;
    }
  }
  return maxIteration + 1;
}

} // close the extern "C"
'''
    # ok now we compile the kernel by saving it to a file and then calling nvcc
    file_name = f"cg_{'_'.join([f'{x[0]}_{x[1]}' for x in blockDimensions])}"
    f = open(f"{file_name}.cu", 'w')
    f.write(self.__kernelString)
    f.close()

    # now we compile the kernel
    import os
    os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda")
    self.__cg_kernel = ctypes.CDLL(f"./{file_name}.so").computeSolution
    self.__cg_kernel.argtypes = [
        ctypes.c_void_p, # cuda context
        ctypes.c_uint,   # maxIteration
        ctypes.c_double, # threshold
        ctypes.c_void_p, # block_values (device pointer to double)
        ctypes.c_void_p, # block_positions (device pointer to unsigned int)
        ctypes.POINTER(ctypes.c_uint), # block_values_start (unsigned int list from numpy array)
        ctypes.POINTER(ctypes.c_uint), # block_counts (unsigned int list from numpy)
        ctypes.c_void_p, # diagonal (device pointer to double)
        ctypes.c_void_p, # gradient (device pointer to double)
        ctypes.c_uint,   # MATRIX_SIZE
        ctypes.c_void_p, # d_p1_b (device pointer)
        ctypes.c_void_p, # d_r (device pointer)
        ctypes.c_void_p, # d_c (device pointer)
        ctypes.c_void_p, # d_q (device pointer)
        ctypes.c_void_p, # d_s (device pointer)
        ctypes.c_void_p  # solution (device pointer)
    ]

  def __to_void_p(self, x: gpuarray.GPUArray):
    return ctypes.c_void_p(int(x.gpudata))

  def computeSolution(self, cuda_context, maxIteration, threshold, block_values: gpuarray.GPUArray, block_positions: gpuarray.GPUArray, block_values_start: List[int], block_counts: List[int], diagonal: gpuarray.GPUArray, gradient: gpuarray.GPUArray, d_p1_b: gpuarray.GPUArray, d_r: gpuarray.GPUArray, d_c: gpuarray.GPUArray, d_q: gpuarray.GPUArray, d_s: gpuarray.GPUArray, solution: gpuarray.GPUArray):
    start_call = cuda.Event()
    end_call = cuda.Event()
    start_call.record()
    result = self.__cg_kernel(
      cuda_context,
      maxIteration,
      threshold,
      self.__to_void_p(block_values),
      self.__to_void_p(block_positions),
      np.array(block_values_start, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      np.array(block_counts, dtype = np.uint32).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
      self.__to_void_p(diagonal),
      self.__to_void_p(gradient),
      gradient.shape[0],
      self.__to_void_p(d_p1_b),
      self.__to_void_p(d_r),
      self.__to_void_p(d_c),
      self.__to_void_p(d_q),
      self.__to_void_p(d_s),
      self.__to_void_p(solution)
    )
    # Record the end event
    end_call.record()
    # Wait for the end event to complete
    end_call.synchronize()
    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_call.time_till(end_call)
    print(f"Solver converged in {result} iterations")
    print(f"Solver time: {elapsed_time_ms:.5f} ms")

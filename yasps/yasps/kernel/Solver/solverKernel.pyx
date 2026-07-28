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
import fcntl
import subprocess
import threading
from yasps.helper import timed
from yasps.context import context

SOLVER_KERNEL_CACHE_VERSION = "v6_content_addressed"
SOLVER_NVCC_COMMAND_TEMPLATE = (
  "nvcc", "-Xcompiler", "-fPIC", "-shared", "-o", "<OUTPUT>", "<SOURCE>",
  "-O3", "-arch=sm_89", "-cudart=shared", "-lcuda",
  "--expt-relaxed-constexpr", "-std=c++17",
)

def _stable_content_signature(payload) -> str:
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()

def _atomic_write_text(path: str, text: str) -> None:
  build_token = f"{os.getpid()}_{threading.get_ident()}"
  temporary_path = f"{path}.tmp_{build_token}"
  try:
    with open(temporary_path, "w", encoding="utf-8") as f:
      f.write(text)
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary_path, path)
  finally:
    if os.path.exists(temporary_path):
      os.remove(temporary_path)

def _update_cache_index(cache_index_file: str, entry: dict) -> None:
  lock_path = f"{cache_index_file}.lock"
  with open(lock_path, "a+", encoding="utf-8") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
      try:
        with open(cache_index_file, "r", encoding="utf-8") as f:
          data = json.load(f)
      except (FileNotFoundError, json.JSONDecodeError):
        data = []
      if not isinstance(data, list):
        data = []
      data = [
        item for item in data
        if isinstance(item, dict) and item.get("file_hashed_name") != entry["file_hashed_name"]
      ]
      data.append(entry)
      data.sort(key=lambda item: item.get("file_hashed_name", ""))
      _atomic_write_text(cache_index_file, json.dumps(data, indent=2, sort_keys=True) + "\n")
    finally:
      fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

class solverKernel:
  def __init__(self, blockDimensions: List[int]):
    self.__max_row_size = 0
    self.__cg_kernel = None
    self.__cg_library = None
    self.__cleanup_streams = None
    self.__saved_block_dimensions = set([])
    self.__context = context()

  def __loadKernelLibrary(self, file_hashed_name: str) -> None:
    library = ctypes.CDLL(f"{file_hashed_name}.so")
    kernel = library.computeSolution
    kernel.argtypes = [
      ctypes.c_uint,   # maxIteration
      ctypes.c_double, # threshold
      ctypes.c_void_p, # block_values (device pointer to double)
      ctypes.c_void_p, # block_positions (device pointer to unsigned int)
      ctypes.POINTER(ctypes.c_uint), # block_values_start
      ctypes.POINTER(ctypes.c_uint), # block_counts
      ctypes.POINTER(ctypes.c_uint), # block_dimensions
      ctypes.c_uint,   # NUM_BLOCK_DIMENSIONS
      ctypes.c_void_p, # block_values_dynamic
      ctypes.c_void_p, # block_positions_dynamic
      ctypes.POINTER(ctypes.c_uint), # block_values_start_dynamic
      ctypes.POINTER(ctypes.c_uint), # block_counts_dynamic
      ctypes.POINTER(ctypes.c_uint), # block_dimensions_dynamic
      ctypes.c_uint,   # NUM_BLOCK_DIMENSIONS_DYNAMIC
      ctypes.c_void_p, # diagonal
      ctypes.c_void_p, # diagonalBlockInverse
      ctypes.POINTER(ctypes.c_uint), # diagonalBlocksStart
      ctypes.POINTER(ctypes.c_uint), # diagonalBlocksCount
      ctypes.POINTER(ctypes.c_uint), # diagonalBlocksSize
      ctypes.POINTER(ctypes.c_uint), # gradientSegmentsStart
      ctypes.c_int,    # numAttributes
      ctypes.c_void_p, # gradient
      ctypes.c_uint,   # MATRIX_SIZE
      ctypes.c_void_p, # d_p1_b
      ctypes.c_void_p, # d_r
      ctypes.c_void_p, # d_c
      ctypes.c_void_p, # d_q
      ctypes.c_void_p, # d_s
      ctypes.c_void_p, # solution
      ctypes.c_void_p, # initial_guess
      ctypes.c_int,    # zeroInitialGuess
      ctypes.c_void_p  # cg_scalars
    ]
    kernel.restype = ctypes.c_int
    cleanup_streams = library.cleanupSolverStreams
    cleanup_streams.argtypes = []
    cleanup_streams.restype = ctypes.c_int

    # Resolve the replacement fully before cleaning the active library's
    # thread-local stream pool. A bad cache entry must leave the old kernel usable.
    self.cleanupStreams()
    self.__cg_library = library
    self.__cg_kernel = kernel
    self.__cleanup_streams = cleanup_streams

  def cleanupStreams(self) -> None:
    if self.__cleanup_streams is None:
      return
    self.__context.useDefaultContext()
    result = self.__cleanup_streams()
    if result != 0:
      raise RuntimeError(f"solverKernel.cleanupStreams: CUDA error {result}")

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
      candidate_block_dimensions = self.__saved_block_dimensions | blockDimensionsTuplesSet
      max_modded_row_size = (max(dim[0] for dim in candidate_block_dimensions) + 2) // 3 * 3
      os.makedirs(".yasps_constant", exist_ok=True)
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

namespace {
thread_local std::vector<cudaStream_t> solver_spmv_streams;
thread_local std::vector<cudaStream_t> solver_preconditioner_streams;
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
      for dim in sorted(candidate_block_dimensions):
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
      for dim in sorted(candidate_block_dimensions):
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

// Persistent cg_scalars layout. Slots 6 and 7 form the only host status packet
// copied after each iteration; the remaining recurrence values stay on device.
enum CgScalarSlot : unsigned int {
  CG_DELTA_0 = 0,
  CG_DELTA_NEW = 1,
  CG_DENOMINATOR = 2,
  CG_DELTA_OLD = 3,
  CG_ALPHA = 4,
  CG_BETA = 5,
  CG_STATUS = 6,
  CG_STATUS_VALUE = 7,
  CG_SCALAR_COUNT = 8
};

constexpr double CG_STATUS_NON_SPD = -1.0;
constexpr double CG_STATUS_CONTINUE = 0.0;
constexpr double CG_STATUS_CONVERGED = 1.0;

// Alpha and beta use the same single double-precision divisions and dependency
// order as the former host recurrence. Moving them here only removes transfers.
__global__ void prepareCgIteration(double* cg_scalars) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const double denominator = cg_scalars[CG_DENOMINATOR];
    cg_scalars[CG_DELTA_OLD] = cg_scalars[CG_DELTA_NEW];
    if (!isfinite(denominator) || denominator <= 0.0) {
      cg_scalars[CG_ALPHA] = 0.0;
      cg_scalars[CG_STATUS] = CG_STATUS_NON_SPD;
      cg_scalars[CG_STATUS_VALUE] = denominator;
    } else {
      cg_scalars[CG_ALPHA] = cg_scalars[CG_DELTA_NEW] / denominator;
      cg_scalars[CG_STATUS] = CG_STATUS_CONTINUE;
      cg_scalars[CG_STATUS_VALUE] = cg_scalars[CG_DELTA_NEW];
    }
  }
}

// These updates are independent. Fusing their traversal can interleave stores,
// but preserves each element's arithmetic expression and data dependencies.
__global__ void updateSolutionAndResidual(double* solution,
                                          const double* c,
                                          double* r,
                                          const double* q,
                                          const double* cg_scalars,
                                          unsigned int n) {
  if (cg_scalars[CG_STATUS] == CG_STATUS_NON_SPD) {
    return;
  }
  const double alpha = cg_scalars[CG_ALPHA];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    solution[i] = solution[i] + c[i] * alpha;
    r[i] = r[i] - q[i] * alpha;
  }
}

__global__ void finishCgIteration(double* cg_scalars,
                                  double relativeTolerance) {
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      cg_scalars[CG_STATUS] != CG_STATUS_NON_SPD) {
    const double deltaNew = cg_scalars[CG_DELTA_NEW];
    cg_scalars[CG_BETA] = deltaNew / cg_scalars[CG_DELTA_OLD];
    cg_scalars[CG_STATUS] = deltaNew <= relativeTolerance
      ? CG_STATUS_CONVERGED
      : CG_STATUS_CONTINUE;
    cg_scalars[CG_STATUS_VALUE] = deltaNew;
  }
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

__global__ void updateDirection(const double* s,
                                double* c,
                                const double* cg_scalars,
                                unsigned int n) {
  if (cg_scalars[CG_STATUS] == CG_STATUS_NON_SPD) {
    return;
  }
  const double beta = cg_scalars[CG_BETA];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    c[i] = s[i] + c[i] * beta;
  }
}
extern "C" {
int cleanupSolverStreams() {
  cudaError_t first_error = cudaSuccess;
  for (cudaStream_t stream : solver_spmv_streams) {
    const cudaError_t error = cudaStreamDestroy(stream);
    if (first_error == cudaSuccess && error != cudaSuccess) {
      first_error = error;
    }
  }
  solver_spmv_streams.clear();
  for (cudaStream_t stream : solver_preconditioner_streams) {
    const cudaError_t error = cudaStreamDestroy(stream);
    if (first_error == cudaSuccess && error != cudaSuccess) {
      first_error = error;
    }
  }
  solver_preconditioner_streams.clear();
  return static_cast<int>(first_error);
}

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
                            double* initial_guess,
                            const int zeroInitialGuess,
                            double* cg_scalars
                            ) {
  std::vector<cudaStream_t>& spmv_streams = solver_spmv_streams;
  std::vector<cudaStream_t>& preconditioner_streams = solver_preconditioner_streams;
  const unsigned int required_spmv_streams = NUM_BLOCK_DIMENSIONS + NUM_BLOCK_DIMENSIONS_DYNAMIC;
  while (spmv_streams.size() < required_spmv_streams) {
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));
    spmv_streams.push_back(stream);
  }
  while (preconditioner_streams.size() < static_cast<unsigned int>(numAttributes)) {
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreate(&stream));
    preconditioner_streams.push_back(stream);
  }

  double* d_delta0 = cg_scalars + CG_DELTA_0;
  double* d_delta_new = cg_scalars + CG_DELTA_NEW;
  double* d_denominator = cg_scalars + CG_DENOMINATOR;
  double* d_status = cg_scalars + CG_STATUS;
  double initial_residuals[2];
  double iteration_status[2];
  double h_delta_0;
  double h_delta_new;
  CUDA_CHECK_ERROR(cudaMemsetAsync(cg_scalars, 0, CG_SCALAR_COUNT * sizeof(double)));

  if (zeroInitialGuess) {
    CUDA_CHECK_ERROR(cudaMemsetAsync(solution, 0, MATRIX_SIZE * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpyAsync(d_r, gradient, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
    blockJacobiPreconditioner(
      diagonalBlockInverse,
      gradient,
      d_c,
      diagonalBlocksStart,
      diagonalBlocksCount,
      diagonalBlocksSize,
      gradientSegmentsStart,
      numAttributes,
      preconditioner_streams
    );
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_c, gradient, d_delta0, MATRIX_SIZE);
    CUDA_CHECK_ERROR(cudaMemcpyAsync(d_delta_new, d_delta0, sizeof(double), cudaMemcpyDeviceToDevice));
  } else {
    CUDA_CHECK_ERROR(cudaMemcpyAsync(solution, initial_guess, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice));
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
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_p1_b, gradient, d_delta0, MATRIX_SIZE);

    CUDA_CHECK_ERROR(cudaMemsetAsync(d_r, 0, MATRIX_SIZE * sizeof(double)));
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
                   spmv_streams);
    vecAddWithScalar<<<(MATRIX_SIZE + 255) / 256, 256>>>(gradient, d_r, d_r, -1.0, MATRIX_SIZE);
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
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_c, d_delta_new, MATRIX_SIZE);
  }
  CUDA_CHECK_ERROR(cudaMemcpy(initial_residuals, cg_scalars, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  h_delta_0 = initial_residuals[0];
  h_delta_new = initial_residuals[1];

  // check tolerance
  if (!isfinite(h_delta_0) || h_delta_0 < 0.0 ||
      !isfinite(h_delta_new) || h_delta_new < 0.0) {
    printf("Invalid initial preconditioned residuals: delta_0=%lf, delta_new=%lf\\n",
           h_delta_0, h_delta_new);
    return -5;
  }
  double relativeTolerance = threshold * h_delta_0;
  if (!isfinite(relativeTolerance) || relativeTolerance < 0.0) {
    printf("Invalid relative tolerance: %lf\\n", relativeTolerance);
    return -5;
  }
  // printf("Initial residual %lf, relative tolerance: %lf\\n", h_delta_new, relativeTolerance);
  if (h_delta_new <= relativeTolerance){
    return 0;
  }

  for (unsigned int iteration = 1; iteration <= maxIteration; iteration++){
    // q = A * c
    CUDA_CHECK_ERROR(cudaMemsetAsync(d_q, 0, MATRIX_SIZE * sizeof(double)));
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
                   spmv_streams);
    CUDA_CHECK_ERROR(cudaMemsetAsync(d_denominator, 0, sizeof(double)));
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_c, d_q, d_denominator, MATRIX_SIZE);
    prepareCgIteration<<<1, 1>>>(cg_scalars);

    // solution += alpha * c and r -= alpha * q in one vector traversal.
    updateSolutionAndResidual<<<(MATRIX_SIZE + 255) / 256, 256>>>(
      solution, d_c, d_r, d_q, cg_scalars, MATRIX_SIZE);

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

    // delta_new = r * s
    CUDA_CHECK_ERROR(cudaMemsetAsync(d_delta_new, 0, sizeof(double)));
    dotProduct<<<(MATRIX_SIZE + 255) / 256, 256>>>(d_r, d_s, d_delta_new, MATRIX_SIZE);
    finishCgIteration<<<1, 1>>>(cg_scalars, relativeTolerance);

    // c = s + beta * c. This intentionally still runs on the converged
    // iteration to preserve the previous solver's update ordering.
    updateDirection<<<(MATRIX_SIZE + 255) / 256, 256>>>(
      d_s, d_c, cg_scalars, MATRIX_SIZE);

    // One D2H transfer per iteration: [status, residual-or-denominator].
    CUDA_CHECK_ERROR(cudaMemcpy(iteration_status, d_status,
                                2 * sizeof(double), cudaMemcpyDeviceToHost));
    if (iteration_status[0] == CG_STATUS_NON_SPD) {
      printf("Non SPD matrix detected in %d iterations with residual %lf and alpha %lf\\n",
             iteration, h_delta_new, iteration_status[1]);
      return -iteration - 4;
    }
    h_delta_new = iteration_status[1];
    if (iteration_status[0] == CG_STATUS_CONVERGED){
      printf("Converged in %d iterations with residual %lf\\n", iteration, h_delta_new);
      return iteration;
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error during kernel execution: %s\\n", cudaGetErrorString(err));
    return -3;  // Return error to Python
  }
  printf("Did not converge in %d iterations; residual %lf\\n", maxIteration, h_delta_new);
  return -4;
}

} // close the extern "C"
'''
      sorted_dimensions = sorted(candidate_block_dimensions)
      cache_identity = {
        "cache_version": SOLVER_KERNEL_CACHE_VERSION,
        "dimensions": sorted_dimensions,
        "source": kernelString,
        "compiler_command": SOLVER_NVCC_COMMAND_TEMPLATE,
      }
      cache_signature = _stable_content_signature(cache_identity)
      dimension_to_text = "__".join(f"{dim[0]}_{dim[1]}" for dim in sorted_dimensions)
      file_original_name = f".yasps_constant/cg_dims_{dimension_to_text}__{SOLVER_KERNEL_CACHE_VERSION}"
      file_hashed_name = f".yasps_constant/cg_dims_{cache_signature}"
      source_path = f"{file_hashed_name}.cu"
      shared_library_path = f"{file_hashed_name}.so"
      cache_index_file = ".yasps_constant/cg_dimension_to_file_v6.json"

      _atomic_write_text(source_path, kernelString)
      if not os.path.exists(shared_library_path):
        build_token = f"{os.getpid()}_{threading.get_ident()}"
        temporary_library_path = f"{shared_library_path}.tmp_{build_token}"
        compile_command = [
          temporary_library_path if item == "<OUTPUT>" else
          source_path if item == "<SOURCE>" else item
          for item in SOLVER_NVCC_COMMAND_TEMPLATE
        ]
        try:
          subprocess.run(compile_command, check=True)
          os.replace(temporary_library_path, shared_library_path)
        finally:
          if os.path.exists(temporary_library_path):
            os.remove(temporary_library_path)

      self.__loadKernelLibrary(file_hashed_name)
      _update_cache_index(cache_index_file, {
        "cache_signature": cache_signature,
        "cache_version": SOLVER_KERNEL_CACHE_VERSION,
        "compiler_command": list(SOLVER_NVCC_COMMAND_TEMPLATE),
        "dimensions": sorted_dimensions,
        "file_hashed_name": file_hashed_name,
        "file_original_name": file_original_name,
        "source_sha256": hashlib.sha256(kernelString.encode("utf-8")).hexdigest(),
      })
      self.__saved_block_dimensions = candidate_block_dimensions
      self.__max_row_size = max_modded_row_size

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
    initial_guess: gpuarray.GPUArray,
    zero_initial_guess: bool = False,
    cg_scalars = None
  ):
    self.__context.useDefaultContext()
    # self.__context.useNamedContext("solver") # 14624

    if cg_scalars is None:
      cg_scalars = gpuarray.empty(8, dtype=np.float64)
    elif cg_scalars.size < 8 or cg_scalars.dtype != np.float64:
      raise ValueError("solverKernel.computeSolution: cg_scalars must contain at least 8 float64 values")

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
      self.__to_void_p(initial_guess),
      int(zero_initial_guess),
      self.__to_void_p(cg_scalars)
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

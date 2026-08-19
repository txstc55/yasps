from yasps.attribute import attribute
from yasps.deviceKernel import deviceKernel
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
from typing import List, Set
class hessianKernelHeader:
  def __init__(self, att: attribute, unique_gradient_sizes: Set[int], sortedDependency: List[deviceKernel]):
    self.__att = att
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    self.__kernelString = ""
    self.__kernelString += '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>
#ifndef EIGEN_PROJECTION
#define EIGEN_PROJECTION
// Collapse identical atomic destinations within a contiguous warp segment.
// The caller enables this only for terms whose outputs are highly shared.
__device__ __forceinline__ void atomic_add_grouped(double *address, double value) {
#if __CUDA_ARCH__ >= 700
  const unsigned int active = __activemask();
  const unsigned long long key = reinterpret_cast<unsigned long long>(address);
  const unsigned int peers = __match_any_sync(active, key);
  if (__popc(peers) > 1) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int first = __ffs(peers) - 1;
    const unsigned int last = 31 - __clz(peers);
    const unsigned int width = last - first + 1;
    const unsigned int contiguous = width == 32 ? 0xffffffffu : (((1u << width) - 1u) << first);
    if (peers == contiguous) {
      const unsigned int rank = lane - first;
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const double other = __shfl_down_sync(active, value, offset);
        if ((rank & (2 * offset - 1)) == 0 && lane + offset <= last) {
          value += other;
        }
      }
      if (lane == first) {
        atomicAdd(address, value);
      }
      return;
    }
  }
#endif
  atomicAdd(address, value);
}

// Row-major packed storage for the upper triangle, including the diagonal.
// Reflect lower-triangular reads so callers can treat the packed data as a
// dense symmetric matrix without materializing its lower half.
template <unsigned int N>
__device__ __forceinline__ unsigned int symmetric_upper_index(
    unsigned int row, unsigned int column) {
  if (row > column) {
    const unsigned int temporary = row;
    row = column;
    column = temporary;
  }
  return row * N - row * (row + 1) / 2 + column;
}

template <unsigned int N>
__device__ __forceinline__ double symmetric_upper_get(
    const double *packed, unsigned int row, unsigned int column) {
  return packed[symmetric_upper_index<N>(row, column)];
}

// Fast FP64 LDLT-equivalent PSD test using an in-place packed Schur
// complement. It preserves A and uses half the explicit scratch of the
// dense lower-plus-diagonal implementation.
template <unsigned int N>
__device__ __forceinline__ bool is_positive_semidefinite(
    const double *__restrict__ A) {
  static_assert(N > 0);

  constexpr unsigned int PACKED_SIZE = N * (N + 1) / 2;
  double schur[PACKED_SIZE];
  double scale = 1.0;

#pragma unroll 1
  for (unsigned int row = 0; row < N; ++row) {
    const unsigned int row_base = row * (row + 1) / 2;
#pragma unroll 4
    for (unsigned int column = 0; column <= row; ++column) {
      schur[row_base + column] = A[row * N + column];
    }
    scale = fmax(scale, fabs(schur[row_base + row]));
  }

  const double tolerance = scale * 1.0e-10;

#pragma unroll 1
  for (unsigned int column = 0; column < N; ++column) {
    const unsigned int column_base = column * (column + 1) / 2;
    const double pivot = schur[column_base + column];

    if (pivot < -tolerance) return false;

    // A zero diagonal in a PSD Schur complement requires the corresponding
    // remaining column to be zero as well.
    if (fabs(pivot) <= tolerance) {
#pragma unroll 4
      for (unsigned int row = column + 1; row < N; ++row) {
        const unsigned int row_base = row * (row + 1) / 2;
        if (fabs(schur[row_base + column]) > tolerance) return false;
      }
      continue;
    }

    // One correctly-rounded reciprocal per column instead of one division
    // for every remaining row.
    const double inverse_pivot = __drcp_rn(pivot);

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
// For small matrix < 4
template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output, int choice) {
  if (choice == 0){
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
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
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
    for (unsigned int i = 0; i < N * N; i++){
      output[i] = A[i];
    }
    return;
  }
  if (is_positive_semidefinite<N>(A)) {
    for (unsigned int i = 0; i < N * N; ++i) output[i] = A[i];
    return;
  }
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }
  }

  Eigen::Matrix<double, N, N> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();

  Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> outputMap(output);
  outputMap = A_reconstructed;
  return;
}

template <unsigned int N>
__device__ void spd_projection_inplace(double *A, int choice) {
  if (choice == 0){
    return;
  }
  if (is_positive_semidefinite<N>(A)) return;
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }
  }

  // Reconstruct the matrix directly without using an intermediate matrix
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += B(i, k) * eigenValues[k] * B(j, k);
      }
      A[i * N + j] = sum;
    }
  }
  return;
}
#endif // EIGEN_PROJECTION
'''

    # we first generate the header file
    for item in (sortedDependency+ [self.__att.deviceKernel]):
      self.__kernelString += f'''
extern "C" {{
{item.kernelHeader};
}}'''
    for unique_gradient_size in unique_gradient_sizes:
      if unique_gradient_size == 0:
        continue
      self.__kernelString += f'''
extern "C" {{
__global__ void compute_hessian_and_gradient_global_function_final_gradient_size_{unique_gradient_size}(
{"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
{"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
{"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
{"".join([f"const unsigned int* {x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
const unsigned int* segment_indices,            // where to place the gradient for each segment of the local gradient / hessian we generated
const unsigned short int* segment_sizes,        // how large is each segment of the gradient before compression
const short int* local_permutations,            // how do i locally compress the hessian and gradient
const unsigned int* lookups,                    // how to place the current block inside the hessian
const unsigned int* coordinatesOuter,           // this will tell us for each instance, the starting and ending index in the lookup table for putting the hessian blocks into the global hessian data array
const unsigned int* groupedIndicesInner, // we need to know which instance will correspond to the current size
const unsigned int* groupedIndicesOuter, // the outer indices that will indicate for each gradient size, what's the start and end in the inner array
const unsigned int nth_gradient_size,    // this indicates which position we are in the outer array
const unsigned int projection_method,
double* gradient,   // the gradient output
double* hessian_blocks, // the blocks that will constitute the hessian
double* diagonal,    // the diagonal, we will use it for preconditioning
double* diagonal_blocks, // store the diagonal blocks, use it for block preconditioning
const unsigned int* diagonal_blocks_start, // for each attribute, where does the diagonal block start
const unsigned int* gradient_segments_start // for each attribute, where does the gradient start
);
}}
'''


  @property
  def kernelString(self):
    return self.__kernelString

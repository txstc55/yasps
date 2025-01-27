# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from pycuda.compiler import SourceModule
import pycuda.driver as pd
from typing import Optional, List
from yasps.helper import get_mangled_name
from yasps.helper import prune_duplicate_functions

testing_kernel = ""

class hessianAndGradientKernel:
  def __init__(self, att: attribute, block_sizes: List[int], project_entire_hessian: bool, projection_method: int = 1):
    self.__kernelString: str = ""
    self.__kernel: Optional[pd.Function] = None
    self.__block_sizes = block_sizes
    self.__project_entire_hessian = project_entire_hessian
    self.__projection_method = projection_method
    self.__generateKernel(att)



  def __generateKernel(self, attr: attribute) -> None:
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = attr.deviceKernel.dependents
    sortedDatas: List[attribute] = attr.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = attr.deviceKernel.kernelConnectivity
    # add the includes
    self.__kernelString += f'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#include <Eigen/Core>
#include <Eigen/Dense>
// here we add code for spd projection

// for small matrix < 4
template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output, int choice){{
  const int M = 4;
  // Initialize an M x M matrix with zeros
  Eigen::Matrix<double, M, M> symMtr = Eigen::Matrix<double, M, M>::Identity();

  // Copy the input N x N matrix into the top-left corner of the M x M matrix
  for (int row = 0; row < N; ++row) {{
    for (int col = 0; col < N; ++col) {{
      symMtr(row, col) = A[row * N + col];
    }}
  }}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, M, M>> eigenSolver(symMtr);
  Eigen::Matrix<double, M, M> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < M; i++){{
    if (eigenValues[i] < 0) {{
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }}
  }}
  // Reconstruct the matrix without using a diagonal matrix
  // Scale columns of B by corresponding eigenvalues
  Eigen::Matrix<double, M, M> C;
  for (int i = 0; i < M; ++i) {{
    C.col(i) = B.col(i) * eigenValues[i];
  }}
  // Compute the reconstructed matrix: A_reconstructed = C * B.transpose()
  Eigen::Matrix<double, M, M> A_reconstructed = C * B.transpose();
  // Copy the top-left N x N submatrix back to A
  for (int row = 0; row < N; ++row) {{
    for (int col = 0; col < N; ++col) {{
      output[row * N + col] = A_reconstructed(row, col);
    }}
  }}
  return;
}}

template <unsigned int N>
__device__ void spd_projection(const double *A, double* output, int choice){{
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  Eigen::Matrix<double, N, N> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < N; i++){{
    if (eigenValues[i] < 0) {{
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }}
  }}
  // Reconstruct the matrix without using a diagonal matrix
  // Scale columns of B by corresponding eigenvalues
  Eigen::Matrix<double, N, N> C;
  for (int i = 0; i < N; ++i) {{
    C.col(i) = B.col(i) * eigenValues[i];
  }}
  // Compute the reconstructed matrix: A_reconstructed = C * B.transpose()
  Eigen::Matrix<double, N, N> A_reconstructed = C * B.transpose();
  // Copy the top-left N x N submatrix back to output
  Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> outputMap(output);
  outputMap = A_reconstructed;
  return;
}}

template <unsigned int N>
__device__ void spd_projection_inplace(double *A, int choice){{
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  Eigen::Matrix<double, N, N> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < N; i++){{
    if (eigenValues[i] < 0) {{
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }}
  }}
  // Reconstruct the matrix directly without using an intermediate matrix
  for (int i = 0; i < N; ++i) {{
    for (int j = 0; j < N; ++j) {{
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {{
        sum += B(i, k) * eigenValues[k] * B(j, k);
      }}
      A[i * N + j] = sum;
    }}
  }}
  return;
}}
'''

    for item in sortedDependency:
      self.__kernelString += f"{item.kernelHeader};"
    self.__kernelString += f"{attr.deviceKernel.kernelHeader};"

    for item in sortedDependency:
      self.__kernelString += item.kernelString
    self.__kernelString += attr.deviceKernel.kernelString

    # now actually generate the global kernel
    attributeName: str = ""
    if attr.name == "":
      attributeName = f'attr_{attr.hash}'.replace("-", "_neg_")
    else:
      attributeName = attr.fullName

    kernelRawName = f'''
__global__ void accumulate_hessian_and_gradient_global_function({"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}{"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}{"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}const unsigned int* gradient_placements, const unsigned int* block_sizes, const unsigned int* hessian_blocks_start_indices, const unsigned int* hessian_blocks_where_to_check, const unsigned int* hessian_blocks_indices, double* gradient, double* hessian_blocks, double* diagonal, unsigned int MAX_INDEX)'''
    self.__kernelString += f'''
{kernelRawName}{{
  // gradient_placements: for the gradient generated for each local element, and for all the small segments inside, where to place it
  // block_sizes: the gradient is segmented into small parts, for each parts, what's the dimension. This is also used for hessian block sizes
  // hessian_blocks_start_indices: the hessian generated can be segmented into smaller blocks, each block may have different dimensions. This array tells us, for each dimension, where does the block start
  // hessian_blocks_where_to_check: for each block, we have its dimension, we need to know for this dimension, where does the segment start
  // hessian_blocks_indices: for each block, we have its dimension, we need to know for this dimension, where to put it in the corresponding segment
  // gradient: the accumulated gradient
  // hessian_blocks: the accumulated blocks
  // diagonal: the accumulated diagonal

  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  Eigen::Matrix<double, {sum(self.__block_sizes) + 1}, {sum(self.__block_sizes)}{", Eigen::RowMajor" if sum(self.__block_sizes) > 1 else ""}> hg_mat = Eigen::Matrix<double, {sum(self.__block_sizes) + 1}, {sum(self.__block_sizes)}, Eigen::RowMajor>::Zero(); // get the merged gradient and hessian
  // now we call the device function
  {attributeName}_device_function({"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}{"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}{"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}index, hg_mat.data());
  // now maybe we need to project the entire hessian
  // the true false value is generated at compile time
  if ({int(self.__project_entire_hessian)}){{
    // project the hessian
    {"spd_projection_inplace" if sum(self.__block_sizes) >= 4 else "spd_projection_small"}<{sum(self.__block_sizes)}>(hg_mat.data(), {"hg_mat.data(), " if sum(self.__block_sizes) < 4 else ""}{self.__projection_method});
  }}
  // now we need to place the hessian into the correct places
  unsigned int row_offset = 0;
  unsigned int off_diagonal_counts = 0;
  for (unsigned int i = 0; i < {len(self.__block_sizes)}; i++){{
    unsigned int col_offset = row_offset;
    unsigned int block_rows = block_sizes[i];
    unsigned int raw_position_i = gradient_placements[index * {len(self.__block_sizes)} + i];
    for (unsigned int j = i; j < {len(self.__block_sizes)}; j++){{
      unsigned int raw_position_j = gradient_placements[index * {len(self.__block_sizes)} + j];
      unsigned int block_cols = block_sizes[j];
      // now we know the size of the block, we need to put it to the correct block
      // we need to put it in the off diagonal blocks
      unsigned int where_to_check = (raw_position_i <= raw_position_j) ? (hessian_blocks_where_to_check[off_diagonal_counts]) : (hessian_blocks_where_to_check[off_diagonal_counts + {len(self.__block_sizes) * (len(self.__block_sizes) + 1) // 2}]); // know which off diagonal block we are in
      unsigned int off_diagonal_block_start_index = hessian_blocks_start_indices[where_to_check]; // get the start index of this off diagonal block
      unsigned int placement_index = hessian_blocks_indices[index * {len(self.__block_sizes) * (len(self.__block_sizes) + 1) // 2} + off_diagonal_counts]; // get the placement index
      unsigned int off_diagonal_block_placement = off_diagonal_block_start_index + placement_index * block_rows * block_cols; // get the placement index
      // place the block
      if (raw_position_i <= raw_position_j){{
        // we are in the upper triangle
        for (unsigned int k = 0; k < block_rows; k++){{
          for (unsigned int l = 0; l < block_cols; l++){{
            // printf("k: %u, l: %u, row offset: %u, col offset: %u\\n", k, l, row_offset, col_offset);
            // printf("hg_mat: %lf\\n", hg_mat(row_offset + k, col_offset + l));
            atomicAdd(&hessian_blocks[off_diagonal_block_placement + k * block_cols + l], hg_mat(row_offset + k, col_offset + l));
          }}
        }}
      }} else{{
        // we are in the lower triangle
        for (unsigned int k = 0; k < block_cols; k++){{
          for (unsigned int l = 0; l < block_rows; l++){{
            atomicAdd(&hessian_blocks[off_diagonal_block_placement + k * block_cols + l], hg_mat(row_offset + l, col_offset + k));
          }}
        }}
      }}
      if (i == j){{
        // we know it is a diagonal block
        // we need to put the diagonal element into diagonal
        for (unsigned int k = 0; k < block_rows; k++){{
          atomicAdd(&diagonal[raw_position_i + k], hg_mat(row_offset + k, col_offset + k));
        }}
      }}
      if (raw_position_i == raw_position_j && i != j){{
        // we need to place the transpose too, this is a special case, because we are doing accumulation
        // this block needs to be added twice
        for (unsigned int k = 0; k < block_rows; k++){{
          for (unsigned int l = 0; l < block_cols; l++){{
            // printf("k: %u, l: %u, row offset: %u, col offset: %u\\n", k, l, row_offset, col_offset);
            // printf("hg_mat: %lf\\n", hg_mat(row_offset + k, col_offset + l));
            atomicAdd(&hessian_blocks[off_diagonal_block_placement + k * block_cols + l], hg_mat(col_offset + l, row_offset + k));
          }}
        }}
        // this is a off diagonal block, but when assembled, it is a diagonal block
        // which means we need to put 2 * the value into the diagonal
        for (unsigned int k = 0; k < block_rows; k++){{
          atomicAdd(&diagonal[raw_position_i + k], 2.0 * hg_mat(row_offset + k, col_offset + k));
        }}
      }}
      off_diagonal_counts += 1;
      col_offset += block_cols; // move the column offset
    }}
    row_offset += block_rows; // move the row offset
  }}
  // now we need to place the gradient
  unsigned int count = 0;
  for (unsigned int i = 0; i < {len(self.__block_sizes)}; i++){{
    unsigned int placement_index = gradient_placements[index * {len(self.__block_sizes)} + i];
    for (unsigned int j = 0; j < block_sizes[i]; j++){{
      atomicAdd(&gradient[placement_index + j], hg_mat({sum(self.__block_sizes)}, count));
      count += 1;
    }}
  }}
}}
'''
    # prune duplicate functions
    self.__kernelString = prune_duplicate_functions(self.__kernelString)
    # generate the code to check
    f = open(".yasps_tmp/hessian_and_gradient_kernel.cu", "w")
    f.write(self.__kernelString)
    f.close()
    mod = SourceModule(
      self.__kernelString,
      options = ["-std=c++11", '-O3', '-I/usr/include/eigen3', "--expt-relaxed-constexpr", "--disable-warnings"],
      no_extern_c = True
    )
    kernel_name: str = get_mangled_name(kernelRawName, 'accumulate_hessian_and_gradient_global_function')
    self.__kernel = mod.get_function(kernel_name)



  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self) -> pd.Function:
    return self.__kernel

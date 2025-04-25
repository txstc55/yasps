# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from pycuda.compiler import SourceModule
import pycuda.driver as pd
from typing import Optional, List, Set
from yasps.helper import get_mangled_name
from yasps.helper import prune_duplicate_functions

testing_kernel = ""

class hessianAndGradientKernel:
  def __init__(self, att: attribute, project_entire_hessian: bool, projection_method: int = 1, gradeient_only: bool = False):
    self.__kernelString: str = ""
    self.__kernel = None # the kernel for computhing the gradient and hessians
    self.__unique_gradient_sizes: Set[int] = set([]) # this will tell us the unique gradient sizes, we will use this to generate and regenerate kernel when there are new gradient sizes
    self.__project_entire_hessian = project_entire_hessian
    self.__projection_method = projection_method
    self.__gradient_only = gradeient_only
    self.__att = att
    # self.__generateKernel(att)



  def __generateKernel(self, unique_gradient_sizes: List[int]) -> None:
    # check if our unique gradient sizes contains the input gradient sizes
    if set(unique_gradient_sizes).issubset(self.__unique_gradient_sizes):
      return
    self.__unique_gradient_sizes.update(unique_gradient_sizes)

    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    # add the includes and the evd function
    self.__kernelString += '''
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
  const Eigen::Matrix<double, M, M> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < M; i++){{
    if (eigenValues[i] < 0) {{
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }}
  }}
  Eigen::Matrix<double, M, M> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();
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
  const Eigen::Matrix<double, N, N> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < N; i++){{
    if (eigenValues[i] < 0) {{
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }}
  }}
  // Compute the reconstructed matrix: A_reconstructed = C * B.transpose()
  Eigen::Matrix<double, N, N> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();
  // A_reconstructed.noalias() = C * B.transpose();
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
  const Eigen::Matrix<double, N, N> B = eigenSolver.eigenvectors();
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
    self.__kernelString += f"{self.__att.deviceKernel.kernelHeader};"

    for item in sortedDependency:
      self.__kernelString += item.kernelString
    self.__kernelString += self.__att.deviceKernel.kernelString

    # now actually generate the global kernel
    attributeName: str = ""
    if self.__att.name == "":
      attributeName = f'attr_{self.__att.hash}'.replace("-", "_neg_")
    else:
      attributeName = self.__att.fullName

    kernelRawName = f'''
template <unsigned int N>
__global__ void accumulate_hessian_and_gradient_global_function(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  const unsigned int* segment_placements,         // where to place the gradient for each segment of the local gradient / hessian we generated
  const unsigned short int* segment_sizes,        // how large is each segment of the gradient
  const short int* local_permutations,   // how do i locally compress the hessian and gradient
  const unsigned int* lookups,                    // how to place the current block inside the hessian
  const unsigned int* current_gradient_size_indices, // for the current gradient size, we need to know which instance will produce this gradient size
  const unsigned int max_num_indices, // the maximum number of indices for each instance
  const unsigned int projection_method,
  unsigned int begin, // for this gradient size, where do i begin looking
  unsigned int end,   // for this gradient size, where do i end looking
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal    // the diagonal, we will use it for preconditioning
)'''
    self.__kernelString += f'''

  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= end - begin){{
    return;
  }}
  index = begin + index; // add to begin
  unsigned int instance = current_gradient_size_indices[index];

// determine if we are computing both the hessian and gradient
#if {int(not self.__gradient_only)} // are we computing both the hessian and gradient
  Eigen::Matrix<double, {self.__att.rows + 1}, {self.__att.rows}{", Eigen::RowMajor" if self.__att.rows > 1 else ""}> hg_mat = Eigen::Matrix<double, {self.__att.rows + 1}, {self.__att.rows}, Eigen::RowMajor>::Zero(); // get the merged gradient and hessian
#else // we are only computing the gradient
  Eigen::Matrix<double, 1, {self.__att.rows}> hg_mat = Eigen::Matrix<double, 1, {self.__att.rows}>::Zero(); // get the gradient
#endif

  // now we call the device function
  {attributeName}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    instance,
    hg_mat.data()
  );
  // ok we now first put the gradient into the correct place
  unsigned int gradient_offset = 0;
  for (unsigned int i = 0; i < max_num_indices; i++){{
    // we will first get the segment size
    unsigned short int segment_size = segment_sizes[instance * max_num_indices + i];
    // and the position for this segment
    unsigned int segment_placement = segment_placements[instance * max_num_indices + i];
    // now we access the gradient and put it into the correct place
    for (unsigned int j = 0; j < segment_size; j++){{
#if {int(not self.__gradient_only)} // did we compute the hessian
      atomicAdd(&gradient[segment_placement + j], hg_mat({self.__att.rows}, gradient_offset + j));
#else
      atomicAdd(&gradient[segment_placement + j], hg_mat(0, gradient_offset + j));
#endif
    }}
    gradient_offset += segment_size;
  }}

  // ok now we have the gradient, we need to compress the hessian locally
  // we have the permutation, and we know the size of the matrix
#if {int(not self.__gradient_only)}
  // we will only start this part if we are not just doing gradient
  // first of all, allocate a matrix
  Eigen::Matrix<double, N, N> compressed_hessian = Eigen::Matrix<double, N, N>::Zero();
  unsigned int row_offset = 0;
  for (unsigned int i = 0; i < max_num_indices; j++){{
    unsigned int col_offset = 0;
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = permutation[instance * max_num_indices + i];
    if (permutation_i < 0){{
      // this block position exists, we need to get the negative of it
      permutation_i = -permutation_i;
    }}
    permutation_i -= 1; // back to 0 indexed
    unsigned short int segment_size_i = segment_sizes[instance * max_num_indices + i];
    for (unsigned int j = i; j < max_num_indices; j++){{
      short int permutation_j = permutation[instance * max_num_indices + j];
      if (permutation_j < 0){{
        // this block position exists, we need to get the negative of it
        permutation_j = -permutation_j;
      }}
      permutation_j -= 1; // back to 0 indexed
      // ok at this point we know the correct position to put in the compressed hessian
      unsigned short int segment_size_j = segment_sizes[instance * max_num_indices + j];
      for (unsigned int k = 0; k < segment_size_i; k++){{
        for (unsigned int l = 0; l < segment_size_j; l++){{
          // we put the block into the compressed hessian
          compressed_hessian(permutation_i + k, permutation_j + l) = hg_mat(row_offset + k, col_offset + l);
          // put the transpose into the compressed hessian if not a diagonal block
          if (i != j){{
            compressed_hessian(permutation_j + l, permutation_i + k) = hg_mat(col_offset + l, row_offset + k);
          }}
        }}
      }}
      col_offset += segment_size_j;
    }}
    row_offset += segment_size_i;
  }}
  // now we have the compressed hessian
  // we will project it if needed

#if {int(self.__project_entire_hessian)} // do we need to project the hessian here
  // project the hessian
  if (N < 4){{
    spd_projection_small<N>(compressed_hessian.data(), compressed_hessian.data(), projection_method);
  }}else{{
    spd_projection_inplace<N>(compressed_hessian.data(), projection_method);
  }}
#endif // end of projection
  // now we finished the projection
  // we will need to place the hessian back to blocks


#endif



#if {int(not self.__gradient_only)} // project the hessian, and put the hessian inplace
  // now maybe we need to project the entire hessian
  // the true false value is generated at compile time
#if {int(self.__project_entire_hessian)} // do we need to project the hessian here
  // project the hessian
  {"spd_projection_inplace" if self.__att.rows >= 4 else "spd_projection_small"}<{self.__att.rows}>(hg_mat.data(), {"hg_mat.data(), " if self.__att.rows < 4 else ""}{self.__projection_method});
#endif // and of projection
  // now we need to place the hessian into the correct places
  unsigned int row_offset = 0;
  unsigned int off_diagonal_counts = 0;
  for (unsigned int i = 0; i < {len(self.__block_sizes)}; i++){{
    unsigned int col_offset = row_offset;
    unsigned int block_rows = segment_sizes[i];
    unsigned int raw_position_i = segment_placements[index * {len(self.__block_sizes)} + i];
    for (unsigned int j = i; j < {len(self.__block_sizes)}; j++){{
      unsigned int raw_position_j = segment_placements[index * {len(self.__block_sizes)} + j];
      unsigned int block_cols = segment_sizes[j];
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
#endif // end projecting and putting the hessian into the global matrix
  // now we need to place the gradient
  unsigned int count = 0;
  for (unsigned int i = 0; i < {len(self.__block_sizes)}; i++){{
    unsigned int placement_index = segment_placements[index * {len(self.__block_sizes)} + i];
    for (unsigned int j = 0; j < segment_sizes[i]; j++){{
#if {int(not self.__gradient_only)} // did we compute the hessian
      atomicAdd(&gradient[placement_index + j], hg_mat({self.__att.rows}, count));
#else
      atomicAdd(&gradient[placement_index + j], hg_mat(0, count));
#endif
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

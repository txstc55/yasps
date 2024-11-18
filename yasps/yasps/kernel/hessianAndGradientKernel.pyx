# cython: language_level=3
from __future__ import annotations
from sympy.logic.boolalg import Boolean
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
  def __init__(self, att: attribute, block_sizes: List[int], needs_projection):
    self.__kernelString: str = ""
    self.__kernel: Optional[pd.Function] = None
    self.__block_sizes = block_sizes
    self.__needs_projection = needs_projection
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
template <unsigned int N>
__device__ void spd_projection(const double *A, double* output, int choice){{
  // Define M as the maximum of N and 4, because 3 by 3 evd is wrong somehow
  const int M = (N < 4) ? 4 : N;

  // Initialize an M x M matrix with zeros
  Eigen::Matrix<double, M, M> symMtr = Eigen::Matrix<double, M, M>::Zero();

  // Copy the input N x N matrix into the top-left corner of the M x M matrix
  for (int row = 0; row < N; ++row) {{
    for (int col = 0; col < N; ++col) {{
      symMtr(row, col) = A[row * N + col];
    }}
  }}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, M, M>> eigenSolver(symMtr);
  Eigen::Matrix<double, M, M> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();
  if (choice == 1) {{
    for (int i = 0; i < M; i++) {{
      if (eigenValues.data()[i] < 0) {{
        eigenValues.data()[i] = abs(eigenValues.data()[i]);
      }}
    }}
  }}else{{
    for (int i = 0; i < M; i++) {{
      if (eigenValues.data()[i] < 0) {{
        eigenValues.data()[i] = 0.0;
      }}
    }}
  }}
  // Reconstruct the matrix without using a diagonal matrix
  // Scale columns of B by corresponding eigenvalues
  Eigen::Matrix<double, M, M> C;
  for (int i = 0; i < N; ++i) {{
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
__global__ void accumulate_hessian_and_gradient_global_function({"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}{"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}{"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}const unsigned int* gradient_placements, const unsigned int* block_sizes, const unsigned int* hessian_diagonal_blocks_start_indices, const unsigned int* hessian_off_diagonal_blocks_start_indices, const unsigned int* diagonal_block_infos, const unsigned int* off_diagonal_blocks_where_to_check, const unsigned int* off_diagonal_blocks_indices, double* gradient, double* hessian_diagonal_blocks, double* hessian_off_diagonal_blocks, unsigned int MAX_INDEX)'''
    self.__kernelString += f'''
{kernelRawName}{{
  // gradient_placements: for the gradient generated for each local element, and for all the small segments inside, where to place it
  // block_sizes: the gradient is segmented into small parts, for each parts, what's the size. This is also used for hessian block sizes
  // hessian_diagonal_blocks_start_indices: say if we are differentiating wrt N attributes, then each of those attributes will have a list of diagonal blocks, positioned sequentially. This array tells us where each of those blocks start
  // hessian_off_diagonal_blocks_start_indices: the hessian generated can be segmented into smaller blocks, each block may have different dimensions. This array tells us, for each dimension, where does the block start
  // diagonal_block_infos: for each diagonal block, which attribute (index) it is, the start index of this attribute(used to compute offset)
  // off_diagonal_blocks_where_to_check: for each off diagonal block, we have its dimension, we need to know for this dimension, where does the segment start
  // off_diagonal_blocks_indices: for each off diagonal block, we have its dimension, we need to know for this dimension, where to put it in the corresponding segment
  // gradient: the accumulated gradient
  // hessian_diagonal_blocks: the accumulated diagonal blocks
  // hessian_off_diagonal_blocks: the accumulated off diagonal blocks

  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  Eigen::Matrix<double, {sum(self.__block_sizes) + 1}, {sum(self.__block_sizes)}, Eigen::RowMajor> hg_mat = Eigen::Matrix<double, {sum(self.__block_sizes) + 1}, {sum(self.__block_sizes)}, Eigen::RowMajor>::Zero(); // get the merged gradient and hessian
  // now we call the device function
  {attributeName}_device_function({"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}{"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}{"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}index, hg_mat.data());
  printf("gradient: ");
  for (unsigned int i = 0; i < {sum(self.__block_sizes)}; i++){{
    printf("%lf, ", hg_mat({sum(self.__block_sizes)}, i));
  }}
  printf("\\n");
  // now maybe we need to project the entire hessian
  // the true false value is generated at compile time
  // if ({int(self.__needs_projection)}){{
  if (0){{
    // project the hessian
    spd_projection<{sum(self.__block_sizes)}>(hg_mat.data(), hg_mat.data(), 1);
  }}
  printf("Hessian projected\\n");
  // now we need to place the hessian into the correct places
  unsigned int row_offset = 0;
  unsigned int off_diagonal_counts = 0;
  for (unsigned int i = 0; i < {len(self.__block_sizes)}; i++){{
    unsigned int col_offset = 0;
    unsigned int block_rows = block_sizes[i];
    for (unsigned int j = i; j < {len(self.__block_sizes)}; j++){{
      unsigned int block_cols = block_sizes[j];
      printf("block rows: %u, block cols: %u, ij: %u, %u\\n", block_rows, block_cols, i, j);
      // now we know the size of the block, we need to put it to the correct block
      if (i == j){{
        // we need to put it in the diagonal block
        unsigned int attribute_pos_in_all_attributes = diagonal_block_infos[i * 2];
        unsigned int attribute_offset = diagonal_block_infos[i * 2 + 1];
        unsigned int diagonal_block_start_index = hessian_diagonal_blocks_start_indices[attribute_pos_in_all_attributes];
        unsigned int placement_index = gradient_placements[index * {len(self.__block_sizes)} + i];
        unsigned int diagonal_block_placement = diagonal_block_start_index + (placement_index - attribute_offset) * block_rows; // the diagonal block starts at start index, then we need to know this attribute's position in the whole attribute array, we should technically divide by block_rows to get the true index, but since we need to multiply by block_rows * block_rows, we can eliminate one division
        printf("diagonal block placement: %u, %u, %u, %u\\n", attribute_pos_in_all_attributes, attribute_offset, placement_index, diagonal_block_placement);
        for (unsigned int k = 0; k < block_rows; k++){{
          for (unsigned int l = 0; l < block_cols; l++){{
            atomicAdd(&hessian_diagonal_blocks[diagonal_block_placement + k * block_cols + l], hg_mat(row_offset + k, col_offset + l));
          }}
        }}
      }}else{{
        // we need to put it in the off diagonal blocks
        unsigned int where_to_check = off_diagonal_blocks_where_to_check[off_diagonal_counts]; // know which off diagonal block we are in
        printf("where to check: %u\\n", where_to_check);
        unsigned int off_diagonal_block_start_index = hessian_off_diagonal_blocks_start_indices[where_to_check]; // get the start index of this off diagonal block
        printf("off diagonal block start index: %u\\n", off_diagonal_block_start_index);
        unsigned int placement_index = off_diagonal_blocks_indices[index * {len(self.__block_sizes) * (len(self.__block_sizes) - 1) // 2} + off_diagonal_counts]; // get the placement index
        printf("off diagonal block placement index: %u\\n", placement_index);
        unsigned int off_diagonal_block_placement = off_diagonal_block_start_index + placement_index * block_rows * block_cols; // get the placement index
        printf("off diagonal block placement: %u\\n", off_diagonal_block_placement);
        // place the block
        for (unsigned int k = 0; k < block_rows; k++){{
          for (unsigned int l = 0; l < block_cols; l++){{
            atomicAdd(&hessian_off_diagonal_blocks[off_diagonal_block_placement + k * block_cols + l], hg_mat(row_offset + k, col_offset + l));
          }}
        }}
        off_diagonal_counts += 1;
      }}
      col_offset += block_cols; // move the column offset
    }}
    row_offset += block_rows; // move the row offset
  }}
  printf("Hessian assembled\\n");
  // now we need to place the gradient
  unsigned int count = 0;
  for (unsigned int i = 0; i < {len(self.__block_sizes)}; i++){{
    unsigned int placement_index = gradient_placements[index * {len(self.__block_sizes)} + i];
    for (unsigned int j = 0; j < block_sizes[i]; j++){{
      atomicAdd(&gradient[placement_index + j], hg_mat({sum(self.__block_sizes)}, count));
      count += 1;
    }}
  }}
  printf("Gradient assembled\\n");
}}
'''
    # prune duplicate functions
    self.__kernelString = prune_duplicate_functions(self.__kernelString)
    # generate the code to check
    f = open("testing_hessian_and_gradient_kernel.cu", "w")
    f.write(self.__kernelString)
    f.close()
    # f = open("/home/xuan/Desktop/research/yasps/tests/energy/testing_hessian_and_gradient_kernel.cu", 'r')
    # codes = f.read()
    # mod = SourceModule(
    #   codes,
    #   options = ["-std=c++11", '-O3', '-I/usr/include/eigen3', "--expt-relaxed-constexpr", "--disable-warnings"],
    #   no_extern_c = True
    # )
    mod = SourceModule(
      self.__kernelString,
      options = ["-std=c++11", '-O2', '-I/usr/include/eigen3', "--expt-relaxed-constexpr", "--disable-warnings"],
      no_extern_c = True
    )
    kernel_name: str = get_mangled_name(kernelRawName, f'accumulate_hessian_and_gradient_global_function')
    self.__kernel = mod.get_function(kernel_name)



  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self) -> pd.Function:
    return self.__kernel

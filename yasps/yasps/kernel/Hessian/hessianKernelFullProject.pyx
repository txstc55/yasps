from typing import List, Set
from yasps.attribute import attribute
from yasps.deviceKernel import deviceKernel
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
class hessianKernelFullProject:
  def __init__(self, att: attribute, unique_gradient_size: int, gradient_only: bool, max_num_indices: int, attributeName: str):
    self.__att = att
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    self.__kernelString = f'''
#include "allHeaders.cuh"
extern "C"{{
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
){{
  const unsigned int N = {unique_gradient_size}; // the size of the gradient and hessian, this is the unique gradient size
  // get the start and end position of the current gradient size
  const unsigned int start = groupedIndicesOuter[nth_gradient_size];
  const unsigned int end = groupedIndicesOuter[nth_gradient_size + 1];
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= end - start){{
    return;
  }}
  index = start + index; // add to begin
  const unsigned int instance = groupedIndicesInner[index]; // this will tell us which instance of the hessian we are computing
// determine if we are computing both the hessian and gradient
#if {int(not gradient_only)} // are we computing both the hessian and gradient
  Eigen::Matrix<double, {self.__att.rows}, {self.__att.cols}{", Eigen::RowMajor" if self.__att.cols > 1 else ""}> hg_mat = Eigen::Matrix<double, {self.__att.rows}, {self.__att.cols}, Eigen::RowMajor>::Zero(); // get the merged gradient and hessian
#else // we are only computing the gradient
  Eigen::Matrix<double, 1, {self.__att.cols}> hg_mat = Eigen::Matrix<double, 1, {self.__att.cols}>::Zero(); // get the gradient
#endif


  // now we call the device function
  {attributeName}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
    instance,
    hg_mat.data()
  );
  // ok we now first put the gradient into the correct place
  unsigned int gradient_offset = 0;
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    // we will first get the segment size
    unsigned short int segment_size = segment_sizes[instance * {max_num_indices} + i];
    // and the position for this segment
    unsigned int segment_placement = segment_indices[instance * {max_num_indices} + i];
    if (segment_placement == 0){{
      gradient_offset += segment_size;
      continue; // we encountered space reserved for union, skip
    }}else if (segment_placement == 1){{
      // this is a special case where we want the variable to be in the matrix, but not in the final hessian
      // we keep it because it's necessary for the hessian projection
      gradient_offset += segment_size; // skip this segment
      continue; // skip
    }}
    segment_placement -= 2; // make it 0 indexed
    // now we access the gradient and put it into the correct place
    for (unsigned int j = 0; j < segment_size; j++){{
  #if {int(not gradient_only)} // did we compute the hessian
      atomicAdd(&gradient[segment_placement + j], hg_mat({self.__att.rows - 1}, gradient_offset + j));
  #else
      atomicAdd(&gradient[segment_placement + j], hg_mat(0, gradient_offset + j));
  #endif
    }}
    gradient_offset += segment_size;
  }}
  // Now check if we are also computing the Hessian


#if {int(not gradient_only)}
  unsigned int row_offset = 0;
  // we are projecting the entire Hessian
  Eigen::Matrix<double, N, N> compressed_hessian = Eigen::Matrix<double, N, N>::Zero(); // first we allocate the matrix
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    unsigned int col_offset = 0;
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = local_permutations[instance * {max_num_indices} + i];
    if (permutation_i == 0){{
      row_offset += segment_sizes[instance * {max_num_indices} + i]; // done with the row since it's reserved for union empty space
      continue; // we encountered space reserved for union, skip
    }}
    if (permutation_i < 0){{
      // this block position exists, we need to get the negative of it
      permutation_i = -permutation_i;
    }}
    permutation_i -= 1; // back to 0 indexed
    unsigned short int segment_size_i = segment_sizes[instance * {max_num_indices} + i];
    for (unsigned int j = 0; j < {max_num_indices}; j++){{
      short int permutation_j = local_permutations[instance * {max_num_indices} + j];
      if (permutation_j == 0){{
        col_offset += segment_sizes[instance * {max_num_indices} + j]; // done with the column since it's reserved for union empty space
        continue; // we encountered space reserved for union, skip
      }}
      if (permutation_j < 0){{
        // this block position exists, we need to get the negative of it
        permutation_j = -permutation_j;
      }}
      permutation_j -= 1; // back to 0 indexed
      // ok at this point we know the correct position to put in the compressed hessian
      unsigned short int segment_size_j = segment_sizes[instance * {max_num_indices} + j];
      for (unsigned int k = 0; k < segment_size_i; k++){{
        for (unsigned int l = 0; l < segment_size_j; l++){{
          // we put the block into the compressed hessian
          compressed_hessian(permutation_i + k, permutation_j + l) += hg_mat(row_offset + k, col_offset + l);
        }}
      }}
      col_offset += segment_size_j;
    }}
    row_offset += segment_size_i;
  }}
  // now we have the compressed hessian
  // we will project it if needed
  // project the hessian
  if (N < 4){{
    spd_projection_small<N>(compressed_hessian.data(), compressed_hessian.data(), projection_method);
  }}else{{
    spd_projection_inplace<N>(compressed_hessian.data(), projection_method);
  }}


  // we will now put the compressed hessian into the global hessian blocks
  // as well as the diagonal blocks
  const unsigned int coordinate_start = coordinatesOuter[instance];
  const unsigned int coordinate_end = coordinatesOuter[instance];
  row_offset = 0;
  unsigned int valid_block_counts = 0;
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = local_permutations[instance * {max_num_indices} + i]; // get the permuted placement
    unsigned int segment_index_i = segment_indices[instance * {max_num_indices} + i];
    if (permutation_i > 0 && segment_index_i >= 2){{
      // make it 0 indexed first
      permutation_i -= 1;
      unsigned short int segment_size_i = segment_sizes[instance * {max_num_indices} + i];
      segment_index_i -= 2;
      // we know exactly the row block, we now check for column block
      for (unsigned int j = i; j < {max_num_indices}; j++){{
        short int permutation_j = local_permutations[instance * {max_num_indices} + j]; // get the permuted placement
        unsigned int segment_index_j = segment_indices[instance * {max_num_indices} + j];
        if (permutation_j > 0 && segment_index_j >= 2){{
          // ok we have found a valid block
          // first again we make it 0 indexed
          permutation_j -= 1;
          unsigned short int segment_size_j = segment_sizes[instance * {max_num_indices} + j];
          segment_index_j -= 2;
          // we now need to get the index
          unsigned int placement_index = lookups[coordinate_start + valid_block_counts];
          // now we put the block in
          if (segment_index_i < segment_index_j){{
            for (unsigned int k = 0; k < segment_size_i; k++){{
              for (unsigned int l = 0; l < segment_size_j; l++){{
                // this is a block in the upper triangle
                atomicAdd(&hessian_blocks[placement_index + k * segment_size_j + l], compressed_hessian(permutation_i + k, permutation_j + l));
              }}
            }}
          }}else{{
            for (unsigned int k = 0; k < segment_size_j; k++){{
              for (unsigned int l = 0; l < segment_size_i; l++){{
                // put the transpose block in
                atomicAdd(&hessian_blocks[placement_index + k * segment_size_i + l], compressed_hessian(permutation_i + l, permutation_j + k));
              }}
            }}
          }}
          // additionally, if it is a diagonal block, we also need to put the diagonal elements
          if (i == j){{
            // get the placement
            unsigned int segment_index = segment_indices[instance * {max_num_indices} + i] - 2;
            for (unsigned int k = 0; k < segment_size_i; k++){{
              atomicAdd(&diagonal[segment_index + k], compressed_hessian(permutation_i + k, permutation_j + k));
            }}
            // now we do the block diagonal placement
            // we first need to determine where to put it in the global diagonal blocks array
            int which_attribute = 0;
            for (int k = 0; k < 123456; k++){{ // for now lets just make 123456 our default, need to change it later
              if (segment_index < gradient_segments_start[k + 1]){{
                break;
              }}
              which_attribute += 1;
            }}
            // now determine which instance in that attribute
            const unsigned int diagonal_block_start = diagonal_blocks_start[which_attribute];
            const unsigned int diff = segment_index - gradient_segments_start[which_attribute];
            const unsigned int which_instance = diff / (segment_size_i);
            const unsigned int diagonal_block_placement = diagonal_block_start + which_instance * segment_size_i * segment_size_i;
            // now we put the diagonal block
            for (unsigned int k = 0; k < segment_size_i; k++){{
              for (unsigned int l = 0; l < segment_size_i; l++){{
              atomicAdd(&diagonal_blocks[diagonal_block_placement + k * segment_size_i + l], compressed_hessian(permutation_i + k, permutation_j + l));
              }}
            }}
          }}
          valid_block_counts++;
        }}
      }}
    }}
  }}
#endif // end for gradient only check
}}
}}
'''

  @property
  def kernelString(self):
    return self.__kernelString

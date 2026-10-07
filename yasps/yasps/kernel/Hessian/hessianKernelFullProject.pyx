from typing import List
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
class hessianKernelFullProject:
  def __init__(self, evaluation, unique_gradient_size: int, gradient_only: bool, max_num_indices: int, attributeName: str, num_attributes: int, hessian_row_size: int, grouped_add: bool = False):
    atomic_add = "atomic_add_grouped" if grouped_add else "atomicAdd"
    packed_size = hessian_row_size * (hessian_row_size + 1) // 2
    dense_size = hessian_row_size * hessian_row_size
    packed_compression = packed_size + unique_gradient_size * unique_gradient_size < dense_size
    compressed_offset = packed_size if packed_compression else 0
    scratch_size = evaluation.gradient.size
    if not gradient_only:
      scratch_size = max(scratch_size, min(dense_size, packed_size + unique_gradient_size * unique_gradient_size))
    sortedDatas: List[attribute] = evaluation.kernelDatas
    sortedConnectivities: List[connectivity] = evaluation.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = evaluation.kernelPrimitiveUnions
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
  constexpr unsigned int HESSIAN_ROWS = {hessian_row_size};
  double intermediates[{scratch_size}];

  {evaluation.call(evaluation.gradient, "intermediates")}
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
      {atomic_add}(&gradient[segment_placement + j], intermediates[gradient_offset + j]);
    }}
    gradient_offset += segment_size;
  }}
  // Now check if we are also computing the Hessian


#if {int(not gradient_only)}
  {evaluation.call(evaluation.hessian, "intermediates") if not gradient_only else ""}

#if {int(not packed_compression)}
  // Expand the packed upper triangle backwards. Both symmetric destinations
  // are at or above the packed source index, so unread inputs remain intact.
  for (int row = HESSIAN_ROWS - 1; row >= 0; --row){{
    for (int column = HESSIAN_ROWS - 1; column >= row; --column){{
      const double value = symmetric_upper_get<HESSIAN_ROWS>(intermediates, row, column);
      intermediates[row * HESSIAN_ROWS + column] = value;
      intermediates[column * HESSIAN_ROWS + row] = value;
    }}
  }}
#endif
  Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> compressed_hessian(intermediates + {compressed_offset});
#if {int(packed_compression)}
  // For heavy compression, packed input plus a separate region for the
  // compressed result is smaller than expanding to the original dense size.
  compressed_hessian.setZero();
#endif

  // computePermutation preserves first-occurrence order: a compressed scalar
  // index never exceeds its original index, and N <= HESSIAN_ROWS. Traverse
  // the dense source in row-major order, clearing each consumed cell before
  // accumulating into an earlier/equal cell. No unread source is overwritten.
  unsigned int row_offset = 0;
  if (N != HESSIAN_ROWS){{ // Equal sizes imply the compression is the identity.
    for (unsigned int i = 0; i < {max_num_indices}; i++){{
      short int permutation_i = local_permutations[instance * {max_num_indices} + i];
      if (permutation_i < 0) permutation_i = -permutation_i;
      const unsigned short int segment_size_i = segment_sizes[instance * {max_num_indices} + i];
      for (unsigned int k = 0; k < segment_size_i; k++){{
        unsigned int col_offset = 0;
        for (unsigned int j = 0; j < {max_num_indices}; j++){{
          short int permutation_j = local_permutations[instance * {max_num_indices} + j];
          if (permutation_j < 0) permutation_j = -permutation_j;
          const unsigned short int segment_size_j = segment_sizes[instance * {max_num_indices} + j];
          for (unsigned int l = 0; l < segment_size_j; l++){{
#if {int(packed_compression)}
            const double value = symmetric_upper_get<HESSIAN_ROWS>(intermediates, row_offset + k, col_offset + l);
#else
            const unsigned int source = (row_offset + k) * HESSIAN_ROWS + col_offset + l;
            const double value = intermediates[source];
            intermediates[source] = 0.0;
#endif
            if (permutation_i != 0 && permutation_j != 0){{
              compressed_hessian(permutation_i - 1 + k, permutation_j - 1 + l) += value;
            }}
          }}
          col_offset += segment_size_j;
        }}
      }}
      row_offset += segment_size_i;
    }}
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
                {atomic_add}(&hessian_blocks[placement_index + k * segment_size_j + l], compressed_hessian(permutation_i + k, permutation_j + l));
              }}
            }}
          }}else{{
            for (unsigned int k = 0; k < segment_size_j; k++){{
              for (unsigned int l = 0; l < segment_size_i; l++){{
                // put the transpose block in
                {atomic_add}(&hessian_blocks[placement_index + k * segment_size_i + l], compressed_hessian(permutation_i + l, permutation_j + k));
              }}
            }}
          }}
          // additionally, if it is a diagonal block, we also need to put the diagonal elements
          if (i == j){{
            // get the placement
            unsigned int segment_index = segment_indices[instance * {max_num_indices} + i] - 2;
            for (unsigned int k = 0; k < segment_size_i; k++){{
              {atomic_add}(&diagonal[segment_index + k], compressed_hessian(permutation_i + k, permutation_j + k));
            }}
            // now we do the block diagonal placement
            // we first need to determine where to put it in the global diagonal blocks array
            int which_attribute = 0;
            for (int k = 0; k < {num_attributes}; k++){{
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
              {atomic_add}(&diagonal_blocks[diagonal_block_placement + k * segment_size_i + l], compressed_hessian(permutation_i + k, permutation_j + l));
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

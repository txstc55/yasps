from typing import List, Set
from yasps.attribute import attribute
from yasps.deviceKernel import deviceKernel
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
class hessianKernelNoProject:
  def __init__(self, att: attribute, unique_gradient_size: int, gradient_only: bool, max_num_indices: int, attributeName: str, num_attributes: int, hessian_row_size: int, grouped_add: bool = False):
    self.__att = att
    atomic_add = "atomic_add_grouped" if grouped_add else "atomicAdd"
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
  constexpr unsigned int HESSIAN_ROWS = {hessian_row_size};
  constexpr unsigned int PACKED_HESSIAN_SIZE = HESSIAN_ROWS * (HESSIAN_ROWS + 1) / 2;
  double hg_mat[{self.__att.size}]; // [packed upper Hessian, gradient]


  // now we call the device function
  {attributeName}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
    instance,
    hg_mat
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
      {atomic_add}(&gradient[segment_placement + j], hg_mat[PACKED_HESSIAN_SIZE + gradient_offset + j]);
#else
      {atomic_add}(&gradient[segment_placement + j], hg_mat[gradient_offset + j]);
  #endif
    }}
    gradient_offset += segment_size;
  }}
  // Now check if we are also computing the Hessian


#if {int(not gradient_only)}
  // first we allocate a new array, which computes that for each index
  short int unique_segment_placements[{max_num_indices}] = {{0}}; // this will count how many unique positions we can put the segment, and this is 0 based index
  unsigned short int unique_segment_placements_counts[{max_num_indices}] = {{0}}; // this will count how many segments are placed in each unique position, this is used for the compression
  short int inverse_map[{max_num_indices}] = {{0}}; // this will map the original segment index to the unique position index, this is also 0 based index
  short int unique_segment_sizes[{max_num_indices}] = {{0}}; // this will store the size of the segment for each unique position, this is used for the compression
  unsigned short int unique_segment_placement_first_i[{max_num_indices}] = {{0}}; // this will store, the first i for each unique segment placement


  unsigned short int current_unique_position_index = 0;
  short int last_placement = -1;
  for (unsigned short int i = 0; i < {max_num_indices}; i++){{
    short int permutation_i = local_permutations[instance * {max_num_indices} + i]; // get the permuted placement
    if (permutation_i == 0){{
      continue; // we encountered space reserved for union, skip
    }}
    if (permutation_i < 0) {{
      permutation_i = -permutation_i;
    }}
    permutation_i -= 1; // make it 0 indexed
    if (permutation_i > last_placement){{
      // this means its a new placement, record it
      unique_segment_placements[current_unique_position_index] = permutation_i;
      unique_segment_placements_counts[current_unique_position_index] += 1;
      unique_segment_sizes[current_unique_position_index] = segment_sizes[instance * {max_num_indices} + i];
      unique_segment_placement_first_i[current_unique_position_index] = i;
      inverse_map[i] = current_unique_position_index;
      current_unique_position_index += 1;
      last_placement = permutation_i;

    }}else{{
      // increment the count
      for (unsigned short int j = 0; j < current_unique_position_index; j++){{
        if (unique_segment_placements[j] == permutation_i){{
          unique_segment_placements_counts[j] += 1;
          inverse_map[i] = j;
          break;
        }}
      }}
    }}
  }}

  // now we construct the outer index array, which marks the range
  short int unique_segment_placements_outer[{max_num_indices} + 1] = {{0}};
  for (unsigned short int i = 0; i < current_unique_position_index; i++){{
    unique_segment_placements_outer[i + 1] = unique_segment_placements_outer[i] + unique_segment_placements_counts[i];
  }}
  // now we actually construct the inner array, which marks for each placement, where in the matrix column can we find the start of the block
  short int unique_segment_placements_inner[{max_num_indices}] = {{0}};
  for (unsigned short int i = 0; i < {max_num_indices}; i++){{
    unique_segment_placements_counts[i] = 0;  // reuse this array for local index counting
  }}



  unsigned int column_offset = 0;
  for (unsigned short int i = 0; i < {max_num_indices}; i++){{
    // first check if it is a unique permutation
    short int permutation_i = local_permutations[instance * {max_num_indices} + i];
    if (permutation_i == 0){{
      column_offset += segment_sizes[instance * {max_num_indices} + i];
      continue; // we encountered space reserved for union, skip
    }}
    // now we get the outer index
    const unsigned short int gid = inverse_map[i]; // find out which unique group it belongs to
    const unsigned short int start = unique_segment_placements_outer[gid]; // get the start of the group
    const unsigned short int local_count = unique_segment_placements_counts[gid]; // get how many segments have been placed in this group so far

    unique_segment_placements_inner[start + local_count] = column_offset; // record the column offset for this segment
    unique_segment_placements_counts[gid] = local_count + 1; // increment the count for this group
    column_offset += segment_sizes[instance * {max_num_indices} + i]; // increment the column offset
  }}


  // now we know for each placement index, the range of columns (and also rows) that corresponds to the same placement
  // we can start the accumulation and placement
  unsigned short int valid_block_counts = 0;
  const unsigned int coordinate_start = coordinatesOuter[instance];
  for (unsigned short int i = 0; i < current_unique_position_index; i++){{
    // this is one group
    // let's get the group's segment length
    const unsigned short int segment_size_i = unique_segment_sizes[i]; // what is the block row size
    unsigned int segment_index_i = segment_indices[instance * {max_num_indices} + unique_segment_placement_first_i[i]]; // what is its position in the global atrix
    if (segment_index_i < 2){{
      continue; // this is reserved for attributes that participate in the differentiation, but not in the final hessian, we skip it
    }}
    unsigned short int group_count_1 = unique_segment_placements_counts[i]; // how many rows in this block
    // let's also get the placement index
    for (unsigned short int j = i; j < current_unique_position_index; j++){{
      const unsigned short int segment_size_j = unique_segment_sizes[j]; // what is the block column size
      unsigned int segment_index_j = segment_indices[instance * {max_num_indices} + unique_segment_placement_first_i[j]]; // what is its position in the global matrix
      if (segment_index_j < 2){{
        continue; // this is reserved for attributes that participate in the differentiation, but not in the final hessian, we skip it
      }}
      unsigned short int group_count_2 = unique_segment_placements_counts[j]; // how many columns in this block
      // now we will start to accumulate the block
      unsigned int placement_index = lookups[coordinate_start + valid_block_counts];
      unsigned int diagonal_block_placement = 0; // initialize the diagonal block placement
      if (i == j){{
        const unsigned int segment_index = segment_index_i - 2;
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
        diagonal_block_placement = diagonal_block_start + which_instance * segment_size_i * segment_size_i;
      }}

      if (segment_index_i < segment_index_j) {{
        for (unsigned int k = 0; k < segment_size_i; k++) {{
          for (unsigned int l = 0; l < segment_size_j; l++) {{
            double acc = 0.0;

            for (unsigned short int g1_index = 0; g1_index < group_count_1; g1_index++) {{
              const unsigned int column_offset_i =
                unique_segment_placements_inner[unique_segment_placements_outer[i] + g1_index];

              for (unsigned short int g2_index = 0; g2_index < group_count_2; g2_index++) {{
                const unsigned int column_offset_j =
                  unique_segment_placements_inner[unique_segment_placements_outer[j] + g2_index];

                acc += symmetric_upper_get<HESSIAN_ROWS>(hg_mat, column_offset_i + k, column_offset_j + l);
              }}
            }}

            {atomic_add}(
              &hessian_blocks[placement_index + k * segment_size_j + l],
              acc
            );
          }}
        }}
      }}else {{
        for (unsigned int k = 0; k < segment_size_j; k++) {{
          for (unsigned int l = 0; l < segment_size_i; l++) {{
            double acc = 0.0;

            for (unsigned short int g1_index = 0; g1_index < group_count_1; g1_index++) {{
              const unsigned int column_offset_i =
                unique_segment_placements_inner[unique_segment_placements_outer[i] + g1_index];

              for (unsigned short int g2_index = 0; g2_index < group_count_2; g2_index++) {{
                const unsigned int column_offset_j =
                  unique_segment_placements_inner[unique_segment_placements_outer[j] + g2_index];

                // This computes block[l, k], matching your transpose placement.
                acc += symmetric_upper_get<HESSIAN_ROWS>(hg_mat, column_offset_i + l, column_offset_j + k);
              }}
            }}
            {atomic_add}(
              &hessian_blocks[placement_index + k * segment_size_i + l],
              acc
            );
            if (i == j) {{
              const unsigned int segment_index = segment_index_i - 2;
              {atomic_add}(
                &diagonal_blocks[diagonal_block_placement + k * segment_size_i + l],
                acc
              );
            }}
          }}
        }}
      }}
      valid_block_counts++; // finally increment
    }}
  }}
#endif // end for gradient only check
}}
}}
'''

  @property
  def kernelString(self):
    return self.__kernelString

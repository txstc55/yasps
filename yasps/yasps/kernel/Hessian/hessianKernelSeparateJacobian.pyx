from yasps.attribute import attribute
from typing import List
from yasps.scene import scene
from yasps.codeGenerator import codeGenerator
from yasps.deviceKernel import deviceKernel
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
from yasps.backend import is_metal
class hessianKernelSeparateJacobian:
  def __init__(
    self,
    att: attribute,
    gradient_only: bool = False, # whether we only want to compute the gradient, if true, we will skip the hessian computation and only compute the gradient, this is useful for the cases where we only need the gradient, and it can save us a lot of computation
  ):
    self.__stored_multiplied_blocks: List[attribute] = []
    self.__dependents: List[deviceKernel] = []
    self.__kernelString: str = ""
    self.__gradient_size: str = ""
    self.__att = att
    self.__merged_hessian_jacobian_nonzeros = 0
    self.__global_jacobian_children_sizes: List[int] = []
    self.__global_jacobian_children_spans: List[int] = []
    self.__gradient_only = gradient_only

  def create_multiplied_blocks(
    self,
    global_jacobian_block_nonzero_attributes: List[attribute],
    global_jacobian_block_nonzero_local_positions: List[attribute],
    global_jacobian_children_sizes: List[int],
    global_jacobian_children_spans: List[int]
  ):
    self.__global_jacobian_children_sizes = global_jacobian_children_sizes
    self.__global_jacobian_children_spans = global_jacobian_children_spans
    # if we need to separate the jacobian and hessian, the first thing we need to do is reconstruct the jacobian and hessian symbolically
    # which we will use to figure out how to compute the final hessian block by performing J_i^T H_ij J_j for each block
    base_scene_name = f"{self.__att.fullName}_tmp_variables_scene"
    tmp_scene = None
    candidate = base_scene_name
    self.__merged_hessian_jacobian_nonzeros = sum(global_jacobian_children_sizes) * sum(global_jacobian_children_sizes) + len(global_jacobian_block_nonzero_attributes)
    counter = 1
    while True:
      try:
        tmp_scene = scene(candidate)
        break
      except ValueError:
        candidate = f"{base_scene_name}_trial_{counter}"
        counter += 1

    # ok now we create the attributes with the sparsity pattern
    children_sizes = global_jacobian_children_sizes
    children_spans = global_jacobian_children_spans
    local_hessian_rows = sum(children_sizes)
    jacobian_cols = sum(children_spans)
    # we don't actually care about the actual attributes
    # we just want to make replacement attributes so we can do the multiplications
    local_hessian_replaced_att = tmp_scene.addAttribute("local_hessian", rows = local_hessian_rows, cols = local_hessian_rows)
    global_jacobian_replaced_att = tmp_scene.addAttribute("global_jacobian_nonzeros", rows = 1, cols = len(global_jacobian_block_nonzero_attributes))


    # ok we have created the local hessian, we now need to create the global jacobians
    global_jacobian_array = [0.0 for _ in range(local_hessian_rows * jacobian_cols)] # initialize the array
    for i in range(len(global_jacobian_block_nonzero_attributes)):
      local_position_row = global_jacobian_block_nonzero_local_positions[i * 2]
      local_position_col = global_jacobian_block_nonzero_local_positions[i * 2 + 1]
      global_jacobian_array[local_position_row * jacobian_cols + local_position_col] = global_jacobian_replaced_att[i]
    global_jacobian = attribute.to_array(global_jacobian_array, rows = local_hessian_rows, cols = jacobian_cols)

    # now the both have been created, we will need to do the block multiplications ourselves
    for i in range(len(children_sizes)):
      # first we get the diagonal block from the jacobian matrix
      # which is J_i
      J_i_block_rows = children_sizes[i]
      J_i_block_cols = children_spans[i]
      ji_block = self.__getSubBlock(global_jacobian, row_offset = sum(children_sizes[:i]), col_offset = sum(children_spans[:i]), block_rows = J_i_block_rows, block_cols = J_i_block_cols)
      ji_block = ji_block.transpose()

      for j in range(i, len(children_sizes)):
        # we need to get the H_ij block from the local hessian
        # as well as the J_j block from the jacobian matrix
        J_j_block_rows = children_sizes[j]
        J_j_block_cols = children_spans[j]
        jj_block = self.__getSubBlock(global_jacobian, row_offset = sum(children_sizes[:j]), col_offset = sum(children_spans[:j]), block_rows = J_j_block_rows, block_cols = J_j_block_cols)

        hij_block = self.__getSubBlock(local_hessian_replaced_att, row_offset = sum(children_sizes[:i]), col_offset = sum(children_sizes[:j]), block_rows = J_i_block_rows, block_cols = J_j_block_rows)
        multiplied_block = ji_block.mul_explicit(hij_block).mul_explicit(jj_block)
        multiplied_block = tmp_scene.addAttribute(f"multiplied_block_{i}_{j}", computed_attribute = multiplied_block)
        self.__stored_multiplied_blocks.append(multiplied_block)

    for item in self.__stored_multiplied_blocks:
      codegen: codeGenerator = codeGenerator(item)
      codegen.generateCode()
      device_kernel_string = item.deviceKernel.kernelString
      # now we do a bit of pruning and a bit of magic
      split_lines = device_kernel_string.splitlines()
      do_skip_lines = False
      corrected_lines = []
      for i in range(len(split_lines)):
        line = split_lines[i]
        if not do_skip_lines:
          # we hit the declaration
          if line.strip().startswith(f"__device__ void "):
            corrected_lines.append(f"__device__ void {item.fullName}_device_function(")
            corrected_lines.append(f" double* {local_hessian_replaced_att.fullName}_data,")
            corrected_lines.append(f" double* {global_jacobian_replaced_att.fullName}_data,")
            corrected_lines.append(" double* result\n){")
            do_skip_lines = True
            # let's also modify the kernel header here
            item.deviceKernel.kernelHeader = f"__device__ void {item.fullName}_device_function(double* {local_hessian_replaced_att.fullName}_data, double* {global_jacobian_replaced_att.fullName}_data, double* result)"
          # we hit a device function, which should be removed
          elif line.strip().startswith(f"{local_hessian_replaced_att.fullName}_device_function") or line.strip().startswith(f"{global_jacobian_replaced_att.fullName}_device_function"):
            do_skip_lines = True
            # we also need to remove the last line
            variable_declaration = corrected_lines.pop()
            intermediate_name = variable_declaration.strip().split()[-1].rstrip(";")
            computed_attribute = None
            if line.strip().startswith(f"{local_hessian_replaced_att.fullName}_device_function"):
              computed_attribute = local_hessian_replaced_att
            elif line.strip().startswith(f"{global_jacobian_replaced_att.fullName}_device_function"):
              computed_attribute = global_jacobian_replaced_att
            else:
              raise ValueError(f"hessianKernelSeparateJacobian: unexpected line when parsing device kernel string: {line.strip()}")

            # now recreate the variable
            redone_line = "  "
            if computed_attribute.size == 1:
              redone_line += f"float {intermediate_name} = {computed_attribute.fullName}_data[0];"
            elif computed_attribute.rows == 1 or computed_attribute.cols == 1:
              redone_line += f"Eigen::Map<Eigen::Matrix<double, {computed_attribute.rows}, {computed_attribute.cols}>> {intermediate_name}({computed_attribute.fullName}_data);"
            else:
              redone_line += f"Eigen::Map<Eigen::Matrix<double, {computed_attribute.rows}, {computed_attribute.cols}, Eigen::RowMajor>> {intermediate_name}({computed_attribute.fullName}_data);"
            corrected_lines.append(redone_line)
          else:
            corrected_lines.append(line)
        else:
          if (line.strip().startswith(");") or line.strip().startswith("){")) and do_skip_lines:
            do_skip_lines = False
      item.deviceKernel.kernelString = "\n".join(corrected_lines) # recreate the device kernel string with the corrected lines
      if is_metal():
        from yasps.backend.metal_codegen import translate_device_kernel

        function_start = item.deviceKernel.kernelString.find(
          f"__device__ void {item.fullName}_device_function("
        )
        function_body = item.deviceKernel.kernelString.find(
          "{",
          function_start
        )
        normalized_source = (
          item.deviceKernel.kernelString[:function_start]
          + item.deviceKernel.kernelHeader
          + item.deviceKernel.kernelString[function_body:]
        )
        metal_source, metal_header = translate_device_kernel(
          normalized_source,
          item.deviceKernel.kernelHeader,
          item.rows,
          item.cols
        )
        item.deviceKernel.metalKernelString = metal_source
        item.deviceKernel.metalKernelHeader = metal_header
      self.__dependents.append(item.deviceKernel) # add the device kernel



  def generateKernelString(
    self,
    unique_gradient_size: int,
    max_num_indices: int,
    attributeName: str,
  ):
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    global_jacobian_children_spans_outer = [0]
    hessian_rows = sum(self.__global_jacobian_children_sizes)
    for i in range(len(self.__global_jacobian_children_spans)):
      global_jacobian_children_spans_outer.append(global_jacobian_children_spans_outer[-1] + self.__global_jacobian_children_spans[i])
    self.__kernelString = f'''
#include "allHeaders.cuh"
extern "C"{{
__device__ __constant__ unsigned short int jacobian_block_spans_outer[{len(global_jacobian_children_spans_outer)}] = {{{', '.join(str(span) for span in global_jacobian_children_spans_outer)}}}; // this is the size of each jacobian block, this will also tell us how to segment the hessian blocks


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
  constexpr unsigned int N = {max(self.__global_jacobian_children_spans)}; // this is the size that we care about, this is the maximum of jacobian children's span, which means the largest column size for each block in the jacobian matrix. We will use this size for our allocation
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
#if {int(not self.__gradient_only)} // are we computing both the hessian and gradient
  Eigen::Matrix<double, 1, {self.__att.cols}> hg_mat = Eigen::Matrix<double, 1, {self.__att.cols}>::Zero(); // get the merged gradient and hessian
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
  #if {int(not self.__gradient_only)} // did we compute the hessian
      atomicAdd(&gradient[segment_placement + j], hg_mat(0, gradient_offset + {self.__merged_hessian_jacobian_nonzeros} + j));
  #else
      atomicAdd(&gradient[segment_placement + j], hg_mat(0, gradient_offset + j));
  #endif
    }}
    gradient_offset += segment_size;
  }}
  // Now check if we are also computing the Hessian


#if {int(not self.__gradient_only)}
  // first we allocate an array, this array tells us for the segmenets, which larger blocks they are in
  unsigned short int segment_in_large_block_outer[{len(global_jacobian_children_spans_outer)}] = {{0}};
  const unsigned int coordinate_start = coordinatesOuter[instance];
  unsigned short int segment_length = 0;
  unsigned short int current_large_block_index = 0;
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    unsigned short int segment_size = segment_sizes[instance * {max_num_indices} + i];
    segment_length += segment_size;
    if (segment_length == jacobian_block_spans_outer[current_large_block_index + 1]){{
      segment_in_large_block_outer[current_large_block_index + 1] = i + 1; // record the ending of a large segment
      current_large_block_index++;
    }}
  }}
  // now we have an outer index array, which tells us that for each large segment, what are the small segments inside it
  // now we will compute the large blocks one by one
  // first we allocate a data array
  double multiplied_block[N * N] = {{0.0}}; // this will store the multiplied block, we will reuse this for each block, since we are doing it sequentially
  unsigned short int row_offset = 0;
  unsigned short int col_offset = 0;
  unsigned short int valid_block_counts = 0; // this will count how many valid blocks we have encountered, which will be used for the lookup table to place the block in the correct position in the global hessian
'''
    block_count = 0
    for i in range(len(self.__global_jacobian_children_sizes)):
      for j in range(i, len(self.__global_jacobian_children_sizes)):
        # we only stored the upper ones anyway
        self.__kernelString += f'''
  // zero out the multiplied block first
  for (unsigned int i = 0; i < N * N; i++){{
    multiplied_block[i] = 0.0;
  }}
  {self.__stored_multiplied_blocks[block_count].fullName}_device_function(hg_mat.data(), hg_mat.data() + {hessian_rows * hessian_rows}, multiplied_block);
  row_offset = 0;
  col_offset = 0;
  // now we have the block for place {i}, {j}, we will want to know which small blocks needs to be placed back
'''

        self.__kernelString += f'''
  // process this large block now
  for (unsigned short int local_i = segment_in_large_block_outer[{i}]; local_i < segment_in_large_block_outer[{i + 1}]; local_i++){{
    short int segment_permutation_i = local_permutations[instance * {max_num_indices} + local_i];
    unsigned short int row_size = segment_sizes[instance * {max_num_indices} + local_i];
    if (segment_permutation_i <= 0){{
      row_offset += row_size;
      continue;
    }}
    unsigned int segment_placement_i = segment_indices[instance * {max_num_indices} + local_i];
    col_offset = 0; // reset the col offset for each new row segment
    for (unsigned short int local_j = segment_in_large_block_outer[{j}]; local_j < segment_in_large_block_outer[{j + 1}]; local_j++){{
      short int segment_permutation_j = local_permutations[instance * {max_num_indices} + local_j];
      unsigned short int col_size = segment_sizes[instance * {max_num_indices} + local_j];
      if (segment_permutation_j <= 0 || local_j < local_i){{
        // either this is a segment we don't need to place, or we are in the local triangle
        col_offset += col_size;
        continue;
      }}

      unsigned int segment_placement_j = segment_indices[instance * {max_num_indices} + local_j];
      const unsigned int placement = lookups[coordinate_start + valid_block_counts];
      // now we know the position to place the block, we will finally place it
      if (segment_placement_i <= segment_placement_j){{
        // correct placement, no transpose
        for (unsigned short int k = 0; k < row_size; k++){{
          for (unsigned short int l = 0; l < col_size; l++){{
            atomicAdd(&hessian_blocks[placement + k * col_size + l], multiplied_block[(row_offset + k) * {self.__stored_multiplied_blocks[block_count].cols} + col_offset + l]);
          }}
        }}
      }}else{{
        // we need to do transpose
        for (unsigned short int k = 0; k < col_size; k++){{
          for (unsigned short int l = 0; l < row_size; l++){{
            atomicAdd(&hessian_blocks[placement + k * row_size + l], multiplied_block[(row_offset + l) * {self.__stored_multiplied_blocks[block_count].cols} + col_offset + k]);
          }}
        }}
      }}
      if (segment_placement_i == segment_placement_j && segment_permutation_i != segment_permutation_j){{
        // we are hitting a diagonal block, but in the uncompressed Hessian, they are on off diagonals
        // so we also add the transpose
        for (unsigned short int k = 0; k < col_size; k++){{
          for (unsigned short int l = 0; l < row_size; l++){{
            atomicAdd(&hessian_blocks[placement + k * row_size + l], multiplied_block[(row_offset + l) * {self.__stored_multiplied_blocks[block_count].cols} + col_offset + k]);
          }}
        }}
      }}
      if (segment_placement_i == segment_placement_j){{
        const unsigned int segment_index = segment_placement_i - 2;
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
        const unsigned int which_instance = diff / (row_size);
        const unsigned int diagonal_block_placement = diagonal_block_start + which_instance * row_size * row_size;
        // now we place the diagonal block
        for (unsigned short int k = 0; k < row_size; k++){{
          for (unsigned short int l = 0; l < row_size; l++){{
            atomicAdd(
              &diagonal_blocks[diagonal_block_placement + k * row_size + l],
              multiplied_block[(row_offset + k) * {self.__stored_multiplied_blocks[block_count].cols} + col_offset + l]
            );
          }}
        }}
        if (segment_permutation_i != segment_permutation_j){{
          // we are on the diagonal, but they are permuted differently, which means in the uncompressed hessian, they are on the off diagonal, so we also need to place the transpose
          for (unsigned short int k = 0; k < row_size; k++){{
            for (unsigned short int l = 0; l < row_size; l++){{
              atomicAdd(
                &diagonal_blocks[diagonal_block_placement + k * row_size + l],
                multiplied_block[(row_offset + l) * {self.__stored_multiplied_blocks[block_count].cols} + col_offset + k]
              );
            }}
          }}
        }}
      }}


      col_offset += col_size; // increment the col offset
      valid_block_counts++; // we have placed a valid block, increment the count
    }}
    row_offset += row_size; // increment the row offset
  }}
'''
        block_count += 1
    self.__kernelString += f'''
#endif // end for gradient only check
}}
}}
'''
    return self.__kernelString


  def __getSubBlock(self, mat, row_offset, col_offset, block_rows, block_cols) -> attribute:
    block = [0.0 for _ in range(block_rows * block_cols)]
    for i in range(block_rows):
      for j in range(block_cols):
        block[i * block_cols + j] = mat[row_offset + i, col_offset + j]
    return attribute.to_array(block, rows = block_rows, cols = block_cols)

  @property
  def stored_multiplied_blocks(self) -> List[attribute]:
    return self.__stored_multiplied_blocks

  @property
  def dependents(self) -> List[deviceKernel]:
    return self.__dependents

  @property
  def kernelString(self) -> str:
    return self.__kernelString

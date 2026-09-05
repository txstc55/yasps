# cython: language_level=3
from yasps.attribute import attribute
from yasps.jacobianBlockLayout import generate_jacobian_block_layout


class hessianKernelSeparateJacobian:
  def __init__(self, att: attribute, gradient_only: bool = False, grouped_add: bool = False):
    self.__att = att
    self.__gradient_only = gradient_only
    self.__atomic_add = "atomic_add_grouped" if grouped_add else "atomicAdd"
    self.__kernelString = ""
    self.__layout = None
    self.__block_patterns = []
    self.__local_hessian_nonzero_count = 0
    self.__merged_hessian_jacobian_nonzeros = 0

  def create_multiplied_blocks(self, global_jacobian_block_nonzero_attributes, global_jacobian_block_nonzero_local_positions, global_jacobian_children_sizes, global_jacobian_children_spans, local_hessian_nonzero_upper_positions, global_jacobian_block_layout=None):
    # The permutations change multiplication order, not packed H/J value order.
    if len(local_hessian_nonzero_upper_positions) % 2:
      raise ValueError("Separate Hessian: expected row/column pairs for the inner Hessian.")
    if len(global_jacobian_block_nonzero_local_positions) != 2 * len(global_jacobian_block_nonzero_attributes):
      raise ValueError("Separate Hessian: Jacobian positions and values have different lengths.")
    self.__local_hessian_nonzero_count = len(local_hessian_nonzero_upper_positions) // 2
    self.__merged_hessian_jacobian_nonzeros = self.__local_hessian_nonzero_count + len(global_jacobian_block_nonzero_attributes)
    self.__block_patterns = []
    if self.__gradient_only:
      return
    if global_jacobian_block_layout is None:
      global_jacobian_block_layout = generate_jacobian_block_layout(sum(global_jacobian_children_sizes), sum(global_jacobian_children_spans), global_jacobian_block_nonzero_local_positions)
    self.__layout = global_jacobian_block_layout
    hessian_positions = {}
    for index in range(self.__local_hessian_nonzero_count):
      row, col = local_hessian_nonzero_upper_positions[2 * index:2 * index + 2]
      if not 0 <= row <= col < self.__layout["rows"] or (row, col) in hessian_positions:
        raise ValueError(f"Separate Hessian: invalid or repeated upper position ({row}, {col}).")
      hessian_positions[row, col] = index
      hessian_positions[col, row] = index
    jacobian_positions = {}
    for index in range(len(global_jacobian_block_nonzero_attributes)):
      row, col = global_jacobian_block_nonzero_local_positions[2 * index:2 * index + 2]
      jacobian_positions[row, col] = index

    # Generate J_i^T H_ij J_j from structural nonzeros. Reuse each left
    # contraction across an output row, without materializing dense J or H.
    for i, block_i in enumerate(self.__layout["blocks"]):
      for j in range(i, len(self.__layout["blocks"])):
        block_j = self.__layout["blocks"][j]
        lines = []
        for local_col_i, col_i in enumerate(block_i["cols"]):
          lines.append("  {")
          active_rows = []
          for local_row_j, row_j in enumerate(block_j["rows"]):
            products = []
            for row_i in block_i["rows"]:
              h = hessian_positions.get((row_i, row_j))
              left = jacobian_positions.get((row_i, col_i))
              if h is not None and left is not None:
                products.append(f"h[{h}] * jac[{left}]")
            if products:
              lines.append(f"    const double t{local_row_j} = {' + '.join(products)};")
              active_rows.append((local_row_j, row_j))
          for local_col_j, col_j in enumerate(block_j["cols"]):
            products = []
            for local_row_j, row_j in active_rows:
              right = jacobian_positions.get((row_j, col_j))
              if right is not None:
                products.append(f"t{local_row_j} * jac[{right}]")
            expression = " + ".join(products) if products else "0.0"
            lines.append(f"    result[{local_col_i * len(block_j['cols']) + local_col_j}] = {expression};")
          lines.append("  }")
        self.__block_patterns.append((i, j, "\n".join(lines)))

  def generateKernelString(self, unique_gradient_size: int, max_num_indices: int, attributeName: str, num_attributes: int):
    data = self.__att.deviceKernel.kernelDatas
    connectivity = self.__att.deviceKernel.kernelConnectivity
    unions = self.__att.deviceKernel.kernelPrimitiveUnions
    suffix = str(unique_gradient_size)
    declarations = "".join(f"const double* {x.code_generation_data_name}, " for x in data)
    declarations += "".join(f"const unsigned int* {x.code_generation_index_name}, " for x in connectivity)
    declarations += "".join(f"const unsigned int* {x.code_generation_csr_name}, " for x in connectivity if x.dimension == 0)
    declarations += "".join(f"const unsigned int* {x.code_generation_counts_name}, " for x in unions)
    arguments = "".join(f"{x.code_generation_data_name}, " for x in data)
    arguments += "".join(f"{x.code_generation_index_name}, " for x in connectivity)
    arguments += "".join(f"{x.code_generation_csr_name}, " for x in connectivity if x.dimension == 0)
    arguments += "".join(f"{x.code_generation_counts_name}, " for x in unions)
    source = ['#include "allHeaders.cuh"', 'extern "C" {']
    if not self.__gradient_only:
      for i, j, pattern in self.__block_patterns:
        source.append(f"static __device__ void multiply_sparse_block_{i}_{j}_{suffix}(const double* h, const double* jac, double* result) {{\n{pattern}\n}}")
      permutation = self.__layout["column_permutation"]
      # Symbolic column IDs; joins/unions resolve global DOFs at runtime.
      source.append(f"static __device__ __constant__ unsigned int jacobian_columns_{suffix}[{max(1, len(permutation))}] = {{{', '.join(map(str, permutation)) or '0'}}};")
      source.append(self.__scatterFunction(suffix, num_attributes))
    gradient_start = 0 if self.__gradient_only else self.__merged_hessian_jacobian_nonzeros
    source.append(f'''
__global__ void compute_hessian_and_gradient_global_function_final_gradient_size_{suffix}(
  {declarations}
  const unsigned int* segment_indices,
  const unsigned short int* segment_sizes,
  const short int* local_permutations,
  const unsigned int* lookups,
  const unsigned int* coordinatesOuter,
  const unsigned int* groupedIndicesInner,
  const unsigned int* groupedIndicesOuter,
  const unsigned int nth_gradient_size,
  const unsigned int projection_method,
  double* gradient,
  double* hessian_blocks,
  double* diagonal,
  double* diagonal_blocks,
  const unsigned int* diagonal_blocks_start,
  const unsigned int* gradient_segments_start
) {{
  const unsigned int start = groupedIndicesOuter[nth_gradient_size];
  const unsigned int end = groupedIndicesOuter[nth_gradient_size + 1];
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= end - start) return;
  const unsigned int instance = groupedIndicesInner[start + index];
  double hg_mat[{self.__att.size}];
  {attributeName}_device_function({arguments}instance, hg_mat);
  const unsigned int* indices = segment_indices + instance * {max_num_indices};
  const unsigned short int* sizes = segment_sizes + instance * {max_num_indices};
  const short int* permutations = local_permutations + instance * {max_num_indices};
  unsigned int gradient_offset = 0;
  for (unsigned int i = 0; i < {max_num_indices}; ++i) {{
    if (indices[i] >= 2) {{
      for (unsigned int k = 0; k < sizes[i]; ++k) {{
        {self.__atomic_add}(&gradient[indices[i] - 2 + k], hg_mat[{gradient_start} + gradient_offset + k]);
      }}
    }}
    gradient_offset += sizes[i];
  }}
''')
    if not self.__gradient_only and self.__block_patterns:
      jacobian_cols = self.__layout["cols"]
      max_block_entries = max(len(self.__layout['blocks'][i]['cols']) * len(self.__layout['blocks'][j]['cols']) for i, j, _ in self.__block_patterns)
      source.append(f'''
  // Invert segmentation once, retaining union padding in the original axes.
  unsigned short int column_segment[{max(1, jacobian_cols)}];
  unsigned int segment_outer[{max_num_indices + 1}];
  unsigned int valid_rank[{max_num_indices}]; // basically this tells us for each segment, which valid coordinates shall we look at. For example, if we have A, SKIP, B, C, then because we save only the valid ones when having the coordinates etc, so we know that B is the 1st valid one (0 indexed).
  unsigned int valid_count = 0;
  segment_outer[0] = 0;
  for (unsigned int i = 0; i < {max_num_indices}; ++i) {{
    segment_outer[i + 1] = segment_outer[i] + sizes[i];
    valid_rank[i] = valid_count;
    if (permutations[i] > 0 && indices[i] >= 2) ++valid_count; // if this is an actual value that we need to place back into the Hessian, then increment the valid rank
    for (unsigned int k = segment_outer[i]; k < segment_outer[i + 1]; ++k) {{
      column_segment[k] = i; // this records for each original column, which segment it belongs to, so that we can look up the segment for each original column when scattering back into the Hessian
    }}
  }}
  const unsigned int* instance_lookups = lookups + coordinatesOuter[instance];
  double multiplied_block[{max_block_entries}];
''')
      spans_outer = [0]
      for block in self.__layout["blocks"]:
        spans_outer.append(spans_outer[-1] + len(block["cols"]))
      for i, j, _ in self.__block_patterns:
        rows = len(self.__layout["blocks"][i]["cols"])
        cols = len(self.__layout["blocks"][j]["cols"])
        source.append(f'''
  multiply_sparse_block_{i}_{j}_{suffix}(hg_mat, hg_mat + {self.__local_hessian_nonzero_count}, multiplied_block);
  for (unsigned int a = 0; a < {rows}; ++a) {{
    for (unsigned int b = {'a' if i == j else '0'}; b < {cols}; ++b) {{
      const unsigned int original_a = jacobian_columns_{suffix}[{spans_outer[i]} + a]; // this is the row in the original Hessian
      const unsigned int original_b = jacobian_columns_{suffix}[{spans_outer[j]} + b]; // this is the col in the original Hessian
      scatter_sparse_hessian_{suffix}(multiplied_block[a * {cols} + b], original_a, original_b, column_segment, segment_outer, valid_rank, valid_count, indices, sizes, permutations, instance_lookups, hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);
    }}
  }}
''')
    source.append("}\n}\n")
    self.__kernelString = "\n".join(source)
    return self.__kernelString

  def __scatterFunction(self, suffix, num_attributes):
    # Component pairs supply one scalar triangle. Global storage contains
    # complete blocks: mirror off-diagonal scalars landing in a diagonal block,
    # including different local vertex occurrences mapping to the same vertex.
    return f'''
static __device__ __forceinline__ void scatter_sparse_hessian_{suffix}(
  double value, // the value
  unsigned int original_a, // the original row index in the Hessian
  unsigned int original_b, // the original column index in the Hessian
  const unsigned short int* column_segment, // the mapping from original column index to segment index
  const unsigned int* segment_outer, // the outer array for each segment, recording the starting index of each segment in the original column indices
  const unsigned int* valid_rank, unsigned int valid_count, // stores for each segment, basically if it is avalid
  const unsigned int* indices, const unsigned short int* sizes,
  const short int* permutations, const unsigned int* lookups,
  double* hessian_blocks, double* diagonal_blocks,
  const unsigned int* diagonal_blocks_start, const unsigned int* gradient_segments_start
) {{
  const unsigned int segment_a = column_segment[original_a]; // which segment (in row space)
  const unsigned int segment_b = column_segment[original_b]; // which segment (in column space)
  if (permutations[segment_a] <= 0 || permutations[segment_b] <= 0 || indices[segment_a] < 2 || indices[segment_b] < 2) return; // check if we want to place it back at all
  const unsigned int rank_a = valid_rank[segment_a];
  const unsigned int rank_b = valid_rank[segment_b];
  const unsigned int first = min(rank_a, rank_b);
  const unsigned int last = max(rank_a, rank_b);
  // Coordinate generation enumerates original valid segment pairs in this
  // upper-triangular order, independently of the Jacobian permutation.
  const unsigned int lookup_index = first * valid_count - first * (first + 1) / 2 + last;
  const unsigned int placement = lookups[lookup_index];
  const unsigned int offset_a = original_a - segment_outer[segment_a];
  const unsigned int offset_b = original_b - segment_outer[segment_b];
  const unsigned int start_a = indices[segment_a];
  const unsigned int start_b = indices[segment_b];
  if (start_a <= start_b) {{
    {self.__atomic_add}(&hessian_blocks[placement + offset_a * sizes[segment_b] + offset_b], value);
  }} else {{
    {self.__atomic_add}(&hessian_blocks[placement + offset_b * sizes[segment_a] + offset_a], value);
  }}
  if (start_a == start_b) {{
    const unsigned int size = sizes[segment_a];
    if (original_a != original_b) {{
      {self.__atomic_add}(&hessian_blocks[placement + offset_b * size + offset_a], value);
    }}
    const unsigned int segment_start = start_a - 2;
    unsigned int which_attribute = 0;
    while (which_attribute + 1 < {num_attributes} && segment_start >= gradient_segments_start[which_attribute + 1]) ++which_attribute;
    const unsigned int local_instance = (segment_start - gradient_segments_start[which_attribute]) / size;
    const unsigned int diagonal_start = diagonal_blocks_start[which_attribute] + local_instance * size * size;
    {self.__atomic_add}(&diagonal_blocks[diagonal_start + offset_a * size + offset_b], value);
    if (original_a != original_b) {{
      {self.__atomic_add}(&diagonal_blocks[diagonal_start + offset_b * size + offset_a], value);
    }}
  }}
}}
'''

  @property
  def dependents(self):
    return []

  @property
  def kernelString(self):
    return self.__kernelString

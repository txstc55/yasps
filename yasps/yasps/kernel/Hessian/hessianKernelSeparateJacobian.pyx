# cython: language_level=3
from yasps.attribute import attribute
from yasps.jacobianBlockLayout import generate_jacobian_block_layout, pack_jacobian_block_nonzeros
from yasps.jacobianBlockLayout import generate_inner_hessian_block_layout, pack_inner_hessian_block_nonzeros, generate_segment_block_layout
import ctypes
import numpy as np
import pycuda.gpuarray as gpuarray


class hessianKernelSeparateJacobian:
  def __init__(self, att: attribute, gradient_only: bool = False, grouped_add: bool = False, auto_partition: int = 0, segment_sizes = None):
    self.__att = att
    self.__gradient_only = gradient_only
    self.__auto_partition = auto_partition
    self.__segment_sizes = segment_sizes
    self.__atomic_add = "atomic_add_grouped" if grouped_add else "atomicAdd"
    self.__kernelString = ""
    self.__layout = None
    self.__block_patterns = []
    self.__patterns = []
    self.__hessian_operands = []
    self.__packed_jacobian = None
    self.__local_hessian_nonzero_count = 0
    self.__merged_hessian_jacobian_nonzeros = 0
    self.__left_patterns = []
    self.__inner_products = []
    self.__activity_kernel = None
    self.__block_activity = None
    self.__activity_state = None

  def bindBlockActivityKernel(self, library):
    if self.__gradient_only or not self.__block_patterns:
      return
    self.__activity_kernel = library.recompute_hessian_block_activity
    self.__activity_kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32]
    self.__activity_kernel.restype = ctypes.c_int

  def invalidateBlockActivity(self):
    self.__activity_state = None

  def recomputeBlockActivity(self, indices_kernel, force=False):
    if self.__activity_kernel is None:
      return
    count = indices_kernel.numInstances
    state = (indices_kernel, count)
    if not force and self.__activity_state == state:
      return
    length = count * len(self.__layout["blocks"])
    if self.__block_activity is None or self.__block_activity.size < length:
      self.__block_activity = gpuarray.empty(length, np.uint8)
    if count:
      inputs = [indices_kernel.outputIndices, indices_kernel.outputSizes, indices_kernel.outputPermutations, self.__block_activity]
      error = self.__activity_kernel(*[ctypes.c_void_p(int(value.gpudata)) for value in inputs], ctypes.c_uint32(count))
      if error:
        raise RuntimeError(f"Separate Hessian: block activity kernel failed with CUDA error {error}.")
    self.__activity_state = state

  @property
  def blockActivity(self):
    return self.__block_activity

  def __activitySource(self, suffix, max_num_indices):
    # Map ORIGINAL scalar columns to multiplication blocks. Runtime segment
    # widths include union padding, so a symbolic block need not align with a
    # single segment. Zero Jacobian columns cannot activate a block.
    column_blocks = [65535] * self.__layout["cols"]
    for block_id, block in enumerate(self.__layout["blocks"]):
      for _, column in self.__packed_jacobian["block_local_positions"][block_id]:
        column_blocks[block["cols"][column]] = block_id
    if len(self.__layout["blocks"]) > 65535:
      raise ValueError("Separate Hessian: block activity exceeds uint16 local index capacity.")
    # Accumulate flags in a few registers, then write each byte just once.
    # Directly setting a byte for every valid scalar repeats many global stores.
    words = (len(self.__layout["blocks"]) + 31) // 32
    masks = "\n  ".join(f"unsigned int active_{word} = 0;" for word in range(words))
    mark = "\n        ".join(f"{'if' if word == 0 else 'else if'} (block / 32 == {word}) active_{word} |= 1u << (block % 32);" for word in range(words))
    reduce_masks = "\n    ".join(f"active_{word} |= __shfl_xor_sync(0xffffffffu, active_{word}, offset);" for word in range(words))
    stores = "\n  ".join(f"if (lane < {min(32, len(self.__layout['blocks']) - 32 * word)}) activity[(size_t)({32 * word} + lane) * count + instance] = (active_{word} >> lane) & 1u;" for word in range(words))
    return f'''
static __device__ const unsigned short int activity_column_blocks_{suffix}[{max(1, len(column_blocks))}] = {{{', '.join(map(str, column_blocks)) or '65535'}}};
__global__ void recompute_hessian_block_activity_kernel(
  const unsigned int* segment_indices, const unsigned short int* segment_sizes,
  const short int* local_permutations, unsigned char* activity, unsigned int count
) {{
  // One warp cooperates on an instance's contiguous runtime segments.
  const unsigned int instance = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const unsigned int lane = threadIdx.x & 31;
  if (instance >= count) return;
  {masks}
  unsigned int column = 0;
  #pragma unroll 1
  for (unsigned int first = 0; first < {max_num_indices}; first += 32) {{
    const unsigned int segment = first + lane;
    const size_t index = (size_t)instance * {max_num_indices} + segment;
    const unsigned int size = segment < {max_num_indices} ? segment_sizes[index] : 0;
    unsigned int prefix = size;
    #pragma unroll
    for (unsigned int offset = 1; offset < 32; offset *= 2) {{
      const unsigned int other = __shfl_up_sync(0xffffffffu, prefix, offset);
      if (lane >= offset) prefix += other;
    }}
    const unsigned int end = column + prefix;
    if (segment < {max_num_indices} && local_permutations[index] > 0 && segment_indices[index] >= 2) {{
      for (unsigned int c = end - size; c < end; ++c) {{
        const unsigned short int block = activity_column_blocks_{suffix}[c];
        {mark}
      }}
    }}
    column += __shfl_sync(0xffffffffu, prefix, 31);
  }}
  #pragma unroll
  for (unsigned int offset = 1; offset < 32; offset *= 2) {{
    {reduce_masks}
  }}
  // Block-major bytes make the later assembly's instance reads coalesced.
  {stores}
}}
int recompute_hessian_block_activity(
  const unsigned int* indices, const unsigned short int* sizes,
  const short int* permutations, unsigned char* activity, unsigned int count
) {{
  if (count) recompute_hessian_block_activity_kernel<<<(count + 3) / 4, 128>>>(indices, sizes, permutations, activity, count);
  // The following assembly uses separate streams; complete the prepass first.
  return (int)cudaDeviceSynchronize();
}}
'''

  def create_multiplied_blocks(self,
    global_jacobian_block_nonzero_attributes,
    global_jacobian_block_nonzero_local_positions,
    global_jacobian_children_sizes,
    global_jacobian_children_spans,
    local_hessian_nonzero_upper_positions,
    global_jacobian_block_layout=None):
    # Only J values change storage order; original scalar axes and scatter stay
    # unchanged. The symbolic HJ producer uses this same packing helper.
    if len(local_hessian_nonzero_upper_positions) % 2:
      raise ValueError("Separate Hessian: expected row/column pairs for the inner Hessian.")
    if len(global_jacobian_block_nonzero_local_positions) != 2 * len(global_jacobian_block_nonzero_attributes):
      raise ValueError("Separate Hessian: Jacobian positions and values have different lengths.")
    self.__local_hessian_nonzero_count = len(local_hessian_nonzero_upper_positions) // 2
    self.__merged_hessian_jacobian_nonzeros = self.__local_hessian_nonzero_count + len(global_jacobian_block_nonzero_attributes)
    self.__block_patterns = []
    self.__patterns = []
    self.__hessian_operands = []
    if self.__gradient_only:
      return
    if global_jacobian_block_layout is None:
      global_jacobian_block_layout = generate_jacobian_block_layout(sum(global_jacobian_children_sizes), sum(global_jacobian_children_spans), global_jacobian_block_nonzero_local_positions)
    if self.__auto_partition == 1:
      self.__layout = global_jacobian_block_layout
      self.__packed_jacobian = pack_jacobian_block_nonzeros(self.__layout, global_jacobian_block_nonzero_local_positions)
    else:
      rows, cols = global_jacobian_block_layout["rows"], global_jacobian_block_layout["cols"]
      self.__layout = generate_segment_block_layout(rows, cols, self.__segment_sizes) if self.__auto_partition >= 2 else generate_inner_hessian_block_layout(rows, cols)
      self.__packed_jacobian = pack_inner_hessian_block_nonzeros(self.__layout, global_jacobian_block_nonzero_local_positions)

    # first we construct the hessian nonzero positions
    hessian_positions = {}
    for index in range(self.__local_hessian_nonzero_count):
      row, col = local_hessian_nonzero_upper_positions[2 * index:2 * index + 2]
      if not 0 <= row <= col < self.__layout["rows"] or (row, col) in hessian_positions:
        raise ValueError(f"Separate Hessian: invalid or repeated upper position ({row}, {col}).")
      hessian_positions[row, col] = index
      hessian_positions[col, row] = index
    if self.__auto_partition != 1:
      self.__createInnerBlocks(hessian_positions)
      return
    patterns = {}
    blocks = self.__layout["blocks"]
    for i, block_i in enumerate(blocks):
      for j in range(i, len(blocks)):
        block_j = blocks[j]
        # we have the block i and j
        left = self.__packed_jacobian["block_local_positions"][i]
        right = self.__packed_jacobian["block_local_positions"][j]
        # get the active rows and columns sets
        active_left_rows = {r for r, _ in left} # get the rows of all the nonzero left jacobians block
        active_right_rows = {r for r, _ in right} # get the rows of all the nonzero right jacobians block
        # construct which Hessian values do we need for the computation
        h_ids = {}
        h_values = []
        h_support = {}
        for r, original_r in enumerate(block_i["rows"]):
          for s, original_s in enumerate(block_j["rows"]):
            h = hessian_positions.get((original_r, original_s))
            if h is None:
              continue
            if h not in h_ids:
              h_ids[h] = len(h_values)
              h_values.append(h)
            h_support[r, s] = h_ids[h]
        # Omit structurally zero products and their scatter, not numeric zeros.
        if not h_values:
          continue
        h_start = h_values[0]
        mapped_h = h_values != list(range(h_start, h_start + len(h_values)))
        key = (len(block_i["cols"]), len(block_j["cols"]), tuple(left), tuple(right), tuple(h_support.items()), mapped_h)
        if key in patterns:
          pattern_id = patterns[key]
          self.__block_patterns.append((i, j, pattern_id))
          self.__hessian_operands.append(h_values if mapped_h else h_start)
          continue
        # if no pattern has been found, we generate the multiplication code.
        lines = []
        for local_col_i in range(len(block_i["cols"])):
          lines.append("  {")
          active_rows = []
          for local_row_j in range(len(block_j["rows"])):
            products = []
            for local_row_i in range(len(block_i["rows"])):
              h = h_support.get((local_row_i, local_row_j))
              left_id = left.get((local_row_i, local_col_i))
              if h is not None and left_id is not None:
                h_access = f"h[h_indices[{h}]]" if mapped_h else f"h[{h}]"
                products.append(f"{h_access} * left_jac[{left_id}]")
            if products:
              lines.append(f"    const double t{local_row_j} = {' + '.join(products)};")
              active_rows.append(local_row_j)
          for local_col_j in range(len(block_j["cols"])):
            products = []
            for local_row_j in active_rows:
              right_id = right.get((local_row_j, local_col_j))
              if right_id is not None:
                products.append(f"t{local_row_j} * right_jac[{right_id}]")
            expression = " + ".join(products) if products else "0.0"
            lines.append(f"    result[{local_col_i * len(block_j['cols']) + local_col_j}] = {expression};")
          lines.append("  }")
        pattern_id = len(self.__patterns)
        patterns[key] = pattern_id
        self.__patterns.append({"source": "\n".join(lines), "mapped_h": mapped_h})
        self.__block_patterns.append((i, j, pattern_id))
        self.__hessian_operands.append(h_values if mapped_h else h_start)

  def __createInnerBlocks(self, hessian_positions):
    # All tiles share H's original row axes. Pattern IDs depend on local
    # structural positions, never on offsets into the packed numeric J array.
    size = self.__layout["rows"]
    self.__left_patterns = []
    self.__inner_products = []
    left_ids = {}
    for block_id, block in enumerate(self.__layout["blocks"]):
      jacobian = self.__packed_jacobian["block_local_positions"][block_id]
      if not jacobian:
        continue
      width = len(block["cols"])
      key = (width, tuple(jacobian))
      if key in left_ids:
        self.__left_patterns[left_ids[key]]["blocks"].append(block_id)
        continue
      lines, support, expressions = [], set(), {}
      for a in range(width):
        for s in range(size):
          products = []
          for r in range(size):
            h = hessian_positions.get((r, s))
            j = jacobian.get((r, a))
            if h is not None and j is not None:
              products.append(f"left_jac[{j}] * h[{h}]")
          if products:
            support.add((a, s))
            expressions[a, s] = " + ".join(products)
            lines.append(f"  result[{a * size + s}] = {' + '.join(products)};")
          elif self.__auto_partition == 2:
            lines.append(f"  result[{a * size + s}] = 0.0;")
      left_ids[key] = len(self.__left_patterns)
      self.__left_patterns.append({"source": "\n".join(lines), "support": support, "expressions": expressions, "width": width, "jacobian": jacobian, "blocks": [block_id]})

    patterns = {}
    for left_id, left in enumerate(self.__left_patterns):
      for right_id, right in enumerate(self.__left_patterns):
        pairs = [(i, j) for i in left["blocks"] for j in right["blocks"] if i <= j]
        if not pairs:
          continue
        key = (left["width"], right["width"], tuple(sorted(left["support"])), tuple(right["jacobian"]))
        if self.__auto_partition == 3:
          key += (tuple(left["jacobian"]),)
        if key not in patterns:
          lines, positions = [], []
          result_stride = right["width"] if self.__auto_partition >= 2 else size
          for a in range(left["width"]):
            if self.__auto_partition == 3:
              lines.append("  {")
              for s in range(size):
                if (a, s) in left["expressions"]:
                  lines.append(f"    const double t{s} = {left['expressions'][a, s]};")
            for b in range(right["width"]):
              products = []
              for s in range(size):
                j = right["jacobian"].get((s, b))
                if (a, s) in left["support"] and j is not None:
                  partial = f"t{s}" if self.__auto_partition == 3 else f"partial[{a * size + s}]"
                  products.append(f"{partial} * right_jac[{j}]")
              if products:
                positions.append((a, b))
                lines.append(f"  result[{a * result_stride + b}] = {' + '.join(products)};")
            if self.__auto_partition == 3:
              lines.append("  }")
          # A structurally zero tile pair needs neither multiplication nor scatter.
          patterns[key] = len(self.__patterns) if positions else None
          if positions:
            self.__patterns.append({"source": "\n".join(lines), "mapped_h": False, "positions": positions, "dense": len(positions) == left["width"] * right["width"], "rows": left["width"], "cols": right["width"]})
        pattern_id = patterns[key]
        if pattern_id is not None:
          self.__inner_products.append((left_id, right_id, pattern_id))
          self.__block_patterns.extend((i, j, pattern_id) for i, j in pairs)

  def __innerFunctionSource(self, suffix, constant_bytes):
    source = []
    for i, pattern in enumerate(self.__left_patterns):
      source.append(f"static __device__ __noinline__ void multiply_left_pattern_{i}_{suffix}(const double* left_jac, const double* h, double* __restrict__ result) {{\n{pattern['source']}\n}}")
    for i, pattern in enumerate(self.__patterns):
      source.append(f"static __device__ __noinline__ void multiply_right_pattern_{i}_{suffix}(const double* partial, const double* right_jac, double* __restrict__ result) {{\n{pattern['source']}\n}}")
    tables = [("tile_nonzero_offsets", self.__packed_jacobian["block_offsets"])]
    tables += [(f"tile_group_{i}", pattern["blocks"]) for i, pattern in enumerate(self.__left_patterns)]
    tables += [(f"product_coordinates_{i}", [axis for position in pattern["positions"] for axis in position]) for i, pattern in enumerate(self.__patterns) if not pattern["dense"]]
    for name, values in tables:
      if max(values, default=0) > 65535:
        raise ValueError("Inner Hessian tiles exceed uint16 local offset capacity.")
      table_bytes = max(1, len(values)) * 2
      storage = "__constant__" if constant_bytes + table_bytes <= 65536 else "const"
      if storage == "__constant__":
        constant_bytes += table_bytes
      source.append(f"static __device__ {storage} unsigned short int {name}_{suffix}[{max(1, len(values))}] = {{{', '.join(map(str, values)) or '0'}}};")
    return source

  def __innerCallerSource(self, suffix, max_num_indices):
    size = self.__layout["rows"]
    source = [f'''
  double left_product[{size * size}];
''']
    for left_id, left in enumerate(self.__left_patterns):
      products = [(right_id, pattern_id) for current, right_id, pattern_id in self.__inner_products if current == left_id]
      if not products:
        continue
      source.append(f'''
  #pragma unroll 1
  for (unsigned short int left_id = 0; left_id < {len(left['blocks'])}; ++left_id) {{
    const unsigned short int i = tile_group_{left_id}_{suffix}[left_id];
    if (!block_activity[(size_t)i * activity_stride]) continue;
    multiply_left_pattern_{left_id}_{suffix}(hg_mat + {self.__local_hessian_nonzero_count} + tile_nonzero_offsets_{suffix}[i], hg_mat, left_product);
''')
      for right_id, pattern_id in products:
        right = self.__left_patterns[right_id]
        source.append(f'''
    #pragma unroll 1
    for (unsigned short int right_id = 0; right_id < {len(right['blocks'])}; ++right_id) {{
      const unsigned short int j = tile_group_{right_id}_{suffix}[right_id];
      if (j < i || !block_activity[(size_t)j * activity_stride]) continue;
      multiply_right_pattern_{pattern_id}_{suffix}(left_product, hg_mat + {self.__local_hessian_nonzero_count} + tile_nonzero_offsets_{suffix}[j], multiplied_block);
''')
        pattern = self.__patterns[pattern_id]
        if pattern["dense"] and size == 1:
          # Scalar inner energies have no repeated segment lookup to hoist.
          # Keep the inline scalar path rather than adding a device call.
          source.append(f'''
      scatter_sparse_hessian_{suffix}(multiplied_block[0], i, j, column_segment, segment_outer, valid_rank, valid_count, indices, sizes, permutations, instance_lookups, hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);
''')
        elif pattern["dense"]:
          source.append(f'''
      scatter_dense_tile_{suffix}(multiplied_block, i * {size}, j * {size}, {left['width']}, {right['width']}, column_segment, segment_outer, valid_rank, valid_count, indices, sizes, permutations, instance_lookups, hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);
''')
        else:
          # Sparse inner energies (for example one-axis walls) must not
          # compute or atomically scatter the other structural-zero scalars.
          source.append(f'''
      #pragma unroll 1
      for (unsigned int entry = 0; entry < {len(pattern['positions'])}; ++entry) {{
        const unsigned short int a = product_coordinates_{pattern_id}_{suffix}[2 * entry];
        const unsigned short int b = product_coordinates_{pattern_id}_{suffix}[2 * entry + 1];
        if (i == j && b < a) continue;
        scatter_sparse_hessian_{suffix}(multiplied_block[a * {size} + b], i * {size} + a, j * {size} + b, column_segment, segment_outer, valid_rank, valid_count, indices, sizes, permutations, instance_lookups, hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);
      }}
''')
        source.append("    }")
      source.append("  }")
    return "\n".join(source)

  def __segmentFunctionSource(self, suffix, num_attributes):
    # All metadata is shared read-only device data, never a per-thread copy.
    tables = [("segment_offsets", [block["cols"][0] for block in self.__layout["blocks"]]), ("segment_jacobian_offsets", self.__packed_jacobian["block_offsets"])]
    source = []
    if self.__auto_partition == 2:
      left_ids = [65535] * len(self.__layout["blocks"])
      for pattern_id, pattern in enumerate(self.__left_patterns):
        for block in pattern["blocks"]:
          left_ids[block] = pattern_id
        source.append(f"static __device__ __noinline__ void multiply_segment_left_{pattern_id}_{suffix}(const double* left_jac, const double* h, double* __restrict__ result) {{\n{pattern['source']}\n}}")
      tables.append(("segment_left_patterns", left_ids))
    pairs = sorted(self.__block_patterns)
    tables.append(("segment_pairs", [value for pair in pairs for value in pair]))
    for pattern_id, pattern in enumerate(self.__patterns):
      operands = "const double* partial, const double* right_jac" if self.__auto_partition == 2 else "const double* left_jac, const double* h, const double* right_jac"
      source.append(f"static __device__ __noinline__ void multiply_segment_{pattern_id}_{suffix}({operands}, double* __restrict__ result) {{\n{pattern['source']}\n}}")
      if not pattern["dense"]:
        tables.append((f"segment_product_coordinates_{pattern_id}", [axis for position in pattern["positions"] for axis in position]))
    constant_bytes = 0
    for name, values in tables:
      if max(values, default=0) > 65535:
        raise ValueError("Segment assembly exceeds uint16 local metadata capacity.")
      table_bytes = max(1, len(values)) * 2
      storage = "__constant__" if constant_bytes + table_bytes <= 65536 else "const"
      if storage == "__constant__":
        constant_bytes += table_bytes
      source.append(f"static __device__ {storage} unsigned short int {name}_{suffix}[{max(1, len(values))}] = {{{', '.join(map(str, values)) or '0'}}};")
    for pattern_id, pattern in enumerate(self.__patterns):
      source.append(self.__segmentScatterSource(suffix, pattern_id, pattern, num_attributes))
    return source

  def __segmentScatterSource(self, suffix, pattern_id, pattern, num_attributes):
    rows, cols = pattern["rows"], pattern["cols"]
    if pattern["dense"]:
      entries = f"const unsigned int a = entry / {cols}, b = entry % {cols};"
    else:
      entries = f"const unsigned int a = segment_product_coordinates_{pattern_id}_{suffix}[2 * entry], b = segment_product_coordinates_{pattern_id}_{suffix}[2 * entry + 1];"
    return f'''
static __device__ __noinline__ void scatter_segment_{pattern_id}_{suffix}(
  const double* block, unsigned int i, unsigned int j,
  const unsigned short int* valid_rank, unsigned int valid_count,
  const unsigned int* indices, const unsigned int* lookups,
  double* hessian_blocks, double* diagonal_blocks,
  const unsigned int* diagonal_blocks_start, const unsigned int* gradient_segments_start
) {{
  // One lookup and orientation decision for the WHOLE destination block.
  // i <= j is the original local segment order, not the global vertex order.
  const unsigned int first = valid_rank[i], last = valid_rank[j];
  const unsigned int placement = lookups[first * valid_count - first * (first + 1) / 2 + last];
  const unsigned int start_a = indices[i], start_b = indices[j];
  const bool transpose = start_a > start_b;
  const bool diagonal = start_a == start_b;
  unsigned int diagonal_start = 0;
  if (diagonal) {{
    const unsigned int segment_start = start_a - 2;
    unsigned int attribute_id = 0;
    while (attribute_id + 1 < {num_attributes} && segment_start >= gradient_segments_start[attribute_id + 1]) ++attribute_id;
    const unsigned int local_instance = (segment_start - gradient_segments_start[attribute_id]) / {rows};
    diagonal_start = diagonal_blocks_start[attribute_id] + local_instance * {rows * rows};
  }}
  #pragma unroll 1
  for (unsigned int entry = 0; entry < {len(pattern['positions'])}; ++entry) {{
    {entries}
    if (i == j && b < a) continue;
    const double value = block[a * {cols} + b];
    {self.__atomic_add}(&hessian_blocks[placement + (transpose ? b * {rows} + a : a * {cols} + b)], value);
    if (diagonal) {{
      {self.__atomic_add}(&diagonal_blocks[diagonal_start + a * {rows} + b], value);
      // Distinct local occurrences of the same vertex contribute B + B^T,
      // including twice their diagonal. A single segment supplies one triangle.
      if (i != j || a != b) {{
        {self.__atomic_add}(&hessian_blocks[placement + b * {rows} + a], value);
        {self.__atomic_add}(&diagonal_blocks[diagonal_start + b * {rows} + a], value);
      }}
    }}
  }}
}}
'''

  def __segmentCallerSource(self, suffix, max_num_indices):
    size = self.__layout["rows"]
    source = []
    if self.__auto_partition == 2:
      source.append(f'''
  // Store J^T H in original outer-row order. Only active rows are read later.
  double left_product[{self.__layout['cols'] * size}];
  #pragma unroll 1
  for (unsigned int i = 0; i < {max_num_indices}; ++i) {{
    if (!block_activity[(size_t)i * activity_stride]) continue;
    switch (segment_left_patterns_{suffix}[i]) {{
''')
      for pattern_id in range(len(self.__left_patterns)):
        source.append(f"      case {pattern_id}: multiply_segment_left_{pattern_id}_{suffix}(hg_mat + {self.__local_hessian_nonzero_count} + segment_jacobian_offsets_{suffix}[i], hg_mat, left_product + segment_offsets_{suffix}[i] * {size}); break;")
      source.append("    }\n  }")
    source.append(f'''
  // Structurally nonzero final blocks, in original row-wise upper-triangle order.
  #pragma unroll 1
  for (unsigned int pair = 0; pair < {len(self.__block_patterns)}; ++pair) {{
    const unsigned int i = segment_pairs_{suffix}[3 * pair], j = segment_pairs_{suffix}[3 * pair + 1];
    if (!block_activity[(size_t)i * activity_stride] || !block_activity[(size_t)j * activity_stride]) continue;
''')
    if len(self.__patterns) > 1:
      source.append(f"    switch (segment_pairs_{suffix}[3 * pair + 2]) {{")
    for pattern_id in range(len(self.__patterns)):
      if len(self.__patterns) > 1:
        source.append(f"      case {pattern_id}: {{")
      operands = f"left_product + segment_offsets_{suffix}[i] * {size}" if self.__auto_partition == 2 else f"hg_mat + {self.__local_hessian_nonzero_count} + segment_jacobian_offsets_{suffix}[i], hg_mat"
      source.append(f"    multiply_segment_{pattern_id}_{suffix}({operands}, hg_mat + {self.__local_hessian_nonzero_count} + segment_jacobian_offsets_{suffix}[j], multiplied_block);")
      source.append(f"    scatter_segment_{pattern_id}_{suffix}(multiplied_block, i, j, valid_rank, valid_count, indices, instance_lookups, hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);")
      if len(self.__patterns) > 1:
        source.append("        break;\n      }")
    if len(self.__patterns) > 1:
      source.append("    }")
    source.append("  }")
    return "\n".join(source)

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
    if not self.__gradient_only and self.__block_patterns:
      if self.__auto_partition >= 2 and max_num_indices != len(self.__segment_sizes):
        raise ValueError("Segment assembly layout does not match the coordinate segment count.")
      source.append(self.__activitySource(suffix, max_num_indices))
      if not 1 <= max_num_indices <= 65536:
        raise ValueError("Separate Hessian: local segment indices exceed unsigned short int capacity.")
      permutation = self.__layout["column_permutation"]
      column_bytes = max(1, len(permutation)) * 2
      column_storage = "__constant__" if column_bytes <= 65536 else "const"
      constant_bytes = column_bytes if column_storage == "__constant__" else 0
      # Only result is restrict-qualified: it always points to the separate
      # multiplied_block buffer, never the H/J operands within hg_mat.
      if self.__auto_partition >= 2:
        source.extend(self.__segmentFunctionSource(suffix, num_attributes))
      if self.__auto_partition == 0:
        source.extend(self.__innerFunctionSource(suffix, constant_bytes))
      for pattern_id, pattern in enumerate(self.__patterns if self.__auto_partition == 1 else []):
        h_map_argument = ", const unsigned short int* h_indices" if pattern["mapped_h"] else ""
        source.append(f"static __device__ void multiply_sparse_pattern_{pattern_id}_{suffix}(const double* left_jac, const double* h, const double* right_jac{h_map_argument}, double* __restrict__ result) {{\n{pattern['source']}\n}}")
      for pair, (i, j, pattern_id) in enumerate(self.__block_patterns):
        pattern = self.__patterns[pattern_id]
        if pattern["mapped_h"]:
          values = self.__hessian_operands[pair]
          # Reserve the scatter's column table first. Large collections of H
          # maps fall back to static read-only device storage, rather than
          # exceeding CUDA's 64 KiB constant space.
          map_bytes = len(values) * 2
          if constant_bytes + map_bytes <= 65536:
            storage = "__constant__"
            constant_bytes += map_bytes
          else:
            storage = "const"
          source.append(f"static __device__ {storage} unsigned short int hessian_indices_{i}_{j}_{suffix}[{len(values)}] = {{{', '.join(map(str, values))}}};")
      # Symbolic column IDs; joins/unions resolve global DOFs at runtime. A
      # table larger than constant space uses the same read-only fallback.
      if self.__auto_partition < 2:
        source.append(f"static __device__ {column_storage} unsigned short int jacobian_columns_{suffix}[{max(1, len(permutation))}] = {{{', '.join(map(str, permutation)) or '0'}}};")
        source.append(self.__scatterFunction(suffix, num_attributes))
      if self.__auto_partition == 0 and self.__layout["rows"] > 1:
        source.append(self.__denseTileScatterFunction(suffix, num_attributes))
    gradient_start = 0 if self.__gradient_only else self.__merged_hessian_jacobian_nonzeros
    entry_source = f'''
__global__ void compute_hessian_and_gradient_global_function_final_gradient_size_{suffix}(
  {declarations}
  const unsigned int* segment_indices,
  const unsigned short int* segment_sizes,
  const short int* local_permutations,
  const unsigned int* lookups,
  const unsigned int* coordinatesOuter,
  const unsigned int* groupedIndicesInner,
  const unsigned int* groupedIndicesOuter,
  const unsigned char* block_activity,
  const unsigned int activity_stride,
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
'''
    if not self.__gradient_only and self.__block_patterns:
      jacobian_cols = self.__layout["cols"]
      max_block_entries = max(len(self.__layout['blocks'][i]['cols']) * len(self.__layout['blocks'][j]['cols']) for i, j, _ in self.__block_patterns) if self.__auto_partition == 1 else (max(self.__segment_sizes) ** 2 if self.__auto_partition >= 2 else self.__layout["rows"] ** 2)
      # Segment modes already know every local block boundary at compile time.
      # Only the rank among included targets still depends on runtime indices.
      mapping_declarations = "" if self.__auto_partition >= 2 else f"unsigned short int column_segment[{max(1, jacobian_cols)}];\n  unsigned short int segment_outer[{max_num_indices + 1}];\n  segment_outer[0] = 0;"
      mapping_update = "" if self.__auto_partition >= 2 else "segment_outer[i + 1] = segment_outer[i] + sizes[i];\n    for (unsigned int k = segment_outer[i]; k < segment_outer[i + 1]; ++k) column_segment[k] = i;"
      source.append(f'''
// Evaluation and assembly are sequential. Keep assembly scratch in its own
// call frame so it is not reserved during the nested H/J/gradient evaluation.
static __device__ __noinline__ void assemble_sparse_hessian_{suffix}(
  const double* hg_mat, const unsigned int* indices,
  const unsigned short int* sizes, const short int* permutations,
  const unsigned char* block_activity, const unsigned int activity_stride,
  const unsigned int* instance_lookups, double* hessian_blocks,
  double* diagonal_blocks, const unsigned int* diagonal_blocks_start,
  const unsigned int* gradient_segments_start
) {{
  {mapping_declarations}
  unsigned short int valid_rank[{max_num_indices}]; // Map original segments to valid coordinate ranks, skipping union padding.
  unsigned int valid_count = 0;
  for (unsigned int i = 0; i < {max_num_indices}; ++i) {{
    {mapping_update}
    valid_rank[i] = valid_count;
    if (permutations[i] > 0 && indices[i] >= 2) ++valid_count; // if this is an actual value that we need to place back into the Hessian, then increment the valid rank
  }}
  double multiplied_block[{max_block_entries}];
''')
      if self.__auto_partition >= 2:
        source.append(self.__segmentCallerSource(suffix, max_num_indices))
      elif self.__auto_partition == 0:
        source.append(self.__innerCallerSource(suffix, max_num_indices))
      else:
        spans_outer = [0]
        for block in self.__layout["blocks"]:
          spans_outer.append(spans_outer[-1] + len(block["cols"]))
        for pair, (i, j, pattern_id) in enumerate(self.__block_patterns):
          rows = len(self.__layout["blocks"][i]["cols"])
          cols = len(self.__layout["blocks"][j]["cols"])
          left_offset = self.__local_hessian_nonzero_count + self.__packed_jacobian["block_offsets"][i]
          right_offset = self.__local_hessian_nonzero_count + self.__packed_jacobian["block_offsets"][j]
          mapped_h = self.__patterns[pattern_id]["mapped_h"]
          h_argument = "hg_mat" if mapped_h else f"hg_mat + {self.__hessian_operands[pair]}"
          h_map_argument = f", hessian_indices_{i}_{j}_{suffix}" if mapped_h else ""
          source.append(f'''
  if (block_activity[(size_t){i} * activity_stride] && block_activity[(size_t){j} * activity_stride]) {{
  multiply_sparse_pattern_{pattern_id}_{suffix}(hg_mat + {left_offset}, {h_argument}, hg_mat + {right_offset}{h_map_argument}, multiplied_block);
  for (unsigned int a = 0; a < {rows}; ++a) {{
    for (unsigned int b = {'a' if i == j else '0'}; b < {cols}; ++b) {{
      const unsigned int original_a = jacobian_columns_{suffix}[{spans_outer[i]} + a]; // this is the row in the original Hessian
      const unsigned int original_b = jacobian_columns_{suffix}[{spans_outer[j]} + b]; // this is the col in the original Hessian
      scatter_sparse_hessian_{suffix}(multiplied_block[a * {cols} + b], original_a, original_b, column_segment, segment_outer, valid_rank, valid_count, indices, sizes, permutations, instance_lookups, hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);
    }}
  }}
  }}
''')
      source.append("}")
      entry_source += f'''
  assemble_sparse_hessian_{suffix}(hg_mat, indices, sizes, permutations, block_activity + instance, activity_stride, lookups + coordinatesOuter[instance], hessian_blocks, diagonal_blocks, diagonal_blocks_start, gradient_segments_start);
'''
    source.append(entry_source + "}\n}\n")
    self.__kernelString = "\n".join(source)
    return self.__kernelString

  def __denseTileScatterFunction(self, suffix, num_attributes):
    # Inner-Hessian tiles have contiguous original axes. Intersect each tile
    # with the runtime segments, and resolve one destination per segment pair
    # rather than repeating coordinate and diagonal lookup for every scalar.
    # Keep this loop out of the outer evaluator/multiply kernel's live state.
    size = self.__layout["rows"]
    return f'''
static __device__ __noinline__ void scatter_dense_tile_{suffix}(
  const double* values, unsigned int row_begin, unsigned int column_begin,
  unsigned int rows, unsigned int columns,
  const unsigned short int* column_segment, const unsigned short int* segment_outer,
  const unsigned short int* valid_rank, unsigned int valid_count,
  const unsigned int* indices, const unsigned short int* sizes,
  const short int* permutations, const unsigned int* lookups,
  double* hessian_blocks, double* diagonal_blocks,
  const unsigned int* diagonal_blocks_start, const unsigned int* gradient_segments_start
) {{
  const unsigned int row_end = row_begin + rows;
  const unsigned int column_end = column_begin + columns;
  const bool diagonal_tile = row_begin == column_begin;
  #pragma unroll 1
  for (unsigned int sa = column_segment[row_begin]; sa <= column_segment[row_end - 1]; ++sa) {{
    if (permutations[sa] <= 0 || indices[sa] < 2 || sizes[sa] == 0) continue;
    const unsigned int a_begin = max(row_begin, (unsigned int)segment_outer[sa]);
    const unsigned int a_end = min(row_end, (unsigned int)segment_outer[sa + 1]);
    const unsigned int start_a = indices[sa];
    const unsigned int rank_a = valid_rank[sa];
    const unsigned int size_a = sizes[sa];
    #pragma unroll 1
    for (unsigned int sb = column_segment[column_begin]; sb <= column_segment[column_end - 1]; ++sb) {{
      if (permutations[sb] <= 0 || indices[sb] < 2 || sizes[sb] == 0) continue;
      const unsigned int b_begin = max(column_begin, (unsigned int)segment_outer[sb]);
      const unsigned int b_end = min(column_end, (unsigned int)segment_outer[sb + 1]);
      if (diagonal_tile && b_end <= a_begin) continue;
      const unsigned int start_b = indices[sb];
      const unsigned int first = min(rank_a, (unsigned int)valid_rank[sb]);
      const unsigned int last = max(rank_a, (unsigned int)valid_rank[sb]);
      const unsigned int placement = lookups[first * valid_count - first * (first + 1) / 2 + last];
      const unsigned int row_stride = start_a <= start_b ? sizes[sb] : 1;
      const unsigned int column_stride = start_a <= start_b ? 1 : size_a;
      const bool same_target = start_a == start_b;
      unsigned int diagonal_start = 0;
      if (same_target) {{
        const unsigned int segment_start = start_a - 2;
        unsigned int which_attribute = 0;
        while (which_attribute + 1 < {num_attributes} && segment_start >= gradient_segments_start[which_attribute + 1]) ++which_attribute;
        const unsigned int local_instance = (segment_start - gradient_segments_start[which_attribute]) / size_a;
        diagonal_start = diagonal_blocks_start[which_attribute] + local_instance * size_a * size_a;
      }}
      #pragma unroll 1
      for (unsigned int a = a_begin; a < a_end; ++a) {{
        const unsigned int offset_a = a - segment_outer[sa];
        #pragma unroll 1
        for (unsigned int b = diagonal_tile ? max(b_begin, a) : b_begin; b < b_end; ++b) {{
          const unsigned int offset_b = b - segment_outer[sb];
          const double value = values[(a - row_begin) * {size} + b - column_begin];
          {self.__atomic_add}(&hessian_blocks[placement + offset_a * row_stride + offset_b * column_stride], value);
          if (same_target) {{
            {self.__atomic_add}(&diagonal_blocks[diagonal_start + offset_a * size_a + offset_b], value);
            // Different local occurrences can resolve to the same vertex.
            if (a != b) {{
              {self.__atomic_add}(&hessian_blocks[placement + offset_b * size_a + offset_a], value);
              {self.__atomic_add}(&diagonal_blocks[diagonal_start + offset_b * size_a + offset_a], value);
            }}
          }}
        }}
      }}
    }}
  }}
}}
'''

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
  const unsigned short int* segment_outer, // starting local scalar column of each segment
  const unsigned short int* valid_rank, unsigned int valid_count, // rank among valid segments; arithmetic stays uint
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
  def packedInfo(self):
    if self.__packed_jacobian is None:
      return {}
    if self.__auto_partition >= 2:
      temporary_entries = max(self.__segment_sizes, default=0) ** 2
      if self.__auto_partition == 2:
        temporary_entries += self.__layout["rows"] * self.__layout["cols"]
    else:
      temporary_entries = 2 * self.__layout["rows"] ** 2 if self.__auto_partition == 0 else max((len(self.__layout["blocks"][i]["cols"]) * len(self.__layout["blocks"][j]["cols"]) for i, j, _ in self.__block_patterns), default=0)
    return {
      "jacobian_nonzero_permutation": list(self.__packed_jacobian["nonzero_permutation"]),
      "block_offsets": list(self.__packed_jacobian["block_offsets"]),
      "block_counts": [len(positions) for positions in self.__packed_jacobian["block_local_positions"]],
      "block_pairs": len(self.__block_patterns),
      "multiplication_patterns": len(self.__patterns),
      "mapped_hessian_pairs": sum(isinstance(values, list) for values in self.__hessian_operands),
      "auto_partition": self.__auto_partition,
      "left_patterns": len(self.__left_patterns),
      "temporary_entries": temporary_entries,
    }

  @property
  def kernelString(self):
    return self.__kernelString

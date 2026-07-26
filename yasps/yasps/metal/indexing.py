"""Topology-specific generated Metal kernels for sparse gradient indices."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import mlx.core as mx

from yasps.backend import GPUArray
from yasps.metal.scan import outer_indices

import importlib

ya = importlib.import_module("yasps.attribute")


def _symbol(kind: str, name: str) -> str:
  return f"yasps_{kind}_{sha256(f'{kind}:{name}'.encode()).hexdigest()[:16]}"


def _array(value):
  return value._array if isinstance(value, GPUArray) else value


class MetalIndexPipeline:
  """Generated counterpart of CUDA ``gradientIndicesKernel``."""

  def __init__(
    self,
    path_dict,
    unioned_child_to_children,
    wrt,
    energy,
    gradient_sizes,
    index_sizes,
    max_gradient_size,
    max_indices,
    no_local_permutation,
  ):
    self.path = path_dict
    self.unioned_children = unioned_child_to_children
    self.wrt_positions = {item: index for index, item in enumerate(wrt)}
    self.energy = energy
    self.gradient_sizes = gradient_sizes
    self.index_sizes = index_sizes
    self.max_gradient_size = max_gradient_size
    self.max_indices = max_indices
    self.no_local_permutation = no_local_permutation
    self.joins = sorted(
      {
        item.hash: item
        for item in path_dict
        if item.operator == ya.JOIN
      }.values(),
      key=lambda item: item.fullName,
    )
    self.unions = sorted(
      {
        item.correspondance.fullName: item.correspondance
        for item in path_dict
        if item.operator == ya.UNION
      }.values(),
      key=lambda item: item.fullName,
    )
    self.input_names = (
      [_symbol("indices", item.through.fullName) for item in self.joins]
      + [_symbol("union", item.fullName) for item in self.unions]
      + ["wrt_starts", "instance_count"]
    )
    self.raw_source = self._generate_raw_source()
    digest = sha256(self.raw_source.encode()).hexdigest()[:16]
    self.raw_kernel = mx.fast.metal_kernel(
      name=f"yasps_raw_indices_{digest}",
      input_names=self.input_names,
      output_names=["output_indices", "output_sizes"],
      source=self.raw_source,
    )
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / f"yasps_raw_indices_{digest}.metal").write_text(
      self.raw_source
    )
    self.compression_kernel = self._compression_kernel()
    self.histogram_kernel = self._histogram_kernel()
    self.compact_sizes_kernel = self._compact_sizes_kernel()
    self.group_positions_kernel = self._group_positions_kernel()
    self.scatter_groups_kernel = self._scatter_groups_kernel()
    self.coordinate_kernel = self._coordinate_kernel()

  def _emit_leaf(self, attribute, index, base, lines, indent):
    if attribute in self.wrt_positions:
      position = self.wrt_positions[attribute]
      stride = attribute.size if attribute.correspondance.type == "primitive" else 0
      lines.append(
        f"{indent}output_indices[output_base + {base}] = "
        f"wrt_starts[{position}] + ({index}) * {stride} + 2u;"
      )
    else:
      lines.append(
        f"{indent}output_indices[output_base + {base}] = 1u;"
      )
    lines.append(
      f"{indent}output_sizes[output_base + {base}] = {attribute.size};"
    )

  def _emit_node(self, attribute, index, base, lines, indent):
    if attribute.operator == ya.DATA:
      self._emit_leaf(attribute, index, base, lines, indent)
      return
    if attribute.operator == ya.JOIN:
      connection = _symbol("indices", attribute.through.fullName)
      per_slot = self.index_sizes[attribute] // attribute.through.dimension
      lines.append(
        f"{indent}for (uint slot = 0; slot < "
        f"{attribute.through.dimension}; ++slot) {{"
      )
      lines.append(
        f"{indent}  const uint joined_index = "
        f"{connection}[({index}) * {attribute.through.dimension} + slot];"
      )
      offset = 0
      for child in self.path[attribute]:
        self._emit_node(
          child,
          "joined_index",
          f"({base}) + slot * {per_slot} + {offset}",
          lines,
          indent + "  ",
        )
        offset += self.index_sizes[child]
      lines.append(f"{indent}}}")
      return
    if attribute.operator == ya.UNION:
      counts = _symbol("union", attribute.correspondance.fullName)
      lines.append(f"{indent}uint union_offset = 0u;")
      for child_index, union_child in enumerate(attribute.children):
        lines.append(
          f"{indent}if (({index}) >= union_offset && ({index}) < union_offset "
          f"+ {counts}[{child_index}]) {{"
        )
        used = [
          candidate
          for candidate in self.path[attribute]
          if candidate in self.unioned_children[union_child]
        ]
        offset = 0
        actual_size = 0
        for candidate in used:
          self._emit_node(
            candidate,
            f"({index}) - union_offset",
            f"({base}) + {offset}",
            lines,
            indent + "  ",
          )
          offset += self.index_sizes[candidate]
          actual_size += self.gradient_sizes[candidate]
        lines.append(
          f"{indent}  output_sizes[output_base + ({base}) "
          f"+ {self.index_sizes[attribute] - 1}] = "
          f"{self.gradient_sizes[attribute] - actual_size};"
        )
        lines.append(f"{indent}}}")
        if child_index + 1 < len(attribute.children):
          lines.append(
            f"{indent}union_offset += {counts}[{child_index}];"
          )
      return

    children = self.path.get(attribute)
    if children is None:
      children = [
        child
        for child in self.unioned_children.get(attribute, [])
        if child in self.path or child in self.wrt_positions
      ]
    offset = 0
    for child in children:
      self._emit_node(
        child, index, f"({base}) + {offset}", lines, indent
      )
      offset += self.index_sizes[child]

  def _generate_raw_source(self):
    lines = [
      "const uint instance = thread_position_in_grid.x;",
      "if (instance >= instance_count[0]) return;",
      f"const uint output_base = instance * {self.max_indices};",
      f"for (uint i = 0; i < {self.max_indices}; ++i) {{",
      "  output_indices[output_base + i] = 0u;",
      "  output_sizes[output_base + i] = 0;",
      "}",
    ]
    self._emit_node(self.energy, "instance", "0", lines, "")
    return "\n".join(lines)

  def _compression_kernel(self):
    no_permutation = "true" if self.no_local_permutation else "false"
    source = f"""
      const uint instance = thread_position_in_grid.x;
      if (instance >= instance_count[0]) return;
      constexpr uint K = {self.max_indices};
      constexpr bool NO_PERMUTATION = {no_permutation};
      const uint base = instance * K;
      uint gradient_offset = 0u;
      ushort total_size = 0;
      uint compressed_count = 0u;
      for (uint i = 0; i < K; ++i) {{
        permutations[base + i] = 0;
        const uint current = indices[base + i];
        if (current == 0u) continue;
        bool found = false;
        if (!NO_PERMUTATION && current >= 2u) {{
          for (uint j = 0; j < i; ++j) {{
            if (indices[base + j] == current) {{
              permutations[base + i] = -permutations[base + j];
              found = true;
              break;
            }}
          }}
        }}
        if (!found) {{
          short permutation = short(gradient_offset + 1u);
          if (NO_PERMUTATION && current == 1u) permutation = -permutation;
          permutations[base + i] = permutation;
          gradient_offset += uint(index_sizes[base + i]);
          total_size += index_sizes[base + i];
          if (current >= 2u) ++compressed_count;
        }}
      }}
      gradient_sizes[instance] = total_size;
      coordinate_counts[instance] =
          compressed_count * (compressed_count + 1u) / 2u;
    """
    digest = sha256(source.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
      name=f"yasps_compress_local_indices_{digest}",
      input_names=["indices", "index_sizes", "instance_count"],
      output_names=[
        "permutations",
        "gradient_sizes",
        "coordinate_counts",
      ],
      source=source,
    )

  def _histogram_kernel(self):
    source = """
      const uint instance = thread_position_in_grid.x;
      if (instance >= instance_count[0]) return;
      atomic_fetch_add_explicit(
          &histogram[gradient_sizes[instance]], 1u, memory_order_relaxed);
    """
    return mx.fast.metal_kernel(
      name=f"yasps_gradient_size_histogram_{self.max_gradient_size}",
      input_names=["gradient_sizes", "instance_count"],
      output_names=["histogram"],
      source=source,
      atomic_outputs=True,
    )

  def _compact_sizes_kernel(self):
    source = f"""
      if (thread_position_in_grid.x != 0) return;
      uint unique_count = 0u;
      uint running_offset = 0u;
      for (uint size = 0; size <= {self.max_gradient_size}; ++size) {{
        size_offsets[size] = running_offset;
        const uint count = histogram[size];
        if (count != 0u) {{
          unique_sizes[unique_count] = ushort(size);
          grouped_outer[unique_count] = running_offset;
          ++unique_count;
          running_offset += count;
        }}
      }}
      grouped_outer[unique_count] = running_offset;
      num_unique[0] = ushort(unique_count);
    """
    return mx.fast.metal_kernel(
      name=f"yasps_compact_gradient_sizes_{self.max_gradient_size}",
      input_names=["histogram"],
      output_names=[
        "size_offsets",
        "unique_sizes",
        "grouped_outer",
        "num_unique",
      ],
      source=source,
    )

  def _group_positions_kernel(self):
    source = """
      const uint instance = thread_position_in_grid.x;
      if (instance >= instance_count[0]) return;
      const uint size = uint(gradient_sizes[instance]);
      const uint position = atomic_fetch_add_explicit(
          &counters[size], 1u, memory_order_relaxed);
      atomic_store_explicit(
          &positions[instance], position, memory_order_relaxed);
    """
    return mx.fast.metal_kernel(
      name=f"yasps_group_positions_{self.max_gradient_size}",
      input_names=["gradient_sizes", "instance_count"],
      output_names=["positions", "counters"],
      source=source,
      atomic_outputs=True,
    )

  def _scatter_groups_kernel(self):
    source = """
      const uint instance = thread_position_in_grid.x;
      if (instance >= instance_count[0]) return;
      const uint size = uint(gradient_sizes[instance]);
      grouped_indices[size_offsets[size] + positions[instance]] = instance;
    """
    return mx.fast.metal_kernel(
      name=f"yasps_scatter_gradient_groups_{self.max_gradient_size}",
      input_names=[
        "gradient_sizes",
        "size_offsets",
        "positions",
        "instance_count",
      ],
      output_names=["grouped_indices"],
      source=source,
    )

  def _coordinate_kernel(self):
    source = f"""
      const uint instance = thread_position_in_grid.x;
      if (instance >= instance_count[0]) return;
      constexpr uint K = {self.max_indices};
      const uint base = instance * K;
      uint coordinate = coordinate_outer[instance] * 2u;
      for (uint i = 0; i < K; ++i) {{
        const uint index_i = indices[base + i];
        if (permutations[base + i] <= 0 || index_i < 2u) continue;
        for (uint j = i; j < K; ++j) {{
          const uint index_j = indices[base + j];
          if (permutations[base + j] <= 0 || index_j < 2u) continue;
          const bool ordered = index_i < index_j;
          coordinates[coordinate] = (ordered ? index_i : index_j) - 2u;
          coordinates[coordinate + 1u] =
              (ordered ? index_j : index_i) - 2u;
          dimensions[coordinate] =
              ordered ? index_sizes[base + i] : index_sizes[base + j];
          dimensions[coordinate + 1u] =
              ordered ? index_sizes[base + j] : index_sizes[base + i];
          coordinate += 2u;
        }}
      }}
    """
    digest = sha256(source.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
      name=f"yasps_generate_coordinates_{digest}",
      input_names=[
        "indices",
        "permutations",
        "index_sizes",
        "coordinate_outer",
        "instance_count",
      ],
      output_names=["coordinates", "dimensions"],
      source=source,
    )

  def run(self, wrt_starts, count):
    if count == 0:
      return self._empty()
    count_array = mx.array([count], dtype=mx.uint32)
    inputs = [_array(item.through.value) for item in self.joins]
    union_counts = [item.children_primitive_counts_gpu for item in self.unions]
    inputs.extend(_array(item) for item in union_counts)
    inputs.extend([mx.array(wrt_starts, dtype=mx.uint32), count_array])
    raw_indices, index_sizes = self.raw_kernel(
      inputs=inputs,
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[
        (count * self.max_indices,),
        (count * self.max_indices,),
      ],
      output_dtypes=[mx.uint32, mx.uint16],
      init_value=0,
    )
    permutations, gradient_sizes, coordinate_counts = self.compression_kernel(
      inputs=[raw_indices, index_sizes, count_array],
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[
        (count * self.max_indices,),
        (count,),
        (count,),
      ],
      output_dtypes=[mx.int16, mx.uint16, mx.uint32],
      init_value=0,
    )
    histogram = self.histogram_kernel(
      inputs=[gradient_sizes, count_array],
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[(self.max_gradient_size + 1,)],
      output_dtypes=[mx.uint32],
      init_value=0,
    )[0]
    size_offsets, unique_sizes, grouped_outer, num_unique = (
      self.compact_sizes_kernel(
        inputs=[histogram],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[
          (self.max_gradient_size + 1,),
          (self.max_gradient_size + 1,),
          (self.max_gradient_size + 2,),
          (1,),
        ],
        output_dtypes=[mx.uint32, mx.uint16, mx.uint32, mx.uint16],
        init_value=0,
      )
    )
    positions, _ = self.group_positions_kernel(
      inputs=[gradient_sizes, count_array],
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[(count,), (self.max_gradient_size + 1,)],
      output_dtypes=[mx.uint32, mx.uint32],
      init_value=0,
    )
    grouped_indices = self.scatter_groups_kernel(
      inputs=[gradient_sizes, size_offsets, positions, count_array],
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[(count,)],
      output_dtypes=[mx.uint32],
      init_value=0,
    )[0]
    coordinate_outer = outer_indices(coordinate_counts)
    mx.eval(coordinate_outer)
    total_coordinates = int(coordinate_outer[-1].item())
    coordinates, dimensions = self.coordinate_kernel(
      inputs=[
        raw_indices,
        permutations,
        index_sizes,
        coordinate_outer,
        count_array,
      ],
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[
        (total_coordinates * 2,),
        (total_coordinates * 2,),
      ],
      output_dtypes=[mx.uint32, mx.uint16],
      init_value=0,
    )
    return {
      "indices": GPUArray._wrap(raw_indices),
      "sizes": GPUArray._wrap(index_sizes),
      "permutations": GPUArray._wrap(permutations),
      "gradient_sizes": GPUArray._wrap(gradient_sizes),
      "unique_sizes": GPUArray._wrap(unique_sizes),
      "grouped_inner": GPUArray._wrap(grouped_indices),
      "grouped_outer": GPUArray._wrap(grouped_outer),
      "num_unique": GPUArray._wrap(num_unique),
      "coordinate_outer": GPUArray._wrap(coordinate_outer),
      "coordinates": GPUArray._wrap(coordinates),
      "dimensions": GPUArray._wrap(dimensions),
      "total_coordinates": total_coordinates,
    }

  def _empty(self):
    return {
      "indices": GPUArray._wrap(mx.empty((0,), dtype=mx.uint32)),
      "sizes": GPUArray._wrap(mx.empty((0,), dtype=mx.uint16)),
      "permutations": GPUArray._wrap(mx.empty((0,), dtype=mx.int16)),
      "gradient_sizes": GPUArray._wrap(mx.empty((0,), dtype=mx.uint16)),
      "unique_sizes": GPUArray._wrap(mx.empty((0,), dtype=mx.uint16)),
      "grouped_inner": GPUArray._wrap(mx.empty((0,), dtype=mx.uint32)),
      "grouped_outer": GPUArray._wrap(mx.zeros((1,), dtype=mx.uint32)),
      "num_unique": GPUArray._wrap(mx.zeros((1,), dtype=mx.uint16)),
      "coordinate_outer": GPUArray._wrap(mx.zeros((1,), dtype=mx.uint32)),
      "coordinates": GPUArray._wrap(mx.empty((0,), dtype=mx.uint32)),
      "dimensions": GPUArray._wrap(mx.empty((0,), dtype=mx.uint16)),
      "total_coordinates": 0,
    }

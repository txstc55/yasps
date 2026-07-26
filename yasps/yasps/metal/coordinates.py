"""Generated Metal compression for sparse block coordinates."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import mlx.core as mx

from yasps.backend import GPUArray
from yasps.metal.scan import exclusive_scan, outer_indices


_THREADS = 256
_SENTINEL = 0xFFFFFFFF


def _array(value):
  return value._array if isinstance(value, GPUArray) else value


def _grid(count):
  return (max(1, count), 1, 1)


def _threadgroup(count):
  return (max(1, min(count, _THREADS)), 1, 1)


def _next_power_of_two(value):
  return 1 << (max(1, value) - 1).bit_length()


@lru_cache(maxsize=None)
def _pack_kernel(counts: tuple[int, ...]):
  input_names = []
  lines = [
    "const uint index = thread_position_in_grid.x;",
    f"if (index >= {sum(counts)}) return;",
  ]
  offset = 0
  for term, count in enumerate(counts):
    input_names.extend([f"coordinates_{term}", f"dimensions_{term}"])
    condition = "if" if term == 0 else "else if"
    lines.extend(
      [
        f"{condition} (index < {offset + count}u) {{",
        f"  const uint local = index - {offset}u;",
        f"  rows[index] = coordinates_{term}[local * 2u];",
        f"  columns[index] = coordinates_{term}[local * 2u + 1u];",
        f"  heights[index] = uint(dimensions_{term}[local * 2u]);",
        f"  widths[index] = uint(dimensions_{term}[local * 2u + 1u]);",
        "  original[index] = index;",
        "}",
      ]
    )
    offset += count
  source = "\n".join(lines)
  digest = sha256(source.encode()).hexdigest()[:16]
  return mx.fast.metal_kernel(
    name=f"yasps_pack_coordinates_{digest}",
    input_names=input_names,
    output_names=["rows", "columns", "heights", "widths", "original"],
    source=source,
  )


@lru_cache(maxsize=1)
def _pad_kernel():
  source = f"""
    const uint index = thread_position_in_grid.x;
    if (index >= padded_count[0]) return;
    if (index < count[0]) {{
      rows_out[index] = rows[index];
      columns_out[index] = columns[index];
      heights_out[index] = heights[index];
      widths_out[index] = widths[index];
      original_out[index] = original[index];
    }} else {{
      rows_out[index] = {_SENTINEL}u;
      columns_out[index] = {_SENTINEL}u;
      heights_out[index] = {_SENTINEL}u;
      widths_out[index] = {_SENTINEL}u;
      original_out[index] = {_SENTINEL}u;
    }}
  """
  return mx.fast.metal_kernel(
    name="yasps_pad_coordinate_records",
    input_names=[
      "rows",
      "columns",
      "heights",
      "widths",
      "original",
      "count",
      "padded_count",
    ],
    output_names=[
      "rows_out",
      "columns_out",
      "heights_out",
      "widths_out",
      "original_out",
    ],
    source=source,
  )


@lru_cache(maxsize=1)
def _bitonic_stage_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    const uint partner = index ^ distance[0];
    const bool ascending = (index & sequence[0]) == 0u;
    const bool lower_lane = index < partner;
    const bool take_less = ascending == lower_lane;

    bool left_less = false;
    if (heights[index] != heights[partner]) {
      left_less = heights[index] < heights[partner];
    } else if (widths[index] != widths[partner]) {
      left_less = widths[index] < widths[partner];
    } else if (rows[index] != rows[partner]) {
      left_less = rows[index] < rows[partner];
    } else if (columns[index] != columns[partner]) {
      left_less = columns[index] < columns[partner];
    } else {
      left_less = original[index] < original[partner];
    }
    const uint selected = (take_less == left_less) ? index : partner;
    rows_out[index] = rows[selected];
    columns_out[index] = columns[selected];
    heights_out[index] = heights[selected];
    widths_out[index] = widths[selected];
    original_out[index] = original[selected];
  """
  return mx.fast.metal_kernel(
    name="yasps_bitonic_coordinate_stage",
    input_names=[
      "rows",
      "columns",
      "heights",
      "widths",
      "original",
      "count",
      "distance",
      "sequence",
    ],
    output_names=[
      "rows_out",
      "columns_out",
      "heights_out",
      "widths_out",
      "original_out",
    ],
    source=source,
  )


@lru_cache(maxsize=1)
def _flag_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    const bool unique = index == 0u
        || heights[index] != heights[index - 1u]
        || widths[index] != widths[index - 1u]
        || rows[index] != rows[index - 1u]
        || columns[index] != columns[index - 1u];
    const bool new_dimension = unique && (
        index == 0u
        || heights[index] != heights[index - 1u]
        || widths[index] != widths[index - 1u]);
    unique_flags[index] = unique ? 1u : 0u;
    dimension_flags[index] = new_dimension ? 1u : 0u;
  """
  return mx.fast.metal_kernel(
    name="yasps_flag_unique_coordinates",
    input_names=["rows", "columns", "heights", "widths", "count"],
    output_names=["unique_flags", "dimension_flags"],
    source=source,
  )


@lru_cache(maxsize=1)
def _scatter_metadata_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    const uint unique_index = unique_prefix[index]
        - (unique_flags[index] == 0u ? 1u : 0u);
    const uint dimension_index = dimension_prefix[index]
        - (dimension_flags[index] == 0u ? 1u : 0u);
    original_to_unique[original[index]] = unique_index;
    if (unique_flags[index] != 0u) {
      unique_coordinates[unique_index * 2u] = rows[index];
      unique_coordinates[unique_index * 2u + 1u] = columns[index];
      unique_to_dimension[unique_index] = dimension_index;
    }
    if (dimension_flags[index] != 0u) {
      unique_dimensions[dimension_index * 2u] = ushort(heights[index]);
      unique_dimensions[dimension_index * 2u + 1u] = ushort(widths[index]);
      dimension_block_starts[dimension_index] = unique_index;
    }
  """
  return mx.fast.metal_kernel(
    name="yasps_scatter_coordinate_metadata",
    input_names=[
      "rows",
      "columns",
      "heights",
      "widths",
      "original",
      "unique_flags",
      "dimension_flags",
      "unique_prefix",
      "dimension_prefix",
      "count",
    ],
    output_names=[
      "unique_coordinates",
      "unique_dimensions",
      "unique_to_dimension",
      "dimension_block_starts",
      "original_to_unique",
    ],
    source=source,
  )


@lru_cache(maxsize=1)
def _finish_dimension_starts_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index > num_dimensions[0]) return;
    finished[index] = index == num_dimensions[0]
        ? num_unique[0]
        : dimension_block_starts[index];
  """
  return mx.fast.metal_kernel(
    name="yasps_finish_dimension_block_starts",
    input_names=[
      "dimension_block_starts",
      "num_dimensions",
      "num_unique",
    ],
    output_names=["finished"],
    source=source,
  )


@lru_cache(maxsize=1)
def _dimension_sizes_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= num_dimensions[0]) return;
    const uint count =
        dimension_block_starts[index + 1u] - dimension_block_starts[index];
    block_counts[index] = count;
    block_storage_sizes[index] = count
        * uint(unique_dimensions[index * 2u])
        * uint(unique_dimensions[index * 2u + 1u]);
  """
  return mx.fast.metal_kernel(
    name="yasps_coordinate_dimension_sizes",
    input_names=[
      "dimension_block_starts",
      "unique_dimensions",
      "num_dimensions",
    ],
    output_names=["block_counts", "block_storage_sizes"],
    source=source,
  )


@lru_cache(maxsize=1)
def _lookup_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    const uint unique_index = original_to_unique[index];
    const uint dimension_index = unique_to_dimension[unique_index];
    const uint local =
        unique_index - dimension_block_starts[dimension_index];
    lookup[index] = dimension_offsets[dimension_index] + local
        * uint(unique_dimensions[dimension_index * 2u])
        * uint(unique_dimensions[dimension_index * 2u + 1u]);
  """
  return mx.fast.metal_kernel(
    name="yasps_coordinate_storage_lookup",
    input_names=[
      "original_to_unique",
      "unique_to_dimension",
      "dimension_block_starts",
      "unique_dimensions",
      "dimension_offsets",
      "count",
    ],
    output_names=["lookup"],
    source=source,
  )


class MetalCoordinateCompressor:
  """Sort, deduplicate, and group sparse coordinates entirely on Metal."""

  def run(self, coordinates, dimensions, counts):
    counts = tuple(int(count) for count in counts if count > 0)
    count = sum(counts)
    if count == 0:
      return self._empty()

    inputs = []
    for coordinate, dimension in zip(coordinates, dimensions):
      inputs.extend([_array(coordinate), _array(dimension)])
    records = _pack_kernel(counts)(
      inputs=inputs,
      grid=_grid(count),
      threadgroup=_threadgroup(count),
      output_shapes=[(count,)] * 5,
      output_dtypes=[mx.uint32] * 5,
    )

    padded = _next_power_of_two(count)
    count_array = mx.array([count], dtype=mx.uint32)
    padded_array = mx.array([padded], dtype=mx.uint32)
    if padded != count:
      records = _pad_kernel()(
        inputs=[*records, count_array, padded_array],
        grid=_grid(padded),
        threadgroup=_threadgroup(padded),
        output_shapes=[(padded,)] * 5,
        output_dtypes=[mx.uint32] * 5,
      )

    sequence = 2
    while sequence <= padded:
      distance = sequence // 2
      while distance:
        records = _bitonic_stage_kernel()(
          inputs=[
            *records,
            padded_array,
            mx.array([distance], dtype=mx.uint32),
            mx.array([sequence], dtype=mx.uint32),
          ],
          grid=_grid(padded),
          threadgroup=_threadgroup(padded),
          output_shapes=[(padded,)] * 5,
          output_dtypes=[mx.uint32] * 5,
        )
        distance //= 2
      # Bound the lazy graph and release obsolete stage buffers per merge level.
      mx.eval(*records)
      sequence *= 2

    rows, columns, heights, widths, original = records
    unique_flags, dimension_flags = _flag_kernel()(
      inputs=[rows, columns, heights, widths, count_array],
      grid=_grid(count),
      threadgroup=_threadgroup(count),
      output_shapes=[(count,), (count,)],
      output_dtypes=[mx.uint32, mx.uint32],
    )
    unique_prefix = exclusive_scan(unique_flags)
    dimension_prefix = exclusive_scan(dimension_flags)
    mx.eval(unique_flags, dimension_flags, unique_prefix, dimension_prefix)
    num_unique = int(
      (unique_prefix[count - 1] + unique_flags[count - 1]).item()
    )
    num_dimensions = int(
      (dimension_prefix[count - 1] + dimension_flags[count - 1]).item()
    )
    num_unique_array = mx.array([num_unique], dtype=mx.uint32)
    num_dimensions_array = mx.array([num_dimensions], dtype=mx.uint32)

    (
      unique_coordinates,
      unique_dimensions,
      unique_to_dimension,
      dimension_block_starts,
      original_to_unique,
    ) = _scatter_metadata_kernel()(
      inputs=[
        rows,
        columns,
        heights,
        widths,
        original,
        unique_flags,
        dimension_flags,
        unique_prefix,
        dimension_prefix,
        count_array,
      ],
      grid=_grid(count),
      threadgroup=_threadgroup(count),
      output_shapes=[
        (num_unique * 2,),
        (num_dimensions * 2,),
        (num_unique,),
        (num_dimensions + 1,),
        (count,),
      ],
      output_dtypes=[
        mx.uint32,
        mx.uint16,
        mx.uint32,
        mx.uint32,
        mx.uint32,
      ],
      init_value=0,
    )
    dimension_block_starts = _finish_dimension_starts_kernel()(
      inputs=[
        dimension_block_starts,
        num_dimensions_array,
        num_unique_array,
      ],
      grid=_grid(num_dimensions + 1),
      threadgroup=_threadgroup(num_dimensions + 1),
      output_shapes=[(num_dimensions + 1,)],
      output_dtypes=[mx.uint32],
      init_value=0,
    )[0]
    block_counts, block_storage_sizes = _dimension_sizes_kernel()(
      inputs=[
        dimension_block_starts,
        unique_dimensions,
        num_dimensions_array,
      ],
      grid=_grid(num_dimensions),
      threadgroup=_threadgroup(num_dimensions),
      output_shapes=[(num_dimensions,), (num_dimensions,)],
      output_dtypes=[mx.uint32, mx.uint32],
    )
    dimension_offsets = outer_indices(block_storage_sizes)
    lookup = _lookup_kernel()(
      inputs=[
        original_to_unique,
        unique_to_dimension,
        dimension_block_starts,
        unique_dimensions,
        dimension_offsets,
        count_array,
      ],
      grid=_grid(count),
      threadgroup=_threadgroup(count),
      output_shapes=[(count,)],
      output_dtypes=[mx.uint32],
    )[0]
    mx.eval(
      unique_coordinates,
      unique_dimensions,
      block_counts,
      dimension_offsets,
      lookup,
    )
    return {
      "unique_coordinates": GPUArray._wrap(unique_coordinates),
      "unique_dimensions": GPUArray._wrap(unique_dimensions),
      "dimension_offsets": GPUArray._wrap(dimension_offsets),
      "block_counts": GPUArray._wrap(block_counts),
      "lookup": GPUArray._wrap(lookup),
      "num_unique": num_unique,
      "num_dimensions": num_dimensions,
      "total_block_size": int(dimension_offsets[num_dimensions].item()),
    }

  @staticmethod
  def _empty():
    return {
      "unique_coordinates": GPUArray._wrap(
        mx.empty((0,), dtype=mx.uint32)
      ),
      "unique_dimensions": GPUArray._wrap(
        mx.empty((0,), dtype=mx.uint16)
      ),
      "dimension_offsets": GPUArray._wrap(
        mx.zeros((1,), dtype=mx.uint32)
      ),
      "block_counts": GPUArray._wrap(
        mx.empty((0,), dtype=mx.uint32)
      ),
      "lookup": GPUArray._wrap(mx.empty((0,), dtype=mx.uint32)),
      "num_unique": 0,
      "num_dimensions": 0,
      "total_block_size": 0,
    }


class MetalPlacementReorder:
  """Group separate-Jacobian lookup entries by large Hessian block."""

  def __init__(self, spans, max_num_indices):
    self.spans = tuple(int(span) for span in spans)
    self.max_num_indices = int(max_num_indices)
    outer = [0]
    for span in self.spans:
      outer.append(outer[-1] + span)
    span_count = len(self.spans)
    large_blocks = span_count * (span_count + 1) // 2
    source = f"""
      const uint instance = thread_position_in_grid.x;
      if (instance >= instance_count[0]) return;
      constexpr ushort K = {self.max_num_indices};
      constexpr ushort L = {span_count};
      constexpr ushort B = {large_blocks};
      ushort span_outer[L + 1] = {{{", ".join(f"{value}u" for value in outer)}}};
      ushort block_counts[B];
      ushort block_outer[B + 1];
      ushort block_added[B];
      for (ushort block = 0; block < B; ++block) {{
        block_counts[block] = 0;
        block_added[block] = 0;
      }}
      block_outer[0] = 0;

      uint current_row = 0u;
      for (ushort i = 0; i < K; ++i) {{
        current_row += uint(segment_sizes[instance * K + i]);
        if (local_permutations[instance * K + i] <= 0) continue;
        ushort large_row = 0;
        while (large_row + 1u < L
            && current_row > uint(span_outer[large_row + 1u])) {{
          ++large_row;
        }}
        uint current_column = current_row - uint(segment_sizes[instance * K + i]);
        for (ushort j = i; j < K; ++j) {{
          current_column += uint(segment_sizes[instance * K + j]);
          if (local_permutations[instance * K + j] <= 0) continue;
          ushort large_column = 0;
          while (large_column + 1u < L
              && current_column > uint(span_outer[large_column + 1u])) {{
            ++large_column;
          }}
          const ushort upper = large_row * L
              - (large_row * (large_row - 1u)) / 2u
              + (large_column - large_row);
          ++block_counts[upper];
        }}
      }}
      for (ushort block = 0; block < B; ++block) {{
        block_outer[block + 1u] =
            block_outer[block] + block_counts[block];
      }}

      const uint lookup_start = coordinate_outer[instance];
      current_row = 0u;
      ushort original_block = 0;
      for (ushort i = 0; i < K; ++i) {{
        current_row += uint(segment_sizes[instance * K + i]);
        if (local_permutations[instance * K + i] <= 0) continue;
        ushort large_row = 0;
        while (large_row + 1u < L
            && current_row > uint(span_outer[large_row + 1u])) {{
          ++large_row;
        }}
        uint current_column = current_row - uint(segment_sizes[instance * K + i]);
        for (ushort j = i; j < K; ++j) {{
          current_column += uint(segment_sizes[instance * K + j]);
          if (local_permutations[instance * K + j] <= 0) continue;
          ushort large_column = 0;
          while (large_column + 1u < L
              && current_column > uint(span_outer[large_column + 1u])) {{
            ++large_column;
          }}
          const ushort upper = large_row * L
              - (large_row * (large_row - 1u)) / 2u
              + (large_column - large_row);
          const uint destination = lookup_start + block_outer[upper]
              + block_added[upper];
          reordered[destination] = lookups[lookup_start + original_block];
          ++block_added[upper];
          ++original_block;
        }}
      }}
    """
    digest = sha256(source.encode()).hexdigest()[:16]
    self.source = source
    self.kernel = mx.fast.metal_kernel(
      name=f"yasps_reorder_placements_{digest}",
      input_names=[
        "segment_sizes",
        "local_permutations",
        "lookups",
        "coordinate_outer",
        "instance_count",
      ],
      output_names=["reordered"],
      source=source,
    )
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / f"yasps_reorder_placements_{digest}.metal").write_text(
      source
    )

  def run(self, gi_kernel, lookups, instance_count):
    if instance_count == 0 or lookups.size == 0:
      return GPUArray._wrap(mx.empty((0,), dtype=mx.uint32))
    reordered = self.kernel(
      inputs=[
        _array(gi_kernel.outputSizes),
        _array(gi_kernel.outputPermutations),
        _array(lookups),
        _array(gi_kernel.outputCompressedCoordinateCountsOuter),
        mx.array([instance_count], dtype=mx.uint32),
      ],
      grid=_grid(instance_count),
      threadgroup=_threadgroup(instance_count),
      output_shapes=[lookups.shape],
      output_dtypes=[mx.uint32],
      init_value=0,
    )[0]
    return GPUArray._wrap(reordered)

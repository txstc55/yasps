"""Metal prefix scans shared by the coordinate kernels in this directory."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_BLOCK_SIZE = 256
_ELEMENTS_PER_BLOCK = _BLOCK_SIZE * 2


@lru_cache(maxsize=1)
def _block_scan_kernel():
  source = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint base = group * {_ELEMENTS_PER_BLOCK};
    const uint first = base + tid;
    const uint second = first + {_BLOCK_SIZE};
    threadgroup uint scratch[{_ELEMENTS_PER_BLOCK}];
    scratch[tid] = first < count[0] ? values[first] : 0u;
    scratch[tid + {_BLOCK_SIZE}] =
        second < count[0] ? values[second] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < {_ELEMENTS_PER_BLOCK}; offset <<= 1) {{
      const uint position = (tid + 1) * offset * 2 - 1;
      if (position < {_ELEMENTS_PER_BLOCK}) {{
        scratch[position] += scratch[position - offset];
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0) {{
      block_sums[group] = scratch[{_ELEMENTS_PER_BLOCK - 1}];
      scratch[{_ELEMENTS_PER_BLOCK - 1}] = 0u;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = {_ELEMENTS_PER_BLOCK // 2}; offset > 0; offset >>= 1) {{
      const uint position = (tid + 1) * offset * 2 - 1;
      if (position < {_ELEMENTS_PER_BLOCK}) {{
        const uint temporary = scratch[position - offset];
        scratch[position - offset] = scratch[position];
        scratch[position] += temporary;
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (first < count[0]) scanned[first] = scratch[tid];
    if (second < count[0]) scanned[second] = scratch[tid + {_BLOCK_SIZE}];
  """
  return mx.fast.metal_kernel(
    name="yasps_exclusive_scan_blocks_u32",
    input_names=["values", "count"],
    output_names=["scanned", "block_sums"],
    source=source,
  )


@lru_cache(maxsize=1)
def _add_block_offsets_kernel():
  source = f"""
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    output[index] = scanned[index]
        + block_offsets[index / {_ELEMENTS_PER_BLOCK}];
  """
  return mx.fast.metal_kernel(
    name="yasps_add_scan_block_offsets_u32",
    input_names=["scanned", "block_offsets", "count"],
    output_names=["output"],
    source=source,
  )


@lru_cache(maxsize=1)
def _outer_indices_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    const uint n = count[0];
    if (index > n) return;
    if (index < n) {
      outer[index] = scanned[index];
    } else {
      outer[index] = n == 0 ? 0u : scanned[n - 1] + values[n - 1];
    }
  """
  return mx.fast.metal_kernel(
    name="yasps_scan_to_outer_indices_u32",
    input_names=["values", "scanned", "count"],
    output_names=["outer"],
    source=source,
  )


def exclusive_scan(values: mx.array) -> mx.array:
  """Return an exclusive uint32 scan without leaving Metal."""

  count = values.size
  if count == 0:
    return mx.empty((0,), dtype=mx.uint32)
  groups = (count + _ELEMENTS_PER_BLOCK - 1) // _ELEMENTS_PER_BLOCK
  count_array = mx.array([count], dtype=mx.uint32)
  scanned, block_sums = _block_scan_kernel()(
    inputs=[values, count_array],
    grid=(groups * _BLOCK_SIZE, 1, 1),
    threadgroup=(_BLOCK_SIZE, 1, 1),
    output_shapes=[(count,), (groups,)],
    output_dtypes=[mx.uint32, mx.uint32],
  )
  if groups == 1:
    return scanned
  block_offsets = exclusive_scan(block_sums)
  return _add_block_offsets_kernel()(
    inputs=[scanned, block_offsets, count_array],
    grid=(count, 1, 1),
    threadgroup=(min(count, _BLOCK_SIZE), 1, 1),
    output_shapes=[(count,)],
    output_dtypes=[mx.uint32],
  )[0]


def outer_indices(values: mx.array) -> mx.array:
  """Return ``[exclusive_scan(values), sum(values)]`` on Metal."""

  count = values.size
  scanned = exclusive_scan(values)
  return _outer_indices_kernel()(
    inputs=[values, scanned, mx.array([count], dtype=mx.uint32)],
    grid=(count + 1, 1, 1),
    threadgroup=(min(count + 1, _BLOCK_SIZE), 1, 1),
    output_shapes=[(count + 1,)],
    output_dtypes=[mx.uint32],
  )[0]

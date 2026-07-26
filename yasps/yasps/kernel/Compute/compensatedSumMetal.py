"""Accurate float32 reductions executed by generated Metal kernels."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_THREADS = 256
_WIDTH = 2 * _THREADS


_DOUBLE_SINGLE_HEADER = """
inline void yasps_ds_add(thread float& hi, thread float& lo, float value) {
  const float sum = hi + value;
  const float virtual_value = sum - hi;
  const float error =
      (hi - (sum - virtual_value)) + (value - virtual_value);
  const float corrected = lo + error;
  const float new_hi = sum + corrected;
  lo = corrected - (new_hi - sum);
  hi = new_hi;
}

inline void yasps_ds_add_pair(
    thread float& hi,
    thread float& lo,
    float added_hi,
    float added_lo) {
  yasps_ds_add(hi, lo, added_hi);
  yasps_ds_add(hi, lo, added_lo);
}
"""


def _reduction_body(load_second_component):
  second_load_first = (
    "yasps_ds_add(hi, lo, input_lo[first]);"
    if load_second_component
    else ""
  )
  second_load_second = (
    "yasps_ds_add(hi, lo, input_lo[second]);"
    if load_second_component
    else ""
  )
  return f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint first = group * {_WIDTH}u + tid;
    const uint second = first + {_THREADS}u;
    threadgroup float scratch_hi[{_THREADS}];
    threadgroup float scratch_lo[{_THREADS}];

    float hi = 0.0f;
    float lo = 0.0f;
    if (first < count[0]) {{
      yasps_ds_add(hi, lo, input_hi[first]);
      {second_load_first}
    }}
    if (second < count[0]) {{
      yasps_ds_add(hi, lo, input_hi[second]);
      {second_load_second}
    }}
    scratch_hi[tid] = hi;
    scratch_lo[tid] = lo;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = {_THREADS // 2}u; offset > 0u; offset >>= 1u) {{
      if (tid < offset) {{
        hi = scratch_hi[tid];
        lo = scratch_lo[tid];
        yasps_ds_add_pair(
            hi,
            lo,
            scratch_hi[tid + offset],
            scratch_lo[tid + offset]);
        scratch_hi[tid] = hi;
        scratch_lo[tid] = lo;
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
      output_hi[group] = scratch_hi[0];
      output_lo[group] = scratch_lo[0];
    }}
  """


@lru_cache(maxsize=1)
def _first_kernel():
  return mx.fast.metal_kernel(
    name="yasps_compensated_sum_first",
    input_names=["input_hi", "count"],
    output_names=["output_hi", "output_lo"],
    header=_DOUBLE_SINGLE_HEADER,
    source=_reduction_body(False),
  )


@lru_cache(maxsize=1)
def _pair_kernel():
  return mx.fast.metal_kernel(
    name="yasps_compensated_sum_pair",
    input_names=["input_hi", "input_lo", "count"],
    output_names=["output_hi", "output_lo"],
    header=_DOUBLE_SINGLE_HEADER,
    source=_reduction_body(True),
  )


@lru_cache(maxsize=1)
def _dot_kernel():
  source = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint first = group * {_WIDTH}u + tid;
    const uint second = first + {_THREADS}u;
    threadgroup float scratch_hi[{_THREADS}];
    threadgroup float scratch_lo[{_THREADS}];

    float hi = 0.0f;
    float lo = 0.0f;
    if (first < count[0]) {{
      yasps_ds_add(hi, lo, left[first] * right[first]);
    }}
    if (second < count[0]) {{
      yasps_ds_add(hi, lo, left[second] * right[second]);
    }}
    scratch_hi[tid] = hi;
    scratch_lo[tid] = lo;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = {_THREADS // 2}u; offset > 0u; offset >>= 1u) {{
      if (tid < offset) {{
        hi = scratch_hi[tid];
        lo = scratch_lo[tid];
        yasps_ds_add_pair(
            hi,
            lo,
            scratch_hi[tid + offset],
            scratch_lo[tid + offset]);
        scratch_hi[tid] = hi;
        scratch_lo[tid] = lo;
      }}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (tid == 0u) {{
      output_hi[group] = scratch_hi[0];
      output_lo[group] = scratch_lo[0];
    }}
  """
  return mx.fast.metal_kernel(
    name="yasps_compensated_dot_first",
    input_names=["left", "right", "count"],
    output_names=["output_hi", "output_lo"],
    header=_DOUBLE_SINGLE_HEADER,
    source=source,
  )


def _finish_pairs(hi, lo) -> float:
  while hi.size > 1:
    groups = (hi.size + _WIDTH - 1) // _WIDTH
    hi, lo = _pair_kernel()(
      inputs=[hi, lo, mx.array([hi.size], dtype=mx.uint32)],
      grid=(groups * _THREADS, 1, 1),
      threadgroup=(_THREADS, 1, 1),
      output_shapes=[(groups,), (groups,)],
      output_dtypes=[mx.float32, mx.float32],
    )
  return float(hi[0].item()) + float(lo[0].item())


def compensated_sum(values) -> float:
  """Return a host scalar after doing all arithmetic reduction on Metal."""

  values = values.reshape((-1,))
  if values.size == 0:
    return 0.0

  groups = (values.size + _WIDTH - 1) // _WIDTH
  hi, lo = _first_kernel()(
    inputs=[values, mx.array([values.size], dtype=mx.uint32)],
    grid=(groups * _THREADS, 1, 1),
    threadgroup=(_THREADS, 1, 1),
    output_shapes=[(groups,), (groups,)],
    output_dtypes=[mx.float32, mx.float32],
  )
  return _finish_pairs(hi, lo)


def compensated_dot(left, right) -> float:
  """Return an accurate float32 dot product reduced entirely on Metal."""

  left = left.reshape((-1,))
  right = right.reshape((-1,))
  if left.size != right.size:
    raise ValueError("compensated_dot inputs must have the same size")
  if left.size == 0:
    return 0.0
  groups = (left.size + _WIDTH - 1) // _WIDTH
  hi, lo = _dot_kernel()(
    inputs=[
      left,
      right,
      mx.array([left.size], dtype=mx.uint32),
    ],
    grid=(groups * _THREADS, 1, 1),
    threadgroup=(_THREADS, 1, 1),
    output_shapes=[(groups,), (groups,)],
    output_dtypes=[mx.float32, mx.float32],
  )
  return _finish_pairs(hi, lo)

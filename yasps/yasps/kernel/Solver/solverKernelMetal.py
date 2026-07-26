"""Generated Metal counterpart to ``solverKernel.pyx``.

The Python layer specializes and persists one Metal library per block layout.
The compiled MLX extension owns the complete PCG host loop and dispatches these
kernels directly against MLX-allocated Metal buffers.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

import mlx.core as mx

from yasps import _metal_solver_ext


_COMMON_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

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

kernel void yasps_solver_clear(
    device float* output [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = 0.0f;
}

kernel void yasps_solver_copy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = input[index];
}

kernel void yasps_solver_combine(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) output[index] = left[index] + scale * right[index];
}

kernel void yasps_solver_dot_first(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output_hi [[buffer(2)]],
    device float* output_lo [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 256u;
  const uint first = group * (2u * threads) + tid;
  const uint second = first + threads;
  threadgroup float scratch_hi[threads];
  threadgroup float scratch_lo[threads];

  float hi = 0.0f;
  float lo = 0.0f;
  if (first < count) yasps_ds_add(hi, lo, left[first] * right[first]);
  if (second < count) yasps_ds_add(hi, lo, left[second] * right[second]);
  scratch_hi[tid] = hi;
  scratch_lo[tid] = lo;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = threads / 2u; offset > 0u; offset >>= 1u) {
    if (tid < offset) {
      hi = scratch_hi[tid];
      lo = scratch_lo[tid];
      yasps_ds_add_pair(
          hi,
          lo,
          scratch_hi[tid + offset],
          scratch_lo[tid + offset]);
      scratch_hi[tid] = hi;
      scratch_lo[tid] = lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    output_hi[group] = scratch_hi[0];
    output_lo[group] = scratch_lo[0];
  }
}

kernel void yasps_solver_reduce_pairs(
    device const float* input_hi [[buffer(0)]],
    device const float* input_lo [[buffer(1)]],
    device float* output_hi [[buffer(2)]],
    device float* output_lo [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 256u;
  const uint first = group * (2u * threads) + tid;
  const uint second = first + threads;
  threadgroup float scratch_hi[threads];
  threadgroup float scratch_lo[threads];

  float hi = 0.0f;
  float lo = 0.0f;
  if (first < count) {
    yasps_ds_add_pair(hi, lo, input_hi[first], input_lo[first]);
  }
  if (second < count) {
    yasps_ds_add_pair(hi, lo, input_hi[second], input_lo[second]);
  }
  scratch_hi[tid] = hi;
  scratch_lo[tid] = lo;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = threads / 2u; offset > 0u; offset >>= 1u) {
    if (tid < offset) {
      hi = scratch_hi[tid];
      lo = scratch_lo[tid];
      yasps_ds_add_pair(
          hi,
          lo,
          scratch_hi[tid + offset],
          scratch_lo[tid + offset]);
      scratch_hi[tid] = hi;
      scratch_lo[tid] = lo;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0u) {
    output_hi[group] = scratch_hi[0];
    output_lo[group] = scratch_lo[0];
  }
}
"""


def _spmv_source(rows: int, columns: int) -> str:
  return f"""
kernel void yasps_solver_spmv_{rows}x{columns}(
    device const float* block_values [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device atomic<float>* result [[buffer(3)]],
    constant uint& value_start [[buffer(4)]],
    constant uint& position_start [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint block [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {{
  threadgroup float forward[{rows * 32}];
  threadgroup uint row_keys[32];
  for (ushort row = 0; row < {rows}; ++row) {{
    forward[tid * {rows}u + row] = 0.0f;
  }}

  bool valid = block < count;
  uint output_row = 0xffffffffu;
  uint column = 0xffffffffu;
  if (valid) {{
    const uint position = position_start + block;
    output_row = positions[2u * position];
    column = positions[2u * position + 1u];
    const uint block_start = value_start + block * {rows * columns}u;
    for (ushort row = 0; row < {rows}; ++row) {{
      float value = 0.0f;
      for (ushort component = 0; component < {columns}; ++component) {{
        value += block_values[
            block_start + row * {columns}u + component]
            * x[column + component];
      }}
      forward[tid * {rows}u + row] = value;
    }}
    if (output_row != column) {{
      for (ushort component = 0; component < {columns}; ++component) {{
        float value = 0.0f;
        for (ushort row = 0; row < {rows}; ++row) {{
          value += block_values[
              block_start + row * {columns}u + component]
              * x[output_row + row];
        }}
        atomic_fetch_add_explicit(
            &result[column + component], value, memory_order_relaxed);
      }}
    }}
  }}
  row_keys[tid] = output_row;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (valid && (tid == 0u || row_keys[tid - 1u] != output_row)) {{
    for (ushort component = 0; component < {rows}; ++component) {{
      float value = 0.0f;
      for (uint lane = tid; lane < 32u; ++lane) {{
        if (row_keys[lane] != output_row) break;
        value += forward[lane * {rows}u + component];
      }}
      atomic_fetch_add_explicit(
          &result[output_row + component], value, memory_order_relaxed);
    }}
  }}
}}
"""


def _preconditioner_source(size: int) -> str:
  return f"""
kernel void yasps_solver_precondition_{size}(
    device const float* inverse_blocks [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& inverse_start [[buffer(3)]],
    constant uint& vector_start [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint block [[thread_position_in_grid]]) {{
  if (block >= count) return;
  const uint matrix = inverse_start + block * {size * size}u;
  const uint vector = vector_start + block * {size}u;
  for (ushort row = 0; row < {size}; ++row) {{
    float value = 0.0f;
    for (ushort column = 0; column < {size}; ++column) {{
      value += inverse_blocks[matrix + row * {size}u + column]
          * input[vector + column];
    }}
    output[vector + row] = value;
  }}
}}
"""


def _pairs(dimensions):
  return zip(dimensions[::2], dimensions[1::2])


class MetalCGSolver:
  """Specialize Metal kernels and call the compiled C++ PCG driver."""

  def __init__(self):
    self._programs = {}

  def _program(self, static_dimensions, dynamic_dimensions, diagonal_sizes):
    block_dimensions = sorted(
      {
        (int(rows), int(columns))
        for rows, columns in (
          list(_pairs(static_dimensions))
          + list(_pairs(dynamic_dimensions))
        )
      }
    )
    block_sizes = sorted({int(size) for size in diagonal_sizes})
    key = (tuple(block_dimensions), tuple(block_sizes))
    cached = self._programs.get(key)
    if cached is not None:
      return cached

    modules = [_COMMON_SOURCE]
    modules.extend(_spmv_source(*dimension) for dimension in block_dimensions)
    modules.extend(_preconditioner_source(size) for size in block_sizes)
    source = "\n".join(modules)
    digest = sha256(source.encode()).hexdigest()[:20]
    name = f"yasps_solver_{digest}"
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    source_path = output_directory / f"{name}.metal"
    if not source_path.exists() or source_path.read_text() != source:
      source_path.write_text(source)
    self._programs[key] = (source, name)
    return source, name

  def solve(
    self,
    max_iterations,
    threshold,
    block_values,
    block_positions,
    block_starts,
    block_counts,
    block_dimensions,
    dynamic_values,
    dynamic_positions,
    dynamic_starts,
    dynamic_counts,
    dynamic_dimensions,
    inverse_blocks,
    diagonal_starts,
    diagonal_counts,
    diagonal_sizes,
    gradient_starts,
    gradient,
    p1_b,
    residual,
    direction,
    product,
    preconditioned,
    solution,
    initial_guess,
  ):
    source, library_name = self._program(
      block_dimensions, dynamic_dimensions, diagonal_sizes
    )
    trace = os.environ.get("YASPS_SOLVER_TRACE", "").strip().lower() in {
      "1",
      "true",
      "yes",
    }
    solved, status, residual_value = _metal_solver_ext.solve_pcg(
      int(max_iterations),
      float(threshold),
      block_values._array,
      block_positions._array,
      [int(value) for value in block_starts],
      [int(value) for value in block_counts],
      [int(value) for value in block_dimensions],
      dynamic_values._array,
      dynamic_positions._array,
      [int(value) for value in dynamic_starts],
      [int(value) for value in dynamic_counts],
      [int(value) for value in dynamic_dimensions],
      inverse_blocks._array,
      [int(value) for value in diagonal_starts],
      [int(value) for value in diagonal_counts],
      [int(value) for value in diagonal_sizes],
      [int(value) for value in gradient_starts],
      gradient._array,
      p1_b._array,
      residual._array,
      direction._array,
      product._array,
      preconditioned._array,
      solution._array,
      initial_guess._array,
      source,
      library_name,
      trace=trace,
    )
    solution._array = solved
    if trace:
      print(
        f"Metal PCG status {status}, "
        f"preconditioned residual {residual_value:.9g}"
      )
    return status

"""Metal counterpart to ``solverKernel.pyx``."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import mlx.core as mx


_THREADS = 256
_REDUCTION_WIDTH = _THREADS * 2


def _metadata(values):
  return mx.array([int(value) for value in values], dtype=mx.uint32)


@lru_cache(maxsize=1)
def _copy_kernel():
  return mx.fast.metal_kernel(
    name="yasps_solver_copy",
    input_names=["source", "count"],
    output_names=["output"],
    source="""
      const uint index = thread_position_in_grid.x;
      if (index < count[0]) output[index] = source[index];
    """,
  )


@lru_cache(maxsize=1)
def _combine_kernel():
  return mx.fast.metal_kernel(
    name="yasps_solver_vector_combine",
    input_names=["left", "right", "scale", "count"],
    output_names=["output"],
    source="""
      const uint index = thread_position_in_grid.x;
      if (index < count[0]) {
        output[index] = left[index] + scale[0] * right[index];
      }
    """,
  )


@lru_cache(maxsize=1)
def _dot_blocks_kernel():
  return mx.fast.metal_kernel(
    name="yasps_solver_dot_blocks",
    input_names=["left", "right", "count"],
    output_names=["block_sums"],
    source=f"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint group = threadgroup_position_in_grid.x;
      const uint base = group * {_REDUCTION_WIDTH}u;
      const uint first = base + tid;
      const uint second = first + {_THREADS}u;
      threadgroup float scratch[{_THREADS}];
      float value = 0.0f;
      if (first < count[0]) value += left[first] * right[first];
      if (second < count[0]) value += left[second] * right[second];
      scratch[tid] = value;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint offset = {_THREADS // 2}u; offset > 0u; offset >>= 1u) {{
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }}
      if (tid == 0u) block_sums[group] = scratch[0];
    """,
  )


@lru_cache(maxsize=1)
def _sum_blocks_kernel():
  return mx.fast.metal_kernel(
    name="yasps_solver_sum_blocks",
    input_names=["values", "count"],
    output_names=["block_sums"],
    source=f"""
      const uint tid = thread_position_in_threadgroup.x;
      const uint group = threadgroup_position_in_grid.x;
      const uint base = group * {_REDUCTION_WIDTH}u;
      const uint first = base + tid;
      const uint second = first + {_THREADS}u;
      threadgroup float scratch[{_THREADS}];
      float value = 0.0f;
      if (first < count[0]) value += values[first];
      if (second < count[0]) value += values[second];
      scratch[tid] = value;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint offset = {_THREADS // 2}u; offset > 0u; offset >>= 1u) {{
        if (tid < offset) scratch[tid] += scratch[tid + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }}
      if (tid == 0u) block_sums[group] = scratch[0];
    """,
  )


def _copy(source):
  count = source.size
  if count == 0:
    return mx.empty((0,), dtype=mx.float32)
  return _copy_kernel()(
    inputs=[source, mx.array([count], dtype=mx.uint32)],
    grid=(count, 1, 1),
    threadgroup=(min(count, _THREADS), 1, 1),
    output_shapes=[source.shape],
    output_dtypes=[mx.float32],
  )[0]


def _combine(left, right, scale):
  count = left.size
  return _combine_kernel()(
    inputs=[
      left,
      right,
      mx.array([scale], dtype=mx.float32),
      mx.array([count], dtype=mx.uint32),
    ],
    grid=(count, 1, 1),
    threadgroup=(min(count, _THREADS), 1, 1),
    output_shapes=[left.shape],
    output_dtypes=[mx.float32],
  )[0]


def _reduce(values):
  while values.size > 1:
    groups = (values.size + _REDUCTION_WIDTH - 1) // _REDUCTION_WIDTH
    values = _sum_blocks_kernel()(
      inputs=[values, mx.array([values.size], dtype=mx.uint32)],
      grid=(groups * _THREADS, 1, 1),
      threadgroup=(_THREADS, 1, 1),
      output_shapes=[(groups,)],
      output_dtypes=[mx.float32],
    )[0]
  return values


def _dot(left, right):
  groups = (left.size + _REDUCTION_WIDTH - 1) // _REDUCTION_WIDTH
  values = _dot_blocks_kernel()(
    inputs=[left, right, mx.array([left.size], dtype=mx.uint32)],
    grid=(groups * _THREADS, 1, 1),
    threadgroup=(_THREADS, 1, 1),
    output_shapes=[(groups,)],
    output_dtypes=[mx.float32],
  )[0]
  return float(_reduce(values)[0].item())


def _atomic_add(output, index, value):
  return (
    "atomic_fetch_add_explicit("
    f"&{output}[{index}], {value}, memory_order_relaxed);"
  )


class MetalCGSolver:
  """CUDA solverKernel counterpart built from generated Metal kernels."""

  def __init__(self):
    self.spmv_kernels = {}
    self.preconditioner_kernels = {}

  def _spmv_kernel(self, static_dimensions, dynamic_dimensions):
    key = (tuple(static_dimensions), tuple(dynamic_dimensions))
    if key in self.spmv_kernels:
      return self.spmv_kernels[key]
    lines = [
      "const uint block = thread_position_in_grid.x;",
      "uint remaining = block;",
      "uint position_start = 0u;",
    ]
    groups = [
      ("static", row, column, group)
      for group, (row, column) in enumerate(static_dimensions)
    ] + [
      ("dynamic", row, column, group)
      for group, (row, column) in enumerate(dynamic_dimensions)
    ]
    for kind, rows, columns, group in groups:
      counts = f"{kind}_counts"
      starts = f"{kind}_starts"
      positions = f"{kind}_positions"
      values = f"{kind}_values"
      lines.extend(
        [
          f"if (remaining < {counts}[{group}]) {{",
          "  const uint position = position_start + remaining;",
          f"  const uint row = {positions}[position * 2u];",
          f"  const uint column = {positions}[position * 2u + 1u];",
          (
            f"  const uint value_start = {starts}[{group}] "
            f"+ remaining * {rows * columns}u;"
          ),
          f"  for (ushort i = 0; i < {rows}; ++i) {{",
          "    float value = 0.0f;",
          f"    for (ushort j = 0; j < {columns}; ++j) {{",
          f"      value += {values}[value_start + i * {columns}u + j] * x[column + j];",
          "    }",
          "    " + _atomic_add("result", "row + i", "value"),
          "  }",
          "  if (row != column) {",
          f"    for (ushort j = 0; j < {columns}; ++j) {{",
          "      float value = 0.0f;",
          f"      for (ushort i = 0; i < {rows}; ++i) {{",
          f"        value += {values}[value_start + i * {columns}u + j] * x[row + i];",
          "      }",
          "      " + _atomic_add("result", "column + j", "value"),
          "    }",
          "  }",
          "  return;",
          "}",
          f"remaining -= {counts}[{group}];",
          f"position_start += {counts}[{group}];",
        ]
      )
    source = "\n".join(lines)
    digest = sha256(source.encode()).hexdigest()[:16]
    kernel = mx.fast.metal_kernel(
      name=f"yasps_block_spmv_{digest}",
      input_names=[
        "static_values",
        "static_positions",
        "static_starts",
        "static_counts",
        "dynamic_values",
        "dynamic_positions",
        "dynamic_starts",
        "dynamic_counts",
        "x",
      ],
      output_names=["result"],
      source=source,
      atomic_outputs=True,
      compile_options={"math_mode": "fast"},
    )
    self._persist(f"yasps_block_spmv_{digest}", source)
    self.spmv_kernels[key] = kernel
    return kernel

  def _preconditioner_kernel(self, sizes):
    key = tuple(int(size) for size in sizes)
    if key in self.preconditioner_kernels:
      return self.preconditioner_kernels[key]
    lines = [
      "const uint block = thread_position_in_grid.x;",
      "uint remaining = block;",
    ]
    for attribute, size in enumerate(key):
      lines.extend(
        [
          f"if (remaining < block_counts[{attribute}]) {{",
          (
            f"  const uint inverse_start = block_starts[{attribute}] "
            f"+ remaining * {size * size}u;"
          ),
          (
            f"  const uint vector_start = gradient_starts[{attribute}] "
            f"+ remaining * {size}u;"
          ),
          f"  for (ushort row = 0; row < {size}; ++row) {{",
          "    float value = 0.0f;",
          f"    for (ushort column = 0; column < {size}; ++column) {{",
          (
            f"      value += inverse_blocks[inverse_start + row * {size}u "
            "+ column] * x[vector_start + column];"
          ),
          "    }",
          "    result[vector_start + row] = value;",
          "  }",
          "  return;",
          "}",
          f"remaining -= block_counts[{attribute}];",
        ]
      )
    source = "\n".join(lines)
    digest = sha256(source.encode()).hexdigest()[:16]
    kernel = mx.fast.metal_kernel(
      name=f"yasps_block_jacobi_{digest}",
      input_names=[
        "inverse_blocks",
        "x",
        "block_starts",
        "block_counts",
        "gradient_starts",
      ],
      output_names=["result"],
      source=source,
      compile_options={"math_mode": "fast"},
    )
    self._persist(f"yasps_block_jacobi_{digest}", source)
    self.preconditioner_kernels[key] = kernel
    return kernel

  @staticmethod
  def _persist(name, source):
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"{name}.metal"
    if not path.exists() or path.read_text() != source:
      path.write_text(source)

  def _spmv(
    self,
    static_values,
    static_positions,
    static_starts,
    static_counts,
    static_dimensions,
    dynamic_values,
    dynamic_positions,
    dynamic_starts,
    dynamic_counts,
    dynamic_dimensions,
    x,
  ):
    static_pairs = list(zip(static_dimensions[::2], static_dimensions[1::2]))
    dynamic_pairs = list(
      zip(dynamic_dimensions[::2], dynamic_dimensions[1::2])
    )
    total_blocks = sum(static_counts) + sum(dynamic_counts)
    if total_blocks == 0:
      return mx.zeros_like(x)
    return self._spmv_kernel(static_pairs, dynamic_pairs)(
      inputs=[
        static_values._array,
        static_positions._array,
        _metadata(static_starts),
        _metadata(static_counts),
        dynamic_values._array,
        dynamic_positions._array,
        _metadata(dynamic_starts),
        _metadata(dynamic_counts),
        x,
      ],
      grid=(total_blocks, 1, 1),
      threadgroup=(min(total_blocks, _THREADS), 1, 1),
      output_shapes=[x.shape],
      output_dtypes=[mx.float32],
      init_value=0.0,
    )[0]

  def _precondition(
    self,
    inverse_blocks,
    x,
    block_starts,
    block_counts,
    block_sizes,
    gradient_starts,
  ):
    total_blocks = sum(block_counts)
    return self._preconditioner_kernel(block_sizes)(
      inputs=[
        inverse_blocks._array,
        x,
        _metadata(block_starts),
        _metadata(block_counts),
        _metadata(gradient_starts),
      ],
      grid=(total_blocks, 1, 1),
      threadgroup=(min(total_blocks, _THREADS), 1, 1),
      output_shapes=[x.shape],
      output_dtypes=[mx.float32],
      init_value=0.0,
    )[0]

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
    def spmv(vector):
      return self._spmv(
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
        vector,
      )

    gradient_array = gradient._array
    solution._array = _copy(initial_guess._array)
    p1_b._array = self._precondition(
      inverse_blocks,
      gradient_array,
      diagonal_starts,
      diagonal_counts,
      diagonal_sizes,
      gradient_starts,
    )
    delta_zero = _dot(p1_b._array, gradient_array)
    residual._array = _combine(gradient_array, spmv(initial_guess._array), -1.0)
    direction._array = self._precondition(
      inverse_blocks,
      residual._array,
      diagonal_starts,
      diagonal_counts,
      diagonal_sizes,
      gradient_starts,
    )
    delta_new = _dot(residual._array, direction._array)
    relative_tolerance = float(threshold) * delta_zero
    if delta_new <= relative_tolerance:
      return 0

    for iteration in range(1, int(max_iterations) + 1):
      product._array = spmv(direction._array)
      denominator = _dot(direction._array, product._array)
      if denominator < 0.0:
        return -iteration - 4
      if denominator == 0.0:
        return -iteration - 4
      alpha = delta_new / denominator
      solution._array = _combine(solution._array, direction._array, alpha)
      residual._array = _combine(residual._array, product._array, -alpha)
      preconditioned._array = self._precondition(
        inverse_blocks,
        residual._array,
        diagonal_starts,
        diagonal_counts,
        diagonal_sizes,
        gradient_starts,
      )
      delta_old = delta_new
      delta_new = _dot(residual._array, preconditioned._array)
      if delta_new <= relative_tolerance:
        return iteration
      direction._array = _combine(
        preconditioned._array,
        direction._array,
        delta_new / delta_old,
      )
    return int(max_iterations) + 1

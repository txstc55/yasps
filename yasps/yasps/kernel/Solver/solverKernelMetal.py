"""Metal counterpart to ``solverKernel.pyx``."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import math
import os
from pathlib import Path

import mlx.core as mx

from yasps.kernel.Compute.compensatedSumMetal import compensated_dot


_THREADS = 256


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


def _dot(left, right):
  return compensated_dot(left, right)


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
    group_width = 32
    maximum_rows = max(
      [rows for rows, _ in static_dimensions + dynamic_dimensions],
      default=1,
    )
    lines = [
      "const uint tid = thread_position_in_threadgroup.x;",
      "uint remaining_group = threadgroup_position_in_grid.x;",
      "bool assigned = false;",
      "bool valid = false;",
      "uint output_row = 0xffffffffu;",
      f"ushort active_rows = {maximum_rows}u;",
      f"threadgroup float forward[{group_width * maximum_rows}];",
      f"threadgroup uint row_keys[{group_width}];",
      (
        f"for (ushort i = 0; i < {maximum_rows}; ++i) "
        f"forward[tid * {maximum_rows}u + i] = 0.0f;"
      ),
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
      position_starts = f"{kind}_position_starts"
      threadgroups = f"{kind}_threadgroups"
      values = f"{kind}_values"
      lines.extend(
        [
          "if (!assigned) {",
          f"  if (remaining_group < {threadgroups}[{group}]) {{",
          (
            f"    const uint block = remaining_group * {group_width}u "
            "+ tid;"
          ),
          f"    active_rows = {rows}u;",
          f"    if (block < {counts}[{group}]) {{",
          (
            f"      const uint position = {position_starts}[{group}] "
            "+ block;"
          ),
          f"      output_row = {positions}[position * 2u];",
          f"      const uint column = {positions}[position * 2u + 1u];",
          (
            f"      const uint value_start = {starts}[{group}] "
            f"+ block * {rows * columns}u;"
          ),
          f"      for (ushort i = 0; i < {rows}; ++i) {{",
          "        float value = 0.0f;",
          f"        for (ushort j = 0; j < {columns}; ++j) {{",
          (
            f"          value += {values}[value_start + i * "
            f"{columns}u + j] * x[column + j];"
          ),
          "        }",
          f"        forward[tid * {maximum_rows}u + i] = value;",
          "      }",
          "      if (output_row != column) {",
          f"        for (ushort j = 0; j < {columns}; ++j) {{",
          "          float value = 0.0f;",
          f"          for (ushort i = 0; i < {rows}; ++i) {{",
          (
            f"            value += {values}[value_start + i * "
            f"{columns}u + j] * x[output_row + i];"
          ),
          "          }",
          "          " + _atomic_add("result", "column + j", "value"),
          "        }",
          "      }",
          "      row_keys[tid] = output_row;",
          "      valid = true;",
          "    }",
          "    assigned = true;",
          "  } else {",
          f"    remaining_group -= {threadgroups}[{group}];",
          "  }",
          "}",
        ]
      )
    lines.extend(
      [
        "if (!valid) row_keys[tid] = 0xffffffffu;",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        (
          "if (valid && "
          "(tid == 0u || row_keys[tid - 1u] != row_keys[tid])) {"
        ),
        "  for (ushort component = 0; component < active_rows; ++component) {",
        "    float value = 0.0f;",
        f"    for (uint other = tid; other < {group_width}u; ++other) {{",
        "      if (row_keys[other] != row_keys[tid]) break;",
        (
          f"      value += forward[other * {maximum_rows}u + component];"
        ),
        "    }",
        "    " + _atomic_add(
          "result", "row_keys[tid] + component", "value"
        ),
        "  }",
        "}",
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
        "static_position_starts",
        "static_threadgroups",
        "dynamic_values",
        "dynamic_positions",
        "dynamic_starts",
        "dynamic_counts",
        "dynamic_position_starts",
        "dynamic_threadgroups",
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
    static_position_starts = []
    position_start = 0
    for count in static_counts:
      static_position_starts.append(position_start)
      position_start += count
    dynamic_position_starts = []
    position_start = 0
    for count in dynamic_counts:
      dynamic_position_starts.append(position_start)
      position_start += count
    group_width = 32
    static_threadgroups = [
      (count + group_width - 1) // group_width for count in static_counts
    ]
    dynamic_threadgroups = [
      (count + group_width - 1) // group_width for count in dynamic_counts
    ]
    total_threadgroups = sum(static_threadgroups) + sum(dynamic_threadgroups)
    return self._spmv_kernel(static_pairs, dynamic_pairs)(
      inputs=[
        static_values._array,
        static_positions._array,
        _metadata(static_starts),
        _metadata(static_counts),
        _metadata(static_position_starts),
        _metadata(static_threadgroups),
        dynamic_values._array,
        dynamic_positions._array,
        _metadata(dynamic_starts),
        _metadata(dynamic_counts),
        _metadata(dynamic_position_starts),
        _metadata(dynamic_threadgroups),
        x,
      ],
      grid=(total_threadgroups * group_width, 1, 1),
      threadgroup=(group_width, 1, 1),
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
    trace = os.environ.get("YASPS_SOLVER_TRACE", "").strip().lower() in {
      "1",
      "true",
      "yes",
    }

    def finish(status, residual_value):
      if trace:
        print(
          "Metal PCG status "
          f"{status}, preconditioned residual {residual_value:.9g}, "
          f"threshold {relative_tolerance:.9g}"
        )
      return status

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
      return finish(0, delta_new)
    best_delta = delta_new
    best_solution = _copy(solution._array)
    stagnant_restarts = 0

    for iteration in range(1, int(max_iterations) + 1):
      product._array = spmv(direction._array)
      denominator = _dot(direction._array, product._array)
      if not math.isfinite(denominator) or denominator <= 0.0:
        # A long float32 recurrence can lose A-conjugacy and make an SPD
        # product look non-positive.  Recompute the true residual and restart
        # from preconditioned steepest descent before declaring breakdown.
        residual._array = _combine(
          gradient_array, spmv(solution._array), -1.0
        )
        direction._array = self._precondition(
          inverse_blocks,
          residual._array,
          diagonal_starts,
          diagonal_counts,
          diagonal_sizes,
          gradient_starts,
        )
        delta_new = _dot(residual._array, direction._array)
        product._array = spmv(direction._array)
        denominator = _dot(direction._array, product._array)
        if math.isfinite(delta_new) and delta_new < best_delta:
          best_delta = delta_new
          best_solution = _copy(solution._array)
          mx.eval(best_solution)
        if (
          not math.isfinite(denominator)
          or denominator <= 0.0
        ):
          solution._array = best_solution
          return finish(-iteration - 4, best_delta)
      alpha = delta_new / denominator
      solution._array = _combine(solution._array, direction._array, alpha)
      # The solution is not consumed by either scalar reduction below.  Without
      # materializing it here, MLX retains one more vector-update node per PCG
      # iteration and eventually replays an increasingly deep lazy graph.
      mx.eval(solution._array)
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
        # A recursively updated float32 residual can look converged even after
        # drifting away from b - A*x.  Confirm against the true residual.
        residual._array = _combine(
          gradient_array, spmv(solution._array), -1.0
        )
        preconditioned._array = self._precondition(
          inverse_blocks,
          residual._array,
          diagonal_starts,
          diagonal_counts,
          diagonal_sizes,
          gradient_starts,
        )
        delta_new = _dot(residual._array, preconditioned._array)
        if delta_new <= relative_tolerance:
          return finish(iteration, delta_new)
        if math.isfinite(delta_new) and delta_new < best_delta:
          best_delta = delta_new
          best_solution = _copy(solution._array)
          mx.eval(best_solution)
        direction._array = _copy(preconditioned._array)
        continue
      if iteration % 32 == 0:
        # Residual replacement bounds recurrence drift in float32 and restarts
        # PCG with the residual of the solution actually stored on the GPU.
        residual._array = _combine(
          gradient_array, spmv(solution._array), -1.0
        )
        preconditioned._array = self._precondition(
          inverse_blocks,
          residual._array,
          diagonal_starts,
          diagonal_counts,
          diagonal_sizes,
          gradient_starts,
        )
        delta_new = _dot(residual._array, preconditioned._array)
        if delta_new <= relative_tolerance:
          return finish(iteration, delta_new)
        previous_best = best_delta
        if math.isfinite(delta_new) and delta_new < best_delta:
          best_delta = delta_new
          best_solution = _copy(solution._array)
          mx.eval(best_solution)
        if (
          math.isfinite(delta_new)
          and delta_new < previous_best * 0.99
        ):
          stagnant_restarts = 0
        else:
          stagnant_restarts += 1
        if not math.isfinite(delta_new) or stagnant_restarts >= 8:
          solution._array = best_solution
          return finish(-iteration - 4, best_delta)
        direction._array = _copy(preconditioned._array)
      else:
        direction._array = _combine(
          preconditioned._array,
          direction._array,
          delta_new / delta_old,
        )
    solution._array = best_solution
    return finish(-int(max_iterations) - 5, best_delta)

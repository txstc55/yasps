"""Generated Metal counterpart to the CUDA CCD implementation in this folder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import math
import os
import time

import mlx.core as mx
import numpy as np

from yasps.backend import GPUArray, gpuarray, synchronize
from yasps.kernel.Coordinate.scanMetal import outer_indices


_THREADS = 256
_HEADER = Path(__file__).with_name("ccd.metal").read_text()


def _trace_enabled():
  return os.environ.get(
    "YASPS_METAL_CCD_TRACE", ""
  ).strip().lower() in {"1", "true", "yes"}


def _append_enabled():
  return os.environ.get(
    "YASPS_METAL_CCD_APPEND", "1"
  ).strip().lower() not in {"0", "false", "no"}


def _array(value):
  return value._array if isinstance(value, GPUArray) else value


def _wrap(value):
  return GPUArray._wrap(value)


def _grid(count):
  return (max(1, int(count)), 1, 1)


def _threadgroup(count):
  return (max(1, min(int(count), _THREADS)), 1, 1)


@dataclass
class _SpatialGrid:
  lower: object
  upper: object
  keys: object
  elements: object
  parameters: object
  gap: float

  @property
  def reference_count(self):
    return int(self.keys.size)


@lru_cache(maxsize=2)
def _aabb_kernel(arity):
  source = f"""
    const uint element_index = thread_position_in_grid.x;
    if (element_index >= settings[0]) return;
    float3 lower_bound = float3(INFINITY);
    float3 upper_bound = float3(-INFINITY);
    for (uint local = 0u; local < {arity}u; ++local) {{
      const uint vertex_index = elements[element_index * {arity}u + local];
      const float3 start = yasps_ccd_load3(vertices, vertex_index);
      float3 finish = start;
      if (settings[1] != 0u) {{
        finish -= yasps_ccd_load3(movement, vertex_index) * options[0];
      }}
      lower_bound = min(lower_bound, min(start, finish));
      upper_bound = max(upper_bound, max(start, finish));
    }}
    lower_bound -= options[1];
    upper_bound += options[1];
    const uint offset = element_index * 3u;
    for (uint axis = 0u; axis < 3u; ++axis) {{
      lower_bounds[offset + axis] = lower_bound[axis];
      upper_bounds[offset + axis] = upper_bound[axis];
    }}
  """
  return mx.fast.metal_kernel(
    name=f"yasps_ccd_aabb_{arity}",
    input_names=["vertices", "movement", "elements", "settings", "options"],
    output_names=["lower_bounds", "upper_bounds"],
    header=_HEADER,
    source=source,
    compile_options={"math_mode": "fast"},
  )


@lru_cache(maxsize=1)
def _grid_parameters_kernel():
  source = """
    if (thread_position_in_grid.x != 0u) return;
    const float3 lower_bound =
        float3(minimums[0], minimums[1], minimums[2]);
    const float3 upper_bound =
        float3(maximums[0], maximums[1], maximums[2]);
    const float extent = max(
        upper_bound.x - lower_bound.x,
        max(upper_bound.y - lower_bound.y, upper_bound.z - lower_bound.z));
    parameters[0] = lower_bound.x;
    parameters[1] = lower_bound.y;
    parameters[2] = lower_bound.z;
    parameters[3] = max(
        max(extent / float(settings[0]), settings[1] * 2.0f), 1.0e-6f);
  """
  return mx.fast.metal_kernel(
    name="yasps_ccd_grid_parameters",
    input_names=["minimums", "maximums", "settings"],
    output_names=["parameters"],
    source=source,
  )


@lru_cache(maxsize=1)
def _cell_count_kernel():
  source = """
    const uint element_index = thread_position_in_grid.x;
    if (element_index >= count[0]) return;
    const uint offset = element_index * 3u;
    const int3 first = yasps_ccd_cell(
        float3(lower_bounds[offset], lower_bounds[offset + 1u],
               lower_bounds[offset + 2u]),
        parameters);
    const int3 last = yasps_ccd_cell(
        float3(upper_bounds[offset], upper_bounds[offset + 1u],
               upper_bounds[offset + 2u]),
        parameters);
    const ulong cells =
        ulong(last.x - first.x + 1)
        * ulong(last.y - first.y + 1)
        * ulong(last.z - first.z + 1);
    cell_counts[element_index] = uint(min(cells, ulong(0xffffffffu)));
  """
  return mx.fast.metal_kernel(
    name="yasps_ccd_cell_counts",
    input_names=["lower_bounds", "upper_bounds", "parameters", "count"],
    output_names=["cell_counts"],
    header=_HEADER,
    source=source,
  )


@lru_cache(maxsize=1)
def _cell_scatter_kernel():
  source = """
    const uint element_index = thread_position_in_grid.x;
    if (element_index >= count[0]) return;
    const uint offset = element_index * 3u;
    const int3 first = yasps_ccd_cell(
        float3(lower_bounds[offset], lower_bounds[offset + 1u],
               lower_bounds[offset + 2u]),
        parameters);
    const int3 last = yasps_ccd_cell(
        float3(upper_bounds[offset], upper_bounds[offset + 1u],
               upper_bounds[offset + 2u]),
        parameters);
    uint destination = cell_offsets[element_index];
    for (int x = first.x; x <= last.x; ++x) {
      for (int y = first.y; y <= last.y; ++y) {
        for (int z = first.z; z <= last.z; ++z) {
          keys[destination] = yasps_ccd_cell_key(int3(x, y, z));
          element_ids[destination] = element_index;
          ++destination;
        }
      }
    }
  """
  return mx.fast.metal_kernel(
    name="yasps_ccd_scatter_cell_references",
    input_names=[
      "lower_bounds",
      "upper_bounds",
      "parameters",
      "cell_offsets",
      "count",
    ],
    output_names=["keys", "element_ids"],
    header=_HEADER,
    source=source,
  )


@lru_cache(maxsize=1)
def _sort_reorder_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    const uint source_index = order[index];
    sorted_keys[index] = keys[source_index];
    sorted_elements[index] = elements[source_index];
  """
  return mx.fast.metal_kernel(
    name="yasps_ccd_reorder_sorted_cells",
    input_names=["keys", "elements", "order", "count"],
    output_names=["sorted_keys", "sorted_elements"],
    source=source,
  )


def _build_grid(vertices, movement, elements, arity, count, gap, alpha):
  trace = _trace_enabled()
  started = time.perf_counter() if trace else None
  if count == 0:
    return _SpatialGrid(
      mx.empty((0,), dtype=mx.float32),
      mx.empty((0,), dtype=mx.float32),
      mx.empty((0,), dtype=mx.uint64),
      mx.empty((0,), dtype=mx.uint32),
      mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32),
      float(gap),
    )

  has_motion = movement is not None
  movement_array = _array(movement) if has_motion else _array(vertices)
  lower, upper = _aabb_kernel(arity)(
    inputs=[
      _array(vertices),
      movement_array,
      _array(elements),
      mx.array([count, int(has_motion)], dtype=mx.uint32),
      mx.array([alpha, gap], dtype=mx.float32),
    ],
    grid=_grid(count),
    threadgroup=_threadgroup(count),
    output_shapes=[(count * 3,), (count * 3,)],
    output_dtypes=[mx.float32, mx.float32],
  )
  reshaped_lower = lower.reshape((count, 3))
  reshaped_upper = upper.reshape((count, 3))
  minimums = mx.min(reshaped_lower, axis=0)
  maximums = mx.max(reshaped_upper, axis=0)
  target_resolution = max(1, math.ceil(count ** (1.0 / 3.0)))
  parameters = _grid_parameters_kernel()(
    inputs=[
      minimums,
      maximums,
      mx.array([target_resolution, gap], dtype=mx.float32),
    ],
    grid=(1, 1, 1),
    threadgroup=(1, 1, 1),
    output_shapes=[(4,)],
    output_dtypes=[mx.float32],
  )[0]
  cell_counts = _cell_count_kernel()(
    inputs=[
      lower,
      upper,
      parameters,
      mx.array([count], dtype=mx.uint32),
    ],
    grid=_grid(count),
    threadgroup=_threadgroup(count),
    output_shapes=[(count,)],
    output_dtypes=[mx.uint32],
  )[0]
  cell_offsets = outer_indices(cell_counts)
  reference_count = int(cell_offsets[-1].item())
  if trace:
    prefix_finished = time.perf_counter()
    print(
      "Metal CCD grid prefix "
      f"arity {arity}: {(prefix_finished - started) * 1000.0:.3f} ms, "
      f"{count} elements, {reference_count} cell references"
    )
  if reference_count == 0:
    return _SpatialGrid(
      lower,
      upper,
      mx.empty((0,), dtype=mx.uint64),
      mx.empty((0,), dtype=mx.uint32),
      parameters,
      float(gap),
    )
  keys, element_ids = _cell_scatter_kernel()(
    inputs=[
      lower,
      upper,
      parameters,
      cell_offsets,
      mx.array([count], dtype=mx.uint32),
    ],
    grid=_grid(count),
    threadgroup=_threadgroup(count),
    output_shapes=[(reference_count,), (reference_count,)],
    output_dtypes=[mx.uint64, mx.uint32],
  )

  # CUDA uses Thrust for this step. MLX's argsort is the equivalent Metal
  # library primitive; the data reorder remains an explicitly generated kernel.
  order = mx.argsort(keys)
  sorted_keys, sorted_elements = _sort_reorder_kernel()(
    inputs=[
      keys,
      element_ids,
      order,
      mx.array([reference_count], dtype=mx.uint32),
    ],
    grid=_grid(reference_count),
    threadgroup=_threadgroup(reference_count),
    output_shapes=[(reference_count,), (reference_count,)],
    output_dtypes=[mx.uint64, mx.uint32],
  )
  if trace:
    mx.eval(sorted_keys, sorted_elements)
    print(
      "Metal CCD grid sort "
      f"arity {arity}: "
      f"{(time.perf_counter() - prefix_finished) * 1000.0:.3f} ms"
    )
  return _SpatialGrid(
    lower,
    upper,
    sorted_keys,
    sorted_elements,
    parameters,
    float(gap),
  )


def _query_source(kind, continuous, mode):
  if kind == "faces":
    prepare = """
      const uint vertex_id = surface_vertices[query_index];
      const float3 start = yasps_ccd_load3(vertices, vertex_id);
      float3 finish = start;
      if (continuous) {
        finish -= yasps_ccd_load3(movement, vertex_id) * options[1];
      }
      const float3 query_lower = min(start, finish);
      const float3 query_upper = max(start, finish);
      const int3 first = yasps_ccd_cell(query_lower, parameters);
      const int3 last = yasps_ccd_cell(query_upper, parameters);
    """
    candidate = """
          const uint3 triangle = uint3(
              topology[element_id * 3u],
              topology[element_id * 3u + 1u],
              topology[element_id * 3u + 2u]);
          if (vertex_id == triangle.x || vertex_id == triangle.y
              || vertex_id == triangle.z) {
            ++position;
            continue;
          }
          const uint mesh = mesh_indices[vertex_id];
          if (mesh != 0u
              && mesh_indices[triangle.x] == mesh
              && mesh_indices[triangle.y] == mesh
              && mesh_indices[triangle.z] == mesh) {
            ++position;
            continue;
          }
          if (boundary_types[vertex_id] >= 2
              && boundary_types[triangle.x] >= 2
              && boundary_types[triangle.y] >= 2
              && boundary_types[triangle.z] >= 2) {
            ++position;
            continue;
          }
          const uint element_offset = element_id * 3u;
          const float3 element_lower = float3(
              lower_bounds[element_offset],
              lower_bounds[element_offset + 1u],
              lower_bounds[element_offset + 2u]);
          const float3 element_upper = float3(
              upper_bounds[element_offset],
              upper_bounds[element_offset + 1u],
              upper_bounds[element_offset + 2u]);
          if (!yasps_ccd_overlap(
                  query_lower, query_upper, element_lower, element_upper)
              || key != yasps_ccd_cell_key(max(
                  first, yasps_ccd_cell(element_lower, parameters)))) {
            ++position;
            continue;
          }
          int4 full_pair = int4(
              -int(vertex_id) - 1,
              int(triangle.x),
              int(triangle.y),
              int(triangle.z));
          int4 discrete_pair = full_pair;
          bool hit = continuous;
          if (!continuous) {
            hit = yasps_ccd_discrete_pt(
                start,
                yasps_ccd_load3(vertices, triangle.x),
                yasps_ccd_load3(vertices, triangle.y),
                yasps_ccd_load3(vertices, triangle.z),
                vertex_id,
                triangle,
                options[0],
                discrete_pair);
          }
    """
  else:
    prepare = """
      const uint2 query_edge = uint2(
          topology[query_index * 2u], topology[query_index * 2u + 1u]);
      const uint query_offset = query_index * 3u;
      const float3 query_lower = float3(
          lower_bounds[query_offset],
          lower_bounds[query_offset + 1u],
          lower_bounds[query_offset + 2u]);
      const float3 query_upper = float3(
          upper_bounds[query_offset],
          upper_bounds[query_offset + 1u],
          upper_bounds[query_offset + 2u]);
      const int3 first = yasps_ccd_cell(query_lower, parameters);
      const int3 last = yasps_ccd_cell(query_upper, parameters);
    """
    candidate = """
          if (element_id <= query_index) {
            ++position;
            continue;
          }
          const uint2 other_edge = uint2(
              topology[element_id * 2u], topology[element_id * 2u + 1u]);
          if (query_edge.x == other_edge.x || query_edge.x == other_edge.y
              || query_edge.y == other_edge.x
              || query_edge.y == other_edge.y) {
            ++position;
            continue;
          }
          const uint mesh0 = mesh_indices[query_edge.x];
          const uint mesh1 = mesh_indices[query_edge.y];
          if (!(mesh0 == 0u || mesh1 == 0u
                || (mesh0 == mesh1
                    && (mesh0 != mesh_indices[other_edge.x]
                        || mesh0 != mesh_indices[other_edge.y])))) {
            ++position;
            continue;
          }
          if (boundary_types[query_edge.x] >= 2
              && boundary_types[query_edge.y] >= 2
              && boundary_types[other_edge.x] >= 2
              && boundary_types[other_edge.y] >= 2) {
            ++position;
            continue;
          }
          const uint element_offset = element_id * 3u;
          const float3 element_lower = float3(
              lower_bounds[element_offset],
              lower_bounds[element_offset + 1u],
              lower_bounds[element_offset + 2u]);
          const float3 element_upper = float3(
              upper_bounds[element_offset],
              upper_bounds[element_offset + 1u],
              upper_bounds[element_offset + 2u]);
          if (!yasps_ccd_overlap(
                  query_lower, query_upper, element_lower, element_upper)
              || key != yasps_ccd_cell_key(max(
                  first, yasps_ccd_cell(element_lower, parameters)))) {
            ++position;
            continue;
          }
          int4 full_pair = int4(
              int(query_edge.x),
              int(query_edge.y),
              int(other_edge.x),
              int(other_edge.y));
          int4 discrete_pair = full_pair;
          bool hit = continuous;
          if (!continuous) {
            hit = yasps_ccd_discrete_ee(
                yasps_ccd_load3(vertices, query_edge.x),
                yasps_ccd_load3(vertices, query_edge.y),
                yasps_ccd_load3(vertices, other_edge.x),
                yasps_ccd_load3(vertices, other_edge.y),
                query_edge,
                other_edge,
                options[0],
                discrete_pair);
          }
    """

  if mode == "write":
    on_hit = """
          if (hit) {
            const uint destination = pair_offsets[query_index] + local_count;
            for (uint component = 0u; component < 4u; ++component) {
              pairs[destination * 4u + component] = discrete_pair[component];
              full_pairs[destination * 4u + component] = full_pair[component];
            }
            ++local_count;
          }
    """
    finish = ""
  elif mode == "append":
    pair_store = "" if continuous else """
              atomic_store_explicit(
                  atomic_pairs + destination * 4u + component,
                  discrete_pair[component],
                  memory_order_relaxed);
    """
    on_hit = f"""
          if (hit) {{
            const uint destination = atomic_fetch_add_explicit(
                atomic_pair_count, 1u, memory_order_relaxed);
            if (destination < settings[2]) {{
              for (uint component = 0u; component < 4u; ++component) {{
                {pair_store}
                atomic_store_explicit(
                    atomic_full_pairs + destination * 4u + component,
                    full_pair[component],
                    memory_order_relaxed);
              }}
            }}
          }}
    """
    declarations = """
      device atomic_uint* atomic_pair_count =
          reinterpret_cast<device atomic_uint*>(pair_count);
      device atomic_int* atomic_full_pairs =
          reinterpret_cast<device atomic_int*>(full_pairs);
    """
    if not continuous:
      declarations += """
      device atomic_int* atomic_pairs =
          reinterpret_cast<device atomic_int*>(pairs);
      """
    finish = ""
  else:
    on_hit = """
          if (hit) ++local_count;
    """
    finish = "counts[query_index] = local_count;"
  if mode != "append":
    declarations = ""

  source = f"""
    const uint query_index = thread_position_in_grid.x;
    if (query_index >= settings[0]) return;
    constexpr bool continuous = {"true" if continuous else "false"};
    {declarations}
    {prepare}
    uint local_count = 0u;
    for (int x = first.x; x <= last.x; ++x) {{
      for (int y = first.y; y <= last.y; ++y) {{
        for (int z = first.z; z <= last.z; ++z) {{
          const ulong key = yasps_ccd_cell_key(int3(x, y, z));
          uint left = 0u;
          uint right = settings[1];
          while (left < right) {{
            const uint middle = left + (right - left) / 2u;
            if (cell_keys[middle] < key) left = middle + 1u;
            else right = middle;
          }}
          uint position = left;
          while (position < settings[1] && cell_keys[position] == key) {{
            const uint element_id = cell_elements[position];
            {candidate}
            {on_hit}
            ++position;
          }}
        }}
      }}
    }}
    {finish}
  """
  return source


@lru_cache(maxsize=12)
def _query_kernel(kind, continuous, mode):
  source = _query_source(kind, continuous, mode)
  digest = sha256(source.encode()).hexdigest()[:16]
  input_names = [
    "vertices",
    "movement",
    "topology",
    "surface_vertices",
    "boundary_types",
    "mesh_indices",
    "lower_bounds",
    "upper_bounds",
    "cell_keys",
    "cell_elements",
    "parameters",
    "settings",
    "options",
  ]
  if mode == "write":
    input_names.append("pair_offsets")
    output_names = ["pairs", "full_pairs"]
  elif mode == "append":
    output_names = (
      ["full_pairs", "pair_count"]
      if continuous
      else ["pairs", "full_pairs", "pair_count"]
    )
  else:
    output_names = ["counts"]
  return mx.fast.metal_kernel(
    name=f"yasps_ccd_query_{kind}_{digest}",
    input_names=input_names,
    output_names=output_names,
    header=_HEADER,
    source=source,
    compile_options={"math_mode": "fast"},
    atomic_outputs=mode == "append",
  )


def _initial_pair_capacity(continuous, query_count, maximum_pairs):
  divisor = 2 if continuous else 16
  return min(
    maximum_pairs,
    max(1024, (int(query_count) + divisor - 1) // divisor),
  )


def _detect_append(
    kind,
    continuous,
    inputs,
    query_count,
    maximum_pairs,
    capacity,
    trace,
    started,
):
  if capacity is None:
    capacity = _initial_pair_capacity(
      continuous, query_count, maximum_pairs
    )
  capacity = min(maximum_pairs, max(1, int(capacity)))
  empty = mx.empty((0,), dtype=mx.int32)
  attempts = 0
  while True:
    settings = mx.array(
      [
        query_count,
        int(inputs[9].size),
        capacity,
      ],
      dtype=mx.uint32,
    )
    output_shapes = (
      [(capacity * 4,), (1,)]
      if continuous
      else [(capacity * 4,), (capacity * 4,), (1,)]
    )
    output_dtypes = (
      [mx.int32, mx.uint32]
      if continuous
      else [mx.int32, mx.int32, mx.uint32]
    )
    outputs = _query_kernel(kind, continuous, "append")(
      inputs=[*inputs[:11], settings, inputs[12]],
      grid=_grid(query_count),
      threadgroup=_threadgroup(query_count),
      output_shapes=output_shapes,
      output_dtypes=output_dtypes,
      init_value=0,
    )
    if continuous:
      full_pairs, count = outputs
      pairs = empty
    else:
      pairs, full_pairs, count = outputs
    pair_count = int(count.item())
    attempts += 1
    if pair_count > maximum_pairs:
      raise RuntimeError(
        f"Metal CCD found {pair_count} pairs, exceeding the configured "
        f"maximum of {maximum_pairs}."
      )
    if pair_count <= capacity:
      pairs = pairs[:pair_count * 4]
      full_pairs = full_pairs[:pair_count * 4]
      if trace:
        mx.eval(pairs, full_pairs)
        print(
          f"Metal CCD {kind} append: "
          f"{(time.perf_counter() - started) * 1000.0:.3f} ms, "
          f"{pair_count} pairs, capacity {capacity}, "
          f"{attempts} pass{'es' if attempts != 1 else ''}"
        )
      return pairs, full_pairs, max(capacity, pair_count)
    capacity = min(
      maximum_pairs,
      max(pair_count, min(maximum_pairs, capacity * 2)),
    )


def _detect(
    kind,
    continuous,
    vertices,
    movement,
    topology,
    surface_vertices,
    boundary_types,
    mesh_indices,
    element_count,
    query_count,
    d_hat,
    alpha,
    maximum_pairs,
    capacity=None,
):
  trace = _trace_enabled()
  started = time.perf_counter() if trace else None
  arity = 3 if kind == "faces" else 2
  grid_data = _build_grid(
    vertices,
    movement if continuous else None,
    topology,
    arity,
    element_count,
    math.sqrt(max(float(d_hat), 0.0)),
    alpha,
  )
  if query_count == 0 or grid_data.reference_count == 0:
    empty = mx.empty((0,), dtype=mx.int32)
    return empty, empty, grid_data, capacity
  movement_array = _array(movement) if continuous else _array(vertices)
  surface_array = (
    _array(surface_vertices)
    if surface_vertices is not None
    else mx.empty((0,), dtype=mx.uint32)
  )
  inputs = [
    _array(vertices),
    movement_array,
    _array(topology),
    surface_array,
    _array(boundary_types),
    _array(mesh_indices),
    grid_data.lower,
    grid_data.upper,
    grid_data.keys,
    grid_data.elements,
    grid_data.parameters,
    mx.array(
      [query_count, grid_data.reference_count], dtype=mx.uint32
    ),
    mx.array([d_hat, alpha], dtype=mx.float32),
  ]
  if _append_enabled():
    pairs, full_pairs, capacity = _detect_append(
      kind,
      continuous,
      inputs,
      query_count,
      maximum_pairs,
      capacity,
      trace,
      started,
    )
    return pairs, full_pairs, grid_data, capacity
  counts = _query_kernel(kind, continuous, "count")(
    inputs=inputs,
    grid=_grid(query_count),
    threadgroup=_threadgroup(query_count),
    output_shapes=[(query_count,)],
    output_dtypes=[mx.uint32],
  )[0]
  offsets = outer_indices(counts)
  pair_count = int(offsets[-1].item())
  if trace:
    counted = time.perf_counter()
    print(
      f"Metal CCD {kind} count: "
      f"{(counted - started) * 1000.0:.3f} ms, "
      f"{pair_count} pairs"
    )
  if pair_count > maximum_pairs:
    raise RuntimeError(
      f"Metal CCD found {pair_count} pairs, exceeding the configured "
      f"maximum of {maximum_pairs}."
    )
  if pair_count == 0:
    empty = mx.empty((0,), dtype=mx.int32)
    return empty, empty, grid_data
  pairs, full_pairs = _query_kernel(kind, continuous, "write")(
    inputs=[*inputs, offsets],
    grid=_grid(query_count),
    threadgroup=_threadgroup(query_count),
    output_shapes=[(pair_count * 4,), (pair_count * 4,)],
    output_dtypes=[mx.int32, mx.int32],
  )
  if trace:
    mx.eval(pairs, full_pairs)
    print(
      f"Metal CCD {kind} write: "
      f"{(time.perf_counter() - counted) * 1000.0:.3f} ms"
    )
  return pairs, full_pairs, grid_data, capacity


@lru_cache(maxsize=None)
def _concatenate_kernel(counts):
  total = sum(counts)
  lines = [
    "const uint index = thread_position_in_grid.x;",
    f"if (index >= {total}u) return;",
  ]
  inputs = []
  offset = 0
  for array_index, count in enumerate(counts):
    inputs.append(f"pairs_{array_index}")
    condition = "if" if array_index == 0 else "else if"
    lines.extend(
      [
        f"{condition} (index < {offset + count}u) {{",
        f"  result[index] = pairs_{array_index}[index - {offset}u];",
        "}",
      ]
    )
    offset += count
  source = "\n".join(lines)
  digest = sha256(source.encode()).hexdigest()[:16]
  return mx.fast.metal_kernel(
    name=f"yasps_ccd_concatenate_{digest}",
    input_names=inputs,
    output_names=["result"],
    source=source,
  )


def _concatenate(*arrays):
  arrays = tuple(array for array in arrays if array.size)
  if not arrays:
    return mx.empty((0,), dtype=mx.int32)
  if len(arrays) == 1:
    return arrays[0]
  counts = tuple(int(array.size) for array in arrays)
  total = sum(counts)
  return _concatenate_kernel(counts)(
    inputs=list(arrays),
    grid=_grid(total),
    threadgroup=_threadgroup(total),
    output_shapes=[(total,)],
    output_dtypes=[arrays[0].dtype],
  )[0]


@lru_cache(maxsize=1)
def _separate_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    const int4 pair = int4(
        pairs[index * 4u],
        pairs[index * 4u + 1u],
        pairs[index * 4u + 2u],
        pairs[index * 4u + 3u]);
    device atomic_uint* atomic_counts =
        reinterpret_cast<device atomic_uint*>(counts);
    device atomic_uint* atomic_pp =
        reinterpret_cast<device atomic_uint*>(pp);
    device atomic_uint* atomic_pe =
        reinterpret_cast<device atomic_uint*>(pe);
    device atomic_uint* atomic_pt =
        reinterpret_cast<device atomic_uint*>(pt);
    device atomic_uint* atomic_ee =
        reinterpret_cast<device atomic_uint*>(ee);
    uint destination;
    if (pair.x >= 0) {
      destination = atomic_fetch_add_explicit(
          atomic_counts + 3u, 1u, memory_order_relaxed);
      for (uint i = 0u; i < 4u; ++i)
        atomic_store_explicit(
            atomic_ee + destination * 4u + i,
            uint(pair[i]),
            memory_order_relaxed);
    } else if (pair.z < 0) {
      destination = atomic_fetch_add_explicit(
          atomic_counts, 1u, memory_order_relaxed);
      atomic_store_explicit(
          atomic_pp + destination * 2u,
          uint(-pair.x - 1),
          memory_order_relaxed);
      atomic_store_explicit(
          atomic_pp + destination * 2u + 1u,
          uint(pair.y),
          memory_order_relaxed);
    } else if (pair.w < 0) {
      destination = atomic_fetch_add_explicit(
          atomic_counts + 1u, 1u, memory_order_relaxed);
      atomic_store_explicit(
          atomic_pe + destination * 3u,
          uint(-pair.x - 1),
          memory_order_relaxed);
      atomic_store_explicit(
          atomic_pe + destination * 3u + 1u,
          uint(pair.y),
          memory_order_relaxed);
      atomic_store_explicit(
          atomic_pe + destination * 3u + 2u,
          uint(pair.z),
          memory_order_relaxed);
    } else {
      destination = atomic_fetch_add_explicit(
          atomic_counts + 2u, 1u, memory_order_relaxed);
      atomic_store_explicit(
          atomic_pt + destination * 4u,
          uint(-pair.x - 1),
          memory_order_relaxed);
      atomic_store_explicit(
          atomic_pt + destination * 4u + 1u,
          uint(pair.y),
          memory_order_relaxed);
      atomic_store_explicit(
          atomic_pt + destination * 4u + 2u,
          uint(pair.z),
          memory_order_relaxed);
      atomic_store_explicit(
          atomic_pt + destination * 4u + 3u,
          uint(pair.w),
          memory_order_relaxed);
    }
  """
  return mx.fast.metal_kernel(
    name="yasps_ccd_separate_contact_types",
    input_names=["pairs", "count"],
    output_names=["pp", "pe", "pt", "ee", "counts"],
    source=source,
    atomic_outputs=True,
  )


def _separate(pairs):
  pair_count = int(pairs.size // 4)
  if pair_count == 0:
    return (
      mx.empty((0,), dtype=mx.uint32),
      mx.empty((0,), dtype=mx.uint32),
      mx.empty((0,), dtype=mx.uint32),
      mx.empty((0,), dtype=mx.uint32),
      mx.zeros((4,), dtype=mx.uint32),
    )
  return _separate_kernel()(
    inputs=[pairs, mx.array([pair_count], dtype=mx.uint32)],
    grid=_grid(pair_count),
    threadgroup=_threadgroup(pair_count),
    output_shapes=[
      (pair_count * 2,),
      (pair_count * 3,),
      (pair_count * 4,),
      (pair_count * 4,),
      (4,),
    ],
    output_dtypes=[
      mx.uint32,
      mx.uint32,
      mx.uint32,
      mx.uint32,
      mx.uint32,
    ],
    init_value=0,
  )


@lru_cache(maxsize=1)
def _step_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index >= count[0]) return;
    int4 pair = int4(
        pairs[index * 4u],
        pairs[index * 4u + 1u],
        pairs[index * 4u + 2u],
        pairs[index * 4u + 3u]);
    const float eta = 1.0f - slackness[0];
    float step;
    if (pair.x < 0) {
      pair.x = -pair.x - 1;
      step = yasps_ccd_pt_toc(
          yasps_ccd_load3(vertices, uint(pair.x)),
          yasps_ccd_load3(vertices, uint(pair.y)),
          yasps_ccd_load3(vertices, uint(pair.z)),
          yasps_ccd_load3(vertices, uint(pair.w)),
          -yasps_ccd_load3(movement, uint(pair.x)),
          -yasps_ccd_load3(movement, uint(pair.y)),
          -yasps_ccd_load3(movement, uint(pair.z)),
          -yasps_ccd_load3(movement, uint(pair.w)),
          eta,
          0.0f);
    } else {
      step = yasps_ccd_ee_toc(
          yasps_ccd_load3(vertices, uint(pair.x)),
          yasps_ccd_load3(vertices, uint(pair.y)),
          yasps_ccd_load3(vertices, uint(pair.z)),
          yasps_ccd_load3(vertices, uint(pair.w)),
          -yasps_ccd_load3(movement, uint(pair.x)),
          -yasps_ccd_load3(movement, uint(pair.y)),
          -yasps_ccd_load3(movement, uint(pair.z)),
          -yasps_ccd_load3(movement, uint(pair.w)),
          eta,
          0.0f);
    }
    steps[index] = clamp(step, 0.0f, 1.0f);
  """
  return mx.fast.metal_kernel(
    name="yasps_ccd_additive_step_sizes",
    input_names=["vertices", "pairs", "movement", "slackness", "count"],
    output_names=["steps"],
    header=_HEADER,
    source=source,
    compile_options={"math_mode": "fast"},
  )


class MetalCCD:
  """Float32 Metal CCD with generated broad-, narrow-, and step kernels."""

  def __init__(
      self,
      num_vertices,
      all_vertices,
      max_cd_pairs=10_000_000,
      max_ccd_pairs=100_000_000,
      mesh_indices=None):
    self._num_surface_vertices = int(num_vertices)
    self._all_vertices = int(all_vertices)
    self._max_cd_pairs = int(max_cd_pairs)
    self._max_ccd_pairs = int(max_ccd_pairs)
    self._boundary_types = gpuarray.zeros(all_vertices, dtype=np.int32)
    if mesh_indices is None or len(mesh_indices) == 0:
      self._mesh_indices = gpuarray.zeros(all_vertices, dtype=np.uint32)
    else:
      if len(mesh_indices) != all_vertices:
        raise ValueError("Length of mesh_indices must be equal to all_vertices")
      self._mesh_indices = gpuarray.to_gpu(
        np.asarray(mesh_indices, dtype=np.uint32)
      )
    self._faces = None
    self._surface_vertices = None
    self._face_count = 0
    self._edges = None
    self._edge_count = 0
    self._face_grid = None
    self._edge_grid = None
    self._pair_capacities = {}
    self.reset()

  @property
  def collision_pairs(self):
    return self._collision_pairs

  @property
  def collision_pairs_ccd(self):
    return self._collision_pairs_ccd

  @property
  def cp_num(self):
    return self._cp_num

  @property
  def separated_counts(self):
    return self._separated_counts.get().tolist()

  @property
  def pp(self):
    return self._pp

  @property
  def pe(self):
    return self._pe

  @property
  def pt(self):
    return self._pt

  @property
  def ee(self):
    return self._ee

  def init_edges(self, vertices, vertices_rest, edges, edge_num):
    del vertices, vertices_rest
    self._edges = edges
    self._edge_count = int(edge_num)

  def init_faces(self, vertices, faces, surface_vertices, face_num):
    del vertices
    self._faces = faces
    self._surface_vertices = surface_vertices
    self._face_count = int(face_num)

  def _detect_faces(self, vertices, d_hat, movement=None, alpha=1.0):
    if self._faces is None:
      empty = mx.empty((0,), dtype=mx.int32)
      return empty, empty
    continuous = movement is not None
    key = ("faces", continuous)
    (
      pairs,
      full_pairs,
      self._face_grid,
      self._pair_capacities[key],
    ) = _detect(
      "faces",
      continuous,
      vertices,
      movement,
      self._faces,
      self._surface_vertices,
      self._boundary_types,
      self._mesh_indices,
      self._face_count,
      self._num_surface_vertices,
      d_hat,
      alpha,
      self._max_ccd_pairs if continuous else self._max_cd_pairs,
      self._pair_capacities.get(key),
    )
    return pairs, full_pairs

  def _detect_edges(self, vertices, d_hat, movement=None, alpha=1.0):
    if self._edges is None:
      empty = mx.empty((0,), dtype=mx.int32)
      return empty, empty
    continuous = movement is not None
    key = ("edges", continuous)
    (
      pairs,
      full_pairs,
      self._edge_grid,
      self._pair_capacities[key],
    ) = _detect(
      "edges",
      continuous,
      vertices,
      movement,
      self._edges,
      None,
      self._boundary_types,
      self._mesh_indices,
      self._edge_count,
      self._edge_count,
      d_hat,
      alpha,
      self._max_ccd_pairs if continuous else self._max_cd_pairs,
      self._pair_capacities.get(key),
    )
    return pairs, full_pairs

  def _set_pair_buffers(self, pairs, full_pairs):
    pair_count = int(pairs.size // 4)
    self._collision_pairs = _wrap(pairs.reshape((pair_count, 4)))
    self._collision_pairs_ccd = _wrap(
      full_pairs.reshape((int(full_pairs.size // 4), 4))
    )
    self._cp_num = gpuarray.to_gpu(
      np.array([full_pairs.size // 4, 0, 0, 0, 0], dtype=np.uint32)
    )

  def _set_separated(self, pairs):
    pp, pe, pt, ee, counts = _separate(pairs)
    self._pp = _wrap(pp)
    self._pe = _wrap(pe)
    self._pt = _wrap(pt)
    self._ee = _wrap(ee)
    self._separated_counts = _wrap(counts)

  def construct_faces(self, vertices):
    if self._faces is not None:
      self._face_grid = _build_grid(
        vertices, None, self._faces, 3, self._face_count, 0.0, 1.0
      )

  def construct_edges(self, vertices):
    if self._edges is not None:
      self._edge_grid = _build_grid(
        vertices, None, self._edges, 2, self._edge_count, 0.0, 1.0
      )

  def construct_full_ccd_faces(self, vertices, moving_directions, alpha):
    if self._faces is not None:
      self._face_grid = _build_grid(
        vertices,
        moving_directions,
        self._faces,
        3,
        self._face_count,
        0.0,
        alpha,
      )

  def construct_full_ccd_edges(self, vertices, moving_directions, alpha):
    if self._edges is not None:
      self._edge_grid = _build_grid(
        vertices,
        moving_directions,
        self._edges,
        2,
        self._edge_count,
        0.0,
        alpha,
      )

  def cd_faces(self, vertices, dhat):
    pairs, full_pairs = self._detect_faces(vertices, dhat)
    self._set_pair_buffers(pairs, full_pairs)
    self._set_separated(pairs)

  def cd_edges(self, vertices, dhat):
    pairs, full_pairs = self._detect_edges(vertices, dhat)
    self._set_pair_buffers(pairs, full_pairs)
    self._set_separated(pairs)

  def ccd_faces(self, vertices, dhat, moving_directions, alpha):
    pairs, full_pairs = self._detect_faces(
      vertices, dhat, moving_directions, alpha
    )
    self._set_pair_buffers(pairs, full_pairs)

  def ccd_edges(self, vertices, dhat, moving_directions, alpha):
    pairs, full_pairs = self._detect_edges(
      vertices, dhat, moving_directions, alpha
    )
    self._set_pair_buffers(pairs, full_pairs)

  def reset(self):
    self._collision_pairs = _wrap(mx.empty((0, 4), dtype=mx.int32))
    self._collision_pairs_ccd = _wrap(mx.empty((0, 4), dtype=mx.int32))
    self._cp_num = gpuarray.zeros(5, dtype=np.uint32)
    self._pp = _wrap(mx.empty((0,), dtype=mx.uint32))
    self._pe = _wrap(mx.empty((0,), dtype=mx.uint32))
    self._pt = _wrap(mx.empty((0,), dtype=mx.uint32))
    self._ee = _wrap(mx.empty((0,), dtype=mx.uint32))
    self._separated_counts = gpuarray.zeros(4, dtype=np.uint32)

  def cd(self, vertices, dhat):
    started = time.perf_counter()
    self.reset()
    face_pairs, face_full = self._detect_faces(vertices, dhat)
    edge_pairs, edge_full = self._detect_edges(vertices, dhat)
    pairs = _concatenate(face_pairs, edge_pairs)
    full_pairs = _concatenate(face_full, edge_full)
    self._set_pair_buffers(pairs, full_pairs)
    self._set_separated(pairs)
    synchronize()
    print(f"Collision detection took {(time.perf_counter() - started) * 1000:.2f} ms")

  def ccd(self, vertices, dhat, moving_directions, alpha):
    started = time.perf_counter()
    self.reset()
    _, face_pairs = self._detect_faces(
      vertices, dhat, moving_directions, alpha
    )
    _, edge_pairs = self._detect_edges(
      vertices, dhat, moving_directions, alpha
    )
    pairs = _concatenate(face_pairs, edge_pairs)
    self._set_pair_buffers(mx.empty((0,), dtype=mx.int32), pairs)
    synchronize()
    print(
      "Continuous collision detection took "
      f"{(time.perf_counter() - started) * 1000:.2f} ms"
    )

  def compute_largest_step_size(
      self, slackness, vertices, moving_directions):
    pair_count = int(self._collision_pairs_ccd.size // 4)
    if pair_count == 0:
      return 1.0
    started = time.perf_counter()
    steps = _step_kernel()(
      inputs=[
        _array(vertices),
        self._collision_pairs_ccd._array.reshape((-1,)),
        _array(moving_directions),
        mx.array([slackness], dtype=mx.float32),
        mx.array([pair_count], dtype=mx.uint32),
      ],
      grid=_grid(pair_count),
      threadgroup=_threadgroup(pair_count),
      output_shapes=[(pair_count,)],
      output_dtypes=[mx.float32],
    )[0]
    step = float(mx.min(steps).item())
    print(
      "Computing largest step size took "
      f"{(time.perf_counter() - started) * 1000:.2f} ms"
    )
    if _trace_enabled():
      print(f"Metal CCD step reduction processed {pair_count} pairs")
    return step

  @staticmethod
  def _scene_size(grid_data):
    if grid_data is None or grid_data.lower.size == 0:
      return 0.0
    element_count = int(grid_data.lower.size // 3)
    lower = mx.min(grid_data.lower.reshape((element_count, 3)), axis=0)
    upper = mx.max(grid_data.upper.reshape((element_count, 3)), axis=0)
    extent = mx.maximum(upper - lower - 2.0 * grid_data.gap, 0.0)
    return float(mx.sum(extent * extent).item())

  def get_scene_size_faces(self):
    return self._scene_size(self._face_grid)

  def get_scene_size_edges(self):
    return self._scene_size(self._edge_grid)

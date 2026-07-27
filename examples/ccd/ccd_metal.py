"""Metal LBVH and additive CCD backend generated beside the CUDA sources."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time

import numpy as np

from yasps.backend import cuda, gpuarray


_DIRECTORY = Path(__file__).resolve().parent
_EMPTY = np.uint32(0xFFFFFFFF)


def _extract(source, start_marker, end_marker):
  start = source.index(start_marker)
  end = source.index(end_marker, start)
  return source[start:end]


def _transpile_device_source(source):
  result = source
  result = result.replace(
    "const double3* _vertexes",
    "device const float* _vertexes"
  )
  result = result.replace(
    "const double3* _rest_vertexes",
    "device const float* _rest_vertexes"
  )
  result = re.sub(
    r"_rest_vertexes\[([^\]]+)\]",
    r"yasps_load_float3(_rest_vertexes, \1)",
    result
  )
  result = re.sub(
    r"(?<!_rest)_vertexes\[([^\]]+)\]",
    r"yasps_load_float3(_vertexes, \1)",
    result
  )
  result = result.replace("__device__", "")
  result = result.replace("__host__", "")
  result = result.replace("__forceinline__", "inline")
  result = result.replace("noexcept", "")
  result = result.replace("double3", "float3")
  result = result.replace("double2", "float2")
  result = result.replace("double", "float")
  result = result.replace("uint32_t", "uint")
  result = result.replace("make_int4", "int4")
  result = result.replace("cbrt(", "yasps_cbrt(")
  result = result.replace("const float3&", "float3")
  result = result.replace("const uint&", "uint")
  result = result.replace("const float&", "float")
  result = result.replace("float&", "thread float&")
  result = result.replace(
    "uint* _cpNum",
    "device atomic_uint* _cpNum"
  )
  result = result.replace(
    "uint* _meshIndices",
    "device uint* _meshIndices"
  )
  result = result.replace(
    "int4* _collisionPair",
    "device int4* _collisionPair"
  )
  result = result.replace(
    "int4* _ccd_collisionPair",
    "device int4* _ccd_collisionPair"
  )
  result = result.replace(
    "inline bool _check",
    "inline void _check"
  )
  result = re.sub(
    r"^[ \t]*printf\s*\([^;]*\);",
    "",
    result,
    flags=re.MULTILINE | re.DOTALL
  )
  return result


def _geigen_source():
  return r'''
namespace __GEIGEN__ {

struct Matrix3x3d {
  float m[3][3];
};

inline float3 __minus(float3 left, float3 right) {
  return left - right;
}

inline float3 __add(float3 left, float3 right) {
  return left + right;
}

inline float3 __s_vec_multiply3(float3 value, float scalar) {
  return value * scalar;
}

inline float3 __v_vec_cross(float3 left, float3 right) {
  return cross(left, right);
}

inline float __v_vec_dot(float3 left, float3 right) {
  return dot(left, right);
}

inline float __squaredNorm3(float3 value) {
  return dot(value, value);
}

inline float __squaredNorm(float3 value) {
  return dot(value, value);
}

inline float __norm(float3 value) {
  return length(value);
}

inline float __mabs(float value) {
  return fabs(value);
}

inline void __set_Mat_val(
  thread Matrix3x3d& matrix,
  float a00,
  float a01,
  float a02,
  float a10,
  float a11,
  float a12,
  float a20,
  float a21,
  float a22
) {
  matrix.m[0][0] = a00;
  matrix.m[0][1] = a01;
  matrix.m[0][2] = a02;
  matrix.m[1][0] = a10;
  matrix.m[1][1] = a11;
  matrix.m[1][2] = a12;
  matrix.m[2][0] = a20;
  matrix.m[2][1] = a21;
  matrix.m[2][2] = a22;
}

inline float __Determiant_output(
  thread const Matrix3x3d& input
) {
  return input.m[0][0] * input.m[1][1] * input.m[2][2]
    + input.m[1][0] * input.m[2][1] * input.m[0][2]
    + input.m[2][0] * input.m[0][1] * input.m[1][2]
    - input.m[2][0] * input.m[1][1] * input.m[0][2]
    - input.m[0][0] * input.m[1][2] * input.m[2][1]
    - input.m[0][1] * input.m[1][0] * input.m[2][2];
}

inline float __cubic_value(
  float x,
  float a,
  float b,
  float c,
  float d
) {
  return a * x * x * x + b * x * x + c * x + d;
}

inline float __cubic_derivative(
  float x,
  float a,
  float b,
  float c
) {
  return 3.0f * a * x * x + 2.0f * b * x + c;
}

inline void __NewtonSolverForCubicEquation(
  float a,
  float b,
  float c,
  float d,
  thread float* results,
  thread int& num_solutions,
  float epsilon
) {
  float delta_x = 0.0f;
  num_solutions = 0;
  float special_point = -b / a / 3.0f;
  float positions[2];
  int solution_count = 1;
  float delta = 4.0f * b * b - 12.0f * a * c;
  if (delta > 0.0f) {
    positions[0] = (sqrt(delta) - 2.0f * b) / (6.0f * a);
    positions[1] = (-sqrt(delta) - 2.0f * b) / (6.0f * a);
    float value0 = __cubic_value(positions[0], a, b, c, d);
    float value1 = __cubic_value(positions[1], a, b, c, d);
    if (fabs(value0) < epsilon * epsilon) {
      value0 = 0.0f;
    }
    if (fabs(value1) < epsilon * epsilon) {
      value1 = 0.0f;
    }
    delta_x = positions[0] - positions[1];
    if (value0 * value1 <= 0.0f) {
      solution_count = 3;
    } else if (
      (a < 0.0f && value0 > 0.0f)
      || (a > 0.0f && value0 < 0.0f)
    ) {
      delta_x = -delta_x;
    }
  } else if (delta == 0.0f) {
    float value = __cubic_value(special_point, a, b, c, d);
    if (fabs(value) < epsilon * epsilon) {
      for (int index = 0; index < 3; ++index) {
        results[num_solutions++] = special_point;
      }
      return;
    }
    if (
      (a > 0.0f && value > 0.0f)
      || (a < 0.0f && value < 0.0f)
    ) {
      delta_x = 1.0f;
    } else {
      delta_x = -1.0f;
    }
  }

  float start = special_point - delta_x;
  float x0 = start;
  for (int solution = 0; solution < solution_count; ++solution) {
    float x1 = 0.0f;
    int iteration = 0;
    do {
      if (iteration != 0) {
        x0 = x1;
      }
      float derivative = __cubic_derivative(x0, a, b, c);
      x1 = x0 - __cubic_value(x0, a, b, c, d) / derivative;
      ++iteration;
    } while (
      fabs(x1 - x0) > epsilon
      && iteration < 100000
    );
    results[num_solutions++] = x1;
    start += delta_x;
    x0 = start;
  }
}

}  // namespace __GEIGEN__
'''


def _metal_prefix():
  return r'''
#include <metal_stdlib>
using namespace metal;

inline float3 yasps_load_float3(
  device const float* values,
  uint index
) {
  uint start = index * 3;
  return float3(values[start], values[start + 1], values[start + 2]);
}

inline float yasps_cbrt(float value) {
  return copysign(pow(fabs(value), 1.0f / 3.0f), value);
}

inline uint atomicAdd(device atomic_uint* target, uint value) {
  return atomic_fetch_add_explicit(
    target,
    value,
    memory_order_relaxed
  );
}

inline float __m_min(float left, float right) {
  return min(left, right);
}

inline float __m_max(float left, float right) {
  return max(left, right);
}
''' + _geigen_source()


def _lbvh_source():
  return r'''
struct AABB {
  float3 upper;
  float3 lower;
};

struct Node {
  uint parent_idx;
  uint left_idx;
  uint right_idx;
  uint element_idx;
};

inline AABB empty_aabb() {
  AABB result;
  result.lower = float3(INFINITY);
  result.upper = float3(-INFINITY);
  return result;
}

inline void include_point(thread AABB& box, float3 point) {
  box.lower = min(box.lower, point);
  box.upper = max(box.upper, point);
}

inline AABB merge_aabb(AABB left, AABB right) {
  AABB result;
  result.lower = min(left.lower, right.lower);
  result.upper = max(left.upper, right.upper);
  return result;
}

inline bool aabb_overlap(AABB left, AABB right, float gap) {
  if (
    right.lower.x - left.upper.x >= gap
    || left.lower.x - right.upper.x >= gap
  ) {
    return false;
  }
  if (
    right.lower.y - left.upper.y >= gap
    || left.lower.y - right.upper.y >= gap
  ) {
    return false;
  }
  if (
    right.lower.z - left.upper.z >= gap
    || left.lower.z - right.upper.z >= gap
  ) {
    return false;
  }
  return true;
}

inline uint expand_bits(uint value) {
  value = (value * 0x00010001u) & 0xFF0000FFu;
  value = (value * 0x00000101u) & 0x0F00F00Fu;
  value = (value * 0x00000011u) & 0xC30C30C3u;
  value = (value * 0x00000005u) & 0x49249249u;
  return value;
}

inline uint morton_code(float x, float y, float z) {
  constexpr float resolution = 1024.0f;
  x = clamp(x * resolution, 0.0f, resolution - 1.0f);
  y = clamp(y * resolution, 0.0f, resolution - 1.0f);
  z = clamp(z * resolution, 0.0f, resolution - 1.0f);
  uint xx = expand_bits(uint(x));
  uint yy = expand_bits(uint(y));
  uint zz = expand_bits(uint(z));
  return (xx << 2) + (yy << 1) + zz;
}

inline int common_upper_bits(ulong left, ulong right) {
  return int(clz(left ^ right));
}

inline uint2 determine_range(
  device const ulong* node_code,
  uint leaf_count,
  uint index
) {
  if (index == 0) {
    return uint2(0, leaf_count - 1);
  }
  ulong self_code = node_code[index];
  int left_delta = common_upper_bits(
    self_code,
    node_code[index - 1]
  );
  int right_delta = common_upper_bits(
    self_code,
    node_code[index + 1]
  );
  int direction = right_delta > left_delta ? 1 : -1;
  int minimum_delta = min(left_delta, right_delta);
  int maximum_length = 2;
  int delta = -1;
  int temporary = int(index) + direction * maximum_length;
  if (temporary >= 0 && temporary < int(leaf_count)) {
    delta = common_upper_bits(self_code, node_code[temporary]);
  }
  while (delta > minimum_delta) {
    maximum_length <<= 1;
    temporary = int(index) + direction * maximum_length;
    delta = -1;
    if (temporary >= 0 && temporary < int(leaf_count)) {
      delta = common_upper_bits(self_code, node_code[temporary]);
    }
  }
  int length = 0;
  int step = maximum_length >> 1;
  while (step > 0) {
    temporary = int(index) + (length + step) * direction;
    delta = -1;
    if (temporary >= 0 && temporary < int(leaf_count)) {
      delta = common_upper_bits(self_code, node_code[temporary]);
    }
    if (delta > minimum_delta) {
      length += step;
    }
    step >>= 1;
  }
  uint other = uint(int(index) + length * direction);
  return direction < 0 ? uint2(other, index) : uint2(index, other);
}

inline uint find_split(
  device const ulong* node_code,
  uint first,
  uint last
) {
  ulong first_code = node_code[first];
  ulong last_code = node_code[last];
  if (first_code == last_code) {
    return (first + last) >> 1;
  }
  int node_delta = common_upper_bits(first_code, last_code);
  uint split = first;
  uint stride = last - first;
  do {
    stride = (stride + 1) >> 1;
    uint middle = split + stride;
    if (
      middle < last
      && common_upper_bits(first_code, node_code[middle]) > node_delta
    ) {
      split = middle;
    }
  } while (stride > 1);
  return split;
}

kernel void calculate_face_leaf_boxes(
  device const float* vertices [[buffer(0)]],
  device const uint* faces [[buffer(1)]],
  device const float* directions [[buffer(2)]],
  device AABB* output [[buffer(3)]],
  constant float& alpha [[buffer(4)]],
  constant uint& count [[buffer(5)]],
  constant uint& swept [[buffer(6)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= count) {
    return;
  }
  AABB box = empty_aabb();
  for (uint corner = 0; corner < 3; ++corner) {
    uint vertex_index = faces[index * 3 + corner];
    float3 point = yasps_load_float3(vertices, vertex_index);
    include_point(box, point);
    if (swept != 0) {
      include_point(
        box,
        point - yasps_load_float3(directions, vertex_index) * alpha
      );
    }
  }
  output[index] = box;
}

kernel void calculate_edge_leaf_boxes(
  device const float* vertices [[buffer(0)]],
  device const uint* edges [[buffer(1)]],
  device const float* directions [[buffer(2)]],
  device AABB* output [[buffer(3)]],
  constant float& alpha [[buffer(4)]],
  constant uint& count [[buffer(5)]],
  constant uint& swept [[buffer(6)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= count) {
    return;
  }
  AABB box = empty_aabb();
  for (uint corner = 0; corner < 2; ++corner) {
    uint vertex_index = edges[index * 2 + corner];
    float3 point = yasps_load_float3(vertices, vertex_index);
    include_point(box, point);
    if (swept != 0) {
      include_point(
        box,
        point - yasps_load_float3(directions, vertex_index) * alpha
      );
    }
  }
  output[index] = box;
}

kernel void reduce_aabbs(
  device const AABB* input [[buffer(0)]],
  device AABB* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  threadgroup AABB values[256];
  values[local_index] =
    index < count ? input[index] : empty_aabb();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (local_index < stride) {
      values[local_index] = merge_aabb(
        values[local_index],
        values[local_index + stride]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    output[group_index] = values[0];
  }
}

kernel void copy_aabb(
  device const AABB* input [[buffer(0)]],
  device AABB* output [[buffer(1)]],
  uint index [[thread_position_in_grid]]
) {
  if (index == 0) {
    output[0] = input[0];
  }
}

kernel void calculate_morton_hashes(
  device ulong* hashes [[buffer(0)]],
  device uint* indices [[buffer(1)]],
  device const AABB* scene_box [[buffer(2)]],
  device const AABB* leaf_boxes [[buffer(3)]],
  constant uint& count [[buffer(4)]],
  constant uint& padded_count [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= padded_count) {
    return;
  }
  if (index >= count) {
    hashes[index] = 0xfffffffffffffffful;
    indices[index] = 0xffffffffu;
    return;
  }
  AABB scene = scene_box[0];
  AABB leaf = leaf_boxes[index];
  float3 scene_size = scene.upper - scene.lower;
  float3 center = (leaf.upper + leaf.lower) * 0.5f;
  float3 normalized = (center - scene.lower) / scene_size;
  ulong code = ulong(
    morton_code(normalized.x, normalized.y, normalized.z)
  );
  hashes[index] = (code << 32) | ulong(index);
  indices[index] = index;
}

kernel void bitonic_morton_step(
  device ulong* hashes [[buffer(0)]],
  device uint* indices [[buffer(1)]],
  constant uint& compare_distance [[buffer(2)]],
  constant uint& sequence_length [[buffer(3)]],
  constant uint& count [[buffer(4)]],
  uint index [[thread_position_in_grid]]
) {
  uint other = index ^ compare_distance;
  if (index >= count || other >= count || other <= index) {
    return;
  }
  bool ascending = (index & sequence_length) == 0;
  bool should_swap = ascending
    ? hashes[other] < hashes[index]
    : hashes[index] < hashes[other];
  if (should_swap) {
    ulong hash = hashes[index];
    hashes[index] = hashes[other];
    hashes[other] = hash;
    uint element = indices[index];
    indices[index] = indices[other];
    indices[other] = element;
  }
}

kernel void sort_leaf_boxes(
  device const uint* indices [[buffer(0)]],
  device const AABB* original [[buffer(1)]],
  device AABB* sorted [[buffer(2)]],
  constant uint& count [[buffer(3)]],
  uint index [[thread_position_in_grid]]
) {
  if (index < count) {
    sorted[index] = original[indices[index]];
  }
}

kernel void calculate_leaf_nodes(
  device Node* nodes [[buffer(0)]],
  device const uint* indices [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= count) {
    return;
  }
  if (index < count - 1) {
    nodes[index] = Node{
      0xffffffffu,
      0xffffffffu,
      0xffffffffu,
      0xffffffffu
    };
  }
  nodes[index + count - 1] = Node{
    0xffffffffu,
    0xffffffffu,
    0xffffffffu,
    indices[index]
  };
}

kernel void calculate_internal_nodes(
  device Node* nodes [[buffer(0)]],
  device const ulong* hashes [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= count - 1) {
    return;
  }
  uint2 range = determine_range(hashes, count, index);
  uint split = find_split(hashes, range.x, range.y);
  uint left = split;
  uint right = split + 1;
  if (min(range.x, range.y) == split) {
    left += count - 1;
  }
  if (max(range.x, range.y) == split + 1) {
    right += count - 1;
  }
  nodes[index].left_idx = left;
  nodes[index].right_idx = right;
  nodes[left].parent_idx = index;
  nodes[right].parent_idx = index;
}

kernel void fill_uint(
  device uint* output [[buffer(0)]],
  constant uint& value [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {
  if (index < count) {
    output[index] = value;
  }
}

kernel void calculate_internal_boxes(
  device const Node* nodes [[buffer(0)]],
  device AABB* boxes [[buffer(1)]],
  device atomic_uint* flags [[buffer(2)]],
  constant uint& count [[buffer(3)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= count) {
    return;
  }
  uint parent = nodes[index + count - 1].parent_idx;
  while (parent != 0xffffffffu) {
    uint expected = 0xffffffffu;
    bool first = atomic_compare_exchange_weak_explicit(
      &flags[parent],
      &expected,
      0u,
      memory_order_relaxed,
      memory_order_relaxed
    );
    if (first) {
      return;
    }
    uint left = nodes[parent].left_idx;
    uint right = nodes[parent].right_idx;
    boxes[parent] = merge_aabb(boxes[left], boxes[right]);
    parent = nodes[parent].parent_idx;
  }
}

kernel void calculate_internal_boxes_independent(
  device const ulong* hashes [[buffer(0)]],
  device AABB* boxes [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  if (group_index >= count - 1) {
    return;
  }
  // Metal device atomics only provide relaxed ordering. Compute each
  // node from its contiguous Morton leaf range so parent nodes never
  // race the child writes produced by another threadgroup.
  uint2 range = determine_range(hashes, count, group_index);
  AABB result = empty_aabb();
  for (
    uint leaf = range.x + local_index;
    leaf <= range.y;
    leaf += 32
  ) {
    result = merge_aabb(result, boxes[leaf + count - 1]);
  }
  threadgroup AABB partial[32];
  partial[local_index] = result;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 16; stride > 0; stride >>= 1) {
    if (local_index < stride) {
      partial[local_index] = merge_aabb(
        partial[local_index],
        partial[local_index + stride]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    boxes[group_index] = partial[0];
  }
}

kernel void scene_size_squared(
  device const AABB* boxes [[buffer(0)]],
  device float* output [[buffer(1)]],
  uint index [[thread_position_in_grid]]
) {
  if (index == 0) {
    float3 diagonal = boxes[0].upper - boxes[0].lower;
    output[0] = dot(diagonal, diagonal);
  }
}
'''


def _query_source():
  return r'''
kernel void query_faces_cd(
  device const int* body_types [[buffer(0)]],
  device const float* vertices [[buffer(1)]],
  device const uint* faces [[buffer(2)]],
  device const uint* surface_vertices [[buffer(3)]],
  device const AABB* boxes [[buffer(4)]],
  device const Node* nodes [[buffer(5)]],
  device int4* collision_pairs [[buffer(6)]],
  device int4* ccd_pairs [[buffer(7)]],
  device atomic_uint* pair_count [[buffer(8)]],
  device uint* mesh_indices [[buffer(9)]],
  constant float& distance_squared [[buffer(10)]],
  constant uint& count [[buffer(11)]],
  uint query_index [[thread_position_in_grid]]
) {
  if (query_index >= count) {
    return;
  }
  uint stack[64];
  uint stack_size = 1;
  stack[0] = 0;
  uint point_index = surface_vertices[query_index];
  float3 point = yasps_load_float3(vertices, point_index);
  AABB query_box;
  query_box.upper = point;
  query_box.lower = point;
  float gap = sqrt(distance_squared);
  uint current_mesh = mesh_indices[point_index];

  while (stack_size > 0) {
    uint node_index = stack[--stack_size];
    uint children[2] = {
      nodes[node_index].left_idx,
      nodes[node_index].right_idx
    };
    for (uint side = 0; side < 2; ++side) {
      uint child = children[side];
      if (
        child == 0xffffffffu
        || !aabb_overlap(query_box, boxes[child], gap)
      ) {
        continue;
      }
      uint face_index = nodes[child].element_idx;
      if (face_index == 0xffffffffu) {
        stack[stack_size++] = child;
        continue;
      }
      uint i0 = faces[face_index * 3];
      uint i1 = faces[face_index * 3 + 1];
      uint i2 = faces[face_index * 3 + 2];
      bool different_mesh =
        current_mesh == 0
        || current_mesh != mesh_indices[i0]
        || current_mesh != mesh_indices[i1]
        || current_mesh != mesh_indices[i2];
      if (
        different_mesh
        && point_index != i0
        && point_index != i1
        && point_index != i2
        && !(
          body_types[point_index] >= 2
          && body_types[i0] >= 2
          && body_types[i1] >= 2
          && body_types[i2] >= 2
        )
      ) {
        _checkPTintersection(
          vertices,
          point_index,
          i0,
          i1,
          i2,
          distance_squared,
          pair_count,
          mesh_indices,
          collision_pairs,
          ccd_pairs
        );
      }
    }
  }
}

kernel void query_faces_ccd(
  device const int* body_types [[buffer(0)]],
  device const float* vertices [[buffer(1)]],
  device const float* directions [[buffer(2)]],
  device const uint* faces [[buffer(3)]],
  device const uint* surface_vertices [[buffer(4)]],
  device const AABB* boxes [[buffer(5)]],
  device const Node* nodes [[buffer(6)]],
  device int4* ccd_pairs [[buffer(7)]],
  device atomic_uint* pair_count [[buffer(8)]],
  device const uint* mesh_indices [[buffer(9)]],
  constant float& distance_squared [[buffer(10)]],
  constant float& alpha [[buffer(11)]],
  constant uint& count [[buffer(12)]],
  uint query_index [[thread_position_in_grid]]
) {
  if (query_index >= count) {
    return;
  }
  uint stack[64];
  uint stack_size = 1;
  stack[0] = 0;
  uint point_index = surface_vertices[query_index];
  float3 point = yasps_load_float3(vertices, point_index);
  AABB query_box;
  query_box.upper = point;
  query_box.lower = point;
  include_point(
    query_box,
    point - yasps_load_float3(directions, point_index) * alpha
  );
  float gap = sqrt(distance_squared);
  uint current_mesh = mesh_indices[point_index];

  while (stack_size > 0) {
    uint node_index = stack[--stack_size];
    uint children[2] = {
      nodes[node_index].left_idx,
      nodes[node_index].right_idx
    };
    for (uint side = 0; side < 2; ++side) {
      uint child = children[side];
      if (
        child == 0xffffffffu
        || !aabb_overlap(query_box, boxes[child], gap)
      ) {
        continue;
      }
      uint face_index = nodes[child].element_idx;
      if (face_index == 0xffffffffu) {
        stack[stack_size++] = child;
        continue;
      }
      uint i0 = faces[face_index * 3];
      uint i1 = faces[face_index * 3 + 1];
      uint i2 = faces[face_index * 3 + 2];
      bool different_mesh =
        current_mesh == 0
        || current_mesh != mesh_indices[i0]
        || current_mesh != mesh_indices[i1]
        || current_mesh != mesh_indices[i2];
      if (
        different_mesh
        && point_index != i0
        && point_index != i1
        && point_index != i2
        && !(
          body_types[point_index] >= 2
          && body_types[i0] >= 2
          && body_types[i1] >= 2
          && body_types[i2] >= 2
        )
      ) {
        uint output = atomicAdd(pair_count, 1u);
        ccd_pairs[output] = int4(
          -int(point_index) - 1,
          int(i0),
          int(i1),
          int(i2)
        );
      }
    }
  }
}

kernel void query_edges_cd(
  device const int* body_types [[buffer(0)]],
  device const float* vertices [[buffer(1)]],
  device const float* rest_vertices [[buffer(2)]],
  device const uint* edges [[buffer(3)]],
  device const AABB* boxes [[buffer(4)]],
  device const Node* nodes [[buffer(5)]],
  device int4* collision_pairs [[buffer(6)]],
  device int4* ccd_pairs [[buffer(7)]],
  device atomic_uint* pair_count [[buffer(8)]],
  device uint* mesh_indices [[buffer(9)]],
  constant float& distance_squared [[buffer(10)]],
  constant uint& count [[buffer(11)]],
  uint leaf_index [[thread_position_in_grid]]
) {
  if (leaf_index >= count) {
    return;
  }
  uint node_leaf = leaf_index + count - 1;
  AABB query_box = boxes[node_leaf];
  uint self_edge = nodes[node_leaf].element_idx;
  uint self0 = edges[self_edge * 2];
  uint self1 = edges[self_edge * 2 + 1];
  uint mesh0 = mesh_indices[self0];
  uint mesh1 = mesh_indices[self1];
  float gap = sqrt(distance_squared);
  uint stack[64];
  uint stack_size = 1;
  stack[0] = 0;

  while (stack_size > 0) {
    uint node_index = stack[--stack_size];
    uint children[2] = {
      nodes[node_index].left_idx,
      nodes[node_index].right_idx
    };
    for (uint side = 0; side < 2; ++side) {
      uint child = children[side];
      if (
        child == 0xffffffffu
        || !aabb_overlap(query_box, boxes[child], gap)
      ) {
        continue;
      }
      uint other_edge = nodes[child].element_idx;
      if (other_edge == 0xffffffffu) {
        stack[stack_size++] = child;
        continue;
      }
      uint other0 = edges[other_edge * 2];
      uint other1 = edges[other_edge * 2 + 1];
      bool different_mesh =
        mesh0 == 0
        || mesh1 == 0
        || (
          mesh0 == mesh1
          && (
            mesh0 != mesh_indices[other0]
            || mesh0 != mesh_indices[other1]
          )
        );
      if (
        self_edge != other_edge
        && different_mesh
        && self0 != other0
        && self0 != other1
        && self1 != other0
        && self1 != other1
        && other_edge >= self_edge
        && !(
          body_types[self0] >= 2
          && body_types[self1] >= 2
          && body_types[other0] >= 2
          && body_types[other1] >= 2
        )
      ) {
        _checkEEintersection(
          vertices,
          rest_vertices,
          self0,
          self1,
          other0,
          other1,
          other_edge,
          distance_squared,
          pair_count,
          collision_pairs,
          ccd_pairs,
          int(count)
        );
      }
    }
  }
}

kernel void query_edges_ccd(
  device const int* body_types [[buffer(0)]],
  device const float* vertices [[buffer(1)]],
  device const uint* edges [[buffer(2)]],
  device const AABB* boxes [[buffer(3)]],
  device const Node* nodes [[buffer(4)]],
  device int4* ccd_pairs [[buffer(5)]],
  device atomic_uint* pair_count [[buffer(6)]],
  device const uint* mesh_indices [[buffer(7)]],
  constant float& distance_squared [[buffer(8)]],
  constant uint& count [[buffer(9)]],
  uint leaf_index [[thread_position_in_grid]]
) {
  if (leaf_index >= count) {
    return;
  }
  uint node_leaf = leaf_index + count - 1;
  AABB query_box = boxes[node_leaf];
  uint self_edge = nodes[node_leaf].element_idx;
  uint self0 = edges[self_edge * 2];
  uint self1 = edges[self_edge * 2 + 1];
  uint mesh0 = mesh_indices[self0];
  uint mesh1 = mesh_indices[self1];
  float gap = sqrt(distance_squared);
  uint stack[64];
  uint stack_size = 1;
  stack[0] = 0;

  while (stack_size > 0) {
    uint node_index = stack[--stack_size];
    uint children[2] = {
      nodes[node_index].left_idx,
      nodes[node_index].right_idx
    };
    for (uint side = 0; side < 2; ++side) {
      uint child = children[side];
      if (
        child == 0xffffffffu
        || !aabb_overlap(query_box, boxes[child], gap)
      ) {
        continue;
      }
      uint other_edge = nodes[child].element_idx;
      if (other_edge == 0xffffffffu) {
        stack[stack_size++] = child;
        continue;
      }
      uint other0 = edges[other_edge * 2];
      uint other1 = edges[other_edge * 2 + 1];
      bool different_mesh =
        mesh0 == 0
        || mesh1 == 0
        || (
          mesh0 == mesh1
          && (
            mesh0 != mesh_indices[other0]
            || mesh0 != mesh_indices[other1]
          )
        );
      if (
        self_edge != other_edge
        && different_mesh
        && self0 != other0
        && self0 != other1
        && self1 != other0
        && self1 != other1
        && other_edge >= self_edge
        && !(
          body_types[self0] >= 2
          && body_types[self1] >= 2
          && body_types[other0] >= 2
          && body_types[other1] >= 2
        )
      ) {
        uint output = atomicAdd(pair_count, 1u);
        ccd_pairs[output] = int4(
          int(self0),
          int(self1),
          int(other0),
          int(other1)
        );
      }
    }
  }
}

kernel void separate_face_pairs(
  device const int4* pairs [[buffer(0)]],
  device uint* pp [[buffer(1)]],
  device uint* pe [[buffer(2)]],
  device uint* pt [[buffer(3)]],
  device const uint* pair_count [[buffer(4)]],
  device atomic_uint* counts [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= pair_count[0]) {
    return;
  }
  int4 pair = pairs[index];
  if (pair.z < 0) {
    uint output = atomicAdd(&counts[0], 1u);
    pp[output * 2] = uint(-pair.x - 1);
    pp[output * 2 + 1] = uint(pair.y);
  } else if (pair.w < 0) {
    uint output = atomicAdd(&counts[1], 1u);
    pe[output * 3] = uint(-pair.x - 1);
    pe[output * 3 + 1] = uint(pair.y);
    pe[output * 3 + 2] = uint(pair.z);
  } else {
    uint output = atomicAdd(&counts[2], 1u);
    pt[output * 4] = uint(-pair.x - 1);
    pt[output * 4 + 1] = uint(pair.y);
    pt[output * 4 + 2] = uint(pair.z);
    pt[output * 4 + 3] = uint(pair.w);
  }
}

kernel void separate_edge_pairs(
  device const int4* pairs [[buffer(0)]],
  device uint* pp [[buffer(1)]],
  device uint* pe [[buffer(2)]],
  device uint* ee [[buffer(3)]],
  device const uint* pair_count [[buffer(4)]],
  device atomic_uint* counts [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= pair_count[0]) {
    return;
  }
  int4 pair = pairs[index];
  if (pair.x >= 0) {
    uint output = atomicAdd(&counts[3], 1u);
    ee[output * 4] = uint(pair.x);
    ee[output * 4 + 1] = uint(pair.y);
    ee[output * 4 + 2] = uint(pair.z);
    ee[output * 4 + 3] = uint(pair.w);
  } else if (pair.z < 0) {
    uint output = atomicAdd(&counts[0], 1u);
    pp[output * 2] = uint(-pair.x - 1);
    pp[output * 2 + 1] = uint(pair.y);
  } else if (pair.w < 0) {
    uint output = atomicAdd(&counts[1], 1u);
    pe[output * 3] = uint(-pair.x - 1);
    pe[output * 3 + 1] = uint(pair.y);
    pe[output * 3 + 2] = uint(pair.z);
  }
}
'''


def _accd_kernel_source():
  return r'''
kernel void calculate_step_reciprocals(
  device const float* vertices [[buffer(0)]],
  device const int4* pairs [[buffer(1)]],
  device const float* directions [[buffer(2)]],
  device float* output [[buffer(3)]],
  constant float& slackness [[buffer(4)]],
  constant uint& count [[buffer(5)]],
  uint index [[thread_position_in_grid]]
) {
  if (index >= count) {
    return;
  }
  int4 pair = pairs[index];
  float distance_ratio = 1.0f - slackness;
  float collision_time;
  if (pair.x < 0) {
    pair.x = -pair.x - 1;
    collision_time = point_triangle_ccd(
      yasps_load_float3(vertices, uint(pair.x)),
      yasps_load_float3(vertices, uint(pair.y)),
      yasps_load_float3(vertices, uint(pair.z)),
      yasps_load_float3(vertices, uint(pair.w)),
      -yasps_load_float3(directions, uint(pair.x)),
      -yasps_load_float3(directions, uint(pair.y)),
      -yasps_load_float3(directions, uint(pair.z)),
      -yasps_load_float3(directions, uint(pair.w)),
      distance_ratio,
      0.0f
    );
  } else {
    collision_time = edge_edge_ccd(
      yasps_load_float3(vertices, uint(pair.x)),
      yasps_load_float3(vertices, uint(pair.y)),
      yasps_load_float3(vertices, uint(pair.z)),
      yasps_load_float3(vertices, uint(pair.w)),
      -yasps_load_float3(directions, uint(pair.x)),
      -yasps_load_float3(directions, uint(pair.y)),
      -yasps_load_float3(directions, uint(pair.z)),
      -yasps_load_float3(directions, uint(pair.w)),
      distance_ratio,
      0.0f
    );
  }
  output[index] = 1.0f / collision_time;
}

kernel void reduce_max_float(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  threadgroup float values[256];
  values[local_index] =
    index < count ? input[index] : -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (local_index < stride) {
      values[local_index] = max(
        values[local_index],
        values[local_index + stride]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    output[group_index] = values[0];
  }
}
'''


def _build_source():
  mlbvh_cuda = (_DIRECTORY / "mlbvh.cu").read_text(
    encoding="utf-8"
  )
  accd_cuda = (_DIRECTORY / "ACCD.cu").read_text(
    encoding="utf-8"
  )
  intersection_cuda = _extract(
    mlbvh_cuda,
    "__device__\nvoid _d_PP",
    "__global__\nvoid _reduct_max_box"
  )
  accd_core_cuda = _extract(
    accd_cuda,
    "__device__\nint _dType_point_triangle",
    "__global__\nvoid _reduct_min_selfTimeStep_to_double"
  )
  return "\n".join([
    _metal_prefix(),
    _transpile_device_source(intersection_cuda),
    _lbvh_source(),
    _query_source(),
    _transpile_device_source(accd_core_cuda),
    _accd_kernel_source(),
  ])


class _MetalKernels:
  _instance = None

  def __new__(cls):
    if cls._instance is not None:
      return cls._instance
    instance = super().__new__(cls)
    source = _build_source()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    source_path = Path(
      f".yasps_constant/ccd_metal_{digest}.metal"
    )
    library_path = Path(
      f".yasps_constant/ccd_metal_{digest}.metallib"
    )
    if not library_path.exists():
      source_path.write_text(source, encoding="utf-8")
      gpuarray.compile_metal([source_path], library_path)

    names = [
      "calculate_face_leaf_boxes",
      "calculate_edge_leaf_boxes",
      "reduce_aabbs",
      "copy_aabb",
      "calculate_morton_hashes",
      "bitonic_morton_step",
      "sort_leaf_boxes",
      "calculate_leaf_nodes",
      "calculate_internal_nodes",
      "fill_uint",
      "calculate_internal_boxes",
      "calculate_internal_boxes_independent",
      "scene_size_squared",
      "query_faces_cd",
      "query_faces_ccd",
      "query_edges_cd",
      "query_edges_ccd",
      "separate_face_pairs",
      "separate_edge_pairs",
      "calculate_step_reciprocals",
      "reduce_max_float",
    ]
    instance.library_path = library_path
    instance.kernels = {
      name: gpuarray.MetalKernel(library_path, name)
      for name in names
    }
    cls._instance = instance
    return instance

  def __getitem__(self, name):
    return self.kernels[name]


def _next_power_of_two(value):
  return 1 << (int(value) - 1).bit_length()


class _MetalBVH:
  """Persistent buffers and the original LBVH construction sequence."""

  _aabb_bytes = 32
  _node_bytes = 16

  def __init__(self, primitive_count, primitive_kind):
    if primitive_count < 1:
      raise ValueError("Metal LBVH requires at least one primitive")
    self.count = int(primitive_count)
    self.kind = primitive_kind
    self.kernels = _MetalKernels()
    self.padded_count = _next_power_of_two(self.count)

    tree_count = 2 * self.count - 1
    self.boxes = gpuarray.empty(
      tree_count * self._aabb_bytes,
      np.uint8,
    )
    self.nodes = gpuarray.empty(
      tree_count * self._node_bytes,
      np.uint8,
    )
    self.original_boxes = gpuarray.empty(
      self.count * self._aabb_bytes,
      np.uint8,
    )
    self.reduction_a = gpuarray.empty(
      self.count * self._aabb_bytes,
      np.uint8,
    )
    self.reduction_b = gpuarray.empty(
      self.count * self._aabb_bytes,
      np.uint8,
    )
    self.hashes = gpuarray.empty(self.padded_count, np.uint64)
    self.indices = gpuarray.empty(self.padded_count, np.uint32)
    self.flags = gpuarray.empty(max(self.count - 1, 1), np.uint32)
    self.scene_size_output = gpuarray.empty(1, np.float32)
    self.dummy_directions = gpuarray.zeros(3, np.float32)
    self.last_timings = {}

  @property
  def leaf_boxes(self):
    byte_offset = (self.count - 1) * self._aabb_bytes
    return self.boxes[byte_offset:]

  def _dispatch(
    self,
    name,
    arguments,
    grid_size,
    threadgroup_size=0,
  ):
    start = time.perf_counter()
    kernel = self.kernels[name]
    kernel.dispatch(arguments, grid_size, threadgroup_size)
    elapsed = (time.perf_counter() - start) * 1000.0
    timing = self.last_timings.setdefault(
      name,
      {"wall_ms": 0.0, "gpu_ms": 0.0, "calls": 0},
    )
    timing["wall_ms"] += elapsed
    timing["gpu_ms"] += kernel.last_gpu_time_ms
    timing["calls"] += 1

  def _construct_dispatches(
    self,
    vertices,
    primitives,
    directions=None,
    alpha=0.0,
  ):
    self.last_timings = {}
    dispatches = []
    swept = directions is not None
    direction_buffer = (
      directions if swept else self.dummy_directions
    )
    leaf_kernel = (
      "calculate_face_leaf_boxes"
      if self.kind == "face"
      else "calculate_edge_leaf_boxes"
    )
    dispatches.append((
      self.kernels[leaf_kernel],
      [
        vertices,
        primitives,
        direction_buffer,
        self.original_boxes,
        np.float32(alpha),
        np.uint32(self.count),
        np.uint32(swept),
      ],
      self.count,
      0,
    ))

    source = self.original_boxes
    target = self.reduction_a
    reduction_count = self.count
    while reduction_count > 1:
      output_count = (reduction_count + 255) // 256
      dispatches.append((
        self.kernels["reduce_aabbs"],
        [source, target, np.uint32(reduction_count)],
        output_count * 256,
        256,
      ))
      reduction_count = output_count
      source, target = (
        target,
        self.reduction_b
        if target is self.reduction_a
        else self.reduction_a,
      )
    dispatches.append((
      self.kernels["copy_aabb"],
      [source, self.boxes],
      1,
      0,
    ))

    dispatches.append((
      self.kernels["calculate_morton_hashes"],
      [
        self.hashes,
        self.indices,
        self.boxes,
        self.original_boxes,
        np.uint32(self.count),
        np.uint32(self.padded_count),
      ],
      self.padded_count,
      0,
    ))
    sequence_length = 2
    while sequence_length <= self.padded_count:
      compare_distance = sequence_length // 2
      while compare_distance:
        dispatches.append((
          self.kernels["bitonic_morton_step"],
          [
            self.hashes,
            self.indices,
            np.uint32(compare_distance),
            np.uint32(sequence_length),
            np.uint32(self.padded_count),
          ],
          self.padded_count,
          0,
        ))
        compare_distance //= 2
      sequence_length *= 2

    dispatches.append((
      self.kernels["sort_leaf_boxes"],
      [
        self.indices,
        self.original_boxes,
        self.leaf_boxes,
        np.uint32(self.count),
      ],
      self.count,
      0,
    ))
    dispatches.append((
      self.kernels["calculate_leaf_nodes"],
      [self.nodes, self.indices, np.uint32(self.count)],
      self.count,
      0,
    ))
    dispatches.append((
      self.kernels["calculate_internal_nodes"],
      [self.nodes, self.hashes, np.uint32(self.count)],
      max(self.count - 1, 1),
      0,
    ))
    if self.count > 1:
      dispatches.append((
        self.kernels["calculate_internal_boxes_independent"],
        [
          self.hashes,
          self.boxes,
          np.uint32(self.count),
        ],
        (self.count - 1) * 32,
        32,
      ))
    return dispatches

  def construct(
    self,
    vertices,
    primitives,
    directions=None,
    alpha=0.0,
  ):
    gpuarray.dispatch_batch(
      self._construct_dispatches(
        vertices,
        primitives,
        directions,
        alpha,
      ),
      f"ccd_construct_{self.kind}",
    )

  def scene_size_squared(self):
    self._dispatch(
      "scene_size_squared",
      [self.boxes, self.scene_size_output],
      1,
    )
    return float(self.scene_size_output.get()[0])


class MetalCCD:
  """Float32 Metal implementation of the CUDA CCD public API."""

  def __init__(
    self,
    num_vertices,
    all_vertices,
    max_cd_pairs=10000000,
    max_ccd_pairs=100000000,
    mesh_indices=(),
  ):
    self.kernels = _MetalKernels()
    self.num_surface_vertices = int(num_vertices)
    self.all_vertices = int(all_vertices)
    self.max_cd_pairs = int(max_cd_pairs)
    self.max_ccd_pairs = int(max_ccd_pairs)
    self.face_bvh = None
    self.edge_bvh = None
    self.faces = None
    self.edges = None
    self.surface_vertices = None
    self.rest_vertices = None
    self.face_vertices = None
    self.edge_vertices = None

    self._collision_pairs = gpuarray.zeros(
      (self.max_cd_pairs, 4),
      np.int32,
    )
    self._collision_pairs_ccd = gpuarray.zeros(
      (self.max_ccd_pairs, 4),
      np.int32,
    )
    self._cp_num = gpuarray.zeros(5, np.uint32)
    self._btypes = gpuarray.zeros(self.all_vertices, np.int32)
    self._pp = gpuarray.zeros(self.max_cd_pairs * 2, np.uint32)
    self._pe = gpuarray.zeros(self.max_cd_pairs * 3, np.uint32)
    self._pt = gpuarray.zeros(self.max_cd_pairs * 4, np.uint32)
    self._ee = gpuarray.zeros(self.max_cd_pairs * 4, np.uint32)
    self._separated_counts = gpuarray.zeros(4, np.uint32)
    self._mqueue = gpuarray.zeros(self.max_ccd_pairs, np.float32)
    self._reduce_scratch = gpuarray.empty(
      max((self.max_ccd_pairs + 255) // 256, 1),
      np.float32,
    )
    self.last_timings = {}

    if len(mesh_indices) == 0:
      self._mesh_indices = gpuarray.zeros(
        self.all_vertices,
        np.uint32,
      )
    else:
      if len(mesh_indices) != self.all_vertices:
        raise ValueError(
          "Length of mesh_indices must be equal to all_vertices"
        )
      self._mesh_indices = gpuarray.to_gpu(
        np.asarray(mesh_indices, dtype=np.uint32)
      )

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

  def _dispatch(
    self,
    name,
    arguments,
    grid_size,
    threadgroup_size=0,
  ):
    start = time.perf_counter()
    kernel = self.kernels[name]
    kernel.dispatch(arguments, grid_size, threadgroup_size)
    elapsed = (time.perf_counter() - start) * 1000.0
    timing = self.last_timings.setdefault(
      name,
      {"wall_ms": 0.0, "gpu_ms": 0.0, "calls": 0},
    )
    timing["wall_ms"] += elapsed
    timing["gpu_ms"] += kernel.last_gpu_time_ms
    timing["calls"] += 1

  def _fill_words(self, array, value=0):
    word_count = array.nbytes // 4
    self._dispatch(
      "fill_uint",
      [array, np.uint32(value), np.uint32(word_count)],
      word_count,
    )

  def _pair_count(self):
    return int(self._cp_num.get()[0])

  def _check_pair_capacity(self, continuous):
    pair_count = self._pair_count()
    capacity = (
      self.max_ccd_pairs if continuous else self.max_cd_pairs
    )
    if pair_count > capacity:
      kind = "CCD" if continuous else "CD"
      raise RuntimeError(
        f"{kind} generated {pair_count} pairs, exceeding "
        f"the configured capacity of {capacity}"
      )
    return pair_count

  def init_edges(
    self,
    vertices,
    vertices_rest,
    edges,
    edge_num,
  ):
    self.edges = edges
    self.rest_vertices = vertices_rest
    self.edge_vertices = vertices
    self.edge_bvh = _MetalBVH(edge_num, "edge")

  def construct_edges(self, vertices):
    if self.edge_bvh is None:
      raise RuntimeError("init_edges must be called first")
    self.edge_vertices = vertices
    self.edge_bvh.construct(vertices, self.edges)

  def construct_full_ccd_edges(
    self,
    vertices,
    moving_directions,
    alpha,
  ):
    if self.edge_bvh is None:
      raise RuntimeError("init_edges must be called first")
    self.edge_vertices = vertices
    self.edge_bvh.construct(
      vertices,
      self.edges,
      moving_directions,
      alpha,
    )

  def cd_edges(self, vertices, dhat):
    self.construct_edges(vertices)
    count = self.edge_bvh.count
    self._dispatch(
      "query_edges_cd",
      [
        self._btypes,
        vertices,
        self.rest_vertices,
        self.edges,
        self.edge_bvh.boxes,
        self.edge_bvh.nodes,
        self._collision_pairs,
        self._collision_pairs_ccd,
        self._cp_num,
        self._mesh_indices,
        np.float32(dhat),
        np.uint32(count),
      ],
      count,
    )
    pair_count = self._check_pair_capacity(False)
    if pair_count:
      self._dispatch(
        "separate_edge_pairs",
        [
          self._collision_pairs,
          self._pp,
          self._pe,
          self._ee,
          self._cp_num,
          self._separated_counts,
        ],
        pair_count,
      )

  def ccd_edges(
    self,
    vertices,
    dhat,
    moving_directions,
    alpha,
  ):
    self.construct_full_ccd_edges(
      vertices,
      moving_directions,
      alpha,
    )
    count = self.edge_bvh.count
    self._dispatch(
      "query_edges_ccd",
      [
        self._btypes,
        vertices,
        self.edges,
        self.edge_bvh.boxes,
        self.edge_bvh.nodes,
        self._collision_pairs_ccd,
        self._cp_num,
        self._mesh_indices,
        np.float32(dhat),
        np.uint32(count),
      ],
      count,
    )
    self._check_pair_capacity(True)

  def init_faces(
    self,
    vertices,
    faces,
    surface_vertices,
    face_num,
  ):
    self.faces = faces
    self.surface_vertices = surface_vertices
    self.face_vertices = vertices
    self.face_bvh = _MetalBVH(face_num, "face")

  def construct_faces(self, vertices):
    if self.face_bvh is None:
      raise RuntimeError("init_faces must be called first")
    self.face_vertices = vertices
    self.face_bvh.construct(vertices, self.faces)

  def construct_full_ccd_faces(
    self,
    vertices,
    moving_directions,
    alpha,
  ):
    if self.face_bvh is None:
      raise RuntimeError("init_faces must be called first")
    self.face_vertices = vertices
    self.face_bvh.construct(
      vertices,
      self.faces,
      moving_directions,
      alpha,
    )

  def cd_faces(self, vertices, dhat):
    self.construct_faces(vertices)
    self._dispatch(
      "query_faces_cd",
      [
        self._btypes,
        vertices,
        self.faces,
        self.surface_vertices,
        self.face_bvh.boxes,
        self.face_bvh.nodes,
        self._collision_pairs,
        self._collision_pairs_ccd,
        self._cp_num,
        self._mesh_indices,
        np.float32(dhat),
        np.uint32(self.num_surface_vertices),
      ],
      self.num_surface_vertices,
    )
    pair_count = self._check_pair_capacity(False)
    if pair_count:
      self._dispatch(
        "separate_face_pairs",
        [
          self._collision_pairs,
          self._pp,
          self._pe,
          self._pt,
          self._cp_num,
          self._separated_counts,
        ],
        pair_count,
      )

  def ccd_faces(
    self,
    vertices,
    dhat,
    moving_directions,
    alpha,
  ):
    self.construct_full_ccd_faces(
      vertices,
      moving_directions,
      alpha,
    )
    self._dispatch(
      "query_faces_ccd",
      [
        self._btypes,
        vertices,
        moving_directions,
        self.faces,
        self.surface_vertices,
        self.face_bvh.boxes,
        self.face_bvh.nodes,
        self._collision_pairs_ccd,
        self._cp_num,
        self._mesh_indices,
        np.float32(dhat),
        np.float32(alpha),
        np.uint32(self.num_surface_vertices),
      ],
      self.num_surface_vertices,
    )
    self._check_pair_capacity(True)

  def _reset_dispatches(self):
    return [
      (
        self.kernels["fill_uint"],
        [
          self._separated_counts,
          np.uint32(0),
          np.uint32(self._separated_counts.size),
        ],
        self._separated_counts.size,
        0,
      ),
      (
        self.kernels["fill_uint"],
        [
          self._cp_num,
          np.uint32(0),
          np.uint32(self._cp_num.size),
        ],
        self._cp_num.size,
        0,
      ),
    ]

  def reset(self):
    gpuarray.dispatch_batch(
      self._reset_dispatches(),
      "ccd_reset",
    )

  def cd(self, vertices, dhat):
    start = time.perf_counter()
    self.last_timings = {}
    self.reset()
    if self.face_bvh is not None:
      self.cd_faces(vertices, dhat)
    self._fill_words(self._cp_num)
    if self.edge_bvh is not None:
      self.cd_edges(vertices, dhat)
    elapsed = (time.perf_counter() - start) * 1000.0
    print(f"Collision detection took {elapsed:.2f} ms")

  def ccd(
    self,
    vertices,
    dhat,
    moving_directions,
    alpha,
  ):
    start = time.perf_counter()
    self.last_timings = {}
    dispatches = self._reset_dispatches()
    if self.face_bvh is not None:
      self.face_vertices = vertices
      dispatches.extend(self.face_bvh._construct_dispatches(
        vertices,
        self.faces,
        moving_directions,
        alpha,
      ))
      dispatches.append((
        self.kernels["query_faces_ccd"],
        [
          self._btypes,
          vertices,
          moving_directions,
          self.faces,
          self.surface_vertices,
          self.face_bvh.boxes,
          self.face_bvh.nodes,
          self._collision_pairs_ccd,
          self._cp_num,
          self._mesh_indices,
          np.float32(dhat),
          np.float32(alpha),
          np.uint32(self.num_surface_vertices),
        ],
        self.num_surface_vertices,
        0,
      ))
    if self.edge_bvh is not None:
      self.edge_vertices = vertices
      dispatches.extend(self.edge_bvh._construct_dispatches(
        vertices,
        self.edges,
        moving_directions,
        alpha,
      ))
      count = self.edge_bvh.count
      dispatches.append((
        self.kernels["query_edges_ccd"],
        [
          self._btypes,
          vertices,
          self.edges,
          self.edge_bvh.boxes,
          self.edge_bvh.nodes,
          self._collision_pairs_ccd,
          self._cp_num,
          self._mesh_indices,
          np.float32(dhat),
          np.uint32(count),
        ],
        count,
        0,
      ))
    gpuarray.dispatch_batch(
      dispatches,
      "ccd_continuous",
    )
    self._check_pair_capacity(True)
    elapsed = (time.perf_counter() - start) * 1000.0
    print(
      f"Continuous collision detection took {elapsed:.2f} ms"
    )

  def compute_largest_step_size(
    self,
    slackness,
    vertices,
    moving_directions,
  ):
    start = time.perf_counter()
    pair_count = self._pair_count()
    print("number of collision pairs:", pair_count)
    if pair_count < 1:
      return 1.0
    dispatches = [(
      self.kernels["calculate_step_reciprocals"],
      [
        vertices,
        self._collision_pairs_ccd,
        moving_directions,
        self._mqueue,
        np.float32(slackness),
        np.uint32(pair_count),
      ],
      pair_count,
      0,
    )]
    source = self._mqueue
    target = self._reduce_scratch
    reduction_count = pair_count
    while reduction_count > 1:
      output_count = (reduction_count + 255) // 256
      dispatches.append((
        self.kernels["reduce_max_float"],
        [source, target, np.uint32(reduction_count)],
        output_count * 256,
        256,
      ))
      reduction_count = output_count
      source, target = target, source
    gpuarray.dispatch_batch(
      dispatches,
      "ccd_largest_step",
    )
    reciprocal = float(source.get()[0])
    step = 1.0 / reciprocal
    elapsed = (time.perf_counter() - start) * 1000.0
    print(
      f"Computing largest step size took {elapsed:.2f} ms"
    )
    return step

  def get_scene_size_faces(self):
    if self.face_bvh is None:
      raise RuntimeError("init_faces must be called first")
    self.face_bvh.construct(
      self.face_vertices,
      self.faces,
    )
    return self.face_bvh.scene_size_squared()

  def get_scene_size_edges(self):
    if self.edge_bvh is None:
      raise RuntimeError("init_edges must be called first")
    return self.edge_bvh.scene_size_squared()

#include <metal_stdlib>
using namespace metal;

#ifndef YASPS_CCD_METAL
#define YASPS_CCD_METAL

inline float3 yasps_ccd_load3(const device float* values, const uint index) {
  const uint offset = index * 3u;
  return float3(values[offset], values[offset + 1u], values[offset + 2u]);
}

inline int3 yasps_ccd_cell(const float3 point, const constant float* grid) {
  const float inverse_cell_size = 1.0f / grid[3];
  const float3 scaled = floor(
      (point - float3(grid[0], grid[1], grid[2])) * inverse_cell_size);
  return clamp(int3(scaled), int3(0), int3(0x1fffff));
}

inline ulong yasps_ccd_cell_key(const int3 cell) {
  return (ulong(uint(cell.x)) << 42)
      | (ulong(uint(cell.y)) << 21)
      | ulong(uint(cell.z));
}

inline bool yasps_ccd_overlap(
    const float3 lower_a,
    const float3 upper_a,
    const float3 lower_b,
    const float3 upper_b) {
  return all(lower_a <= upper_b) && all(lower_b <= upper_a);
}

inline float yasps_ccd_det3(
    const float3 a, const float3 b, const float3 c) {
  return dot(a, cross(b, c));
}

inline float yasps_ccd_point_point(const float3 a, const float3 b) {
  return length_squared(a - b);
}

inline float yasps_ccd_point_edge(
    const float3 p, const float3 a, const float3 b) {
  const float3 edge = b - a;
  const float denominator = length_squared(edge);
  if (denominator <= 1.0e-20f) return length_squared(p - a);
  return length_squared(cross(a - p, b - p)) / denominator;
}

inline float yasps_ccd_point_triangle(
    const float3 p,
    const float3 a,
    const float3 b,
    const float3 c) {
  const float3 normal = cross(b - a, c - a);
  const float denominator = length_squared(normal);
  if (denominator <= 1.0e-20f) {
    return min(
        yasps_ccd_point_edge(p, a, b),
        min(yasps_ccd_point_edge(p, b, c),
            yasps_ccd_point_edge(p, c, a)));
  }
  const float numerator = dot(p - a, normal);
  return numerator * numerator / denominator;
}

inline float yasps_ccd_edge_edge(
    const float3 a0,
    const float3 a1,
    const float3 b0,
    const float3 b1) {
  const float3 normal = cross(a1 - a0, b1 - b0);
  const float denominator = length_squared(normal);
  if (denominator <= 1.0e-20f) {
    return min(
        min(yasps_ccd_point_edge(a0, b0, b1),
            yasps_ccd_point_edge(a1, b0, b1)),
        min(yasps_ccd_point_edge(b0, a0, a1),
            yasps_ccd_point_edge(b1, a0, a1)));
  }
  const float numerator = dot(b0 - a0, normal);
  return numerator * numerator / denominator;
}

inline int yasps_ccd_point_triangle_type(
    const float3 p,
    const float3 t0,
    const float3 t1,
    const float3 t2) {
  float3 basis0 = t1 - t0;
  float3 basis1 = t2 - t0;
  const float3 basis2 = p - t0;
  const float3 normal = cross(basis0, basis1);
  if (length_squared(normal) <= 1.0e-20f) {
    float distances[6] = {
      yasps_ccd_point_point(p, t0),
      yasps_ccd_point_point(p, t1),
      yasps_ccd_point_point(p, t2),
      yasps_ccd_point_edge(p, t0, t1),
      yasps_ccd_point_edge(p, t1, t2),
      yasps_ccd_point_edge(p, t2, t0),
    };
    int selected = 0;
    for (int i = 1; i < 6; ++i) {
      if (distances[i] < distances[selected]) selected = i;
    }
    return selected;
  }

  basis1 = cross(basis0, normal);
  float denominator = yasps_ccd_det3(basis0, basis1, normal);
  const float2 parameter0 = float2(
      yasps_ccd_det3(basis2, basis1, normal) / denominator,
      yasps_ccd_det3(basis0, basis2, normal) / denominator);
  if (parameter0.x > 0.0f && parameter0.x < 1.0f
      && parameter0.y >= 0.0f) {
    return 3;
  }

  basis0 = t2 - t1;
  basis1 = cross(basis0, normal);
  const float3 basis_from_t1 = p - t1;
  denominator = yasps_ccd_det3(basis0, basis1, normal);
  const float2 parameter1 = float2(
      yasps_ccd_det3(basis_from_t1, basis1, normal) / denominator,
      yasps_ccd_det3(basis0, basis_from_t1, normal) / denominator);
  if (parameter1.x > 0.0f && parameter1.x < 1.0f
      && parameter1.y >= 0.0f) {
    return 4;
  }

  basis0 = t0 - t2;
  basis1 = cross(basis0, normal);
  const float3 basis_from_t2 = p - t2;
  denominator = yasps_ccd_det3(basis0, basis1, normal);
  const float2 parameter2 = float2(
      yasps_ccd_det3(basis_from_t2, basis1, normal) / denominator,
      yasps_ccd_det3(basis0, basis_from_t2, normal) / denominator);
  if (parameter2.x > 0.0f && parameter2.x < 1.0f
      && parameter2.y >= 0.0f) {
    return 5;
  }
  if (parameter0.x <= 0.0f && parameter2.x >= 1.0f) return 0;
  if (parameter1.x <= 0.0f && parameter0.x >= 1.0f) return 1;
  if (parameter2.x <= 0.0f && parameter1.x >= 1.0f) return 2;
  return 6;
}

inline int yasps_ccd_edge_edge_type(
    const float3 a0,
    const float3 a1,
    const float3 b0,
    const float3 b1) {
  const float3 u = a1 - a0;
  const float3 v = b1 - b0;
  const float3 w = a0 - b0;
  const float a = length_squared(u);
  const float b = dot(u, v);
  const float c = length_squared(v);
  const float d = dot(u, w);
  const float e = dot(v, w);
  const float determinant = a * c - b * b;
  float t_denominator = determinant;
  const float s_numerator = b * e - c * d;
  float t_numerator;
  int default_case = 8;
  if (s_numerator <= 0.0f) {
    t_numerator = e;
    t_denominator = c;
    default_case = 2;
  } else if (s_numerator >= determinant) {
    t_numerator = e + b;
    t_denominator = c;
    default_case = 5;
  } else {
    t_numerator = a * e - b * d;
    const float3 crossed = cross(u, v);
    if (t_numerator > 0.0f && t_numerator < t_denominator
        && (dot(w, crossed) == 0.0f
            // CUDA's 1e-20 relative test is appropriate for float64 but is
            // below a useful classification scale in float32.  Treat edges
            // within roughly 1e-6 radians as parallel so the EE distance
            // never divides by an unresolved cross product.
            || length_squared(crossed) < 1.0e-12f * a * c)) {
      if (s_numerator < determinant * 0.5f) {
        t_numerator = e;
        t_denominator = c;
        default_case = 2;
      } else {
        t_numerator = e + b;
        t_denominator = c;
        default_case = 5;
      }
    }
  }
  if (t_numerator <= 0.0f) {
    if (-d <= 0.0f) return 0;
    if (-d >= a) return 3;
    return 6;
  }
  if (t_numerator >= t_denominator) {
    if (-d + b <= 0.0f) return 1;
    if (-d + b >= a) return 4;
    return 7;
  }
  return default_case;
}

inline float yasps_ccd_point_triangle_distance(
    const float3 p,
    const float3 t0,
    const float3 t1,
    const float3 t2) {
  switch (yasps_ccd_point_triangle_type(p, t0, t1, t2)) {
    case 0: return yasps_ccd_point_point(p, t0);
    case 1: return yasps_ccd_point_point(p, t1);
    case 2: return yasps_ccd_point_point(p, t2);
    case 3: return yasps_ccd_point_edge(p, t0, t1);
    case 4: return yasps_ccd_point_edge(p, t1, t2);
    case 5: return yasps_ccd_point_edge(p, t2, t0);
    default: return yasps_ccd_point_triangle(p, t0, t1, t2);
  }
}

inline float yasps_ccd_edge_edge_distance(
    const float3 a0,
    const float3 a1,
    const float3 b0,
    const float3 b1) {
  switch (yasps_ccd_edge_edge_type(a0, a1, b0, b1)) {
    case 0: return yasps_ccd_point_point(a0, b0);
    case 1: return yasps_ccd_point_point(a0, b1);
    case 2: return yasps_ccd_point_edge(a0, b0, b1);
    case 3: return yasps_ccd_point_point(a1, b0);
    case 4: return yasps_ccd_point_point(a1, b1);
    case 5: return yasps_ccd_point_edge(a1, b0, b1);
    case 6: return yasps_ccd_point_edge(b0, a0, a1);
    case 7: return yasps_ccd_point_edge(b1, a0, a1);
    default: return yasps_ccd_edge_edge(a0, a1, b0, b1);
  }
}

inline bool yasps_ccd_discrete_pt(
    const float3 p,
    const float3 t0,
    const float3 t1,
    const float3 t2,
    const uint p_id,
    const uint3 triangle,
    const float d_hat,
    thread int4& pair) {
  const int type = yasps_ccd_point_triangle_type(p, t0, t1, t2);
  float distance;
  if (type == 0) {
    distance = yasps_ccd_point_point(p, t0);
    pair = int4(-int(p_id) - 1, int(triangle.x), -1, -1);
  } else if (type == 1) {
    distance = yasps_ccd_point_point(p, t1);
    pair = int4(-int(p_id) - 1, int(triangle.y), -1, -1);
  } else if (type == 2) {
    distance = yasps_ccd_point_point(p, t2);
    pair = int4(-int(p_id) - 1, int(triangle.z), -1, -1);
  } else if (type == 3) {
    distance = yasps_ccd_point_edge(p, t0, t1);
    pair = int4(-int(p_id) - 1, int(triangle.x), int(triangle.y), -1);
  } else if (type == 4) {
    distance = yasps_ccd_point_edge(p, t1, t2);
    pair = int4(-int(p_id) - 1, int(triangle.y), int(triangle.z), -1);
  } else if (type == 5) {
    distance = yasps_ccd_point_edge(p, t2, t0);
    pair = int4(-int(p_id) - 1, int(triangle.z), int(triangle.x), -1);
  } else {
    distance = yasps_ccd_point_triangle(p, t0, t1, t2);
    pair = int4(
        -int(p_id) - 1,
        int(triangle.x),
        int(triangle.y),
        int(triangle.z));
  }
  return distance < d_hat;
}

inline bool yasps_ccd_discrete_ee(
    const float3 a0,
    const float3 a1,
    const float3 b0,
    const float3 b1,
    const uint2 edge_a,
    const uint2 edge_b,
    const float d_hat,
    thread int4& pair) {
  const int type = yasps_ccd_edge_edge_type(a0, a1, b0, b1);
  float distance;
  if (type == 0) {
    distance = yasps_ccd_point_point(a0, b0);
    pair = int4(-int(edge_a.x) - 1, int(edge_b.x), -1, -1);
  } else if (type == 1) {
    distance = yasps_ccd_point_point(a0, b1);
    pair = int4(-int(edge_a.x) - 1, int(edge_b.y), -1, -1);
  } else if (type == 2) {
    distance = yasps_ccd_point_edge(a0, b0, b1);
    pair = int4(-int(edge_a.x) - 1, int(edge_b.x), int(edge_b.y), -1);
  } else if (type == 3) {
    distance = yasps_ccd_point_point(a1, b0);
    pair = int4(-int(edge_a.y) - 1, int(edge_b.x), -1, -1);
  } else if (type == 4) {
    distance = yasps_ccd_point_point(a1, b1);
    pair = int4(-int(edge_a.y) - 1, int(edge_b.y), -1, -1);
  } else if (type == 5) {
    distance = yasps_ccd_point_edge(a1, b0, b1);
    pair = int4(-int(edge_a.y) - 1, int(edge_b.x), int(edge_b.y), -1);
  } else if (type == 6) {
    distance = yasps_ccd_point_edge(b0, a0, a1);
    pair = int4(-int(edge_b.x) - 1, int(edge_a.x), int(edge_a.y), -1);
  } else if (type == 7) {
    distance = yasps_ccd_point_edge(b1, a0, a1);
    pair = int4(-int(edge_b.y) - 1, int(edge_a.x), int(edge_a.y), -1);
  } else {
    distance = yasps_ccd_edge_edge(a0, a1, b0, b1);
    pair = int4(
        int(edge_a.x), int(edge_a.y), int(edge_b.x), int(edge_b.y));
  }
  return distance < d_hat;
}

inline float yasps_ccd_pt_toc(
    float3 p,
    float3 t0,
    float3 t1,
    float3 t2,
    float3 dp,
    float3 dt0,
    float3 dt1,
    float3 dt2,
    const float eta,
  const float thickness) {
  const float3 average = -0.25f * (dp + dt0 + dt1 + dt2);
  dp += average;
  dt0 += average;
  dt1 += average;
  dt2 += average;
  const float max_displacement =
      length(dp) + sqrt(max(length_squared(dt0),
          max(length_squared(dt1), length_squared(dt2))));
  if (max_displacement == 0.0f) return 1.0f;
  float distance_squared =
      yasps_ccd_point_triangle_distance(p, t0, t1, t2);
  float distance = sqrt(max(distance_squared, 0.0f));
  const float gap =
      eta * (distance_squared - thickness * thickness)
      / (distance + thickness);
  float toc = 0.0f;
  for (uint iteration = 0; iteration < 10000u; ++iteration) {
    const float lower_bound =
        (1.0f - eta) * (distance_squared - thickness * thickness)
        / ((distance + thickness) * max_displacement);
    if (!isfinite(lower_bound) || lower_bound <= 0.0f) return toc;
    p += dp * lower_bound;
    t0 += dt0 * lower_bound;
    t1 += dt1 * lower_bound;
    t2 += dt2 * lower_bound;
    distance_squared =
        yasps_ccd_point_triangle_distance(p, t0, t1, t2);
    distance = sqrt(max(distance_squared, 0.0f));
    if (toc > 0.0f
        && (distance_squared - thickness * thickness)
            / (distance + thickness) < gap) {
      return toc;
    }
    toc += lower_bound;
    if (toc > 1.0f) return 1.0f;
  }
  return min(toc, 1.0f);
}

inline float yasps_ccd_ee_toc(
    float3 a0,
    float3 a1,
    float3 b0,
    float3 b1,
    float3 da0,
    float3 da1,
    float3 db0,
    float3 db1,
    const float eta,
    const float thickness) {
  const float3 average = -0.25f * (da0 + da1 + db0 + db1);
  da0 += average;
  da1 += average;
  db0 += average;
  db1 += average;
  const float max_displacement =
      sqrt(max(length_squared(da0), length_squared(da1)))
      + sqrt(max(length_squared(db0), length_squared(db1)));
  if (max_displacement == 0.0f) return 1.0f;
  float distance_squared = yasps_ccd_edge_edge_distance(a0, a1, b0, b1);
  float distance_function = distance_squared - thickness * thickness;
  if (distance_function <= 0.0f) {
    distance_squared = min(
        min(length_squared(a0 - b0), length_squared(a0 - b1)),
        min(length_squared(a1 - b0), length_squared(a1 - b1)));
    distance_function = distance_squared - thickness * thickness;
  }
  float distance = sqrt(max(distance_squared, 0.0f));
  const float gap = eta * distance_function / (distance + thickness);
  float toc = 0.0f;
  for (uint iteration = 0; iteration < 10000u; ++iteration) {
    const float lower_bound =
        (1.0f - eta) * distance_function
        / ((distance + thickness) * max_displacement);
    if (!isfinite(lower_bound) || lower_bound <= 0.0f) return toc;
    a0 += da0 * lower_bound;
    a1 += da1 * lower_bound;
    b0 += db0 * lower_bound;
    b1 += db1 * lower_bound;
    distance_squared = yasps_ccd_edge_edge_distance(a0, a1, b0, b1);
    distance_function = distance_squared - thickness * thickness;
    if (distance_function <= 0.0f) {
      distance_squared = min(
          min(length_squared(a0 - b0), length_squared(a0 - b1)),
          min(length_squared(a1 - b0), length_squared(a1 - b1)));
      distance_function = distance_squared - thickness * thickness;
    }
    distance = sqrt(max(distance_squared, 0.0f));
    if (toc > 0.0f && distance_function / (distance + thickness) < gap) {
      return toc;
    }
    toc += lower_bound;
    if (toc > 1.0f) return 1.0f;
  }
  return min(toc, 1.0f);
}

#endif

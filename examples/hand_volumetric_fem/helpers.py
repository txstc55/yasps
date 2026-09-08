"""Geometry, YASPS energies, collision updates, and output helpers."""
from pathlib import Path
import os
import subprocess

import numpy as np


##################################################################
## Rest geometry and physical mass
##################################################################
def tetrahedron_signed_six_volumes(positions, tetrahedra):
  points = positions[tetrahedra]
  return np.einsum("ij,ij->i", np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), points[:, 3] - points[:, 0])


def tetrahedron_vertex_masses(positions, tetrahedra, density):
  masses = density * tetrahedron_signed_six_volumes(positions, tetrahedra) / 6.0
  if np.any(masses <= 0.0):
    raise RuntimeError("Physical mass construction encountered a non-positive tetrahedron.")
  result = np.bincount(tetrahedra.ravel(), weights=np.repeat(0.25 * masses, 4), minlength=len(positions)).astype(np.float64)
  if np.any(result <= 0.0):
    raise RuntimeError("A tetrahedral vertex received no positive volume mass.")
  return result, masses


def closed_piece_vertex_masses(positions, triangles, vertex_offsets, triangle_offsets, density):
  result = np.zeros(len(positions), dtype=np.float64)
  piece_volumes = []
  for piece in range(len(vertex_offsets) - 1):
    vertex_start, vertex_end = map(int, vertex_offsets[piece:piece + 2])
    triangle_start, triangle_end = map(int, triangle_offsets[piece:piece + 2])
    local_positions = positions[vertex_start:vertex_end]
    local_triangles = triangles[triangle_start:triangle_end].astype(np.int64) - vertex_start
    triangle_points = local_positions[local_triangles]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]), axis=1)
    volume = abs(float(np.einsum("ij,ij->i", triangle_points[:, 0], np.cross(triangle_points[:, 1], triangle_points[:, 2])).sum() / 6.0))
    incident_area = np.bincount(local_triangles.ravel(), weights=np.repeat(triangle_areas / 3.0, 3), minlength=len(local_positions))
    if volume <= 0.0 or np.any(incident_area <= 0.0):
      raise RuntimeError(f"Rigid piece {piece} is not a closed positive-volume surface.")
    result[vertex_start:vertex_end] = density * volume * incident_area / incident_area.sum()
    piece_volumes.append(volume)
  return result, np.asarray(piece_volumes)


def compact_triangles(full_indices, full_triangles, full_vertex_count):
  lookup = np.full(full_vertex_count, -1, dtype=np.int64)
  lookup[full_indices] = np.arange(len(full_indices))
  result = lookup[full_triangles]
  if np.any(result < 0):
    raise RuntimeError("A compact surface triangle references a non-surface vertex.")
  return result.astype(np.uint32)


def unique_edges(triangles):
  edges = np.concatenate((triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]))
  return np.unique(np.sort(edges, axis=1), axis=0).astype(np.uint32)


##################################################################
## Inversion-free tetrahedron step bound
##################################################################
class TetrahedronStepLimiter:
  """Find the first positive-volume boundary along a linear GPU step."""

  _module = None
  _step_kernel = None
  _jacobian_kernel = None

  def __init__(self, rest_positions, tetrahedra, vertex_offset=0):
    from pycuda import gpuarray

    rest_positions = np.ascontiguousarray(rest_positions, dtype=np.float64)
    tetrahedra = np.ascontiguousarray(tetrahedra, dtype=np.uint32)
    if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
      raise ValueError("Rest positions must have shape (num_vertices, 3).")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or len(tetrahedra) == 0:
      raise ValueError("Tetrahedra must have nonzero shape (num_tetrahedra, 4).")
    if int(tetrahedra.max()) >= len(rest_positions):
      raise ValueError("A tetrahedron references a rest vertex outside the array.")
    if vertex_offset < 0:
      raise ValueError("The vertex offset must be nonnegative.")
    rest_six_volumes = tetrahedron_signed_six_volumes(rest_positions, tetrahedra)
    if np.any(~np.isfinite(rest_six_volumes)) or np.any(rest_six_volumes == 0.0):
      raise ValueError("The rest tetrahedra must have finite, nonzero signed volume.")

    self._compile()
    self.tetrahedra = gpuarray.to_gpu(tetrahedra.ravel())
    self.rest_six_volumes = gpuarray.to_gpu(np.ascontiguousarray(rest_six_volumes))
    self.candidate_steps = gpuarray.empty(len(tetrahedra), dtype=np.float64)
    self.vertex_offset = int(vertex_offset)
    self.required_vertex_count = self.vertex_offset + int(tetrahedra.max()) + 1
    self.num_tetrahedra = len(tetrahedra)

  @classmethod
  def _compile(cls):
    if cls._module is not None:
      return
    from pycuda.compiler import SourceModule

    source = r'''
#include <float.h>
#include <math.h>

extern "C" {

__device__ __forceinline__ double determinant(
  const double ax, const double ay, const double az,
  const double bx, const double by, const double bz,
  const double cx, const double cy, const double cz
) {
  return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
}

__device__ __forceinline__ double polynomial(
  const double c0, const double c1, const double c2, const double c3, const double alpha
) {
  return ((c3 * alpha + c2) * alpha + c1) * alpha + c0;
}

__global__ void tetrahedron_step_bound(
  const double* positions,
  const double* directions,
  const unsigned int* tetrahedra,
  const double* rest_six_volumes,
  const unsigned int vertex_offset,
  const unsigned int num_tetrahedra,
  const double maximum_step,
  const double minimum_jacobian,
  const double safety,
  double* candidate_steps
) {
  const unsigned int tet = blockIdx.x * blockDim.x + threadIdx.x;
  if (tet >= num_tetrahedra) return;

  const unsigned int i0 = vertex_offset + tetrahedra[4 * tet];
  const unsigned int i1 = vertex_offset + tetrahedra[4 * tet + 1];
  const unsigned int i2 = vertex_offset + tetrahedra[4 * tet + 2];
  const unsigned int i3 = vertex_offset + tetrahedra[4 * tet + 3];
  const double ax = positions[3 * i1] - positions[3 * i0];
  const double ay = positions[3 * i1 + 1] - positions[3 * i0 + 1];
  const double az = positions[3 * i1 + 2] - positions[3 * i0 + 2];
  const double bx = positions[3 * i2] - positions[3 * i0];
  const double by = positions[3 * i2 + 1] - positions[3 * i0 + 1];
  const double bz = positions[3 * i2 + 2] - positions[3 * i0 + 2];
  const double cx = positions[3 * i3] - positions[3 * i0];
  const double cy = positions[3 * i3 + 1] - positions[3 * i0 + 1];
  const double cz = positions[3 * i3 + 2] - positions[3 * i0 + 2];
  const double dax = directions[3 * i1] - directions[3 * i0];
  const double day = directions[3 * i1 + 1] - directions[3 * i0 + 1];
  const double daz = directions[3 * i1 + 2] - directions[3 * i0 + 2];
  const double dbx = directions[3 * i2] - directions[3 * i0];
  const double dby = directions[3 * i2 + 1] - directions[3 * i0 + 1];
  const double dbz = directions[3 * i2 + 2] - directions[3 * i0 + 2];
  const double dcx = directions[3 * i3] - directions[3 * i0];
  const double dcy = directions[3 * i3 + 1] - directions[3 * i0 + 1];
  const double dcz = directions[3 * i3 + 2] - directions[3 * i0 + 2];

  // The determinant is cubic along x(alpha) = x - alpha * direction.
  // Normalize alpha to u in [0, 1] before solving so small CCD bounds do not
  // make the polynomial degree tests ill-conditioned.
  const double inverse_rest_volume = 1.0 / rest_six_volumes[tet];
  double c0 = determinant(ax, ay, az, bx, by, bz, cx, cy, cz) * inverse_rest_volume - minimum_jacobian;
  double c1 = -(
    determinant(dax, day, daz, bx, by, bz, cx, cy, cz) +
    determinant(ax, ay, az, dbx, dby, dbz, cx, cy, cz) +
    determinant(ax, ay, az, bx, by, bz, dcx, dcy, dcz)
  ) * inverse_rest_volume * maximum_step;
  double c2 = (
    determinant(dax, day, daz, dbx, dby, dbz, cx, cy, cz) +
    determinant(dax, day, daz, bx, by, bz, dcx, dcy, dcz) +
    determinant(ax, ay, az, dbx, dby, dbz, dcx, dcy, dcz)
  ) * inverse_rest_volume * maximum_step * maximum_step;
  double c3 = -determinant(dax, day, daz, dbx, dby, dbz, dcx, dcy, dcz) * inverse_rest_volume * maximum_step * maximum_step * maximum_step;
  if (!(c0 > 0.0) || !isfinite(c0) || !isfinite(c1) || !isfinite(c2) || !isfinite(c3)) {
    candidate_steps[tet] = 0.0;
    return;
  }
  const double coefficient_scale = fmax(fabs(c0), fmax(fabs(c1), fmax(fabs(c2), fabs(c3))));
  c0 /= coefficient_scale;
  c1 /= coefficient_scale;
  c2 /= coefficient_scale;
  c3 /= coefficient_scale;

  // Derivative roots split the cubic into monotone intervals. Checking those
  // endpoints detects two crossings even when both u=0 and u=1 are feasible.
  const double derivative_a = 3.0 * c3;
  const double derivative_b = 2.0 * c2;
  const double derivative_c = c1;
  const double derivative_scale = fmax(1.0, fmax(fabs(derivative_a), fmax(fabs(derivative_b), fabs(derivative_c))));
  const double derivative_tolerance = 64.0 * DBL_EPSILON * derivative_scale;
  double critical0 = 1.0;
  double critical1 = 1.0;
  if (fabs(derivative_a) > derivative_tolerance) {
    const double discriminant = derivative_b * derivative_b - 4.0 * derivative_a * derivative_c;
    const double discriminant_tolerance = 128.0 * DBL_EPSILON * (derivative_b * derivative_b + fabs(4.0 * derivative_a * derivative_c) + 1.0);
    if (discriminant >= -discriminant_tolerance) {
      const double square_root = sqrt(fmax(0.0, discriminant));
      const double stable_term = -0.5 * (derivative_b + copysign(square_root, derivative_b));
      if (fabs(stable_term) > derivative_tolerance) {
        critical0 = stable_term / derivative_a;
        critical1 = derivative_c / stable_term;
      } else {
        critical0 = -derivative_b / (2.0 * derivative_a);
        critical1 = critical0;
      }
    }
  } else if (fabs(derivative_b) > derivative_tolerance) {
    critical0 = -derivative_c / derivative_b;
  }
  if (!(critical0 > 0.0 && critical0 < 1.0 && isfinite(critical0))) critical0 = 1.0;
  if (!(critical1 > 0.0 && critical1 < 1.0 && isfinite(critical1))) critical1 = 1.0;
  if (critical1 < critical0) {
    const double temporary = critical0;
    critical0 = critical1;
    critical1 = temporary;
  }

  const double candidates[3] = {critical0, critical1, 1.0};
  int crossing_interval = -1;
  double high_value = 0.0;
  for (int index = 0; index < 3; ++index) {
    const double alpha = candidates[index];
    const double value = polynomial(c0, c1, c2, c3, alpha);
    const double evaluation_tolerance = 256.0 * DBL_EPSILON * (
      fabs(c3 * alpha * alpha * alpha) + fabs(c2 * alpha * alpha) + fabs(c1 * alpha) + fabs(c0) + 1.0
    );
    if (value <= evaluation_tolerance) {
      crossing_interval = index;
      high_value = value;
      break;
    }
  }
  if (crossing_interval < 0) {
    candidate_steps[tet] = maximum_step;
    return;
  }

  double high = candidates[crossing_interval];
  // A small positive value at a stationary point represents a tangential root
  // within floating-point tolerance. A negative endpoint has a sign-changing
  // root, which is isolated accurately by bisection on its monotone interval.
  if (high_value < 0.0) {
    double low = crossing_interval == 0 ? 0.0 : candidates[crossing_interval - 1];
    for (int iteration = 0; iteration < 52; ++iteration) {
      const double middle = 0.5 * (low + high);
      if (polynomial(c0, c1, c2, c3, middle) <= 0.0) high = middle;
      else low = middle;
    }
  }
  candidate_steps[tet] = fmin(maximum_step, safety * maximum_step * high);
}

__global__ void tetrahedron_jacobians(
  const double* positions,
  const unsigned int* tetrahedra,
  const double* rest_six_volumes,
  const unsigned int vertex_offset,
  const unsigned int num_tetrahedra,
  double* jacobians
) {
  const unsigned int tet = blockIdx.x * blockDim.x + threadIdx.x;
  if (tet >= num_tetrahedra) return;
  const unsigned int i0 = vertex_offset + tetrahedra[4 * tet];
  const unsigned int i1 = vertex_offset + tetrahedra[4 * tet + 1];
  const unsigned int i2 = vertex_offset + tetrahedra[4 * tet + 2];
  const unsigned int i3 = vertex_offset + tetrahedra[4 * tet + 3];
  jacobians[tet] = determinant(
    positions[3 * i1] - positions[3 * i0], positions[3 * i1 + 1] - positions[3 * i0 + 1], positions[3 * i1 + 2] - positions[3 * i0 + 2],
    positions[3 * i2] - positions[3 * i0], positions[3 * i2 + 1] - positions[3 * i0 + 1], positions[3 * i2 + 2] - positions[3 * i0 + 2],
    positions[3 * i3] - positions[3 * i0], positions[3 * i3 + 1] - positions[3 * i0 + 1], positions[3 * i3 + 2] - positions[3 * i0 + 2]
  ) / rest_six_volumes[tet];
}

}
'''
    cls._module = SourceModule(source, no_extern_c=True, options=["-std=c++14"])
    cls._step_kernel = cls._module.get_function("tetrahedron_step_bound")
    cls._jacobian_kernel = cls._module.get_function("tetrahedron_jacobians")

  def _validate_gpu_positions(self, positions, name):
    from pycuda import gpuarray

    if not isinstance(positions, gpuarray.GPUArray) or positions.dtype != np.float64:
      raise TypeError(f"{name} must be a float64 GPUArray.")
    if positions.size % 3 != 0 or positions.size // 3 < self.required_vertex_count:
      raise ValueError(f"{name} does not contain every tetrahedron vertex.")

  def compute_largest_step_size(self, positions, directions, maximum_step, minimum_jacobian=1.0e-6, safety=0.9):
    from pycuda import gpuarray

    self._validate_gpu_positions(positions, "positions")
    self._validate_gpu_positions(directions, "directions")
    if positions.size != directions.size:
      raise ValueError("Positions and directions must have the same size.")
    if not np.isfinite(maximum_step) or not 0.0 < maximum_step <= 1.0:
      raise ValueError("The maximum step must be finite and in (0, 1].")
    if not np.isfinite(minimum_jacobian) or minimum_jacobian < 0.0:
      raise ValueError("The minimum deformation Jacobian must be finite and nonnegative.")
    if not np.isfinite(safety) or not 0.0 < safety < 1.0:
      raise ValueError("The volume step safety must be finite and in (0, 1).")
    block_size = 256
    self._step_kernel(positions, directions, self.tetrahedra, self.rest_six_volumes, np.uint32(self.vertex_offset), np.uint32(self.num_tetrahedra), np.float64(maximum_step), np.float64(minimum_jacobian), np.float64(safety), self.candidate_steps, block=(block_size, 1, 1), grid=((self.num_tetrahedra + block_size - 1) // block_size, 1, 1))
    return float(gpuarray.min(self.candidate_steps).get())

  def minimum_jacobian(self, positions):
    from pycuda import gpuarray

    self._validate_gpu_positions(positions, "positions")
    block_size = 256
    self._jacobian_kernel(positions, self.tetrahedra, self.rest_six_volumes, np.uint32(self.vertex_offset), np.uint32(self.num_tetrahedra), self.candidate_steps, block=(block_size, 1, 1), grid=((self.num_tetrahedra + block_size - 1) // block_size, 1, 1))
    return float(gpuarray.min(self.candidate_steps).get())


##################################################################
## Implicit Euler and constitutive energies
##################################################################
def inertia(last_position, velocity, dt, position, mass, gravity):
  from yasps import attribute
  target = last_position + velocity * dt + attribute.to_array([0.0, -gravity * dt * dt, 0.0], rows=3, cols=1)
  difference = position - target
  return 0.5 * mass * difference.dot(difference)


def stable_neo_hookean(rest_position, position, mu, lam, dt):
  from yasps import attribute
  rest0, rest1, rest2, rest3 = rest_position.row(0), rest_position.row(1), rest_position.row(2), rest_position.row(3)
  rest_edge0, rest_edge1, rest_edge2 = rest1 - rest0, rest2 - rest0, rest3 - rest0
  rest_basis = attribute.to_array([rest_edge0[0], rest_edge0[1], rest_edge0[2], rest_edge1[0], rest_edge1[1], rest_edge1[2], rest_edge2[0], rest_edge2[1], rest_edge2[2]], rows=3, cols=3).transpose()
  volume = rest_basis.determinant() / 6.0
  current0, current1, current2, current3 = position.row(0), position.row(1), position.row(2), position.row(3)
  edge0, edge1, edge2 = current1 - current0, current2 - current0, current3 - current0
  current_basis = attribute.to_array([edge0[0], edge0[1], edge0[2], edge1[0], edge1[1], edge1[2], edge2[0], edge2[1], edge2[2]], rows=3, cols=3).transpose()
  deformation = current_basis * rest_basis.inverse()
  jacobian = deformation.determinant()
  invariant = (deformation.transpose() * deformation).trace()
  volume_residual = jacobian - (1.0 + 0.75 * mu / lam)
  return volume * (0.5 * mu * (invariant - 3.0) - 0.5 * mu * (invariant + 1.0).log() + 0.5 * lam * volume_residual * volume_residual) * dt * dt


def tetrahedron_inversion_barrier(rest_position, position, stiffness, activation_jacobian, dt):
  from yasps import attribute
  rest0, rest1, rest2, rest3 = rest_position.row(0), rest_position.row(1), rest_position.row(2), rest_position.row(3)
  rest_edge0, rest_edge1, rest_edge2 = rest1 - rest0, rest2 - rest0, rest3 - rest0
  rest_basis = attribute.to_array([rest_edge0[0], rest_edge0[1], rest_edge0[2], rest_edge1[0], rest_edge1[1], rest_edge1[2], rest_edge2[0], rest_edge2[1], rest_edge2[2]], rows=3, cols=3).transpose()
  current0, current1, current2, current3 = position.row(0), position.row(1), position.row(2), position.row(3)
  edge0, edge1, edge2 = current1 - current0, current2 - current0, current3 - current0
  current_basis = attribute.to_array([edge0[0], edge0[1], edge0[2], edge1[0], edge1[1], edge1[2], edge2[0], edge2[1], edge2[2]], rows=3, cols=3).transpose()
  jacobian = (current_basis * rest_basis.inverse()).determinant()
  residual = jacobian - activation_jacobian
  active_barrier = -stiffness * (rest_basis.determinant() / 6.0) * residual * residual * (jacobian / activation_jacobian).log() * dt * dt
  return attribute.select(jacobian < activation_jacobian, active_barrier, attribute(float_value=0.0))


def affine_target(affine, target, matrix_weight, translation_weight, dt):
  linear_residual = sum((affine[i, j] - target[i, j]) * (affine[i, j] - target[i, j]) for i in range(3) for j in range(3))
  translation_residual = sum((affine[i, 3] - target[i, 3]) * (affine[i, 3] - target[i, 3]) for i in range(3))
  return 0.5 * dt * dt * (matrix_weight * linear_residual + translation_weight * translation_residual)


def affine_orthogonality(affine, weight, dt):
  from yasps import attribute
  linear = attribute.to_array([affine[i, j] for i in range(3) for j in range(3)], rows=3, cols=3)
  orthogonality = linear.transpose() * linear - attribute.identity(3)
  orthogonality_residual = sum(orthogonality.row(i).dot(orthogonality.row(i)) for i in range(3))
  return 0.5 * weight * dt * dt * orthogonality_residual


def affine_determinant(affine, weight, dt):
  from yasps import attribute
  linear = attribute.to_array([affine[i, j] for i in range(3) for j in range(3)], rows=3, cols=3)
  determinant_residual = linear.determinant() - 1.0
  return 0.5 * weight * dt * dt * determinant_residual * determinant_residual


##################################################################
## IPC barrier energies
##################################################################
def point_point(position, dhat, kappa):
  difference = position.row(1) - position.row(0)
  squared_distance = difference.dot(difference)
  logarithm = (squared_distance / dhat).log()
  return kappa * (squared_distance - dhat) * (squared_distance - dhat) * logarithm * logarithm


def point_edge(position, dhat, kappa):
  point, edge0, edge1 = position.row(0), position.row(1), position.row(2)
  cross = (edge0 - point).cross(edge1 - point)
  squared_distance = cross.dot(cross) / (edge1 - edge0).dot(edge1 - edge0)
  logarithm = (squared_distance / dhat).log()
  return kappa * (squared_distance - dhat) * (squared_distance - dhat) * logarithm * logarithm


def point_triangle(position, dhat, kappa):
  point, triangle0, triangle1, triangle2 = position.row(0), position.row(1), position.row(2), position.row(3)
  normal = (triangle1 - triangle0).cross(triangle2 - triangle0)
  numerator = (point - triangle0).dot(normal)
  squared_distance = numerator * numerator / normal.dot(normal)
  logarithm = (squared_distance / dhat).log()
  return kappa * (squared_distance - dhat) * (squared_distance - dhat) * logarithm * logarithm


def edge_edge(position, dhat, kappa):
  edge00, edge01, edge10, edge11 = position.row(0), position.row(1), position.row(2), position.row(3)
  normal = (edge01 - edge00).cross(edge11 - edge10)
  numerator = (edge10 - edge00).dot(normal)
  squared_distance = numerator * numerator / normal.dot(normal)
  logarithm = (squared_distance / dhat).log()
  return kappa * (squared_distance - dhat) * (squared_distance - dhat) * logarithm * logarithm


##################################################################
## Dynamic contact updates and numerical validation
##################################################################
def update_collision_pairs(detector, positions, dhat, primitives, connectivities, cached_alpha=None):
  if cached_alpha is None:
    counts = detector.cd(positions, dhat)
  else:
    try:
      counts = detector.cd_from_cached_ccd(positions, dhat, cached_alpha)
    except (ValueError, RuntimeError) as error:
      if str(error) not in ("Trial vertices lie outside the cached swept bounds", "ccd() must create a candidate cache before cached filtering"):
        raise
      counts = detector.cd(positions, dhat)
  for primitive, connectivity, width, pair_values, count in zip(primitives, connectivities, (2, 3, 4, 4), (detector.pp, detector.pe, detector.pt, detector.ee), counts):
    primitive.updateNumInstances(int(count))
    if count:
      connectivity.updateConnectivity(pair_values[:width * int(count)])
  return tuple(map(int, counts))


def validate_collision_pairs(detector, counts, vertex_count, surface_mask, mesh_ids):
  for width, pair_values, count in zip((2, 3, 4, 4), (detector.pp, detector.pe, detector.pt, detector.ee), counts):
    if count == 0:
      continue
    pairs = pair_values[:width * count].get().reshape(count, width)
    if np.any(pairs < 0) or np.any(pairs >= vertex_count):
      raise RuntimeError("CCD emitted an index outside the position union.")
    if not np.all(surface_mask[pairs]):
      raise RuntimeError("CCD emitted a volume-interior vertex.")
    pair_mesh_ids = mesh_ids[pairs]
    suppressed = np.all(pair_mesh_ids == pair_mesh_ids[:, :1], axis=1) & (pair_mesh_ids[:, 0] != 0)
    if np.any(suppressed):
      raise RuntimeError("CCD emitted a self-contact entirely inside one rigid bone mesh ID.")


def apply_affine(rest_homogeneous, vertex_to_body, affine):
  return np.einsum("vij,vj->vi", affine[vertex_to_body], rest_homogeneous)


##################################################################
## Compact surface output and fixed-camera rendering
##################################################################
def triangle_cells(triangles):
  return np.column_stack((np.full(len(triangles), 3, dtype=np.uint32), triangles)).ravel()


def make_plotter(asset, include_bunny, off_screen, window_size):
  import pyvista as pv
  hand_vertex_count = len(asset["hand_rest_positions"])
  outer_indices = asset["outer_surface_indices"]
  outer = pv.PolyData(asset["hand_rest_positions"][outer_indices].copy(), triangle_cells(asset["outer_compact_triangles"]))
  bones = pv.PolyData(asset["bone_rest_positions"].copy(), triangle_cells(asset["bone_surface_triangles"]))
  plotter = pv.Plotter(window_size=window_size, off_screen=off_screen, title="Volumetric affine-bone hand")
  plotter.set_background("#eef1f5")
  plotter.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0.0)
  plotter.add_mesh(outer, color="#d6aa82", opacity=0.58, smooth_shading=True, ambient=0.32, diffuse=0.68, specular=0.18)
  plotter.add_mesh(bones, color="#f0dfb2", opacity=1.0, smooth_shading=True, ambient=0.40, diffuse=0.70, specular=0.25)
  bunny = None
  if include_bunny:
    bunny_compact = compact_triangles(asset["bunny_surface_indices"], asset["bunny_surface_triangles"], len(asset["bunny_rest_positions"]))
    bunny = pv.PolyData(asset["bunny_rest_positions"][asset["bunny_surface_indices"]].copy(), triangle_cells(bunny_compact))
    plotter.add_mesh(bunny, color="#438fca", opacity=0.94, smooth_shading=True, ambient=0.30, diffuse=0.70, specular=0.24)
  plotter.camera_position = [(-0.17, 0.24, -0.30), (0.075, 0.025, 0.0), (0.0, 1.0, 0.0)]
  plotter.enable_parallel_projection()
  plotter.camera.parallel_scale = 0.12
  plotter.camera.clipping_range = (0.005, 2.0)
  if not off_screen:
    plotter.show(interactive_update=True, auto_close=False)
  return plotter, outer, bones, bunny


def save_surface_frame(path, frame, time_value, asset, hand_positions, affine, target_affine, bunny_positions=None):
  path = Path(path)
  values = {"frame": np.asarray(frame), "time": np.asarray(time_value), "dt": asset["dt"], "outer_skin_positions": hand_positions[asset["outer_surface_indices"]], "outer_skin_triangles": asset["outer_compact_triangles"], "bone_surface_positions": hand_positions[:int(asset["bone_vertex_count"])], "bone_surface_triangles": asset["bone_surface_triangles"], "piece_names": asset["piece_names"], "piece_vertex_offsets": asset["piece_vertex_offsets"], "affine": affine, "target_affine": target_affine, "length_unit": np.asarray("meter")}
  if bunny_positions is not None:
    values.update({"bunny_surface_positions": bunny_positions[asset["bunny_surface_indices"]], "bunny_surface_triangles": compact_triangles(asset["bunny_surface_indices"], asset["bunny_surface_triangles"], len(bunny_positions))})
  temporary = path.with_name(path.stem + ".temporary.npz")
  np.savez_compressed(temporary, **values)
  os.replace(temporary, path)


def encode_video(frame_directory, output_path, frame_count, fps):
  command = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(frame_directory) / "frame_%04d.jpg"), "-frames:v", str(frame_count), "-c:v", "libx264", "-threads", "8", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]
  subprocess.run(command, check=True)

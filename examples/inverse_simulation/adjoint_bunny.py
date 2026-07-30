"""
Differentiable bunny drop using YASPS matrices directly.

The forward step is the converged minimizer of

  Phi_k(x; x_k, v_k, theta) =
      inertia(x, x_k, v_k, mass)
    + h^2 elasticity(x, young)
    + floor_contact(x)
    + self_contact(x),

followed by v_{k+1} = (x_{k+1} - x_k) / h.  No scene energy registry or
scene.minimizeEnergy call is used.  Newton steps are built from explicit
Hessian objects.  After the complete trajectory has been recorded, the
discrete adjoint pass rebuilds each converged Hessian and uses mixed
second-order Jacobians for

  B_x = d residual / d previous_position
  B_v = d residual / d previous_velocity
  C   = d residual / d design_parameter.

Every one of those rectangular matrices retains the strict second-order
chain rule with two outer Jacobians, one inner Hessian, and the recursive
second-order term.  The loss is the final-frame mean squared position error
to one target bunny configuration.  The three supported design variables
are per-element Young/Poisson fields, initial translation, and per-vertex
mass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yasps-adjoint-matplotlib")

import numpy as np
import pycuda.gpuarray as gpuarray
import pyvista as pv
from scipy import sparse as scipy_sparse
from scipy.sparse.linalg import minres

from yasps import attribute, differentiator, scene, vector


SCRIPT_DIR = Path(__file__).resolve().parent
CCD_DIR = SCRIPT_DIR.parent / "ccd"
if str(CCD_DIR) not in sys.path:
  sys.path.append(str(CCD_DIR))
from ccd import CCD


DESIGN_YOUNG = "young"
DESIGN_INITIAL_POSITION = "initial-position"
DESIGN_MASS = "mass"
DESIGN_PARAMETERS = (
  DESIGN_YOUNG,
  DESIGN_INITIAL_POSITION,
  DESIGN_MASS,
)


@dataclass
class LinearizedContactState:
  connectivity: np.ndarray
  normals: np.ndarray
  weights: np.ndarray

  @property
  def count(self) -> int:
    return self.connectivity.shape[0]


@dataclass
class ContactState:
  floor_vertices: np.ndarray
  point_point: LinearizedContactState
  point_edge: LinearizedContactState
  point_triangle: LinearizedContactState
  edge_edge: LinearizedContactState

  @property
  def self_pair_count(self) -> int:
    return (
      self.point_point.count
      + self.point_edge.count
      + self.point_triangle.count
      + self.edge_edge.count
    )


@dataclass
class StepCheckpoint:
  previous_position: np.ndarray
  previous_velocity: np.ndarray
  position: np.ndarray
  velocity: np.ndarray
  contacts: ContactState
  newton_iterations: int
  residual_inf: float


@dataclass
class Trajectory:
  initial_position: np.ndarray
  initial_velocity: np.ndarray
  checkpoints: List[StepCheckpoint]
  state_loss_gradients: List[np.ndarray]
  loss: float
  final_center_offset: np.ndarray
  final_min_height: float
  initial_self_pairs: int
  maximum_self_pairs: int
  maximum_point_point_pairs: int
  maximum_point_edge_pairs: int
  maximum_point_triangle_pairs: int
  maximum_edge_edge_pairs: int
  frames_with_self_contact: int
  maximum_floor_contacts: int
  maximum_residual_inf: float
  mean_newton_iterations: float
  elapsed_seconds: float

  @property
  def center_distance(self) -> float:
    return float(np.linalg.norm(self.final_center_offset))

  @property
  def horizontal_center_distance(self) -> float:
    return float(
      np.linalg.norm(self.final_center_offset[[0, 2]])
    )


def extract_surface_triangles(tets: np.ndarray) -> np.ndarray:
  face_counts: Dict[Tuple[int, int, int], int] = {}
  for tet in tets:
    for local_face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
      face = tuple(sorted(int(tet[i]) for i in local_face))
      face_counts[face] = face_counts.get(face, 0) + 1
  return np.asarray(
    [face for face, count in face_counts.items() if count == 1],
    dtype=np.uint32
  )


def extract_surface_edges(triangles: np.ndarray) -> np.ndarray:
  edges = set()
  for a, b, c in triangles:
    edges.add(tuple(sorted((int(a), int(b)))))
    edges.add(tuple(sorted((int(b), int(c)))))
    edges.add(tuple(sorted((int(c), int(a)))))
  return np.asarray(sorted(edges), dtype=np.uint32)


def load_tetrahedral_bunny(
  node_path: Path,
  element_path: Path,
  height: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Load the repository TetGen bunny, normalize it, and orient every
  tetrahedron positively for the elastic energy.
  """
  with node_path.open("r", encoding="utf-8") as node_file:
    node_header = [int(value) for value in node_file.readline().split()]
  with element_path.open("r", encoding="utf-8") as element_file:
    element_header = [
      int(value) for value in element_file.readline().split()
    ]
  if len(node_header) < 2 or node_header[1] != 3:
    raise ValueError(f"{node_path} is not a three-dimensional .node file.")
  if len(element_header) < 2 or element_header[1] != 4:
    raise ValueError(
      f"{element_path} is not a tetrahedral .ele file."
    )

  node_data = np.loadtxt(node_path, skiprows=1, dtype=np.float64)
  element_data = np.loadtxt(
    element_path, skiprows=1, dtype=np.int64
  )
  if node_data.shape != (node_header[0], 4):
    raise ValueError(
      f"{node_path} contains {node_data.shape[0]} nodes with "
      f"{node_data.shape[1] - 1} coordinates; expected "
      f"{node_header[0]} nodes with 3 coordinates."
    )
  if element_data.shape != (element_header[0], 7):
    raise ValueError(
      f"{element_path} contains an unexpected element layout."
    )
  expected_node_ids = np.arange(1, node_header[0] + 1)
  expected_element_ids = np.arange(1, element_header[0] + 1)
  if not np.array_equal(node_data[:, 0].astype(np.int64), expected_node_ids):
    raise ValueError(f"{node_path} node IDs must be contiguous and 1-based.")
  if not np.array_equal(element_data[:, 0], expected_element_ids):
    raise ValueError(
      f"{element_path} element IDs must be contiguous and 1-based."
    )
  nodes = node_data[:, 1:4].copy()
  elements = (element_data[:, 3:7] - 1).astype(np.uint32)

  center_xz = nodes[:, (0, 2)].mean(axis=0)
  nodes[:, 0] -= center_xz[0]
  nodes[:, 2] -= center_xz[1]
  nodes[:, 1] -= nodes[:, 1].min()
  source_height = float(nodes[:, 1].max())
  if source_height <= 0.0:
    raise RuntimeError("The bunny mesh has zero height.")
  nodes *= height / source_height

  tet_positions = nodes[elements]
  signed_volumes = np.linalg.det(np.stack((
    tet_positions[:, 1] - tet_positions[:, 0],
    tet_positions[:, 2] - tet_positions[:, 0],
    tet_positions[:, 3] - tet_positions[:, 0],
  ), axis=2))
  inverted = signed_volumes < 0.0
  swapped = elements[inverted, 2].copy()
  elements[inverted, 2] = elements[inverted, 3]
  elements[inverted, 3] = swapped

  surface_triangles = extract_surface_triangles(elements)
  surface_edges = extract_surface_edges(surface_triangles)
  return nodes, elements, surface_triangles, surface_edges


def stable_neo_hookean(F, TB, young, poisson, dt):
  mu = young / (2.0 * (1.0 + poisson))
  lam = (
    young * poisson
    / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
  )
  IB = TB.transpose().inverse()
  volume = TB.transpose().determinant() / 6.0
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  density = (
    0.5 * mu * (IC - 3.0)
    - 0.5 * mu * I3.log()
    + 0.5
    * lam
    * (
      J - (1.0 + 0.75 * mu / lam)
    )
    * (
      J - (1.0 + 0.75 * mu / lam)
    )
  )
  return volume * density * dt * dt


def linearized_contact(position, normal, weights, width, d_hat, kappa):
  """
  IPC barrier along a lagged closest-feature normal.

  ``normal`` and ``weights`` are constants refreshed by collision detection.
  Consequently the signed separation is linear in position and the exact
  Hessian of this one-dimensional convex barrier is PSD.  They intentionally
  remain constants in all differentiations.
  """
  delta = position.row(0) * weights[0]
  for index in range(1, width):
    delta = delta + position.row(index) * weights[index]
  separation = delta.dot(normal)
  d = separation * separation
  ratio = d / d_hat
  return kappa * (d - d_hat) * (d - d_hat) * ratio.log() * ratio.log()


def exact_point_point_contact(position, d_hat, kappa):
  difference = position.row(1) - position.row(0)
  d = difference.dot(difference)
  ratio = d / d_hat
  return kappa * (d - d_hat) * (d - d_hat) * ratio.log() * ratio.log()


def exact_point_edge_contact(position, d_hat, kappa):
  point = position.row(0)
  edge0 = position.row(1)
  edge1 = position.row(2)
  cross = (edge0 - point).cross(edge1 - point)
  d = cross.dot(cross) / ((edge1 - edge0).dot(edge1 - edge0))
  ratio = d / d_hat
  return kappa * (d - d_hat) * (d - d_hat) * ratio.log() * ratio.log()


def exact_point_triangle_contact(position, d_hat, kappa):
  point = position.row(0)
  triangle0 = position.row(1)
  triangle1 = position.row(2)
  triangle2 = position.row(3)
  normal = (triangle1 - triangle0).cross(triangle2 - triangle0)
  projected = (point - triangle0).dot(normal)
  d = projected * projected / normal.dot(normal)
  ratio = d / d_hat
  return kappa * (d - d_hat) * (d - d_hat) * ratio.log() * ratio.log()


def exact_edge_edge_contact(position, d_hat, kappa):
  edge00 = position.row(0)
  edge01 = position.row(1)
  edge10 = position.row(2)
  edge11 = position.row(3)
  normal = (edge01 - edge00).cross(edge11 - edge10)
  projected = (edge10 - edge00).dot(normal)
  d = projected * projected / normal.dot(normal)
  ratio = d / d_hat
  return kappa * (d - d_hat) * (d - d_hat) * ratio.log() * ratio.log()


def floor_collision(position, floor_height, d_hat, kappa):
  distance = position[1] - floor_height
  d = distance * distance
  ratio = d / d_hat
  return kappa * (d - d_hat) * (d - d_hat) * ratio.log() * ratio.log()


def sum_matrices(matrices: Iterable):
  matrices = list(matrices)
  if len(matrices) == 0:
    raise ValueError("sum_matrices requires at least one matrix.")
  result = matrices[0]
  for current in matrices[1:]:
    result = result + current
  return result


def gpu_inf_norm(value: gpuarray.GPUArray) -> float:
  """
  Synchronize once and compute a robust infinity norm on the host.

  The Newton convergence check already requires a scalar synchronization.
  Computing the norm after that transfer keeps this small validation path
  straightforward and avoids an extra temporary GPU allocation.
  """
  host_value = value.get()
  if host_value.size == 0:
    return 0.0
  return float(np.max(np.abs(host_value)))


def hessian_to_scipy(hessian) -> scipy_sparse.csr_matrix:
  """
  Materialize YASPS's symmetric upper-block storage as a SciPy CSR matrix.

  The block coordinates are already in the matrix ordering selected during
  index generation.  Off-diagonal blocks are stored once; YASPS's CG kernel
  applies their transpose implicitly, which is reproduced here.
  """
  row_parts = []
  column_parts = []
  value_parts = []

  def append_storage(
    dimensions,
    counts,
    starts,
    positions_gpu,
    values_gpu,
  ):
    total_count = sum(int(count) for count in counts)
    if not dimensions or total_count == 0:
      return
    positions = positions_gpu.get()[:2 * total_count].reshape((-1, 2))
    values = values_gpu.get()
    position_start = 0
    for dimension_index, count in enumerate(counts):
      row_size = int(dimensions[2 * dimension_index])
      column_size = int(dimensions[2 * dimension_index + 1])
      count = int(count)
      if count == 0:
        continue
      position_end = position_start + count
      block_positions = positions[position_start:position_end]
      value_start = int(starts[dimension_index])
      value_end = value_start + count * row_size * column_size
      blocks = values[value_start:value_end].reshape(
        (count, row_size, column_size)
      )
      rows = (
        block_positions[:, 0, None, None]
        + np.arange(row_size, dtype=np.uint32)[None, :, None]
      )
      columns = (
        block_positions[:, 1, None, None]
        + np.arange(column_size, dtype=np.uint32)[None, None, :]
      )
      rows = np.broadcast_to(rows, blocks.shape).reshape(-1)
      columns = np.broadcast_to(columns, blocks.shape).reshape(-1)
      row_parts.append(rows)
      column_parts.append(columns)
      value_parts.append(blocks.reshape(-1))

      off_diagonal = block_positions[:, 0] != block_positions[:, 1]
      if np.any(off_diagonal):
        transpose_blocks = blocks[off_diagonal].transpose((0, 2, 1))
        transpose_positions = block_positions[off_diagonal]
        transpose_rows = (
          transpose_positions[:, 1, None, None]
          + np.arange(column_size, dtype=np.uint32)[None, :, None]
        )
        transpose_columns = (
          transpose_positions[:, 0, None, None]
          + np.arange(row_size, dtype=np.uint32)[None, None, :]
        )
        row_parts.append(
          np.broadcast_to(
            transpose_rows, transpose_blocks.shape
          ).reshape(-1)
        )
        column_parts.append(
          np.broadcast_to(
            transpose_columns, transpose_blocks.shape
          ).reshape(-1)
        )
        value_parts.append(transpose_blocks.reshape(-1))
      position_start = position_end

  append_storage(
    hessian.block_dimensions,
    hessian.block_counts,
    hessian.blocks_start_indices,
    hessian.block_positions,
    hessian.blocks_flattened,
  )
  append_storage(
    hessian.block_dimensions_dynamic,
    hessian.block_counts_dynamic,
    hessian.blocks_start_indices_dynamic,
    hessian.block_positions_dynamic,
    hessian.blocks_flattened_dynamic,
  )
  if not value_parts:
    return scipy_sparse.csr_matrix(
      (hessian.rows, hessian.cols), dtype=np.float64
    )
  return scipy_sparse.coo_matrix(
    (
      np.concatenate(value_parts),
      (np.concatenate(row_parts), np.concatenate(column_parts)),
    ),
    shape=(hessian.rows, hessian.cols),
  ).tocsr()


def solve_symmetric_indefinite(
  hessian,
  rhs: vector,
  tolerance: float,
  max_iterations: int,
) -> Tuple[vector, float, int]:
  """
  Solve an exact, potentially indefinite stationarity Hessian with MINRES.

  This is an example-side fallback for the backward system only.  Forward
  Newton continues to use YASPS's GPU CG solve on its PSD globalization.
  """
  matrix_cpu = hessian_to_scipy(hessian)
  rhs_cpu = rhs.value.get()
  solution_cpu, info = minres(
    matrix_cpu,
    rhs_cpu,
    rtol=tolerance,
    maxiter=max_iterations,
    show=False,
    check=False,
  )
  rhs_norm = max(np.linalg.norm(rhs_cpu), 1.0e-30)
  residual = rhs_cpu - matrix_cpu @ solution_cpu
  relative_residual = float(np.linalg.norm(residual) / rhs_norm)
  for _ in range(3):
    if relative_residual <= max(10.0 * tolerance, 1.0e-7):
      break
    correction, correction_info = minres(
      matrix_cpu,
      residual,
      rtol=tolerance,
      maxiter=max_iterations,
      show=False,
      check=False,
    )
    if correction_info != 0:
      info = correction_info
      break
    solution_cpu += correction
    residual = rhs_cpu - matrix_cpu @ solution_cpu
    relative_residual = float(np.linalg.norm(residual) / rhs_norm)
  if (
    info != 0
    or not np.all(np.isfinite(solution_cpu))
    or relative_residual > max(10.0 * tolerance, 1.0e-5)
  ):
    raise RuntimeError(
      "MINRES failed for the exact adjoint Hessian: "
      f"info={info}, relative_residual={relative_residual:.6e}."
    )
  result = vector(hessian.rows)
  result.updateValue(solution_cpu)
  return result, relative_residual, info


class AdjointBunnySimulation:
  def __init__(
    self,
    frames: int = 500,
    dt: float = 0.01,
    bunny_height: float = 0.35,
    d_hat: float = 1.0e-6,
    kappa: float = 5.0e3,
    young: float = 1.0e4,
    poisson: float = 0.32,
    mass: float = 4.0e-5,
    velocity_retention: float = 0.90,
    initial_translation: np.ndarray = np.array(
      [-0.28, 0.15, -0.20], dtype=np.float64
    ),
    target_position: Optional[np.ndarray] = None,
    node_path: Optional[Path] = None,
    element_path: Optional[Path] = None,
    newton_tolerance: float = 1.0e-12,
    contact_newton_tolerance: float = 2.0e-7,
    position_tolerance: float = 1.0e-12,
    max_newton_iterations: int = 300,
    linear_tolerance: float = 1.0e-10,
    max_linear_iterations: int = 2000,
    verbose: bool = True
  ):
    if frames <= 0:
      raise ValueError("frames must be positive.")
    if dt <= 0.0:
      raise ValueError("dt must be positive.")
    if d_hat <= 0.0:
      raise ValueError("d_hat must be positive.")
    if newton_tolerance <= 0.0 or contact_newton_tolerance <= 0.0:
      raise ValueError("Newton tolerances must be positive.")
    if max_newton_iterations <= 0:
      raise ValueError("max_newton_iterations must be positive.")
    if young <= 0.0 or mass <= 0.0:
      raise ValueError("young and mass must be positive.")
    if not 0.0 <= velocity_retention <= 1.0:
      raise ValueError("velocity_retention must lie in [0, 1].")
    if not 0.0 < poisson < 0.49:
      raise ValueError("poisson must lie in (0, 0.49).")
    self.frames = int(frames)
    self.dt_value = float(dt)
    self.d_hat_value = float(d_hat)
    self.contact_distance = math.sqrt(self.d_hat_value)
    self.kappa_value = float(kappa)
    self.velocity_retention_value = float(velocity_retention)
    self.floor_height_value = 0.0
    self.newton_tolerance = float(newton_tolerance)
    self.contact_newton_tolerance = float(contact_newton_tolerance)
    self.position_tolerance = float(position_tolerance)
    self.max_newton_iterations = int(max_newton_iterations)
    self.linear_tolerance = float(linear_tolerance)
    self.max_linear_iterations = int(max_linear_iterations)
    self.verbose = bool(verbose)

    (
      self.rest_positions,
      self.tet_indices,
      self.surface_triangles,
      self.surface_edges,
    ) = load_tetrahedral_bunny(
      node_path or SCRIPT_DIR.parent / "data" / "bunny.node",
      element_path or SCRIPT_DIR.parent / "data" / "bunny.ele",
      bunny_height
    )
    self.num_vertices = self.rest_positions.shape[0]
    self.num_tets = self.tet_indices.shape[0]
    self.num_dofs = 3 * self.num_vertices
    self.default_initial_translation = np.asarray(
      initial_translation, dtype=np.float64
    ).copy()
    if self.default_initial_translation.shape != (3,):
      raise ValueError("initial_translation must contain three values.")
    self.initial_translation = self.default_initial_translation.copy()
    if target_position is None:
      target_position = np.asarray(
        [0.0, self.contact_distance, 0.0],
        dtype=np.float64,
      )
    self.target_position = np.asarray(
      target_position, dtype=np.float64
    ).copy()
    if self.target_position.shape != (3,):
      raise ValueError("target_position must contain three values.")
    self.target_configuration = (
      self.rest_positions + self.target_position
    )
    self.target_center = self.target_configuration.mean(axis=0)

    self.default_young = np.full(
      self.num_tets, float(young), dtype=np.float64
    )
    self.default_poisson = np.full(
      self.num_tets, float(poisson), dtype=np.float64
    )
    self.default_vertex_mass = np.full(
      self.num_vertices,
      float(mass),
      dtype=np.float64,
    )

    self._build_scene()
    self._build_collision_detector()
    self._build_derivatives()

    if self.verbose:
      print(
        "adjoint bunny mesh: "
        f"{self.num_vertices} vertices, {self.tet_indices.shape[0]} tets, "
        f"{self.surface_triangles.shape[0]} surface triangles"
      )

  def _constant(self, owner, name: str, value: float):
    result = owner.addConstant(name, rows=1, cols=1)
    result.updateValue(np.asarray([value], dtype=np.float64))
    return result

  def _build_scene(self):
    self.model = scene("adjoint_bunny_drop")
    self.dt = self._constant(self.model, "dt", self.dt_value)
    self.d_hat = self._constant(
      self.model, "d_hat", self.d_hat_value
    )
    self.kappa = self._constant(
      self.model, "kappa", self.kappa_value
    )
    self.velocity_retention = self._constant(
      self.model,
      "velocity_retention",
      self.velocity_retention_value,
    )
    self.floor_height = self._constant(
      self.model, "floor_height", self.floor_height_value
    )

    self.bunny = self.model.addMesh("bunny")

    self.vertices = self.bunny.addPrimitive(
      "vertices", numInstances=self.num_vertices
    )
    self.position = self.vertices.addAttribute(
      "position", rows=3, cols=1
    )
    self.previous_position = self.vertices.addConstant(
      "previous_position", rows=3, cols=1
    )
    self.previous_velocity = self.vertices.addConstant(
      "previous_velocity", rows=3, cols=1
    )
    self.rest_position = self.vertices.addConstant(
      "rest_position", rows=3, cols=1
    )
    self.rest_position.updateValue(self.rest_positions.reshape(-1))
    self.vertex_mass = self.vertices.addConstant(
      "mass", rows=1, cols=1
    )
    self.vertex_mass.updateValue(self.default_vertex_mass)

    self.tets = self.bunny.addPrimitive(
      "tets", numInstances=self.tet_indices.shape[0]
    )
    self.young = self.tets.addConstant("young", rows=1, cols=1)
    self.young.updateValue(self.default_young)
    self.poisson = self.tets.addConstant("poisson", rows=1, cols=1)
    self.poisson.updateValue(self.default_poisson)
    self.tet_to_vertices = self.tets.addConnectivity(
      "tet_to_vertices",
      self.vertices,
      self.tet_indices,
      4
    )
    tet_positions = self.tets.addAttribute(
      "positions",
      through=self.tet_to_vertices,
      source=self.position
    )
    tet_rest_positions = self.tets.addAttribute(
      "rest_positions",
      through=self.tet_to_vertices,
      source=self.rest_position
    )

    rest0 = tet_rest_positions.row(0)
    rest1 = tet_rest_positions.row(1)
    rest2 = tet_rest_positions.row(2)
    rest3 = tet_rest_positions.row(3)
    rest_x0 = rest1 - rest0
    rest_x1 = rest2 - rest0
    rest_x2 = rest3 - rest0
    TB = self.tets.addAttribute(
      "TB",
      computed_attribute=attribute.to_array(
        [
          rest_x0[0], rest_x0[1], rest_x0[2],
          rest_x1[0], rest_x1[1], rest_x1[2],
          rest_x2[0], rest_x2[1], rest_x2[2],
        ],
        rows=3,
        cols=3
      )
    )

    current0 = tet_positions.row(0)
    current1 = tet_positions.row(1)
    current2 = tet_positions.row(2)
    current3 = tet_positions.row(3)
    current_x0 = current1 - current0
    current_x1 = current2 - current0
    current_x2 = current3 - current0
    F = self.tets.addAttribute(
      "F",
      computed_attribute=attribute.to_array(
        [
          current_x0[0], current_x0[1], current_x0[2],
          current_x1[0], current_x1[1], current_x1[2],
          current_x2[0], current_x2[1], current_x2[2],
        ],
        rows=3,
        cols=3
      )
    )

    self.elastic_energy = self.tets.addAttribute(
      "elastic_energy",
      computed_attribute=stable_neo_hookean(
        F, TB, self.young, self.poisson, self.dt
      )
    )
    # A distinct source name keeps the exact converged residual Hessian
    # separate from the PSD-projected matrix used only to globalize Newton.
    self.elastic_energy_exact = self.tets.addAttribute(
      "elastic_energy_exact",
      computed_attribute=stable_neo_hookean(
        F, TB, self.young, self.poisson, self.dt
      )
    )

    gravity_step = attribute.to_array(
      [0.0, -9.81 * self.dt_value * self.dt_value, 0.0],
      rows=3,
      cols=1
    )
    predicted_position = (
      self.previous_position
      + self.previous_velocity * self.velocity_retention * self.dt
      + gravity_step
    )
    displacement = self.position - predicted_position
    self.inertia_energy = self.vertices.addAttribute(
      "inertia_energy",
      computed_attribute=(
        0.5 * self.vertex_mass * displacement.dot(displacement)
      )
    )

    self.floor_contacts = self.bunny.addPrimitive(
      "floor_contacts", numInstances=0, isDynamic=True
    )
    self.floor_to_vertices = self.floor_contacts.addConnectivity(
      "floor_to_vertices", self.vertices, [], 1
    )
    floor_positions = self.floor_contacts.addAttribute(
      "positions",
      through=self.floor_to_vertices,
      source=self.position
    )
    self.floor_energy = self.floor_contacts.addAttribute(
      "floor_energy",
      computed_attribute=floor_collision(
        floor_positions,
        self.floor_height,
        self.d_hat,
        self.kappa
      )
    )

    self.point_point_contacts = self.bunny.addPrimitive(
      "point_point_contacts", numInstances=0, isDynamic=True
    )
    self.point_edge_contacts = self.bunny.addPrimitive(
      "point_edge_contacts", numInstances=0, isDynamic=True
    )
    self.point_triangle_contacts = self.bunny.addPrimitive(
      "point_triangle_contacts", numInstances=0, isDynamic=True
    )
    self.edge_edge_contacts = self.bunny.addPrimitive(
      "edge_edge_contacts", numInstances=0, isDynamic=True
    )
    self.pp_to_vertices = self.point_point_contacts.addConnectivity(
      "point_point_to_vertices", self.vertices, [], 2
    )
    self.pe_to_vertices = self.point_edge_contacts.addConnectivity(
      "point_edge_to_vertices", self.vertices, [], 3
    )
    self.pt_to_vertices = self.point_triangle_contacts.addConnectivity(
      "point_triangle_to_vertices", self.vertices, [], 4
    )
    self.ee_to_vertices = self.edge_edge_contacts.addConnectivity(
      "edge_edge_to_vertices", self.vertices, [], 4
    )

    pp_positions = self.point_point_contacts.addAttribute(
      "positions", through=self.pp_to_vertices, source=self.position
    )
    pe_positions = self.point_edge_contacts.addAttribute(
      "positions", through=self.pe_to_vertices, source=self.position
    )
    pt_positions = self.point_triangle_contacts.addAttribute(
      "positions", through=self.pt_to_vertices, source=self.position
    )
    ee_positions = self.edge_edge_contacts.addAttribute(
      "positions", through=self.ee_to_vertices, source=self.position
    )
    self.pp_normal = self.point_point_contacts.addConstant(
      "normal", rows=3, cols=1
    )
    self.pe_normal = self.point_edge_contacts.addConstant(
      "normal", rows=3, cols=1
    )
    self.pt_normal = self.point_triangle_contacts.addConstant(
      "normal", rows=3, cols=1
    )
    self.ee_normal = self.edge_edge_contacts.addConstant(
      "normal", rows=3, cols=1
    )
    self.pp_weights = self.point_point_contacts.addConstant(
      "weights", rows=2, cols=1
    )
    self.pe_weights = self.point_edge_contacts.addConstant(
      "weights", rows=3, cols=1
    )
    self.pt_weights = self.point_triangle_contacts.addConstant(
      "weights", rows=4, cols=1
    )
    self.ee_weights = self.edge_edge_contacts.addConstant(
      "weights", rows=4, cols=1
    )
    self.point_point_energy = self.point_point_contacts.addAttribute(
      "point_point_energy",
      computed_attribute=linearized_contact(
        pp_positions,
        self.pp_normal,
        self.pp_weights,
        2,
        self.d_hat,
        self.kappa
      )
    )
    self.point_edge_energy = self.point_edge_contacts.addAttribute(
      "point_edge_energy",
      computed_attribute=linearized_contact(
        pe_positions,
        self.pe_normal,
        self.pe_weights,
        3,
        self.d_hat,
        self.kappa
      )
    )
    self.point_triangle_energy = (
      self.point_triangle_contacts.addAttribute(
        "point_triangle_energy",
        computed_attribute=linearized_contact(
          pt_positions,
          self.pt_normal,
          self.pt_weights,
          4,
          self.d_hat,
          self.kappa
        )
      )
    )
    self.edge_edge_energy = self.edge_edge_contacts.addAttribute(
      "edge_edge_energy",
      computed_attribute=linearized_contact(
        ee_positions,
        self.ee_normal,
        self.ee_weights,
        4,
        self.d_hat,
        self.kappa
      )
    )
    self.point_point_energy_exact = (
      self.point_point_contacts.addAttribute(
        "point_point_energy_exact",
        computed_attribute=exact_point_point_contact(
          pp_positions, self.d_hat, self.kappa
        )
      )
    )
    self.point_edge_energy_exact = (
      self.point_edge_contacts.addAttribute(
        "point_edge_energy_exact",
        computed_attribute=exact_point_edge_contact(
          pe_positions, self.d_hat, self.kappa
        )
      )
    )
    self.point_triangle_energy_exact = (
      self.point_triangle_contacts.addAttribute(
        "point_triangle_energy_exact",
        computed_attribute=exact_point_triangle_contact(
          pt_positions, self.d_hat, self.kappa
        )
      )
    )
    self.edge_edge_energy_exact = (
      self.edge_edge_contacts.addAttribute(
        "edge_edge_energy_exact",
        computed_attribute=exact_edge_edge_contact(
          ee_positions, self.d_hat, self.kappa
        )
      )
    )

    self.static_energies = [
      self.inertia_energy,
      self.elastic_energy,
    ]
    self.dynamic_energies = [
      self.floor_energy,
      self.point_point_energy,
      self.point_edge_energy,
      self.point_triangle_energy,
      self.edge_edge_energy,
    ]
    self.exact_dynamic_energies = [
      self.floor_energy,
      self.point_point_energy_exact,
      self.point_edge_energy_exact,
      self.point_triangle_energy_exact,
      self.edge_edge_energy_exact,
    ]

  def _build_collision_detector(self):
    surface_indices = np.unique(
      self.surface_triangles.reshape(-1)
    ).astype(np.uint32)
    broad_phase_capacity = max(
      25_000_000,
      512 * (
        self.surface_triangles.shape[0]
        + self.surface_edges.shape[0]
      ),
    )
    self.ccd = CCD(
      surface_indices.size,
      self.num_vertices,
      max_cd_pairs=1_000_000,
      max_ccd_pairs=broad_phase_capacity,
      mesh_indices=np.zeros(self.num_vertices, dtype=np.uint32).tolist()
    )
    position_gpu = gpuarray.to_gpu(
      self.rest_positions.astype(np.float64).reshape(-1)
    )
    self.ccd.init_faces(
      position_gpu,
      gpuarray.to_gpu(self.surface_triangles.reshape(-1)),
      gpuarray.to_gpu(surface_indices),
      self.surface_triangles.shape[0]
    )
    self.ccd.init_edges(
      position_gpu,
      position_gpu,
      gpuarray.to_gpu(self.surface_edges.reshape(-1)),
      self.surface_edges.shape[0]
    )

  def _differentiate_hessian(
    self,
    energy,
    projection_method: int,
    dynamic_instances: bool
  ):
    return differentiator().diff2(
      [energy],
      [self.position],
      [self.position],
      projection_method=projection_method,
      dynamic_instances=dynamic_instances
    )

  def _build_derivatives(self):
    # Newton is globalized with a PSD elastic Hessian and lagged-normal
    # contact Hessians.  The backward pass differentiates the converged
    # stationarity condition itself, so it uses the exact elastic/contact
    # Hessian.  This is A = dr/d(position), not the matrix from an
    # intermediate Newton iteration.
    forward_terms = [
      self._differentiate_hessian(
        self.inertia_energy, -1, False
      ),
      self._differentiate_hessian(
        self.elastic_energy, 2, False
      ),
    ]
    forward_terms += [
      self._differentiate_hessian(energy, -1, True)
      for energy in self.dynamic_energies
    ]
    self.forward_system = sum_matrices(forward_terms)

    adjoint_terms = [
      self._differentiate_hessian(
        self.inertia_energy, -1, False
      ),
      self._differentiate_hessian(
        self.elastic_energy_exact, -1, False
      ),
    ]
    adjoint_terms += [
      self._differentiate_hessian(energy, -1, True)
      for energy in self.exact_dynamic_energies
    ]
    self.adjoint_system = sum_matrices(adjoint_terms)

    # Each mixed matrix below is assembled by secondOrderJacobian with
    #
    #   J_position^T H_inner J_column + H_recursive.
    #
    # In particular, previous_position and previous_velocity are separate
    # column targets.  They are not collapsed into a first-order derivative
    # or inferred from a Hessian block.
    self.previous_position_jacobian = differentiator().diff2(
      [self.inertia_energy],
      [self.position],
      [self.previous_position],
      compress_coordinates=False
    )
    self.previous_velocity_jacobian = differentiator().diff2(
      [self.inertia_energy],
      [self.position],
      [self.previous_velocity],
      compress_coordinates=False
    )
    self.material_jacobian = differentiator().diff2(
      [self.elastic_energy_exact],
      [self.position],
      [self.young, self.poisson],
      compress_coordinates=False
    )
    self.mass_jacobian = differentiator().diff2(
      [self.inertia_energy],
      [self.position],
      [self.vertex_mass],
      compress_coordinates=False
    )

  def restore_default_design(self):
    self.young.updateValue(self.default_young)
    self.poisson.updateValue(self.default_poisson)
    self.vertex_mass.updateValue(self.default_vertex_mass)
    self.initial_translation = (
      self.default_initial_translation.copy()
    )

  def prepare_design(self, design: str):
    """
    Reset every control before an independent optimization.

    Every first optimization trajectory starts offset from the world origin.
    Each design then minimizes the same final-configuration position loss.
    """
    self.restore_default_design()

  def get_design_value(self, design: str):
    if design == DESIGN_YOUNG:
      return {
        "young": self.young.value.get().copy(),
        "poisson": self.poisson.value.get().copy(),
      }
    if design == DESIGN_MASS:
      return self.vertex_mass.value.get().copy()
    if design == DESIGN_INITIAL_POSITION:
      return self.initial_translation.copy()
    raise ValueError(f"Unknown design parameter: {design}")

  def set_design_value(self, design: str, value):
    if design == DESIGN_YOUNG:
      if not isinstance(value, dict):
        raise TypeError(
          "Material design value must contain young and poisson fields."
        )
      young = np.asarray(value["young"], dtype=np.float64)
      poisson = np.asarray(value["poisson"], dtype=np.float64)
      if young.shape != (self.num_tets,):
        raise ValueError("Young field must contain one value per element.")
      if poisson.shape != (self.num_tets,):
        raise ValueError("Poisson field must contain one value per element.")
      if np.any(young <= 0.0):
        raise ValueError("Every Young's modulus must be positive.")
      if np.any(poisson <= 0.0) or np.any(poisson >= 0.49):
        raise ValueError("Every Poisson ratio must lie in (0, 0.49).")
      self.young.updateValue(young)
      self.poisson.updateValue(poisson)
      return
    if design == DESIGN_MASS:
      mass = np.asarray(value, dtype=np.float64)
      if mass.shape != (self.num_vertices,):
        raise ValueError("Mass field must contain one value per vertex.")
      if np.any(mass <= 0.0):
        raise ValueError("Every vertex mass must be positive.")
      self.vertex_mass.updateValue(mass)
      return
    if design == DESIGN_INITIAL_POSITION:
      numeric = np.asarray(value, dtype=np.float64)
      if numeric.shape != (3,):
        raise ValueError("Initial translation must contain three values.")
      self.initial_translation = numeric.copy()
      return
    raise ValueError(f"Unknown design parameter: {design}")

  def _empty_pairs(self, width: int) -> np.ndarray:
    return np.empty((0, width), dtype=np.uint32)

  def _copy_ccd_pairs(
    self, values, count: int, width: int
  ) -> np.ndarray:
    if count == 0:
      return self._empty_pairs(width)
    return (
      values[:count * width]
      .get()
      .astype(np.uint32)
      .reshape((-1, width))
    )

  def _make_linearized_contact(
    self,
    connectivity: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray
  ) -> LinearizedContactState:
    width = connectivity.shape[1]
    if connectivity.shape[0] == 0:
      return LinearizedContactState(
        connectivity=connectivity,
        normals=np.empty((0, 3), dtype=np.float64),
        weights=np.empty((0, width), dtype=np.float64)
      )
    deltas = np.einsum(
      "ij,ijk->ik", weights, positions[connectivity]
    )
    lengths = np.linalg.norm(deltas, axis=1)
    normals = np.zeros_like(deltas)
    regular = lengths > 1.0e-12
    normals[regular] = deltas[regular] / lengths[regular, None]
    normals[~regular, 1] = 1.0
    return LinearizedContactState(
      connectivity=connectivity,
      normals=normals,
      weights=weights
    )

  def _linearize_point_point(
    self, connectivity: np.ndarray, positions: np.ndarray
  ) -> LinearizedContactState:
    weights = np.tile(
      np.asarray([1.0, -1.0]), (connectivity.shape[0], 1)
    )
    return self._make_linearized_contact(
      connectivity, positions, weights
    )

  def _linearize_point_edge(
    self, connectivity: np.ndarray, positions: np.ndarray
  ) -> LinearizedContactState:
    weights = np.empty(
      (connectivity.shape[0], 3), dtype=np.float64
    )
    for index, (point_index, edge0_index, edge1_index) in enumerate(
      connectivity
    ):
      point = positions[point_index]
      edge0 = positions[edge0_index]
      edge = positions[edge1_index] - edge0
      denominator = float(np.dot(edge, edge))
      parameter = (
        float(np.dot(point - edge0, edge)) / denominator
        if denominator > 1.0e-24
        else 0.0
      )
      parameter = float(np.clip(parameter, 0.0, 1.0))
      weights[index] = [
        1.0, -(1.0 - parameter), -parameter
      ]
    return self._make_linearized_contact(
      connectivity, positions, weights
    )

  def _linearize_point_triangle(
    self, connectivity: np.ndarray, positions: np.ndarray
  ) -> LinearizedContactState:
    weights = np.empty(
      (connectivity.shape[0], 4), dtype=np.float64
    )
    for index, indices in enumerate(connectivity):
      point, triangle0, triangle1, triangle2 = positions[indices]
      edge0 = triangle1 - triangle0
      edge1 = triangle2 - triangle0
      relative = point - triangle0
      dot00 = float(np.dot(edge0, edge0))
      dot01 = float(np.dot(edge0, edge1))
      dot11 = float(np.dot(edge1, edge1))
      dot20 = float(np.dot(relative, edge0))
      dot21 = float(np.dot(relative, edge1))
      denominator = dot00 * dot11 - dot01 * dot01
      if abs(denominator) > 1.0e-24:
        bary1 = (dot11 * dot20 - dot01 * dot21) / denominator
        bary2 = (dot00 * dot21 - dot01 * dot20) / denominator
        barycentric = np.asarray(
          [1.0 - bary1 - bary2, bary1, bary2],
          dtype=np.float64
        )
        barycentric = np.maximum(barycentric, 0.0)
        bary_sum = float(barycentric.sum())
        if bary_sum > 1.0e-24:
          barycentric /= bary_sum
        else:
          barycentric[:] = [1.0, 0.0, 0.0]
      else:
        barycentric = np.asarray(
          [1.0, 0.0, 0.0], dtype=np.float64
        )
      weights[index, 0] = 1.0
      weights[index, 1:] = -barycentric
    return self._make_linearized_contact(
      connectivity, positions, weights
    )

  def _linearize_edge_edge(
    self, connectivity: np.ndarray, positions: np.ndarray
  ) -> LinearizedContactState:
    weights = np.empty(
      (connectivity.shape[0], 4), dtype=np.float64
    )
    for index, indices in enumerate(connectivity):
      edge00, edge01, edge10, edge11 = positions[indices]
      first = edge01 - edge00
      second = edge11 - edge10
      relative = edge00 - edge10
      first_squared = float(np.dot(first, first))
      second_squared = float(np.dot(second, second))
      cross_dot = float(np.dot(first, second))
      first_relative = float(np.dot(first, relative))
      second_relative = float(np.dot(second, relative))
      denominator = (
        first_squared * second_squared - cross_dot * cross_dot
      )
      if abs(denominator) > 1.0e-24:
        first_parameter = (
          cross_dot * second_relative
          - second_squared * first_relative
        ) / denominator
        second_parameter = (
          first_squared * second_relative
          - cross_dot * first_relative
        ) / denominator
      else:
        first_parameter = 0.0
        second_parameter = (
          second_relative / second_squared
          if second_squared > 1.0e-24
          else 0.0
        )
      first_parameter = float(np.clip(first_parameter, 0.0, 1.0))
      second_parameter = float(np.clip(second_parameter, 0.0, 1.0))
      weights[index] = [
        1.0 - first_parameter,
        first_parameter,
        -(1.0 - second_parameter),
        -second_parameter,
      ]
    return self._make_linearized_contact(
      connectivity, positions, weights
    )

  def detect_contacts(self) -> ContactState:
    positions = self.position.value.get().reshape((-1, 3))
    floor_vertices = np.flatnonzero(
      positions[:, 1] <= self.contact_distance
    ).astype(np.uint32)

    self.ccd.cd(self.position.value, self.d_hat_value)
    pp_count, pe_count, pt_count, ee_count = (
      int(x) for x in self.ccd.separated_counts
    )
    point_point = self._copy_ccd_pairs(self.ccd.pp, pp_count, 2)
    point_edge = self._copy_ccd_pairs(self.ccd.pe, pe_count, 3)
    point_triangle = self._copy_ccd_pairs(
      self.ccd.pt, pt_count, 4
    )
    edge_edge = self._copy_ccd_pairs(self.ccd.ee, ee_count, 4)
    return ContactState(
      floor_vertices=floor_vertices,
      point_point=self._linearize_point_point(
        point_point, positions
      ),
      point_edge=self._linearize_point_edge(point_edge, positions),
      point_triangle=self._linearize_point_triangle(
        point_triangle, positions
      ),
      edge_edge=self._linearize_edge_edge(edge_edge, positions),
    )

  def _set_dynamic_connectivity(
    self, primitive, connectivity, values: np.ndarray
  ):
    primitive.updateNumInstances(values.shape[0])
    if values.shape[0] > 0:
      connectivity.updateConnectivity(values)

  def _set_dynamic_contact(
    self,
    primitive,
    connectivity,
    normal_attribute,
    weight_attribute,
    state: LinearizedContactState
  ):
    primitive.updateNumInstances(state.count)
    if state.count > 0:
      connectivity.updateConnectivity(state.connectivity)
      normal_attribute.updateValue(state.normals.reshape(-1))
      weight_attribute.updateValue(state.weights.reshape(-1))

  def apply_contacts(self, contacts: ContactState):
    self._set_dynamic_connectivity(
      self.floor_contacts,
      self.floor_to_vertices,
      contacts.floor_vertices.reshape((-1, 1))
    )
    self._set_dynamic_contact(
      self.point_point_contacts,
      self.pp_to_vertices,
      self.pp_normal,
      self.pp_weights,
      contacts.point_point
    )
    self._set_dynamic_contact(
      self.point_edge_contacts,
      self.pe_to_vertices,
      self.pe_normal,
      self.pe_weights,
      contacts.point_edge
    )
    self._set_dynamic_contact(
      self.point_triangle_contacts,
      self.pt_to_vertices,
      self.pt_normal,
      self.pt_weights,
      contacts.point_triangle
    )
    self._set_dynamic_contact(
      self.edge_edge_contacts,
      self.ee_to_vertices,
      self.ee_normal,
      self.ee_weights,
      contacts.edge_edge
    )

  def update_contacts(self) -> ContactState:
    contacts = self.detect_contacts()
    self.apply_contacts(contacts)
    return contacts

  def total_energy(self) -> float:
    result = 0.0
    for energy in self.static_energies:
      result += float(gpuarray.sum(energy.compute().value).get())
    dynamic_primitives = [
      self.floor_contacts,
      self.point_point_contacts,
      self.point_edge_contacts,
      self.point_triangle_contacts,
      self.edge_edge_contacts,
    ]
    for primitive, energy in zip(
      dynamic_primitives, self.dynamic_energies
    ):
      if primitive.numInstances > 0:
        result += float(gpuarray.sum(energy.compute().value).get())
    return result

  def _floor_step_limit(
    self, positions: np.ndarray, direction: np.ndarray
  ) -> float:
    clearance = positions[:, 1] - self.floor_height_value
    downward_step = direction[:, 1]
    moving_down = downward_step > 0.0
    if not np.any(moving_down):
      return 1.0
    # Keep ten percent of the current positive clearance.  The old
    # contact-distance-based cap incorrectly made 0.8 * d_hat**0.5 a hard
    # floor, even though it is only the barrier activation distance.
    denominator = downward_step[moving_down]
    numer = 0.9 * clearance[moving_down]
    valid = (denominator > 1.0e-14) & (numer > 0.0)
    if not np.any(valid):
      return 1.0
    steps = numer[valid] / denominator[valid]
    positive = steps[steps > 0.0]
    if positive.size == 0:
      return 1.0
    return float(np.clip(positive.min(), 0.0, 1.0))

  def _collision_step_limit(self, direction) -> float:
    current = self.position.value.copy()
    self.ccd.ccd(current, self.d_hat_value, direction, 1.0)
    return float(
      self.ccd.compute_largest_step_size(
        0.8, current, direction
      )
    )

  def _line_search(self, step: vector) -> Tuple[float, float]:
    current_gpu = self.position.value.copy()
    current = current_gpu.get().reshape((-1, 3))
    direction = step.value.get().reshape((-1, 3))
    base_energy = self.total_energy()

    floor_limit = self._floor_step_limit(current, direction)
    collision_limit = self._collision_step_limit(step.value)
    alpha = min(1.0, floor_limit, collision_limit)
    initial_alpha = alpha
    accepted_energy = base_energy
    accepted = False
    candidate_energy = math.inf
    for _ in range(12):
      self.position.updateValue(
        current_gpu - step.value * alpha,
        deepCopy=True
      )
      self.update_contacts()
      candidate_energy = self.total_energy()
      if np.isfinite(candidate_energy) and (
        candidate_energy <= base_energy + 1.0e-12
      ):
        accepted_energy = candidate_energy
        accepted = True
        break
      alpha *= 0.5

    if not accepted:
      self.position.updateValue(current_gpu, deepCopy=True)
      self.update_contacts()
      alpha = 0.0
    self._last_line_search = {
      "accepted": accepted,
      "alpha": alpha,
      "initial_alpha": initial_alpha,
      "floor_limit": floor_limit,
      "collision_limit": collision_limit,
      "base_energy": base_energy,
      "candidate_energy": candidate_energy,
    }
    return alpha, accepted_energy

  def _solve_frame(self, frame: int) -> StepCheckpoint:
    previous = self.position.value.get().reshape((-1, 3)).copy()
    previous_velocity = (
      self.previous_velocity.value.get().reshape((-1, 3)).copy()
    )
    self.previous_position.updateValue(previous.reshape(-1))
    residual_inf = math.inf
    residual_history: List[float] = []
    iterations = 0
    converged = False
    contacts = self.update_contacts()

    for iteration in range(self.max_newton_iterations):
      contacts = self.update_contacts()
      self.forward_system.compute()
      residual_inf = gpu_inf_norm(self.forward_system.gradient.value)
      residual_history.append(residual_inf)
      iterations = iteration + 1
      convergence_tolerance = (
        max(self.newton_tolerance, self.contact_newton_tolerance)
        if contacts.floor_vertices.size > 0
        else self.newton_tolerance
      )
      if residual_inf <= convergence_tolerance:
        converged = True
        break

      step = self.forward_system.solve(
        tolerance=self.linear_tolerance,
        maxIterations=self.max_linear_iterations,
        recompute=False,
        zero_initial_guess=True
      )
      if self.forward_system.last_solve_error_code < 0:
        raise RuntimeError(
          "Forward Hessian solve did not converge at frame "
          f"{frame}, Newton iteration {iteration}."
        )
      step_inf = gpu_inf_norm(step.value)
      if not np.isfinite(step_inf):
        raise RuntimeError(
          f"Newton step became non-finite at frame {frame}."
        )
      alpha, _ = self._line_search(step)
      if alpha == 0.0 or alpha * step_inf <= self.position_tolerance:
        contacts = self.update_contacts()
        self.forward_system.compute()
        residual_inf = gpu_inf_norm(
          self.forward_system.gradient.value
        )
        convergence_tolerance = (
          max(self.newton_tolerance, self.contact_newton_tolerance)
          if contacts.floor_vertices.size > 0
          else self.newton_tolerance
        )
        converged = residual_inf <= convergence_tolerance
        break

    if not converged:
      line_search = getattr(self, "_last_line_search", {})
      raise RuntimeError(
        f"Frame {frame} did not converge: residual_inf="
        f"{residual_inf:.6e}, tolerance={convergence_tolerance:.6e}, "
        f"iterations={iterations}, recent_residuals="
        f"{residual_history[-10:]}, contacts="
        f"{{'floor': {contacts.floor_vertices.size}, "
        f"'point_point': {contacts.point_point.count}, "
        f"'point_edge': {contacts.point_edge.count}, "
        f"'point_triangle': {contacts.point_triangle.count}, "
        f"'edge_edge': {contacts.edge_edge.count}}}, "
        f"line_search={line_search}."
      )
    if (
      self.verbose
      and (
        frame == 0
        or frame + 1 == self.frames
        or (frame + 1) % max(1, self.frames // 10) == 0
      )
    ):
      print(
        f"forward frame {frame + 1:04d}/{self.frames}: "
        f"Newton={iterations:02d}, residual_inf={residual_inf:.3e}, "
        f"floor={contacts.floor_vertices.size}, "
        f"self={contacts.self_pair_count}"
      )

    current = self.position.value.get().reshape((-1, 3)).copy()
    current_velocity = (current - previous) / self.dt_value
    self.previous_velocity.updateValue(current_velocity.reshape(-1))
    return StepCheckpoint(
      previous_position=previous,
      previous_velocity=previous_velocity,
      position=current,
      velocity=current_velocity,
      contacts=contacts,
      newton_iterations=iterations,
      residual_inf=residual_inf
    )

  def _trajectory_loss(
    self, positions: List[np.ndarray]
  ) -> Tuple[float, List[np.ndarray], np.ndarray]:
    gradients = [np.zeros_like(position) for position in positions]
    offset = positions[-1] - self.target_configuration
    gradients[-1] = 2.0 * offset / float(self.num_vertices)
    return (
      float(np.mean(np.sum(offset * offset, axis=1))),
      gradients,
      offset.mean(axis=0),
    )

  def forward(self) -> Trajectory:
    start = time.perf_counter()
    initial = self.rest_positions + self.initial_translation
    initial_velocity = np.zeros_like(initial)
    self.position.updateValue(initial.reshape(-1))
    self.previous_position.updateValue(initial.reshape(-1))
    self.previous_velocity.updateValue(initial_velocity.reshape(-1))
    initial_contacts = self.update_contacts()

    checkpoints = [
      self._solve_frame(frame) for frame in range(self.frames)
    ]
    final_position = checkpoints[-1].position
    positions = [
      initial,
      *(checkpoint.position for checkpoint in checkpoints),
    ]
    loss, state_loss_gradients, center_offset = (
      self._trajectory_loss(positions)
    )
    return Trajectory(
      initial_position=initial,
      initial_velocity=initial_velocity,
      checkpoints=checkpoints,
      state_loss_gradients=state_loss_gradients,
      loss=loss,
      final_center_offset=center_offset,
      final_min_height=float(final_position[:, 1].min()),
      initial_self_pairs=initial_contacts.self_pair_count,
      maximum_self_pairs=max(
        checkpoint.contacts.self_pair_count
        for checkpoint in checkpoints
      ),
      maximum_point_point_pairs=max(
        checkpoint.contacts.point_point.count
        for checkpoint in checkpoints
      ),
      maximum_point_edge_pairs=max(
        checkpoint.contacts.point_edge.count
        for checkpoint in checkpoints
      ),
      maximum_point_triangle_pairs=max(
        checkpoint.contacts.point_triangle.count
        for checkpoint in checkpoints
      ),
      maximum_edge_edge_pairs=max(
        checkpoint.contacts.edge_edge.count
        for checkpoint in checkpoints
      ),
      frames_with_self_contact=sum(
        checkpoint.contacts.self_pair_count > 0
        for checkpoint in checkpoints
      ),
      maximum_floor_contacts=max(
        checkpoint.contacts.floor_vertices.size
        for checkpoint in checkpoints
      ),
      maximum_residual_inf=max(
        checkpoint.residual_inf for checkpoint in checkpoints
      ),
      mean_newton_iterations=float(np.mean([
        checkpoint.newton_iterations for checkpoint in checkpoints
      ])),
      elapsed_seconds=time.perf_counter() - start
    )

  def _restore_checkpoint(self, checkpoint: StepCheckpoint):
    self.previous_position.updateValue(
      checkpoint.previous_position.reshape(-1)
    )
    self.previous_velocity.updateValue(
      checkpoint.previous_velocity.reshape(-1)
    )
    self.position.updateValue(checkpoint.position.reshape(-1))
    self.apply_contacts(checkpoint.contacts)

  def adjoint(
    self, trajectory: Trajectory, design: str
  ):
    position_adjoint = vector(self.num_dofs)
    position_adjoint.updateValue(
      trajectory.state_loss_gradients[-1].reshape(-1)
    )
    velocity_adjoint = vector(self.num_dofs)
    velocity_adjoint.updateValue(np.zeros(self.num_dofs))
    direct_position_gradient = vector(self.num_dofs)
    if design == DESIGN_YOUNG:
      parameter_gradient = {
        "young": np.zeros(self.num_tets, dtype=np.float64),
        "poisson": np.zeros(self.num_tets, dtype=np.float64),
      }
    elif design == DESIGN_MASS:
      parameter_gradient = np.zeros(
        self.num_vertices, dtype=np.float64
      )
    else:
      parameter_gradient = np.zeros(3, dtype=np.float64)
    parameter_jacobian = None
    if design == DESIGN_YOUNG:
      parameter_jacobian = self.material_jacobian
    elif design == DESIGN_MASS:
      parameter_jacobian = self.mass_jacobian

    backward_start = time.perf_counter()
    for reverse_index, checkpoint in enumerate(
      reversed(trajectory.checkpoints)
    ):
      self._restore_checkpoint(checkpoint)
      self.adjoint_system.compute()

      # The next velocity is (x_{k+1} - x_k) / dt, so both next-state
      # adjoints act on the implicit x_{k+1} solve.
      implicit_rhs_value = (
        position_adjoint.value
        + velocity_adjoint.value / self.dt_value
      )
      implicit_rhs_scale = max(
        1.0, gpu_inf_norm(implicit_rhs_value)
      )
      if not np.isfinite(implicit_rhs_scale):
        raise RuntimeError(
          "Adjoint right-hand side became non-finite at reverse step "
          f"{reverse_index}."
        )
      implicit_rhs = vector(self.num_dofs)
      implicit_rhs.updateValue(
        implicit_rhs_value / implicit_rhs_scale
      )
      auxiliary = self.adjoint_system.solve(
        implicit_rhs,
        tolerance=self.linear_tolerance,
        maxIterations=self.max_linear_iterations,
        recompute=False,
        zero_initial_guess=True
      )
      if self.adjoint_system.last_solve_error_code < 0:
        auxiliary, minres_residual, _ = solve_symmetric_indefinite(
          self.adjoint_system,
          implicit_rhs,
          tolerance=self.linear_tolerance,
          max_iterations=self.max_linear_iterations,
        )
        if self.verbose:
          print(
            "adjoint exact-Hessian MINRES fallback at reverse step "
            f"{reverse_index}: relative_residual="
            f"{minres_residual:.3e}"
          )
      auxiliary.updateValue(
        auxiliary.value * implicit_rhs_scale
      )
      if not np.isfinite(gpu_inf_norm(auxiliary.value)):
        raise RuntimeError(
          "Adjoint solution became non-finite at reverse step "
          f"{reverse_index}."
        )

      self.previous_position_jacobian.compute()
      previous_position_contribution = (
        self.previous_position_jacobian
        .transposeMatVecProduct(auxiliary)
      )
      self.previous_velocity_jacobian.compute()
      previous_velocity_contribution = (
        self.previous_velocity_jacobian
        .transposeMatVecProduct(auxiliary)
      )

      if parameter_jacobian is not None:
        parameter_jacobian.compute()
        parameter_contribution = (
          parameter_jacobian
          .transposeMatVecProduct(auxiliary)
          .value
          .get()
        )
        if design == DESIGN_YOUNG:
          parameter_gradient["young"] -= (
            parameter_contribution[:self.num_tets]
          )
          parameter_gradient["poisson"] -= (
            parameter_contribution[self.num_tets:]
          )
        else:
          parameter_gradient -= parameter_contribution

      # Reverse the complete position/velocity state transition and then add
      # the loss derivative at x_k:
      #
      # lambda_x[k] = dL/dx_k - B_x^T mu
      #               - lambda_v[k+1] / dt
      # lambda_v[k] = -B_v^T mu.
      next_velocity_adjoint = velocity_adjoint.value.copy()
      state_index = self.frames - reverse_index - 1
      direct_position_gradient.updateValue(
        trajectory.state_loss_gradients[state_index].reshape(-1)
      )
      position_adjoint.updateValue(
        direct_position_gradient.value
        - previous_position_contribution.value
        - next_velocity_adjoint / self.dt_value
      )
      velocity_adjoint.updateValue(
        -previous_velocity_contribution.value
      )

      if (
        self.verbose
        and (
          reverse_index == 0
          or reverse_index + 1 == self.frames
          or (reverse_index + 1) % max(1, self.frames // 10) == 0
        )
      ):
        print(
          f"adjoint step {reverse_index + 1:04d}/{self.frames}: "
          f"|lambda_x|_inf="
          f"{gpu_inf_norm(position_adjoint.value):.3e}, "
          f"|lambda_v|_inf="
          f"{gpu_inf_norm(velocity_adjoint.value):.3e}"
        )

    if design == DESIGN_INITIAL_POSITION:
      parameter_gradient = (
        position_adjoint.value.get().reshape((-1, 3)).sum(axis=0)
      )

    if self.verbose:
      print(
        f"adjoint {design}: gradient_l2="
        f"{gradient_l2(design, parameter_gradient):.6e}, "
        f"time={time.perf_counter() - backward_start:.3f}s"
      )
    return parameter_gradient

  def finite_difference_gradient(
    self,
    design: str,
    epsilon: float,
    direction
  ) -> float:
    base_value = self.get_design_value(design)
    plus_value = perturb_parameter(
      design, base_value, direction, epsilon
    )
    minus_value = perturb_parameter(
      design, base_value, direction, -epsilon
    )

    self.set_design_value(design, plus_value)
    plus_loss = self.forward().loss
    self.set_design_value(design, minus_value)
    minus_loss = self.forward().loss
    self.set_design_value(design, base_value)
    return (plus_loss - minus_loss) / (2.0 * epsilon)


def _json_value(value):
  if isinstance(value, dict):
    return {key: _json_value(item) for key, item in value.items()}
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  return value


POISSON_MIN = 0.05
POISSON_MAX = 0.45
YOUNG_MIN = 100.0
YOUNG_MAX = 2.0e4
VERTEX_MASS_MIN = 1.0e-9
VERTEX_MASS_MAX = 1.0


def field_summary(value: np.ndarray) -> Dict:
  value = np.asarray(value, dtype=np.float64)
  return {
    "count": int(value.size),
    "minimum": float(value.min()),
    "maximum": float(value.max()),
    "mean": float(value.mean()),
    "standard_deviation": float(value.std()),
    "l2_norm": float(np.linalg.norm(value)),
  }


def summarize_design_value(design: str, value) -> Dict:
  if design == DESIGN_YOUNG:
    return {
      "young_per_element": field_summary(value["young"]),
      "poisson_per_element": field_summary(value["poisson"]),
    }
  if design == DESIGN_MASS:
    result = field_summary(value)
    result["total_mass"] = float(np.sum(value))
    result["layout"] = "per-vertex"
    return result
  return {"translation": np.asarray(value, dtype=np.float64).tolist()}


def summarize_gradient(design: str, value) -> Dict:
  if design == DESIGN_YOUNG:
    return {
      "young_per_element": field_summary(value["young"]),
      "poisson_per_element": field_summary(value["poisson"]),
    }
  if design == DESIGN_MASS:
    return {"mass_per_vertex": field_summary(value)}
  return {"translation": np.asarray(value, dtype=np.float64).tolist()}


def gradient_l2(design: str, value) -> float:
  if design == DESIGN_YOUNG:
    return float(math.sqrt(
      np.dot(value["young"], value["young"])
      + np.dot(value["poisson"], value["poisson"])
    ))
  return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _rms_normalized(value: np.ndarray) -> np.ndarray:
  value = np.asarray(value, dtype=np.float64)
  maximum = float(np.max(np.abs(value)))
  if maximum <= 1.0e-30:
    return np.zeros_like(value)
  scaled = value / maximum
  rms = float(np.sqrt(np.mean(scaled * scaled)))
  return np.clip(scaled / rms, -2.0, 2.0)


def _poisson_coordinate(poisson: np.ndarray) -> np.ndarray:
  normalized = (
    (np.asarray(poisson, dtype=np.float64) - POISSON_MIN)
    / (POISSON_MAX - POISSON_MIN)
  )
  normalized = np.clip(normalized, 1.0e-9, 1.0 - 1.0e-9)
  return np.log(normalized / (1.0 - normalized))


def _poisson_coordinate_derivative(poisson: np.ndarray) -> np.ndarray:
  poisson = np.asarray(poisson, dtype=np.float64)
  return (
    (poisson - POISSON_MIN)
    * (POISSON_MAX - poisson)
    / (POISSON_MAX - POISSON_MIN)
  )


def parameter_descent_direction(design: str, value, gradient_value):
  if design == DESIGN_YOUNG:
    return {
      "young": _rms_normalized(
        np.asarray(value["young"]) * gradient_value["young"]
      ),
      "poisson": _rms_normalized(
        _poisson_coordinate_derivative(value["poisson"])
        * gradient_value["poisson"]
      ),
    }
  if design == DESIGN_MASS:
    return _rms_normalized(
      np.asarray(value, dtype=np.float64)
      * np.asarray(gradient_value, dtype=np.float64)
    )
  gradient_value = np.asarray(gradient_value, dtype=np.float64)
  return gradient_value / max(np.linalg.norm(gradient_value), 1.0e-30)


def perturb_parameter(design: str, value, direction, delta: float):
  if design == DESIGN_YOUNG:
    young = np.clip(
      np.asarray(value["young"], dtype=np.float64)
      * np.exp(delta * np.asarray(direction["young"])),
      YOUNG_MIN,
      YOUNG_MAX,
    )
    poisson_coordinate = (
      _poisson_coordinate(value["poisson"])
      + delta * np.asarray(direction["poisson"])
    )
    sigmoid = 1.0 / (1.0 + np.exp(-poisson_coordinate))
    poisson = POISSON_MIN + (
      POISSON_MAX - POISSON_MIN
    ) * sigmoid
    return {"young": young, "poisson": poisson}
  if design == DESIGN_MASS:
    return np.clip(
      np.asarray(value, dtype=np.float64)
      * np.exp(delta * np.asarray(direction, dtype=np.float64)),
      VERTEX_MASS_MIN,
      VERTEX_MASS_MAX,
    )
  candidate = np.asarray(value, dtype=np.float64) + (
    delta * np.asarray(direction, dtype=np.float64)
  )
  candidate[[0, 2]] = np.clip(candidate[[0, 2]], -0.5, 0.5)
  candidate[1] = np.clip(candidate[1], 0.03, 1.0)
  return candidate


def parameter_directional_derivative(
  design: str, value, gradient_value, direction
) -> float:
  if design == DESIGN_YOUNG:
    return float(
      np.dot(
        gradient_value["young"],
        np.asarray(value["young"]) * direction["young"],
      )
      + np.dot(
        gradient_value["poisson"],
        _poisson_coordinate_derivative(value["poisson"])
        * direction["poisson"],
      )
    )
  if design == DESIGN_MASS:
    return float(np.dot(
      gradient_value,
      np.asarray(value, dtype=np.float64) * direction,
    ))
  return float(np.dot(gradient_value, direction))


def design_values_equal(design: str, first, second) -> bool:
  if design == DESIGN_YOUNG:
    return (
      np.array_equal(first["young"], second["young"])
      and np.array_equal(first["poisson"], second["poisson"])
    )
  return np.array_equal(np.asarray(first), np.asarray(second))


def optimize_design(
  simulation: AdjointBunnySimulation,
  design: str,
  optimization_steps: int,
  step_size: float,
  check_gradient: bool,
  convergence_tolerance: float = 1.0e-5
) -> Tuple[Dict, Trajectory, Trajectory]:
  simulation.prepare_design(design)
  initial_value = simulation.get_design_value(design)

  history = []
  trajectory = simulation.forward()
  initial_trajectory = trajectory

  for optimization_step in range(optimization_steps):
    value = simulation.get_design_value(design)
    gradient_value = simulation.adjoint(trajectory, design)
    direction = parameter_descent_direction(
      design, value, gradient_value
    )
    record = {
      "iteration": optimization_step,
      "value": summarize_design_value(design, value),
      "loss": trajectory.loss,
      "center_distance": trajectory.center_distance,
      "horizontal_center_distance": (
        trajectory.horizontal_center_distance
      ),
      "gradient": summarize_gradient(design, gradient_value),
      "forward_seconds": trajectory.elapsed_seconds,
    }

    if check_gradient and optimization_step == 0:
      epsilon = (
        1.0e-3
        if design in (DESIGN_INITIAL_POSITION, DESIGN_MASS)
        else 1.0e-2
      )
      finite_difference = simulation.finite_difference_gradient(
        design, epsilon, direction
      )
      record["finite_difference_epsilon"] = epsilon
      adjoint_directional = parameter_directional_derivative(
        design, value, gradient_value, direction
      )
      record["finite_difference"] = finite_difference
      record["adjoint_directional"] = adjoint_directional
      record["gradient_relative_error"] = (
        abs(finite_difference - adjoint_directional)
        / max(
          1.0e-14,
          abs(finite_difference),
          abs(adjoint_directional)
        )
      )
      simulation.set_design_value(design, value)

    accepted = False
    trial_step = step_size
    record["failed_trials"] = []
    for _ in range(6):
      candidate = perturb_parameter(
        design, value, direction, -trial_step
      )
      if design_values_equal(design, candidate, value):
        trial_step *= 0.5
        continue
      simulation.set_design_value(design, candidate)
      try:
        candidate_trajectory = simulation.forward()
      except RuntimeError as error:
        record["failed_trials"].append({
          "step": trial_step,
          "reason": str(error),
        })
        trial_step *= 0.5
        continue
      if candidate_trajectory.loss < trajectory.loss:
        previous_loss = trajectory.loss
        trajectory = candidate_trajectory
        accepted = True
        record["accepted_step"] = trial_step
        record["accepted_value"] = summarize_design_value(
          design, candidate
        )
        record["accepted_loss"] = trajectory.loss
        record["relative_improvement"] = (
          (previous_loss - trajectory.loss)
          / max(abs(previous_loss), 1.0e-30)
        )
        break
      trial_step *= 0.5
    if not accepted:
      simulation.set_design_value(design, value)
      record["accepted_step"] = 0.0
      record["relative_improvement"] = 0.0
    history.append(record)

    print(
      f"optimization {design} {optimization_step + 1}/"
      f"{optimization_steps}: loss={record['loss']:.6e}, "
      f"center={record['center_distance']:.6e}, "
      f"gradient_l2={gradient_l2(design, gradient_value):.6e}, "
      f"accepted_step={record['accepted_step']:.4g}"
    )
    if not accepted:
      print(
        f"optimization {design} stopped: line search found no "
        "loss-decreasing update."
      )
      break
    if record["relative_improvement"] <= convergence_tolerance:
      print(
        f"optimization {design} converged: relative loss improvement "
        f"{record['relative_improvement']:.3e} <= "
        f"{convergence_tolerance:.3e}."
      )
      break

  final_value = simulation.get_design_value(design)
  result = {
    "design": design,
    "frames": simulation.frames,
    "dt": simulation.dt_value,
    "velocity_retention": simulation.velocity_retention_value,
    "target_position": simulation.target_position.tolist(),
    "initial_value": summarize_design_value(design, initial_value),
    "initial_loss": initial_trajectory.loss,
    "final_loss": trajectory.loss,
    "initial_forward_seconds": initial_trajectory.elapsed_seconds,
    "final_forward_seconds": trajectory.elapsed_seconds,
    "initial_center_distance": initial_trajectory.center_distance,
    "final_center_distance": trajectory.center_distance,
    "initial_horizontal_center_distance": (
      initial_trajectory.horizontal_center_distance
    ),
    "final_horizontal_center_distance": (
      trajectory.horizontal_center_distance
    ),
    "final_min_height": trajectory.final_min_height,
    "initial_maximum_self_pairs": (
      initial_trajectory.maximum_self_pairs
    ),
    "initial_self_pairs": initial_trajectory.initial_self_pairs,
    "maximum_self_pairs": trajectory.maximum_self_pairs,
    "maximum_point_point_pairs": (
      trajectory.maximum_point_point_pairs
    ),
    "maximum_point_edge_pairs": trajectory.maximum_point_edge_pairs,
    "maximum_point_triangle_pairs": (
      trajectory.maximum_point_triangle_pairs
    ),
    "maximum_edge_edge_pairs": trajectory.maximum_edge_edge_pairs,
    "frames_with_self_contact": trajectory.frames_with_self_contact,
    "maximum_floor_contacts": trajectory.maximum_floor_contacts,
    "maximum_residual_inf": trajectory.maximum_residual_inf,
    "mean_newton_iterations": trajectory.mean_newton_iterations,
    "optimization_iterations": len(history),
    "final_value": summarize_design_value(design, final_value),
    "history": history,
  }
  print(
    f"result {design}: loss {result['initial_loss']:.6e} -> "
    f"{result['final_loss']:.6e}, center "
    f"{result['initial_center_distance']:.6e} -> "
    f"{result['final_center_distance']:.6e}, "
    f"self_pairs_max={result['maximum_self_pairs']}, "
    f"floor_contacts_max={result['maximum_floor_contacts']}"
  )
  return result, initial_trajectory, trajectory


def render_comparison_video(
  simulation: AdjointBunnySimulation,
  design: str,
  before: Trajectory,
  after: Trajectory,
  output_directory: Path,
  fps: int = 30
) -> Tuple[Path, Path]:
  """
  Render only the first and last optimization trajectories side by side.

  Every rendered PNG is retained; ffmpeg consumes those images directly, so
  the visualization path never writes an intermediate OBJ file.
  """
  if fps <= 0:
    raise ValueError("fps must be positive.")
  before_positions = [
    before.initial_position,
    *(checkpoint.position for checkpoint in before.checkpoints),
  ]
  after_positions = [
    after.initial_position,
    *(checkpoint.position for checkpoint in after.checkpoints),
  ]
  if len(before_positions) != len(after_positions):
    raise ValueError(
      "Before and after trajectories must have equal frame counts."
    )

  output_directory.mkdir(parents=True, exist_ok=True)
  image_directory = output_directory / design / "frames"
  image_directory.mkdir(parents=True, exist_ok=True)
  video_path = output_directory / f"{design}_before_after.mp4"
  faces = np.column_stack((
    np.full(simulation.surface_triangles.shape[0], 3, dtype=np.int64),
    simulation.surface_triangles.astype(np.int64),
  )).reshape(-1)
  before_mesh = pv.PolyData(before_positions[0], faces)
  after_mesh = pv.PolyData(after_positions[0], faces)

  all_positions = np.concatenate(
    [*before_positions, *after_positions], axis=0
  )
  horizontal_radius = max(
    0.45,
    float(np.max(np.linalg.norm(all_positions[:, [0, 2]], axis=1)))
    * 1.35,
  )
  maximum_height = max(0.45, float(all_positions[:, 1].max()) * 1.15)
  floor = pv.Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 1.0, 0.0),
    i_size=1000.0,
    j_size=1000.0,
    i_resolution=1,
    j_resolution=1,
  )
  plotter = pv.Plotter(
    shape=(1, 2),
    off_screen=True,
    window_size=(1280, 640),
    border=False,
  )
  titles = (
    f"Before optimization\nloss={before.loss:.5e}",
    f"After optimization\nloss={after.loss:.5e}",
  )
  meshes = (before_mesh, after_mesh)
  camera_position = [
    (2.6 * horizontal_radius, 1.15 * maximum_height,
     3.8 * horizontal_radius),
    (0.0, 0.42 * maximum_height, 0.0),
    (0.0, 1.0, 0.0),
  ]
  for column, (mesh, title) in enumerate(zip(meshes, titles)):
    plotter.subplot(0, column)
    plotter.set_background("#f7f7f4")
    plotter.add_mesh(
      floor,
      color="#d6d9dd",
      opacity=0.65,
      show_edges=True,
      edge_color="#a8adb3",
      line_width=1.0,
    )
    plotter.add_mesh(
      mesh,
      color="#d9784a",
      smooth_shading=True,
      specular=0.18,
      show_edges=False,
    )
    plotter.add_text(
      title,
      position="upper_left",
      font_size=13,
      color="#202124",
    )
    plotter.camera_position = camera_position
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = max(
      0.52,
      0.74 * maximum_height,
      1.10 * horizontal_radius,
    )
    plotter.camera.clipping_range = (
      0.01,
      8.0 * max(horizontal_radius, maximum_height),
    )

  plotter.link_views()
  for frame_index, (before_position, after_position) in enumerate(
    zip(before_positions, after_positions)
  ):
    before_mesh.points = before_position
    after_mesh.points = after_position
    plotter.render()
    plotter.screenshot(
      str(image_directory / f"frame_{frame_index:04d}.png")
    )
  plotter.close()

  subprocess.run(
    [
      "ffmpeg",
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      str(fps),
      "-i",
      str(image_directory / "frame_%04d.png"),
      "-c:v",
      "libx264",
      "-crf",
      "18",
      "-pix_fmt",
      "yuv420p",
      str(video_path),
    ],
    check=True,
  )
  return image_directory, video_path


def build_parser(
  default_parameter: Optional[str] = None
) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Optimize a dropping bunny with a discrete adjoint built from "
      "explicit YASPS Hessian and mixed-Jacobian objects."
    )
  )
  parser.add_argument(
    "--parameter",
    choices=(*DESIGN_PARAMETERS, "all"),
    default=default_parameter or "all"
  )
  parser.add_argument("--frames", type=int, default=500)
  parser.add_argument("--dt", type=float, default=0.01)
  parser.add_argument(
    "--optimization-steps", type=int, default=6
  )
  parser.add_argument("--bunny-height", type=float, default=0.35)
  parser.add_argument(
    "--node-path",
    type=Path,
    default=SCRIPT_DIR.parent / "data" / "bunny.node",
  )
  parser.add_argument(
    "--element-path",
    type=Path,
    default=SCRIPT_DIR.parent / "data" / "bunny.ele",
  )
  parser.add_argument("--d-hat", type=float, default=1.0e-6)
  parser.add_argument("--kappa", type=float, default=5.0e3)
  parser.add_argument("--young", type=float, default=1.0e4)
  parser.add_argument("--poisson", type=float, default=0.32)
  parser.add_argument(
    "--velocity-retention",
    type=float,
    default=0.90,
    help="Fraction of velocity retained by each implicit time step.",
  )
  parser.add_argument(
    "--mass",
    type=float,
    default=4.0e-5,
    help="Initial mass assigned independently to each vertex.",
  )
  parser.add_argument(
    "--target-position",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    help=(
      "Translation of the target rest configuration. The default is "
      "(0, sqrt(d_hat), 0)."
    ),
  )
  parser.add_argument("--initial-x", type=float, default=-0.28)
  parser.add_argument("--drop-height", type=float, default=0.15)
  parser.add_argument("--initial-z", type=float, default=-0.20)
  parser.add_argument(
    "--newton-tolerance", type=float, default=1.0e-12
  )
  parser.add_argument(
    "--contact-newton-tolerance", type=float, default=2.0e-7
  )
  parser.add_argument(
    "--max-newton-iterations", type=int, default=300
  )
  parser.add_argument("--young-step", type=float, default=0.15)
  parser.add_argument("--mass-step", type=float, default=0.15)
  parser.add_argument(
    "--initial-position-step", type=float, default=0.08
  )
  parser.add_argument(
    "--convergence-tolerance", type=float, default=1.0e-5
  )
  parser.add_argument("--check-gradient", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("--json-output", type=Path)
  parser.add_argument(
    "--video-directory",
    type=Path,
    help=(
      "Save side-by-side before/after PNG frames and an MP4 for each "
      "optimized design."
    ),
  )
  parser.add_argument("--video-fps", type=int, default=30)
  return parser


def main(default_parameter: Optional[str] = None):
  args = build_parser(default_parameter).parse_args()
  simulation = AdjointBunnySimulation(
    frames=args.frames,
    dt=args.dt,
    bunny_height=args.bunny_height,
    node_path=args.node_path,
    element_path=args.element_path,
    d_hat=args.d_hat,
    kappa=args.kappa,
    young=args.young,
    poisson=args.poisson,
    mass=args.mass,
    velocity_retention=args.velocity_retention,
    initial_translation=np.asarray(
      [args.initial_x, args.drop_height, args.initial_z],
      dtype=np.float64,
    ),
    target_position=(
      None
      if args.target_position is None
      else np.asarray(args.target_position, dtype=np.float64)
    ),
    newton_tolerance=args.newton_tolerance,
    contact_newton_tolerance=args.contact_newton_tolerance,
    max_newton_iterations=args.max_newton_iterations,
    verbose=not args.quiet
  )
  parameters = (
    DESIGN_PARAMETERS
    if args.parameter == "all"
    else (args.parameter,)
  )
  step_sizes = {
    DESIGN_YOUNG: args.young_step,
    DESIGN_INITIAL_POSITION: args.initial_position_step,
    DESIGN_MASS: args.mass_step,
  }
  results = {}
  for design in parameters:
    result, initial_trajectory, final_trajectory = optimize_design(
      simulation,
      design,
      args.optimization_steps,
      step_sizes[design],
      args.check_gradient,
      args.convergence_tolerance,
    )
    if args.video_directory is not None:
      image_directory, video_path = render_comparison_video(
        simulation,
        design,
        initial_trajectory,
        final_trajectory,
        args.video_directory,
        args.video_fps,
      )
      result["frame_directory"] = str(image_directory)
      result["video"] = str(video_path)
      print(
        f"rendered {design}: frames={image_directory}, "
        f"video={video_path}"
      )
    results[design] = result
    if args.json_output is not None:
      args.json_output.parent.mkdir(parents=True, exist_ok=True)
      args.json_output.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
      )
  return results


if __name__ == "__main__":
  main()

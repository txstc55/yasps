"""Geometry, energies, and output helpers used only by the hand/bunny example."""
from pathlib import Path
import subprocess

import numpy as np
from yasps import attribute


##################################################################
## Geometry, physical masses, and output
##################################################################
def load_bunny_tets(data_directory):
  # The repository's .ele file stores tet indices after three header columns.
  data_directory = Path(data_directory)
  with open(data_directory / "bunny.node") as stream:
    next(stream)
    vertices = [[float(value) for value in line.split()[1:4]] for line in stream if line.strip() and not line.lstrip().startswith("#")]
  with open(data_directory / "bunny.ele") as stream:
    next(stream)
    tets = [[int(value) - 1 for value in line.split()[3:7]] for line in stream if line.strip() and not line.lstrip().startswith("#")]
  vertices = np.asarray(vertices, dtype=np.float64)
  tets = np.asarray(tets, dtype=np.uint32)
  signed_volumes = tet_signed_volumes(vertices, tets)
  reversed_tets = signed_volumes < 0.0
  tets[reversed_tets, 1], tets[reversed_tets, 2] = tets[reversed_tets, 2].copy(), tets[reversed_tets, 1].copy()
  if np.any(tet_signed_volumes(vertices, tets) <= 0.0):
    raise ValueError("The bunny must contain only nondegenerate, positively oriented tetrahedra.")
  return vertices, tets


def tet_signed_volumes(vertices, tets):
  points = vertices[tets]
  return np.einsum("ij,ij->i", np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), points[:, 3] - points[:, 0]) / 6.0


def tet_vertex_masses(vertices, tets, density):
  volumes = tet_signed_volumes(vertices, tets)
  if np.any(volumes <= 0.0):
    raise ValueError("Mass construction requires positively oriented rest tetrahedra.")
  masses = np.zeros(len(vertices), dtype=np.float64)
  np.add.at(masses, tets.ravel(), np.repeat(density * volumes / 4.0, 4))
  return masses, volumes


def surface_vertex_masses(vertices, triangles, total_mass):
  # Area lumping preserves the requested hand mass independently of tessellation.
  points = vertices[triangles]
  areas = 0.5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
  masses = np.zeros(len(vertices), dtype=np.float64)
  np.add.at(masses, triangles.ravel(), np.repeat(areas / 3.0, 3))
  return masses * (total_mass / masses.sum())


def extract_surface_triangles(tets):
  # These orientations point outwards for positive-volume tetrahedra.
  oriented_faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
  _, first, counts = np.unique(np.sort(oriented_faces, axis=1), axis=0, return_index=True, return_counts=True)
  return oriented_faces[first[counts == 1]].astype(np.uint32)


def extract_edges_from_triangles(triangles):
  edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
  return np.unique(np.sort(edges, axis=1), axis=0).astype(np.uint32)


def triangle_cells(triangles):
  return np.column_stack([np.full(len(triangles), 3, dtype=np.uint32), triangles]).ravel()


def save_video(output_directory, frame_count, fps):
  output_directory = Path(output_directory)
  path = output_directory / "hand_bunny_grasp.mp4"
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(output_directory / "frame_%04d.jpg"), "-frames:v", str(frame_count), "-c:v", "libx264", "-threads", "8", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
  return path


##################################################################
## Implicit-Euler inertia and the original stable neo-Hookean model
##################################################################
def inertia(last_position, velocity, dt, position, mass, gravity):
  target = last_position + velocity * dt - attribute.to_array([0.0, gravity, 0.0], rows=3, cols=1) * dt * dt
  difference = position - target
  return 0.5 * difference.transpose() * mass * difference


def stable_neo_hookean(tet_position_rest, tet_position, mu, lam, dt):
  x0 = tet_position_rest.row(1) - tet_position_rest.row(0)
  x1 = tet_position_rest.row(2) - tet_position_rest.row(0)
  x2 = tet_position_rest.row(3) - tet_position_rest.row(0)
  rest_edges = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows=3, cols=3).transpose()
  volume = rest_edges.determinant() / 6.0
  x0 = tet_position.row(1) - tet_position.row(0)
  x1 = tet_position.row(2) - tet_position.row(0)
  x2 = tet_position.row(3) - tet_position.row(0)
  current_edges = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows=3, cols=3).transpose()
  deformation = current_edges * rest_edges.inverse()
  jacobian = deformation.determinant()
  invariant = (deformation.transpose() * deformation).trace()
  volume_difference = jacobian - (1.0 + 0.75 * mu / lam)
  return volume * (0.5 * mu * (invariant - 3.0) - 0.5 * mu * (invariant + 1.0).log() + 0.5 * lam * volume_difference * volume_difference) * dt * dt


##################################################################
## Affine pose targets and independent proper-rotation constraints
##################################################################
def affine_matrix_target(matrix, target, weight, dt):
  difference = matrix - target
  squared_norm = sum(difference.row(i).dot(difference.row(i)) for i in range(3))
  return 0.5 * weight * squared_norm * dt * dt


def affine_translation_target(translation, target, weight, dt):
  difference = translation - target
  return 0.5 * weight * difference.dot(difference) * dt * dt


def orthogonality(matrix, weight, dt):
  difference = matrix.transpose() * matrix - attribute.identity(3)
  squared_norm = sum(difference.row(i).dot(difference.row(i)) for i in range(3))
  return 0.5 * weight * squared_norm * dt * dt


def determinant(matrix, weight, dt):
  difference = matrix.determinant() - 1.0
  return 0.5 * weight * difference * difference * dt * dt


##################################################################
## Parent/child constraints in the two bones' own bind-local frames
##################################################################
def joint_pivot(pair_matrices, pair_translations, pivot_parent, pivot_child, weight, dt):
  parent = pair_matrices.row(0).resize(3, 3)
  child = pair_matrices.row(1).resize(3, 3)
  parent_origin = pair_translations.row(0).transpose()
  child_origin = pair_translations.row(1).transpose()
  difference = parent * pivot_parent + parent_origin - child * pivot_child - child_origin
  return 0.5 * weight * difference.dot(difference) * dt * dt


def hinge_axis(pair_matrices, axis_parent, axis_child, weight, dt):
  # With proper rotations, equal global hinge axes permit only relative twist.
  # axis_parent is Q_rest * axis_child, NOT simply the parent's local Z axis.
  parent = pair_matrices.row(0).resize(3, 3)
  child = pair_matrices.row(1).resize(3, 3)
  difference = parent * axis_parent - child * axis_child
  return 0.5 * weight * difference.dot(difference) * dt * dt


def fixed_joint_rotation(pair_matrices, rest_relative_rotation, weight, dt):
  # Fingertip/forearm rig markers retain their entire bind-relative rotation.
  parent = pair_matrices.row(0).resize(3, 3)
  child = pair_matrices.row(1).resize(3, 3)
  difference = parent * rest_relative_rotation - child
  squared_norm = sum(difference.row(i).dot(difference.row(i)) for i in range(3))
  return 0.5 * weight * squared_norm * dt * dt


##################################################################
## Collision barriers, matching the existing mixed-separation example
##################################################################
def point_point(position, dHat, kappa):
  difference = position.row(1) - position.row(0)
  distance_squared = difference.dot(difference)
  logarithm = (distance_squared / dHat).log()
  difference = distance_squared - dHat
  return kappa * difference * difference * logarithm * logarithm


def point_edge(position, dHat, kappa):
  p0, p1, p2 = position.row(0), position.row(1), position.row(2)
  cross = (p1 - p0).cross(p2 - p0)
  distance_squared = cross.dot(cross) / (p2 - p1).dot(p2 - p1)
  logarithm = (distance_squared / dHat).log()
  difference = distance_squared - dHat
  return kappa * difference * difference * logarithm * logarithm


def point_triangle(position, dHat, kappa):
  p0, p1, p2, p3 = position.row(0), position.row(1), position.row(2), position.row(3)
  normal = (p2 - p1).cross(p3 - p1)
  signed_volume = (p0 - p1).dot(normal)
  distance_squared = signed_volume * signed_volume / normal.dot(normal)
  logarithm = (distance_squared / dHat).log()
  difference = distance_squared - dHat
  return kappa * difference * difference * logarithm * logarithm


def edge_edge(position, dHat, kappa):
  p0, p1, p2, p3 = position.row(0), position.row(1), position.row(2), position.row(3)
  normal = (p1 - p0).cross(p3 - p2)
  signed_volume = (p2 - p0).dot(normal)
  distance_squared = signed_volume * signed_volume / normal.dot(normal)
  logarithm = (distance_squared / dHat).log()
  difference = distance_squared - dHat
  return kappa * difference * difference * logarithm * logarithm

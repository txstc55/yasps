import numpy as np

from yasps import attribute


##################################################################
## Mesh construction and output
##################################################################
def load_triangle_obj(path):
  vertices = []
  triangles = []
  with open(path, "r") as obj_file:
    for line in obj_file:
      if line.startswith("v "):
        vertices.append([float(value) for value in line.split()[1:4]])
      elif line.startswith("f "):
        face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
        if len(face) != 3:
          raise ValueError(f"Cubic stylization requires triangles; got a {len(face)}-gon in {path}.")
        triangles.append(face)
  if not vertices or not triangles:
    raise ValueError(f"No triangular surface mesh was found in {path}.")
  return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.uint32)


def normalize_bunny(vertices):
  normalized = vertices[:, [0, 2, 1]].copy()
  normalized[:, 2] *= -1.0
  normalized -= 0.5 * (normalized.min(axis=0) + normalized.max(axis=0))
  normalized *= 2.0 / np.max(np.ptp(normalized, axis=0))
  normalized[:, 1] -= normalized[:, 1].min()
  return normalized


def place_on_ground(vertices):
  grounded = vertices.copy()
  grounded[:, 1] -= grounded[:, 1].min()
  return grounded


def build_geometry(vertices, triangles):
  vertex_normals = np.zeros_like(vertices)
  vertex_areas = np.zeros(vertices.shape[0], dtype=np.float64)
  edge_weights = {}
  degenerate_faces = 0
  for triangle in triangles:
    i, j, k = [int(index) for index in triangle]
    p0, p1, p2 = vertices[i], vertices[j], vertices[k]
    cross = np.cross(p1 - p0, p2 - p0)
    twice_area = np.linalg.norm(cross)
    if twice_area <= 1e-14:
      degenerate_faces += 1
      continue
    vertex_normals[triangle] += cross
    vertex_areas[triangle] += twice_area / 6.0
    for edge, opposite in (((j, k), i), ((k, i), j), ((i, j), k)):
      a, b = edge
      u = vertices[a] - vertices[opposite]
      v = vertices[b] - vertices[opposite]
      cotangent = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
      key = (min(a, b), max(a, b))
      edge_weights[key] = edge_weights.get(key, 0.0) + 0.5 * cotangent
  normal_lengths = np.linalg.norm(vertex_normals, axis=1)
  if np.any(normal_lengths <= 1e-14):
    raise ValueError("The input contains vertices without a valid incident surface normal.")
  vertex_normals /= normal_lengths[:, None]
  edges = np.asarray(list(edge_weights), dtype=np.uint32)
  raw_weights = np.asarray(list(edge_weights.values()), dtype=np.float64)
  return vertex_normals, vertex_areas, edges, np.maximum(raw_weights, 1e-8), int(np.count_nonzero(raw_weights <= 0.0)), degenerate_faces


def build_triangle_cells(triangles):
  return np.hstack([np.full((triangles.shape[0], 1), 3, dtype=np.uint32), triangles])


def save_obj_without_normals(path, vertices, triangles):
  with open(path, "w") as obj_file:
    for vertex in vertices:
      obj_file.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
    for triangle in triangles:
      obj_file.write(f"f {int(triangle[0]) + 1} {int(triangle[1]) + 1} {int(triangle[2]) + 1}\n")


##################################################################
## Cubic stylization energies
##################################################################
def edge_arap_energy(weight, positions, rotations, rest_positions, penalty):
  p0, p1 = positions.row(0).transpose(), positions.row(1).transpose()
  q0, q1 = rest_positions.row(0).transpose(), rest_positions.row(1).transpose()
  a0, a1 = rotations.row(0).resize(3, 3), rotations.row(1).resize(3, 3)
  deformed_edge = p1 - p0
  rest_edge = q1 - q0
  residual0 = deformed_edge - a0 * rest_edge
  residual1 = deformed_edge - a1 * rest_edge
  return 0.5 * penalty * weight * (residual0.dot(residual0) + residual1.dot(residual1))


def vertex_cubic_energy(area, rotation, normal, penalty):
  rotated_normal = rotation * normal
  return penalty * area * (rotated_normal[0].abs() + rotated_normal[1].abs() + rotated_normal[2].abs())


def vertex_orthogonality_energy(area, rotation, penalty):
  residual = rotation.transpose() * rotation - attribute.identity(3)
  squared_norm = residual.row(0).dot(residual.row(0)) + residual.row(1).dot(residual.row(1)) + residual.row(2).dot(residual.row(2))
  return 0.5 * penalty * area * squared_norm


def vertex_determinant_energy(area, rotation, penalty):
  residual = rotation.determinant() - 1.0
  return 0.5 * penalty * area * residual * residual


def vertex_regularization_energy(position, rest_position, area, penalty):
  difference = position - rest_position
  return difference.transpose() * penalty * area * difference


def vertex_inertia_energy(last_position, velocity, time_step, position, mass, gravity):
  target = last_position + velocity * time_step - attribute.to_array([0.0, gravity, 0.0], rows=3, cols=1) * time_step * time_step
  difference = position - target
  return 0.5 * difference.transpose() * mass * difference

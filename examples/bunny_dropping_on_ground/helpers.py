import numpy as np

from yasps import attribute


def extract_surface_triangles(tets):
  from collections import defaultdict

  face_count = defaultdict(int)
  tet_faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3],
  ])
  for tet in tets:
    for face in tet_faces:
      sorted_face = tuple(sorted(tet[face]))
      face_count[sorted_face] += 1
  return np.array(
    [list(face) for face, count in face_count.items() if count == 1],
    dtype=np.uint32,
  )


def extract_edges_from_triangles(triangles):
  edges = set()
  for triangle in triangles:
    edges.add(tuple(sorted((triangle[0], triangle[1]))))
    edges.add(tuple(sorted((triangle[1], triangle[2]))))
    edges.add(tuple(sorted((triangle[2], triangle[0]))))
  return np.array(list(edges), dtype=np.uint32)


def stable_neo_hookean(tet_position_rest, tet_position, mu, lam, dt):
  rest_row0 = tet_position_rest.row(0)
  rest_row1 = tet_position_rest.row(1)
  rest_row2 = tet_position_rest.row(2)
  rest_row3 = tet_position_rest.row(3)
  rest_x0 = rest_row1 - rest_row0
  rest_x1 = rest_row2 - rest_row0
  rest_x2 = rest_row3 - rest_row0
  TB = attribute.to_array([rest_x0[0], rest_x0[1], rest_x0[2], rest_x1[0], rest_x1[1], rest_x1[2], rest_x2[0], rest_x2[1], rest_x2[2]], rows=3, cols=3)
  volume = TB.transpose().determinant() / 6.0
  IB = TB.transpose().inverse()

  row0 = tet_position.row(0)
  row1 = tet_position.row(1)
  row2 = tet_position.row(2)
  row3 = tet_position.row(3)
  x0 = row1 - row0
  x1 = row2 - row0
  x2 = row3 - row0
  F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows=3, cols=3)
  deformation = F.transpose() * IB
  determinant = deformation.determinant()
  trace = (deformation.transpose() * deformation).trace()
  shifted_trace = trace + 1.0
  return volume * (
    0.5 * mu * (trace - 3.0)
    - 0.5 * mu * shifted_trace.log()
    + 0.5 * lam * (
      determinant - (1.0 + 0.75 * mu / lam)
    ) * (
      determinant - (1.0 + 0.75 * mu / lam)
    )
  ) * dt * dt


def inertia(last_position, velocity, dt, position, mass):
  target = (
    last_position
    + velocity * dt
    - attribute.to_array(
      [0.0, 9.8 * dt * dt, 0.0],
      rows=3,
      cols=1,
    )
  )
  difference = position - target
  return 0.5 * difference.transpose() * mass * difference


def point_point(position, dhat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  squared_distance = (p1 - p0).dot(p1 - p0)
  normalized_distance = squared_distance / dhat
  distance_offset = squared_distance - dhat
  log_distance = normalized_distance.log()
  return kappa * distance_offset * distance_offset * log_distance * log_distance


def point_edge(position, dhat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  cross = (p1 - p0).cross(p2 - p0)
  squared_distance = cross.dot(cross) / (p2 - p1).dot(p2 - p1)
  normalized_distance = squared_distance / dhat
  distance_offset = squared_distance - dhat
  log_distance = normalized_distance.log()
  return kappa * distance_offset * distance_offset * log_distance * log_distance


def point_triangle(position, dhat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  p3 = position.row(3)
  normal = (p2 - p1).cross(p3 - p1)
  numerator = (p0 - p1).dot(normal)
  squared_distance = numerator * numerator / normal.dot(normal)
  normalized_distance = squared_distance / dhat
  distance_offset = squared_distance - dhat
  log_distance = normalized_distance.log()
  return kappa * distance_offset * distance_offset * log_distance * log_distance


def edge_edge(position, dhat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  p3 = position.row(3)
  normal = (p1 - p0).cross(p3 - p2)
  numerator = (p2 - p0).dot(normal)
  squared_distance = numerator * numerator / normal.dot(normal)
  normalized_distance = squared_distance / dhat
  distance_offset = squared_distance - dhat
  log_distance = normalized_distance.log()
  return kappa * distance_offset * distance_offset * log_distance * log_distance

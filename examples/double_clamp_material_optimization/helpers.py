import numpy as np

from yasps import attribute


def extract_surface_triangles(tetrahedra):
  face_pattern = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype = np.int64)
  faces = tetrahedra[:, face_pattern].reshape((-1, 3))
  face_keys = np.sort(faces, axis = 1)
  _, first_indices, counts = np.unique(face_keys, axis = 0, return_index = True, return_counts = True)
  return faces[first_indices[counts == 1]].astype(np.uint32)


def stable_neo_hookean(tet_position_rest, tet_position, mu, lam, dt):
  row0 = tet_position_rest.row(0)
  row1 = tet_position_rest.row(1)
  row2 = tet_position_rest.row(2)
  row3 = tet_position_rest.row(3)
  x0 = row1 - row0
  x1 = row2 - row0
  x2 = row3 - row0
  TB = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
  volume = TB.transpose().determinant() / 6.0
  IB = TB.transpose().inverse()

  row0 = tet_position.row(0)
  row1 = tet_position.row(1)
  row2 = tet_position.row(2)
  row3 = tet_position.row(3)
  x0 = row1 - row0
  x1 = row2 - row0
  x2 = row3 - row0
  F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
  deformation = F.transpose() * IB
  determinant = deformation.determinant()
  trace = (deformation.transpose() * deformation).trace()
  shifted_trace = trace + 1.0
  return volume * (0.5 * mu * (trace - 3.0) - 0.5 * mu * shifted_trace.log() + 0.5 * lam * (determinant - (1.0 + 0.75 * mu / lam)) * (determinant - (1.0 + 0.75 * mu / lam))) * dt * dt


def inertia(last_position, velocity, dt, position, mass):
  target_position = last_position + velocity * dt
  difference = position - target_position
  return 0.5 * difference.transpose() * mass * difference


def moving_energy(position, target_position, dt, stiffness):
  difference = position - target_position
  return 0.5 * stiffness * dt * dt * difference.dot(difference)

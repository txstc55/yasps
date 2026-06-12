import numpy as np
from yasps import attribute
import pycuda.gpuarray as gpuarray
def extract_surface_triangles(tets):
  from collections import defaultdict
  face_count = defaultdict(int)

  # Define all faces of a tetrahedron (4 faces per tetrahedron)
  tet_faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
  ])
  for tet in tets:
    for face in tet_faces:
      face_vertices = tet[face]
      sorted_face = tuple(sorted(face_vertices))
      face_count[sorted_face] += 1
  # Extract faces that occur only once (surface triangles)
  surface_triangles = [list(face) for face, count in face_count.items() if count == 1]
  return np.array(surface_triangles)

def extract_edges_from_triangles(triangles):
  edge_set = set()
  for tri in triangles:
    edges = [
      tuple(sorted([tri[0], tri[1]])),
      tuple(sorted([tri[1], tri[2]])),
      tuple(sorted([tri[2], tri[0]]))
    ]
    for edge in edges:
      edge_set.add(edge)
  return np.array([list(e) for e in edge_set])

def safe_gpu_sum(arr: gpuarray.GPUArray) -> float:
  if arr.size == 0:
      return 0.0
  return gpuarray.sum(arr).get()

def generate_cloth_mesh(length_of_square, num_segments):
  num_vertices_per_side = num_segments + 1
  # Compute step size along one edge
  step = length_of_square / num_segments
  # Generate vertex positions
  vertices = []
  for j in range(num_vertices_per_side):
    for i in range(num_vertices_per_side):
      x = i * step - length_of_square / 2
      y = 0.0
      z = j * step - length_of_square / 2
      vertices.append((x, y, z))
  faces = []
  for j in range(num_segments):
    for i in range(num_segments):
      # Calculate the vertex indices
      v0 = j * num_vertices_per_side + i
      v1 = v0 + 1
      v2 = (j + 1) * num_vertices_per_side + i
      v3 = v2 + 1
      faces.append((v2, v1, v0))
      faces.append((v3, v1, v2))
  return np.array(vertices, dtype = np.float64), np.array(faces, dtype = np.uint32)

def generate_edge_to_vertices_list(faces):
  edge_to_triangle = {}
  for i in range(faces.shape[0]):
    v0 = faces[i, 0]
    v1 = faces[i, 1]
    v2 = faces[i, 2]
    for edge in [(v0, v1), (v1, v2), (v2, v0)]:
      starting_index = edge[0]
      ending_index = edge[1]
      flipped = False
      if starting_index > ending_index:
        starting_index, ending_index = ending_index, starting_index
        flipped = True
      if (starting_index, ending_index) not in edge_to_triangle:
        edge_to_triangle[(starting_index, ending_index)] = []
      edge_to_triangle[(starting_index, ending_index)].append(-i - 1 if flipped else i)
  edge_to_triangle_vertices = []
  for item in edge_to_triangle:
    if len(edge_to_triangle[item]) == 2:
      indices = [item[0], item[1], -1, -1]
      for triangle_ind in edge_to_triangle[item]:
        # determine the third vertex
        true_ind = triangle_ind
        if triangle_ind < 0:
          true_ind = -triangle_ind - 1
        new_ind = -1
        for i in range(3):
          if faces[true_ind, i] != item[0] and faces[true_ind, i] != item[1]:
            new_ind = faces[true_ind, i]
            break
        if triangle_ind < 0:
          indices[3] = new_ind
        else:
          indices[2] = new_ind
      edge_to_triangle_vertices.append(indices)
  return np.array(edge_to_triangle_vertices, dtype = np.uint32)

def stable_neo_hookean(tet_position_rest, tet_position, mu, lam, dt):
  # here we compute the rest position deformation gradient
  row0 = tet_position_rest.row(0)
  row1 = tet_position_rest.row(1)
  row2 = tet_position_rest.row(2)
  row3 = tet_position_rest.row(3)
  x0 = row1 - row0
  x1 = row2 - row0
  x2 = row3 - row0
  TB = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
  vol = TB.transpose().determinant() / 6.0
  IB = TB.transpose().inverse()
  # add deformation gradient
  row0 = tet_position.row(0)
  row1 = tet_position.row(1)
  row2 = tet_position.row(2)
  row3 = tet_position.row(3)
  x0 = row1 - row0
  x1 = row2 - row0
  x2 = row3 - row0
  F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam)))) * dt * dt

def stable_neo_hookean_modified(F, TB, mu, lam, dt):
  IB = TB.transpose().inverse()
  vol = TB.transpose().determinant() / 6.0
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam)))) * dt * dt

def inertia(x_before, vel, dt, x, mass):
  x_target = x_before + vel * dt - attribute.to_array([0.0, 9.8 * dt * dt, 0.0], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))


def point_point(position, dHat, kappa):
  # 1E-6
  p0 = position.row(0)
  p1 = position.row(1)
  d = (p1 - p0).dot(p1 - p0)
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log

def point_edge(position, dHat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  cross = (p1 - p0).cross(p2 - p0)
  cross = cross.dot(cross)
  d = cross / ((p2 - p1).dot(p2 - p1))
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log

def point_triangle(position, dHat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  p3 = position.row(3)
  b = (p2 - p1).cross(p3 - p1)
  atb = (p0 - p1).dot(b)
  d = atb * atb / (b.dot(b))
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log

def edge_edge(position, dHat, kappa):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  p3 = position.row(3)
  b = (p1 - p0).cross(p3 - p2)
  atb = (p2 - p0).dot(b)
  d = atb * atb / (b.dot(b))
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log

from pycuda.reduction import ReductionKernel
# Define the reduction kernel once
abs_max_reduce = ReductionKernel(
  np.float64,
  neutral="0",
  reduce_expr="max(a, b)",
  map_expr="fabs(x[i])",
  arguments="double *x"
)

def affine_energy(rotation, weight):
  identity = attribute.to_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], rows=3, cols=3)
  rot_square = rotation.transpose() * rotation
  diff = rot_square - identity
  energy = 0.5 * (diff.row(0).dot(diff.row(0)) +
                  diff.row(1).dot(diff.row(1)) +
                  diff.row(2).dot(diff.row(2)))
  return weight * energy




def angle(v, w, axis):
  theta = 2.0 * (v.cross(w).dot(axis) / axis.norm()).atan2(v.dot(w) + v.norm() * w.norm())
  return theta
def edgeTheta(q0, q1, q2, q3):
  n0 = (q0 - q2).cross(q1 - q2)
  n1 = (q1 - q3).cross(q0 - q3)
  axis = q1 - q0
  theta = angle(n0, n1, axis)
  return theta
def bending(x, x_init, bendStiff, dt):
  x0 = x.row(0)
  x1 = x.row(1)
  x2 = x.row(2)
  x3 = x.row(3)
  t = edgeTheta(x0, x1, x2, x3)
  x_init0 = x_init.row(0)
  x_init1 = x_init.row(1)
  x_init2 = x_init.row(2)
  x_init3 = x_init.row(3)
  t_init = edgeTheta(x_init0, x_init1, x_init2, x_init3)
  bend_energy = bendStiff * (t - t_init) * (t - t_init) * (x_init1 - x_init0).norm()
  return bend_energy * dt * dt

def edgeTheta_modified(e, a, b):
  n0 = e.cross(a)
  n1 = b.cross(e)
  axis = e
  theta = angle(n0, n1, axis)
  return theta

def bending_modified(g, g_init, bendStiff, dt):
  t = edgeTheta_modified(g.row(0), g.row(1), g.row(2))
  t_init = edgeTheta_modified(g_init.row(0), g_init.row(1), g_init.row(2))
  bend_energy = bendStiff * (t - t_init) * (t - t_init) * g_init.row(0).norm()
  return bend_energy * dt * dt

def compute_rest_shape(v0, v1, v2):
  v01 = v1 - v0
  v02 = v2 - v0
  normal = v01.cross(v02)
  normal = normal / normal.norm()
  target = attribute.to_array([0, 1, 0], rows = 1, cols = 3)
  vec = normal.cross(target)
  cos = normal.dot(target)
  rotation = attribute.identity(3)
  cross_vec = attribute.to_array([0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0], rows = 3, cols = 3)
  rotation = rotation + cross_vec + cross_vec * cross_vec / (1 + cos)
  rotate_uv0 = rotation * v0.transpose()
  rotate_uv1 = rotation * v1.transpose()
  rotate_uv2 = rotation * v2.transpose()
  uv0 = attribute.to_array([rotate_uv0[0], rotate_uv0[2]], rows = 1, cols = 2)
  uv1 = attribute.to_array([rotate_uv1[0], rotate_uv1[2]], rows = 1, cols = 2)
  uv2 = attribute.to_array([rotate_uv2[0], rotate_uv2[2]], rows = 1, cols = 2)
  uv1_minus_uv0 = uv1 - uv0
  uv2_minus_uv0 = uv2 - uv0
  M = attribute.to_array([uv1_minus_uv0[0], uv2_minus_uv0[0], uv1_minus_uv0[1], uv2_minus_uv0[1]], rows = 2, cols = 2)
  return M


def baraff_witkin(x_init, x, stretchS, shearS, thickness, dt):
  anisotropic_a = attribute.to_array([1, 0], rows = 1, cols = 2)
  anisotropic_b = attribute.to_array([0, 1], rows = 1, cols = 2)
  x10 = x.row(1) - x.row(0)
  x20 = x.row(2) - x.row(0)
  F = attribute.to_array([x10[0], x10[1], x10[2], x20[0], x20[1], x20[2]], rows = 2, cols = 3)
  F = F.transpose()
  F = F * compute_rest_shape(x_init.row(0), x_init.row(1), x_init.row(2)).inverse()
  I6 = (anisotropic_a * F.transpose() * F * anisotropic_b.transpose())
  shear_energy = I6 * I6
  I5u = (F * anisotropic_a.transpose()).norm()
  I5v = (F * anisotropic_b.transpose()).norm()
  ucoeff = 1.0
  vcoeff = 1.0
  ucoeff = attribute.select(I5u < attribute(float_value = 1.0), attribute(float_value = 0.01), attribute(float_value = 1.0))
  vcoeff = attribute.select(I5v < attribute(float_value = 1.0), attribute(float_value = 0.01), attribute(float_value = 1.0))
  stretch_energy = ucoeff * (I5u - 1.0) * (I5u - 1.0) + vcoeff * (I5v - 1.0) * (I5v - 1.0)
  v01 = x_init.row(1) - x_init.row(0)
  v02 = x_init.row(2) - x_init.row(0)
  area = thickness * v01.cross(v02).norm() / 2.0
  return (stretchS * stretch_energy + shearS * shear_energy) * area * dt * dt

def baraff_witkin_modified(x_init, F, stretchS, shearS, thickness, dt):
  anisotropic_a = attribute.to_array([1, 0], rows = 1, cols = 2)
  anisotropic_b = attribute.to_array([0, 1], rows = 1, cols = 2)
  F = F.transpose()
  F = F * compute_rest_shape(x_init.row(0), x_init.row(1), x_init.row(2)).inverse()
  I6 = (anisotropic_a * F.transpose() * F * anisotropic_b.transpose())
  shear_energy = I6 * I6
  I5u = (F * anisotropic_a.transpose()).norm()
  I5v = (F * anisotropic_b.transpose()).norm()
  ucoeff = 1.0
  vcoeff = 1.0
  ucoeff = attribute.select(I5u < attribute(float_value = 1.0), attribute(float_value = 0.01), attribute(float_value = 1.0))
  vcoeff = attribute.select(I5v < attribute(float_value = 1.0), attribute(float_value = 0.01), attribute(float_value = 1.0))
  stretch_energy = ucoeff * (I5u - 1.0) * (I5u - 1.0) + vcoeff * (I5v - 1.0) * (I5v - 1.0)
  v01 = x_init.row(1) - x_init.row(0)
  v02 = x_init.row(2) - x_init.row(0)
  area = thickness * v01.cross(v02).norm() / 2.0
  return (stretchS * stretch_energy + shearS * shear_energy) * area * dt * dt



def rotation_energy(rotation, rotation_weight):
  det = rotation.determinant()
  energy = 0.5 * rotation_weight * (det - 1.0) * (det - 1.0)
  return energy

def constrained_energy(cp, ctp, dt, weight):
  distance = (cp - ctp).dot(cp - ctp)
  energy = 0.5 * weight * distance * distance
  return energy * dt * dt

def random_rotation_matrix(rng):
  # random angles in radians
  ax = rng.uniform(0.0, 2.0 * np.pi)
  ay = rng.uniform(0.0, 2.0 * np.pi)
  az = rng.uniform(0.0, 2.0 * np.pi)

  Rx = np.array([
    [1.0,        0.0,         0.0],
    [0.0, np.cos(ax), -np.sin(ax)],
    [0.0, np.sin(ax),  np.cos(ax)],
  ])

  Ry = np.array([
    [ np.cos(ay), 0.0, np.sin(ay)],
    [        0.0, 1.0,        0.0],
    [-np.sin(ay), 0.0, np.cos(ay)],
  ])

  Rz = np.array([
    [np.cos(az), -np.sin(az), 0.0],
    [np.sin(az),  np.cos(az), 0.0],
    [       0.0,         0.0, 1.0],
  ])
  return Rz @ Ry @ Rx

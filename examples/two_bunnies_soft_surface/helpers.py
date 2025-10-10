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

def extract_edge_to_vertices(faces):
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
  return np.array(edge_to_triangle_vertices, dtype=np.uint32)


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

def angle(v, w, axis):
  theta = 2.0 * (v.cross(w).dot(axis) / axis.norm()).atan2(v.dot(w) + v.norm() * w.norm())
  return theta
def edgeTheta(q0, q1, q2, q3):
  n0 = (q0 - q2).cross(q1 - q2)
  n1 = (q1 - q3).cross(q0 - q3)
  axis = q1 - q0
  theta = angle(n0, n1, axis)
  return theta


def bending(x, x_init, bendStiff):
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
  return bend_energy

def compute_rest_shape(v0, v1, v2):
  # Rest edges in 3D
  e1 = v1 - v0
  e2 = v2 - v0

  # Tangent basis by Gram–Schmidt
  t1 = e1 / e1.norm()
  n  = e1.cross(e2)
  n_norm = n.norm()
  n  = n / n_norm
  t2 = n.cross(t1)  # orthonormal to t1 in the triangle plane

  u1 = t1.dot(e1); v1p = t2.dot(e1)   # v1' relative to v0
  u2 = t1.dot(e2); v2p = t2.dot(e2)   # v2' relative to v0
  M = attribute.to_array([u1, u2, v1p, v2p], rows=2, cols=2)

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
  stretch_energy = ucoeff * (I5u - 1.0) * (I5u - 1.0) + vcoeff * (I5v - 1.0) * (I5v - 1.0)
  v01 = x_init.row(1) - x_init.row(0)
  v02 = x_init.row(2) - x_init.row(0)
  area = thickness * v01.cross(v02).norm() / 2.0
  return (stretchS * stretch_energy + shearS * shear_energy) * area * dt * dt

def inertia(x_before, vel, dt, x, mass):
  x_target = x_before + vel * dt - attribute.to_array([0.0, 9.8 * dt * dt, 0.0], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))

from pycuda.reduction import ReductionKernel
# Define the reduction kernel once
abs_max_reduce = ReductionKernel(
  np.float64,
  neutral="0",
  reduce_expr="max(a, b)",
  map_expr="fabs(x[i])",
  arguments="double *x"
)


def affine_energy(rotation):
  identity = attribute.to_array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], rows=3, cols=3)
  rot_square = rotation.transpose() * rotation
  diff = rot_square - identity
  energy = 0.5 * (diff.row(0).dot(diff.row(0)) +
                  diff.row(1).dot(diff.row(1)) +
                  diff.row(2).dot(diff.row(2)))
  return energy


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
  return dt * dt * vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam))))

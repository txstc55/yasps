import numpy as np
from yasps import attribute
import pycuda.gpuarray as gpuarray

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

def extract_edges_2_tri(triangles, edges):
  edge_indices = {}
  for i, edge in enumerate(edges):
    edge_indices[tuple(edge)] = i
  tri_list = []
  for edge in edges:
    tri_list.append([])
  for tri in triangles:
    edge0 = tuple([tri[0], tri[1]])
    edge1 = tuple([tri[1], tri[2]])
    edge2 = tuple([tri[2], tri[0]])
    edge0r = tuple([tri[1], tri[0]])
    edge1r = tuple([tri[2], tri[1]])
    edge2r = tuple([tri[0], tri[2]])
    if edge0 in edge_indices:
      tri_list[edge_indices[edge0]].append(tri)
    if edge1 in edge_indices:
      tri_list[edge_indices[edge1]].append(tri)
    if edge2 in edge_indices:
      tri_list[edge_indices[edge2]].append(tri)
    if edge0r in edge_indices:
      tri_list[edge_indices[edge0r]].append(tri)
    if edge1r in edge_indices:
      tri_list[edge_indices[edge1r]].append(tri)
    if edge2r in edge_indices:
      tri_list[edge_indices[edge2r]].append(tri)

  return np.array(tri_list)






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

def angle_energy(p0, p1, p2, p0r, p1r, p2r):
  p0p1 = p1 - p0
  p0p2 = p2 - p0
  p0p1r = p1r - p0r
  p0p2r = p2r - p0r
  p0p1_len = p0p1.norm()
  p0p2_len = p0p2.norm()
  p0p1r_len = p0p1r.norm()
  p0p2r_len = p0p2r.norm()
  cos_theta = p0p1.dot(p0p2) / (p0p1_len * p0p2_len)
  cos_theta_r = p0p1r.dot(p0p2r) / (p0p1r_len * p0p2r_len)
  energy = 0.5 * (cos_theta - cos_theta_r) * (cos_theta - cos_theta_r)
  return energy

from pycuda.reduction import ReductionKernel
# Define the reduction kernel once
abs_max_reduce = ReductionKernel(
  np.float64,
  neutral="0",
  reduce_expr="max(a, b)",
  map_expr="fabs(x[i])",
  arguments="double *x"
)

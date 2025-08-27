import numpy as np
from yasps import attribute
import pycuda.gpuarray as gpuarray


def extract_boundary_edges(F: np.ndarray) -> np.ndarray:
  """
  F: (n,3) triangle indices (any integer dtype).
  Returns: (m,2) array of undirected boundary edges with u < v.
  """
  # List all triangle edges
  E = np.vstack([
    F[:, [0, 1]],
    F[:, [1, 2]],
    F[:, [2, 0]],
  ])

  # Treat edges as undirected by sorting endpoints
  E_sorted = np.sort(E, axis=1)

  # Count occurrences of each undirected edge
  uniq, counts = np.unique(E_sorted, axis=0, return_counts=True)

  # Boundary edges occur exactly once
  boundary = uniq[counts == 1]
  return boundary







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


# add the repulsive force
def repulsive_loop(points, alpha, beta):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)
  # r = p0.dot_explicit(p0) + p1.dot_explicit(p1) + p2.dot_explicit(p2) + p3.dot_explicit(p3)
  T01 = (p1 - p0) / (p1 - p0).norm()
  T23 = (p3 - p2) / (p3 - p2).norm()
  m01 = 0.5 * (p0 + p1)
  m23 = 0.5 * (p2 + p3)
  r = ((T01.cross(m01 - m23)).norm()).pow(alpha) / (((m01 - m23).norm()).pow(beta))
  r += ((T23.cross(m23 - m01)).norm()).pow(alpha) / (((m23 - m01).norm()).pow(beta))
  # r += ((T01.cross(m01 - p2)).norm()).pow(alpha) / (((m01 - p2).norm()).pow(beta))
  # r += ((T01.cross(m01 - p3)).norm()).pow(alpha) / (((m01 - p3).norm()).pow(beta))
  # r += ((T23.cross(m23 - p0)).norm()).pow(alpha) / (((m23 - p0).norm()).pow(beta))
  # r += ((T23.cross(m23 - p1)).norm()).pow(alpha) / (((m23 - p1).norm()).pow(beta))
  return r

def smooth_loop(points):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)

  t01 = p1 - p0
  t01 = t01 / t01.norm()
  t12 = p2 - p1
  t12 = t12 / t12.norm()
  t23 = p3 - p2
  t23 = t23 / t23.norm()
  return 1.0 - t01.dot(t12) + 1.0 - t12.dot(t23)
  # e0 = p0 - 2.0 * p1 + p2
  # e0 = e0.dot(e0)
  # e1 = p1 - 2.0 * p2 + p3
  # e1 = e1.dot(e1)
  # return e0 + e1

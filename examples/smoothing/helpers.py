from yasps import attribute
def compute_edge_to_triangle_vertices(faces):
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
  return edge_to_triangle_vertices


def compute_area(p0, p1, p2):
  v1 = p1 - p0
  v2 = p2 - p0
  return v1.cross(v2).norm() / 2.0
def cot_between_edge(p0, p1, p2):
  v1 = p0 - p1
  v2 = p2 - p1
  return v1.dot(v2) / v1.cross(v2).norm()



def quadratic_bending_energy(x, x_init, bending_stiff, dt):
  x0 = x.row(0)
  x1 = x.row(1)
  x2 = x.row(2)
  x3 = x.row(3)

  x0_init = x_init.row(0)
  x1_init = x_init.row(1)
  x2_init = x_init.row(2)
  x3_init = x_init.row(3)

  area_init0 = compute_area(x0_init, x1_init, x2_init)
  area_init1 = compute_area(x1_init, x0_init, x3_init)

  c03 = cot_between_edge(x2, x1, x0)
  c04 = cot_between_edge(x3, x1, x0)
  c01 = cot_between_edge(x2, x0, x1)
  c02 = cot_between_edge(x3, x0, x1)

  k = attribute.to_array([c03 + c04, c01 + c02, -c01 - c03, -c02 - c04], rows = 1, cols = 4)
  K = k.transpose() * k
  Q = K * 3.0 / (area_init0 + area_init1)

  x_dim = attribute.to_array([x0[0], x1[0], x2[0], x3[0]], rows = 1, cols = 4)
  y_dim = attribute.to_array([x0[1], x1[1], x2[1], x3[1]], rows = 1, cols = 4)
  z_dim = attribute.to_array([x0[2], x1[2], x2[2], x3[2]], rows = 1, cols = 4)

  eb = x_dim * (Q * x_dim.transpose()) + y_dim * (Q * y_dim.transpose()) + z_dim * (Q * z_dim.transpose())
  return dt * dt * bending_stiff * eb

def inertia(x_before, vel, dt, x, mass):
  x_target = x_before + vel * dt - attribute.to_array([0.0, 0.0, 0.0], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))


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
  bend_energy = bendStiff * t * t * (x_init1 - x_init0).norm()
  return bend_energy * dt * dt

def radius_energy(x, center, target, penalty):
  len = (x - center).transpose() * (x - center)
  rlen = target * target
  diff = len - rlen
  return penalty * diff * diff

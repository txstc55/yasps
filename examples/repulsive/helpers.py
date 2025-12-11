from yasps import attribute


def inertia(x_before, vel, dt, x, mass):
  x_target = x_before + vel * dt - attribute.to_array([0.0, 0.0, 0.0], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))



def point_point(position, dHat, kappa, mass):
  # 1E-6
  p0 = position.row(0)
  p1 = position.row(1)
  d = (p1 - p0).dot(p1 - p0)
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log / (mass.row(0) + mass.row(1))

def point_edge(position, dHat, kappa, mass):
  p0 = position.row(0)
  p1 = position.row(1)
  p2 = position.row(2)
  cross = (p1 - p0).cross(p2 - p0)
  cross = cross.dot(cross)
  d = cross / ((p2 - p1).dot(p2 - p1))
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log / (mass.row(0) + mass.row(1) + mass.row(2))

def point_triangle(position, dHat, kappa, mass):
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
  return kappa * lenE * lenE * I5log * I5log / (mass.row(0) + mass.row(1) + mass.row(2) + mass.row(3))

def edge_edge(position, dHat, kappa, mass):
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
  return kappa * lenE * lenE * I5log * I5log / (mass.row(0) + mass.row(1) + mass.row(2) + mass.row(3))


def smooth_energy(x, weights, penalty, dt):
  # target = x.row(0) * weights[0] + x.row(1) * weights[1] + x.row(2) * weights[2] + x.row(3) * weights[3] + x.row(4) * weights[4]
  # diff = x.row(2) - target
  p0 = x.row(0)
  p1 = x.row(1)
  p2 = x.row(2)
  p3 = x.row(3)
  p4 = x.row(4)

  l01 = (p1 - p0).dot(p1 - p0)
  l12 = (p2 - p1).dot(p2 - p1)
  l23 = (p3 - p2).dot(p3 - p2)
  l34 = (p4 - p3).dot(p4 - p3)

  average = (l01 + l12 + l23 + l34) / 4.0
  diff = (l01 - average + l12 - average + l23 - average + l34 - average)

  return dt * dt * penalty * diff * diff

def length_energy(x, target, penalty, mass, dt, mass_scale):
  p0 = x.row(0)
  p1 = x.row(1)
  length = (p0 - p1).dot(p0 - p1)
  m0 = mass.row(0)
  m1 = mass.row(1)

  target_length = target / ((m0 + m1) / (2.0 * mass_scale))
  target_length *= target_length

  return dt * dt * penalty * (length - target_length) * (length - target_length)

def repulsive(points, alpha, beta, weight, dt):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)
  p0 = p0 / p0.norm()
  p1 = p1 / p1.norm()
  p2 = p2 / p2.norm()
  p3 = p3 / p3.norm()
  # r = p0.dot_explicit(p0) + p1.dot_explicit(p1) + p2.dot_explicit(p2) + p3.dot_explicit(p3)
  T01 = (p1 - p0) / ((p1 - p0).dot_explicit(p1 - p0)).sqrt()
  T23 = (p3 - p2) / ((p3 - p2).dot_explicit(p3 - p2)).sqrt()
  r = T01.dot_explicit(p0 - p2).pow(alpha) / (p0 - p2).norm().pow(beta)
  r += T01.dot_explicit(p0 - p3).pow(alpha) / (p0 - p3).norm().pow(beta)
  r += T01.dot_explicit(p1 - p2).pow(alpha) / (p1 - p2).norm().pow(beta)
  r += T01.dot_explicit(p1 - p3).pow(alpha) / (p1 - p3).norm().pow(beta)
  r += T23.dot_explicit(p2 - p0).pow(alpha) / (p2 - p0).norm().pow(beta)
  r += T23.dot_explicit(p2 - p1).pow(alpha) / (p2 - p1).norm().pow(beta)
  r += T23.dot_explicit(p3 - p0).pow(alpha) / (p3 - p0).norm().pow(beta)
  r += T23.dot_explicit(p3 - p1).pow(alpha) / (p3 - p1).norm().pow(beta)
  return weight * dt * dt * r

def radius_energy(x, target_radius, penalty, dt):
  r = x.dot(x)
  diff = r - target_radius * target_radius
  return dt * dt * penalty * (diff * diff)

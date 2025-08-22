from yasps import attribute
def closest_point_on_triangle(pt, v0, v1, v2):
  v0p = pt - v0
  v0v1 = v1 - v0
  v0v2 = v2 - v0
  # Dot products for barycentric
  dot00 = v0v2.dot(v0v2)
  dot01 = v0v2.dot(v0v1)
  dot02 = v0v2.dot(v0p)
  dot11 = v0v1.dot(v0v1)
  dot12 = v0v1.dot(v0p)
  denom = dot00 * dot11 - dot01 * dot01
  alpha = (dot11 * dot02 - dot01 * dot12) / denom
  beta  = (dot00 * dot12 - dot01 * dot02) / denom
  alpha = attribute.select(alpha < attribute(float_value = 0.0), attribute(float_value = 0.0), alpha)
  alpha = attribute.select(alpha > attribute(float_value = 1.0), attribute(float_value = 1.0), alpha)
  beta = attribute.select(beta < attribute(float_value = 0.0), attribute(float_value = 0.0), beta)
  beta = attribute.select(alpha + beta <= attribute(float_value = 1.0), beta, 1.0 - alpha)
  return v0 + alpha * v0v2 + beta * v0v1

def triangle_sphere_collision(v0, v1, v2, center, radius, dHat, kappa):
  # 1) Get closest point on triangle to the sphere center
  cp = closest_point_on_triangle(center, v0, v1, v2)
  to_cp = cp - center
  to_cp = to_cp / to_cp.norm() * radius + center
  d = ((cp - to_cp).dot(cp - to_cp)).sqrt()
  d = attribute.select(d < dHat, d, dHat)
  b = -((d - dHat) * (d - dHat)) * (d / dHat).log()
  return kappa * b

def inertia(x_before, vel, dt, x, mass):
  x_target = x_before + vel * dt - attribute.to_array([0.0, 9.8 * dt * dt, 0.0], rows = 1, cols = 3)
  return (0.5 * (x - x_target) * mass * (x - x_target).transpose())





def triangle_center_distance_energy(v0, v1, v2, p, velocity, DT, epsilon=1e-3):
  center = (v0 + v1 + v2) / 3.0
  normal = (v2 - v0).cross(v1 - v0).transpose()
  target_p = p + velocity * DT
  A = normal.dot(target_p - v0)
  B = normal[1]
  t = - A / B
  q = p + t * attribute.to_array([0.0, 1.0, 0.0], rows = 1, cols = 3)
  return (q - center).norm()

def tangent_energy(v0, v1, v2, p, target, velocity, DT, epsilon = 1e-3):
  center = (v0 + v1 + v2) / 3.0
  normal = (v2 - v0).cross(v1 - v0).transpose()
  normal = normal / normal.norm()
  target_p = p + velocity * DT
  A = normal.dot(target_p - v0)
  B = normal[1]
  t = - A / B
  q = target_p + t * attribute.to_array([0.0, 1.0, 0.0], rows = 1, cols = 3)
  target_direction = target - target_p
  target_direction = target_direction / target_direction.norm()
  tangent_energy = -target_direction.dot(normal) + velocity[0] * normal[0] + velocity[2] * normal[2] + 0.01 * (center - q).norm()
  return tangent_energy

def sigmoid(x):
  return 1.0 / (1.0 + attribute(float_value = 2.718281828459045).pow(-x))

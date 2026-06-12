from yasps import attribute


def normalize(v):
  return v / v.norm()


def scalar_zero():
  return attribute(float_value=0.0)


def lambda_last_h_rank2(distance2, dHat, kappa):
  I5 = distance2 / dHat
  I5log = I5.log()
  lenE = distance2 - dHat

  inner = (
    I5log * I5log * (2.0 * distance2 - 2.0 * dHat)
    + 2.0 * I5log * lenE * lenE / distance2
  )

  return -kappa * 2.0 * distance2.sqrt() * inner


###########################################################
## common friction smoothing / energy
###########################################################

def smooth_friction_phi(y2, fDhat, dt):
  eps = fDhat.sqrt() * dt
  fricDHat = fDhat * dt * dt

  y = (y2 + 1e-16).sqrt()

  smooth = (
    eps / 3.0
    + y2 / eps
    - y2 * y / (3.0 * eps * eps)
  )

  nonsmooth = y

  return attribute.select(
    y2 > fricDHat,
    nonsmooth,
    smooth
  )


def friction_energy_from_rel_dx(
  relDX,
  fDhat,
  dt,
  frictionRate,
  tangent0,
  tangent1,
  lambda_last_h
):
  u0 = tangent0.dot(relDX)
  u1 = tangent1.dot(relDX)

  y2 = u0 * u0 + u1 * u1

  phi = smooth_friction_phi(y2, fDhat, dt)

  return frictionRate * lambda_last_h * phi


###########################################################
## point triangle
###########################################################

def closest_point_coord_and_tangent_basis_pt(position):
  p = position.row(0)
  t0 = position.row(1)
  t1 = position.row(2)
  t2 = position.row(3)

  e01 = t1 - t0
  e02 = t2 - t0

  B = attribute.to_array([
    e01[0], e02[0],
    e01[1], e02[1],
    e01[2], e02[2]
  ], rows=3, cols=2)

  rhs = (p - t0).transpose()

  beta = (B.transpose() * B).inverse() * B.transpose() * rhs

  beta1 = beta[0]
  beta2 = beta[1]

  tangent0 = normalize(e01)
  tangent1 = normalize(e01.cross(e02).cross(e01))

  coord = attribute.to_array([
    beta1,
    beta2
  ], rows=1, cols=2)

  tangent_basis = attribute.to_array([
    tangent0[0], tangent0[1], tangent0[2],
    tangent1[0], tangent1[1], tangent1[2]
  ], rows=2, cols=3)

  return coord, tangent_basis


def distance2_pt_from_coord(position, coord):
  p = position.row(0)
  t0 = position.row(1)
  t1 = position.row(2)
  t2 = position.row(3)

  beta1 = coord[0]
  beta2 = coord[1]

  q = t0 + beta1 * (t1 - t0) + beta2 * (t2 - t0)

  r = p - q

  return r.dot(r)


def lambda_last_h_pt(position, coord, dHat, kappa):
  distance2 = distance2_pt_from_coord(position, coord)
  lambda_last_h = lambda_last_h_rank2(distance2, dHat, kappa)

  return lambda_last_h


def rel_dx_pt(position, old_position, coord):
  p = position.row(0)
  t0 = position.row(1)
  t1 = position.row(2)
  t2 = position.row(3)

  p_old = old_position.row(0)
  t0_old = old_position.row(1)
  t1_old = old_position.row(2)
  t2_old = old_position.row(3)

  dxp = p - p_old
  dxt0 = t0 - t0_old
  dxt1 = t1 - t1_old
  dxt2 = t2 - t2_old

  beta1 = coord[0]
  beta2 = coord[1]

  relDX = dxp - (
    dxt0
    + beta1 * (dxt1 - dxt0)
    + beta2 * (dxt2 - dxt0)
  )

  return relDX


def friction_energy_pt(
  position,
  old_position,
  fDhat,
  dt,
  frictionRate,
  coord,
  tangent0,
  tangent1,
  lambda_last_h
):
  relDX = rel_dx_pt(position, old_position, coord)

  return friction_energy_from_rel_dx(
    relDX,
    fDhat,
    dt,
    frictionRate,
    tangent0,
    tangent1,
    lambda_last_h
  )

###########################################################
## point edge
###########################################################

def closest_point_coord_and_tangent_basis_pe(position):
  p = position.row(0)
  e0 = position.row(1)
  e1 = position.row(2)

  edge = e1 - e0

  eta = (p - e0).dot(edge) / edge.dot(edge)

  tangent0 = normalize(edge)
  tangent1 = normalize(edge.cross(p - e0))

  coord = attribute.to_array([
    eta,
    scalar_zero()
  ], rows=1, cols=2)

  tangent_basis = attribute.to_array([
    tangent0[0], tangent0[1], tangent0[2],
    tangent1[0], tangent1[1], tangent1[2]
  ], rows=2, cols=3)

  return coord, tangent_basis


def distance2_pe_from_coord(position, coord):
  p = position.row(0)
  e0 = position.row(1)
  e1 = position.row(2)

  eta = coord[0]

  q = e0 + eta * (e1 - e0)

  r = p - q

  return r.dot(r)


def lambda_last_h_pe(position, coord, dHat, kappa):
  distance2 = distance2_pe_from_coord(position, coord)
  lambda_last_h = lambda_last_h_rank2(distance2, dHat, kappa)

  return lambda_last_h


def rel_dx_pe(position, old_position, coord):
  p = position.row(0)
  e0 = position.row(1)
  e1 = position.row(2)

  p_old = old_position.row(0)
  e0_old = old_position.row(1)
  e1_old = old_position.row(2)

  dxp = p - p_old
  dxe0 = e0 - e0_old
  dxe1 = e1 - e1_old

  eta = coord[0]

  relDX = dxp - (
    dxe0
    + eta * (dxe1 - dxe0)
  )

  return relDX


def friction_energy_pe(
  position,
  old_position,
  fDhat,
  dt,
  frictionRate,
  coord,
  tangent0,
  tangent1,
  lambda_last_h
):
  relDX = rel_dx_pe(position, old_position, coord)

  return friction_energy_from_rel_dx(
    relDX,
    fDhat,
    dt,
    frictionRate,
    tangent0,
    tangent1,
    lambda_last_h
  )


###########################################################
## edge edge
###########################################################

def closest_point_coord_and_tangent_basis_ee(position):
  a0 = position.row(0)
  a1 = position.row(1)
  b0 = position.row(2)
  b1 = position.row(3)

  e20 = a0 - b0
  e01 = a1 - a0
  e23 = b1 - b0

  A = attribute.to_array([
    e01.dot(e01), -e23.dot(e01),
    -e23.dot(e01), e23.dot(e23)
  ], rows=2, cols=2)

  rhs = attribute.to_array([
    -e20.dot(e01),
    e20.dot(e23)
  ], rows=2, cols=1)

  gamma = A.inverse() * rhs

  gamma1 = gamma[0]
  gamma2 = gamma[1]

  tangent0 = normalize(e01)
  tangent1 = normalize(e01.cross(e23).cross(e01))

  coord = attribute.to_array([
    gamma1,
    gamma2
  ], rows=1, cols=2)

  tangent_basis = attribute.to_array([
    tangent0[0], tangent0[1], tangent0[2],
    tangent1[0], tangent1[1], tangent1[2]
  ], rows=2, cols=3)

  return coord, tangent_basis


def distance2_ee_from_coord(position, coord):
  a0 = position.row(0)
  a1 = position.row(1)
  b0 = position.row(2)
  b1 = position.row(3)

  gamma1 = coord[0]
  gamma2 = coord[1]

  qa = a0 + gamma1 * (a1 - a0)
  qb = b0 + gamma2 * (b1 - b0)

  r = qa - qb

  return r.dot(r)


def lambda_last_h_ee(position, coord, dHat, kappa):
  distance2 = distance2_ee_from_coord(position, coord)
  lambda_last_h = lambda_last_h_rank2(distance2, dHat, kappa)

  return lambda_last_h


def rel_dx_ee(position, old_position, coord):
  a0 = position.row(0)
  a1 = position.row(1)
  b0 = position.row(2)
  b1 = position.row(3)

  a0_old = old_position.row(0)
  a1_old = old_position.row(1)
  b0_old = old_position.row(2)
  b1_old = old_position.row(3)

  dxa0 = a0 - a0_old
  dxa1 = a1 - a1_old
  dxb0 = b0 - b0_old
  dxb1 = b1 - b1_old

  gamma1 = coord[0]
  gamma2 = coord[1]

  relDX = (
    dxa0
    + gamma1 * (dxa1 - dxa0)
    - dxb0
    - gamma2 * (dxb1 - dxb0)
  )

  return relDX


def friction_energy_ee(
  position,
  old_position,
  fDhat,
  dt,
  frictionRate,
  coord,
  tangent0,
  tangent1,
  lambda_last_h
):
  relDX = rel_dx_ee(position, old_position, coord)

  return friction_energy_from_rel_dx(
    relDX,
    fDhat,
    dt,
    frictionRate,
    tangent0,
    tangent1,
    lambda_last_h
  )

###########################################################
## point point
###########################################################

def closest_point_coord_and_tangent_basis_pp(position):
  p0 = position.row(0)
  p1 = position.row(1)

  v01 = p1 - p0

  x_cross = attribute.to_array([
    0.0,
    -v01[2],
    v01[1]
  ], rows=1, cols=3)

  y_cross = attribute.to_array([
    v01[2],
    0.0,
    -v01[0]
  ], rows=1, cols=3)

  use_x = x_cross.dot(x_cross) > y_cross.dot(y_cross)

  tangent0_raw_x = attribute.select(use_x, x_cross[0], y_cross[0])
  tangent0_raw_y = attribute.select(use_x, x_cross[1], y_cross[1])
  tangent0_raw_z = attribute.select(use_x, x_cross[2], y_cross[2])

  tangent0_raw = attribute.to_array([
    tangent0_raw_x,
    tangent0_raw_y,
    tangent0_raw_z
  ], rows=1, cols=3)

  tangent0 = normalize(tangent0_raw)
  tangent1 = normalize(v01.cross(tangent0_raw))

  coord = attribute.to_array([
    scalar_zero(),
    scalar_zero()
  ], rows=1, cols=2)

  tangent_basis = attribute.to_array([
    tangent0[0], tangent0[1], tangent0[2],
    tangent1[0], tangent1[1], tangent1[2]
  ], rows=2, cols=3)

  return coord, tangent_basis


def distance2_pp_from_coord(position, coord):
  p0 = position.row(0)
  p1 = position.row(1)

  r = p0 - p1

  return r.dot(r)


def lambda_last_h_pp(position, coord, dHat, kappa):
  distance2 = distance2_pp_from_coord(position, coord)
  lambda_last_h = lambda_last_h_rank2(distance2, dHat, kappa)

  return lambda_last_h


def rel_dx_pp(position, old_position, coord):
  p0 = position.row(0)
  p1 = position.row(1)

  p0_old = old_position.row(0)
  p1_old = old_position.row(1)

  dx0 = p0 - p0_old
  dx1 = p1 - p1_old

  relDX = dx0 - dx1

  return relDX


def friction_energy_pp(
  position,
  old_position,
  fDhat,
  dt,
  frictionRate,
  coord,
  tangent0,
  tangent1,
  lambda_last_h
):
  relDX = rel_dx_pp(position, old_position, coord)

  return friction_energy_from_rel_dx(
    relDX,
    fDhat,
    dt,
    frictionRate,
    tangent0,
    tangent1,
    lambda_last_h
  )

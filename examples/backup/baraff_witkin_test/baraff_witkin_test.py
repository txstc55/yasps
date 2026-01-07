from yasps import scene
from yasps import attribute
import numpy as np
s0 = scene("scene0")
cloth = s0.addMesh("cloth")
default_positions = [
  [ 0.11397225, -0.19635172, -0.00979915],
  [-0.011099  , -0.01585265,  0.01079282],
  [ 0.13759403, -0.22779129, -0.00730297],
]

STRETCH_STIFFNESS = 335570.469799
SHEAR_STIFFNESS = 100607.114094

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

def baraff_witkin(x_init, x, stretchS, shearS, thickness):
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
  return (stretchS * stretch_energy + shearS * shear_energy) * area

def baraff_witkin_modified(x_init, F, stretchS, shearS, thickness):
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
  return (stretchS * stretch_energy + shearS * shear_energy) * area

stretch = cloth.addConstant("stretch", rows = 1, cols = 1)
stretch.updateValue(np.array([STRETCH_STIFFNESS]))
shear = cloth.addConstant("lambda", rows = 1, cols = 1)
shear.updateValue(np.array([SHEAR_STIFFNESS]))

cv = cloth.addPrimitive("vertices", numInstances = 60000)
cvp = cv.addAttribute("position", rows = 3, cols = 1)
cvp.updateValue(np.array(default_positions * 20000).flatten())
cvrp = cv.addConstant("rest_position", rows = 3, cols = 1)
cvrp.updateValue(np.array(default_positions * 20000).flatten())


ct = cloth.addPrimitive("tetrahedra", numInstances = 20000)
ct2cv = ct.addConnectivity("ct2cv", cv, np.arange(60000), 3)

ctp = ct.addAttribute("positions", through = ct2cv, source = cvp)
ctrp = ct.addAttribute("rest_positions", through = ct2cv, source = cvrp)

row0 = ctp.row(0)
row1 = ctp.row(1)
row2 = ctp.row(2)
x0 = row1 - row0
x1 = row2 - row0
F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2]], rows = 2, cols = 3)
btf = ct.addAttribute("F", computed_attribute = F)

cdg = cloth.addPrimitive("deformation_gradient", numInstances = 20000)
cdg2ct = cdg.addConnectivity("cdg2ct", ct, np.arange(20000), 1)
cdg_F = cdg.addAttribute("F", through = cdg2ct, source = btf)
cdg_F = cdg_F.resize(2, 3)

cdgrp = cdg.addAttribute("rest_positions", through = cdg2ct, source = ctrp)
cdgrp = cdgrp.resize(3, 3)

bw_original = baraff_witkin(ctrp, ctp, stretch, shear, 0.01)
ct.addAttribute("baraff_witkin_original", computed_attribute = bw_original)

bw_modified = baraff_witkin_modified(cdgrp, cdg_F, stretch, shear, 0.001)
cdg.addAttribute("baraff_witkin_modified", computed_attribute = bw_modified)

# for original, no projection, 0.11
# for original, with projection, 0.54
# for modified, no projection, 0.13
# for modified, with projection, 0.27
# s0.addEnergy(bw_original, projection_method = 1)
s0.addEnergy(bw_modified, projection_method = 0)
s0.addMinimizeTarget([cvp])




s0.minimizeEnergy(tolerance = 1e-4)

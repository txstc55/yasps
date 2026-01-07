from yasps import scene
from yasps import attribute
import numpy as np
s0 = scene("scene0")
bunny = s0.addMesh("bunny")
default_positions = [
  [ 0.11397225, -0.19635172, -0.00979915],
  [-0.011099  , -0.01585265,  0.01079282],
  [ 0.13759403, -0.22779129, -0.00730297],
  [ 0.15212258, -0.1978514 , -0.02847163]
]
POISSON_VALUE = 0.2045697005781997
YOUNG_VALUE = 10259.25455816859

MU_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))

def stable_neo_hookean(tet_position_rest, tet_position, mu, lam):
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
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam))))

def stable_neo_hookean_modified(F, TB, mu, lam):
  IB = TB.transpose().inverse()
  vol = TB.transpose().determinant() / 6.0
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam))))


mu = bunny.addConstant("mu", rows = 1, cols = 1)
mu.updateValue(np.array([MU_VALUE]))
lmbda = bunny.addConstant("lambda", rows = 1, cols = 1)
lmbda.updateValue(np.array([LAMBDA_VALUE]))

bv = bunny.addPrimitive("vertices", numInstances = 80000)
bvp = bv.addAttribute("position", rows = 3, cols = 1)
bvp.updateValue(np.array(default_positions * 20000).flatten())
bvrp = bv.addConstant("rest_position", rows = 3, cols = 1)
bvrp.updateValue(np.array(default_positions * 20000).flatten())


bt = bunny.addPrimitive("tetrahedra", numInstances = 20000)
bt2bv = bt.addConnectivity("bt2bv", bv, np.arange(80000), 4)

btp = bt.addAttribute("positions", through = bt2bv, source = bvp)
btrp = bt.addAttribute("rest_positions", through = bt2bv, source = bvrp)

row0 = btrp.row(0)
row1 = btrp.row(1)
row2 = btrp.row(2)
row3 = btrp.row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
TB = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
bttb = bt.addAttribute("TB", computed_attribute = TB)

row0 = btp.row(0)
row1 = btp.row(1)
row2 = btp.row(2)
row3 = btp.row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
btf = bt.addAttribute("F", computed_attribute = F)

bdg = bunny.addPrimitive("deformation_gradient", numInstances = 20000)
bdg2bt = bdg.addConnectivity("bdg2bt", bt, np.arange(20000), 1)
bdg_F = bdg.addAttribute("F", through = bdg2bt, source = btf)
bdg_F = bdg_F.resize(3, 3)

bdg_TB = bdg.addAttribute("TB", through = bdg2bt, source = bttb)
bdg_TB = bdg_TB.resize(3, 3)

snh_original = stable_neo_hookean(btrp, btp, mu, lmbda)
bt.addAttribute("snh_original", computed_attribute = snh_original)

snh_modified = stable_neo_hookean_modified(bdg_F, bdg_TB, mu, lmbda)
bdg.addAttribute("snh_modified", computed_attribute = snh_modified)

# for original, no projection, 0.2
# for original, with projection, 1.106
# for modified, no projection, 0.3
# for modified, with projection, 0.71
# s0.addEnergy(snh_original, projection_method = 0)
# s0.addEnergy(snh_modified, projection_method = 0)
s0.addMinimizeTarget([bvp])




s0.minimizeEnergy(tolerance = 1e-4)

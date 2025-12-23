from yasps import scene
import numpy as np

DHAT = 1.0
DT = 0.01
KAPPA = 1000.0
NUM_INSTANCES = 100000
positions = [[0, 0, 0], [0, DHAT / 2.0, 0]] * NUM_INSTANCES
positions = np.array(positions, dtype=np.float64)
indices = list(range(NUM_INSTANCES * 2))
indices = np.array(indices, dtype=np.uint32)


s0 = scene("scene0")
dhat = s0.addConstant("dhat")
dhat.updateValue([DHAT])
dt = s0.addConstant("dt")
dt.updateValue([DT])
kappa = s0.addConstant("kappa")
kappa.updateValue([KAPPA])


m = s0.addMesh("mesh0")

mv = m.addPrimitive("vertices", numInstances = NUM_INSTANCES * 2)
mvp = mv.addAttribute("position", rows = 3, cols = 1)
mvp.updateValue(positions)
print(mvp.compute().value.get().reshape(-1, 3))

pps = m.addPrimitive("pps", numInstances = 2)
pp2v = pps.addConnectivity("pp2v", mv, indices, 2)
ppp = pps.addAttribute("positions", through = pp2v, source = mvp)
print(ppp.compute().value.get().reshape(-1, 3))

def point_point(position, dHat, kappa):
  # 1E-6
  p0 = position.row(0)
  p1 = position.row(1)
  d = (p1 - p0).dot(p1 - p0)
  I5 = d / dHat
  lenE = d - dHat
  I5log = I5.log()
  return kappa * lenE * lenE * I5log * I5log

pp_energy = point_point(ppp, dhat, kappa)
pps.addAttribute("pp_energy", computed_attribute = pp_energy)
# pp_energy.compute()
# print(pp_energy.value.get())

s0.addEnergy(pp_energy, projection_method = 0)
s0.addMinimizeTarget([mvp])

x = s0.minimizeEnergy()
x = s0.minimizeEnergy()
x = s0.minimizeEnergy()
x = s0.minimizeEnergy()
x = s0.minimizeEnergy()
x = s0.minimizeEnergy()
x = s0.minimizeEnergy()
print("Solution")

tmp_mat6 = mv.addAttribute("mat6", rows =6, cols = 6)
tmp_mat6.updateValue(np.random.rand(NUM_INSTANCES * 2 * 6 * 6).astype(np.float64))
tmp_mat6 = tmp_mat6.spd()
tmp_mat6.compute()

tmp_mat15 = mv.addAttribute("mat15", rows =15, cols = 15)
tmp_mat15.updateValue(np.random.rand(NUM_INSTANCES * 2 * 15 * 15).astype(np.float64))
tmp_mat15 = tmp_mat15.spd()
tmp_mat15.compute()

tmp_mat24 = mv.addAttribute("mat24", rows =24, cols = 24)
tmp_mat24.updateValue(np.random.rand(NUM_INSTANCES * 2 * 24 * 24).astype(np.float64))
tmp_mat24 = tmp_mat24.spd()
tmp_mat24.compute()

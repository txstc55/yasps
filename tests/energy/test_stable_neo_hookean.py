from yasps import scene
from yasps import attribute
import numpy as np

# initialize some data
position = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 2.0], [0.0, 2.0, 2.0]], dtype = np.float64)

# tet_indices = np.array([[0, 1, 3, 7], [1, 2, 3, 7], [0, 1, 4, 7], [1, 5, 4, 7], [1, 2, 6, 7], [1, 5, 6, 7]])
tet_indices = np.array([[0, 1, 3, 7]])


s0 = scene("scene0")
bunny = s0.addMesh("bunny")
bunny.addAttribute("lam", rows = 1, cols = 1)
bunny.attributes["lam"].updateValue(1.0)
bunny.addAttribute("mu", rows = 1, cols = 1)
bunny.attributes["mu"].updateValue(2.0)
bunny.addAttribute("size", rows = 1, cols = 1)
bunny.attributes["size"].updateValue(1.0)

# add vertices
vertices = bunny.addPrimitive("vertices", numInstances = 8)
vertices.addAttribute("position", rows = 1, cols = 3)
vertices["position"].updateValue(position)

tets = bunny.addPrimitive("tets", numInstances = tet_indices.shape[0])
tet_to_vertex = tets.addConnectivity("tet_to_vertex", vertices, tet_indices, 4)
tet_positions = tets.addAttribute("position", through = tet_to_vertex)
tet_position_rest = tets.addAttribute("position_rest", rows = 4, cols = 3)
tet_position_rest.updateValue(tet_positions.compute().value.get() / 2.0, deepCopy = True) # make a deep copy
# here we compute the rest position deformation gradient
row0 = tet_position_rest.row(0)
row1 = tet_position_rest.row(1)
row2 = tet_position_rest.row(2)
row3 = tet_position_rest.row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
TB = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
vol = TB.determinant() / 6.0
IB = TB.inverse()
tets.addAttribute("vol", rows = 1, cols = 1)
tets["vol"].updateValue(vol.compute().value.get())
tets.addAttribute("IB", rows = 3, cols = 3)
tets["IB"].updateValue(IB.compute().value.get())
print("vol and IB")
print(vol.compute().value.get())
print(IB.compute().value.get().reshape(3, 3))

print("tet_positions")
print(tet_positions.compute().value.get())

# add deformation gradient
row0 = tet_positions.row(0)
row1 = tet_positions.row(1)
row2 = tet_positions.row(2)
row3 = tet_positions.row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
deformation = tets.addAttribute("deformation_gradient", computed_attribute = F)

# we add one more level of join for the deformation gradient, which has 1 to 1 relationship to tets
tet_deform = bunny.addPrimitive("tet_deform", numInstances = tet_indices.shape[0])
tet_deform.addConnectivity("tet_deform_to_tet", tets, np.arange(tet_indices.shape[0]), 1)

# ok we now accumulate the vol, IB and deformation gradient to tet_deform
vol = tet_deform.addAttribute("vol", through = tet_deform.connectivities["tet_deform_to_tet"])
IB = tet_deform.addAttribute("IB", through = tet_deform.connectivities["tet_deform_to_tet"])
IB.reshape(3, 3) # need to reshape
deformation = tet_deform.addAttribute("deformation_gradient", through = tet_deform.connectivities["tet_deform_to_tet"])
deformation.reshape(3, 3) # need to reshape
print("Deformation gradient")
print(deformation.compute().value.get().reshape(3, 3))

def stable_neo_hookean(mu, lam, vol, IB, F):
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  # return IC
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam))))
  # return 0.5 * mu


snh = tet_deform.addAttribute("stable_neo_hookean", computed_attribute = stable_neo_hookean(bunny["mu"], bunny["lam"], vol, IB, deformation))

print("stable neo hookean")
# print(snh.compute().value.get().reshape(3, 3))
print(snh.compute().value.get())

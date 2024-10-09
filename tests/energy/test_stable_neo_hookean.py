from yasps import scene
from yasps import attribute
import numpy as np

s0 = scene("scene0")
bunny = s0.addMesh("bunny")
bunny.addAttribute("lam", rows = 1, cols = 1)
bunny.attributes["lam"].updateValue(1.0)
bunny.addAttribute("mu", rows = 1, cols = 1)
bunny.attributes["mu"].updateValue(2.0)

# add vertices
vertices = bunny.addPrimitive("vertices", numInstances = 8)
vertices.addAttribute("position", rows = 1, cols = 3)
vertices["position"].updateValue(np.array([
    [0.0, 0.0, 0.0],  # Vertex 0
    [1.0, 0.0, 0.0],  # Vertex 1
    [1.0, 1.0, 0.0],  # Vertex 2
    [0.0, 1.0, 0.0],  # Vertex 3
    [0.0, 0.0, 1.0],  # Vertex 4
    [1.0, 0.0, 1.0],  # Vertex 5
    [1.0, 1.0, 1.0],  # Vertex 6
    [0.0, 1.0, 1.0]   # Vertex 7
], dtype = np.float64))

tets = bunny.addPrimitive("tets", numInstances = 6)
tet_to_vertex = tets.addConnectivity("tet_to_vertex", vertices, np.array([
    [0, 1, 3, 7],  # Tetrahedron 0
    [1, 2, 3, 7],  # Tetrahedron 1
    [0, 1, 4, 7],  # Tetrahedron 2
    [1, 5, 4, 7],  # Tetrahedron 3
    [1, 2, 6, 7],  # Tetrahedron 4
    [1, 5, 6, 7]   # Tetrahedron 5
]), 4)
tet_positions = tets.addAttribute("position", through = tet_to_vertex)
tet_position_rest = tets.addAttribute("position_rest", rows = 4, cols = 3)
tet_position_rest.updateValue(tet_positions.compute().value.get(), deepCopy = True) # make a deep copy


# here we compute the rest position deformation gradient
row0 = tet_position_rest.row(0)
row1 = tet_position_rest.row(1)
row2 = tet_position_rest.row(2)
row3 = tet_position_rest.row(3)

x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0

IB = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
vol = IB.determinant() / 6.0
tets.addAttribute("vol", computed_attribute = vol)
tets.addAttribute("IB", computed_attribute = IB.inverse())
# print(tets.attributes["vol"].compute().value.get())
# print(tets.attributes["IB"].compute().value.get())



def stable_neo_hookean(mu, lam, vol, IB, position):
  row0 = position.row(0)
  row1 = position.row(1)
  row2 = position.row(2)
  x0 = row1 - row0
  x1 = row2 - row0
  x2 = row3 - row0
  F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
  FI = F.transpose() * IB
  J = FI.determinant()
  lnJ = J.log()
  return 0.5 * mu * ((FI.transpose() * FI).trace() - 3.0) - mu * lnJ + 0.5 * lam * lnJ * lnJ

tets.addAttribute("energy", computed_attribute = stable_neo_hookean(bunny["mu"], bunny["lam"], tets["vol"], tets["IB"], tet_positions))
# tets["energy"].compute()
print(tets.attributes["energy"].compute().value.get())

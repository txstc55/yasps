from yasps import scene
from yasps import attribute
import numpy as np
import time
def extract_surface_triangles(tets):
  from collections import defaultdict
  face_count = defaultdict(int)

  # Define all faces of a tetrahedron (4 faces per tetrahedron)
  tet_faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
  ])
  for tet in tets:
    for face in tet_faces:
      face_vertices = tet[face]
      sorted_face = tuple(sorted(face_vertices))
      face_count[sorted_face] += 1
  # Extract faces that occur only once (surface triangles)
  surface_triangles = [list(face) for face, count in face_count.items() if count == 1]
  return np.array(surface_triangles)

##################################################
## initialize with fake data
##################################################
# position = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 2.0], [0.0, 2.0, 2.0]], dtype = np.float64)

# tet_indices = np.array([[0, 1, 3, 7], [1, 2, 3, 7], [0, 1, 4, 7], [1, 5, 4, 7], [1, 2, 6, 7], [1, 5, 6, 7]])
# # tet_indices = np.array([[0, 1, 3, 7], [1, 2, 3, 7]])

##################################################
## initialize with real data
##################################################

f = open("../data/bunny.ele", 'r')
f.readline()
tet_indices = []
for line in f:
  tet_indices.append([int(x) - 1 for x in line.split()[3:]])
f.close()
tet_indices = np.array(tet_indices)

f = open("../data/bunny.node", 'r')
f.readline()
position = []
for line in f:
  position.append([float(x) for x in line.split()[1:]])
f.close()
position = np.array(position, dtype = np.float64)

##################################################
## extract surface triangles
##################################################
surface_triangle_indices = extract_surface_triangles(tet_indices)

s0 = scene("scene0")
bunny = s0.addMesh("bunny")
bunny.addAttribute("lam", rows = 1, cols = 1)
bunny.attributes["lam"].updateValue(1000.0)
bunny.addAttribute("mu", rows = 1, cols = 1)
bunny.attributes["mu"].updateValue(2000.0)
bunny.addAttribute("size", rows = 1, cols = 1)
bunny.attributes["size"].updateValue(1.0)

# add vertices
vertices = bunny.addPrimitive("vertices", numInstances = position.shape[0])
vertex_positions = vertices.addAttribute("position", rows = 1, cols = 3)
vertices["position"].updateValue(position)

tets = bunny.addPrimitive("tets", numInstances = tet_indices.shape[0])
tet_to_vertex = tets.addConnectivity("tet_to_vertex", vertices, tet_indices, 4)
tet_positions = tets.addAttribute("position", through = tet_to_vertex)
tet_position_rest = tets.addAttribute("position_rest", rows = 4, cols = 3)
tet_position_rest.updateValue(tet_positions.compute().value.get(), deepCopy = True) # make a deep copy



# rotate each vertex by degree of x value
x = position[:, 0]  # Shape: (N,)
y = position[:, 1]  # Shape: (N,)
z = position[:, 2]  # Shape: (N,)
# Convert x-coordinates to rotation angles in degrees
theta_degrees = x  # Each x_i is the rotation angle in degrees
# Convert degrees to radians
theta_radians = np.deg2rad(theta_degrees * 100.0)  # np.deg2rad converts degrees to radians
cos_theta = np.cos(theta_radians)  # Shape: (N,)
sin_theta = np.sin(theta_radians)  # Shape: (N,)
# Compute the new y and z coordinates after rotation
y_rotated = y * cos_theta - z * sin_theta
z_rotated = y * sin_theta + z * cos_theta
# x remains the same
x_rotated = x
rotated_positions = np.column_stack((x_rotated, y_rotated, z_rotated))
# rotated_positions = position
vertices["position"].updateValue(rotated_positions)
print("Vertex positions rotated")

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
tets.addAttribute("vol", rows = 1, cols = 1)
tets["vol"].updateValue(vol.compute().value.get())
tets.addAttribute("IB", rows = 3, cols = 3)
tets["IB"].updateValue(IB.compute().value.get())

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

def stable_neo_hookean(mu, lam, vol, IB, F):
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  # return IC
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam))))


snh = tet_deform.addAttribute("stable_neo_hookean", computed_attribute = stable_neo_hookean(bunny["mu"], bunny["lam"], vol, IB, deformation))

s0.addEnergy(snh)
s0.addMinimizeTarget([vertices["position"]])


import pyvista as pv
triangles = np.array(surface_triangle_indices)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh = pv.PolyData(np.array(position), cells)
mesh2 = pv.PolyData(np.array(position * 2.0), cells)

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.add_mesh(mesh2, opacity=0.5, color='red')
# plotter.show()
plotter.show(interactive_update=True)



total_frames = 0
start = time.time()
iteration = 0
weight = 0.01
def update_position():
  global total_frames, start
  change_value = s0.minimizeEnergy()[0].get()
  for i in range(len(change_value)):
    if np.isnan(change_value[i]):
      print("NaN detected at position", i)
      exit()
  new_positions = vertex_positions.compute().value.get().flatten() - weight * change_value
  vertex_positions.updateValue(new_positions)
  # Update the mesh points
  mesh.points = new_positions.reshape(-1, 3)

  # Refresh the plotter to reflect the updated mesh
  plotter.update_coordinates(mesh.points, mesh=mesh)
  plotter.render()
  print(f"Total energy is: {sum(snh.compute().value.get() )}")


while True:
  update_position()
  total_frames += 1
  # if total_frames % 1000 == 0:
  #   # plotter.export_obj(f"bunny_{total_frames}.obj")
  #   weight *= 0.9

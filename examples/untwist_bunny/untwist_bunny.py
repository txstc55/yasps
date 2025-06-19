from yasps import scene
from yasps import attribute
import numpy as np
from yasps import minimizer # we directly use minimizer for this example
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
## read the bunny file
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
# center the bunny
center = np.mean(position, axis = 0)
position -= center

##################################################
# rotate the position and translate it a bit
# ################################################
# rotate each vertex by degree of x value
x = position[:, 0]  # Shape: (N,)
y = position[:, 1]  # Shape: (N,)
z = position[:, 2]  # Shape: (N,)
# Convert x-coordinates to rotation angles in degrees
theta_degrees = 10.0  # Each x_i is the rotation angle in degrees
# Convert degrees to radians
theta_radians = np.deg2rad(theta_degrees * 1000.0)  # np.deg2rad converts degrees to radians
cos_theta = np.cos(theta_radians)  # Shape: (N,)
sin_theta = np.sin(theta_radians)  # Shape: (N,)
# Compute the new y and z coordinates after rotation
y_rotated = y * cos_theta - z * sin_theta
z_rotated = y * sin_theta + z * cos_theta
# x remains the same
x_rotated = x
moved_position = np.column_stack((x_rotated, y_rotated, z_rotated))

##################################################
## extract surface triangles
##################################################
surface_triangle_indices = extract_surface_triangles(tet_indices)

##################################################
## set up bunny
##################################################
s0 = scene("scene0")
bunny = s0.addMesh("bunny")
bunny.addPrimitive("vertex", numInstances = position.shape[0])
bunny.addPrimitive("triangle", numInstances = surface_triangle_indices.shape[0])
bunny.addPrimitive("tet", numInstances = tet_indices.shape[0])
# bunny.addPrimitive("tet", numInstances = 5)

##################################################
## define connectivity for bunny
##################################################
tet_to_vertex = bunny.tet.addConnectivity("tet_to_vertex", bunny.vertex, tet_indices, 4)


##################################################
## add attributes to bunny
##################################################
roll = bunny.addAttribute("roll", rows = 1, cols = 1) # for rotation
pitch = bunny.addAttribute("pitch", rows = 1, cols = 1) # for rotation
yaw = bunny.addAttribute("yaw", rows = 1, cols = 1) # for rotation
mu = bunny.addAttribute("mu", rows = 1, cols = 1) # for stable neo hookean
lam = bunny.addAttribute("lam", rows = 1, cols = 1) # for stable neo hookean
translation = bunny.addAttribute("translation", rows = 3, cols = 1) # for translation

roll.updateValue([0.0])
pitch.updateValue([0.0])
yaw.updateValue([0.0])
translation.updateValue([0.0, 0.0, 0.0])
mu.updateValue([2000.0])
lam.updateValue([1000.0])
# rotation matrix from roll, pitch, yaw
rotation_matrix = attribute.to_array([
  # Row 1
  yaw.cos() * pitch.cos(),
  yaw.cos() * pitch.sin() * roll.sin() - yaw.sin() * roll.cos(),
  yaw.cos() * pitch.sin() * roll.cos() + yaw.sin() * roll.sin(),
  # Row 2
  yaw.sin() * pitch.cos(),
  yaw.sin() * pitch.sin() * roll.sin() + yaw.cos() * roll.cos(),
  yaw.sin() * pitch.sin() * roll.cos() - yaw.cos() * roll.sin(),
  # Row 3
  -pitch.sin(),
  pitch.cos() * roll.sin(),
  pitch.cos() * roll.cos()
], rows = 3, cols = 3)
rot = bunny.addAttribute("rotation_matrix", computed_attribute = rotation_matrix)

##################################################
## add position to bunny
##################################################
bunny.vertex.addAttribute("rest_position", rows = 3, cols = 1,)
bunny.vertex["rest_position"].updateValue(position)
bunny.vertex.addAttribute("current_position", rows = 3, cols = 1)
bunny.vertex["current_position"].updateValue(moved_position)
bunny.vertex.addAttribute("rotated_position", computed_attribute = (rot * (bunny.vertex["current_position"] + translation)))
tet_position = bunny.tet.addAttribute("position", through = tet_to_vertex, source = bunny.vertex["rotated_position"])
tet_position_rest = bunny.tet.addAttribute("position_rest", through = tet_to_vertex, source = bunny.vertex["rest_position"])
# print("Current position")
# print(tet_position.compute().value.get().reshape(-1, 3))
# print("Rest position")
# print(tet_position_rest.compute().value.get().reshape(-1, 3))
# exit(0)

##################################################
## define stable neo hookean energy
##################################################
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
bunny.tet.addAttribute("vol", rows = 1, cols = 1)
bunny.tet["vol"].updateValue(vol.compute().value.get())
bunny.tet.addAttribute("IB", rows = 3, cols = 3)
bunny.tet["IB"].updateValue(IB.compute().value.get())

# add deformation gradient
row0 = tet_position.row(0)
row1 = tet_position.row(1)
row2 = tet_position.row(2)
row3 = tet_position.row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
deformation = bunny.tet.addAttribute("deformation_gradient", computed_attribute = F)

# we add one more level of join for the deformation gradient, which has 1 to 1 relationship to tets
tet_deform = bunny.addPrimitive("tet_deform", numInstances = tet_indices.shape[0])
tet_deform.addConnectivity("tet_deform_to_tet", bunny.tet, np.arange(tet_indices.shape[0]), 1)

# ok we now accumulate the vol, IB and deformation gradient to tet_deform
vol = tet_deform.addAttribute("vol", through = tet_deform.connectivities["tet_deform_to_tet"])
IB = tet_deform.addAttribute("IB", through = tet_deform.connectivities["tet_deform_to_tet"])
IB = IB.resize(3, 3) # need to reshape
deformation = tet_deform.addAttribute("deformation_gradient", through = tet_deform.connectivities["tet_deform_to_tet"])
deformation = deformation.resize(3, 3) # need to reshape

def stable_neo_hookean(mu, lam, vol, IB, F):
  FI = F.transpose() * IB
  J = FI.determinant()
  IC = (FI.transpose() * FI).trace()
  I3 = IC + 1.0
  return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * I3.log() + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam)) * (J - (1.0 + 0.75 * mu / lam))))


##################################################
## add the energy
##################################################
bunny.vertex.addAttribute("position_penalty", computed_attribute = (bunny.vertex["rotated_position"] - bunny.vertex["rest_position"]).dot(bunny.vertex["rotated_position"] - bunny.vertex["rest_position"]))
snh = tet_deform.addAttribute("stable_neo_hookean", computed_attribute = stable_neo_hookean(bunny["mu"], bunny["lam"], vol, IB, deformation))

minimizer0 = minimizer()
minimizer1 = minimizer()

minimizer0.addEnergy(bunny.vertex["position_penalty"])
minimizer1.addEnergy(snh)

minimizer0.addWrt([bunny["roll"], bunny["pitch"], bunny["yaw"], bunny["translation"]])
minimizer1.addWrt([bunny.vertex["current_position"]])

minimizer0.generateHessianAndGradient()
minimizer1.generateHessianAndGradient()


##################################################
## plot the result
##################################################
import pyvista as pv
triangles = np.array(surface_triangle_indices)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh_rest = pv.PolyData(position, cells)
mesh_moved = pv.PolyData(moved_position, cells)

plotter = pv.Plotter()
plotter.add_mesh(mesh_rest, opacity=0.5, color='blue')
plotter.add_mesh(mesh_moved, opacity=1.0, color='red')
plotter.show(interactive_update=True)



total_frames = 0
iteration = 0
weight = 0.1
def update_position():
  global total_frames
  result0 = minimizer0.computeSolution()
  result1 = minimizer1.computeSolution()
  d_roll = result0[0].get()
  d_pitch = result0[1].get()
  d_yaw = result0[2].get()
  d_translation = result0[3].get().flatten()
  d_position = result1[0].get().flatten()
  roll_new_value = bunny["roll"].value.get() - weight * d_roll
  pitch_new_value = bunny["pitch"].value.get() - weight * d_pitch
  yaw_new_value = bunny["yaw"].value.get() - weight * d_yaw
  translation_new_value = bunny["translation"].value.get().flatten() - weight * d_translation
  position_new_value = bunny.vertex["current_position"].value.get().flatten() - 0.1 * d_position
  bunny["pitch"].updateValue(roll_new_value)
  bunny["roll"].updateValue(pitch_new_value)
  bunny["yaw"].updateValue(yaw_new_value)
  bunny["translation"].updateValue(translation_new_value)
  bunny.vertex["current_position"].updateValue(position_new_value)
  new_positions = bunny.vertex["rotated_position"].compute().value.get().flatten()
  # Update the mesh points
  mesh_moved.points = new_positions.reshape(-1, 3)
  # Refresh the plotter to reflect the updated mesh
  plotter.update_coordinates(mesh_moved.points, mesh=mesh_moved)
  plotter.render()
  print(f"Iteration {total_frames}")
  print(f"Total energy is: {sum(bunny.vertex["position_penalty"].compute().value.get())}")

# save the original
mesh_moved.save(f"bunny_result/bunny_untwisted_{total_frames}_unrotated.obj")
mesh_rest.save("bunny_result/bunny_rest.obj")
while True:
  update_position()
  exit()
  total_frames += 1
  if total_frames % 100 == 0:
    mesh_moved.save(f"bunny_result/bunny_untwisted_{total_frames}_unrotated.obj")
  # if total_frames == 500:
  #   exit(0)

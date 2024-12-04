from yasps import scene
from yasps import attribute
import numpy as np
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
moved_position = np.column_stack((x_rotated, y_rotated, z_rotated))


# roll = 2.1
# pitch = 3.0
# yaw = 0.1
# translation = np.array([50, 30.0, 20.0], dtype = np.float64)
# R_z = np.array([
#   [np.cos(yaw), -np.sin(yaw), 0],
#   [np.sin(yaw),  np.cos(yaw), 0],
#   [0,            0,           1]
# ])
# R_y = np.array([
#   [np.cos(pitch), 0, np.sin(pitch)],
#   [0,             1, 0],
#   [-np.sin(pitch),0, np.cos(pitch)]
# ])
# R_x = np.array([
#   [1, 0,           0],
#   [0, np.cos(roll), -np.sin(roll)],
#   [0, np.sin(roll),  np.cos(roll)]
# ])
# # Combined rotation matrix
# R = R_z @ R_y @ R_x  # Equivalent to np.dot(R_z, np.dot(R_y, R_x))
# moved_position = np.dot(position, R.T) + translation

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

##################################################
## add attributes to bunny
##################################################
roll = bunny.addAttribute("roll", rows = 1, cols = 1) # for rotation
pitch = bunny.addAttribute("pitch", rows = 1, cols = 1) # for rotation
yaw = bunny.addAttribute("yaw", rows = 1, cols = 1) # for rotation

translation = bunny.addAttribute("translation", rows = 3, cols = 1) # for translation
roll.updateValue([0.0])
pitch.updateValue([0.0])
yaw.updateValue([0.0])
translation.updateValue([0.0, 0.0, 0.0])
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

##################################################
## add the energy
##################################################
bunny.vertex.addAttribute("position_penalty", computed_attribute = (bunny.vertex["rotated_position"] - bunny.vertex["rest_position"]).dot(bunny.vertex["rotated_position"] - bunny.vertex["rest_position"]))
print("Energy at start:", sum(bunny.vertex["position_penalty"].compute().value.get()))
s0.addEnergy(bunny.vertex["position_penalty"])
s0.addMinimizeTarget([bunny["roll"], bunny["pitch"], bunny["yaw"], bunny["translation"]])


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
  result = s0.minimizeEnergy()
  d_roll = result[0].get()
  d_pitch = result[1].get()
  d_yaw = result[2].get()
  d_translation = result[3].get().flatten()
  roll_new_value = bunny["roll"].value.get() - weight * d_roll
  pitch_new_value = bunny["pitch"].value.get() - weight * d_pitch
  yaw_new_value = bunny["yaw"].value.get() - weight * d_yaw
  translation_new_value = bunny["translation"].value.get().flatten() - weight * d_translation
  bunny["pitch"].updateValue(roll_new_value)
  bunny["roll"].updateValue(pitch_new_value)
  bunny["yaw"].updateValue(yaw_new_value)
  bunny["translation"].updateValue(translation_new_value)
  new_positions = bunny.vertex["rotated_position"].compute().value.get().flatten()
  # Update the mesh points
  mesh_moved.points = new_positions.reshape(-1, 3)
  # Refresh the plotter to reflect the updated mesh
  plotter.update_coordinates(mesh_moved.points, mesh=mesh_moved)
  plotter.render()
  print(f"Total energy is: {sum(bunny.vertex["position_penalty"].compute().value.get())}")


while True:
  update_position()
  total_frames += 1
  # if total_frames % 1000 == 0:
  #   # plotter.export_obj(f"bunny_{total_frames}.obj")
  #   weight *= 0.9

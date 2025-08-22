import json
import numpy as np

skeleton_json = {}
skin_json = {}

with open("../data/OculusHand_L.fbx.bones.json", "r") as f:
  skeleton_json = json.load(f)

with open("../data/OculusHand_L.fbx.vertices.json", "r") as f:
  skin_json = json.load(f)

skeleton_points = []
skeleton_lines = []
# now we first get the bones positions
for name, bone in skeleton_json.items():
  parent = bone["parent"]
  # load the two matrices
  M_rest_global  = np.array(bone["matrix_rest"])    # bone.matrix_local for the current bone, but technically this is the global position
  M_delta_local  = np.array(bone["matrix_local"])   # your parent⁻¹ · bone.matrix_local

  θ = bone["theta"]
  R = np.array([
    [ np.cos(θ), -np.sin(θ), 0, 0],
    [ np.sin(θ),  np.cos(θ), 0, 0],
    [         0,           0, 1, 0],
    [         0,           0, 0, 1],
  ])
  # choose left- or right-multiplication based on whether you want to pre- or post-rotate:
  M_delta_local = M_delta_local @ R

  # parent’s current global (or identity for root)
  if parent:
    P_current = skeleton_json[parent]["matrix_global_current"]
  else:
    P_current = np.eye(4)

  # build your globals correctly
  G_rest    = M_rest_global
  G_current = P_current @ M_delta_local

  skeleton_json[name]["matrix_global_rest"]    = G_rest
  skeleton_json[name]["matrix_global_current"] = G_current

  # now transform head/tail (these are in bone-local coords)
  h = np.append(bone["head"], 1.0)
  t = np.append(bone["tail"], 1.0)
  skeleton_points.append((G_current @ h)[:3])
  skeleton_points.append((G_current @ t)[:3])
  skeleton_lines.append([2, len(skeleton_points)-2, len(skeleton_points)-1])

  if θ != 0:
    ## print the name and the global matrix
    print(f"Bone Name: {name}")
    print(f"Global Matrix:\n{G_current}")



skeleton_points = np.array(skeleton_points)

surface_vertices = []
# now we process the skinning data
for vertex in skin_json:
  weights = vertex["weights"]
  rest_position = np.array(vertex["rest_position"])
  rest_position = np.append(rest_position, 1.0)
  pos = np.zeros(3)
  for bone in weights:
    # get the bone name
    name = bone["bone"]
    # get the weight
    weight = bone["weight"]
    bone_matrix_rest = skeleton_json[name]["matrix_rest"]
    bone_matrix_current = skeleton_json[name]["matrix_global_current"]
    # get the global matrix
    pos += (weight * (bone_matrix_current @ np.linalg.inv(bone_matrix_rest) @ rest_position))[:3]
  if (len(weights) == 1):
    print(vertex["rest_position"])
  surface_vertices.append(pos)

# exit()


surface_energy = [1.0 / (pos.dot(pos)) for pos in surface_vertices]
surface_vertices = np.array(surface_vertices)


# print(surface_vertices)




import pyvista as pv

# show all the surface vertices
# Create a PolyData object with the skeleton_points
point_cloud = pv.PolyData(surface_vertices)

# You can add spheres to make the surface_vertices more visible (optional)
point_cloud["size"] = np.full(surface_vertices.shape[0], 10.0)  # optional scalar array

# Plotting
plotter = pv.Plotter()
plotter.add_mesh(point_cloud, color='red', point_size=5, render_points_as_spheres=True)
plotter.show()



# Create PolyData
line_mesh = pv.PolyData()
line_mesh.points = skeleton_points
line_mesh.lines = skeleton_lines
# Plot
plotter = pv.Plotter()
plotter.add_mesh(line_mesh, color="blue", line_width=2)
plotter.show()

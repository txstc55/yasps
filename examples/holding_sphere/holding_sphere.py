from posix import fork
from yasps.scene import scene
from yasps.attribute import attribute
import numpy as np

################################################################################################
## We read the hand json file
## and do the index reorientation
################################################################################################
import json
skeleton_json = {}
with open("../data/OculusHand_L.fbx.bones.json", "r") as f:
  skeleton_json = json.load(f)

max_level = 0
for name, bone in skeleton_json.items():
  parent = bone["parent"]
  if parent:
    skeleton_json[name]["level"] = skeleton_json[parent]["level"] + 1
    max_level = max(max_level, skeleton_json[name]["level"])
  else:
    skeleton_json[name]["level"] = 0
  M_rest_global  = np.array(bone["matrix_rest"])    # bone.matrix_local for the current bone, but technically this is the global position
  M_rest_global_inv = np.linalg.inv(M_rest_global) # directly inverse the rest global matrix for later use
  skeleton_json[name]["matrix_rest_global_inv"] = M_rest_global_inv

print("Bone max level is: ", max_level)
# ok now we need to do an indexing based on level
index = 0
index_local = 0
bones_leveled = []
for i in range(max_level + 1):
  bones_leveled.append([])
  for name, bone in skeleton_json.items():
    if bone["level"] == i:
      bone["index"] = index
      bone["index_local"] = index_local
      index += 1
      index_local += 1
      bones_leveled[i].append(bone)
  index_local = 0

for level in bones_leveled:
  print([f"{x['name']}, {x['index']}" for x in level])


################################################################################################
## We read the skin vertices
## and due to the fact that we reorient the bones, we need to map the correct indices
## furthermore, we need to orient the vertices so that they are categorized
## by the number of bones they are attached to
################################################################################################
skin_vertex_json = {}
skin_vertex_categorized = []
with open("../data/OculusHand_L.fbx.vertices.json", "r") as f:
  skin_vertex_json = json.load(f)

max_affected_num_bones = max([len(x["weights"]) for x in skin_vertex_json])
print(max_affected_num_bones)
for _ in range(max_affected_num_bones):
  skin_vertex_categorized.append([])

total_vertices = 0
for i in range(len(skin_vertex_json)):
  vertex = skin_vertex_json[i]
  affected_num_bones = len(vertex["weights"])
  skin_vertex_categorized[affected_num_bones - 1].append(vertex)
  vertex["connected_bones"] = [skeleton_json[x["bone"]]["index"] for x in vertex["weights"]]
  vertex["bone_weights"] = [x["weight"] for x in vertex["weights"]]
  total_vertices += 1
print("Total vertices:", total_vertices)
print("Skin vertex categorized sizes")
print([len(x) for x in skin_vertex_categorized])
################################################################################################
## initialize scene
################################################################################################
s0 = scene("scene0")
handMesh = s0.addMesh("handMesh")

################################################################################################
## here we construct the level of bones
################################################################################################
bl_primitives = []
for level in range(len(bones_leveled)):
  bl = handMesh.addPrimitive(f"bone_level_{level}", len(bones_leveled[level]))
  bl.addAttribute("matrix_rest", rows = 4, cols = 4)
  bl.addAttribute("matrix_local", rows = 4, cols = 4)
  matrix_rest = np.array([np.array(x["matrix_rest"]) for x in bones_leveled[level]]).flatten()
  matrix_local = np.array([np.array(x["matrix_local"]) for x in bones_leveled[level]]).flatten()
  bl["matrix_rest"].updateValue(matrix_rest) # this is the rest global matrix for each bone
  bl["matrix_local"].updateValue(matrix_local) # this is the rest local matrix for each bone

  ## set the rotation of the bones
  bl.addAttribute("theta", rows = 1, cols = 1)
  bl.addAttribute("theta_target", rows = 1, cols = 1)
  bl["theta"].updateValue([x["theta"] for x in bones_leveled[level]])
  bl["theta_target"].updateValue(np.zeros(len(bones_leveled[level])))
  sin_theta = bl["theta"].sin()
  cos_theta = bl["theta"].cos()
  rotation_matrix = [
     cos_theta, -sin_theta, 0, 0,
     sin_theta,  cos_theta, 0, 0,
             0,           0, 1, 0,
             0,           0, 0, 1,
  ]
  rotation_matrix_derivative = [
      -sin_theta, -cos_theta, 0, 0,
        cos_theta, -sin_theta, 0, 0,
              0,           0, 0, 0,
              0,           0, 0, 0,
  ]
  rotation_matrix = attribute.to_array(rotation_matrix, rows = 4, cols =4)
  rotation_matrix_derivative = attribute.to_array(rotation_matrix_derivative, rows = 4, cols =4)
  local_rotation = bl["matrix_local"] * rotation_matrix
  local_rotation_derivative = bl["matrix_local"] * rotation_matrix_derivative

  if level == 0:
    bl.addAttribute("matrix_global_current", computed_attribute = local_rotation)
    bl.addAttribute("matrix_global_current_derivative", computed_attribute = local_rotation)
    bl.addAttribute("const_matrix", computed_attribute = attribute.to_array([0.0, 0.0, 0.0, 1.0], rows = 4, cols = 1))
    # bl.addAttribute("const_matrix_resized", computed_attribute = bl["const_matrix"].resize(2, 2))
  else:
    bl_parent = handMesh.primitives[f"bone_level_{level - 1}"]
    connectivity = [skeleton_json[x["parent"]]["index_local"] for x in bones_leveled[level]]
    # names = [x["parent"] for x in bones_leveled[level]]
    # print(connectivity)
    # print(names)
    bl_connectivity = bl.addConnectivity(f"bone_level_{level}_to_parent", bl_parent, connectivity, 1)
    bl_parent_matrix_global_current = bl.addAttribute("parent_matrix_global_current", through = bl_connectivity, source = bl_parent["matrix_global_current"]).resize(4, 4)
    bl_parent_matrix_global_current_derivative = bl.addAttribute("parent_matrix_global_current_derivative", through = bl_connectivity, source = bl_parent["matrix_global_current_derivative"]).resize(4, 4)
    bl.addAttribute("matrix_global_current", computed_attribute = bl_parent_matrix_global_current * local_rotation)
    bl.addAttribute("matrix_global_current_derivative", computed_attribute = bl_parent_matrix_global_current_derivative * local_rotation_derivative)
    bl.addAttribute("const_matrix", computed_attribute = attribute.to_array([0.0, 0.0, 0.0, 1.0], rows = 4, cols = 1))
    # bl.addAttribute("const_matrix_resized", computed_attribute = bl["const_matrix"].resize(2, 2))
    print(bl["const_matrix"].compute().value.get())
  bl_primitives.append(bl)

################################################################################################
## here we construct a union of all bones
################################################################################################
bones_union = handMesh.addPrimitiveUnion("bones", bl_primitives)
bones_union.addAttribute("matrix_global_current")
bones_union.addAttribute("matrix_rest")
bones_union.addAttribute("matrix_global_current_derivative")

# print(bones_union["matrix_global_current"].compute().value.get().reshape((-1, 4, 4)))
# exit()
# energy = bones_union["matrix_global_current"][0]
# bones_union.addAttribute("bone_energy", computed_attribute = energy)
# s0.addEnergy(energy)
# s0.addMinimizeTarget([handMesh.bone_level_1["theta"], handMesh.bone_level_2["theta"]])
# exit()

# bones_union.addAttribute("const_matrix")
# print(bones_union["matrix_global_current"].compute().value.get())
# exit()

# print(bones_union["const_matrix"].compute().value.get())
# print([x.fullName for x in bones_union["matrix_global_current"].children])


################################################################################################
## here we construct the levels of skin vertices
################################################################################################
vw_primitives = []
print(len(skin_vertex_categorized))
vw_positions = []

for i in range(len(skin_vertex_categorized)):
  vw = handMesh.addPrimitive(f"skin_with_{i + 1}_weights", len(skin_vertex_categorized[i]))
  # print(len(skin_vertex_categorized[i]))
  # add rest position
  vw.addAttribute("rest_position", rows = 3, cols = 1)
  vw.addAttribute("rest_position_extended", computed_attribute = attribute.to_array([vw["rest_position"].row(0), vw["rest_position"].row(1), vw["rest_position"].row(2), 1.0], rows = 4, cols = 1))
  vw["rest_position"].updateValue(np.array([x["rest_position"] for x in skin_vertex_categorized[i]]).flatten())
  # add weights
  vw.addAttribute("weights", rows = i + 1, cols = 1)
  vw["weights"].updateValue(np.array([x["bone_weights"] for x in skin_vertex_categorized[i]]).flatten())
  v2b = vw.addConnectivity("skin_to_bone", bones_union, np.array([x["connected_bones"] for x in skin_vertex_categorized[i]]).flatten(), i + 1)
  vw.addAttribute("bones_matrix_rest", through = v2b, source = bones_union["matrix_rest"])
  vw.addAttribute("bones_matrix_current", through = v2b, source = bones_union["matrix_global_current"])
  vw.addAttribute("bones_matrix_current_derivative", through = v2b, source = bones_union["matrix_global_current_derivative"])

  vw_current_position = 0.0 * vw["rest_position_extended"] # this is 4 by 1
  vw_current_position_derivative = 0.0 * vw["rest_position_extended"] # this is 4 by 1
  for j in range(i + 1):
    # get the matrix for rest
    mat_rest = vw["bones_matrix_rest"].row(j).resize(4, 4)
    mat_rest_inv = mat_rest.inverse()
    projected_global = mat_rest_inv * vw["rest_position_extended"]
    transformed = vw["bones_matrix_current"].row(j).resize(4, 4) * projected_global
    transformed_derivative = vw["bones_matrix_current_derivative"].row(j).resize(4, 4) * projected_global
    weighted_transformed = vw["weights"][j] * transformed
    weighted_transformed_derivative = vw["weights"][j] * transformed_derivative
    vw_current_position += weighted_transformed
    vw_current_position_derivative += weighted_transformed_derivative
  vw.addAttribute("current_position", computed_attribute = attribute.to_array([vw_current_position.row(0), vw_current_position.row(1), vw_current_position.row(2)], rows = 3, cols = 1))
  vw.addAttribute("current_position_derivative", computed_attribute = attribute.to_array([vw_current_position_derivative.row(0), vw_current_position_derivative.row(1), vw_current_position_derivative.row(2)], rows = 3, cols = 1))
  # vw_positions += (vw["current_position"].compute().value.get().flatten().tolist())
  # if i == 0:
  #   print(vw["current_position"].compute().value.get().reshape(-1, 3))


  vw_primitives.append(vw)
  # print(vw.numInstances)
# ################################################################################################
# ## here we construct the union of skin vertices
# ################################################################################################
vertices_union = handMesh.addPrimitiveUnion("vertices_union", vw_primitives)
# print(vertices_union.numInstances)
vp = vertices_union.addAttribute("current_position")
# print(vp.compute().value.get().reshape(-1, 3))
# exit()


vertices_union.addAttribute("current_position_derivative")

# let's create a fake energy
energy = 1.0 / (vertices_union["current_position"].dot(vertices_union["current_position"]))
vertices_union.addAttribute("surface_repulse_energy", computed_attribute = energy)
s0.addEnergy(energy)
s0.addMinimizeTarget([handMesh.bone_level_1["theta"], handMesh.bone_level_2["theta"]])
exit()




















# surface_vertices = vertices_union["current_position"].compute().value.get().reshape(-1, 3)
# print(surface_vertices[40, :])

# print(surface_vertices[79, :])





# import pyvista as pv

# # show all the surface vertices
# # Create a PolyData object with the skeleton_points
# point_cloud = pv.PolyData(surface_vertices)

# # You can add spheres to make the surface_vertices more visible (optional)
# point_cloud["size"] = np.full(surface_vertices.shape[0], 10.0)  # optional scalar array

# # Plotting
# plotter = pv.Plotter()
# plotter.add_mesh(point_cloud, color='red', point_size=5, render_points_as_spheres=True)
# plotter.show()

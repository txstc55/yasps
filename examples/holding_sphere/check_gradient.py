import numpy as np
computed_gradient = np.load("gradient_check_repulse_energy.npz")
computed_gradient = computed_gradient["arr_0"]
computed_gradient_bone = np.load("gradient_check_bone_energy.npz")
computed_gradient_bone = computed_gradient_bone["arr_0"]
# print(computed_gradient_bone)
# exit()
computed_indices = np.load("output_indices.npz")
outputIndices = computed_indices["outputIndices"]
import json

################################################################################
## separate bones
################################################################################
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

index = 0
index_local = 0
bones_leveled = []
bones_flattened = []
for i in range(max_level + 1):
  bones_leveled.append([])
  for name, bone in skeleton_json.items():
    if bone["level"] == i:
      bone["index"] = index
      bone["index_local"] = index_local
      index += 1
      index_local += 1
      bones_leveled[i].append(bone)
      bones_flattened.append(bone)
  index_local = 0
print("total level")
print(len(bones_leveled))
print("All levels")
for level in bones_leveled:
  print([(bone["name"], bone["index"]) for bone in level])


for i in range(max_level + 1):
  if i == 0:
    # we add the global matrix current with
    for bone in bones_leveled[i]:
      bone["matrix_global_current"] = np.eye(4)
      bone["matrix_global_current_derivative_lvl1"] = np.zeros((4, 4))
      bone["matrix_global_current_derivative_lvl2"] = np.zeros((4, 4))
      # print(bone["matrix_global_current"])
      # print()
  else:
    for bone in bones_leveled[i]:
      # we first get the parent global current
      parent_name = bone["parent"]
      parent_bone = bones_flattened[skeleton_json[parent_name]["index"]]
      assert parent_bone["name"] == parent_name
      bone["parent_matrix_global_current"] = parent_bone["matrix_global_current"]
      theta = bone["theta"]
      rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ])
      matrix_local = bone["matrix_local"]
      # compute the global rotation
      bone["matrix_global_current"] = bone["parent_matrix_global_current"] @ matrix_local @ rotation_matrix
      rotation_matrix_gradient = np.array([
        [-np.sin(theta), -np.cos(theta), 0, 0],
        [np.cos(theta), -np.sin(theta), 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
      ])
      if i == 1:
        bone["matrix_global_current_derivative_lvl1"] = bone["parent_matrix_global_current"] @ matrix_local @ rotation_matrix_gradient
        bone["matrix_global_current_derivative_lvl2"] = np.zeros((4, 4))
        merged_derivative = np.zeros((16, 2))
        merged_derivative[:, 0] = np.array(bone["matrix_global_current_derivative_lvl1"]).flatten()
        # print(merged_derivative)
        # print()
      elif i == 2:
        # we get the two gradients wrt theta at level 1 and level 2
        bone["matrix_global_current_derivative_lvl2"] = bone["parent_matrix_global_current"] @ matrix_local @ rotation_matrix_gradient
        bone["matrix_global_current_derivative_lvl1"] = parent_bone["matrix_global_current_derivative_lvl1"] @ matrix_local @ rotation_matrix
        merged_derivative = np.zeros((16, 2))
        merged_derivative[:, 1] = np.array(bone["matrix_global_current_derivative_lvl1"]).flatten()
        merged_derivative[:, 0] = np.array(bone["matrix_global_current_derivative_lvl2"]).flatten()
        # print(merged_derivative)
        # print()
      else:
        # we get the two gradients wrt theta at level 1 and level 2
        bone["matrix_global_current_derivative_lvl2"] = parent_bone["matrix_global_current_derivative_lvl2"] @ matrix_local @ rotation_matrix
        bone["matrix_global_current_derivative_lvl1"] = parent_bone["matrix_global_current_derivative_lvl1"] @ matrix_local @ rotation_matrix
        merged_derivative = np.zeros((16, 2))
        merged_derivative[:, 1] = np.array(bone["matrix_global_current_derivative_lvl1"]).flatten()
        merged_derivative[:, 0] = np.array(bone["matrix_global_current_derivative_lvl2"]).flatten()
        # print(merged_derivative)
        # print()
################################################################################
## separate vertices
################################################################################
skin_vertex_json = {}
skin_vertex_categorized = []
with open("../data/OculusHand_L.fbx.vertices.json", "r") as f:
  skin_vertex_json = json.load(f)

max_affected_num_bones = max([len(x["weights"]) for x in skin_vertex_json])
# print(max_affected_num_bones)
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


gradient_per_skin_vertex = len(computed_gradient) // total_vertices
print("Gradient length", len(computed_gradient))
print("Gradient per skin vertex:", gradient_per_skin_vertex)

################################################################################
## manually compute gradient
################################################################################
total_count = 0
for vertices in skin_vertex_categorized:
  for vertex in vertices:
    connected_bones = vertex["connected_bones"]
    bone_weights = vertex["bone_weights"]

    rest_pos = vertex["rest_position"]
    rest_pos_expanded = np.append(rest_pos, 1.0)
    # ok now we need to compute the gradient
    pos = np.zeros(3)
    for i in range(len(connected_bones)):
      index = connected_bones[i]
      weight = bone_weights[i]
      bone = bones_flattened[index]
      bone_matrix_rest = bone["matrix_rest"]
      bone_matrix_current = bone["matrix_global_current"]
      bone_matrix_rest_inv = np.linalg.inv(bone_matrix_rest)
      projected_pos = bone_matrix_rest_inv @ rest_pos_expanded
      pos = pos + weight * (bone_matrix_current @ projected_pos)[:3]
    # print(pos)
    # ok now that we have the pos
    # we will compute the gradient
    grad = np.zeros((7, 2))
    for i in range(len(connected_bones)):
      index = connected_bones[i]
      weight = bone_weights[i]
      bone = bones_flattened[index]
      # print(bone["name"])
      bone_matrix_rest = bone["matrix_rest"]
      bone_matrix_rest_inv = np.linalg.inv(bone_matrix_rest)
      projected_pos = bone_matrix_rest_inv @ rest_pos_expanded

      bone_matrix_derivative_lvl1 = bone["matrix_global_current_derivative_lvl1"]
      bone_matrix_derivative_lvl2 = bone["matrix_global_current_derivative_lvl2"]

      derivative_lvl1 = (bone_matrix_derivative_lvl1 @ projected_pos)[:3]
      derivative_lvl2 = (bone_matrix_derivative_lvl2 @ projected_pos)[:3]

      derivative_lvl1 = -2 * weight* (pos / (pos.dot(pos) ** 2)).dot(derivative_lvl1)
      derivative_lvl2 = -2 * weight * (pos / (pos.dot(pos) ** 2)).dot(derivative_lvl2)
      grad[i, 0] = derivative_lvl1
      grad[i, 1] = derivative_lvl2
    # if (len(connected_bones) == 2) and connected_bones[0] > 0 and connected_bones[0] < 7:
    summation_ours = np.sum(grad)
    summation_produced = sum(computed_gradient[total_count * 14: (total_count + 1) * 14])
    assert (summation_ours - summation_produced) < 1e-6, f"summation not equal, {summation_ours} != {summation_produced}"

    summation_squares_ours = np.sum(grad ** 2)
    summation_squares_produced = sum([x**2 for x in computed_gradient[total_count * 14: (total_count + 1) * 14]])
    assert (summation_squares_ours - summation_squares_produced) < 1e-6, f"summation squares not equal, {summation_squares_ours} != {summation_squares_produced}"

    # ok now we need to get the corresponding indices
    indices_ours = np.zeros(14, dtype=int)
    for i in range(len(connected_bones)):
      index = connected_bones[i]
      weight = bone_weights[i]
      bone = bones_flattened[index]
      if index == 0:
        continue
      elif index < 7:
        indices_ours[i * 2] = index
      else:
        parent_bone = bone
        while parent_bone["index"] > 0:
          new_index = parent_bone["index"]
          if new_index < 7:
            indices_ours[i * 2 + 1] = new_index
          elif new_index < 12:
            indices_ours[i * 2] = new_index
          parent_bone = skeleton_json[parent_bone["parent"]]


    # print("--------------------------------------------------------")
    assert (np.linalg.norm(indices_ours - outputIndices[total_count * 14 : (total_count + 1) * 14]) == 0), f"indices not equal, {indices_ours} != {outputIndices[total_count * 14 : (total_count + 1) * 14]}"
    # print("--------------------------------------------------------")

    total_count += 1
    # exit()

  # exit()

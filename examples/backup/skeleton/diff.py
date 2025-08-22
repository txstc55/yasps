import json
import numpy as np

# --- Read Data ---
with open("../data/OculusHand_L.fbx.bones.json", "r") as f:
  skeleton_json = json.load(f)

with open("../data/OculusHand_L.fbx.vertices.json", "r") as f:
  skin_json = json.load(f)
  print(f"There are {len(skin_json)} vertices")

# --- Compute Skeleton Global Matrices ---
def compute_global_matrices(skeleton_json):
  for name, bone in skeleton_json.items():
    parent = bone["parent"]
    M_rest_global = np.array(bone["matrix_rest"])
    M_delta_local = np.array(bone["matrix_local"])

    θ = bone["theta"]
    R = np.array([
        [np.cos(θ), -np.sin(θ), 0, 0],
        [np.sin(θ),  np.cos(θ), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    M_delta_local = M_delta_local @ R

    if parent:
      P_current = skeleton_json[parent]["matrix_global_current"]
    else:
      P_current = np.eye(4)

    G_rest = M_rest_global
    G_current = P_current @ M_delta_local

    skeleton_json[name]["matrix_global_rest"] = G_rest
    skeleton_json[name]["matrix_global_current"] = G_current

# Compute globals initially
compute_global_matrices(skeleton_json)

# --- Compute vertex energy and derivative ---
def compute_energy_and_derivative(vertex_index, skeleton_json, skin_json):
  vertex = skin_json[vertex_index]
  rest_pos = np.array(vertex["rest_position"])
  rest_homog = np.append(rest_pos, 1.0)
  weights = vertex["weights"]

  pos = np.zeros(3)

  # Store individual dpos/dtheta_i
  dpos_dtheta_dict = {}

  # Compute pos first
  for bone in weights:
    name = bone["bone"]
    weight = bone["weight"]
    bone_data = skeleton_json[name]

    G_rest = np.array(bone_data["matrix_rest"])
    G_current = bone_data["matrix_global_current"]

    T = np.linalg.inv(G_rest) @ rest_homog
    pos += weight * (G_current @ T)[:3]

  # Now compute dpos/dtheta for each bone
  for bone in weights:
    name = bone["bone"]
    weight = bone["weight"]
    bone_data = skeleton_json[name]

    if bone_data["theta"] == 0:
      continue  # No derivative contribution if theta=0

    θ = bone_data["theta"]

    dR_dθ = np.array([
      [-np.sin(θ), -np.cos(θ), 0, 0],
      [ np.cos(θ), -np.sin(θ), 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ])

    parent = bone_data["parent"]
    P = np.eye(4) if not parent else skeleton_json[parent]["matrix_global_current"]
    M_local = np.array(bone_data["matrix_local"])

    G_rest = np.array(bone_data["matrix_rest"])
    T = np.linalg.inv(G_rest) @ rest_homog

    dG = P @ M_local @ dR_dθ
    dpos_dtheta = weight * (dG @ T)[:3]

    # Store each derivative separately
    dpos_dtheta_dict[name] = dpos_dtheta

  energy = 1.0 / np.dot(pos, pos)

  # Compute each derivative dE/dtheta_i
  dE_dtheta_dict = {}
  for bone_name, dpos_dtheta in dpos_dtheta_dict.items():
    dE_dtheta = -2.0 * np.dot(pos, dpos_dtheta) / (np.dot(pos, pos)**2)
    dE_dtheta_dict[bone_name] = dE_dtheta

  return energy, dE_dtheta_dict

# --- Example Usage ---
vertex_index = 1156  # replace with your vertex index
energy, dE_dtheta = compute_energy_and_derivative(vertex_index, skeleton_json, skin_json)
print(f"Vertex {vertex_index} Energy: {energy}")
print(f"Vertex {vertex_index} dE/dθ: {dE_dtheta}")

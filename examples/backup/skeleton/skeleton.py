import bpy
import json
import os
import sys
from mathutils import Vector
import numpy as np
# --- Get the FBX file path from command line ---
fbx_path = sys.argv[sys.argv.index("--") + 1]
bone_output_path = fbx_path + ".bones.json"
vertex_output_path = fbx_path + ".vertices.json"

# --- Clear current scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- Import the FBX ---
bpy.ops.import_scene.fbx(filepath=fbx_path)

# --- Find the armature (skeleton) ---
armature_obj = next((obj for obj in bpy.data.objects if obj.type == 'ARMATURE'), None)
if not armature_obj:
  raise Exception("No armature found in the scene.")
armature = armature_obj.data

# --- Extract bone hierarchy and local matrices ---
bone_data = {}

for bone in armature.bones:
  name = bone.name
  parent = bone.parent.name if bone.parent else None
  length = bone.length


  # Compute local matrix relative to parent
  if bone.parent:
    parent_matrix = bone.parent.matrix_local
    local_matrix = parent_matrix.inverted() @ bone.matrix_local
  else:
    local_matrix = bone.matrix_local.copy()

  inv_local = bone.matrix_local.inverted()
  head = inv_local @ Vector(bone.head_local)
  tail = inv_local @ Vector(bone.tail_local)

  matrix = [list(row) for row in local_matrix]

  theta = 0
  if name == "b_l_thumb3" or name == "b_l_index1" or name == "b_l_index2" or name == "b_l_index3" or name == "b_l_middle1" or name == "b_l_middle2" or name == "b_l_middle3":
    theta = np.pi / 3
  if name == "b_l_thumb2":
    theta = np.pi / 6

  bone_data[name] = {
    'name': name,
    'parent': parent,
    'length': length,
    'matrix_local': matrix,
    'matrix_rest': [list(row) for row in bone.matrix_local.copy()],
    'head': head[:],
    'tail': tail[:],
    'theta': theta
  }

# --- Save bones to JSON ---
with open(bone_output_path, 'w') as f:
  json.dump(bone_data, f, indent=2)

# --- Find mesh and extract vertex rest positions and weights ---
mesh_obj = next((obj for obj in bpy.data.objects if obj.type == 'MESH'), None)
if not mesh_obj:
  raise Exception("No mesh object found in the scene.")
mesh = mesh_obj.data

# Transform from mesh local to armature local space
to_armature_space = armature_obj.matrix_world.inverted() @ mesh_obj.matrix_world

vertex_data = []

for v in mesh.vertices:
  rest_pos = to_armature_space @ v.co
  influences = []

  for g in v.groups:
    group = mesh_obj.vertex_groups[g.group]
    influences.append({
        'bone': group.name,
        'weight': g.weight
    })

  vertex_data.append({
    'index': v.index,
    'rest_position': list(rest_pos[:]),
    'weights': influences
  })

# --- Save vertices to JSON ---
with open(vertex_output_path, 'w') as f:
  json.dump(vertex_data, f, indent=2)


print(f"✅ Bone data saved to: {bone_output_path}")
print(f"✅ Vertex data saved to: {vertex_output_path}")

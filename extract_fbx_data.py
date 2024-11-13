import bpy
import json
import math

# Function to convert radians to degrees
def radians_to_degrees(radians):
    return radians * (180.0 / math.pi)

# Function to recursively print bone hierarchy (optional)
def print_bone_hierarchy(bone, level=0):
    print("  " * level + f"Bone: {bone.name}")
    for child in bone.children:
        print_bone_hierarchy(child, level + 1)

# Clear existing scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Path to your FBX file
fbx_file = '/home/xuan/Downloads/Hand Assets/OculusHand_L.fbx'  # Replace with the actual path

# Import the FBX file
bpy.ops.import_scene.fbx(filepath=fbx_file)

# Find all armature objects (skeletons)
armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

# Ensure there's at least one armature
if not armatures:
    print("No armature found in the scene.")
    armature = None
else:
    armature = armatures[0]  # Assuming you have one armature
    bpy.context.view_layer.objects.active = armature  # Make it the active object

# Extract Skeleton Data
bones_data = []

if armature:
    # Switch to Pose Mode to access pose bones and constraints
    bpy.ops.object.mode_set(mode='POSE')

    for bone in armature.pose.bones:
        # Bone local axes
        bone_matrix = bone.bone.matrix_local
        x_axis = bone_matrix.to_3x3()[0][:]  # X-axis
        y_axis = bone_matrix.to_3x3()[1][:]  # Y-axis
        z_axis = bone_matrix.to_3x3()[2][:]  # Z-axis

        bone_info = {
            'name': bone.name,
            'parent': bone.parent.name if bone.parent else None,
            'head': bone.head[:],
            'tail': bone.tail[:],
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis,
            'rotation_mode': bone.rotation_mode,
            'constraints': []
        }

        # Access rotational constraints
        for constraint in bone.constraints:
            if constraint.type == 'LIMIT_ROTATION':
                limit_rot = {
                    'constraint_name': constraint.name,
                    'use_limit_x': constraint.use_limit_x,
                    'min_x': constraint.min_x,
                    'max_x': constraint.max_x,
                    'min_x_degrees': radians_to_degrees(constraint.min_x),
                    'max_x_degrees': radians_to_degrees(constraint.max_x),
                    'use_limit_y': constraint.use_limit_y,
                    'min_y': constraint.min_y,
                    'max_y': constraint.max_y,
                    'min_y_degrees': radians_to_degrees(constraint.min_y),
                    'max_y_degrees': radians_to_degrees(constraint.max_y),
                    'use_limit_z': constraint.use_limit_z,
                    'min_z': constraint.min_z,
                    'max_z': constraint.max_z,
                    'min_z_degrees': radians_to_degrees(constraint.min_z),
                    'max_z_degrees': radians_to_degrees(constraint.max_z),
                    'owner_space': constraint.owner_space
                }
                bone_info['constraints'].append(limit_rot)

        bones_data.append(bone_info)

    # Optionally, print the bone hierarchy
    # for bone in armature.data.bones:
    #     if bone.parent is None:
    #         print_bone_hierarchy(bone)

# Extract Surface Mesh Data
meshes_data = []

# Find all mesh objects (surfaces)
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

for mesh_obj in meshes:
    mesh = mesh_obj.data
    vertices = [v.co[:] for v in mesh.vertices]
    faces = [list(p.vertices) for p in mesh.polygons]

    # Extract mapping data (vertex groups and weights)
    mapping = []
    for vert in mesh.vertices:
        vert_groups = []
        for group in vert.groups:
            group_name = mesh_obj.vertex_groups[group.group].name
            weight = group.weight
            vert_groups.append({'group': group_name, 'weight': weight})
        mapping.append({'vertex_index': vert.index, 'groups': vert_groups})

    mesh_info = {
        'mesh_name': mesh_obj.name,
        'vertices': vertices,
        'faces': faces,
        'mapping': mapping
    }
    meshes_data.append(mesh_info)

# # Save the extracted data to JSON files

# # 1. Save bones data
# with open('bones_data.json', 'w') as f:
#     json.dump(bones_data, f, indent=4)

# # 2. Save meshes data
# with open('meshes_data.json', 'w') as f:
#     json.dump(meshes_data, f, indent=4)

# Optionally, you can combine all data into a single file
export_data = {
    'armature_name': armature.name if armature else None,
    'bones': bones_data,
    'meshes': meshes_data
}

with open('extracted_fbx_data.json', 'w') as f:
    json.dump(export_data, f, indent=4)

# print("Data extraction complete. Files saved:")
# print(" - bones_data.json")
# print(" - meshes_data.json")
# print(" - extracted_fbx_data.json")

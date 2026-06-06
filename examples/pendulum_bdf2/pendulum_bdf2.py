from yasps import scene
from yasps import attribute

import os

import numpy as np
import pycuda.gpuarray as gpuarray
import pyvista as pv
from helpers import extract_surface_triangles, inertia, extract_edges_from_triangles, abs_max_reduce, stable_neo_hookean, string_elasticity_energy, constrained_energy, affine_energy, rotation_energy, generate_edge_to_vertices_list, bending, baraff_witkin, inertia_bdf2
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DT_VALUE = 1.0 / 1000.0 # for time step
DHAT_VALUE = 1e-6 # for collision detection
KAPPA_VALUE = 1.0 # for collision


POISSON_VALUE = 0.365697005781997
YOUNG_VALUE = 19000259.25455816859
MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))

POISSON_VALUE_SPHERE = 0.29
YOUNG_VALUE_SPHERE = 20000000
MU_LAME_VALUE_SPHERE = YOUNG_VALUE_SPHERE / (2 * (1 + POISSON_VALUE_SPHERE))
LAMBDA_LAME_VALUE_SPHERE = YOUNG_VALUE_SPHERE * POISSON_VALUE_SPHERE / ((1 + POISSON_VALUE_SPHERE) * (1 - 2 * POISSON_VALUE_SPHERE))


STRING_VERTEX_COUNT = 500
VERTICAL_STRING_VERTEX_COUNT = 100
HANGING_STRING_LENGTH = 1.0
STRING_ELASTICITY = 1000000000
SHRUNK_VALUE = 1.000


##################################################################
## First we create the strings
##################################################################
string_vertices = []
string_edges = []
vertical_string_end_index = []

# first we do the horizontal hanging strings
for i in range(STRING_VERTEX_COUNT):
  string_vertices.append([-HANGING_STRING_LENGTH / 2 + i * HANGING_STRING_LENGTH / (STRING_VERTEX_COUNT - 1), 1.0, -0.2])
  if i < STRING_VERTEX_COUNT - 1:
    string_edges.append([i, i + 1])

for i in range(STRING_VERTEX_COUNT):
  string_vertices.append([-HANGING_STRING_LENGTH / 2 + i * HANGING_STRING_LENGTH / (STRING_VERTEX_COUNT - 1), 1.0, 0.2])
  if i < STRING_VERTEX_COUNT - 1:
    string_edges.append([i + STRING_VERTEX_COUNT, i + 1 + STRING_VERTEX_COUNT])

# determine the indix of the rods
rod_starting_index = [180, 250, 320]

##################################################################
## Now we create the spheres
##################################################################
sphere_vertices = []
sphere_faces = []
f = open(os.path.join(SCRIPT_DIR, "../data/sphere_small.obj"), 'r')
for line in f:
  if line.startswith('v '):
    sphere_vertices.append([float(x) for x in line.split()[1:]])
  if line.startswith('f '):
    face = []
    for x in line.split()[1:]:
      v_idx = int(x.split('/')[0]) - 1
      face.append(v_idx)
    sphere_faces.append(face)
sphere_vertices = np.array(sphere_vertices, dtype=np.float64)
sphere_faces = np.array(sphere_faces, dtype=np.uint32)

# find the bounding box
min_x, max_x = np.min(sphere_vertices[:, 0]), np.max(sphere_vertices[:, 0])
min_y, max_y = np.min(sphere_vertices[:, 1]), np.max(sphere_vertices[:, 1])
min_z, max_z = np.min(sphere_vertices[:, 2]), np.max(sphere_vertices[:, 2])
print("Sphere bounding box:")
print("X: ", min_x, max_x)
print("Y: ", min_y, max_y)
print("Z: ", min_z, max_z)
# center the sphere
center_x = (min_x + max_x) / 2.0
center_y = (min_y + max_y) / 2.0
center_z = (min_z + max_z) / 2.0
sphere_vertices[:, 0] -= center_x
sphere_vertices[:, 1] -= center_y
sphere_vertices[:, 2] -= center_z
# we add a center vertex
sphere_vertices = np.vstack((sphere_vertices, np.array([[0.0, 0.0, 0.0]])))


candidate_indices = np.where(
  (sphere_vertices[:, 1] > 0) &
  (sphere_vertices[:, 0] >= -0.1) &
  (sphere_vertices[:, 0] <= 0.1)
)[0]

sorted_by_z = candidate_indices[np.argsort(sphere_vertices[candidate_indices, 2])]

if len(sorted_by_z) < 3:
  raise ValueError("Not enough vertices satisfying y > 0 and -0.05 <= x <= 0.05")

third_min_z_index = sorted_by_z[1]
third_max_z_index = sorted_by_z[-2]

min_z_index = third_min_z_index
max_z_index = third_max_z_index

print("Min z index: ", min_z_index)
print("Max z index: ", max_z_index)

# scale it to size 1
scale = 1.0 / np.linalg.norm(max_x - min_x)
sphere_vertices *= scale

# now we need to determine the space or the size needed for the sphere
# we do this by checking the index on the rod
rod_starting_pos1 = [string_vertices[rod_starting_index[i]] for i in range(len(rod_starting_index))]
rod_starting_pos2 = [string_vertices[rod_starting_index[i] + STRING_VERTEX_COUNT] for i in range(len(rod_starting_index))]
ending_pos = [np.array([rod_starting_pos1[i][0], rod_starting_pos1[i][1] - 0.7, 0.0]) for i in range(len(rod_starting_index))]

space_between = np.linalg.norm(np.array(rod_starting_pos1[0]) - np.array(rod_starting_pos1[1]))
print("Space between rods: ", space_between)
sphere_vertices *= space_between * 0.99


# move the indix of the rods towards the center a bit
rod_starting_index = [200, 250, 300]
rod_starting_pos1 = [string_vertices[rod_starting_index[i]] for i in range(len(rod_starting_index))]
rod_starting_pos2 = [string_vertices[rod_starting_index[i] + STRING_VERTEX_COUNT] for i in range(len(rod_starting_index))]
# now we create the lines connecting the rods to the sphere
for i in range(len(rod_starting_index)):
  vertical_string_end_index.append(rod_starting_index[i])
  vertical_string_end_index.append(rod_starting_index[i] + STRING_VERTEX_COUNT)
  # sphere_moved = sphere_vertices.copy()
  # # move the sphere to the center
  # sphere_moved += ending_pos[i]
  # # determine the tip vertices
  # tip_vertex1 = sphere_moved[min_z_index]
  # tip_vertex2 = sphere_moved[max_z_index]
  # print("Tip vertex 1: ", tip_vertex1)
  # print("Tip vertex 2: ", tip_vertex2)
  # # ok we construct the first string, which connects to the bunny's one of the tip
  # starting_pos = rod_starting_pos1[i]
  # end_pos1 = tip_vertex1
  # for j in range(VERTICAL_STRING_VERTEX_COUNT):
  #   string_vertices.append([
  #     starting_pos[0] + (end_pos1[0] - starting_pos[0]) * (j + 1) / (VERTICAL_STRING_VERTEX_COUNT + 1),
  #     starting_pos[1] + (end_pos1[1] - starting_pos[1]) * (j + 1) / (VERTICAL_STRING_VERTEX_COUNT + 1),
  #     starting_pos[2] + (end_pos1[2] - starting_pos[2]) * (j + 1) / (VERTICAL_STRING_VERTEX_COUNT + 1),
  #   ])
  #   if j == 0:
  #     string_edges.append([rod_starting_index[i], len(string_vertices) - 1])
  #   else:
  #     string_edges.append([len(string_vertices) - 2, len(string_vertices) - 1])
  # vertical_string_end_index.append(len(string_vertices) - 1)

  # starting_pos = rod_starting_pos2[i]
  # end_pos2 = tip_vertex2
  # for j in range(VERTICAL_STRING_VERTEX_COUNT):
  #   string_vertices.append([
  #     starting_pos[0] + (end_pos2[0] - starting_pos[0]) * (j + 1) / (VERTICAL_STRING_VERTEX_COUNT + 1),
  #     starting_pos[1] + (end_pos2[1] - starting_pos[1]) * (j + 1) / (VERTICAL_STRING_VERTEX_COUNT + 1),
  #     starting_pos[2] + (end_pos2[2] - starting_pos[2]) * (j + 1) / (VERTICAL_STRING_VERTEX_COUNT + 1),
  #   ])
  #   if j == 0:
  #     string_edges.append([rod_starting_index[i] + STRING_VERTEX_COUNT, len(string_vertices) - 1])
  #   else:
  #     string_edges.append([len(string_vertices) - 2, len(string_vertices) - 1])
  # vertical_string_end_index.append(len(string_vertices) - 1)

# now we put the sphere there
string_vertices = np.array(string_vertices, dtype=np.float64)
string_edges = np.array(string_edges, dtype=np.uint32)
# now for each vertical string, we will have a hanging bunny
all_vertices = string_vertices
all_edges = string_edges
all_tet_indices = []
all_surface_triangle_indices = []
all_surface_indices = list(range(len(string_vertices)))
surface_indices_sphere = np.array(list(range(sphere_vertices.shape[0] - 1)), dtype=np.uint32)
edge_indices_sphere = extract_edges_from_triangles(sphere_faces).astype(np.uint32)
min_z_orig = 0.0
max_z_orig = 0.0

for i in range(len(rod_starting_index)):
  moved_sphere_vertices = sphere_vertices.copy()
  # move the bunny to the center
  moved_sphere_vertices += ending_pos[i]
  min_z_orig = moved_sphere_vertices[min_z_index, 2]
  max_z_orig = moved_sphere_vertices[max_z_index, 2]

  sphere_vertex_offset = all_vertices.shape[0]
  all_surface_triangle_indices.append(sphere_faces + sphere_vertex_offset) # all triangles
  all_surface_indices += (surface_indices_sphere + sphere_vertex_offset).tolist() # surface vertices
  all_edges = np.concatenate((all_edges, edge_indices_sphere + sphere_vertex_offset), axis = 0) # add edges

  all_edges = np.concatenate((all_edges, np.array([[vertical_string_end_index[i * 2], min_z_index + sphere_vertex_offset]], dtype=np.uint32)), axis = 0)
  all_edges = np.concatenate((all_edges, np.array([[vertical_string_end_index[i * 2 + 1], max_z_index + sphere_vertex_offset]], dtype=np.uint32)), axis = 0)
  string_edges = np.concatenate((
    string_edges,
    np.array([[vertical_string_end_index[i * 2], min_z_index + sphere_vertex_offset]], dtype=np.uint32)
  ), axis = 0)
  string_edges = np.concatenate((
    string_edges,
    np.array([[vertical_string_end_index[i * 2 + 1], max_z_index + sphere_vertex_offset]], dtype=np.uint32)
  ), axis = 0)

  all_vertices = np.concatenate((all_vertices, moved_sphere_vertices), axis = 0)

  # flip the face then create the tetrahedrons
  flipped_faces = sphere_faces[:, [0, 2, 1]]

  sphere_tets = np.hstack([
    flipped_faces + sphere_vertex_offset,
    np.full((sphere_faces.shape[0], 1), all_vertices.shape[0] - 1, dtype=np.uint32)
  ]).astype(np.uint32)
  all_tet_indices.append(sphere_tets)


##################################################################
## Now we create the bunnies and put them inside the sphere
##################################################################
f = open(os.path.join(SCRIPT_DIR, "../data/bunny.ele"), 'r')
f.readline()
tet_indices = []
for line in f:
  tet_indices.append([int(x) - 1 for x in line.split()[3:]])
f.close()
tet_indices = np.array(tet_indices)

f = open(os.path.join(SCRIPT_DIR, "../data/bunny.node"), 'r')
f.readline()
position = []
for line in f:
  position.append([float(x) for x in line.split()[1:]])
f.close()
position = np.array(position, dtype = np.float64)
# get the bounding box of the bunny
min_x, max_x = np.min(position[:, 0]), np.max(position[:, 0])
min_y, max_y = np.min(position[:, 1]), np.max(position[:, 1])
min_z, max_z = np.min(position[:, 2]), np.max(position[:, 2])
# now center the bunny
center_x = (min_x + max_x) / 2.0
center_y = (min_y + max_y) / 2.0
center_z = (min_z + max_z) / 2.0
position[:, 0] -= center_x
position[:, 1] -= center_y
position[:, 2] -= center_z
min_x, max_x = np.min(position[:, 0]), np.max(position[:, 0])
min_y, max_y = np.min(position[:, 1]), np.max(position[:, 1])
min_z, max_z = np.min(position[:, 2]), np.max(position[:, 2])
print("Bunny bounding box:")
print("X: ", min_x, max_x)
print("Y: ", min_y, max_y)
print("Z: ", min_z, max_z)

# normalize the bunny to scale 1
scale = 1.0 / max(np.linalg.norm(max_x - min_x), np.linalg.norm(max_y - min_y), np.linalg.norm(max_z - min_z))
position *= scale

# now fit inside the sphere
position *= space_between * 0.7


surface_triangle_indices_bunny = extract_surface_triangles(tet_indices).astype(np.uint32)
edge_indices_bunny = extract_edges_from_triangles(surface_triangle_indices_bunny).astype(np.uint32)
surface_indices_bunny = np.array(list(set(surface_triangle_indices_bunny.flatten().tolist()))).astype(np.uint32)

NUM_BUNNY_VERTICES = position.shape[0]
NUM_BUNNY_TETS = tet_indices.shape[0]
NUM_BUNNY_SURFACE_TRIANGLES = surface_triangle_indices_bunny.shape[0]
NUM_BUNNY_EDGES = edge_indices_bunny.shape[0]
NUM_BUNNY_SURFACE_INDICES = surface_indices_bunny.shape[0]
print("Number of bunny vertices: ", NUM_BUNNY_VERTICES)
print("Number of bunny tets: ", NUM_BUNNY_TETS)
print("Number of bunny surface triangles: ", NUM_BUNNY_SURFACE_TRIANGLES)
print("Number of bunny edges: ", NUM_BUNNY_EDGES)
print("Number of bunny surface indices: ", NUM_BUNNY_SURFACE_INDICES)


for i in range(len(rod_starting_index)):
  moved_bunny_vertices = position.copy()
  # move the bunny to the center
  moved_bunny_vertices += ending_pos[i]

  bunny_vertex_offset = all_vertices.shape[0]
  all_tet_indices.append(tet_indices + bunny_vertex_offset)
  all_surface_triangle_indices.append(surface_triangle_indices_bunny + bunny_vertex_offset)
  all_surface_indices += (surface_indices_bunny + bunny_vertex_offset).tolist()
  all_edges = np.concatenate((all_edges, edge_indices_bunny + bunny_vertex_offset), axis = 0)

  all_vertices = np.concatenate((all_vertices, moved_bunny_vertices), axis = 0)


all_tet_indices = np.vstack(all_tet_indices).astype(np.uint32)
all_surface_triangle_indices = np.vstack(all_surface_triangle_indices).astype(np.uint32)
all_surface_indices = np.array(all_surface_indices, dtype=np.uint32)

##################################################################
## Now we create the scene
##################################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])



kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue(KAPPA_VALUE)

dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])

string_mesh = s0.addMesh("string_mesh")
sphere_mesh = s0.addMesh("sphere_mesh")
bunny_mesh = s0.addMesh("bunny_mesh")


mu = bunny_mesh.addConstant("mu", rows = 1, cols = 1)
mu.updateValue([MU_LAME_VALUE])
lam = bunny_mesh.addConstant("lambda", rows = 1, cols = 1)
lam.updateValue([LAMBDA_LAME_VALUE])


mu_sphere = sphere_mesh.addConstant("mu_sphere", rows = 1, cols = 1)
mu_sphere.updateValue([MU_LAME_VALUE_SPHERE])
lam_sphere = sphere_mesh.addConstant("lambda_sphere", rows = 1, cols = 1)
lam_sphere.updateValue([LAMBDA_LAME_VALUE_SPHERE])

string_vertices_count = len(string_vertices)
sphere_vertices_count = sphere_vertices.shape[0] * len(rod_starting_index)
bunny_vertices_count = NUM_BUNNY_VERTICES * len(rod_starting_index)

##################################################################
## Create vertices for the strings
##################################################################
string_mesh_vertices = string_mesh.addPrimitive("vertices", numInstances = string_vertices.shape[0])
svp = string_mesh_vertices.addAttribute("position", rows = 3, cols = 1)
svp.updateValue(string_vertices.flatten())
svrp = string_mesh_vertices.addConstant("rest_position", rows = 3, cols = 1)
svrp.updateValue(string_vertices.flatten())
svlp = string_mesh_vertices.addConstant("last_position", rows = 3, cols = 1)
svlp.updateValue(string_vertices.flatten())
svllp = string_mesh_vertices.addConstant("last_last_position", rows = 3, cols = 1)
svllp.updateValue(string_vertices.flatten())
svm = string_mesh_vertices.addConstant("mass", rows = 1, cols = 1)
mass = [0.001] * string_vertices.shape[0]
mass = np.array(mass, dtype=np.float64)
svm.updateValue(mass)
svv = string_mesh_vertices.addConstant("velocity", rows = 3, cols = 1)
svv.updateValue(np.zeros_like(string_vertices).flatten())
svlv = string_mesh_vertices.addConstant("last_velocity", rows = 3, cols = 1)
svlv.updateValue(np.zeros_like(string_vertices).flatten())



##################################################################
## Create vertices for the spheres
##################################################################
sphere_mesh_vertices = sphere_mesh.addPrimitive("vertices", numInstances = sphere_vertices.shape[0] * len(rod_starting_index))
spvrp = sphere_mesh_vertices.addConstant("rest_position", rows = 3, cols = 1)
spvrp.updateValue(all_vertices[string_vertices_count: string_vertices_count + sphere_vertices_count].flatten())
spvlp = sphere_mesh_vertices.addConstant("last_position", rows = 3, cols = 1)
spvlp.updateValue(all_vertices[string_vertices_count: string_vertices_count + sphere_vertices_count].flatten())
spvm = sphere_mesh_vertices.addConstant("mass", rows = 1, cols = 1)
spvllp = sphere_mesh_vertices.addConstant("last_last_position", rows = 3, cols = 1)
spvllp.updateValue(all_vertices[string_vertices_count: string_vertices_count + sphere_vertices_count].flatten())
mass = ([0.01 / sphere_vertices.shape[0]] * (sphere_vertices.shape[0] - 1) + [100.0]) * len(rod_starting_index) # create the mass
mass = np.array(mass, dtype=np.float64)
spvm.updateValue(mass)
spvv = sphere_mesh_vertices.addConstant("velocity", rows = 3, cols = 1)
spvv.updateValue(np.zeros_like(all_vertices[string_vertices_count: string_vertices_count + sphere_vertices_count]).flatten())
spvlv = sphere_mesh_vertices.addConstant("last_velocity", rows = 3, cols = 1)
spvlv.updateValue(np.zeros_like(all_vertices[string_vertices_count: string_vertices_count + sphere_vertices_count]).flatten())
spvp = sphere_mesh_vertices.addAttribute("position", rows = 3, cols = 1) # current position
spvp.updateValue(all_vertices[string_vertices_count: string_vertices_count + sphere_vertices_count].flatten())

##################################################################
## Create tetrahedrons
##################################################################
sphere_tets = sphere_mesh.addPrimitive("tets", numInstances = sphere_faces.shape[0] * len(rod_starting_index))
sphere_tets2v = sphere_tets.addConnectivity("t2v", sphere_mesh_vertices, all_tet_indices[: sphere_faces.shape[0] * len(rod_starting_index)].flatten() - string_vertices_count, 4)
sptp = sphere_tets.addAttribute("positions", through = sphere_tets2v, source = spvp)
sptp = sptp.resize(4, 3)
sptrp = sphere_tets.addAttribute("rest_positions", through = sphere_tets2v, source = spvrp)
sptrp = sptrp.resize(4, 3)
sptlp = sphere_tets.addAttribute("last_positions", through = sphere_tets2v, source = spvlp)
sptlp = sptlp.resize(4, 3)



##################################################################
## Create vertices for the bunnies
##################################################################
bunny_mesh_vertices = bunny_mesh.addPrimitive("vertices", numInstances = NUM_BUNNY_VERTICES * len(rod_starting_index))
bvp = bunny_mesh_vertices.addAttribute("position", rows = 3, cols = 1)
bvp.updateValue(all_vertices[string_vertices_count + sphere_vertices_count:].flatten())
bvrp = bunny_mesh_vertices.addConstant("rest_position", rows = 3, cols = 1)
bvrp.updateValue(all_vertices[string_vertices_count + sphere_vertices_count:].flatten())
bvlp = bunny_mesh_vertices.addConstant("last_position", rows = 3, cols = 1)
bvlp.updateValue(all_vertices[string_vertices_count + sphere_vertices_count:].flatten())
bvllp = bunny_mesh_vertices.addConstant("last_last_position", rows = 3, cols = 1)
bvllp.updateValue(all_vertices[string_vertices_count + sphere_vertices_count:].flatten())
bvm = bunny_mesh_vertices.addConstant("mass", rows = 1, cols = 1)
mass = [5.0 / NUM_BUNNY_VERTICES] * NUM_BUNNY_VERTICES * len(rod_starting_index) # create the mass
mass = np.array(mass, dtype=np.float64)
bvm.updateValue(mass)
bvv = bunny_mesh_vertices.addConstant("velocity", rows = 3, cols = 1)
bvv.updateValue(np.zeros_like(all_vertices[string_vertices_count + sphere_vertices_count:]).flatten())
bvlv = bunny_mesh_vertices.addConstant("last_velocity", rows = 3, cols = 1)
bvlv.updateValue(np.zeros_like(all_vertices[string_vertices_count + sphere_vertices_count:]).flatten())

##################################################################
## Create tetrahedrons
##################################################################
# tetrahedrons are only on the bunnies
# so we can directly create the primitive on the bunny
tets = bunny_mesh.addPrimitive("tets", numInstances = all_tet_indices.shape[0] - sphere_faces.shape[0] * len(rod_starting_index))
t2v = tets.addConnectivity("t2v", bunny_mesh_vertices, all_tet_indices[sphere_faces.shape[0] * len(rod_starting_index): ].flatten() - sphere_vertices_count - string_vertices_count, 4)
tp = tets.addAttribute("positions", through = t2v, source = bvp)
tp = tp.resize(4, 3)
trp = tets.addAttribute("rest_positions", through = t2v, source = bvrp)
trp = trp.resize(4, 3)
tlp = tets.addAttribute("last_positions", through = t2v, source = bvlp)
tlp = tlp.resize(4, 3)

##################################################################
## Create the constraint vertices
##################################################################
# constraint vertices are only created for the strings
cv = string_mesh.addPrimitive("constraint_vertices", numInstances = 2 * STRING_VERTEX_COUNT)
cv2p_array = np.array(list(range(STRING_VERTEX_COUNT * 2)))
cv2p = cv.addConnectivity("cv2p", string_mesh_vertices, np.array(cv2p_array), 1)
cvp = cv.addAttribute("positions", through = cv2p, source = svp)
cvtp = cv.addConstant("target_positions", rows = 1, cols = 3)
cvtp_array = string_vertices[: 2 * STRING_VERTEX_COUNT].flatten()
cvtp.updateValue(np.array(cvtp_array))
cvw = cv.addConstant("weights", rows = 1, cols = 1)
cvw.updateValue([10000000000000] * 2 * STRING_VERTEX_COUNT)

cv2 = sphere_mesh.addPrimitive("constraint_vertices", numInstances = 2 * len(rod_starting_index))
cv2_2_spv_array = [min_z_index, max_z_index, min_z_index + sphere_vertices.shape[0], max_z_index + sphere_vertices.shape[0], min_z_index + 2 * sphere_vertices.shape[0], max_z_index + 2 * sphere_vertices.shape[0]]
cv2_2_spv = cv2.addConnectivity("cv2spv", sphere_mesh_vertices, np.array(cv2_2_spv_array), 1)
cv2tp = cv2.addConstant("target_positions", rows = 1, cols = 1)
cv2tp_array = [min_z_orig, max_z_orig, min_z_orig, max_z_orig, min_z_orig, max_z_orig]
cv2tp.updateValue(np.array(cv2tp_array))
cv2pos = cv2.addAttribute("positions", through = cv2_2_spv, source = spvp)
cv2w = cv2.addConstant("weights", rows = 1, cols = 1)
cv2w.updateValue([10000000000000] * 2 * len(rod_starting_index))

##################################################################
## Create the collision mesh, and the primitive union
##################################################################
collision_mesh = s0.addMesh("collision_mesh")
string_elasticity = collision_mesh.addConstant("string_elasticity", rows = 1, cols = 1)
string_elasticity.updateValue([STRING_ELASTICITY])

shrunk_value = collision_mesh.addConstant("shrunk_value", rows = 1, cols = 1)
shrunk_value.updateValue([SHRUNK_VALUE])

cmv = collision_mesh.addPrimitiveUnion("vertices", [string_mesh_vertices, sphere_mesh_vertices, bunny_mesh_vertices])
cmvp = cmv.addAttribute("position")
cmvrp = cmv.addAttribute("rest_position")
cmvlp = cmv.addAttribute("last_position")



##################################################################
## Create edges
##################################################################
# vertices on edges can either be free vertices, or the vertices on the affine bodies
# so it must be created on the primitive union
edges = collision_mesh.addPrimitive("edges", numInstances = string_edges.shape[0])
e2cv = edges.addConnectivity("e2v", cmv, string_edges.flatten(), 2)
ep = edges.addAttribute("positions", through = e2cv, source = cmvp)
ep = ep.resize(2, 3)
erp = edges.addAttribute("rest_positions", through = e2cv, source = cmvrp)
erp = erp.resize(2, 3)


##################################################################
## Create the collision primitives
##################################################################
collision_mesh.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
collision_mesh.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
collision_mesh.addPrimitive("pt", numInstances = 0, isDynamic = True) # # for point triangle collision
collision_mesh.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision
pp2v = collision_mesh.pp.addConnectivity("pp2v", collision_mesh.vertices, [], 2)
pe2v = collision_mesh.pe.addConnectivity("pe2v", collision_mesh.vertices, [], 3)
pt2v = collision_mesh.pt.addConnectivity("pt2v", collision_mesh.vertices, [], 4)
ee2v = collision_mesh.ee.addConnectivity("ee2v", collision_mesh.vertices, [], 4)
pp_positions = collision_mesh.pp.addAttribute("positions", through = pp2v, source = collision_mesh.vertices["position"])
pe_positions = collision_mesh.pe.addAttribute("positions", through = pe2v, source = collision_mesh.vertices["position"])
pt_positions = collision_mesh.pt.addAttribute("positions", through = pt2v, source = collision_mesh.vertices["position"])
ee_positions = collision_mesh.ee.addAttribute("positions", through = ee2v, source = collision_mesh.vertices["position"])
pp_last_positions = collision_mesh.pp.addAttribute("last_positions", through = pp2v, source = collision_mesh.vertices["last_position"])
pe_last_positions = collision_mesh.pe.addAttribute("last_positions", through = pe2v, source = collision_mesh.vertices["last_position"])
pt_last_positions = collision_mesh.pt.addAttribute("last_positions", through = pt2v, source = collision_mesh.vertices["last_position"])
ee_last_positions = collision_mesh.ee.addAttribute("last_positions", through = ee2v, source = collision_mesh.vertices["last_position"])

##################################################################
## Create energies
##################################################################
# stable neo hookean
snh = stable_neo_hookean(trp, tp, mu, lam, dt)
snh = tets.addAttribute("stable_neo_hookean", computed_attribute = snh)

# stable neo hookean for the sphere
snh_sphere = stable_neo_hookean(sptrp, sptp, mu_sphere, lam_sphere, dt)
snh_sphere = sphere_tets.addAttribute("stable_neo_hookean", computed_attribute = snh_sphere)

# inertia will be created for 3 primitives
# yes we can create it for the collision mesh
# but i just want to do it in 3 ways since there's clear separation
inertia_string = inertia_bdf2(svlp, svllp, svv, svlv, dt, svp, svm)
string_mesh_vertices.addAttribute("inertia_bdf2", computed_attribute = inertia_string)

inertia_sphere = inertia_bdf2(spvlp, spvllp, spvv, spvlv, dt, spvp, spvm)
sphere_mesh_vertices.addAttribute("inertia_bdf2", computed_attribute = inertia_sphere)

inertia_bunny = inertia_bdf2(bvlp, bvllp, bvv, bvlv, dt, bvp, bvm)
bunny_mesh_vertices.addAttribute("inertia_bdfe", computed_attribute = inertia_bunny)

# edge elasticity
see = string_elasticity_energy(ep, erp, dt, string_elasticity, shrunk_value)
edges.addAttribute("string_elasticity_energy", computed_attribute = see)

# constraint energy
cve = constrained_energy(cvp, cvtp, dt, cvw)
cv.addAttribute("constrained_energy", computed_attribute = cve)

from helpers import constrained_z_energy
cve2 = constrained_z_energy(cv2pos, cv2tp, dt, cv2w)
cv2.addAttribute("constrained_z_energy", computed_attribute = cve2)

from helpers import point_point, point_edge, point_triangle, edge_edge, affine_energy
pp = point_point(pp_positions, dhat, kappa)
collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

s0.addEnergy(snh, projection_method = 2)
s0.addEnergy(snh_sphere, projection_method = 2)
s0.addEnergy(inertia_string, projection_method = -1)
s0.addEnergy(inertia_sphere, projection_method = -1)
s0.addEnergy(inertia_bunny, projection_method = -1)
s0.addEnergy(see, projection_method = 2)
s0.addEnergy(cve, projection_method = -1)
s0.addEnergy(cve2, projection_method = -1)


s0.addEnergy(pp, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addMinimizeTarget([svp, spvp, bvp])

##################################################################
## add ccd
##################################################################
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
ccd = CCD(all_surface_indices.shape[0], # the number of surface points
  all_vertices.shape[0], # the number of total points
  mesh_indices = [2] * string_vertices_count + [3] * sphere_vertices.shape[0] + [4] * sphere_vertices.shape[0] + [5] * sphere_vertices.shape[0] + [0] * bunny_vertices_count,
  max_ccd_pairs = 200000000,
  max_cd_pairs = 10000000,
)



triangle_indices_all = all_surface_triangle_indices
surface_indices_all = all_surface_indices
edge_indices_all = all_edges

surface_indices_gpu = gpuarray.to_gpu(surface_indices_all.flatten())
edge_indices_gpu = gpuarray.to_gpu(edge_indices_all.flatten())
triangle_indices_gpu = gpuarray.to_gpu(triangle_indices_all.flatten())
position_gpu = gpuarray.to_gpu(cmvp.compute().value.get().reshape(-1, 3)) # basically copy it out

ccd.init_faces(position_gpu, triangle_indices_gpu, surface_indices_gpu, triangle_indices_all.shape[0])
ccd.init_edges(position_gpu, position_gpu, edge_indices_gpu, edge_indices_all.shape[0])


##################################################################
## Draw the scene with pyvista
##################################################################
plotter = pv.Plotter(window_size=[1920, 1080])
string_cells = np.hstack([np.full((string_edges.shape[0], 1), 2, dtype=np.uint32), string_edges])
string_poly = pv.PolyData(cmvp.compute().value.get().reshape(-1, 3), lines=string_cells)
plotter.add_mesh(string_poly, color="black", line_width=4)

sphere_triangle_cells = np.hstack([np.full((sphere_faces.shape[0] * len(ending_pos), 1), 3, dtype=np.uint32), all_surface_triangle_indices[:sphere_faces.shape[0] * len(ending_pos)] - string_vertices_count])
sphere_poly = pv.PolyData(spvp.compute().value.get().reshape(-1, 3), sphere_triangle_cells)
plotter.add_mesh(sphere_poly, color="lightgray", opacity=0.5)


bunny_triangle_cells = np.hstack([np.full((all_surface_triangle_indices.shape[0] - sphere_faces.shape[0] * len(ending_pos), 1), 3, dtype=np.uint32), all_surface_triangle_indices[sphere_faces.shape[0] * len(ending_pos): ] - string_vertices_count - sphere_vertices_count])
bunny_poly = pv.PolyData(bvp.compute().value.get().reshape(-1, 3), bunny_triangle_cells)
plotter.add_mesh(bunny_poly, color="green")

plotter.camera_position = [
  (0.0, HANGING_STRING_LENGTH / 2, HANGING_STRING_LENGTH * 1.5),
  (0.0, HANGING_STRING_LENGTH / 2, 0.0),
  (0.0, 1.0, 0.0),
]
# plotter.show()
# exit()
plotter.show(interactive_update=True, auto_close=False)



import time
position_copy = collision_mesh.vertices["position"].compute().value.copy()
string_position_copy = string_mesh.vertices["position"].value.copy()
sphere_position_copy = sphere_mesh.vertices["position"].value.copy()
bunny_position_copy = bunny_mesh.vertices["position"].value.copy()

start = time.time()
for i in range(30 * 1000):
  start_data_transfer = time.time()
  svllp.updateValue(svlp.value, deepCopy = True)
  spvllp.updateValue(spvlp.value, deepCopy = True)
  bvllp.updateValue(bvlp.value, deepCopy = True)
  svlp.updateValue(svp.value, deepCopy = True)
  spvlp.updateValue(spvp.value, deepCopy = True)
  bvlp.updateValue(bvp.value, deepCopy = True)
  svlv.updateValue(svv.value, deepCopy = True)
  spvlv.updateValue(spvv.value, deepCopy = True)
  bvlv.updateValue(bvv.value, deepCopy = True)

  if (i == 10):
    vv_value = spvv.value.get().reshape((-1, 3))
    vv_value[:sphere_vertices.shape[0], 0] -= 1.5
    spvv.updateValue(vv_value.flatten(), deepCopy=True)

  end_data_transfer = time.time()
  print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
  inner_iteration = 0
  while True:
    print("==================================================================")
    print(f"Iteration {i}, inner iteration {inner_iteration}")
    start_solver = time.time()
    result = s0.minimizeEnergy(tolerance = 1e-4)
    end_solver = time.time()
    print(f"Time taken for solver: {end_solver - start_solver} seconds")
    print("==================================================================")

    start_compute = time.time()
    energies_before = s0.computeTotalEnergy()
    end_compute = time.time()
    print(f"Time taken for computation: {end_compute - start_compute} seconds")
    # we perform CCD here
    # first we get the rotation and translation
    start_data_transfer = time.time()
    d_string_pos = result[0]
    d_sphere_pos = result[1]
    d_bunny_pos = result[2]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(cmvp.compute().value.copy())
    string_position_copy.set(svp.value.copy())
    sphere_position_copy.set(spvp.compute().value.copy())
    bunny_position_copy.set(bvp.value.copy())

    # now update the value
    svp.updateValue(svp.value - d_string_pos, deepCopy = True)
    spvp.updateValue(spvp.compute().value - d_sphere_pos, deepCopy = True)
    bvp.updateValue(bvp.value - d_bunny_pos, deepCopy = True)

    # compute the new positions for the entire scene
    new_positions = cmvp.compute().value
    # now we compute the new direction, remember it's the negative we need to put in
    end_data_transfer = time.time()
    print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
    start_direction_compute = time.time()
    direction_copy = position_copy - new_positions
    end_direction_compute = time.time()
    print(f"Time taken for direction compute: {end_direction_compute - start_direction_compute} seconds")
    start_max_movement = time.time()
    max_movement = gpuarray.max(abs(direction_copy)).get() / DT_VALUE
    end_max_movement = time.time()
    print(f"Time taken for max movement: {end_max_movement - start_max_movement} seconds")
    # check for the largest step size we can take
    start_ccd = time.time()
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 0.95)
    end_ccd = time.time()
    print(f"Time taken for CCD: {end_ccd - start_ccd} seconds")
    start_largest_step = time.time()
    largest_step = ccd.compute_largest_step_size(0.9, position_copy, direction_copy)
    end_largest_step = time.time()
    print("largest step we can take is", largest_step)
    print(f"Time taken for largest step: {end_largest_step - start_largest_step} seconds")
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    while substep <= 8:
      start_data_transfer = time.time()
      svp.updateValue(string_position_copy - d_string_pos * step_taken, deepCopy = True)
      spvp.updateValue(sphere_position_copy - d_sphere_pos * step_taken, deepCopy = True)
      bvp.updateValue(bunny_position_copy - d_bunny_pos * step_taken, deepCopy = True)

      end_data_transfer = time.time()
      print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
      start_cd = time.time()

      # perform collision detection
      ccd.cd(cmvp.compute().value, DHAT_VALUE) # perform collision detection
      end_cd = time.time()
      print(f"Time taken for collision detection: {end_cd - start_cd} seconds")
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      print("The separated counts are", ccd.separated_counts)
      start_update_collision = time.time()
      collision_mesh.pp.updateNumInstances(pp_count)
      collision_mesh.pe.updateNumInstances(pe_count)
      collision_mesh.pt.updateNumInstances(pt_count)
      collision_mesh.ee.updateNumInstances(ee_count)
      if pp_count > 0:
        pp2v.updateConnectivity(ccd.pp[:2 * pp_count])
      if pe_count > 0:
        pe2v.updateConnectivity(ccd.pe[:3 * pe_count])
      if pt_count > 0:
        pt2v.updateConnectivity(ccd.pt[:4 * pt_count])
      if ee_count > 0:
        ee2v.updateConnectivity(ccd.ee[:4 * ee_count])
      end_update_collision = time.time()
      print(f"Time taken for updating connectivity: {end_update_collision - start_update_collision} seconds")
      start_compute = time.time()
      new_energies = s0.computeTotalEnergy()
      end_compute = time.time()
      print(f"Time taken for computation: {end_compute - start_compute} seconds")

      print(f"energy comparison: {new_energies} vs {energies_before}")
      if new_energies <= energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    print("step taken is", step_taken)
    print("substep is", substep)

    string_poly.points = cmvp.compute().value.get().reshape(-1, 3)
    sphere_poly.points = spvp.compute().value.get().reshape(-1, 3)
    bunny_poly.points = bvp.compute().value.get().reshape(-1, 3)
    plotter.render()
    plotter.update()
    inner_iteration += 1
    if max_movement < 0.01:
      print(f"Iteration {inner_iteration} exited with max movement: {max_movement}")
      break
  start_velocity_update = time.time()
  new_velocities_string = (
    3.0 * svp.value
    - 4.0 * svlp.value
    + svllp.value
  ) / (2.0 * DT_VALUE)

  new_velocities_sphere = (
    3.0 * spvp.value
    - 4.0 * spvlp.value
    + spvllp.value
  ) / (2.0 * DT_VALUE)

  new_velocities_bunny = (
    3.0 * bvp.value
    - 4.0 * bvlp.value
    + bvllp.value
  ) / (2.0 * DT_VALUE)
  svv.updateValue(new_velocities_string, deepCopy = True)
  spvv.updateValue(new_velocities_sphere, deepCopy = True)
  bvv.updateValue(new_velocities_bunny, deepCopy = True)

  end_velocity_update = time.time()
  print(f"Time taken for velocity update: {end_velocity_update - start_velocity_update} seconds")
  # save the objs
  if (i % 10 == 0):
    string_poly.save(f"meshes/string_{(i // 10):04d}.obj")
    sphere_poly.save(f"meshes/sphere_{(i // 10):04d}.obj")
    bunny_poly.save(f"meshes/bunny_{(i // 10):04d}.obj") # save screenshot
    plotter.screenshot(f"frames/frame_{(i // 10):04d}.png")

end = time.time()
print("Total time: ", end - start)

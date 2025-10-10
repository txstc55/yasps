from yasps import scene
from yasps import attribute
from helpers import extract_surface_triangles, inertia, extract_edges_from_triangles, abs_max_reduce

import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
import random
random.seed(1313)

DT_VALUE = 0.01 # for time step
DHAT_VALUE = 1e-6 # for collision detection
KAPPA_VALUE = 10000000.0 # for collision


NUM_ABD_BUNNIES = 5
NUM_SOFT_BUNNIES = 5

MU_VALUE_ABDS = []
LAMBDA_VALUE_ABDS = []
for i in range(NUM_ABD_BUNNIES):
  POISSON_VALUE = 0.45 + random.random() * 0.04
  YOUNG_VALUE = 9000000.0 + random.random() * 1000000
  MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
  LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))
  MU_VALUE_ABDS.append(4.0 * MU_LAME_VALUE / 3.0)
  LAMBDA_VALUE_ABDS.append(LAMBDA_LAME_VALUE + 5.0 * MU_LAME_VALUE / 6.0)

MU_VALUE_SOFTS = []
LAMBDA_VALUE_SOFTS = []
for i in range(NUM_SOFT_BUNNIES):
  POISSON_VALUE = 0.20 + random.random() * 0.29
  YOUNG_VALUE = 5000.0 + random.random() * 40000
  MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
  LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))
  MU_VALUE_SOFTS.append(4.0 * MU_LAME_VALUE / 3.0)
  LAMBDA_VALUE_SOFTS.append(LAMBDA_LAME_VALUE + 5.0 * MU_LAME_VALUE / 6.0)

BENDING_STIFFNESS = 1.0
STRETCH_STIFFNESS = 100000000.0
SHEAR_STIFFNESS = 10000000.0
THICKNESS = 0.001

##################################################################
## Load the bunny mesh
##################################################################
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
position = np.array(position, dtype = np.float64) / 10.0
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
# computing the maximum of the minimum distance, which is the diagonal
diag = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)


surface_triangle_indices_bunny = extract_surface_triangles(tet_indices)
edge_indices_bunny = extract_edges_from_triangles(surface_triangle_indices_bunny)
surface_indices_bunny = list(set(surface_triangle_indices_bunny.flatten().tolist()))

NUM_BUNNY_VERTICES = position.shape[0]
NUM_BUNNY_TETS = tet_indices.shape[0]
NUM_BUNNY_SURFACE_TRIANGLES = surface_triangle_indices_bunny.shape[0]
NUM_BUNNY_EDGES = edge_indices_bunny.shape[0]
NUM_BUNNY_SURFACE_INDICES = len(surface_indices_bunny)
print("Number of bunny vertices: ", NUM_BUNNY_VERTICES)
print("Number of bunny tets: ", NUM_BUNNY_TETS)
print("Number of bunny surface triangles: ", NUM_BUNNY_SURFACE_TRIANGLES)
print("Number of bunny edges: ", NUM_BUNNY_EDGES)
print("Number of bunny surface indices: ", NUM_BUNNY_SURFACE_INDICES)

##################################################################
## Now we construct the indices for the bunnies
## also generate random positions and rotations
##################################################################
prev_center = np.array([0.0, 0.0, 0.0])
tet_indices_soft = []
position_softs = []
for i in range(NUM_SOFT_BUNNIES):
  tet_indices_soft.append(tet_indices + i * NUM_BUNNY_VERTICES)
  dir_x = random.random() - 0.5
  dir_y = random.random() * 1.0
  dir_z = random.random() - 0.5
  dir_length = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
  dir_x /= dir_length
  dir_y /= dir_length
  dir_z /= dir_length
  dir_x *= diag * 0.8
  dir_y *= diag * 0.8
  dir_z *= diag * 0.8
  translation = prev_center + np.array([dir_x, dir_y, dir_z])
  prev_center = translation
  # genrate random rotation matrix
  phi = random.random() * 2.0 * np.pi
  theta = random.random() * 2.0 * np.pi
  psi = random.random() * 2.0 * np.pi
  rotation_matrix = np.array([
    [np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
     np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi),
     np.sin(psi) * np.sin(theta)],
    [-np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi),
     -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi),
      np.cos(psi) * np.sin(theta)],
    [np.sin(theta) * np.sin(phi),
      -np.sin(theta) * np.cos(phi),
      np.cos(theta)]
    ])
  position_softs.append(np.dot(position, rotation_matrix) + translation)


tet_indices_abds = []
position_abds = []
translations = []
rotation_matrices = []
for i in range(NUM_ABD_BUNNIES):
  tet_indices_abds.append(tet_indices + i * NUM_BUNNY_VERTICES)
  position_abds.append(position)
  # first we generate a random direction that's always pointing up
  # this doesnt guarantee non-collision, but at least spreads them out
  dir_x = random.random() - 0.5
  dir_y = random.random() * 1.0
  dir_z = random.random() - 0.5
  dir_length = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
  dir_x /= dir_length
  dir_y /= dir_length
  dir_z /= dir_length
  dir_x *= diag * 0.8
  dir_y *= diag * 0.8
  dir_z *= diag * 0.8
  translations.append(prev_center + np.array([dir_x, dir_y, dir_z]))
  prev_center = translations[-1]
  # genrate random rotation matrix
  phi = random.random() * 2.0 * np.pi
  theta = random.random() * 2.0 * np.pi
  psi = random.random() * 2.0 * np.pi
  rotation_matrix = np.array([
    [np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.sin(psi),
     np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi),
     np.sin(psi) * np.sin(theta)],
    [-np.sin(psi) * np.cos(phi) - np.cos(theta) * np.sin(phi) * np.cos(psi),
     -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.cos(psi),
      np.cos(psi) * np.sin(theta)],
    [np.sin(theta) * np.sin(phi),
      -np.sin(theta) * np.cos(phi),
      np.cos(theta)]
    ])
  rotation_matrices.append(rotation_matrix)



##################################################################
## We will now construct the cloth
##################################################################
from helpers import generate_cloth_mesh
CLOTH_LENGTH = diag * 7.0
NUM_SEGMENTS = 50
positions_cloth, triangle_indices_cloth = generate_cloth_mesh(CLOTH_LENGTH, NUM_SEGMENTS)
# we need to pick out the index of the 4 corners
# because we want to mark them as another type of vertices
# indices of the four corners in the original layout
corner_old = [
  0,
  NUM_SEGMENTS,
  NUM_SEGMENTS * (NUM_SEGMENTS + 1),
  (NUM_SEGMENTS + 1) * (NUM_SEGMENTS + 1) - 1
]

# build a mapping from old index -> new index
num_vertices = positions_cloth.shape[0]
mapping = np.arange(num_vertices)

# assign 0,1,2,3 to the corners
for new_i, old_i in enumerate(corner_old):
  mapping[old_i] = new_i

# shift the rest so that they fill after 3
next_free = 4
for old_i in range(num_vertices):
  if old_i not in corner_old:
    mapping[old_i] = next_free
    next_free += 1

# apply the remap to triangles
triangle_indices_cloth = np.vectorize(lambda x: mapping[x])(triangle_indices_cloth)

# also reorder positions so indices line up
positions_cloth_new = np.empty_like(positions_cloth)
for old_i, new_i in enumerate(mapping):
  positions_cloth_new[new_i] = positions_cloth[old_i]

positions_cloth = positions_cloth_new + np.array([0.0, -0.2, 0.0])
from helpers import generate_edge_to_vertices_list
edge_to_vertices_cloth = generate_edge_to_vertices_list(triangle_indices_cloth)
edge_indices_cloth = extract_edges_from_triangles(triangle_indices_cloth)

##################################################################
## construct the abd meshes
##################################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue(DT_VALUE)
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue(DHAT_VALUE)
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue(KAPPA_VALUE)

bunnies_abd = s0.addMesh("bunnies_abd")
abds = bunnies_abd.addPrimitive("affine_bodies", numInstances = NUM_ABD_BUNNIES)
abds_abd_matrices = abds.addAttribute("affine_matrix", rows = 3, cols = 3)
abds_translations = abds.addAttribute("translation", rows = 3, cols = 1)

# update the values
abds_abd_matrices.updateValue(np.array(rotation_matrices, dtype = np.float64).flatten())
abds_translations.updateValue(np.array(translations, dtype = np.float64).flatten())

# now we construct the abd vertices
vertices_abd = bunnies_abd.addPrimitive("vertices_abd", numInstances = NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES)
vertices_abd_rest_position = vertices_abd.addConstant("rest_position", rows = 3, cols = 1)
vertices_abd_last_position = vertices_abd.addConstant("last_position", rows = 3, cols = 1)
vertices_abd_velocity = vertices_abd.addConstant("velocity", rows = 3, cols = 1)
vertices_abd_mass = vertices_abd.addConstant("mass", rows = 1, cols = 1)

# update the values
vertices_abd_rest_position.updateValue(np.array(position_abds, dtype = np.float64).flatten())
vertices_abd_velocity.updateValue(np.zeros(NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES * 3, dtype = np.float64))
vertices_abd_mass.updateValue(np.array([[(i + 1) * 0.5 / NUM_BUNNY_VERTICES] * NUM_BUNNY_VERTICES for i in range(NUM_ABD_BUNNIES)], dtype = np.float64).flatten())

# to actually get the current positions, we will need to first
# get the connectivity between vertices to the affine bodies
v_abd2_abd = vertices_abd.addConnectivity("v_abd2_abd", abds, [[i] * NUM_BUNNY_VERTICES for i in range(NUM_ABD_BUNNIES)], 1)
vertices_abd_abd_matrix = vertices_abd.addAttribute("affine_matrix", through = v_abd2_abd, source = abds_abd_matrices)
vertices_abd_abd_matrix = vertices_abd_abd_matrix.resize(3, 3) # we need to resize it to the correct dimension
vertices_abd_translation = vertices_abd.addAttribute("translation", through = v_abd2_abd, source = abds_translations)
vertices_abd_translation = vertices_abd_translation.resize(3, 1) # we need to resize it to the correct dimension

# now we can compute the current position
vertices_abd_position = vertices_abd.addAttribute("position", computed_attribute = vertices_abd_abd_matrix * vertices_abd_rest_position + vertices_abd_translation)
# and we can compute the current position to update it as the last position
vertices_abd_last_position.updateValue(vertices_abd_position.compute().value.get())

# now that we are done with vertices, we can do the tets
# each tet will have its corresponding mu and lambda values
# and all other attributes will be fetched from the vertices
tets_abds = bunnies_abd.addPrimitive("tets_abd", numInstances = NUM_ABD_BUNNIES * NUM_BUNNY_TETS)
tets_abds.addConstant("mu_abds", rows = 1, cols = 1)
tets_abds.addConstant("lambda_abds", rows = 1, cols = 1)

# add connectivity from tets to vertices
tets_abds2_vertices = tets_abds.addConnectivity("tets_abds2_vertices", vertices_abd, np.array(tet_indices_abds, dtype = np.uint32), 4)

# now we can get the rest positions and the current positions from the vertices
tets_abds_positions = tets_abds.addAttribute("positions", through = tets_abds2_vertices, source = vertices_abd_position)
tets_abds_rest_positions = tets_abds.addAttribute("rest_positions", through = tets_abds2_vertices, source = vertices_abd_rest_position)
tets_abds["mu_abds"].updateValue(np.array([[MU_VALUE_ABDS[i]] * NUM_BUNNY_TETS for i in range(NUM_ABD_BUNNIES)], dtype = np.float64).flatten())
tets_abds["lambda_abds"].updateValue(np.array([[LAMBDA_VALUE_ABDS[i]] * NUM_BUNNY_TETS for i in range(NUM_ABD_BUNNIES)], dtype = np.float64).flatten())

##################################################################
## construct the soft meshes
##################################################################
bunnies_soft = s0.addMesh("bunnies_soft")

# now we construct the soft vertices
vertices_soft = bunnies_soft.addPrimitive("vertices_soft", numInstances = NUM_SOFT_BUNNIES * NUM_BUNNY_VERTICES)
vertices_soft_position = vertices_soft.addAttribute("position", rows = 3, cols = 1)
vertices_soft_rest_position = vertices_soft.addConstant("rest_position", rows = 3, cols = 1)
vertices_soft_last_position = vertices_soft.addConstant("last_position", rows = 3, cols = 1)
vertices_soft_velocity = vertices_soft.addConstant("velocity", rows = 3, cols = 1)
vertices_soft_mass = vertices_soft.addConstant("mass", rows = 1, cols = 1)

# update the values
vertices_soft_position.updateValue(np.array(position_softs, dtype = np.float64).flatten())
vertices_soft_rest_position.updateValue(np.array(position_softs, dtype = np.float64).flatten())
vertices_soft_last_position.updateValue(np.array(position_softs, dtype = np.float64).flatten())
vertices_soft_velocity.updateValue(np.zeros(NUM_SOFT_BUNNIES * NUM_BUNNY_VERTICES * 3, dtype = np.float64))
vertices_soft_mass.updateValue(np.array([[(i + 1) * 1.0 / NUM_BUNNY_VERTICES] * NUM_BUNNY_VERTICES for i in range(NUM_SOFT_BUNNIES)], dtype = np.float64).flatten())


# now that we are done with vertices, we can do the tets
# each tet will have its corresponding mu and lambda values
# and all other attributes will be fetched from the vertices
tets_softs = bunnies_soft.addPrimitive("tets_soft", numInstances = NUM_SOFT_BUNNIES * NUM_BUNNY_TETS)
tets_softs.addConstant("mu_softs", rows = 1, cols = 1)
tets_softs.addConstant("lambda_softs", rows = 1, cols = 1)

# add connectivity from tets to vertices
tets_softs2_vertices = tets_softs.addConnectivity("tets_softs2_vertices", vertices_soft, np.array(tet_indices_soft, dtype = np.uint32), 4)

# now we can get the rest positions and the current positions from the vertices
tets_softs_positions = tets_softs.addAttribute("positions", through = tets_softs2_vertices, source = vertices_soft_position)
tets_softs_rest_positions = tets_softs.addAttribute("rest_positions", through = tets_softs2_vertices, source = vertices_soft_rest_position)
tets_softs["mu_softs"].updateValue(np.array([[MU_VALUE_SOFTS[i]] * NUM_BUNNY_TETS for i in range(NUM_SOFT_BUNNIES)], dtype = np.float64).flatten())
tets_softs["lambda_softs"].updateValue(np.array([[LAMBDA_VALUE_SOFTS[i]] * NUM_BUNNY_TETS for i in range(NUM_SOFT_BUNNIES)], dtype = np.float64).flatten())

##################################################################
## construct the cloth mesh
##################################################################
cloth = s0.addMesh("cloth")
stretch = cloth.addConstant("stretch", rows = 1, cols = 1)
stretch.updateValue(STRETCH_STIFFNESS)
shear = cloth.addConstant("shear", rows = 1, cols = 1)
shear.updateValue(SHEAR_STIFFNESS)
thickness = cloth.addConstant("thickness", rows = 1, cols = 1)
thickness.updateValue(THICKNESS)
bending_stiffness = cloth.addConstant("bending_stiffness", rows = 1, cols = 1)
bending_stiffness.updateValue(BENDING_STIFFNESS)

# first we add the fixed vertices, those are the 4 corners
vertices_fixed = cloth.addPrimitive("vertices_fixed", numInstances = 4)
vertices_fixed_position = vertices_fixed.addAttribute("position", rows = 3, cols = 1)
vertices_fixed_rest_position = vertices_fixed.addConstant("rest_position", rows = 3, cols = 1)
vertices_fixed_last_position = vertices_fixed.addConstant("last_position", rows = 3, cols = 1)
# update the values
vertices_fixed_position.updateValue(positions_cloth[:4].flatten())
vertices_fixed_rest_position.updateValue(positions_cloth[:4].flatten())
vertices_fixed_last_position.updateValue(positions_cloth[:4].flatten())



vertices_free = cloth.addPrimitive("vertices_free", numInstances = positions_cloth.shape[0] - 4)
vertices_free_position = vertices_free.addAttribute("position", rows = 3, cols = 1)
vertices_free_rest_position = vertices_free.addConstant("rest_position", rows = 3, cols = 1)
vertices_free_last_position = vertices_free.addConstant("last_position", rows = 3, cols = 1)
vertices_free_velocity = vertices_free.addConstant("velocity", rows = 3, cols = 1)
vertices_free_mass = vertices_free.addConstant("mass", rows = 1, cols = 1)
# update the values
vertices_free_position.updateValue(positions_cloth[4:].flatten())
vertices_free_rest_position.updateValue(positions_cloth[4:].flatten())
vertices_free_last_position.updateValue(positions_cloth[4:].flatten())
vertices_free_velocity.updateValue(np.zeros((positions_cloth.shape[0] - 4) * 3, dtype = np.float64))
vertices_free_mass.updateValue(np.full((positions_cloth.shape[0] - 4), 0.1 / positions_cloth.shape[0], dtype = np.float64))

# create the union
vertices_cloth = cloth.addPrimitiveUnion("vertices_cloth", [vertices_fixed, vertices_free])
vertices_cloth.addAttribute("position")
vertices_cloth.addAttribute("rest_position")
vertices_cloth.addAttribute("last_position")

# create the triangles
triangles_cloth = cloth.addPrimitive("triangles_cloth", numInstances = triangle_indices_cloth.shape[0])
triangles_cloth2_vertices = triangles_cloth.addConnectivity("triangles_cloth2_vertices", vertices_cloth, triangle_indices_cloth, 3)
triangles_cloth_positions = triangles_cloth.addAttribute("positions", through = triangles_cloth2_vertices, source = vertices_cloth["position"])
triangles_cloth_rest_positions = triangles_cloth.addAttribute("rest_positions", through = triangles_cloth2_vertices, source = vertices_cloth["rest_position"])

# create the edge to 4 vertices pair
edges_cloth = cloth.addPrimitive("edges_cloth", numInstances = edge_to_vertices_cloth.shape[0])
edges_cloth2_vertices = edges_cloth.addConnectivity("edges_cloth2_vertices", vertices_cloth, edge_to_vertices_cloth, 4)
edges_cloth_positions = edges_cloth.addAttribute("positions", through = edges_cloth2_vertices, source = vertices_cloth["position"])
edges_cloth_rest_positions = edges_cloth.addAttribute("rest_positions", through = edges_cloth2_vertices, source = vertices_cloth["rest_position"])

##################################################################
## we have finished the construction of our meshes
## now we need to add a collision mesh
##################################################################
collision_mesh = s0.addMesh("collision_mesh")
collision_vertices = collision_mesh.addPrimitiveUnion("vertices", [vertices_abd, vertices_soft, vertices_fixed, vertices_free])
collision_vertices_position = collision_vertices.addAttribute("position")
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

##################################################################
## ok now we add energies into the scene
##################################################################
from helpers import stable_neo_hookean
snh_abds = stable_neo_hookean(tets_abds_rest_positions, tets_abds_positions, tets_abds["mu_abds"], tets_abds["lambda_abds"], dt)
tets_abds.addAttribute("snh_abds", computed_attribute = snh_abds)
snh_softs = stable_neo_hookean(tets_softs_rest_positions, tets_softs_positions, tets_softs["mu_softs"], tets_softs["lambda_softs"], dt)
tets_softs.addAttribute("snh_softs", computed_attribute = snh_softs)

from helpers import affine_energy
affine = affine_energy(abds_abd_matrices)
abds.addAttribute("affine_energy", computed_attribute = affine)

from helpers import inertia
inertia_abds = inertia(vertices_abd_last_position, vertices_abd_velocity, dt, vertices_abd_position, vertices_abd_mass)
inertia_softs = inertia(vertices_soft_last_position, vertices_soft_velocity, dt, vertices_soft_position, vertices_soft_mass)
inertia_free = inertia(vertices_free_last_position, vertices_free_velocity, dt, vertices_free_position, vertices_free_mass)
vertices_abd.addAttribute("inertia_abds", computed_attribute = inertia_abds)
vertices_soft.addAttribute("inertia_softs", computed_attribute = inertia_softs)
vertices_free.addAttribute("inertia_free", computed_attribute = inertia_free)

from helpers import bending
bending_energy = bending(edges_cloth_positions, edges_cloth_rest_positions, bending_stiffness)
edges_cloth.addAttribute("bending_energy", computed_attribute = bending_energy)

from helpers import baraff_witkin
baraff_witkin_energy = baraff_witkin(triangles_cloth_rest_positions, triangles_cloth_positions,  stretch, shear, thickness, dt)
triangles_cloth.addAttribute("baraff_witkin_energy", computed_attribute = baraff_witkin_energy)

from helpers import point_point, point_edge, point_triangle, edge_edge, affine_energy
pp = point_point(pp_positions, dhat, kappa)
collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

s0.addEnergy(snh_abds, projection_method = 1)
s0.addEnergy(snh_softs, projection_method = 1)
s0.addEnergy(affine, projection_method = 1)
s0.addEnergy(inertia_abds, projection_method = 0)
s0.addEnergy(inertia_softs, projection_method = 0)
s0.addEnergy(inertia_free, projection_method = 0)
s0.addEnergy(bending_energy, projection_method = 2)
s0.addEnergy(baraff_witkin_energy, projection_method = 2)
s0.addEnergy(pp, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2)
s0.addMinimizeTarget([abds_abd_matrices, abds_translations, vertices_soft_position, vertices_free_position])

##################################################################
## add ccd
##################################################################
mesh_indices = []
for i in range(NUM_ABD_BUNNIES):
  mesh_indices += [i + 2] * NUM_BUNNY_VERTICES
for i in range(NUM_SOFT_BUNNIES):
  mesh_indices += [0] * NUM_BUNNY_VERTICES
mesh_indices += [0] * positions_cloth.shape[0]
ccd = CCD(NUM_BUNNY_SURFACE_INDICES * (NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) + positions_cloth.shape[0], # the number of surface points
  NUM_BUNNY_VERTICES * (NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) + positions_cloth.shape[0], # the number of total points
  max_ccd_pairs = 200000000,
  mesh_indices = mesh_indices
)

triangle_indices_all = []
surface_indices_all = []
for i in range(NUM_ABD_BUNNIES):
  triangle_indices_all.append((surface_triangle_indices_bunny + i * NUM_BUNNY_VERTICES).astype(np.uint32))
  surface_indices_all.append((np.array(surface_indices_bunny) + i * NUM_BUNNY_VERTICES).astype(np.uint32))
for i in range(NUM_SOFT_BUNNIES):
  triangle_indices_all.append((surface_triangle_indices_bunny + i * NUM_BUNNY_VERTICES + NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES).astype(np.uint32))
  surface_indices_all.append((np.array(surface_indices_bunny) + i * NUM_BUNNY_VERTICES + NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES).astype(np.uint32))
triangle_indices_all.append((triangle_indices_cloth + (NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES).astype(np.uint32))
surface_indices_all.append((np.array(range(positions_cloth.shape[0])) + (NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES).astype(np.uint32))
triangle_indices_all = np.vstack(triangle_indices_all, dtype = np.uint32)
surface_indices_all = np.hstack(surface_indices_all).astype(np.uint32)

edge_indices_all = []
for i in range(NUM_ABD_BUNNIES):
  edge_indices_all.append((edge_indices_bunny + i * NUM_BUNNY_VERTICES).astype(np.uint32))
for i in range(NUM_SOFT_BUNNIES):
  edge_indices_all.append((edge_indices_bunny + i * NUM_BUNNY_VERTICES + NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES).astype(np.uint32))
edge_indices_all.append((edge_indices_cloth + (NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES).astype(np.uint32))
edge_indices_all = np.vstack(edge_indices_all, dtype = np.uint32)

surface_indices_gpu = gpuarray.to_gpu(surface_indices_all.flatten())
edge_indices_gpu = gpuarray.to_gpu(edge_indices_all.flatten())
triangle_indices_gpu = gpuarray.to_gpu(triangle_indices_all.flatten())

position_gpu = gpuarray.to_gpu(collision_mesh.vertices["position"].compute().value.get()) # basically copy it out

ccd.init_faces(position_gpu, triangle_indices_gpu, surface_indices_gpu, triangle_indices_all.shape[0])
ccd.init_edges(position_gpu, position_gpu, edge_indices_gpu, edge_indices_all.shape[0])


##################################################################
## plot the bunnies
##################################################################
import pyvista as pv
plotter = pv.Plotter(window_size=[3840, 2160])
all_vertices_computed = collision_mesh.vertices["position"].compute().value.get().reshape((-1, 3))
triangles = triangle_indices_all
abd_triangles = triangles[:NUM_ABD_BUNNIES * NUM_BUNNY_SURFACE_TRIANGLES]
soft_triangles = triangles[NUM_ABD_BUNNIES * NUM_BUNNY_SURFACE_TRIANGLES:(NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_SURFACE_TRIANGLES] - NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES
cloth_triangles = triangles[(NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_SURFACE_TRIANGLES:] - (NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES
cells_abd = np.hstack([np.full((abd_triangles.shape[0], 1), 3), abd_triangles])
cells_soft = np.hstack([np.full((soft_triangles.shape[0], 1), 3), soft_triangles])
cells_cloth = np.hstack([np.full((cloth_triangles.shape[0], 1), 3), cloth_triangles])
abd_vertices_computed = all_vertices_computed[:NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES]
soft_vertices_computed = all_vertices_computed[NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES:(NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES]
cloth_vertices_computed = all_vertices_computed[(NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES:]
abd_poly = pv.PolyData(abd_vertices_computed, cells_abd)
soft_poly = pv.PolyData(soft_vertices_computed, cells_soft)
cloth_poly = pv.PolyData(cloth_vertices_computed, cells_cloth)
plotter.add_mesh(abd_poly, color = "lightblue")
plotter.add_mesh(soft_poly, color = "lightgreen")
plotter.add_mesh(cloth_poly, color = "pink", opacity = 0.5)

plotter.camera_position = [(0, 2, 6),
 (0.0, 0.0, 0.0),
 (0, 1, 0)
]
plotter.show(interactive_update=True, auto_close=False)
# plotter.show()
# exit()
position_copy = collision_mesh.vertices["position"].compute().value.copy()
rot_copy = gpuarray.zeros(NUM_ABD_BUNNIES * 9, dtype=np.float64)
trans_copy = gpuarray.zeros(NUM_ABD_BUNNIES * 3, dtype=np.float64)
bunny_abd_position_copy = vertices_abd_position.compute().value.copy()
bunny_soft_position_copy = vertices_soft_position.compute().value.copy()
cloth_free_position_copy = vertices_free["position"].compute().value.copy()

def compute_total_energy():
  total_energy = 0.0
  total_energy += sum(snh_abds.compute().value.get())
  total_energy += sum(snh_softs.compute().value.get())
  total_energy += sum(affine.compute().value.get())
  total_energy += sum(inertia_abds.compute().value.get())
  total_energy += sum(inertia_softs.compute().value.get())
  total_energy += sum(inertia_free.compute().value.get())
  total_energy += sum(bending_energy.compute().value.get())
  total_energy += sum(baraff_witkin_energy.compute().value.get())
  total_energy += sum(pp.compute().value.get())
  total_energy += sum(pe.compute().value.get())
  total_energy += sum(pt.compute().value.get())
  total_energy += sum(ee.compute().value.get())
  return total_energy

for i in range(500):
  bunnies_abd.vertices_abd["last_position"].updateValue(bunnies_abd.vertices_abd["position"].compute().value, deepCopy = True)
  bunnies_soft.vertices_soft["last_position"].updateValue(bunnies_soft.vertices_soft["position"].value, deepCopy = True)
  inner_iteration = 0
  min_inner_iteration_energy = 100000000
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-3)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer
    energies_before = compute_total_energy()
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    # we perform CCD here
    # first we get the rotation and translation
    d_rot = result[0]
    d_trans = result[1]
    d_pos_soft_bunny = result[2]
    d_pos_cloth_free = result[3]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    rot_copy.set(abds_abd_matrices.value)
    trans_copy.set(abds_translations.value)
    bunny_abd_position_copy = vertices_abd_position.compute().value.copy()
    bunny_soft_position_copy.set(vertices_soft_position.compute().value)
    cloth_free_position_copy.set(vertices_free["position"].compute().value)

    # now update the value
    abds_abd_matrices.updateValue(abds_abd_matrices.value - d_rot, deepCopy = True)
    abds_translations.updateValue(abds_translations.value - d_trans, deepCopy = True)
    vertices_soft_position.updateValue(vertices_soft_position.value - d_pos_soft_bunny, deepCopy = True)
    vertices_free["position"].updateValue(vertices_free["position"].value - d_pos_cloth_free, deepCopy = True)

    # compute the new positions for the entire scene
    new_positions = collision_mesh.vertices["position"].compute().value
    # now we compute the new direction, remember it's the negative we need to put in
    direction_copy = position_copy - new_positions

    # check for the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    while substep <= 8:
      abds_abd_matrices.updateValue(rot_copy - d_rot * step_taken, deepCopy = True)
      abds_translations.updateValue(trans_copy - d_trans * step_taken, deepCopy = True)
      vertices_soft_position.updateValue(bunny_soft_position_copy - d_pos_soft_bunny * step_taken, deepCopy = True)
      vertices_free["position"].updateValue(cloth_free_position_copy - d_pos_cloth_free * step_taken, deepCopy = True)

  #     # perform collision detection
      ccd.cd(collision_mesh.vertices["position"].compute().value, DHAT_VALUE) # perform collision detection
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      print("The separated counts are", ccd.separated_counts)
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
      new_energies = compute_total_energy()
      print(f"energy comparison: {new_energies} vs {energies_before}")
      if new_energies <= energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
  #   if substep > 8:
  #     print("failed")
  #     exit(1)
    print("step taken is", step_taken)
    print("substep is", substep)
    all_vertices_computed = collision_mesh.vertices["position"].compute().value.get().reshape((-1, 3))

    abd_vertices_computed = all_vertices_computed[:NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES]
    soft_vertices_computed = all_vertices_computed[NUM_ABD_BUNNIES * NUM_BUNNY_VERTICES:(NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES]
    cloth_vertices_computed = all_vertices_computed[(NUM_ABD_BUNNIES + NUM_SOFT_BUNNIES) * NUM_BUNNY_VERTICES:]
    abd_poly.points = abd_vertices_computed
    soft_poly.points = soft_vertices_computed
    cloth_poly.points = cloth_vertices_computed
    plotter.render()
    plotter.update()

    # print(f"Iteration {inner_iteration} max gradient: {max_grad}")
    if max_grad < 5e-3:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities_abd = (vertices_abd_position.compute().value - vertices_abd_last_position.value) / DT_VALUE
  new_velocities_soft = (vertices_soft_position.compute().value - vertices_soft_last_position.value) / DT_VALUE
  new_velocities_free = (vertices_free["position"].compute().value - vertices_free_last_position.value) / DT_VALUE
  vertices_abd_velocity.updateValue(new_velocities_abd, deepCopy = True)
  vertices_soft_velocity.updateValue(new_velocities_soft, deepCopy = True)
  vertices_free_velocity.updateValue(new_velocities_free, deepCopy = True)
  # plotter.render()
  # plotter.update()
  plotter.screenshot(f"outputs/many_bunny_one_cloth_{i:04d}.jpg")
  # save the mesh obj file
  abd_poly.save(f"meshes/bunny_abd_{i:04d}.obj")
  soft_poly.save(f"meshes/bunny_soft_{i:04d}.obj")
  cloth_poly.save(f"meshes/cloth_{i:04d}.obj")
  # # save the mesh obj file
  # bunny_poly0.save(f"outputs/bunny_abd_soft0_{i:04d}.obj")
  # bunny_poly1.save(f"outputs/bunny_abd_soft1_{i:04d}.obj")

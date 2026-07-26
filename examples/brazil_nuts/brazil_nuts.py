from yasps import scene
from yasps import attribute
from helpers import extract_surface_triangles, inertia, extract_edges_from_triangles, random_rotation_matrix

import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
from yasps.backend import gpuarray
import random
random.seed(1313)
np.random.seed(13)      # for numpy
import time
import random
from pathlib import Path
import os
random.seed(13)
rng = np.random.default_rng(seed=13)
SHOW_EXAMPLE = os.environ.get("YASPS_EXAMPLE_SHOW", "1") != "0"
SAVE_EXAMPLE = os.environ.get("YASPS_EXAMPLE_SAVE", "1") != "0"

NUM_BUNNIES = 1
COUNT_PER_DIM = 15
NUM_Y_SCALE = 24

AFFINE_WEIGHT = 1.0
ROTATION_WEIGHT = 1.0
CONSTRAINED_WEIGHT = 200000000 # for moving the container

DT_VALUE = 0.01 # for time step
DHAT_VALUE = 2e-7 # for collision detection
KAPPA_VALUE = 100000.0 # for collision
FRICTION_RATE = 0.2
MOVEMENT_PER_FRAME = 0.005 # 1CM per frame


POISSON_VALUE = 0.1645697005781997
YOUNG_VALUE = 125925455.816859
MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))

CONTAINER_EDGE_LENGTH = 0.2 # 20 cm container

START_FRAME = 10

A = 0.012          # 1.5 cm amplitude
PERIOD = 0.1      # seconds, 20 frames
OMEGA = 2.0 * np.pi / PERIOD



###########################################################
# Create bunny
###########################################################
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

position = np.array(position, dtype = np.float64) / 80.0
position_z = position[:, 2].copy()
position[:, 2] = position[:, 1].copy()
position[:, 1] = -position_z

# get the bounding box of the bunny
min_x, max_x = np.min(position[:, 0]), np.max(position[:, 0])
min_y, max_y = np.min(position[:, 1]), np.max(position[:, 1])
min_z, max_z = np.min(position[:, 2]), np.max(position[:, 2])
# now center the bunny
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
center_z = (min_z + max_z) / 2
position[:, 0] -= center_x
position[:, 1] -= center_y
position[:, 2] -= center_z
position[:, 1] += 0.03
min_x, max_x = np.min(position[:, 0]), np.max(position[:, 0])
min_y, max_y = np.min(position[:, 1]), np.max(position[:, 1])
min_z, max_z = np.min(position[:, 2]), np.max(position[:, 2])
print("Bunny bounding box:")
print("X: ", min_x, max_x)
print("Y: ", min_y, max_y)
print("Z: ", min_z, max_z)



# computing the maximum of the minimum distance, which is the diagonal
diag = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2)

surface_triangle_indices_bunny = extract_surface_triangles(tet_indices).astype(np.uint32)
edge_indices_bunny = extract_edges_from_triangles(surface_triangle_indices_bunny).astype(np.uint32)
surface_indices_bunny = list(set(surface_triangle_indices_bunny.flatten().tolist()))

NUM_BUNNY_VERTICES = position.shape[0]
NUM_BUNNY_TETS = tet_indices.shape[0]
NUM_BUNNY_SURFACE_TRIANGLES = surface_triangle_indices_bunny.shape[0]
NUM_BUNNY_EDGES = len(edge_indices_bunny)
NUM_BUNNY_SURFACE_INDICES = len(surface_indices_bunny)
print("Number of bunny vertices: ", NUM_BUNNY_VERTICES)
print("Number of bunny tets: ", NUM_BUNNY_TETS)
print("Number of bunny surface triangles: ", NUM_BUNNY_SURFACE_TRIANGLES)
print("Number of bunny edges: ", NUM_BUNNY_EDGES)
print("Number of bunny surface indices: ", NUM_BUNNY_SURFACE_INDICES)

all_tets = tet_indices.copy()
all_vertices = position.copy()
all_surface_indices = surface_indices_bunny
all_edges = edge_indices_bunny.copy()
all_surface_triangles = surface_triangle_indices_bunny.copy()


###########################################################
# Create small spheres
###########################################################
sphere_vertices = []
sphere_faces = []
sphere_path = Path("../data/sphere_low_poly.obj")
if not sphere_path.exists():
  sphere_path = Path("../data/sphere_small.obj")
f = sphere_path.open('r')
for line in f:
  if line.startswith('v '):
    sphere_vertices.append([float(x) for x in line.split()[1:]])
  if line.startswith('f '):
    face = []
    for x in line.split()[1:]:
      v_idx = int(x.split('/')[0]) - 1
      face.append(v_idx)
    sphere_faces.append(face)
sphere_vertices = np.array(sphere_vertices, dtype=np.float64) / 150
sphere_faces = np.array(sphere_faces, dtype=np.uint32)
sphere_edges = extract_edges_from_triangles(sphere_faces).astype(np.uint32)

min_x, max_x = np.min(sphere_vertices[:, 0]), np.max(sphere_vertices[:, 0])
min_y, max_y = np.min(sphere_vertices[:, 1]), np.max(sphere_vertices[:, 1])
min_z, max_z = np.min(sphere_vertices[:, 2]), np.max(sphere_vertices[:, 2])
print("Small sphere bounding box:")
print("X: ", min_x, max_x)
print("Y: ", min_y, max_y)
print("Z: ", min_z, max_z)



container_start = -CONTAINER_EDGE_LENGTH / 2.0
container_end = CONTAINER_EDGE_LENGTH / 2.0
total_spheres = 0
for i in range(COUNT_PER_DIM):
  for j in range(COUNT_PER_DIM):
    for k in range(NUM_Y_SCALE):
      center = container_start + (i + 0.5) * CONTAINER_EDGE_LENGTH / COUNT_PER_DIM, 0.0 + (k + 0.5) * CONTAINER_EDGE_LENGTH / 8.0 + 0.05, container_start + (j + 0.5) * CONTAINER_EDGE_LENGTH / COUNT_PER_DIM,
      center = np.array(center)
      center += np.array([random.uniform(-0.0005, 0.0005), random.uniform(-0.0005, 0.0005), random.uniform(-0.0005, 0.0005)])
      new_points = sphere_vertices.copy()
      new_points[:, 0] *= random.uniform(0.2, 1.0)
      new_points[:, 1] *= random.uniform(0.2, 1.0)
      new_points[:, 2] *= random.uniform(0.2, 1.0)
      R = random_rotation_matrix(rng)
      new_points = new_points @ R.T
      new_points = new_points + np.array(center)

      # randomly sample some points on the inside
      random_weights = rng.random(sphere_vertices.shape[0])
      random_weights = random_weights / sum(random_weights)
      random_point = (random_weights[:, np.newaxis] * new_points).sum(axis = 0)
      new_points = np.concatenate((new_points, random_point[np.newaxis, :]), axis = 0)

      offset = all_vertices.shape[0]
      all_vertices = np.concatenate((all_vertices, new_points), axis = 0)
      all_surface_triangles = np.concatenate((all_surface_triangles, sphere_faces + offset), axis = 0)
      all_surface_indices += list(range(offset, offset + sphere_vertices.shape[0]))
      all_edges = np.concatenate((all_edges, sphere_edges + offset), axis = 0)
      total_spheres += 1

sphere_vertices_count = sphere_vertices.shape[0] + 1

###########################################################
# Create the container
###########################################################
f = open("../data/container.obj", 'r')
container_positions = []
container_surface_triangles = []
for line in f:
  if line.startswith('v '):
    container_positions.append([float(x) for x in line.split()[1:]])
  elif line.startswith('f '):
    container_surface_triangles.append([int(x) - 1 for x in line.split()[1:]])

container_positions = np.array(container_positions, dtype = np.float64).reshape(-1, 3)

container_positions[:, 0] /= 10.0
container_positions[:, 1] /= 8.0
container_positions[:, 2] /= 10.0
min_x, max_x = np.min(container_positions[:, 0]), np.max(container_positions[:, 0])
min_y, max_y = np.min(container_positions[:, 1]), np.max(container_positions[:, 1])
min_z, max_z = np.min(container_positions[:, 2]), np.max(container_positions[:, 2])
print("Container bounding box:")
print("X: ", min_x, max_x)
print("Y: ", min_y, max_y)
print("Z: ", min_z, max_z)


container_surface_triangles = np.array(container_surface_triangles, dtype = np.uint32).reshape(-1, 3)
container_edge_indices = extract_edges_from_triangles(container_surface_triangles).astype(np.uint32)

offset = all_vertices.shape[0]
all_vertices = np.concatenate((all_vertices, container_positions), axis = 0)
all_surface_triangles = np.concatenate((all_surface_triangles, container_surface_triangles + offset), axis = 0)
all_edges = np.concatenate((all_edges, container_edge_indices + offset), axis = 0)
all_surface_indices += list(range(offset, offset + container_positions.shape[0]))

##################################################################
## construct the scene
##################################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue(DT_VALUE)
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue(DHAT_VALUE)
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue(KAPPA_VALUE)
friction_rate = s0.addConstant("friction_rate", rows = 1, cols = 1)
friction_rate.updateValue(FRICTION_RATE)
affine_weight = s0.addConstant("affine_weight", rows = 1, cols = 1)
affine_weight.updateValue(AFFINE_WEIGHT)
rotation_weight = s0.addConstant("rotation_weight", rows = 1, cols = 1)
rotation_weight.updateValue(ROTATION_WEIGHT)
constrained_weight = s0.addConstant("constrained_weight", rows = 1, cols = 1)
constrained_weight.updateValue(CONSTRAINED_WEIGHT)


##################################################################
## construct the bunny meshes
##################################################################
bunnies_soft = s0.addMesh("bunnies_soft")
mu = bunnies_soft.addConstant("mu", rows = 1, cols = 1)
mu.updateValue([MU_LAME_VALUE])
lam = bunnies_soft.addConstant("lambda", rows = 1, cols = 1)
lam.updateValue([LAMBDA_LAME_VALUE])

# now we construct the soft vertices
vertices_bunny = bunnies_soft.addPrimitive("vertices", numInstances = NUM_BUNNIES * NUM_BUNNY_VERTICES)
vertices_bunny_position = vertices_bunny.addAttribute("position", rows = 3, cols = 1)
vertices_bunny_rest_position = vertices_bunny.addConstant("rest_position", rows = 3, cols = 1)
vertices_bunny_last_position = vertices_bunny.addConstant("last_position", rows = 3, cols = 1)
vertices_bunny_velocity = vertices_bunny.addConstant("velocity", rows = 3, cols = 1)
vertices_bunny_mass = vertices_bunny.addConstant("mass", rows = 1, cols = 1)

# update the values
vertices_bunny_position.updateValue(all_vertices[: NUM_BUNNY_VERTICES])
vertices_bunny_rest_position.updateValue(all_vertices[: NUM_BUNNY_VERTICES])
vertices_bunny_last_position.updateValue(all_vertices[: NUM_BUNNY_VERTICES])
vertices_bunny_velocity.updateValue(np.zeros(NUM_BUNNIES * NUM_BUNNY_VERTICES * 3, dtype = np.float64))
vertices_bunny_mass.updateValue(np.array([[3.0 / NUM_BUNNY_VERTICES] * NUM_BUNNY_VERTICES for _ in range(NUM_BUNNIES)], dtype = np.float64).flatten())

##################################################################
## create the tets for the bunnies
##################################################################
tets_bunny = bunnies_soft.addPrimitive("tets", numInstances = NUM_BUNNIES * NUM_BUNNY_TETS)
tb2v = tets_bunny.addConnectivity("tb2v", vertices_bunny, tet_indices, 4)
tbp = tets_bunny.addAttribute("tbp", through = tb2v, source = vertices_bunny_position)
tbrp = tets_bunny.addAttribute("tbrp", through = tb2v, source = vertices_bunny_rest_position)

##################################################################
## construct the small sphere meshes
##################################################################
spheres = s0.addMesh("spheres")

sphere_abds = spheres.addPrimitive("affine_bodies", numInstances = total_spheres)
affine_matrices = sphere_abds.addAttribute("affine_matrix", rows = 3, cols = 3)
affine_matrices.updateValue(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] * total_spheres, dtype = np.float64).flatten())
translations = sphere_abds.addAttribute("translation", rows = 3, cols = 1)
translations.updateValue(np.array([[0.0, 0.0, 0.0]] * total_spheres, dtype = np.float64).flatten())
print(translations.compute().value.get().reshape(-1, 3).shape)

vertices_sphere = spheres.addPrimitive("vertices", numInstances = sphere_vertices_count * total_spheres)
v2abd = vertices_sphere.addConnectivity("v2abd", sphere_abds, np.array([[i] * sphere_vertices_count for i in range(total_spheres)], dtype = np.uint32).flatten(), 1)

vertices_sphere_rest_position = vertices_sphere.addConstant("rest_position", rows = 3, cols = 1)
vertices_sphere_rest_position.updateValue(all_vertices[NUM_BUNNIES * NUM_BUNNY_VERTICES : -container_positions.shape[0]])
vertices_sphere_last_position = vertices_sphere.addConstant("last_position", rows = 3, cols = 1)
vertices_sphere_last_position.updateValue(all_vertices[NUM_BUNNIES * NUM_BUNNY_VERTICES : -container_positions.shape[0]])
vertices_sphere_velocity = vertices_sphere.addConstant("velocity", rows = 3, cols = 1)
vertices_sphere_velocity.updateValue(np.zeros(sphere_vertices_count * 3 * total_spheres, dtype = np.float64))
vertices_sphere_mass = vertices_sphere.addConstant("mass", rows = 1, cols = 1)

random_masses = rng.random(sphere_vertices_count * total_spheres) * 0.08 / (sphere_vertices_count)
random_masses[:: sphere_vertices_count] += 0.04 # make the center point heavier
vertices_sphere_mass.updateValue(random_masses)

# compute the vertices position
vertices_affine_matrices = vertices_sphere.addAttribute("affine_matrices", through = v2abd, source = affine_matrices)
vertices_affine_matrices = vertices_affine_matrices.resize(3, 3)
vertices_translation = vertices_sphere.addAttribute("translation", through = v2abd, source = translations)
vertices_translation = vertices_translation.resize(3, 1)
vertices_sphere_position = vertices_affine_matrices * vertices_sphere_rest_position + vertices_translation
vertices_sphere_position = vertices_sphere.addAttribute("position", computed_attribute = vertices_sphere_position)

##################################################################
## construct the container
##################################################################
container = s0.addMesh("container")
vertices_container = container.addPrimitive("vertices", numInstances = container_positions.shape[0])
# vertices_container_positions = vertices_container.addAttribute("position", rows = 3, cols = 1)
# vertices_container_positions.updateValue(container_positions)
vertices_container_last_positions = vertices_container.addConstant("last_position", rows = 3, cols = 1)
vertices_container_last_positions.updateValue(container_positions)
vertices_container_target_positions = vertices_container.addConstant("target_position", rows = 3, cols = 1)
vertices_container_target_positions.updateValue(container_positions)
vertices_container_mass = vertices_container.addConstant("mass", rows = 1, cols = 1)
vertices_container_mass.updateValue(np.ones(container_positions.shape[0], dtype = np.float64) * 5.0 / container_positions.shape[0])
vertices_container_velocity = vertices_container.addConstant("velocity", rows = 3, cols = 1)
vertices_container_velocity.updateValue(np.zeros(container_positions.shape[0] * 3, dtype = np.float64))
vertices_container_rest_position = vertices_container.addConstant("rest_position", rows = 3, cols = 1)
vertices_container_rest_position.updateValue(container_positions)

container_y_translation = container.addAttribute("y_translation", rows = 1, cols = 1)
container_y_translation.updateValue([0])
vertices_container_positions = attribute.to_array([vertices_container_rest_position[0], vertices_container_rest_position[1] + container_y_translation, vertices_container_rest_position[2]], rows = 3, cols = 1)
vertices_container_positions = vertices_container.addAttribute("position", computed_attribute = vertices_container_positions)

# # add triangles
# triangles_container = container.addPrimitive("triangles", numInstances = container_surface_triangles.shape[0])
# triangles_container2_vertices = triangles_container.addConnectivity("triangles_container2_vertices", vertices_container, np.array(container_surface_triangles, dtype = np.uint32), 3)
# triangles_container_positions = triangles_container.addAttribute("positions", through = triangles_container2_vertices, source = vertices_container_positions)

##################################################################
## we have finished the construction of our meshes
## now we need to add a collision mesh
##################################################################
collision_mesh = s0.addMesh("collision_mesh")
collision_vertices = collision_mesh.addPrimitiveUnion("vertices", [vertices_bunny, vertices_sphere, vertices_container])
collision_vertices_position = collision_vertices.addAttribute("position")
collision_vertices_last_position = collision_vertices.addAttribute("last_position")
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


pp_friction_pairs = collision_mesh.addPrimitive("pp_friction_pairs", numInstances = 0, isDynamic = True)
pe_friction_pairs = collision_mesh.addPrimitive("pe_friction_pairs", numInstances = 0, isDynamic = True)
pt_friction_pairs = collision_mesh.addPrimitive("pt_friction_pairs", numInstances = 0, isDynamic = True)
ee_friction_pairs = collision_mesh.addPrimitive("ee_friction_pairs", numInstances = 0, isDynamic = True)
pp_friction2v = collision_mesh.pp_friction_pairs.addConnectivity("pp_friction2v", collision_mesh.vertices, [], 2)
pe_friction2v = collision_mesh.pe_friction_pairs.addConnectivity("pe_friction2v", collision_mesh.vertices, [], 3)
pt_friction2v = collision_mesh.pt_friction_pairs.addConnectivity("pt_friction2v", collision_mesh.vertices, [], 4)
ee_friction2v = collision_mesh.ee_friction_pairs.addConnectivity("ee_friction2v", collision_mesh.vertices, [], 4)
pp_friction_positions = collision_mesh.pp_friction_pairs.addAttribute("positions", through = pp_friction2v, source = collision_mesh.vertices["position"])
pe_friction_positions = collision_mesh.pe_friction_pairs.addAttribute("positions", through = pe_friction2v, source = collision_mesh.vertices["position"])
pt_friction_positions = collision_mesh.pt_friction_pairs.addAttribute("positions", through = pt_friction2v, source = collision_mesh.vertices["position"])
ee_friction_positions = collision_mesh.ee_friction_pairs.addAttribute("positions", through = ee_friction2v, source = collision_mesh.vertices["position"])
pp_friction_last_positions = collision_mesh.pp_friction_pairs.addAttribute("last_positions", through = pp_friction2v, source = collision_mesh.vertices["last_position"])
pe_friction_last_positions = collision_mesh.pe_friction_pairs.addAttribute("last_positions", through = pe_friction2v, source = collision_mesh.vertices["last_position"])
pt_friction_last_positions = collision_mesh.pt_friction_pairs.addAttribute("last_positions", through = pt_friction2v, source = collision_mesh.vertices["last_position"])
ee_friction_last_positions = collision_mesh.ee_friction_pairs.addAttribute("last_positions", through = ee_friction2v, source = collision_mesh.vertices["last_position"])

from friction_helpers import closest_point_coord_and_tangent_basis_pp, closest_point_coord_and_tangent_basis_pe, closest_point_coord_and_tangent_basis_pt, closest_point_coord_and_tangent_basis_ee, lambda_last_h_pp, lambda_last_h_pe, lambda_last_h_pt, lambda_last_h_ee
pp_coord, pp_tangent_basis = closest_point_coord_and_tangent_basis_pp(pp_friction_last_positions)
pe_coord, pe_tangent_basis = closest_point_coord_and_tangent_basis_pe(pe_friction_last_positions)
pt_coord, pt_tangent_basis = closest_point_coord_and_tangent_basis_pt(pt_friction_last_positions)
ee_coord, ee_tangent_basis = closest_point_coord_and_tangent_basis_ee(ee_friction_last_positions)
pp_friction_pairs.addAttribute("coord", computed_attribute = pp_coord)
pe_friction_pairs.addAttribute("coord", computed_attribute = pe_coord)
pt_friction_pairs.addAttribute("coord", computed_attribute = pt_coord)
ee_friction_pairs.addAttribute("coord", computed_attribute = ee_coord)
pp_friction_pairs.addAttribute("tangent_basis", computed_attribute = pp_tangent_basis)
pe_friction_pairs.addAttribute("tangent_basis", computed_attribute = pe_tangent_basis)
pt_friction_pairs.addAttribute("tangent_basis", computed_attribute = pt_tangent_basis)
ee_friction_pairs.addAttribute("tangent_basis", computed_attribute = ee_tangent_basis)

pp_friction_lambda_last_h = lambda_last_h_pp(pp_friction_last_positions, pp_coord, dhat, kappa)
pe_friction_lambda_last_h = lambda_last_h_pe(pe_friction_last_positions, pe_coord, dhat, kappa)
pt_friction_lambda_last_h = lambda_last_h_pt(pt_friction_last_positions, pt_coord, dhat, kappa)
ee_friction_lambda_last_h = lambda_last_h_ee(ee_friction_last_positions, ee_coord, dhat, kappa)
pp_friction_pairs.addAttribute("lambda_last_h", computed_attribute = pp_friction_lambda_last_h)
pe_friction_pairs.addAttribute("lambda_last_h", computed_attribute = pe_friction_lambda_last_h)
pt_friction_pairs.addAttribute("lambda_last_h", computed_attribute = pt_friction_lambda_last_h)
ee_friction_pairs.addAttribute("lambda_last_h", computed_attribute = ee_friction_lambda_last_h)


##################################################################
## ok now we add energies into the scene
##################################################################
from helpers import stable_neo_hookean
snh_softs = stable_neo_hookean(tbrp, tbp, bunnies_soft["mu"], bunnies_soft["lambda"], dt)
tets_bunny.addAttribute("snh_bunny", computed_attribute = snh_softs)



from helpers import inertia
inertia_softs = inertia(vertices_bunny_last_position, vertices_bunny_velocity, dt, vertices_bunny_position, vertices_bunny_mass)
vertices_bunny.addAttribute("inertia_softs", computed_attribute = inertia_softs)
inertia_sphere = inertia(vertices_sphere_last_position, vertices_sphere_velocity, dt, vertices_sphere_position, vertices_sphere_mass)
vertices_sphere.addAttribute("inertia_sphere", computed_attribute = inertia_sphere)
inertia_container = inertia(vertices_container_last_positions, vertices_container_velocity, dt, vertices_container_positions, vertices_container_mass)
vertices_container.addAttribute("inertia_container", computed_attribute = inertia_container)

from helpers import point_point, point_edge, point_triangle, edge_edge
pp = point_point(pp_positions, dhat, kappa)
collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

from friction_helpers import friction_energy_pp, friction_energy_pe, friction_energy_pt, friction_energy_ee
pp_friction = friction_energy_pp(pp_friction_positions, pp_friction_last_positions, dhat, dt, friction_rate, pp_coord, pp_tangent_basis.row(0), pp_tangent_basis.row(1), pp_friction_lambda_last_h)
pp_friction_pairs.addAttribute("friction_energy", computed_attribute = pp_friction)
pe_friction = friction_energy_pe(pe_friction_positions, pe_friction_last_positions, dhat, dt, friction_rate, pe_coord, pe_tangent_basis.row(0), pe_tangent_basis.row(1), pe_friction_lambda_last_h)
pe_friction_pairs.addAttribute("friction_energy", computed_attribute = pe_friction)
pt_friction = friction_energy_pt(pt_friction_positions, pt_friction_last_positions, dhat, dt, friction_rate, pt_coord, pt_tangent_basis.row(0), pt_tangent_basis.row(1), pt_friction_lambda_last_h)
pt_friction_pairs.addAttribute("friction_energy", computed_attribute = pt_friction)
ee_friction = friction_energy_ee(ee_friction_positions, ee_friction_last_positions, dhat, dt, friction_rate, ee_coord, ee_tangent_basis.row(0), ee_tangent_basis.row(1), ee_friction_lambda_last_h)
ee_friction_pairs.addAttribute("friction_energy", computed_attribute = ee_friction)

from helpers import affine_energy, rotation_energy
affine_constraint = affine_energy(affine_matrices, affine_weight)
sphere_abds.addAttribute("affine_energy", computed_attribute = affine_constraint)
rotation_constraint = rotation_energy(affine_matrices, rotation_weight)
sphere_abds.addAttribute("rotation_energy", computed_attribute = rotation_constraint)

from helpers import constrained_energy
container_vertex_constrained = constrained_energy(vertices_container_positions, vertices_container_target_positions, dt, constrained_weight)
vertices_container.addAttribute("constrained_energy", computed_attribute = container_vertex_constrained)


s0.addEnergy(snh_softs, projection_method = 1)
s0.addEnergy(inertia_softs, projection_method = -1)
s0.addEnergy(inertia_sphere, projection_method = -1)
s0.addEnergy(inertia_container, projection_method = -1)
s0.addEnergy(affine_constraint, projection_method = 2)
s0.addEnergy(rotation_constraint, projection_method = 2)
s0.addEnergy(container_vertex_constrained, projection_method = -1)
s0.addEnergy(pp, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pp_friction, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pe_friction, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(pt_friction, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addEnergy(ee_friction, dynamic_instances = True, projection_method = 2, separate_hessian_jacobian = True)
s0.addMinimizeTarget([vertices_bunny_position, affine_matrices, translations, container_y_translation])
# exit()

##################################################################
## add ccd
##################################################################
mesh_indices = [0] * NUM_BUNNY_VERTICES * NUM_BUNNIES
for i in range(COUNT_PER_DIM * COUNT_PER_DIM * NUM_Y_SCALE):
  mesh_indices += [i + 3] * sphere_vertices_count
mesh_indices += [2] * container_positions.shape[0]
ccd = CCD(len(all_surface_indices), # the number of surface points
  all_vertices.shape[0], # the number of total points
  max_ccd_pairs = 150000000,
  max_cd_pairs = 150000000,
  mesh_indices = mesh_indices
)


triangle_indices_all = all_surface_triangles
surface_indices_all = np.array(all_surface_indices, dtype = np.uint32)
edge_indices_all = all_edges


surface_indices_gpu = gpuarray.to_gpu(surface_indices_all.flatten())
edge_indices_gpu = gpuarray.to_gpu(edge_indices_all.flatten())
triangle_indices_gpu = gpuarray.to_gpu(triangle_indices_all.flatten())
position_gpu = collision_vertices_position.compute().value.copy()

ccd.init_faces(position_gpu, triangle_indices_gpu, surface_indices_gpu, triangle_indices_all.shape[0])
ccd.init_edges(position_gpu, position_gpu, edge_indices_gpu, edge_indices_all.shape[0])


# loaded_positions_gpu = gpuarray.to_gpu(loaded_positions)
# loaded_directions_gpu = gpuarray.to_gpu(loaded_directions)
# ccd.ccd(loaded_positions_gpu, DHAT_VALUE, loaded_directions_gpu, 1.0)
# largest_step = ccd.compute_largest_step_size(0.5, loaded_positions_gpu, loaded_directions_gpu)
# exit()


###########################################################
# draw with pyvista
###########################################################
import pyvista as pv
plotter = pv.Plotter(window_size=[1920, 1080])
triangles = all_surface_triangles
soft_triangles = triangles[0 :(NUM_BUNNIES) * NUM_BUNNY_SURFACE_TRIANGLES]
container_triangles = triangles[-container_surface_triangles.shape[0]:] - (all_vertices.shape[0] - container_positions.shape[0])
sphere_triangles = triangles[(NUM_BUNNIES) * NUM_BUNNY_SURFACE_TRIANGLES : -container_surface_triangles.shape[0]] - (NUM_BUNNIES) * NUM_BUNNY_VERTICES

cells_soft = np.hstack([np.full((soft_triangles.shape[0], 1), 3), soft_triangles])
cells_container = np.hstack([np.full((container_triangles.shape[0], 1), 3), container_triangles])
cells_sphere = np.hstack([np.full((sphere_triangles.shape[0], 1), 3), sphere_triangles])

all_vertices = collision_mesh.vertices["position"].compute().value.get().reshape(-1, 3)

bunny_vertices_computed = all_vertices[0 : (NUM_BUNNIES) * NUM_BUNNY_VERTICES]
container_vertices_computed = all_vertices[-container_positions.shape[0]:]
cells_sphere_computed = all_vertices[(NUM_BUNNIES) * NUM_BUNNY_VERTICES : -container_positions.shape[0]]

bunny_poly = pv.PolyData(bunny_vertices_computed, cells_soft)
container_poly = pv.PolyData(container_vertices_computed, cells_container)
sphere_poly = pv.PolyData(cells_sphere_computed, cells_sphere)

plotter.add_mesh(bunny_poly, color = "lightgreen")
plotter.add_mesh(container_poly, color = "pink", opacity = 0.2)
plotter.add_mesh(sphere_poly, color = "lightblue", opacity = 0.1)

plotter.camera_position = [
  (0, 0.15, 0.8),   # camera position
  (0.0, 0.15, 0.0),    # look-at point
  (0.0, 1.0, 0.0),      # up direction
]
plotter.camera.clipping_range = (0.001, 10.0)
# plotter.show()
if SHOW_EXAMPLE:
  plotter.show(interactive_update=True, auto_close=False)


position_copy = collision_mesh.vertices["position"].compute().value.copy()
bunny_soft_position_copy = vertices_bunny_position.value.copy()
affine_matrices_copy = affine_matrices.value.copy()
translations_copy = translations.value.copy()
# container_positions_copy = vertices_container_positions.value.copy()
container_y_copy = container_y_translation.value.copy()
# sphere_position_copy = vertices_sphere_position.compute().value.copy()


def update_collision_set(candidate_position):
  ccd.cd(candidate_position, DHAT_VALUE)
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


start = time.time()
for i in range(int(os.environ.get("YASPS_EXAMPLE_FRAMES", "2000"))):
  start_data_transfer = time.time()
  bunnies_soft.vertices["last_position"].updateValue(bunnies_soft.vertices["position"].value, deepCopy = True)
  spheres.vertices["last_position"].updateValue(spheres.vertices["position"].compute().value, deepCopy = True)
  container.vertices["last_position"].updateValue(container.vertices["position"].compute().value, deepCopy = True)
  end_data_transfer = time.time()
  if i >= START_FRAME:
    t = (i - START_FRAME) * DT_VALUE
    y_offset = A * (1.0 - np.cos(OMEGA * t)) * random.uniform(-0.5, 2.0)
    # remainder = (i - 10) % 40
    # if remainder < 15:
    #   # moving up
    #   vertices_container_target_positions.updateValue(container_positions + np.array([0.0, MOVEMENT_PER_FRAME * (remainder + 1), 0.0]))
    # elif remainder >= 20 and remainder < 35:
    #   subtracted = 34 - remainder
    #   vertices_container_target_positions.updateValue(container_positions + np.array([0.0, MOVEMENT_PER_FRAME * subtracted, 0.0]))
    vertices_container_target_positions.updateValue(
      container_positions + np.array([0.0, y_offset, 0.0])
    )
  print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
  inner_iteration = 0
  # update the friction set
  print("Updating the friction set, with total count:", ccd.separated_counts)
  pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
  start_update_collision = time.time()
  pp_friction_pairs.updateNumInstances(pp_count)
  pe_friction_pairs.updateNumInstances(pe_count)
  pt_friction_pairs.updateNumInstances(pt_count)
  ee_friction_pairs.updateNumInstances(ee_count)
  if pp_count > 0:
    pp_friction2v.updateConnectivity(ccd.pp[:2 * pp_count])
  if pe_count > 0:
    pe_friction2v.updateConnectivity(ccd.pe[:3 * pe_count])
  if pt_count > 0:
    pt_friction2v.updateConnectivity(ccd.pt[:4 * pt_count])
  if ee_count > 0:
    ee_friction2v.updateConnectivity(ccd.ee[:4 * ee_count])

  while True:
    print("At time step", i, "inner iteration", inner_iteration)

    print("==================================================================")
    start_solver = time.time()
    result = s0.minimizeEnergy(tolerance = 1e-5)
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
    d_pos_soft_bunny = result[0] * 0.9
    d_affine_matrices = result[1] * 0.9
    d_translations = result[2] * 0.9
    d_container_y = result[3] * 0.9
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    bunny_soft_position_copy.set(vertices_bunny_position.value)
    affine_matrices_copy.set(affine_matrices.value)
    translations_copy.set(translations.value)
    # container_positions_copy.set(vertices_container_positions.value)
    container_y_copy.set(container_y_translation.value)

    # now update the value
    vertices_bunny_position.updateValue(vertices_bunny_position.value - d_pos_soft_bunny, deepCopy = True)
    affine_matrices.updateValue(affine_matrices.value - d_affine_matrices, deepCopy = True)
    translations.updateValue(translations.value - d_translations, deepCopy = True)
    # vertices_container_positions.updateValue(vertices_container_positions.value - d_pos_container, deepCopy = True)
    container_y_translation.updateValue(container_y_translation.value - d_container_y, deepCopy = True)

    # compute the new positions for the entire scene
    new_positions = collision_mesh.vertices["position"].compute().value
    # now we compute the new direction, remember it's the negative we need to put in
    end_data_transfer = time.time()
    print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
    start_direction_compute = time.time()
    direction_copy = position_copy - new_positions
    end_direction_compute = time.time()
    print(f"Time taken for direction compute: {end_direction_compute - start_direction_compute} seconds")
    start_max_movement = time.time()
    max_movement = gpuarray.max(abs(direction_copy)).get() / DT_VALUE
    print("max movement we want to take is", max_movement)
    end_max_movement = time.time()
    print(f"Time taken for max movement: {end_max_movement - start_max_movement} seconds")
    # if inner_iteration == 0:
    #   # save the direction
    #   direction = direction_copy.get().reshape(-1, 3)
    #   np.save(f"positions/direction_{i:04d}.npy", direction)
    # check for the largest step size we can take
    start_ccd = time.time()
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    end_ccd = time.time()
    print(f"Time taken for CCD: {end_ccd - start_ccd} seconds")
    start_largest_step = time.time()
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    end_largest_step = time.time()
    print("largest step we can take is", largest_step)
    print(f"Time taken for largest step: {end_largest_step - start_largest_step} seconds")
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    line_search_accepted = False
    while substep <= 8:
      start_data_transfer = time.time()
      vertices_bunny_position.updateValue(bunny_soft_position_copy - d_pos_soft_bunny * step_taken, deepCopy = True)
      affine_matrices.updateValue(affine_matrices_copy - d_affine_matrices * step_taken, deepCopy = True)
      translations.updateValue(translations_copy - d_translations * step_taken, deepCopy = True)
      container_y_translation.updateValue(container_y_copy - d_container_y * step_taken, deepCopy = True)
      # vertices_container_positions.updateValue(container_positions_copy - d_pos_container * step_taken, deepCopy = True)
      end_data_transfer = time.time()
      print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
      start_cd = time.time()

      # perform collision detection
      update_collision_set(collision_mesh.vertices["position"].compute().value)
      end_cd = time.time()
      print(f"Time taken for collision detection: {end_cd - start_cd} seconds")
      start_update_collision = time.time()
      end_update_collision = time.time()
      print(f"Time taken for updating connectivity: {end_update_collision - start_update_collision} seconds")
      start_compute = time.time()
      new_energies = s0.computeTotalEnergy()
      end_compute = time.time()
      print(f"Time taken for computation: {end_compute - start_compute} seconds")

      print(f"energy comparison: {new_energies} vs {energies_before}")
      if new_energies <= energies_before:
        line_search_accepted = True
        break
      step_taken = step_taken / 2.0
      substep += 1
    if not line_search_accepted:
      vertices_bunny_position.updateValue(
        bunny_soft_position_copy, deepCopy=True
      )
      affine_matrices.updateValue(affine_matrices_copy, deepCopy=True)
      translations.updateValue(translations_copy, deepCopy=True)
      container_y_translation.updateValue(container_y_copy, deepCopy=True)
      update_collision_set(position_copy)
      step_taken = 0.0
    print("step taken is", step_taken)
    print("substep is", substep)

    bunny_vertices_computed = vertices_bunny_position.value.get().reshape((-1, 3))
    bunny_poly.points = bunny_vertices_computed
    sphere_vertices_computed = vertices_sphere_position.compute().value.get().reshape((-1, 3))
    sphere_poly.points = sphere_vertices_computed
    container_vertices_computed = vertices_container_positions.compute().value.get().reshape((-1, 3))
    container_poly.points = container_vertices_computed


    if SHOW_EXAMPLE:
      plotter.render()
      plotter.update()


    if not line_search_accepted:
      print(
        f"Iteration {inner_iteration} exited because the line search "
        "found no descending float32 state"
      )
      break
    if max_movement < 2e-2 or inner_iteration >= 100:
      print(f"Iteration {inner_iteration} exited with max movement: {max_movement}")
      break
    inner_iteration += 1
    validation_limit = int(os.environ.get("YASPS_EXAMPLE_INNER_ITERATIONS", "0"))
    if validation_limit > 0 and inner_iteration >= validation_limit:
      print(f"Stopped after {validation_limit} bounded validation iterations")
      break
  start_velocity_update = time.time()
  new_velocities_bunny = (vertices_bunny_position.value - vertices_bunny_last_position.value) / DT_VALUE
  vertices_bunny_velocity.updateValue(new_velocities_bunny, deepCopy = True)
  new_velocities_sphere = (vertices_sphere_position.compute().value - vertices_sphere_last_position.value) / DT_VALUE
  vertices_sphere_velocity.updateValue(new_velocities_sphere, deepCopy = True)
  new_velocities_container = (vertices_container_positions.compute().value - vertices_container_last_positions.value) / DT_VALUE
  vertices_container_velocity.updateValue(new_velocities_container, deepCopy = True)
  end_velocity_update = time.time()
  print(f"Time taken for velocity update: {end_velocity_update - start_velocity_update} seconds")
  if SAVE_EXAMPLE:
    plotter.screenshot(f"outputs/bunny_drop_in_container_{i:04d}.jpg")
    bunny_poly.save(f"meshes/bunny_drop_in_container_{i:04d}.obj")
    container_poly.save(f"meshes/container_{i:04d}.obj")
    sphere_poly.save(f"meshes/spheres_{i:04d}.obj")
  # # save all the positions
  # all_positions = collision_mesh.vertices["position"].compute().value.get().reshape(-1, 3)
  # np.save(f"positions/positions_{i:04d}.npy", all_positions)
end = time.time()
print("Total time: ", end - start)

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
np.random.seed(13)      # for numpy
import time

import argparse

parser = argparse.ArgumentParser(description="Bunny simulation")
parser.add_argument(
    "--num-bunnies",
    type=int,
    default=1,
    help="Number of bunnies (default: 25)"
)

args = parser.parse_args()

NUM_BUNNIES = args.num_bunnies

DT_VALUE = 0.01 # for time step
DHAT_VALUE = 1e-5 # for collision detection
KAPPA_VALUE = 10000.0 # for collision


POISSON_VALUE = 0.3645697005781997
YOUNG_VALUE = 10259.25455816859
MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))

# exit()

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
position = np.array(position, dtype = np.float64) / 5.0
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
position[:, 1] += 1.0
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


##################################################################
## construct the copies of the bunnies
##################################################################
positions_list = []
tet_indices_list = []
surface_triangles_indices_list = []
edges_list = []
surface_indices_list = []
for i in range(NUM_BUNNIES):
  positions_list.append(position + np.array([0.1 * (-1 if i % 2 == 0 else 1), 0.5 * i, 0.3 * (-1 if i % 2 == 0 else 1)]))
  tet_indices_list.append(tet_indices + i * NUM_BUNNY_VERTICES)
  surface_triangles_indices_list.append(surface_triangle_indices_bunny + i * NUM_BUNNY_VERTICES)
  edges_list.append(edge_indices_bunny + i * NUM_BUNNY_VERTICES)
  surface_indices_list.append(surface_indices_bunny + i * NUM_BUNNY_VERTICES)
bunny_positions = np.array(positions_list, dtype = np.float64).flatten().reshape(-1, 3)
bunny_tet_indices = np.array(tet_indices_list, dtype = np.uint32).flatten().reshape(-1, 4)
surface_triangles_indices = np.array(surface_triangles_indices_list, dtype = np.uint32).flatten().reshape(-1, 3)
edges = np.array(edges_list, dtype = np.uint32).flatten().reshape(-1, 2)
surface_indices = np.array(surface_indices_list, dtype = np.uint32).flatten()

##################################################################
## Load the container
##################################################################
f = open("../data/container.obj", 'r')
container_positions = []
container_surface_triangles = []
for line in f:
  if line.startswith('v '):
    container_positions.append([float(x) for x in line.split()[1:]])
  elif line.startswith('f '):
    container_surface_triangles.append([int(x) - 1 for x in line.split()[1:]])

container_positions = np.array(container_positions, dtype = np.float64).reshape(-1, 3)
container_surface_triangles = np.array(container_surface_triangles, dtype = np.uint32).reshape(-1, 3)
container_edge_indices = extract_edges_from_triangles(container_surface_triangles).astype(np.uint32)

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

##################################################################
## construct the soft meshes
##################################################################
bunnies_soft = s0.addMesh("bunnies_soft")
mu = bunnies_soft.addConstant("mu", rows = 1, cols = 1)
mu.updateValue([MU_LAME_VALUE])
lam = bunnies_soft.addConstant("lambda", rows = 1, cols = 1)
lam.updateValue([LAMBDA_LAME_VALUE])

# now we construct the soft vertices
vertices_soft = bunnies_soft.addPrimitive("vertices_soft", numInstances = NUM_BUNNIES * NUM_BUNNY_VERTICES)
vertices_soft_position = vertices_soft.addAttribute("position", rows = 3, cols = 1)
vertices_soft_rest_position = vertices_soft.addConstant("rest_position", rows = 3, cols = 1)
vertices_soft_last_position = vertices_soft.addConstant("last_position", rows = 3, cols = 1)
vertices_soft_velocity = vertices_soft.addConstant("velocity", rows = 3, cols = 1)
vertices_soft_mass = vertices_soft.addConstant("mass", rows = 1, cols = 1)

# update the values
vertices_soft_position.updateValue(np.array(bunny_positions, dtype = np.float64).flatten())
vertices_soft_rest_position.updateValue(np.array(bunny_positions, dtype = np.float64).flatten())
vertices_soft_last_position.updateValue(np.array(bunny_positions, dtype = np.float64).flatten())
vertices_soft_velocity.updateValue(np.zeros(NUM_BUNNIES * NUM_BUNNY_VERTICES * 3, dtype = np.float64))
vertices_soft_mass.updateValue(np.array([[2.0 / NUM_BUNNY_VERTICES] * NUM_BUNNY_VERTICES for _ in range(NUM_BUNNIES)], dtype = np.float64).flatten())


# now that we are done with vertices, we can do the tets
# each tet will have its corresponding mu and lambda values
# and all other attributes will be fetched from the vertices
tets_softs = bunnies_soft.addPrimitive("tets_soft", numInstances = NUM_BUNNIES * NUM_BUNNY_TETS)
# add connectivity from tets to vertices
tets_softs2_vertices = tets_softs.addConnectivity("tets_softs2_vertices", vertices_soft, np.array(bunny_tet_indices, dtype = np.uint32), 4)
# now we can get the rest positions and the current positions from the vertices
tets_softs_positions = tets_softs.addAttribute("positions", through = tets_softs2_vertices, source = vertices_soft_position)
tets_softs_rest_positions = tets_softs.addAttribute("rest_positions", through = tets_softs2_vertices, source = vertices_soft_rest_position)


row0 = tets_softs_rest_positions .row(0)
row1 = tets_softs_rest_positions .row(1)
row2 = tets_softs_rest_positions .row(2)
row3 = tets_softs_rest_positions .row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
TB = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
tets_soft_tb = tets_softs.addAttribute("TB", computed_attribute = TB)

row0 = tets_softs_positions.row(0)
row1 = tets_softs_positions.row(1)
row2 = tets_softs_positions.row(2)
row3 = tets_softs_positions.row(3)
x0 = row1 - row0
x1 = row2 - row0
x2 = row3 - row0
F = attribute.to_array([x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]], rows = 3, cols = 3)
tets_soft_f = tets_softs.addAttribute("F", computed_attribute = F)

bdg = bunnies_soft.addPrimitive("deformation_gradient", numInstances = NUM_BUNNIES * NUM_BUNNY_TETS)
bdg2bt = bdg.addConnectivity("bdg2bt", tets_softs, np.arange(NUM_BUNNIES * NUM_BUNNY_TETS), 1)
bdg_F = bdg.addAttribute("F", through = bdg2bt, source = tets_soft_f)
bdg_F = bdg_F.resize(3, 3)

bdg_TB = bdg.addAttribute("TB", through = bdg2bt, source = tets_soft_tb)
bdg_TB = bdg_TB.resize(3, 3)

##################################################################
## construct the container
##################################################################
container = s0.addMesh("container")
vertices_container = container.addPrimitive("vertices", numInstances = container_positions.shape[0])
vertices_container_positions = vertices_container.addAttribute("position", rows = 3, cols = 1)
vertices_container_positions.updateValue(container_positions)

# add triangles
triangles_container = container.addPrimitive("triangles", numInstances = container_surface_triangles.shape[0])
triangles_container2_vertices = triangles_container.addConnectivity("triangles_container2_vertices", vertices_container, np.array(container_surface_triangles, dtype = np.uint32), 3)
triangles_container_positions = triangles_container.addAttribute("positions", through = triangles_container2_vertices, source = vertices_container_positions)

##################################################################
## we have finished the construction of our meshes
## now we need to add a collision mesh
##################################################################
collision_mesh = s0.addMesh("collision_mesh")
collision_vertices = collision_mesh.addPrimitiveUnion("vertices", [vertices_soft, vertices_container])
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
# from helpers import stable_neo_hookean
# snh_softs = stable_neo_hookean(tets_softs_rest_positions, tets_softs_positions, bunnies_soft["mu"], bunnies_soft["lambda"], dt)
# tets_softs.addAttribute("snh_softs", computed_attribute = snh_softs)

from helpers import stable_neo_hookean_modified
snh_softs = stable_neo_hookean_modified(bdg_F, bdg_TB, bunnies_soft["mu"], bunnies_soft["lambda"], dt)
tets_softs.addAttribute("snh_softs", computed_attribute = snh_softs)


from helpers import inertia
inertia_softs = inertia(vertices_soft_last_position, vertices_soft_velocity, dt, vertices_soft_position, vertices_soft_mass)
vertices_soft.addAttribute("inertia_softs", computed_attribute = inertia_softs)


from helpers import point_point, point_edge, point_triangle, edge_edge, affine_energy
pp = point_point(pp_positions, dhat, kappa)
collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

s0.addEnergy(snh_softs, projection_method = 1)
s0.addEnergy(inertia_softs, projection_method = 0)
s0.addEnergy(pp, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2)
s0.addMinimizeTarget([vertices_soft_position])

##################################################################
## add ccd
##################################################################
ccd = CCD(NUM_BUNNY_SURFACE_INDICES * (NUM_BUNNIES) + container_positions.shape[0], # the number of surface points
  NUM_BUNNY_VERTICES * (NUM_BUNNIES) + container_positions.shape[0], # the number of total points
  max_ccd_pairs = 100000000,
)

triangle_indices_all = surface_triangles_indices_list
surface_indices_all = surface_indices_list

triangle_indices_all.append((container_surface_triangles + (NUM_BUNNIES) * NUM_BUNNY_VERTICES).astype(np.uint32))
surface_indices_all.append((np.array(range(container_positions.shape[0])) + (NUM_BUNNIES) * NUM_BUNNY_VERTICES).astype(np.uint32))


triangle_indices_all = np.vstack(triangle_indices_all, dtype = np.uint32)
surface_indices_all = np.hstack(surface_indices_all).astype(np.uint32)

edge_indices_all = edges_list
edge_indices_all.append((container_edge_indices + (NUM_BUNNIES) * NUM_BUNNY_VERTICES).astype(np.uint32))
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
soft_triangles = triangles[0 :(NUM_BUNNIES) * NUM_BUNNY_SURFACE_TRIANGLES]
container_triangles = triangles[(NUM_BUNNIES) * NUM_BUNNY_SURFACE_TRIANGLES:] - (NUM_BUNNIES) * NUM_BUNNY_VERTICES


cells_soft = np.hstack([np.full((soft_triangles.shape[0], 1), 3), soft_triangles])
cells_container = np.hstack([np.full((container_triangles.shape[0], 1), 3), container_triangles])

soft_vertices_computed = all_vertices_computed[0 : (NUM_BUNNIES) * NUM_BUNNY_VERTICES]
container_vertices_computed = all_vertices_computed[(NUM_BUNNIES) * NUM_BUNNY_VERTICES:]

soft_poly = pv.PolyData(soft_vertices_computed, cells_soft)
container_poly = pv.PolyData(container_vertices_computed, cells_container)

plotter.add_mesh(soft_poly, color = "lightgreen")
plotter.add_mesh(container_poly, color = "pink", opacity = 0.5)

plotter.camera_position = [(0, 3, 15),
 (0.0, 3.0, 0.0),
 (0, 1, 0)
]
# plotter.show()
plotter.show(interactive_update=True, auto_close=False)
# exit()
position_copy = collision_mesh.vertices["position"].compute().value.copy()
bunny_soft_position_copy = vertices_soft_position.compute().value.copy()



start = time.time()
for i in range(400):
  start_data_transfer = time.time()
  bunnies_soft.vertices_soft["last_position"].updateValue(bunnies_soft.vertices_soft["position"].value, deepCopy = True)
  end_data_transfer = time.time()
  print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
  inner_iteration = 0
  while True:
    print("==================================================================")
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
    d_pos_soft_bunny = result[0]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    bunny_soft_position_copy.set(vertices_soft_position.compute().value)

    # now update the value
    vertices_soft_position.updateValue(vertices_soft_position.value - d_pos_soft_bunny, deepCopy = True)

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
    end_max_movement = time.time()
    print(f"Time taken for max movement: {end_max_movement - start_max_movement} seconds")
    if max_movement < 1e-2:
      print(f"Iteration {inner_iteration} exited with max movement: {max_movement}")
      break
    # check for the largest step size we can take
    start_ccd = time.time()
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 0.5)
    end_ccd = time.time()
    print(f"Time taken for CCD: {end_ccd - start_ccd} seconds")
    start_largest_step = time.time()
    largest_step = ccd.compute_largest_step_size(0.5, position_copy, direction_copy)
    end_largest_step = time.time()
    print("largest step we can take is", largest_step)
    print(f"Time taken for largest step: {end_largest_step - start_largest_step} seconds")
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    while substep <= 8:
      start_data_transfer = time.time()
      vertices_soft_position.updateValue(bunny_soft_position_copy - d_pos_soft_bunny * step_taken, deepCopy = True)
      end_data_transfer = time.time()
      print(f"Time taken for data transfer: {end_data_transfer - start_data_transfer} seconds")
      start_cd = time.time()

  #     # perform collision detection
      ccd.cd(collision_mesh.vertices["position"].compute().value, DHAT_VALUE) # perform collision detection
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
  #   if substep > 8:
  #     print("failed")
  #     exit(1)
    print("step taken is", step_taken)
    print("substep is", substep)

    soft_vertices_computed = vertices_soft_position.compute().value.get().reshape((-1, 3))
    soft_poly.points = soft_vertices_computed
    plotter.render()
    plotter.update()

    # print(f"Iteration {inner_iteration} max gradient: {max_grad}")
    # if max_grad < 1e-4:
    #   print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
    #   break
    # if max(abs(direction_copy).get() / DT_VALUE) < 1e-2:
    #   print(f"Iteration {inner_iteration} exited with max movement: {max_grad}")
    #   break
    inner_iteration += 1
  start_velocity_update = time.time()
  new_velocities_soft = (vertices_soft_position.value - vertices_soft_last_position.value) / DT_VALUE
  vertices_soft_velocity.updateValue(new_velocities_soft, deepCopy = True)
  end_velocity_update = time.time()
  print(f"Time taken for velocity update: {end_velocity_update - start_velocity_update} seconds")

  # all_vertices_computed = collision_mesh.vertices["position"].compute().value.get().reshape((-1, 3))
  # soft_vertices_computed = all_vertices_computed[0:(NUM_BUNNIES) * NUM_BUNNY_VERTICES]
  # container_vertices_computed = all_vertices_computed[(NUM_BUNNIES) * NUM_BUNNY_VERTICES:]
  # soft_poly.points = soft_vertices_computed
  # container_poly.points = container_vertices_computed
  # plotter.render()
  # plotter.update()
  # plotter.screenshot(f"outputs/bunny_drop_in_container_{i:04d}.jpg")
  # save the mesh obj file
  # abd_poly.save(f"meshes/bunny_abd_{i:04d}.obj")
  # soft_poly.save(f"meshes/bunny_drop_in_container_{i:04d}.obj")
  # container_poly.save(f"meshes/cloth_3_cloth_{i:04d}.obj")
  # # save the mesh obj file
  # bunny_poly0.save(f"outputs/bunny_abd_soft0_{i:04d}.obj")
  # bunny_poly1.save(f"outputs/bunny_abd_soft1_{i:04d}.obj")
end = time.time()
print("Total time: ", end - start)

from helpers import extract_boundary_edges, point_point, point_edge, point_triangle, edge_edge, abs_max_reduce, repulsive_loop, smooth_loop
import numpy as np
from yasps import scene
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
import random

DT_VALUE = 0.01
KAPPA_VALUE = 1000000000000.0 # for collision
DHAT_VALUE = 1e-4 # for collision detection
NUM_LOOP_POINTS = 6000
ALPHA_VALUE = 3.0
BETA_VALUE = 6.0
SMOOTH_PENALTY = 1.0
LENGTH_PENALTY = 1.0
REPULSIVE_PENALTY = 1.0

######################################################
# Read the bunny mesh
######################################################
v_bunny = []
f_bunny = []
with open("../data/bunny_uv.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("vt "):
      parts = line.split()
      v_bunny.append([float(parts[1]), float(parts[2]), 0.0])
    elif line.startswith("f "):
      parts = line.split()
      f_bunny.append([int(parts[1].split("/")[1]) - 1, int(parts[2].split("/")[1]) - 1, int(parts[3].split("/")[1]) - 1])

v_bunny = np.array(v_bunny, dtype=np.float64)
f_bunny = np.array(f_bunny, dtype=np.uint32)
# get the bounding box of the uv coordinates
min_x = np.min(v_bunny[:, 0])
max_x = np.max(v_bunny[:, 0])
min_y = np.min(v_bunny[:, 1])
max_y = np.max(v_bunny[:, 1])

# center the bunny
v_bunny[:, 0] = v_bunny[:, 0] - (min_x + max_x) / 2.0
v_bunny[:, 1] = v_bunny[:, 1] - (min_y + max_y) / 2.0

# now scale it so its in the -1 to 1 range
scale_x = 1.0 / ((max_x - min_x) / 2.0)
scale_y = 1.0 / ((max_y - min_y) / 2.0)

v_bunny[:, 0] = v_bunny[:, 0] * scale_x
v_bunny[:, 1] = v_bunny[:, 1] * scale_y

min_x = np.min(v_bunny[:, 0])
max_x = np.max(v_bunny[:, 0])
min_y = np.min(v_bunny[:, 1])
max_y = np.max(v_bunny[:, 1])
print("Bunny UV bounding box: ", min_x, max_x, min_y, max_y)

# we also get the boundary edges of the bunny
boundary_bunny = extract_boundary_edges(f_bunny)


######################################################
# create the loops
######################################################
# we create the loop on a unit circle
v_loop = []
e_loop = []
e_loop_expanded = []
for i in range(NUM_LOOP_POINTS):
  angle = 2.0 * np.pi * float(i) / float(NUM_LOOP_POINTS)
  rand_offset = random.random() * 0.000
  v_loop.append([0.5 * np.cos(angle) * (1 + rand_offset), 0.5 * np.sin(angle) * (1 + rand_offset), 0.0])
  e_loop.append([i, (i + 1) % NUM_LOOP_POINTS])
  e_loop_expanded.append([i, (i + 1) % NUM_LOOP_POINTS, (i + 2) % NUM_LOOP_POINTS, (i + 3) % NUM_LOOP_POINTS])

v_loop = np.array(v_loop, dtype=np.float64) * 1.5
rolled_prev = np.roll(v_loop, shift=1, axis=0)
rolled_prev_prev = np.roll(v_loop, shift=2, axis=0)
rolled_next = np.roll(v_loop, shift=-1, axis=0)
rolled_next_next = np.roll(v_loop, shift=-2, axis=0)
v_loop = 0.05 * rolled_prev_prev + 0.05 * rolled_prev + 0.8 * v_loop + 0.05 * rolled_next + 0.05 * rolled_next_next

e_loop = np.array(e_loop, dtype=np.uint32)
e_loop_expanded = np.array(e_loop_expanded, dtype=np.uint32)


######################################################
# start creating meshes
######################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA_VALUE])
alpha = s0.addConstant("alpha", rows = 1, cols = 1)
alpha.updateValue([ALPHA_VALUE])
beta = s0.addConstant("beta", rows = 1, cols = 1)
beta.updateValue([BETA_VALUE])
smooth_penalty = s0.addConstant("smooth_penalty", rows = 1, cols = 1)
smooth_penalty.updateValue([SMOOTH_PENALTY])
length_penalty = s0.addConstant("length_penalty", rows = 1, cols = 1)
length_penalty.updateValue([LENGTH_PENALTY])
repulsive_penalty = s0.addConstant("repulsive_penalty", rows = 1, cols = 1)
repulsive_penalty.updateValue([REPULSIVE_PENALTY])


bunny = s0.addMesh("bunny")
loop = s0.addMesh("loop")
bunny.addPrimitive("vertices", numInstances = v_bunny.shape[0])
bunny.vertices.addAttribute("position", rows = 3, cols = 1)
bunny.vertices["position"].updateValue(v_bunny)


loop.addPrimitive("vertices", numInstances = v_loop.shape[0])
loop.vertices.addAttribute("position", rows = 3, cols = 1)
loop.vertices["position"].updateValue(v_loop)

# we will create the relationship between every two pairs of edges
# to the loop vertices
edge_pair_indices = []
for i in range(v_loop.shape[0]):
  e0 = i
  e1 = (i + 1) % v_loop.shape[0]
  for j in range(i + 2, v_loop.shape[0]):
    e2 = j % v_loop.shape[0]
    e3 = (j + 1) % v_loop.shape[0]
    if e3 != e0 and e3 != e1:
      edge_pair_indices.append([e0, e1, e2, e3])
edge_pair_indices = np.array(edge_pair_indices, dtype=np.uint32)


loop.addPrimitive("edge_pairs", numInstances = edge_pair_indices.shape[0])
l2v = loop.edge_pairs.addConnectivity("l2v", loop.vertices, edge_pair_indices, 4) # each edge pair has 4 vertices
loop.edge_pairs.addAttribute("positions", through = l2v, source = loop.vertices["position"])

# we will also create the relationship between each pair of edges
# to the loop vertices
loop.addPrimitive("edges", numInstances = e_loop.shape[0])
e2v = loop.edges.addConnectivity("e2v", loop.vertices, e_loop, 2)
loop.edges.addAttribute("positions", through = e2v, source = loop.vertices["position"])

# and also the expanded edges
loop.addPrimitive("expanded_edges", numInstances = e_loop_expanded.shape[0])
ee2v_loop = loop.expanded_edges.addConnectivity("ee2v_loop", loop.vertices, e_loop_expanded, 4)
loop.expanded_edges.addAttribute("positions", through = ee2v_loop, source = loop.vertices["position"])

######################################################
# we now create a union of vertices, this will be used
# for collision
######################################################
collision_mesh = s0.addMesh("collision_mesh")
collision_mesh.addPrimitiveUnion("vertices", [bunny.vertices, loop.vertices])
collision_mesh.vertices.addAttribute("position")

# we now add not only the loop edge pairs
# but also the edge pairs formed with the boundary edges
# of the bunny
edge_pairs_global_indices = []
for edge in boundary_bunny:
  b0 = edge[0]
  b1 = edge[1]
  for i in range(v_loop.shape[0]):
    e0 = i
    e1 = (i + 1) % v_loop.shape[0]
    edge_pairs_global_indices.append([b0, b1, e0 + v_bunny.shape[0], e1 + v_bunny.shape[0]])
# concatenate with the previous edge pairs
# edge_pairs_global_indices = np.vstack([edge_pair_indices + v_bunny.shape[0], np.array(edge_pairs_global_indices, dtype=np.uint32)])
edge_pairs_global_indices = np.array(edge_pair_indices + v_bunny.shape[0], dtype = np.uint32)
collision_mesh.addPrimitive("edge_pairs", numInstances = edge_pairs_global_indices.shape[0])
l2v_g = collision_mesh.edge_pairs.addConnectivity("l2v_g", collision_mesh.vertices, edge_pairs_global_indices, 4)
edge_pair_positions = collision_mesh.edge_pairs.addAttribute("positions", through = l2v_g, source = collision_mesh.vertices["position"])



collision_mesh.addPrimitive("pp", numInstances = 0, isDynamic = True)
collision_mesh.addPrimitive("pe", numInstances = 0, isDynamic = True)
collision_mesh.addPrimitive("pt", numInstances = 0, isDynamic = True)
collision_mesh.addPrimitive("ee", numInstances = 0, isDynamic = True)
pp2v = collision_mesh.pp.addConnectivity("pp2v", collision_mesh.vertices, [], 2)
pe2v = collision_mesh.pe.addConnectivity("pe2v", collision_mesh.vertices, [], 3)
pt2v = collision_mesh.pt.addConnectivity("pt2v", collision_mesh.vertices, [], 4)
ee2v = collision_mesh.ee.addConnectivity("ee2v", collision_mesh.vertices, [], 4)
pp_positions = collision_mesh.pp.addAttribute("positions", through = pp2v, source = collision_mesh.vertices["position"])
pe_positions = collision_mesh.pe.addAttribute("positions", through = pe2v, source = collision_mesh.vertices["position"])
pt_positions = collision_mesh.pt.addAttribute("positions", through = pt2v, source = collision_mesh.vertices["position"])
ee_positions = collision_mesh.ee.addAttribute("positions", through = ee2v, source = collision_mesh.vertices["position"])
######################################################
# now we begin adding our energies
######################################################
# the first is the repulsive energy between each pair of edges
repulsive_loop_energy = repulsive_penalty * repulsive_loop(collision_mesh.edge_pairs["positions"], alpha, beta)
collision_mesh.addAttribute("repulsive_loop_energy", computed_attribute = repulsive_loop_energy)

# the second is to expand the loop by adding the inverse of the length
edge_positions = loop.edges["positions"]
v0 = edge_positions.row(0)
v1 = edge_positions.row(1)
edge_length = (v1 - v0).dot(v1 - v0)
# inverse_edge_length = -length_penalty * edge_length
inverse_edge_length = length_penalty / edge_length
inverse_length_energy = loop.addAttribute("inverse_edge_length", computed_attribute = inverse_edge_length)

# the third one is to make the loop smooth
# which requires the 4 countinous points on the loop
edge_positions_expanded = loop.expanded_edges["positions"]
smooth_energy = smooth_penalty * smooth_loop(edge_positions_expanded)
loop.expanded_edges.addAttribute("smooth_energy", computed_attribute = smooth_energy)

# and the rest are the collision energies
pp = point_point(pp_positions, dhat, kappa)
pp_energy = collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

s0.addEnergy(repulsive_loop_energy, projection_method = 1)
s0.addEnergy(inverse_length_energy, projection_method = 1)
s0.addEnergy(smooth_energy, projection_method = 1)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 1)
s0.addMinimizeTarget([loop.vertices["position"]])
# print(sum(repulsive_loop_energy.compute().value.get()))
# print(sum(inverse_length_energy.compute().value.get()))
# print(sum(smooth_energy.compute().value.get()))
# exit()

######################################################
# we now initialize the ccd
######################################################
# here in this case there is no triangle collisions
# only edges
ccd = CCD(v_bunny.shape[0] + v_loop.shape[0], v_bunny.shape[0] + v_loop.shape[0])
triangle_indices_gpu = gpuarray.to_gpu(np.array([], dtype=np.uint32)) # lets make a temp array to avoid null pointer
surface_indices_gpu = gpuarray.to_gpu(np.array(list(range(v_bunny.shape[0], v_bunny.shape[0] + v_loop.shape[0])) + list(set(boundary_bunny.flatten().tolist())), dtype=np.uint32)) # lets make a temp array to avoid null pointer
edge_indices_gpu = gpuarray.to_gpu((np.vstack([boundary_bunny, e_loop + v_bunny.shape[0]])).astype(np.uint32))
ccd.init_faces(collision_mesh.vertices["position"].compute().value, triangle_indices_gpu, surface_indices_gpu, 0)
ccd.init_edges(collision_mesh.vertices["position"].compute().value, collision_mesh.vertices["position"].compute().value, edge_indices_gpu, e_loop.shape[0] + boundary_bunny.shape[0])


######################################################
# plotting
######################################################
import pyvista as pv
# first we add bunny

plotter = pv.Plotter(window_size = [3840, 2160])
cells_bunny = np.hstack([np.full((f_bunny.shape[0], 1), 3), f_bunny])
bunny_poly = pv.PolyData(v_bunny, cells_bunny)
plotter.add_mesh(bunny_poly, show_edges=True, color='cyan', opacity = 0.6)

cells_loop = np.hstack([np.full((e_loop.shape[0], 1), 2), e_loop])
loop_poly = pv.PolyData(v_loop, lines = cells_loop)
plotter.add_mesh(loop_poly, color='red', line_width=3)

plotter.camera_position = [(0, 0, 5.0),
 (0.0, 0.0, 0.0),
 (0.0, 1.0, 0.0)]
plotter.show(interactive_update=True)

# here we do the solve loops
position_copy = collision_mesh.vertices["position"].compute().value.copy()
direction_copy = gpuarray.zeros((v_bunny.shape[0] + v_loop.shape[0]) * 3, dtype=np.float64)
loop_position_copy = loop.vertices["position"].value.copy()
total_iterations = 0
for i in range(2000):
  inner_iteration = 0
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-3, maxIterations = 100000)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer

    # compute the total energy in the scene
    repulsive_loop_energy_sum = sum(repulsive_loop_energy.compute().value.get())
    inverse_length_energy_sum = sum(inverse_length_energy.compute().value.get())
    smooth_energy_sum = sum(smooth_energy.compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = repulsive_loop_energy_sum + inverse_length_energy_sum + smooth_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    print("-------------------------------------------------------------------------------")
    print(f"energy before {energies_before}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    print("-------------------------------------------------------------------------------")

    d_pos = result[0]
    d_pos_cpu = d_pos.get().flatten()
    d_pos_cpu[2::3] = 0.0
    d_pos.set(d_pos_cpu)
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    loop_vertices_copy = loop.vertices["position"].value.get().flatten()
    loop_vertices_copy[2::3] = 0.0
    loop_position_copy.set(loop_vertices_copy)
    direction_copy[v_bunny.shape[0] * 3:].set(d_pos)
    print("position and direction set")

    # we compute the largest step size we can take
    # ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    ccd.reset()
    ccd.ccd_edges(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step

    # perform line search
    while substep <= 8:
      loop.vertices["position"].updateValue(loop_position_copy - d_pos * step_taken, deepCopy = True)
      # perform collision detection
      ccd.reset()
      ccd.cd_edges(collision_mesh.vertices["position"].compute().value, DHAT_VALUE)
      # ccd.cd(collision_mesh.vertices["position"].compute().value, DHAT_VALUE) # perform collision detection
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

      repulsive_loop_energy_sum = sum(repulsive_loop_energy.compute().value.get())
      inverse_length_energy_sum = sum(inverse_length_energy.compute().value.get())
      smooth_energy_sum = sum(smooth_energy.compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = repulsive_loop_energy_sum + inverse_length_energy_sum + smooth_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
      print("===============================================================================")
      print(f"energy comparison: {new_energies} vs {energies_before}")
      print("===============================================================================")

      if new_energies < energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    # if substep > 8:
    #   print("Line search failed, exiting inner loop")
    #   exit(1)

    print("step taken is", step_taken)
    print("substep is", substep)

    loop_positions = loop.vertices["position"].compute().value.get().reshape(-1, 3)
    # loop_positions_prev_prev = np.roll(loop_positions, shift=-2, axis=0)
    # loop_positions_prev = np.roll(loop_positions, shift=-1, axis=0)
    # loop_positions_next = np.roll(loop_positions, shift=1, axis=0)
    # loop_positions_next_next = np.roll(loop_positions, shift=2, axis=0)
    # loop_positions = 0.15 * loop_positions_prev_prev + 0.15 * loop_positions_prev + 0.4 * loop_positions + 0.15 * loop_positions_next + 0.15 * loop_positions_next_next

    loop_poly.points = loop_positions
    plotter.render()
    plotter.update()
    # plotter.screenshot(f"outputs/loop_{total_iterations:06d}.jpg")
    # save the loop positions as obj
    loop_poly.save(f"outputs/loop_{total_iterations:06d}.obj")

    total_iterations += 1
    if total_iterations == 200:
      exit()

    if max_grad < 1000.0:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1

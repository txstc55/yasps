from yasps import scene
from yasps import attribute
from helpers import extract_surface_triangles, stable_neo_hookean, inertia, extract_edges_from_triangles, abs_max_reduce
from helpers import point_point, point_edge, point_triangle, edge_edge, affine_energy
import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
DT_VALUE = 0.01 # for time step
DHAT_VALUE = 1e-6 # for collision detection
KAPPA_VALUE = 10000000.0 # for collision
POISSON_VALUE = 0.49
YOUNG_VALUE = 10000000.0
MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))
MU_VALUE = 4.0 * MU_LAME_VALUE / 3.0
LAMBDA_VALUE = LAMBDA_LAME_VALUE + 5.0 * MU_LAME_VALUE / 6.0
print("Using mu = ", MU_VALUE, " and lambda = ", LAMBDA_VALUE)
##################################################
## read the bunny file
##################################################
f = open("../data/bunny.ele", 'r')
f.readline()
tet_indices = []
for line in f:
  tet_indices.append([int(x) - 1 for x in line.split()[3:]])
f.close()
tet_indices = np.array(tet_indices)
tet_indices_second = tet_indices.copy()

f = open("../data/bunny.node", 'r')
f.readline()
position = []
for line in f:
  position.append([float(x) for x in line.split()[1:]])
f.close()
position = np.array(position, dtype = np.float64)
position_second = position.copy()

position = position + np.array([-2.0, 2.0, 0.0])
position_second = position_second + np.array([2.0, 2.0, -0.0])
tet_indices_second = tet_indices_second + position.shape[0]

position = np.concatenate((position, position_second), axis = 0)
tet_indices = np.concatenate((tet_indices, tet_indices_second), axis = 0)


surface_triangle_indices = extract_surface_triangles(tet_indices)
edge_indices = extract_edges_from_triangles(surface_triangle_indices)
surface_indices = list(set(surface_triangle_indices.flatten().tolist()))


##################################################
## create the mesh with primitives and attributes
##################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA_VALUE])

# create the first bunny, and bind attributes and constants
bunny0 = s0.addMesh("bunny0")
bunny0.addPrimitive("vertices", numInstances = position.shape[0] // 2)
bunny0.addAttribute("rotation", rows = 3, cols = 3)
bunny0["rotation"].updateValue(np.eye(3, dtype=np.float64))
bunny0.addAttribute("translation", rows = 3, cols = 1)
bunny0["translation"].updateValue(np.array([0.0, 0.0, 0.0], dtype=np.float64))
bunny0.vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny0.vertices["rest_position"].updateValue(position[:position.shape[0] // 2, :])
bunny0.vertices.addConstant("last_position", rows = 3, cols = 1)
bunny0.vertices["last_position"].updateValue(position[:position.shape[0] // 2, :])
bunny0.vertices.addConstant("velocity", rows = 3, cols = 1)
velocities = np.zeros_like(position[:position.shape[0] // 2, :], dtype=np.float64)
velocities[:, 0] = 8.0
bunny0.vertices["velocity"].updateValue(velocities)
bunny0.vertices.addAttribute("position", computed_attribute = bunny0["rotation"] * bunny0.vertices["rest_position"] + bunny0["translation"])
bunny0.vertices.addConstant("mass", rows = 1, cols = 1)
bunny0.vertices["mass"].updateValue(np.ones(position.shape[0] // 2, dtype=np.float64) * 1.0)
mu0 = bunny0.addConstant("mu", rows = 1, cols = 1) # for stable neo hookean
lam0 = bunny0.addConstant("lam", rows = 1, cols = 1) # for stable neo hookean
mu0.updateValue([MU_VALUE])
lam0.updateValue([LAMBDA_VALUE])


# now for the second bunny, we make this bunny soft
bunny1 = s0.addMesh("bunny1")
bunny1.addPrimitive("vertices", numInstances = position.shape[0] // 2)
bunny1.vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny1.vertices["rest_position"].updateValue(position[position.shape[0] // 2:, :])
bunny1.vertices.addConstant("last_position", rows = 3, cols = 1)
bunny1.vertices["last_position"].updateValue(position[position.shape[0] // 2:, :])
bunny1.vertices.addConstant("velocity", rows = 3, cols = 1)
velocities = np.zeros_like(position[position.shape[0] // 2:, :], dtype=np.float64)
velocities[:, 0] = -8.0
bunny1.vertices["velocity"].updateValue(velocities)
bunny1.vertices.addAttribute("position", rows = 3, cols = 1)
bunny1.vertices["position"].updateValue(position[position.shape[0] // 2:, :])
bunny1.vertices.addConstant("mass", rows = 1, cols = 1)
bunny1.vertices["mass"].updateValue(np.ones(position.shape[0] // 2, dtype=np.float64) * 1.0)
mu1 = bunny1.addConstant("mu", rows = 1, cols = 1) # for stable neo hookean
lam1 = bunny1.addConstant("lam", rows = 1, cols = 1) # for stable neo hookean
mu1.updateValue([MU_VALUE])
lam1.updateValue([LAMBDA_VALUE])

# we also add the tets for each bunny
bunny0.addPrimitive("tets", numInstances = tet_indices.shape[0] // 2)
bunny1.addPrimitive("tets", numInstances = tet_indices.shape[0] // 2)
# add the connectivities for the tets
tet2v0 = bunny0.tets.addConnectivity("tet2v", bunny0.vertices, tet_indices[:tet_indices.shape[0] // 2], 4)
tet2v1 = bunny1.tets.addConnectivity("tet2v", bunny1.vertices, tet_indices[:tet_indices.shape[0] // 2], 4)
# add things like rest_position and current position
bunny0.tets.addAttribute("rest_positions", through = tet2v0, source = bunny0.vertices["rest_position"])
bunny0.tets.addAttribute("positions", through = tet2v0, source = bunny0.vertices["position"])
bunny1.tets.addAttribute("rest_positions", through = tet2v1, source = bunny1.vertices["rest_position"])
bunny1.tets.addAttribute("positions", through = tet2v1, source = bunny1.vertices["position"])




# ok now we add a new mesh?
bunnies = s0.addMesh("bunnies")
bunnies.addPrimitiveUnion("vertices", [bunny0.vertices, bunny1.vertices])
bunnies.vertices.addAttribute("rest_position")
bunnies.vertices.addAttribute("position")
bunnies.vertices.addAttribute("last_position")
bunnies.vertices.addAttribute("velocity")
bunnies.vertices.addAttribute("mass")


# print("position check")
# print(bunnies.vertices["position"].compute().value.get())
# now for the new mesh we add tets and triangles
# as well as pp pair, pt, pe, ee
# bunnies.addPrimitive("tets", numInstances = tet_indices.shape[0])
bunnies.addPrimitive("surfaceTriangles", numInstances = surface_triangle_indices.shape[0])
bunnies.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
bunnies.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
bunnies.addPrimitive("pt", numInstances = 0, isDynamic = True) # # for point triangle collision
bunnies.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision



##################################################
## add connectivities, and attributes
##################################################
# tet2v = bunnies.tets.addConnectivity("tet2v", bunnies.vertices, tet_indices, 4)
tri2v = bunnies.surfaceTriangles.addConnectivity("tri2v", bunnies.vertices, surface_triangle_indices, 3)
pp2v = bunnies.pp.addConnectivity("pp2v", bunnies.vertices, [], 2)
pe2v = bunnies.pe.addConnectivity("pe2v", bunnies.vertices, [], 3)
pt2v = bunnies.pt.addConnectivity("pt2v", bunnies.vertices, [], 4)
ee2v = bunnies.ee.addConnectivity("ee2v", bunnies.vertices, [], 4)
# tet_positions = bunnies.tets.addAttribute("positions", through = tet2v, source = bunnies.vertices["position"])
# tet_rest_positions = bunnies.tets.addAttribute("rest_positions", through = tet2v, source = bunnies.vertices["rest_position"])
tri_positions = bunnies.surfaceTriangles.addAttribute("positions", through = tri2v, source = bunnies.vertices["position"])
pp_positions = bunnies.pp.addAttribute("positions", through = pp2v, source = bunnies.vertices["position"])
pe_positions = bunnies.pe.addAttribute("positions", through = pe2v, source = bunnies.vertices["position"])
pt_positions = bunnies.pt.addAttribute("positions", through = pt2v, source = bunnies.vertices["position"])
ee_positions = bunnies.ee.addAttribute("positions", through = ee2v, source = bunnies.vertices["position"])

##################################################
# construct ccd
##################################################
ccd = CCD(len(surface_indices), position.shape[0])
surface_indices_gpu = gpuarray.to_gpu(np.array(surface_indices).astype(np.uint32))
edge_indices_gpu = gpuarray.to_gpu(edge_indices.astype(np.uint32))
ccd.init_faces(bunnies.vertices["position"].compute().value, tri2v.value, surface_indices_gpu, surface_triangle_indices.shape[0])
ccd.init_edges(bunnies.vertices["position"].compute().value, bunnies.vertices["rest_position"].compute().value, edge_indices_gpu, edge_indices.shape[0])



##################################################
## add energy to the scene
##################################################
snh0 = stable_neo_hookean(bunny0.tets["rest_positions"], bunny0.tets["positions"], mu0, lam0, dt)
snh_energy0 = bunny0.tets.addAttribute("stable_neo_hookean0", computed_attribute = snh0)
snh1 = stable_neo_hookean(bunny1.tets["rest_positions"], bunny1.tets["positions"], mu1, lam1, dt)
snh_energy1 = bunny1.tets.addAttribute("stable_neo_hookean1", computed_attribute = snh1)

inertia = inertia(bunnies.vertices["last_position"], bunnies.vertices["velocity"], dt, bunnies.vertices["position"], bunnies.vertices["mass"])
inertia_energy = bunnies.vertices.addAttribute("inertia", computed_attribute = inertia)

pp = point_point(pp_positions, dhat, kappa)
pp_energy = bunnies.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = bunnies.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = bunnies.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = bunnies.ee.addAttribute("edge_edge", computed_attribute = ee)

# we also need to remember to make the affine transformation affine
# which means the rotation @rotation.T should be identity
affine0 = affine_energy(bunny0["rotation"])
bunny0.addAttribute("affine0", computed_attribute = affine0)

# s0.addEnergy(snh_energy, projection_method = 2, save_intermediate = True)
s0.addEnergy(snh_energy0, projection_method = 1)
s0.addEnergy(snh_energy1, projection_method = 1)
s0.addEnergy(inertia_energy, projection_method = 1)
s0.addEnergy(affine0, projection_method = 1)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 1)
s0.addMinimizeTarget([bunny0["rotation"], bunny0["translation"], bunny1.vertices["position"]])
##################################################
## plot the scene
##################################################
import pyvista as pv
triangles = np.array(surface_triangle_indices)
triangles = triangles[:triangles.shape[0] // 2]

cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
bunny_poly0 = pv.PolyData(position[:position.shape[0] // 2, :], cells)
bunny_poly1 = pv.PolyData(position[position.shape[0] // 2:, :], cells)
plotter = pv.Plotter(window_size=(3840, 2160))
plotter.add_mesh(bunny_poly0, color='blue', opacity = 0.2)
plotter.add_mesh(bunny_poly1, color='red', opacity = 1.0)
plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]


plotter.show(interactive_update=True)
position_copy = bunnies.vertices["position"].compute().value.copy()
rot0_copy = gpuarray.zeros(9, dtype=np.float64)
trans0_copy = gpuarray.zeros(3, dtype=np.float64)
pos1_copy = bunny1.vertices["position"].value.copy()
for i in range(200):
  bunny0.vertices["last_position"].updateValue(bunny0.vertices["position"].compute().value, deepCopy = True)
  bunny1.vertices["last_position"].updateValue(bunny1.vertices["position"].value, deepCopy = True)
  inner_iteration = 0
  min_inner_iteration_energy = 100000000
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-6)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer
    snh_energy_sum = sum(snh_energy0.compute().value.get()) + sum(snh_energy1.compute().value.get())
    inertia_energy_sum = sum(inertia_energy.compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    affine_energy_sum = sum(affine0.compute().value.get())
    energies_before = snh_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum + affine_energy_sum
    print("snh energy sum before", snh_energy_sum)
    # energies_before = inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum + affine_energy_sum
    if energies_before < min_inner_iteration_energy:
      min_inner_iteration_energy = energies_before
    print(f"energy before {energies_before} vs minimum energy in newton: {min_inner_iteration_energy}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    # we perform CCD here
    # first we get the rotation and translation
    d_rot0 = result[0]
    d_trans0 = result[1]
    d_pos1 = result[2]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(bunnies.vertices["position"].compute().value)
    rot0_copy.set(bunny0["rotation"].value)
    trans0_copy.set(bunny0["translation"].value)
    pos1_copy.set(bunny1.vertices["position"].compute().value)

    # we first compute the new position
    bunny0["rotation"].updateValue(bunny0["rotation"].value - d_rot0, deepCopy = True)
    bunny0["translation"].updateValue(bunny0["translation"].value - d_trans0, deepCopy = True)
    bunny1.vertices["position"].updateValue(bunny1.vertices["position"].value - d_pos1, deepCopy = True)


    new_positions = bunnies.vertices["position"].compute().value
    # now we compute the new direction, remember it's the negative we need to put in
    direction_copy = position_copy - new_positions


    # check for the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.5, position_copy, direction_copy)
    # largest_step = 1.0
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    while substep <= 8:
      bunny0["rotation"].updateValue(rot0_copy - d_rot0 * step_taken, deepCopy = True)
      bunny0["translation"].updateValue(trans0_copy - d_trans0 * step_taken, deepCopy = True)
      bunny1.vertices["position"].updateValue(pos1_copy - d_pos1 * step_taken, deepCopy = True)

      # perform collision detection
      ccd.cd(bunnies.vertices["position"].compute().value, DHAT_VALUE) # perform collision detection
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      print("The separated counts are", ccd.separated_counts)
      bunnies.pp.updateNumInstances(pp_count)
      bunnies.pe.updateNumInstances(pe_count)
      bunnies.pt.updateNumInstances(pt_count)
      bunnies.ee.updateNumInstances(ee_count)
      if pp_count > 0:
        pp2v.updateConnectivity(ccd.pp[:2 * pp_count])
      if pe_count > 0:
        pe2v.updateConnectivity(ccd.pe[:3 * pe_count])
      if pt_count > 0:
        pt2v.updateConnectivity(ccd.pt[:4 * pt_count])
      if ee_count > 0:
        ee2v.updateConnectivity(ccd.ee[:4 * ee_count])

      snh_energy_sum = sum(snh_energy0.compute().value.get()) + sum(snh_energy1.compute().value.get())
      inertia_energy_sum = sum(inertia_energy.compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      affine_energy_sum = sum(affine0.compute().value.get())
      new_energies = snh_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum + affine_energy_sum
      print(f"snh energy sum line search substep {substep}", snh_energy_sum)
      # new_energies = inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum + affine_energy_sum
      print(f"energy comparison: {new_energies} vs {energies_before}")
      if new_energies <= energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    if substep > 8:
      print("failed")
      exit(1)
    print("step taken is", step_taken)
    print("substep is", substep)
    bunny_poly0.points = bunny0.vertices["position"].compute().value.get().reshape(-1, 3)
    bunny_poly1.points = bunny1.vertices["position"].compute().value.get().reshape(-1, 3)
    plotter.render()
    plotter.update()

    # print(f"Iteration {inner_iteration} max gradient: {max_grad}")
    if max_grad < 1e-2:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities0 = (bunny0.vertices["position"].compute().value - bunny0.vertices["last_position"].compute().value) / DT_VALUE
  new_velocities1 = (bunny1.vertices["position"].value - bunny1.vertices["last_position"].value) / DT_VALUE
  bunny0.vertices["velocity"].updateValue(new_velocities0, deepCopy = True)
  bunny1.vertices["velocity"].updateValue(new_velocities1, deepCopy = True)
  bunny_poly0.points = bunny0.vertices["position"].compute().value.get().reshape(-1, 3)
  bunny_poly1.points = bunny1.vertices["position"].compute().value.get().reshape(-1, 3)
  plotter.render()
  plotter.update()
  plotter.screenshot(f"outputs/bunny_abd_soft_{i:04d}.jpg")

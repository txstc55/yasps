from yasps import scene
from helpers import extract_surface_triangles, stable_neo_hookean, inertia, extract_edges_from_triangles, abs_max_reduce
from helpers import point_point, point_edge, point_triangle, edge_edge, safe_gpu_sum
import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
DT_VALUE = 0.01 # for time step
DHAT_VALUE = 1e-4 # for collision detection
KAPPA_VALUE = 10000.0 # for collision
POISSON_VALUE = 0.49
YOUNG_VALUE = 1000000.0
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

position = position - np.array([2.0, 0.0, 0.0])
position_second = position_second + np.array([2.0, 0.0, 0.0])
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

bunnies = s0.addMesh("bunnies")
bunnies.addPrimitive("vertices", numInstances = position.shape[0])
bunnies.addPrimitive("tets", numInstances = tet_indices.shape[0])
bunnies.addPrimitive("surfaceTriangles", numInstances = surface_triangle_indices.shape[0])
bunnies.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
bunnies.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
bunnies.addPrimitive("pt", numInstances = 0, isDynamic = True) # for point triangle collision
bunnies.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision

bunnies.vertices.addConstant("rest_position", rows = 3, cols = 1)
bunnies.vertices["rest_position"].updateValue(position)
bunnies.vertices.addAttribute("position", rows = 3, cols = 1)
# position[0, 0] -= 1.0
bunnies.vertices["position"].updateValue(position)

bunnies.vertices.addAttribute("last_position", rows = 3, cols = 1)
bunnies.vertices["last_position"].updateValue(position)
bunnies.vertices.addAttribute("velocity", rows = 3, cols = 1)
velocities = np.zeros_like(position, dtype=np.float64)
velocities[:position.shape[0] //2, 0] = 8
velocities[position.shape[0] // 2:, 0] = -8
bunnies.vertices["velocity"].updateValue(velocities)

bunnies.vertices.addConstant("mass", rows = 1, cols = 1)
bunnies.vertices["mass"].updateValue(np.ones((position.shape[0]), dtype=np.float64) * 1.0)

mu = bunnies.addConstant("mu", rows = 1, cols = 1) # for stable neo hookean
lam = bunnies.addConstant("lam", rows = 1, cols = 1) # for stable neo hookean
mu.updateValue([MU_VALUE])
lam.updateValue([LAMBDA_VALUE])

##################################################
## add connectivities, and attributes
##################################################
tet2v = bunnies.tets.addConnectivity("tet2v", bunnies.vertices, tet_indices, 4)
tri2v = bunnies.surfaceTriangles.addConnectivity("tri2v", bunnies.vertices, surface_triangle_indices, 3)
pp2v = bunnies.pp.addConnectivity("pp2v", bunnies.vertices, [], 2)
pe2v = bunnies.pe.addConnectivity("pe2v", bunnies.vertices, [], 3)
pt2v = bunnies.pt.addConnectivity("pt2v", bunnies.vertices, [], 4)
ee2v = bunnies.ee.addConnectivity("ee2v", bunnies.vertices, [], 4)
tet_positions = bunnies.tets.addAttribute("positions", through = tet2v, source = bunnies.vertices["position"])
tet_rest_positions = bunnies.tets.addAttribute("rest_positions", through = tet2v, source = bunnies.vertices["rest_position"])
tri_positions = bunnies.surfaceTriangles.addAttribute("positions", through = tri2v, source = bunnies.vertices["position"])
pp_positions = bunnies.pp.addAttribute("positions", through = pp2v, source = bunnies.vertices["position"])
pe_positions = bunnies.pe.addAttribute("positions", through = pe2v, source = bunnies.vertices["position"])
pt_positions = bunnies.pt.addAttribute("positions", through = pt2v, source = bunnies.vertices["position"])
ee_positions = bunnies.ee.addAttribute("positions", through = ee2v, source = bunnies.vertices["position"])

##################################################
# construct ccd
##################################################
ccd = CCD(len(surface_indices))
surface_indices_gpu = gpuarray.to_gpu(np.array(surface_indices).astype(np.uint32))
edge_indices_gpu = gpuarray.to_gpu(edge_indices.astype(np.uint32))
ccd.init_faces(bunnies.vertices["position"].value, tri2v.value, surface_indices_gpu, surface_triangle_indices.shape[0])
ccd.init_edges(bunnies.vertices["position"].value, bunnies.vertices["rest_position"].value, edge_indices_gpu, edge_indices.shape[0])



##################################################
## add energy to the scene
##################################################
snh = stable_neo_hookean(tet_rest_positions, tet_positions, mu, lam, dt)
snh_energy = bunnies.tets.addAttribute("stable_neo_hookean", computed_attribute = snh)
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


s0.addEnergy(snh_energy, projection_method = 2)
s0.addEnergy(inertia_energy, projection_method = 2)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 1)
s0.addMinimizeTarget([bunnies.vertices["position"]])
##################################################
## plot the scene
##################################################
import pyvista as pv
triangles = np.array(surface_triangle_indices)
triangles = triangles[:triangles.shape[0] // 2]

cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
bunny_poly1 = pv.PolyData(position[:position.shape[0] // 2, :], cells)
bunny_poly2 = pv.PolyData(position[position.shape[0] // 2:, :], cells)
plotter = pv.Plotter()
plotter.add_mesh(bunny_poly1, color='blue', opacity = 0.2)
plotter.add_mesh(bunny_poly2, color='red', opacity = 1.0)
plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]


plotter.show(interactive_update=True)
position_copy = bunnies.vertices["position"].value.copy()
direction_copy = gpuarray.zeros_like(bunnies.vertices["position"].value)
direction_zeros = gpuarray.zeros_like(bunnies.vertices["position"].value)
for i in range(200):
  bunnies.vertices["last_position"].updateValue(bunnies.vertices["position"].value, deepCopy = True)
  inner_iteration = 0
  min_inner_iteration_energy = 100000000
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-6)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer
    snh_energy_sum = sum(snh_energy.compute().value.get())
    inertia_energy_sum = sum(inertia_energy.compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = snh_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    if energies_before < min_inner_iteration_energy:
      min_inner_iteration_energy = energies_before
    print(f"energy before {energies_before} vs minimum energy in newton: {min_inner_iteration_energy}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    # we perform CCD here
    d_position = result[0]
    step_taken = 1.0
    # copy the position and direction
    position_copy.set(bunnies.vertices["position"].value)
    direction_copy.set(d_position)

    # check for the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    step_taken = largest_step
    substep = 1
    if step_taken == 2.0:
      bunnies.vertices["position"].updateValue(position_copy - d_position, deepCopy = True)
    else:
      step_taken = step_taken
      while substep <= 8:
        computed_position = position_copy - d_position * step_taken
        bunnies.vertices["position"].updateValue(computed_position, deepCopy = True)
        ccd.cd(bunnies.vertices["position"].value, DHAT_VALUE) # perform collision detection
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

        snh_energy_sum = sum(snh_energy.compute().value.get())
        inertia_energy_sum = sum(inertia_energy.compute().value.get())
        # inertia_energy_sum = 0.0
        pp_energy_sum = sum(pp_energy.compute().value.get())
        pe_energy_sum = sum(pe_energy.compute().value.get())
        pt_energy_sum = sum(pt_energy.compute().value.get())
        ee_energy_sum = sum(ee_energy.compute().value.get())
        # print("snh energy sum: ", snh_energy_sum)
        # print("inertia energy sum: ", inertia_energy_sum)
        # print("pp energy sum: ", pp_energy_sum)
        # print("pe energy sum: ", pe_energy_sum)
        # print("pt energy sum: ", pt_energy_sum)
        # print("ee energy sum: ", ee_energy_sum)
        new_energies = snh_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
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
    new_positions = bunnies.vertices["position"].value.get().reshape(-1, 3)
    bunny_poly1.points = new_positions[:position.shape[0] // 2, :]
    bunny_poly2.points = new_positions[position.shape[0] // 2:, :]
    plotter.render()
    plotter.update()

    # print(f"Iteration {inner_iteration} max gradient: {max_grad}")
    if max_grad < 1e-2:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities = (bunnies.vertices["position"].value - bunnies.vertices["last_position"].value) / DT_VALUE
  bunnies.vertices["velocity"].updateValue(new_velocities)
  new_positions = bunnies.vertices["position"].value.get().reshape(-1, 3)
  bunny_poly1.points = new_positions[:position.shape[0] // 2, :]
  bunny_poly2.points = new_positions[position.shape[0] // 2:, :]
  # export the current positions to obj
  # bunny_poly1.save(f"outputs/bunny1_{i:04d}.obj")
  # bunny_poly2.save(f"outputs/bunny2_{i:04d}.obj")
  plotter.render()
  plotter.update()
  plotter.screenshot(f"outputs/bunny1_{i:04d}.jpg")

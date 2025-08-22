from yasps import scene
from helpers import extract_surface_triangles, stable_neo_hookean, inertia, extract_edges_from_triangles, abs_max_reduce
from helpers import point_point, point_edge, point_triangle, edge_edge, affine_energy
import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
DT_VALUE = 0.01 # for time step
DHAT_VALUE = 1e-4 # for collision detection
NUM_FIXED_POINTS = 1
NUM_RIGID_POINTS = 4000
KAPPA_VALUE = 10000.0 # for collision
POISSON_VALUE = 0.01
YOUNG_VALUE = 10000000.0
MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))
MU_VALUE = 4.0 * MU_LAME_VALUE / 3.0
LAMBDA_VALUE = LAMBDA_LAME_VALUE + 5.0 * MU_LAME_VALUE / 6.0
BUNNY_SCALE_FACTOR = 1.0
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

f = open("../data/bunny.node", 'r')
f.readline()
position = []
for line in f:
  position.append([float(x) for x in line.split()[1:]])
f.close()
position = np.array(position, dtype = np.float64)

# extract surfaces and edges
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

bunny = s0.addMesh("bunny")

bunny.addPrimitive("fixed_vertices", numInstances = NUM_FIXED_POINTS)
bunny.addPrimitive("abd_vertices", numInstances = NUM_RIGID_POINTS)
bunny.addPrimitive("moving_vertices", numInstances = position.shape[0] - NUM_FIXED_POINTS - NUM_RIGID_POINTS)
bunny.addPrimitiveUnion("vertices", [bunny.fixed_vertices, bunny.abd_vertices, bunny.moving_vertices])
bunny.addPrimitive("tets", numInstances = tet_indices.shape[0])
bunny.addPrimitive("surfaceTriangles", numInstances = surface_triangle_indices.shape[0])
bunny.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
bunny.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
bunny.addPrimitive("pt", numInstances = 0, isDynamic = True) # for point triangle collision
bunny.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision

# add attributes for the abd vertices
bunny.addAttribute("rotation", rows = 3, cols = 3)
bunny["rotation"].updateValue(np.eye(3, dtype=np.float64))
bunny.addAttribute("translation", rows = 3, cols = 1)
bunny["translation"].updateValue(np.zeros((3, 1), dtype=np.float64))
bunny.abd_vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny.abd_vertices["rest_position"].updateValue(position[NUM_FIXED_POINTS:NUM_FIXED_POINTS + NUM_RIGID_POINTS, :])
bunny.abd_vertices.addAttribute("position", computed_attribute = bunny["translation"] + bunny["rotation"] * bunny.abd_vertices["rest_position"])
abd_lp = bunny.abd_vertices.addConstant("last_position", rows = 3, cols = 1)
bunny.abd_vertices["last_position"].updateValue(position[NUM_FIXED_POINTS:NUM_FIXED_POINTS + NUM_RIGID_POINTS, :])

# bunny.abd_vertices.addAttribute("test", computed_attribute = abd_lp.dot(abd_lp))
# print(bunny.abd_vertices["test"].compute().value.get())
# exit()



bunny.abd_vertices.addConstant("velocity", rows = 3, cols = 1)
velocities = np.zeros_like(position[NUM_FIXED_POINTS:NUM_FIXED_POINTS + NUM_RIGID_POINTS, :], dtype=np.float64)
bunny.abd_vertices["velocity"].updateValue(velocities)
bunny.abd_vertices.addConstant("mass", rows = 1, cols = 1)
bunny.abd_vertices["mass"].updateValue(np.ones((NUM_RIGID_POINTS), dtype=np.float64) * 0.01)

# add attributes for moving vertices
bunny.moving_vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny.moving_vertices["rest_position"].updateValue(position[NUM_FIXED_POINTS + NUM_RIGID_POINTS:, :])
bunny.moving_vertices.addAttribute("position", rows = 3, cols = 1)
bunny.moving_vertices["position"].updateValue(position[NUM_FIXED_POINTS + NUM_RIGID_POINTS:, :])
bunny.moving_vertices.addConstant("last_position", rows = 3, cols = 1)
bunny.moving_vertices["last_position"].updateValue(position[NUM_FIXED_POINTS + NUM_RIGID_POINTS:, :])
bunny.moving_vertices.addConstant("velocity", rows = 3, cols = 1)
velocities = np.zeros_like(position[NUM_FIXED_POINTS + NUM_RIGID_POINTS:, :], dtype=np.float64)
bunny.moving_vertices["velocity"].updateValue(velocities)
bunny.moving_vertices.addConstant("mass", rows = 1, cols = 1)
bunny.moving_vertices["mass"].updateValue(np.ones((position.shape[0] - 1), dtype=np.float64) * 4.0)

# add attributes for fixed vertices
bunny.fixed_vertices.addAttribute("position", rows = 3, cols = 1)
bunny.fixed_vertices["position"].updateValue(position[:NUM_FIXED_POINTS, :])
bunny.fixed_vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny.fixed_vertices["rest_position"].updateValue(position[:NUM_FIXED_POINTS, :])
bunny.fixed_vertices.addConstant("last_position", rows = 3, cols = 1)
bunny.fixed_vertices["last_position"].updateValue(position[:NUM_FIXED_POINTS, :])
bunny.fixed_vertices.addConstant("velocity", rows = 3, cols = 1)
bunny.fixed_vertices["velocity"].updateValue(np.zeros((1, 3), dtype=np.float64))

# add attribute for the union vertices
bunny.vertices.addAttribute("position")
bunny.vertices.addAttribute("rest_position")
bunny.vertices.addAttribute("last_position")
bunny.vertices.addAttribute("velocity")

mu = bunny.addConstant("mu", rows = 1, cols = 1) # for stable neo hookean
lam = bunny.addConstant("lam", rows = 1, cols = 1) # for stable neo hookean
mu.updateValue([MU_VALUE])
lam.updateValue([LAMBDA_VALUE])

##################################################
## add connectivities, and attributes
##################################################
tet2v = bunny.tets.addConnectivity("tet2v", bunny.vertices, tet_indices, 4)
tri2v = bunny.surfaceTriangles.addConnectivity("tri2v", bunny.vertices, surface_triangle_indices, 3)
pp2v = bunny.pp.addConnectivity("pp2v", bunny.vertices, [], 2)
pe2v = bunny.pe.addConnectivity("pe2v", bunny.vertices, [], 3)
pt2v = bunny.pt.addConnectivity("pt2v", bunny.vertices, [], 4)
ee2v = bunny.ee.addConnectivity("ee2v", bunny.vertices, [], 4)
tet_positions = bunny.tets.addAttribute("positions", through = tet2v, source = bunny.vertices["position"])
tet_rest_positions = bunny.tets.addAttribute("rest_positions", through = tet2v, source = bunny.vertices["rest_position"])
tri_positions = bunny.surfaceTriangles.addAttribute("positions", through = tri2v, source = bunny.vertices["position"])
pp_positions = bunny.pp.addAttribute("positions", through = pp2v, source = bunny.vertices["position"])
pe_positions = bunny.pe.addAttribute("positions", through = pe2v, source = bunny.vertices["position"])
pt_positions = bunny.pt.addAttribute("positions", through = pt2v, source = bunny.vertices["position"])
ee_positions = bunny.ee.addAttribute("positions", through = ee2v, source = bunny.vertices["position"])

##################################################
# construct ccd
##################################################
ccd = CCD(len(surface_indices), position.shape[0])
surface_indices_gpu = gpuarray.to_gpu(np.array(surface_indices).astype(np.uint32).flatten())
edge_indices_gpu = gpuarray.to_gpu(edge_indices.astype(np.uint32).flatten())
triangle_indices_gpu = gpuarray.to_gpu(surface_triangle_indices.astype(np.uint32).flatten())

position_gpu = gpuarray.to_gpu(position.astype(np.float64).flatten())
ccd.init_faces(position_gpu, triangle_indices_gpu, surface_indices_gpu, surface_triangle_indices.shape[0])
ccd.init_edges(position_gpu, position_gpu, edge_indices_gpu, edge_indices.shape[0])

##################################################
## add energy to the scene
##################################################
snh = stable_neo_hookean(tet_rest_positions, tet_positions, mu, lam, dt)
snh_energy = bunny.tets.addAttribute("stable_neo_hookean", computed_attribute = snh)

affine = affine_energy(bunny["rotation"])
affine_energy = bunny.addAttribute("affine_energy", computed_attribute = affine)

inertia_abd = inertia(bunny.abd_vertices["last_position"], bunny.abd_vertices["velocity"], dt, bunny.abd_vertices["position"], bunny.abd_vertices["mass"])
inertia_energy_abd = bunny.abd_vertices.addAttribute("inertia", computed_attribute = inertia_abd)

inertia_moving = inertia(bunny.moving_vertices["last_position"], bunny.moving_vertices["velocity"], dt, bunny.moving_vertices["position"], bunny.moving_vertices["mass"])
inertia_energy_moving = bunny.moving_vertices.addAttribute("inertia", computed_attribute = inertia_moving)



pp = point_point(pp_positions, dhat, kappa)
pp_energy = bunny.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = bunny.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = bunny.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = bunny.ee.addAttribute("edge_edge", computed_attribute = ee)


s0.addEnergy(snh_energy, projection_method = 2)
s0.addEnergy(affine_energy, projection_method = 2)
s0.addEnergy(inertia_energy_abd, projection_method = 2)
s0.addEnergy(inertia_energy_moving, projection_method = 2)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 2)
s0.addMinimizeTarget([bunny.moving_vertices["position"], bunny["rotation"], bunny["translation"]])

##################################################
## plot the scene
##################################################
import pyvista as pv
triangles = np.array(surface_triangle_indices)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
bunny_poly = pv.PolyData(position, cells)
plotter = pv.Plotter(window_size=(3840, 2160))
plotter.add_mesh(bunny_poly, color='blue', opacity = 0.5)
plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]
plotter.show(interactive_update=True)



position_copy = gpuarray.zeros_like(bunny.vertices["position"].compute().value)
position_copy.set(bunny.vertices["position"].compute().value)
direction_copy = gpuarray.zeros_like(bunny.vertices["position"].compute().value)
rotation_copy = bunny["rotation"].value.copy()
translation_copy = bunny["translation"].value.copy()
for i in range(2000):
  # for all the moving vertices we will copy the position to last_position
  bunny.abd_vertices["last_position"].updateValue(bunny.abd_vertices["position"].compute().value, deepCopy = True)
  bunny.moving_vertices["last_position"].updateValue(bunny.moving_vertices["position"].value, deepCopy = True)
  inner_iteration = 0
  min_inner_iteration_energy = 100000000
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-6)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer
    snh_energy_sum = sum(snh_energy.compute().value.get())
    affine_energy_sum = sum(affine_energy.compute().value.get())
    inertia_energy_sum = sum(inertia_energy_abd.compute().value.get()) + sum(inertia_energy_moving.compute().value.get())
    # inertia_energy_sum = 0.0
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
    d_rotation = result[1]
    d_translation = result[2]
    step_taken = 1.0
    # copy the position and direction
    rotation_copy.set(bunny["rotation"].value)
    translation_copy.set(bunny["translation"].value)
    position_copy[3 * (NUM_FIXED_POINTS + NUM_RIGID_POINTS):].set(bunny.moving_vertices["position"].value)
    direction_copy[3 * (NUM_FIXED_POINTS + NUM_RIGID_POINTS):].set(d_position)

    # now we also need to compute the direction for the abd points
    abd_positions = bunny.abd_vertices["position"].compute().value.copy()
    position_copy[3 * NUM_FIXED_POINTS : 3 * (NUM_FIXED_POINTS + NUM_RIGID_POINTS)].set(abd_positions)
    bunny["rotation"].updateValue(rotation_copy - d_rotation) # update the rotation
    bunny["translation"].updateValue(translation_copy - d_translation) # update the translation
    abd_positions_new = bunny.abd_vertices["position"].compute().value # compute the new positions
    direction_copy[3 * NUM_FIXED_POINTS : 3 * (NUM_FIXED_POINTS + NUM_RIGID_POINTS)].set(abd_positions - abd_positions_new)


    # check for the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    step_taken = largest_step
    substep = 1
    # step_taken = 1.0
    while substep <= 8:
      computed_position = position_copy - direction_copy * step_taken
      bunny.moving_vertices["position"].updateValue(computed_position[3 * (NUM_FIXED_POINTS + NUM_RIGID_POINTS):], deepCopy = True)
      bunny["rotation"].updateValue(rotation_copy - d_rotation * step_taken) # update the rotation
      bunny["translation"].updateValue(translation_copy - d_translation * step_taken) # update the translation
      computed_position[3 * NUM_FIXED_POINTS : 3 * (NUM_FIXED_POINTS + NUM_RIGID_POINTS)].set(bunny.abd_vertices["position"].compute().value)



      ccd.cd(computed_position, DHAT_VALUE) # perform collision detection
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      print("The separated counts are", ccd.separated_counts)
      bunny.pp.updateNumInstances(pp_count)
      bunny.pe.updateNumInstances(pe_count)
      bunny.pt.updateNumInstances(pt_count)
      bunny.ee.updateNumInstances(ee_count)
      if pp_count > 0:
        pp2v.updateConnectivity(ccd.pp[:2 * pp_count])
      if pe_count > 0:
        pe2v.updateConnectivity(ccd.pe[:3 * pe_count])
      if pt_count > 0:
        pt2v.updateConnectivity(ccd.pt[:4 * pt_count])
      if ee_count > 0:
        ee2v.updateConnectivity(ccd.ee[:4 * ee_count])

      snh_energy_sum = sum(snh_energy.compute().value.get())
      affine_energy_sum = sum(affine_energy.compute().value.get())
      inertia_energy_sum = sum(inertia_energy_abd.compute().value.get()) + sum(inertia_energy_moving.compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = snh_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
      print(f"energy comparison: {new_energies} vs {energies_before}")
      if new_energies <= energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    # if substep > 8:
    #   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #   print("substep failed")
    #   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #   exit()
    print("step taken is", step_taken)
    print("substep is", substep)
    new_positions = bunny.vertices["position"].compute().value.get().reshape(-1, 3)
    bunny_poly.points = new_positions
    plotter.render()
    plotter.update()

    # print(f"Iteration {inner_iteration} max gradient: {max_grad}")
    if max_grad < 1e-1:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities = (bunny.moving_vertices["position"].value - bunny.moving_vertices["last_position"].value) / DT_VALUE
  bunny.moving_vertices["velocity"].updateValue(new_velocities)

  abd_velocities = (bunny.abd_vertices["position"].compute().value - bunny.abd_vertices["last_position"].value) / DT_VALUE
  bunny.abd_vertices["velocity"].updateValue(abd_velocities)

  new_positions = bunny.vertices["position"].compute().value.get().reshape(-1, 3)
  bunny_poly.points = new_positions
  # # export the current positions to obj
  # bunny_poly.save(f"outputs/bunny1_{i:04d}.obj")
  plotter.render()
  plotter.update()
  plotter.screenshot(f"outputs/bunny1_partial_abd_{i:04d}.jpg")

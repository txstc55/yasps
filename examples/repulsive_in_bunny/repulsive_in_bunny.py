from yasps import scene
import numpy as np
from helpers import inertia, extract_edges_from_triangles, abs_max_reduce
from helpers import point_point, point_edge, point_triangle, edge_edge
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray


DT_VALUE = 0.01
KAPPA_VALUE = 10000000000.0 # for collision
DHAT_VALUE = 1e-6 # for collision detection
NUM_LOOP_POINTS = 3000
LENGTH_PENALTY = 0.1
SMOOTH_PENALTY = 10000.0
##################################################
## Read the mesh
##################################################
v_bunny = []
f_bunny = []
with open("../data/bunny_small.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("v "):
      parts = line.split()
      v_bunny.append([float(parts[1].split("//")[0]), float(parts[2].split("//")[0]), float(parts[3].split("//")[0])])
    elif line.startswith("f "):
      parts = line.split()
      f_bunny.append([int(parts[1].split("//")[0]) - 1, int(parts[2].split("//")[0]) - 1, int(parts[3].split("//")[0]) - 1])

e_bunny = extract_edges_from_triangles(f_bunny)
v_bunny = np.array(v_bunny, dtype=np.float64) * 0.02
f_bunny = np.array(f_bunny, dtype=np.uint32)
# we center the bunny first
v_center = np.mean(v_bunny, axis=0)
v_bunny -= v_center



# we get the bounding box of the bunny
min_x = np.min(v_bunny[:, 0])
max_x = np.max(v_bunny[:, 0])
min_y = np.min(v_bunny[:, 1])
max_y = np.max(v_bunny[:, 1])
min_z = np.min(v_bunny[:, 2])
max_z = np.max(v_bunny[:, 2])
print("Bounding box of the bunny:")
print(f"X: [{min_x}, {max_x}]")
print(f"Y: [{min_y}, {max_y}]")
print(f"Z: [{min_z}, {max_z}]")

##################################################
## now we create the loop
##################################################
v_loop = []
e_loop = []
# create a circular loop in the XY plane
for i in range(NUM_LOOP_POINTS):
  e_loop.append([i, (i + 1) % NUM_LOOP_POINTS])
  v_loop.append([np.cos(2 * np.pi * i / NUM_LOOP_POINTS) * 0.1,
                 np.sin(2 * np.pi * i / NUM_LOOP_POINTS) * 0.1,
                 0.0])
v_loop = np.array(v_loop, dtype=np.float64)
e_loop = np.array(e_loop, dtype=np.uint32)

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
bunny.addPrimitive("vertices", numInstances = v_bunny.shape[0])
bunny.vertices.addAttribute("position", rows = 3, cols = 1)
bunny.vertices["position"].updateValue(v_bunny)

loop = s0.addMesh("loop")
loop.addPrimitive("vertices", numInstances = v_loop.shape[0])
loop.vertices.addAttribute("position", rows = 3, cols = 1)
loop.vertices["position"].updateValue(v_loop)

# just for fun, we gonna add inertia to the points
loop.vertices.addConstant("velocity", rows = 3, cols = 1)
loop.vertices["velocity"].updateValue(np.zeros((v_loop.shape[0], 3), dtype=np.float64))
loop.vertices.addConstant("mass", rows = 1, cols = 1)
loop.vertices["mass"].updateValue(np.ones((v_loop.shape[0], 1), dtype=np.float64) * 0.1)
loop.vertices.addConstant("last_position", rows = 3, cols = 1)
loop.vertices["last_position"].updateValue(np.zeros((v_loop.shape[0], 3), dtype=np.float64))

# now we add the edge
loop.addPrimitive("edges", numInstances = e_loop.shape[0])
l2ev = loop.edges.addConnectivity("le2v", loop.vertices, e_loop, 2)
loop.edges.addAttribute("positions", through = l2ev, source = loop.vertices["position"])

# add the relation of each vertex to its two neighbors
loop.addPrimitive("two_edges", numInstances = e_loop.shape[0])
v2neighbor = []
for i in range(v_loop.shape[0]):
  v2neighbor.append([(i - 1) % v_loop.shape[0], i, (i + 1) % v_loop.shape[0]])
v2neighbor = np.array(v2neighbor, dtype=np.uint32)
v2n = loop.two_edges.addConnectivity("v2n", loop.vertices, v2neighbor, 3)
loop.two_edges.addAttribute("positions", through = v2n, source = loop.vertices["position"])

length_penalty = loop.addAttribute("length_penalty", rows = 1, cols = 1)
loop["length_penalty"].updateValue([LENGTH_PENALTY])
smooth_penalty =loop.addAttribute("smooth_penalty", rows = 1, cols = 1)
loop["smooth_penalty"].updateValue([SMOOTH_PENALTY])


collision_mesh = s0.addMesh("collision_mesh")
collision_mesh.addPrimitiveUnion("vertices", [bunny.vertices,loop.vertices])
collision_mesh.vertices.addAttribute("position")
collision_mesh.addPrimitive("surfaceTriangles", numInstances = f_bunny.shape[0])
collision_mesh.addPrimitive("pp", numInstances = 0, isDynamic = True)
collision_mesh.addPrimitive("pe", numInstances = 0, isDynamic = True)
collision_mesh.addPrimitive("pt", numInstances = 0, isDynamic = True)
collision_mesh.addPrimitive("ee", numInstances = 0, isDynamic = True)

##################################################
## add connectivities, and attributes
##################################################
tri2v = collision_mesh.surfaceTriangles.addConnectivity("tri2v", collision_mesh.vertices, f_bunny, 3)
pp2v = collision_mesh.pp.addConnectivity("pp2v", collision_mesh.vertices, [], 2)
pe2v = collision_mesh.pe.addConnectivity("pe2v", collision_mesh.vertices, [], 3)
pt2v = collision_mesh.pt.addConnectivity("pt2v", collision_mesh.vertices, [], 4)
ee2v = collision_mesh.ee.addConnectivity("ee2v", collision_mesh.vertices, [], 4)
tri_positions = collision_mesh.surfaceTriangles.addAttribute("positions", through = tri2v, source = collision_mesh.vertices["position"])
pp_positions = collision_mesh.pp.addAttribute("positions", through = pp2v, source = collision_mesh.vertices["position"])
pe_positions = collision_mesh.pe.addAttribute("positions", through = pe2v, source = collision_mesh.vertices["position"])
pt_positions = collision_mesh.pt.addAttribute("positions", through = pt2v, source = collision_mesh.vertices["position"])
ee_positions = collision_mesh.ee.addAttribute("positions", through = ee2v, source = collision_mesh.vertices["position"])

##################################################
## Now we can start add energies in the scene
##################################################
inertia = inertia(loop.vertices["last_position"], loop.vertices["velocity"], dt, loop.vertices["position"], loop.vertices["mass"])
inertia_energy = loop.vertices.addAttribute("inertia", computed_attribute = inertia)

inv_length = length_penalty / (loop.edges["positions"].row(0) - loop.edges["positions"].row(1)).dot(loop.edges["positions"].row(0) - loop.edges["positions"].row(1))
inv_length_energy = loop.edges.addAttribute("inv_length", computed_attribute = inv_length)

smooth_angle = smooth_penalty * (loop.two_edges["positions"].row(0) + loop.two_edges["positions"].row(2) -  2.0 * loop.two_edges["positions"].row(1)).dot(loop.two_edges["positions"].row(0) + loop.two_edges["positions"].row(2) -  2.0 * loop.two_edges["positions"].row(1))
smooth_angle_energy = loop.two_edges.addAttribute("smooth_angle", computed_attribute = smooth_angle)

pp = point_point(pp_positions, dhat, kappa)
pp_energy = collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

# s0.addEnergy(inertia_energy, projection_method = 1)
s0.addEnergy(inv_length_energy, projection_method = 2)
s0.addEnergy(smooth_angle_energy, projection_method = 2)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 2)
s0.addMinimizeTarget([loop.vertices["position"]])

##################################################
## initialize CCD
##################################################
ccd = CCD(v_bunny.shape[0] + v_loop.shape[0], v_bunny.shape[0] + v_loop.shape[0])
surface_indices_gpu = gpuarray.to_gpu(np.array(list(range(v_bunny.shape[0] + v_loop.shape[0]))).astype(np.uint32))
edge_indices_gpu = gpuarray.to_gpu((e_loop + v_bunny.shape[0]).astype(np.uint32))
ccd.init_faces(collision_mesh.vertices["position"].compute().value, tri2v.value, surface_indices_gpu, f_bunny.shape[0])
ccd.init_edges(collision_mesh.vertices["position"].compute().value, collision_mesh.vertices["position"].compute().value, edge_indices_gpu, e_loop.shape[0])

##################################################
## we do the plotting here
##################################################
import pyvista as pv
# first we add bunny
cells = np.hstack([np.full((f_bunny.shape[0], 1), 3), f_bunny])
bunny_poly = pv.PolyData(v_bunny, cells)
plotter = pv.Plotter()
plotter.add_mesh(bunny_poly, color='blue', opacity = 0.2)
# now we add the loop
lines = np.hstack([np.full((e_loop.shape[0], 1), 2, dtype=np.int64),
                   e_loop.astype(np.int64)]).ravel()
loop_poly = pv.PolyData(v_loop, lines=lines)
plotter.add_mesh(loop_poly, color='red', line_width=3)
plotter.camera_position = [(0.057117616649138364, -4.958252810522988, 2.699024423785117),
 (0.015973925954570728, 0.04388592494447419, 0.08368562376556503),
 (0.0023182094406358337, 0.46334925930586857, 0.8861727200753207)]
plotter.show(interactive_update=True)

# here we do the solve loops
position_copy = collision_mesh.vertices["position"].compute().value.copy()
direction_copy = gpuarray.zeros((v_bunny.shape[0] + v_loop.shape[0]) * 3, dtype=np.float64)
loop_position_copy = loop.vertices["position"].value.copy()
total_iterations = 0
for i in range(2000):
  loop.vertices["last_position"].updateValue(loop.vertices["position"].value, deepCopy = True)
  inner_iteration = 0
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-3)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer
    # inertia_energy_sum = sum(inertia_energy.compute().value.get())
    inertia_energy_sum = 0.0
    inv_length_energy_sum = sum(inv_length_energy.compute().value.get())
    smooth_angle_energy_sum = sum(smooth_angle_energy.compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = inertia_energy_sum + inv_length_energy_sum + smooth_angle_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    print("-------------------------------------------------------------------------------")
    print(f"energy before {energies_before}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    print("-------------------------------------------------------------------------------")

    d_pos = result[0]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    loop_position_copy.set(loop.vertices["position"].value.copy())
    direction_copy[v_bunny.shape[0] * 3:].set(d_pos)

    # we compute the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step

    # perform line search
    while substep <= 8:
      loop.vertices["position"].updateValue(loop_position_copy - d_pos * step_taken, deepCopy = True)
      # perform collision detection
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

      # inertia_energy_sum = sum(inertia_energy.compute().value.get())
      inertia_energy_sum = 0.0
      inv_length_energy_sum = sum(inv_length_energy.compute().value.get())
      smooth_angle_energy_sum = sum(smooth_angle_energy.compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = inertia_energy_sum + inv_length_energy_sum + smooth_angle_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
      print("===============================================================================")
      print(f"energy comparison: {new_energies} vs {energies_before}")
      print("===============================================================================")

      if new_energies < energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    if substep > 8:
      print("Line search failed, exiting inner loop")
      exit(1)

    print("step taken is", step_taken)
    print("substep is", substep)

    loop_poly.points = loop.vertices["position"].compute().value.get().reshape(-1, 3)
    plotter.render()
    plotter.update()
    plotter.screenshot(f"outputs/loop_{total_iterations:06d}.jpg")
    total_iterations += 1

    if max_grad < 10000.0:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities = (loop.vertices["position"].compute().value - loop.vertices["last_position"].compute().value) / DT_VALUE
  loop.vertices["velocity"].updateValue(new_velocities, deepCopy = True)

from helpers import extract_edges_from_triangles, extract_edges_2_tri, point_point, point_edge, point_triangle, edge_edge, abs_max_reduce
import numpy as np
from yasps import scene
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray


DT_VALUE = 0.01
KAPPA_VALUE = 1000000000000.0 # for collision
DHAT_VALUE = 1e-8 # for collision detection
TARGET_RADIUS = 0.01
RADIUS_ENERGY_WEIGHT = 0.00000000000000000000
TRIANGLE_ENERGY_WEIGHT = 0.0001
EDGE_LENGTH_ENERGY_WEIGHT = 0.00001
NORMAL_ENERGY_WEIGHT = 0.00001

######################################################
# Read the bunny mesh
######################################################
v_bunny = []
f_bunny = []
with open("../data/bunny_small.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("v "):
      parts = line.split()
      v_bunny.append([float(parts[1].split("//")[0]), float(parts[2].split("//")[0]), float(parts[3].split("//")[0])])
    elif line.startswith("f "):
      parts = line.split()
      f_bunny.append([int(parts[1].split("//")[1]) - 1, int(parts[2].split("//")[1]) - 1, int(parts[3].split("//")[1]) - 1])


e_bunny = extract_edges_from_triangles(f_bunny)
v_bunny = np.array(v_bunny, dtype=np.float64) * 0.04
f_bunny = np.array(f_bunny, dtype=np.uint32)
# we center the bunny first
v_center = np.mean(v_bunny, axis=0)
v_bunny -= v_center
e2t_bunny = extract_edges_2_tri(f_bunny, e_bunny)

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


######################################################
# Read the sphere mesh
######################################################
v_sphere = []
f_sphere = []
with open("../data/sphere.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("v "):
      parts = line.split()
      v_sphere.append([float(parts[1].split("/")[0]), float(parts[2].split("/")[0]), float(parts[3].split("/")[0])])
    elif line.startswith("f "):
      parts = line.split()
      f_sphere.append([int(parts[1].split("/")[0]) - 1, int(parts[2].split("/")[0]) - 1, int(parts[3].split("/")[0]) - 1])


e_sphere = extract_edges_from_triangles(f_sphere)
v_sphere = np.array(v_sphere, dtype=np.float64) * 2.0
f_sphere = np.array(f_sphere, dtype=np.uint32)
# we center the bunny first
v_center = np.mean(v_sphere, axis=0)
v_sphere -= v_center
e2t_sphere = extract_edges_2_tri(f_sphere, e_sphere)

# we get the bounding box of the bunny
min_x = np.min(v_sphere[:, 0])
max_x = np.max(v_sphere[:, 0])
min_y = np.min(v_sphere[:, 1])
max_y = np.max(v_sphere[:, 1])
min_z = np.min(v_sphere[:, 2])
max_z = np.max(v_sphere[:, 2])
print("Bounding box of the sphere:")
print(f"X: [{min_x}, {max_x}]")
print(f"Y: [{min_y}, {max_y}]")
print(f"Z: [{min_z}, {max_z}]")


######################################################
# here we create the bunny and the sphere
######################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA_VALUE])
target_radius = s0.addConstant("target_radius", rows = 1, cols = 1)
target_radius.updateValue([TARGET_RADIUS])
radius_energy_weight = s0.addConstant("radius_energy_weight", rows = 1, cols = 1)
radius_energy_weight.updateValue([RADIUS_ENERGY_WEIGHT])
triangle_energy_weight = s0.addConstant("triangle_energy_weight", rows = 1, cols = 1)
triangle_energy_weight.updateValue([TRIANGLE_ENERGY_WEIGHT])
edge_length_energy_weight = s0.addConstant("edge_length_energy_weight", rows = 1, cols = 1)
edge_length_energy_weight.updateValue([EDGE_LENGTH_ENERGY_WEIGHT])
normal_energy_weight = s0.addConstant("normal_energy_weight", rows = 1, cols = 1)
normal_energy_weight.updateValue([NORMAL_ENERGY_WEIGHT])

bunny = s0.addMesh("bunny")
bunny.addPrimitive("vertices", numInstances = v_bunny.shape[0])
bunny.vertices.addConstant("position", rows = 3, cols = 1)
bunny.vertices["position"].updateValue(v_bunny)

sphere = s0.addMesh("sphere")
sphere.addPrimitive("vertices", numInstances = v_sphere.shape[0])
sphere.vertices.addAttribute("position", rows = 3, cols = 1)
sphere.vertices["position"].updateValue(v_sphere)

######################################################
# create faces and bind attributes
######################################################
bunny.addPrimitive("faces", numInstances = f_bunny.shape[0])
f2v_bunny = bunny.faces.addConnectivity("f2v", bunny.vertices, f_bunny, 3)
fp = bunny.faces.addAttribute("positions", through = f2v_bunny, source = bunny.vertices["position"])

sphere.addPrimitive("faces", numInstances = f_sphere.shape[0])
f2v_sphere = sphere.faces.addConnectivity("f2v", sphere.vertices, f_sphere, 3)
fp_sphere = sphere.faces.addAttribute("positions", through = f2v_sphere, source = sphere.vertices["position"])
p0 = fp_sphere.row(0)
p1 = fp_sphere.row(1)
p2 = fp_sphere.row(2)
# compute the normal of the triangle
normal = (p1 - p0).cross(p2 - p0)
normal = normal / normal.norm()
sphere.faces.addAttribute("normal", computed_attribute = normal)

sphere.addPrimitive("edges", numInstances = e_sphere.shape[0])
e2f_sphere = sphere.edges.addConnectivity("e2f", sphere.faces, e2t_sphere, 2)
sphere.edges.addAttribute("normals", through = e2f_sphere, source = sphere.faces["normal"])

######################################################
# create collision mesh
######################################################
collision_mesh = s0.addMesh("collision_mesh")
collision_mesh.addPrimitiveUnion("vertices", [bunny.vertices, sphere.vertices])
collision_mesh.vertices.addAttribute("position")
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
# add energies
######################################################
# the first energy is to shrink the sphere by
# trying to make the distance from sphere's vertex distance to origin
# to a radius we support as close as possible
re = sphere.vertices.addAttribute("radius_energy", computed_attribute = radius_energy_weight * (sphere.vertices["position"].norm() - target_radius) * (sphere.vertices["position"].norm() - target_radius))

# the second energy we want to add is for each triangle on the sphere
# we want to make the length of the three edges as close as possible
e0 = (p1 - p0).norm()
e1 = (p2 - p1).norm()
e2 = (p0 - p2).norm()
diff01 = e0 - e1
diff12 = e1 - e2
diff20 = e2 - e0
te = sphere.faces.addAttribute("triangle_energy", computed_attribute = triangle_energy_weight * (diff01 * diff01 + diff12 * diff12 + diff20 * diff20))


# the third energy is to make the edges shrink, so it should be as short as possible
ele = sphere.faces.addAttribute("edge_length_energy", computed_attribute = edge_length_energy_weight * (e0 + e1 + e2))

# and the last one is saying the two triangles around an edge should be as smooth as possible
# we construct the energy as the dot product of the normals of the two triangles
ne = sphere.edges.addAttribute("edge_normal_energy", computed_attribute = normal_energy_weight * (1.0 - sphere.edges["normals"].row(0).dot(sphere.edges["normals"].row(1))))

# the rest are the energies for collision
# now we just add the collision energies
pp = point_point(pp_positions, dhat, kappa)
pp_energy = collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

######################################################
# we initialize ccd here
######################################################
ccd = CCD(v_bunny.shape[0], v_bunny.shape[0], max_cd_pairs = 200000000, max_ccd_pairs = 100000000)
surface_indices_gpu = gpuarray.to_gpu(np.array(list(range(v_bunny.shape[0] + v_sphere.shape[0]))).astype(np.uint32))
edge_indices_gpu = gpuarray.to_gpu(np.concatenate([e_bunny, e_sphere + v_bunny.shape[0]], axis=0).astype(np.uint32))
triangle_indices_gpu = gpuarray.to_gpu(np.concatenate([f_bunny, f_sphere + v_bunny.shape[0]], axis=0).astype(np.uint32))
ccd.init_faces(collision_mesh.vertices["position"].compute().value, triangle_indices_gpu, surface_indices_gpu, f_bunny.shape[0] + f_sphere.shape[0])
ccd.init_edges(collision_mesh.vertices["position"].compute().value, collision_mesh.vertices["position"].compute().value, edge_indices_gpu, e_bunny.shape[0] + e_sphere.shape[0])

######################################################
# add all energies to the scene
######################################################
s0.addEnergy(sphere.vertices["radius_energy"], projection_method = 1)
s0.addEnergy(sphere.faces["triangle_energy"], projection_method = 1)
s0.addEnergy(sphere.faces["edge_length_energy"], projection_method = 1)
s0.addEnergy(sphere.edges["edge_normal_energy"], projection_method = 1)
s0.addEnergy(collision_mesh.pp["point_point"], dynamic_instances = True, projection_method = 1)
s0.addEnergy(collision_mesh.pe["point_edge"], dynamic_instances = True, projection_method = 1)
s0.addEnergy(collision_mesh.pt["point_triangle"], dynamic_instances = True, projection_method = 1)
s0.addEnergy(collision_mesh.ee["edge_edge"], dynamic_instances = True, projection_method = 1)
s0.addMinimizeTarget([sphere.vertices["position"]])


##################################################
## we do the plotting here
##################################################
import pyvista as pv
# first we add bunny

plotter = pv.Plotter()

cells = np.hstack([np.full((f_bunny.shape[0], 1), 3), f_bunny])
bunny_poly = pv.PolyData(v_bunny, cells)
plotter.add_mesh(bunny_poly, color='blue', opacity = 1.0)

cells_sphere = np.hstack([np.full((f_sphere.shape[0], 1), 3), f_sphere])
sphere_poly = pv.PolyData(v_sphere, cells_sphere)
plotter.add_mesh(sphere_poly, color='red', opacity = 0.5)

plotter.camera_position = [(0.057117616649138364 * 2.0, -4.958252810522988 * 2.0, 2.699024423785117 * 2.0),
 (0.015973925954570728, 0.04388592494447419, 0.08368562376556503),
 (0.0023182094406358337, 0.46334925930586857, 0.8861727200753207)]
plotter.show(interactive_update=True)


# here we do the solve loops
position_copy = collision_mesh.vertices["position"].compute().value.copy()
direction_copy = gpuarray.zeros((v_bunny.shape[0] + v_sphere.shape[0]) * 3, dtype=np.float64)
total_iterations = 0
for i in range(2000):
  inner_iteration = 0
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-6, maxIterations = 100000)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer

    radius_energy_sum = sum(sphere.vertices["radius_energy"].compute().value.get())
    triangle_energy_sum = sum(sphere.faces["triangle_energy"].compute().value.get())
    edge_length_energy_sum = sum(sphere.faces["edge_length_energy"].compute().value.get())
    normal_energy_sum = sum(sphere.edges["edge_normal_energy"].compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = radius_energy_sum + triangle_energy_sum + edge_length_energy_sum + normal_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    print("-------------------------------------------------------------------------------")
    print(f"energy before {energies_before}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    print("-------------------------------------------------------------------------------")

    d_pos = result[0]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    direction_copy[3 * v_bunny.shape[0]:].set(d_pos)

    # we compute the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
#     # largest_step = 1.0
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step

    # perform line search
    while substep <= 8:
      sphere.vertices["position"].updateValue(position_copy[3 * v_bunny.shape[0]:] - d_pos * step_taken, deepCopy = True)
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
      radius_energy_sum = sum(sphere.vertices["radius_energy"].compute().value.get())
      triangle_energy_sum = sum(sphere.faces["triangle_energy"].compute().value.get())
      edge_length_energy_sum = sum(sphere.faces["edge_length_energy"].compute().value.get())
      normal_energy_sum = sum(sphere.edges["edge_normal_energy"].compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = radius_energy_sum + triangle_energy_sum + edge_length_energy_sum + normal_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
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

    sphere_poly.points = sphere.vertices["position"].value.get().reshape(-1, 3)
    plotter.render()
    plotter.update()
    plotter.screenshot(f"outputs/sphere_{total_iterations:06d}.jpg")
    total_iterations += 1

    if max_grad < 1e-4:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  # if (i + 1) % 1000 == 0:
  #   # save the bunny mesh to obj file
  #   bunny_poly.save(f"outputs/bunny_{i + 1:06d}.obj")

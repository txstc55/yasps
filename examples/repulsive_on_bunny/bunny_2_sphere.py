from helpers import extract_edges_from_triangles, extract_edges_2_tri, angle_energy, point_point, point_edge, point_triangle, edge_edge, abs_max_reduce
import numpy as np
from yasps import scene
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray

DT_VALUE = 0.01
KAPPA_VALUE = 1000000000000.0 # for collision
DHAT_VALUE = 1e-3 # for collision detection
TARGET_RADIUS = 0.7
RADIUS_ENERGY_WEIGHT = 10000.0
ANGLE_ENERGY_WEIGHT = 100.0
NORMAL_ENERGY_WEIGHT = 0.01
FACE_NORMAL_ENERGY_WEIGHT = 1000.0
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
      f_bunny.append([int(parts[1].split("//")[0]) - 1, int(parts[2].split("//")[0]) - 1, int(parts[3].split("//")[0]) - 1])


e_bunny = extract_edges_from_triangles(f_bunny)
v_bunny = np.array(v_bunny, dtype=np.float64) * 0.02
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
# here we create the bunny
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
angle_energy_weight = s0.addConstant("angle_energy_weight", rows = 1, cols = 1)
angle_energy_weight.updateValue([ANGLE_ENERGY_WEIGHT])
normal_energy_weight = s0.addConstant("normal_energy_weight", rows = 1, cols = 1)
normal_energy_weight.updateValue([NORMAL_ENERGY_WEIGHT])
face_normal_energy_weight = s0.addConstant("face_normal_energy_weight", rows = 1, cols = 1)
face_normal_energy_weight.updateValue([FACE_NORMAL_ENERGY_WEIGHT])


bunny = s0.addMesh("bunny")
bunny.addPrimitive("vertices", numInstances = v_bunny.shape[0])
bunny.vertices.addAttribute("position", rows = 3, cols = 1)
bunny.vertices["position"].updateValue(v_bunny)
bunny.vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny.vertices["rest_position"].updateValue(v_bunny)


######################################################
# create faces and bind attributes
######################################################
bunny.addPrimitive("faces", numInstances = f_bunny.shape[0])
f2v = bunny.faces.addConnectivity("f2v", bunny.vertices, f_bunny, 3)
fp = bunny.faces.addAttribute("positions", through = f2v, source = bunny.vertices["position"])
fpr = bunny.faces.addAttribute("rest_positions", through = f2v, source = bunny.vertices["rest_position"])

p0 = fp.row(0)
p1 = fp.row(1)
p2 = fp.row(2)
normal = (p1 - p0).cross(p2 - p0)
normal = normal / normal.norm()
fn = bunny.faces.addAttribute("normal", computed_attribute = normal)

######################################################
# create edges and bind attributes
######################################################
bunny.addPrimitive("edges", numInstances = e_bunny.shape[0])
e2f = bunny.edges.addConnectivity("e2f", bunny.faces, e2t_bunny, 2)
en = bunny.edges.addAttribute("normals", through = e2f, source = fn)

######################################################
# add for collision
######################################################
bunny.addPrimitive("pp", numInstances = 0, isDynamic = True)
bunny.addPrimitive("pe", numInstances = 0, isDynamic = True)
bunny.addPrimitive("pt", numInstances = 0, isDynamic = True)
bunny.addPrimitive("ee", numInstances = 0, isDynamic = True)
pp2v = bunny.pp.addConnectivity("pp2v", bunny.vertices, [], 2)
pe2v = bunny.pe.addConnectivity("pe2v", bunny.vertices, [], 3)
pt2v = bunny.pt.addConnectivity("pt2v", bunny.vertices, [], 4)
ee2v = bunny.ee.addConnectivity("ee2v", bunny.vertices, [], 4)
pp_positions = bunny.pp.addAttribute("positions", through = pp2v, source = bunny.vertices["position"])
pe_positions = bunny.pe.addAttribute("positions", through = pe2v, source = bunny.vertices["position"])
pt_positions = bunny.pt.addAttribute("positions", through = pt2v, source = bunny.vertices["position"])
ee_positions = bunny.ee.addAttribute("positions", through = ee2v, source = bunny.vertices["position"])

######################################################
# ok now we need to add couple of energies
# in the scene
######################################################
# the first one is we need to make the points go to target radius
# this is as simple as checking the norm
re = bunny.vertices.addAttribute("radius_energy", computed_attribute = radius_energy_weight * (bunny.vertices["position"].norm() - target_radius) * (bunny.vertices["position"].norm() - target_radius))

# the second one is that we need to add angle energy
# the angle energy tries to preserve the angles of the triangles on the surface
p0 = fp.row(0)
p1 = fp.row(1)
p2 = fp.row(2)
p0r = fpr.row(0)
p1r = fpr.row(1)
p2r = fpr.row(2)

ae = bunny.faces.addAttribute("angle_energy", computed_attribute = angle_energy_weight * angle_energy(p0, p1, p2, p0r, p1r, p2r)
  + angle_energy_weight * angle_energy(p1, p2, p0, p1r, p2r, p0r)
  + angle_energy_weight * angle_energy(p2, p0, p1, p2r, p0r, p1r))

# the third one is the normal energy
# where we try to make the normals of neighboring triangles
# as close as possible
# this will give us the smooth surface
fn0 = en.row(0)
fn1 = en.row(1)
ne = bunny.edges.addAttribute("normal_energy", computed_attribute = normal_energy_weight * ((fn0 - fn1).dot(fn0 - fn1)))

# now we add an energy saying
# the normal of the face should be the normal of the face if it is already
# on a unit sphere
p0n = p0 / p0.norm()
p1n = p1 / p1.norm()
p2n = p2 / p2.norm()
fn = (p1n - p0n).cross(p2n - p0n)
fn = fn / fn.norm()
fne = bunny.faces.addAttribute("face_normal_energy", computed_attribute = face_normal_energy_weight * (3.0 * fn - (p0n + p1n + p2n).transpose()).dot(3.0 * fn - (p0n + p1n + p2n).transpose()))

# now we just add the collision energies
pp = point_point(pp_positions, dhat, kappa)
pp_energy = bunny.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = bunny.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = bunny.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = bunny.ee.addAttribute("edge_edge", computed_attribute = ee)


######################################################
# we initialize ccd here
######################################################
ccd = CCD(v_bunny.shape[0], v_bunny.shape[0], max_cd_pairs = 100000000, max_ccd_pairs = 200000000)
surface_indices_gpu = gpuarray.to_gpu(np.array(list(range(v_bunny.shape[0] ))).astype(np.uint32))
edge_indices_gpu = gpuarray.to_gpu(e_bunny.astype(np.uint32))
ccd.init_faces(bunny.vertices["position"].compute().value, f2v.value, surface_indices_gpu, f_bunny.shape[0])
ccd.init_edges(bunny.vertices["position"].compute().value, bunny.vertices["position"].compute().value, edge_indices_gpu, e_bunny.shape[0])


######################################################
# add all energies to the scene
######################################################
s0.addEnergy(bunny.vertices["radius_energy"], projection_method = 1)
s0.addEnergy(bunny.faces["angle_energy"], projection_method = 1)
s0.addEnergy(bunny.edges["normal_energy"], projection_method = 1)
s0.addEnergy(bunny.faces["face_normal_energy"], projection_method = 1)
s0.addEnergy(bunny.pp["point_point"], dynamic_instances = True, projection_method = 1)
s0.addEnergy(bunny.pe["point_edge"], dynamic_instances = True, projection_method = 1)
s0.addEnergy(bunny.pt["point_triangle"], dynamic_instances = True, projection_method = 1)
s0.addEnergy(bunny.ee["edge_edge"], dynamic_instances = True, projection_method = 1)
s0.addMinimizeTarget([bunny.vertices["position"]])


##################################################
## we do the plotting here
##################################################
import pyvista as pv
# first we add bunny
cells = np.hstack([np.full((f_bunny.shape[0], 1), 3), f_bunny])
bunny_poly = pv.PolyData(v_bunny, cells)
plotter = pv.Plotter()
plotter.add_mesh(bunny_poly, color='blue', opacity = 0.5)
plotter.camera_position = [(0.057117616649138364, -4.958252810522988, 2.699024423785117),
 (0.015973925954570728, 0.04388592494447419, 0.08368562376556503),
 (0.0023182094406358337, 0.46334925930586857, 0.8861727200753207)]
plotter.show(interactive_update=True)


# here we do the solve loops
position_copy = bunny.vertices["position"].compute().value.copy()
direction_copy = gpuarray.zeros((v_bunny.shape[0]) * 3, dtype=np.float64)
total_iterations = 0
for i in range(20000):
  inner_iteration = 0
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-6, maxIterations = 100000)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer

    radius_energy_sum = sum(bunny.vertices["radius_energy"].compute().value.get())
    angle_energy_sum = sum(bunny.faces["angle_energy"].compute().value.get())
    normal_energy_sum = sum(bunny.edges["normal_energy"].compute().value.get())
    face_normal_energy_sum = sum(bunny.faces["face_normal_energy"].compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = radius_energy_sum + angle_energy_sum + normal_energy_sum + face_normal_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    print("-------------------------------------------------------------------------------")
    print(f"energy before {energies_before}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    print("-------------------------------------------------------------------------------")

    d_pos = result[0]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(bunny.vertices["position"].compute().value.copy())
    direction_copy.set(d_pos)

    # we compute the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    # largest_step = 1.0
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step

    # perform line search
    while substep <= 8:
      bunny.vertices["position"].updateValue(position_copy - d_pos * step_taken, deepCopy = True)
      # perform collision detection
      ccd.cd(bunny.vertices["position"].value, DHAT_VALUE) # perform collision detection
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

      # inertia_energy_sum = sum(inertia_energy.compute().value.get())
      radius_energy_sum = sum(bunny.vertices["radius_energy"].compute().value.get())
      angle_energy_sum = sum(bunny.faces["angle_energy"].compute().value.get())
      normal_energy_sum = sum(bunny.edges["normal_energy"].compute().value.get())
      face_normal_energy_sum = sum(bunny.faces["face_normal_energy"].compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = radius_energy_sum + angle_energy_sum + normal_energy_sum + face_normal_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
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

    bunny_poly.points = bunny.vertices["position"].value.get().reshape(-1, 3)
    plotter.render()
    plotter.update()
    # plotter.screenshot(f"outputs/loop_{total_iterations:06d}.jpg")
    total_iterations += 1

    if max_grad < 1e-4:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  if (i + 1) % 1000 == 0:
    # save the bunny mesh to obj file
    bunny_poly.save(f"outputs/bunny_{i + 1:06d}.obj")

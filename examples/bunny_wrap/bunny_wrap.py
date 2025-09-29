from yasps import scene
import numpy as np
from helpers import extract_edges_from_triangles, extract_edge_to_vertices
from helpers import bending, baraff_witkin
from helpers import point_point, point_edge, point_triangle, edge_edge, inertia
from helpers import abs_max_reduce
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray

DT_VALUE = 0.001
KAPPA_VALUE = 1000000.0 # for collision
DHAT_VALUE = 1e-6 # for collision detection
BENDING_STIFFNESS = 0.001
STRETCH_STIFFNESS = 10000.0
SHEAR_STIFFNESS = 0.01
THICKNESS = 1.0
G = 0.0
######################################################
# Read the bunny mesh
######################################################
v_bunny = []
f_bunny = []
with open("../data/bunny_small.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("v "):
      parts = line.split()
      v_bunny.append([float(parts[1]), float(parts[2]), float(parts[3])])
    elif line.startswith("f "):
      parts = line.split()
      f_bunny.append([int(parts[1].split("//")[0]) - 1, int(parts[2].split("//")[0]) - 1, int(parts[3].split("//")[0]) - 1])

v_bunny = np.array(v_bunny, dtype=np.float64)
f_bunny = np.array(f_bunny, dtype=np.uint32)
e_bunny = extract_edges_from_triangles(f_bunny)
# get the bounding box of the uv coordinates
min_x = np.min(v_bunny[:, 0])
max_x = np.max(v_bunny[:, 0])
min_y = np.min(v_bunny[:, 1])
max_y = np.max(v_bunny[:, 1])
min_z = np.min(v_bunny[:, 2])
max_z = np.max(v_bunny[:, 2])
# center the bunny
v_bunny[:, 0] = v_bunny[:, 0] - (min_x + max_x) / 2
v_bunny[:, 1] = v_bunny[:, 1] - (min_y + max_y) / 2
v_bunny[:, 2] = v_bunny[:, 2] - (min_z + max_z) / 2
v_bunny = v_bunny * 0.1
######################################################
# Read the sphere mesh
######################################################
v_sphere = []
f_sphere = []
with open("../data/sphere_large.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("v "):
      parts = line.split()
      v_sphere.append([float(parts[1]), float(parts[2]), float(parts[3])])
    elif line.startswith("f "):
      parts = line.split()
      f_sphere.append([int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1])

v_sphere = np.array(v_sphere, dtype=np.float64)
# make sure v_sphere is projected onto a unit sphere
v_sphere = v_sphere / np.linalg.norm(v_sphere, axis=1, keepdims=True)

f_sphere = np.array(f_sphere, dtype=np.uint32)
e_sphere = extract_edges_from_triangles(f_sphere)
# get the bounding box of the uv coordinates
min_x = np.min(v_sphere[:, 0])
max_x = np.max(v_sphere[:, 0])
min_y = np.min(v_sphere[:, 1])
max_y = np.max(v_sphere[:, 1])
min_z = np.min(v_sphere[:, 2])
max_z = np.max(v_sphere[:, 2])
v_sphere = v_sphere * 5.0
e2v_sphere = extract_edge_to_vertices(f_sphere)

######################################################
# add bunny to the scene
######################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA_VALUE])
bending_stiffness = s0.addConstant("bending_stiffness", rows = 1, cols = 1)
bending_stiffness.updateValue([BENDING_STIFFNESS])
stretch_stiffness = s0.addConstant("stretch_stiffness", rows = 1, cols = 1)
stretch_stiffness.updateValue([STRETCH_STIFFNESS])
shear_stiffness = s0.addConstant("shear_stiffness", rows = 1, cols = 1)
shear_stiffness.updateValue([SHEAR_STIFFNESS])
thickness = s0.addConstant("thickness", rows = 1, cols = 1)
thickness.updateValue([THICKNESS])
g = s0.addConstant("g", rows = 1, cols = 1)
g.updateValue([G])

bunny = s0.addMesh("bunny")
bunny.addPrimitive("vertices", numInstances = v_bunny.shape[0])
bunny.vertices.addAttribute("position", rows = 3, cols = 1)
bunny.vertices["position"].updateValue(v_bunny)
bunny.vertices.addConstant("rest_position", rows = 3, cols = 1)
bunny.vertices["rest_position"].updateValue(v_bunny)

######################################################
# add sphere to the scene
######################################################
sphere = s0.addMesh("sphere")
sphere.addPrimitive("vertices", numInstances = v_sphere.shape[0])
sphere.vertices.addAttribute("position", rows = 3, cols = 1)
sphere.vertices["position"].updateValue(v_sphere)
sphere.vertices.addConstant("rest_position", rows = 3, cols = 1)
sphere.vertices["rest_position"].updateValue(v_sphere / 100.0)
sphere.vertices.addAttribute("mass", rows = 1, cols = 1)
sphere.vertices["mass"].updateValue(np.ones(v_sphere.shape[0]) * 0.01)
sphere.vertices.addAttribute("last_position", rows = 3, cols = 1)
sphere.vertices["last_position"].updateValue(v_sphere)
sphere.vertices.addAttribute("velocity", rows = 3, cols = 1)
sphere.vertices["velocity"].updateValue(np.zeros((v_sphere.shape[0], 3)))
######################################################
# we now build the edge and triangle
# for the sphere
######################################################
sphere.addPrimitive("faces", numInstances = f_sphere.shape[0])
f2v = sphere.faces.addConnectivity("f2v", sphere.vertices, f_sphere, 3)
sphere.faces.addAttribute("positions", through = f2v, source = sphere.vertices["position"])
sphere.faces.addAttribute("rest_positions", through = f2v, source = sphere.vertices["rest_position"])

sphere.addPrimitive("edges", numInstances = e_sphere.shape[0])
e2v = sphere.edges.addConnectivity("e2v", sphere.vertices, e2v_sphere, 4)
sphere.edges.addAttribute("positions", through = e2v, source = sphere.vertices["position"])
sphere.edges.addAttribute("rest_positions", through = e2v, source = sphere.vertices["rest_position"])


######################################################
# build the collision mesh
######################################################
collision_mesh = s0.addMesh("collision_mesh")
collision_mesh.addPrimitiveUnion("vertices", [bunny.vertices, sphere.vertices])
collision_mesh.vertices.addAttribute("position")
collision_mesh.vertices.addAttribute("rest_position")

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
# now we add the energies
######################################################
bending_energy = bending(sphere.edges["positions"], sphere.edges["rest_positions"], bending_stiffness)
sphere.edges.addAttribute("bending_energy", computed_attribute = bending_energy)

baraff_witkin_energy = baraff_witkin(sphere.faces["rest_positions"], sphere.faces["positions"], stretch_stiffness, shear_stiffness, thickness, dt)
sphere.faces.addAttribute("baraff_witkin_energy", computed_attribute = baraff_witkin_energy)

inertia_energy = inertia(sphere.vertices["last_position"], sphere.vertices["velocity"], dt, sphere.vertices["position"], sphere.vertices["mass"], g)
sphere.vertices.addAttribute("inertia_energy", computed_attribute = inertia_energy)

# and the rest are the collision energies
pp = point_point(pp_positions, dhat, kappa)
pp_energy = collision_mesh.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = collision_mesh.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = collision_mesh.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = collision_mesh.ee.addAttribute("edge_edge", computed_attribute = ee)

s0.addEnergy(bending_energy, projection_method = 1)
s0.addEnergy(baraff_witkin_energy, projection_method = 1)
s0.addEnergy(inertia_energy, projection_method = 1)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 1)
s0.addMinimizeTarget([sphere.vertices["position"]])

######################################################
# we now initialize the ccd
######################################################
# here in this case there is no triangle collisions
# only edges
ccd = CCD(v_bunny.shape[0] + v_sphere.shape[0], v_bunny.shape[0] + v_sphere.shape[0], max_cd_pairs = 100000000, mesh_indices = [1] * v_bunny.shape[0] + [0] * v_sphere.shape[0])
triangle_indices_gpu = gpuarray.to_gpu(np.vstack([f_bunny, f_sphere + v_bunny.shape[0]]).astype(np.uint32))
surface_indices_gpu = gpuarray.to_gpu(np.array(list(range(v_bunny.shape[0] + v_sphere.shape[0])), dtype = np.uint32))
edge_indices_gpu = gpuarray.to_gpu((np.vstack([e_bunny, e_sphere + v_bunny.shape[0]])).astype(np.uint32))

ccd.init_faces(collision_mesh.vertices["position"].compute().value, triangle_indices_gpu, surface_indices_gpu, f_bunny.shape[0] + f_sphere.shape[0])
ccd.init_edges(collision_mesh.vertices["position"].compute().value, collision_mesh.vertices["rest_position"].compute().value, edge_indices_gpu, e_bunny.shape[0] + e_sphere.shape[0])


######################################################
# plot the bunny and the sphere
######################################################
import pyvista as pv
# first we add bunny

plotter = pv.Plotter()
cells_bunny = np.hstack([np.full((f_bunny.shape[0], 1), 3), f_bunny])
bunny_poly = pv.PolyData(v_bunny, cells_bunny)
plotter.add_mesh(bunny_poly, color='cyan', opacity = 0.6)

cells_sphere = np.hstack([np.full((f_sphere.shape[0], 1), 3), f_sphere])
sphere_poly = pv.PolyData(v_sphere, cells_sphere)
plotter.add_mesh(sphere_poly, color='yellow', opacity = 0.6)
plotter.camera_position = [(0, -20, 0.0),
 (0.0, 0.0, 0.0),
 (0.0, 0.0, 1.0)]
plotter.show(interactive_update=True)
# plotter.show()
bunny_poly.save("outputs/bunny_start.obj")
######################################################
# do the solve loops
######################################################
# here we do the solve loops
position_copy = collision_mesh.vertices["position"].compute().value.copy()
direction_copy = gpuarray.zeros((v_bunny.shape[0] + v_sphere.shape[0]) * 3, dtype=np.float64)
sphere_position_copy = sphere.vertices["position"].value.copy()
total_iterations = 0

for i in range(1000):
  inner_iteration = 0
  sphere.vertices["last_position"].updateValue(sphere.vertices["position"].value, deepCopy = True)
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-12, maxIterations = 100000)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer

    # compute the total energy in the scene
    bending_energy_sum = sum(bending_energy.compute().value.get())
    baraff_witkin_energy_sum = sum(baraff_witkin_energy.compute().value.get())
    inertia_energy_sum = sum(inertia_energy.compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = bending_energy_sum + baraff_witkin_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    print("-------------------------------------------------------------------------------")
    print(f"energy before {energies_before}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    print("-------------------------------------------------------------------------------")

    d_pos = result[0]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(collision_mesh.vertices["position"].compute().value.copy())
    sphere_position_copy.set(sphere.vertices["position"].value.copy())
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
      sphere.vertices["position"].updateValue(sphere_position_copy - d_pos * step_taken, deepCopy = True)
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

      bending_energy_sum = sum(bending_energy.compute().value.get())
      baraff_witkin_energy_sum = sum(baraff_witkin_energy.compute().value.get())
      inertia_energy_sum = sum(inertia_energy.compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = bending_energy_sum + baraff_witkin_energy_sum + inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
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

    sphere_positions = sphere.vertices["position"].value.get().reshape(-1, 3)
    sphere_poly.points = sphere_positions
    plotter.render()
    plotter.update()

    total_iterations += 1
    if max_grad < 1e-2:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities = (sphere.vertices["position"].value - sphere.vertices["last_position"].value) / DT_VALUE
  sphere.vertices["velocity"].updateValue(new_velocities, deepCopy = True)
  plotter.screenshot(f"outputs/bunny_wrap_{i:04d}.jpg")
  sphere_poly.save(f"outputs/sphere_{i:04d}.obj")

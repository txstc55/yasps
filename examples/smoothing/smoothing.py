from yasps import scene
import numpy as np
from helpers import compute_edge_to_triangle_vertices, baraff_witkin, inertia, quadratic_bending_energy, point_point, point_edge, point_triangle, edge_edge, bending, radius_energy
DT_VALUE = 0.01
BENDING_STIFFNESS = 1000.0
STRETCH_STIFFNESS = 33557.469799
SHEAR_STIFFNESS = 10607.114094
THICKNESS = 0.001
DHAT_VALUE = 0.01
KAPPA = 1000000.0
TARGET_RADIUS = 40.0
RADIUS_PENALTY = 0.000001

bunny_vertices = []
bunny_faces = []

f = open("../data/bunny_small.obj", 'r')
for line in f:
  if line.startswith('v '):
    bunny_vertices.append([float(x) for x in line.strip().split()[1:]])
  if line.startswith('f '):
    bunny_faces.append([int(x.split('//')[0]) - 1 for x in line.strip().split()[1:]])

bunny_faces = np.array(bunny_faces, dtype=np.uint32)
bunny_vertices = np.array(bunny_vertices, dtype=np.float64)
x_max = np.max(bunny_vertices[:, 0])
x_min = np.min(bunny_vertices[:, 0])
y_max = np.max(bunny_vertices[:, 1])
y_min = np.min(bunny_vertices[:, 1])
z_max = np.max(bunny_vertices[:, 2])
z_min = np.min(bunny_vertices[:, 2])
center = np.array([(x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0])
diagonal_size = np.linalg.norm(np.array([x_max - x_min, y_max - y_min, z_max - z_min]))
print(f"Center of bunny: {center}, diagonal size: {diagonal_size}")
bunny_vertices -= center



bunny_edges_to_vertices = compute_edge_to_triangle_vertices(bunny_faces)
bunny_edges_to_vertices = np.array(bunny_edges_to_vertices, dtype=np.uint32)

#####################################################
# Now we create the scene
#####################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA])
radius_penalty = s0.addConstant("radius_penalty", rows = 1, cols = 1)
radius_penalty.updateValue([RADIUS_PENALTY])
target_radius = s0.addConstant("target_radius", rows = 1, cols = 1)
target_radius.updateValue([TARGET_RADIUS])


bunny = s0.addMesh("bunny")
bending_stiffness = bunny.addConstant("bending_stiffness", rows = 1, cols = 1)
bending_stiffness.updateValue([BENDING_STIFFNESS])
stretch_stiffness = bunny.addConstant("stretch_stiffness", rows = 1, cols = 1)
stretch_stiffness.updateValue([STRETCH_STIFFNESS])
shear_stiffness = bunny.addConstant("shear_stiffness", rows = 1, cols = 1)
shear_stiffness.updateValue([SHEAR_STIFFNESS])
thickness = bunny.addConstant("thickness", rows = 1, cols = 1)
thickness.updateValue([THICKNESS])
center = bunny.addConstant("center", rows = 3, cols = 1)
center.updateValue(np.array([0, 0, 0], dtype=np.float64))

#####################################################
# Create vertices
#####################################################
bv = bunny.addPrimitive("vertices", bunny_vertices.shape[0])
bvp = bv.addAttribute("position", rows = 3, cols = 1)
bvp.updateValue(bunny_vertices.flatten())
bvm = bv.addConstant("mass", rows = 1, cols = 1)
bvm.updateValue([1.0] * bunny_vertices.shape[0])
bvlp = bv.addConstant("last_position", rows = 3, cols = 1)
bvlp.updateValue(bunny_vertices.flatten())
bvv = bv.addConstant("velocity", rows = 3, cols = 1)
bvv.updateValue([0.0] * bunny_vertices.shape[0] * 3)
bvip = bv.addConstant("initial_position", rows = 3, cols = 1)
bvip.updateValue(bunny_vertices.flatten())

#####################################################
# Create faces
#####################################################
bf = bunny.addPrimitive("faces", bunny_faces.shape[0])
bf2bv = bf.addConnectivity("bf2bv", bv, bunny_faces, 3)
bfp = bf.addAttribute("positions", through = bf2bv, source = bvp)
bfp.resize(3, 3)
bfip = bf.addAttribute("initial_positions", through = bf2bv, source = bvip)
bfip.resize(3, 3)

#####################################################
# Create edges
#####################################################
be = bunny.addPrimitive("edges", bunny_edges_to_vertices.shape[0])
be2bv = be.addConnectivity("be2bv", bv, bunny_edges_to_vertices, 4)
bep = be.addAttribute("positions", through = be2bv, source = bvp)
bep.resize(4, 3)
beip = be.addAttribute("initial_positions", through = be2bv, source = bvip)
beip.resize(4, 3)

#####################################################
# Create edges but only 2 points
#####################################################
be2 = bunny.addPrimitive("edges2", bunny_edges_to_vertices.shape[0])
be2bv = be2.addConnectivity("be2bv", bv, bunny_edges_to_vertices[:, :2], 2)
be2p = be2.addAttribute("positions", through = be2bv, source = bvp)
be2p.resize(2, 3)

#####################################################
# Create collision pairs
#####################################################
bunny.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
bunny.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
bunny.addPrimitive("pt", numInstances = 0, isDynamic = True) # # for point triangle collision
bunny.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision
pp2v = bunny.pp.addConnectivity("pp2v", bunny.vertices, [], 2)
pe2v = bunny.pe.addConnectivity("pe2v", bunny.vertices, [], 3)
pt2v = bunny.pt.addConnectivity("pt2v", bunny.vertices, [], 4)
ee2v = bunny.ee.addConnectivity("ee2v", bunny.vertices, [], 4)
pp_positions = bunny.pp.addAttribute("positions", through = pp2v, source = bunny.vertices["position"])
pe_positions = bunny.pe.addAttribute("positions", through = pe2v, source = bunny.vertices["position"])
pt_positions = bunny.pt.addAttribute("positions", through = pt2v, source = bunny.vertices["position"])
ee_positions = bunny.ee.addAttribute("positions", through = ee2v, source = bunny.vertices["position"])

#####################################################
# Add energies
#####################################################
# Bending energy
bending_energy = bending(bep, beip, bending_stiffness, dt)
be.addAttribute("bending_energy", computed_attribute = bending_energy)

# Baraff-Witkin energy
baraff_witkin_energy = baraff_witkin(bfip, bfp, stretch_stiffness, shear_stiffness, thickness, dt)
bf.addAttribute("baraff_witkin_energy", computed_attribute = baraff_witkin_energy)

# inertia
inertia_energy = inertia(bvlp, bvv, dt, bvp, bvm)
bv.addAttribute("inertia_energy", computed_attribute = inertia_energy)

# collision
pp = point_point(pp_positions, dhat, kappa)
bunny.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
bunny.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
bunny.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
bunny.ee.addAttribute("edge_edge", computed_attribute = ee)

# meet the target radius
radius_energy_compute = radius_energy(bvp, center, target_radius, radius_penalty)
bv.addAttribute("radius_energy", computed_attribute = radius_energy_compute)

s0.addEnergy(bending_energy, projection_method = 2)
s0.addEnergy(baraff_witkin_energy, projection_method = 2)
s0.addEnergy(inertia_energy, projection_method = 0)
s0.addEnergy(radius_energy_compute, projection_method = 2)
s0.addEnergy(pp, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2)

s0.addMinimizeTarget([bvp])


import pycuda.gpuarray as gpuarray
def compute_total_energy():
  total_energy = 0.0
  total_energy += gpuarray.sum(inertia_energy.compute().value).get()
  total_energy += gpuarray.sum(bending_energy.compute().value).get()
  total_energy += gpuarray.sum(baraff_witkin_energy.compute().value).get()
  total_energy += gpuarray.sum(radius_energy_compute.compute().value).get()
  if pp.correspondance.numInstances > 0:
    total_energy += gpuarray.sum(pp.compute().value).get()
  if pe.correspondance.numInstances > 0:
    total_energy += gpuarray.sum(pe.compute().value).get()
  if pt.correspondance.numInstances > 0:
    total_energy += gpuarray.sum(pt.compute().value).get()
  if ee.correspondance.numInstances > 0:
    total_energy += gpuarray.sum(ee.compute().value).get()
  return total_energy

#####################################################
# Initialize CCD
#####################################################
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD

ccd = CCD(bunny_vertices.shape[0], # the number of surface points
  bunny_vertices.shape[0], # the number of total points
  max_ccd_pairs = 100000000
)

position_gpu = gpuarray.to_gpu(bvp.compute().value.get()) # basically copy it out
indices_gpu = gpuarray.to_gpu(np.array(list(range(bunny_vertices.shape[0]))))
faces_gpu = gpuarray.to_gpu(bunny_faces.flatten())
edges_gpu = gpuarray.to_gpu(bunny_edges_to_vertices[:, :2].flatten())

ccd.init_faces(position_gpu, faces_gpu, indices_gpu, bunny_faces.shape[0])
ccd.init_edges(position_gpu, position_gpu, edges_gpu, bunny_edges_to_vertices.shape[0])


#####################################################
# Visualization
#####################################################
import pyvista as pv
plotter = pv.Plotter(window_size=[3840, 2160])
all_vertices_computed = bvp.compute().value.get().reshape((-1, 3))
triangles = bunny_faces

bunny_poly = pv.PolyData(all_vertices_computed, np.hstack((np.full((triangles.shape[0], 1), 3), triangles)).astype(np.int32))
plotter.add_mesh(bunny_poly)
plotter.show(interactive_update=True, auto_close=False)

position_copy = bvp.compute().value.copy()
direction = gpuarray.to_gpu(np.zeros(bunny_vertices.flatten().shape, dtype=np.float64))
for i in range(2000000):
  bvlp.updateValue(bvp.compute().value, deepCopy = True)
  inner_iteration = 0
  while True:
    print("==================================================================")
    print(f"At iteration {i}, inner iteration {inner_iteration}")
    result = s0.minimizeEnergy(tolerance = 1e-8)
    print("==================================================================")
    energy_before = compute_total_energy()
    d_p = result[0]
    max_movement = gpuarray.max(abs(d_p)).get()
    if max_movement < 1e-4:
      break

    position_copy.set(bvp.compute().value.get())
    direction.set(-d_p.get())
    ccd.ccd(position_copy, DHAT_VALUE, direction, 1.0)


    largest_step = ccd.compute_largest_step_size(0.8, bvp.value, direction)
    position_copy = bvp.compute().value.copy()
    substep = 1
    step_taken = largest_step
    while substep <= 8:
      bvp.updateValue(position_copy - step_taken * d_p, deepCopy = True)
      ccd.cd(bvp.value, DHAT_VALUE)
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
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

      energy_after = compute_total_energy()
      if energy_after < energy_before:
        break

      step_taken = step_taken / 2.0
      substep += 1

    bunny_poly.points = bvp.compute().value.get().reshape((-1, 3))
    plotter.update()
    plotter.render()
    inner_iteration += 1

  bvv.updateValue((bvp.compute().value - bvlp.compute().value) / DT_VALUE, deepCopy = True)
  new_center = bvp.value.get().reshape(-1, 3).mean(axis = 0)
  print(f"New center: {new_center}")
  RADIUS_PENALTY = 0.000000000001 * i
  radius_penalty.updateValue([RADIUS_PENALTY])
  center.updateValue(new_center.flatten())
  if i % 100 == 0:
    bunny_poly.save(f"meshes/smoothing_result_{i:05d}.obj")

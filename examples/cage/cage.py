import numpy as np
from yasps import scene
POISSON_VALUE = 0.25
YOUNG_VALUE = 100
MU_LAME_VALUE = YOUNG_VALUE / (2 * (1 + POISSON_VALUE))
LAMBDA_LAME_VALUE = YOUNG_VALUE * POISSON_VALUE / ((1 + POISSON_VALUE) * (1 - 2 * POISSON_VALUE))

DT_VALUE = 0.01
DHAT = 1e-6
SQRT_DHAT = np.sqrt(DHAT)
KAPPA = 100000

#########################################################
# Load cages
#########################################################
cage_npz = np.load("weights.npz")
cage_points = cage_npz['grid_points_pos'].astype(np.float64)
cages = cage_npz['boxes_grid_point_indices'].astype(np.uint32)
v2g = cage_npz['vertex_grid_point_indices'].astype(np.uint32)
v2w = cage_npz['vertex_weights'].astype(np.float64)

cage_points = cage_points / 100.0
cage_points[:, 1] += 1.0

x_max = max(cage_points[:, 0])
y_max = max(cage_points[:, 1])
z_max = max(cage_points[:, 2])
x_min = min(cage_points[:, 0])
y_min = min(cage_points[:, 1])
z_min = min(cage_points[:, 2])


# print("Cage bounding box:")
# print("x: ", x_min, x_max)
# print("y: ", y_min, y_max)
# print("z: ", z_min, z_max)

tet6 = np.array([
  [0, 1, 3, 7],
  [0, 3, 2, 7],
  [0, 2, 6, 7],
  [0, 6, 4, 7],
  [0, 4, 5, 7],
  [0, 5, 1, 7],
], dtype=np.uint32)
cage_tets = cages[:, tet6].reshape(-1, 4)   # (Nb*6, 4)
tet6_edges = np.array([
  [0, 1], [1, 3], [3, 7], [7, 0],
  [0, 3], [3, 2], [2, 7], [7, 0],
  [0, 2], [2, 6], [6, 7], [7, 0],
  [0, 6], [6, 4], [4, 7], [7, 0],
  [0, 4], [4, 5], [5, 7], [7, 0],
  [0, 5], [5, 1], [1, 7], [7, 0],
], dtype=np.int32)
cage_edges = cages[:, tet6_edges].reshape(-1, 2)   # (Nb*6*12, 2)

#########################################################
# Load bunny mesh
#########################################################
bunny_faces = []
bunny_vertices = []
f = open("../data/bunny_small.obj", 'r')
for line in f:
  if line.startswith('v '):
    bunny_vertices.append([float(x) for x in line.strip().split()[1:]])
  if line.startswith('f '):
    bunny_faces.append([int(x.split('//')[0]) - 1 for x in line.strip().split()[1:]])

bunny_faces = np.array(bunny_faces, dtype=np.uint32)
bunny_vertices = np.array(bunny_vertices, dtype=np.float64)

bunny_vertices = bunny_vertices / 100.0
bunny_vertices[:, 1] += 1.0

x_max = max(bunny_vertices[:, 0])
y_max = max(bunny_vertices[:, 1])
z_max = max(bunny_vertices[:, 2])
x_min = min(bunny_vertices[:, 0])
y_min = min(bunny_vertices[:, 1])
z_min = min(bunny_vertices[:, 2])
# print("Bunny bounding box:")
# print("x: ", x_min, x_max)
# print("y: ", y_min, y_max)
# print("z: ", z_min, z_max)

#########################################################
# Create the scene
#########################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA])

#########################################################
# Add the cage mesh
#########################################################
cage = s0.addMesh("cage")
mu = cage.addConstant("mu", rows = 1, cols = 1)
mu.updateValue([MU_LAME_VALUE])
lam = cage.addConstant("lambda", rows = 1, cols = 1)
lam.updateValue([LAMBDA_LAME_VALUE])

# add the vertices
cv = cage.addPrimitive("vertices", numInstances = cage_points.shape[0])
cvp = cv.addAttribute("position", rows = 3, cols = 1)
cvp.updateValue(cage_points.flatten())
cvlp = cv.addConstant("last_position", rows = 3, cols = 1)
cvlp.updateValue(cage_points.flatten())
cvrp = cv.addConstant("rest_position", rows = 3, cols = 1)
cvrp.updateValue(cage_points.flatten())
cvm = cv.addConstant("mass", rows = 1, cols = 1)
cvm.updateValue(np.ones(cage_points.shape[0], dtype=np.float64) * 0.01)
cvv = cv.addConstant("velocity", rows = 3, cols = 1)
cvv.updateValue(np.zeros((cage_points.shape[0], 3), dtype=np.float64))

# add the tetrahedra
ct = cage.addPrimitive("tets", numInstances = cage_tets.shape[0])
ct2cv = ct.addConnectivity("ct2cv", cv, cage_tets, 4)
ctp = ct.addAttribute("positions", through = ct2cv, source = cvp)
ctrp = ct.addAttribute("rest_positions", through = ct2cv, source = cvrp)

#########################################################
# Add bunny vertices
#########################################################
bv = cage.addPrimitive("bunny_vertices", numInstances = bunny_vertices.shape[0])
bv2cv = bv.addConnectivity("bv2cv", cv, v2g.flatten(), 8)
bvcvp = bv.addAttribute("cage_points_positions", through = bv2cv, source = cvp)
bvw = bv.addConstant("weights", rows = 8, cols = 1)
bvw.updateValue(v2w.flatten())

computed_position = bvcvp.row(0) * bvw.row(0) + \
                    bvcvp.row(1) * bvw.row(1) + \
                    bvcvp.row(2) * bvw.row(2) + \
                    bvcvp.row(3) * bvw.row(3) + \
                    bvcvp.row(4) * bvw.row(4) + \
                    bvcvp.row(5) * bvw.row(5) + \
                    bvcvp.row(6) * bvw.row(6) + \
                    bvcvp.row(7) * bvw.row(7)

bvp = bv.addAttribute("position", computed_attribute = computed_position)

# now we can add the surface triangles
bt = cage.addPrimitive("bunny_triangles", numInstances = bunny_faces.shape[0])
bt2bv = bt.addConnectivity("bt2bv", bv, bunny_faces, 3)
btp = bt.addAttribute("positions", through = bt2bv, source = bvp)
# print(btp.compute().value.get())

#########################################################
# Ground collision points as dynamic connectivity
#########################################################
bv_ground = cage.addPrimitive("bunny_ground_collision_points", numInstances = 0, isDynamic = True)
bv_g2bv = bv_ground.addConnectivity("bv_g2bv", bv, [], 1)
bv_ground_position = bv_ground.addAttribute("position", through = bv_g2bv, source = bvp)

#########################################################
# Let's add some energies
#########################################################
from helpers import stable_neo_hookean, inertia, floor_collision
snh = stable_neo_hookean(ctrp, ctp, mu, lam, dt)
ct.addAttribute("stable_neo_hookean_energy", computed_attribute = snh)

inertia_energy = inertia(cvlp, cvv, dt, cvp, cvm)
cv.addAttribute("inertia_energy", computed_attribute = inertia_energy)

floor_collision_energy = floor_collision(bv_ground_position, dhat, kappa)
bv_ground.addAttribute("floor_collision_energy", computed_attribute = floor_collision_energy)

s0.addEnergy(snh, projection_method = 1)
s0.addEnergy(inertia_energy, projection_method = 0)
s0.addEnergy(floor_collision_energy, dynamic_instances = True, projection_method = 2)
s0.addMinimizeTarget([cvp])
# exit()

import pycuda.gpuarray as gpuarray
def compute_total_energy():
  total_energy = 0.0
  total_energy += gpuarray.sum(snh.compute().value).get()
  total_energy += gpuarray.sum(inertia_energy.compute().value).get()
  if floor_collision_energy.correspondance.numInstances > 0:
    total_energy += gpuarray.sum(floor_collision_energy.compute().value).get()
  return total_energy

#########################################################
# Visualization
#########################################################
import pyvista as pv
plotter = pv.Plotter(window_size=[3840, 2160])
bunny_poly = pv.PolyData(bunny_vertices, np.hstack((np.full((bunny_faces.shape[0], 1), 3), bunny_faces)).astype(np.uint32))
# make it transparent
plotter.add_mesh(bunny_poly, color='blue', opacity=1)

# plot cages
cage_lines = np.hstack([np.full((cage_edges.shape[0], 1), 2), cage_edges])
cage_poly = pv.PolyData(cage_points, lines = cage_lines)
plotter.add_mesh(cage_poly, color='red', line_width=1, opacity = 0.1)

plotter.camera_position = [(0, 2, 6),
 (0.0, 0.0, 0.0),
 (0, 1, 0)
]

floor = pv.Plane(
  center=(0, 0, 0),
  direction=(0, 1, 0),   # plane normal; (0,1,0) => horizontal XZ floor at y=0
  i_size=10,
  j_size=10,
  i_resolution=1,
  j_resolution=1,
)

plotter.add_mesh(floor, color="white", opacity=1.0)

plotter.show(interactive_update=True)

#########################################################
# Simulation
#########################################################
cvp_copy = cvp.compute().value.copy()
for i in range(200):
  cvlp.updateValue(cvp.value, deepCopy = True)
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-6)
    dcvp = result[0]
    energies_before = compute_total_energy()
    print(f"Iteration {i}, energy before: {energies_before}")
    # copy current cell vertex positions
    cvp_copy.set(cvp.value.copy())

    substep = 1
    step_taken = 1.0
    # we first compute how much we can progress
    current_y = bvp.compute().value.get().reshape(-1, 3)[:, 1]
    cvp.updateValue(cvp_copy - dcvp * 1.0)
    new_y = bvp.compute().value.get().reshape(-1, 3)[:, 1]
    dcvp_y = current_y - new_y

    distance_check = np.minimum(SQRT_DHAT * 0.8, current_y * 0.8)
    allowed_step = (current_y - distance_check) / dcvp_y
    close_point_indices = np.where(allowed_step > 0.0)[0]
    # print(close_point_indices)
    if len(close_point_indices) == 0:
      step_taken = 1.0
    else:
      step_taken = min(1.0, (allowed_step[close_point_indices]).min())
      # print(f"  Adjusted step taken to {step_taken} due to ground collision")
      # print("  y positions for 238: ", current_y[238])
      # print("  step taken: ", step_taken)
      # print("  y velocity for 238", dcvp_y[238])
    while substep <=4:
      cvp.updateValue(cvp_copy - dcvp * step_taken)
      distance_to_ground = bvp.compute().value.get().reshape(-1, 3)[:, 1]
      vertex_close_to_ground = np.where(distance_to_ground <= SQRT_DHAT)[0]
      vertex_under_ground = np.where(distance_to_ground < 0.0)[0]
      if len(vertex_under_ground) > 0:
        print("  Warning: vertex under ground detected at indices ", vertex_under_ground)
        exit()
      # print(vertex_close_to_ground)
      # # now update the indices
      bv_ground.updateNumInstances(len(vertex_close_to_ground))
      if len(vertex_close_to_ground) > 0:
        # print(np.array(vertex_close_to_ground, dtype = np.uint32))
        bv_g2bv.updateConnectivity(np.array(vertex_close_to_ground, dtype = np.uint32).reshape(-1, 1))
        # print("y pos for those close to ground: ", distance_to_ground[vertex_close_to_ground])

      energies_after = compute_total_energy()
      print(f"Iteration {i}, substep {substep}, step taken {step_taken}, energy before: {energies_before}, energy after: {energies_after}")
      if energies_after < energies_before:
        break
      substep += 1
      step_taken *= 0.5

    # check movement, if small enough go to next iteration
    max_movement = gpuarray.max(abs(dcvp)).get() / DT_VALUE
    if max_movement < 1e-4:
      break

    cage_poly.points = cvp.compute().value.get().reshape(-1, 3)
    bunny_poly.points = bvp.compute().value.get().reshape(-1, 3)

    plotter.render()
    plotter.update()

  # update velocities
  velocity = (cvp.value.get() - cvlp.value.get()) / DT_VALUE
  cvv.updateValue(velocity)
  cage_poly.points = cvp.compute().value.get().reshape(-1, 3)
  bunny_poly.points = bvp.compute().value.get().reshape(-1, 3)
  plotter.screenshot(f"outputs/cage_{i:04d}.jpg")

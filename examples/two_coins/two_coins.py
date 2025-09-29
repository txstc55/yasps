import numpy as np
NUM_CIRCLE_POINTS = 60
NUM_CIRCLES = 2
DT_VALUE = 0.01
DHAT_VALUE = 1e-3 # for collision detection
KAPPA_VALUE = 10000000000.0 # for collision
AFFINE_PENALTY = 100
THICKNESS = 0.3
ELASTICITY_PENALTY = 1.0
##############################################################
# first we construct the points on the edge of circle
##############################################################
v_circle = []
f_circle_base = []
f_circle_computed = []
e_circle = []
for j in range(NUM_CIRCLES):
  for i in range(NUM_CIRCLE_POINTS):
    theta = 2.0 * np.pi * float(i) / float(NUM_CIRCLE_POINTS)
    v_circle.append([np.cos(theta), 0.0, np.sin(theta)])
    f_circle_base.append([(i + 1) % NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1), NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1), i + j * (NUM_CIRCLE_POINTS + 1)]) # append the face indices
  v_circle.append([0.0, 0.0, 0.0])
v_circle = np.array(v_circle, dtype = np.float64)
f_circle_base = np.array(f_circle_base, dtype=np.uint32)

# now the we construct the face whose vertices' positions are computed
for j in range(NUM_CIRCLES):
  for i in range(NUM_CIRCLE_POINTS):
    f_circle_computed.append([i + j * (NUM_CIRCLE_POINTS + 1), NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1), (i + 1) % NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1)]) # append the face indices
f_circle_computed = np.array(f_circle_computed, dtype=np.uint32) + NUM_CIRCLES * (NUM_CIRCLE_POINTS + 1)

# we now have one circle, but we need to make it two circles for a coin
# we need to add the rims
f_rim = []
for j in range(NUM_CIRCLES):
  for i in range(NUM_CIRCLE_POINTS):
    f_rim.append([i + j * (NUM_CIRCLE_POINTS + 1), i + j * (NUM_CIRCLE_POINTS + 1) + NUM_CIRCLES * (NUM_CIRCLE_POINTS + 1), (i + 1) % NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1) + NUM_CIRCLES * (NUM_CIRCLE_POINTS + 1)])
    f_rim.append([(i + 1) % NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1) + NUM_CIRCLES * (NUM_CIRCLE_POINTS + 1), (i + 1) % NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1), i + j * (NUM_CIRCLE_POINTS + 1)])
f_rim = np.array(f_rim, dtype=np.uint32)


# finally let's stack all the faces together
f_circle = np.vstack([
  f_circle_base,
  f_circle_computed,
  f_rim
])


##############################################################
# now we construct the scene
##############################################################
from yasps import scene
from yasps import attribute
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
coins = s0.addMesh("coins")
thickness = coins.addConstant("thickness", rows = 1, cols = 1)
coins["thickness"].updateValue([THICKNESS])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA_VALUE])
affine_penalty = coins.addConstant("affine_penalty", rows = 1, cols = 1)
affine_penalty.updateValue([AFFINE_PENALTY])
elasticity_penalty = coins.addConstant("elasticity_penalty", rows = 1, cols = 1)
elasticity_penalty.updateValue([ELASTICITY_PENALTY])

# add the affine bodies to the mesh
ca = coins.addPrimitive("affine_bodies", numInstances = NUM_CIRCLES)


caa = ca.addAttribute("affine_matrix", rows = 3, cols = 3)
caa.updateValue(
  np.vstack([
    np.eye(3, dtype = np.float64),
    np.eye(3, dtype = np.float64)
  ])
)
cat = ca.addAttribute("translation", rows = 3, cols = 1)
cat.updateValue(
  np.vstack([
    np.array([-0.7, -1.5, 0.0], dtype = np.float64),
    np.array([0.7, 1.5, 0.0], dtype = np.float64)
  ])
)

ca_temp_v0 = ca.addConstant("tmp_vertices_0", rows = 3, cols = 1)
ca_temp_v1 = ca.addConstant("tmp_vertices_1", rows = 3, cols = 1)
ca_temp_v2 = ca.addConstant("tmp_vertices_2", rows = 3, cols = 1)
ca_temp_v0.updateValue([0, 0, 1.155, 0, 0, 1.155])
ca_temp_v1.updateValue([-1, 0, -0.577, -1, 0, -0.577])
ca_temp_v2.updateValue([1, 0, -0.577, 1, 0, -0.577])



# add the vertices of one face
cv = coins.addPrimitive("vertices", numInstances = v_circle.shape[0])
cvrp = cv.addConstant("rest_position", rows = 3, cols = 1)
cvrp.updateValue(
  v_circle
)
cv2ca = cv.addConnectivity("vertex_to_affine_body", ca, [0] * (v_circle.shape[0] // NUM_CIRCLES) + [1] * (v_circle.shape[0] // NUM_CIRCLES), 1)
cva = cv.addAttribute("affine_matrix", through = cv2ca, source = caa) # grab the affine matrix and translation from the affine bodies
cva = cva.resize(3, 3)
cvt = cv.addAttribute("translation", through = cv2ca, source = cat)
cvt = cvt.resize(3, 1)

cvp = cv.addAttribute("position", computed_attribute = cva * cvrp + cvt)
cvlp = cv.addConstant("last_position", rows = 3, cols = 1) # update the last position
cvlp.updateValue(cvp.compute().value.get())
cvv = cv.addConstant("velocity", rows = 3, cols = 1) # update the velocity
velocities = [0, 10, 0] * (v_circle.shape[0] // NUM_CIRCLES) + [0, 0, 0] * (v_circle.shape[0] // NUM_CIRCLES)
cvv.updateValue(velocities)
cvm = cv.addConstant("mass", rows = 1, cols = 1) # update the mass
cvm.updateValue(np.ones(v_circle.shape[0], dtype = np.float64) * 1.0)


# ok now we create the vertices of the other face
ccv = coins.addPrimitive("computed_vertices", numInstances = v_circle.shape[0])
# each computed vertex has a 1 to 1 mapping to the original vertices
ccv2cv = ccv.addConnectivity("computed_vertices_to_vertices", cv, np.arange(v_circle.shape[0], dtype = np.uint32), 1)
ccv2cvrp = ccv.addAttribute("corresponding_rest_position", through = ccv2cv, source = cvrp)
ccv2cvrp = ccv2cvrp.resize(3, 1)
# now we also need to get th connectivity to the affine bodies
ccv2ca = ccv.addConnectivity("computed_vertices_to_affine_bodies", ca, [0] * (v_circle.shape[0] // NUM_CIRCLES) + [1] * (v_circle.shape[0] // NUM_CIRCLES), 1)
ccva = ccv.addAttribute("affine_matrix", through = ccv2ca, source = caa) # grab the affine matrix and translation from the affine bodies
ccva = ccva.resize(3, 3)
ccvt = ccv.addAttribute("translation", through = ccv2ca, source = cat)
ccvt = ccvt.resize(3, 1)

ccv_temp_v0 = ccv.addAttribute("tmp_vertices_0", through = ccv2ca, source = ca_temp_v0)
ccv_temp_v0 = ccv_temp_v0.resize(3, 1)
ccv_temp_v1 = ccv.addAttribute("tmp_vertices_1", through = ccv2ca, source = ca_temp_v1)
ccv_temp_v1 = ccv_temp_v1.resize(3, 1)
ccv_temp_v2 = ccv.addAttribute("tmp_vertices_2", through = ccv2ca, source = ca_temp_v2)
ccv_temp_v2 = ccv_temp_v2.resize(3, 1)
ccv_rotated_v0 = ccva * ccv_temp_v0 + ccvt
ccv_rotated_v1 = ccva * ccv_temp_v1 + ccvt
ccv_rotated_v2 = ccva * ccv_temp_v2 + ccvt
ccv_normal = (ccv_rotated_v1 - ccv_rotated_v0).cross(ccv_rotated_v2 - ccv_rotated_v0)
ccv_normal = ccv_normal / ccv_normal.norm()

# now, the computed position of those vertices are the corresponding position in the original vertices
# plus a thickness along the normal direction
# then do the affine transformation
ccvp = ccv.addAttribute("position", computed_attribute = ccva * (ccv2cvrp) + thickness * ccv_normal + ccvt)
ccvlp = ccv.addConstant("last_position", rows = 3, cols = 1)
ccvlp.updateValue(ccvp.compute().value.get())
ccvv = ccv.addConstant("velocity", rows = 3, cols = 1)
ccvv.updateValue(velocities)
ccvm = ccv.addConstant("mass", rows = 1, cols = 1)
ccvm.updateValue(np.ones(v_circle.shape[0], dtype = np.float64) * 1.0)

# now, we declare a new primitive union called vertices union, which contains both the original vertices and the computed vertices
cvu = coins.addPrimitiveUnion("vertices_union", [cv, ccv])
cvup = cvu.addAttribute("position")
cvulp = cvu.addAttribute("last_position")
cvuv = cvu.addAttribute("velocity")
cvum = cvu.addAttribute("mass")


# create edge primitives, we use it to force elasticity
ce = coins.addPrimitive("edges", numInstances = NUM_CIRCLES * NUM_CIRCLE_POINTS)
edges = []
for j in range(NUM_CIRCLES):
  for i in range(NUM_CIRCLE_POINTS):
    edges.append([i + j * (NUM_CIRCLE_POINTS + 1), NUM_CIRCLE_POINTS + j * (NUM_CIRCLE_POINTS + 1)])
ce2v = ce.addConnectivity("edge_to_vertices", cv, edges, 2)
cep = ce.addAttribute("positions", through = ce2v, source = cvp)

##############################################################
# now for collision, we use cvu
##############################################################
coins.addPrimitive("pp", numInstances = 0, isDynamic = True)
coins.addPrimitive("pe", numInstances = 0, isDynamic = True)
coins.addPrimitive("pt", numInstances = 0, isDynamic = True)
coins.addPrimitive("ee", numInstances = 0, isDynamic = True)

pp2v = coins.pp.addConnectivity("pp2v", coins.vertices_union, [], 2)
pe2v = coins.pe.addConnectivity("pe2v", coins.vertices_union, [], 3)
pt2v = coins.pt.addConnectivity("pt2v", coins.vertices_union, [], 4)
ee2v = coins.ee.addConnectivity("ee2v", coins.vertices_union, [], 4)
pp_positions = coins.pp.addAttribute("positions", through = pp2v, source = coins.vertices_union["position"])
pe_positions = coins.pe.addAttribute("positions", through = pe2v, source = coins.vertices_union["position"])
pt_positions = coins.pt.addAttribute("positions", through = pt2v, source = coins.vertices_union["position"])
ee_positions = coins.ee.addAttribute("positions", through = ee2v, source = coins.vertices_union["position"])


##################################################
# now we add energies
##################################################
# first we add the affine energy
from helpers import affine_energy
affine_energies = affine_penalty * affine_energy(caa)
ca.addAttribute("affine", computed_attribute = affine_energies)

# now we add inertia
from helpers import inertia
inertia_energies = inertia(cvulp, cvuv, dt, cvup, cvum)
cvu.addAttribute("inertia", computed_attribute = inertia_energies)

# then we add collision energies
from helpers import point_point, point_edge, point_triangle, edge_edge
pp = point_point(pp_positions, dhat, kappa)
pp_energy = coins.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
pe_energy = coins.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
pt_energy = coins.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
ee_energy = coins.ee.addAttribute("edge_edge", computed_attribute = ee)

# we also add an elasticity, we know each edge is length 1, so we just use that
elasticity_energy = elasticity_penalty * ((cep.row(0) - cep.row(1)).norm() - 1.0) * ((cep.row(0) - cep.row(1)).norm() - 1.0)
ce.addAttribute("elasticity", computed_attribute = elasticity_energy)

s0.addEnergy(affine_energies, projection_method = 1)
s0.addEnergy(inertia_energies, projection_method = 1)
s0.addEnergy(elasticity_energy, projection_method = 1)
s0.addEnergy(pp_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pe_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(pt_energy, dynamic_instances = True, projection_method = 1)
s0.addEnergy(ee_energy, dynamic_instances = True, projection_method = 1)

s0.addMinimizeTarget([caa, cat])

##################################################
# construct ccd
##################################################
from helpers import extract_edges_from_triangles
f_coins = f_circle
v_coins = cvup.compute().value.get().reshape(-1, 3)
e_coins = extract_edges_from_triangles(f_coins.tolist())
mesh_indices = [1] * (NUM_CIRCLE_POINTS + 1) + [2] * (NUM_CIRCLE_POINTS + 1) + [1] * (NUM_CIRCLE_POINTS + 1) + [2] * (NUM_CIRCLE_POINTS + 1)
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray


ccd = CCD(v_coins.shape[0], v_coins.shape[0], max_cd_pairs = 10000000, max_ccd_pairs = 10000000, mesh_indices = mesh_indices)
triangle_indices_gpu = gpuarray.to_gpu(f_coins)
edge_indices_gpu = gpuarray.to_gpu(np.array(e_coins, dtype = np.uint32))
surface_indices_gpu = gpuarray.to_gpu(np.arange(v_coins.shape[0], dtype = np.uint32))

ccd.init_faces(cvup.compute().value, triangle_indices_gpu, surface_indices_gpu, f_coins.shape[0])
ccd.init_edges(cvup.compute().value, cvup.compute().value, edge_indices_gpu, e_coins.shape[0])


##################################################
# do some plotting for the coins
##################################################
import pyvista as pv
from helpers import abs_max_reduce
coin_faces = np.array(f_circle)
coin_cells = np.hstack([np.full((f_circle.shape[0], 1), 3), f_circle])

coin_vertices_positions = cvup.compute().value.get().reshape(-1, 3)
coin_poly = pv.PolyData(coin_vertices_positions, coin_cells)

plotter = pv.Plotter()
plotter.add_mesh(
  coin_poly,
  color=[192/255, 192/255, 192/255],  # silver RGB
  opacity=1.0,
  specular=1.0,
  specular_power=15,
)

plotter.camera_position = [
  (4.326129750797788, 10.90376370143946, 16.19853507334642),  # same position
  (0.0, 0.0, 0.0),                                            # same look-at
  (0.0, 1.0, 0.0)                                             # force Y up
]
plotter.camera.SetClippingRange(0.1, 200.0)
plotter.show(interactive_update = True)


position_copy = cvup.compute().value.copy()
rot_copy = gpuarray.zeros(9 * NUM_CIRCLES, dtype=np.float64)
trans_copy = gpuarray.zeros(3 * NUM_CIRCLES, dtype=np.float64)
direction_copy = gpuarray.zeros_like(position_copy)
for i in range(250):
  cvlp.updateValue(cvp.compute().value, deepCopy = True)
  ccvlp.updateValue(ccvp.compute().value, deepCopy = True)
  inner_iteration = 0
  min_inner_iteration_energy = 100000000
  while True:
    result = s0.minimizeEnergy(tolerance = 1e-9, maxIterations = 100000)
    gradient_gpu = s0.gradient
    max_grad = abs_max_reduce(gradient_gpu).get()  # only one scalar transfer

    affine_energy_sum = sum(affine_energies.compute().value.get())
    inertia_energy_sum = sum(inertia_energies.compute().value.get())
    elasticity_energy_sum = sum(elasticity_energy.compute().value.get())
    pp_energy_sum = sum(pp_energy.compute().value.get())
    pe_energy_sum = sum(pe_energy.compute().value.get())
    pt_energy_sum = sum(pt_energy.compute().value.get())
    ee_energy_sum = sum(ee_energy.compute().value.get())
    energies_before = affine_energy_sum + inertia_energy_sum + elasticity_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
    if energies_before < min_inner_iteration_energy:
      min_inner_iteration_energy = energies_before
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"energy before {energies_before} vs minimum energy in newton: {min_inner_iteration_energy}")
    print(f"max gradient at outer iteration {i}, inner iteration {inner_iteration} is {max_grad}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # we perform CCD here
    # first we get the rotation and translation
    d_rot = result[0]
    d_trans = result[1]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(cvup.compute().value)
    rot_copy.set(caa.value)
    trans_copy.set(cat.value)

    # we first compute the new position
    caa.updateValue(caa.value - d_rot, deepCopy = True)
    cat.updateValue(cat.value - d_trans, deepCopy = True)

    new_positions = cvup.compute().value
    # now we compute the new direction, remember it's the negative we need to put in
    direction_copy = position_copy - new_positions


    # check for the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    # largest_step = 1.0
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    while substep <= 16:
      caa.updateValue(rot_copy - d_rot * step_taken, deepCopy = True)
      cat.updateValue(trans_copy - d_trans * step_taken, deepCopy = True)

      # perform collision detection
      ccd.cd(cvup.compute().value, DHAT_VALUE) # perform collision detection
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      print("The separated counts are", ccd.separated_counts)
      coins.pp.updateNumInstances(pp_count)
      coins.pe.updateNumInstances(pe_count)
      coins.pt.updateNumInstances(pt_count)
      coins.ee.updateNumInstances(ee_count)
      if pp_count > 0:
        pp2v.updateConnectivity(ccd.pp[:2 * pp_count])
      if pe_count > 0:
        pe2v.updateConnectivity(ccd.pe[:3 * pe_count])
      if pt_count > 0:
        pt2v.updateConnectivity(ccd.pt[:4 * pt_count])
      if ee_count > 0:
        ee2v.updateConnectivity(ccd.ee[:4 * ee_count])

      affine_energy_sum = sum(affine_energies.compute().value.get())
      inertia_energy_sum = sum(inertia_energies.compute().value.get())
      elasticity_energy_sum = sum(elasticity_energy.compute().value.get())
      pp_energy_sum = sum(pp_energy.compute().value.get())
      pe_energy_sum = sum(pe_energy.compute().value.get())
      pt_energy_sum = sum(pt_energy.compute().value.get())
      ee_energy_sum = sum(ee_energy.compute().value.get())
      new_energies = affine_energy_sum + inertia_energy_sum + elasticity_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum
      # new_energies = inertia_energy_sum + pp_energy_sum + pe_energy_sum + pt_energy_sum + ee_energy_sum + affine_energy_sum
      print(f"energy comparison: {new_energies} vs {energies_before}")
      if new_energies <= energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    # if substep > 8:
    #   print("failed")
    #   exit(1)
    print("step taken is", step_taken)
    print("substep is", substep)
    coin_poly.points = cvup.compute().value.get().reshape(-1, 3)
    plotter.render()
    plotter.update()
    print(plotter.camera_position)

    # print(f"Iteration {inner_iteration} max gradient: {max_grad}")
    if max_grad < 2e-1:
      print(f"Iteration {inner_iteration} exited with max gradient: {max_grad}")
      break
    inner_iteration += 1
  new_velocities0 = (cvp.compute().value - cvlp.value) / DT_VALUE
  new_velocities1 = (ccvp.compute().value - ccvlp.value) / DT_VALUE
  cvv.updateValue(new_velocities0, deepCopy = True)
  ccvv.updateValue(new_velocities1, deepCopy = True)
  coin_poly.points = cvup.compute().value.get().reshape(-1, 3)
  plotter.render()
  plotter.update()
  plotter.screenshot(f"outputs/two_coins_{i:04d}.jpg")

from yasps import scene
from yasps import attribute
from helpers import extract_surface_triangles, inertia, extract_edges_from_triangles
import math
import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
import random
random.seed(1313)
np.random.seed(13)      # for numpy

import pycuda.driver as cuda

def print_mem(tag):
  free, total = cuda.mem_get_info()
  print(f"{tag} | Free: {free/1e6:.2f} MB / Total: {total/1e6:.2f} MB")
  return free, total



SHOW_RENDER = True

DT_VALUE = 0.01 # for time step
DHAT_VALUE = 1e-5 # for collision detection
KAPPA_VALUE = 1000.0 # for collision

BENDING_STIFFNESS = 0.05
STRETCH_STIFFNESS = 35500007.469799
SHEAR_STIFFNESS = 10000067.114094
THICKNESS = 0.001
FRAME = 0
MOVING_WEIGHT = 100000.0

mem_start, _ = print_mem("Start")

##################################################################
## We will now construct the cloth
##################################################################
from helpers import generate_cloth_mesh
CLOTH_LENGTH = 4.0
NUM_VERTICES = 800
DHAT_VALUE = (CLOTH_LENGTH / NUM_VERTICES) * (CLOTH_LENGTH / NUM_VERTICES) * 0.25
NUM_SEGMENTS = NUM_VERTICES - 1
positions_cloth, triangle_indices_cloth = generate_cloth_mesh(CLOTH_LENGTH, NUM_SEGMENTS)
print(positions_cloth.shape)
# positions_cloth = positions_cloth + np.array([0.0, -0.2, 0.0])
from helpers import generate_edge_to_vertices_list
edge_to_vertices = generate_edge_to_vertices_list(triangle_indices_cloth)
edge_indices_cloth = extract_edges_from_triangles(triangle_indices_cloth)
# corners = [
#   0,
#   NUM_SEGMENTS,
#   NUM_SEGMENTS * (NUM_SEGMENTS + 1),
#   (NUM_SEGMENTS + 1) * (NUM_SEGMENTS + 1) - 1
# ]
corners = list(range(NUM_VERTICES)) + list(range(NUM_VERTICES * (NUM_VERTICES - 1), (NUM_VERTICES) * (NUM_VERTICES)))

# print(positions_cloth[corners])
# exit()

##################################################################
## construct the abd meshes
##################################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue(DT_VALUE)
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue(DHAT_VALUE)
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue(KAPPA_VALUE)

##################################################################
## construct the cloth mesh
##################################################################
cloth = s0.addMesh("cloth")
stretch = cloth.addConstant("stretch", rows = 1, cols = 1)
stretch.updateValue(STRETCH_STIFFNESS)
shear = cloth.addConstant("shear", rows = 1, cols = 1)
shear.updateValue(SHEAR_STIFFNESS)
thickness = cloth.addConstant("thickness", rows = 1, cols = 1)
thickness.updateValue(THICKNESS)
bending_stiffness = cloth.addConstant("bending_stiffness", rows = 1, cols = 1)
bending_stiffness.updateValue(BENDING_STIFFNESS)
moving_weight = cloth.addConstant("moving_weight", rows = 1, cols = 1)
moving_weight.updateValue(MOVING_WEIGHT)



vertices = cloth.addPrimitive("vertices", numInstances = positions_cloth.shape[0])
vertices_position = vertices.addAttribute("position", rows = 3, cols = 1)
vertices_rest_position = vertices.addConstant("rest_position", rows = 3, cols = 1)
vertices_last_position = vertices.addConstant("last_position", rows = 3, cols = 1)
vertices_velocity = vertices.addConstant("velocity", rows = 3, cols = 1)
vertices_mass = vertices.addConstant("mass", rows = 1, cols = 1)
# update the values
vertices_position.updateValue(positions_cloth.flatten())
vertices_rest_position.updateValue(positions_cloth.flatten())
vertices_last_position.updateValue(positions_cloth.flatten())
vertices_velocity.updateValue(np.zeros((positions_cloth.shape[0]) * 3, dtype = np.float64))
vertices_mass.updateValue(np.full((positions_cloth.shape[0]), 1.0 / positions_cloth.shape[0], dtype = np.float64))

vertices_controlled = cloth.addPrimitive("vertices_controlled", numInstances = len(corners))
v2v = vertices_controlled.addConnectivity("v2v", vertices, corners, 1)
vertices_controlled_position = vertices_controlled.addAttribute("position", through = v2v, source = vertices_position)
vertices_target_position = vertices_controlled.addConstant("target_position", rows = 1, cols = 3)

def compute_target(frame):
  scaling = 20.0
  theta = frame / scaling

  c = np.cos(theta)
  s = np.sin(theta)
  Rz = np.array([
      [c, -s, 0],
      [s,  c, 0],
      [0,  0, 1]
  ], dtype=np.float64)
  Rz_neg = np.array([
      [c,  s, 0],
      [-s, c, 0],
      [0,  0, 1]
  ], dtype=np.float64)

  half = len(corners) // 2
  pts = positions_cloth[corners]

  targets = np.empty_like(pts, dtype=np.float64)
  targets[:half] = pts[:half] @ Rz.T
  targets[half:] = pts[half:] @ Rz_neg.T

  return targets.flatten()

vertices_target_position.updateValue(compute_target(FRAME))

# create the triangles
triangles_cloth = cloth.addPrimitive("triangles_cloth", numInstances = triangle_indices_cloth.shape[0])
triangles_cloth2_vertices = triangles_cloth.addConnectivity("triangles_cloth2_vertices", vertices, triangle_indices_cloth, 3)
triangles_cloth_positions = triangles_cloth.addAttribute("positions", through = triangles_cloth2_vertices, source = vertices["position"])
triangles_cloth_rest_positions = triangles_cloth.addAttribute("rest_positions", through = triangles_cloth2_vertices, source = vertices["rest_position"])

# create the edge to 4 vertices pair
edges_cloth = cloth.addPrimitive("edges_cloth", numInstances = edge_to_vertices.shape[0])
edges_cloth2_vertices = edges_cloth.addConnectivity("edges_cloth2_vertices", vertices, edge_to_vertices, 4)
edges_cloth_positions = edges_cloth.addAttribute("positions", through = edges_cloth2_vertices, source = vertices["position"])
edges_cloth_rest_positions = edges_cloth.addAttribute("rest_positions", through = edges_cloth2_vertices, source = vertices["rest_position"])

cloth.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
cloth.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
cloth.addPrimitive("pt", numInstances = 0, isDynamic = True) # # for point triangle collision
cloth.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision
pp2v = cloth.pp.addConnectivity("pp2v", cloth.vertices, [], 2)
pe2v = cloth.pe.addConnectivity("pe2v", cloth.vertices, [], 3)
pt2v = cloth.pt.addConnectivity("pt2v", cloth.vertices, [], 4)
ee2v = cloth.ee.addConnectivity("ee2v", cloth.vertices, [], 4)
pp_positions = cloth.pp.addAttribute("positions", through = pp2v, source = cloth.vertices["position"])
pe_positions = cloth.pe.addAttribute("positions", through = pe2v, source = cloth.vertices["position"])
pt_positions = cloth.pt.addAttribute("positions", through = pt2v, source = cloth.vertices["position"])
ee_positions = cloth.ee.addAttribute("positions", through = ee2v, source = cloth.vertices["position"])

##################################################################
## ok now we add energies into the scene
##################################################################
from helpers import inertia
inertia_free = inertia(vertices_last_position, vertices_velocity, dt, vertices_position, vertices_mass)
vertices.addAttribute("inertia_free", computed_attribute = inertia_free)


from helpers import bending
bending_energy = bending(edges_cloth_positions, edges_cloth_rest_positions, bending_stiffness)
edges_cloth.addAttribute("bending_energy", computed_attribute = bending_energy)

from helpers import baraff_witkin
baraff_witkin_energy = baraff_witkin(triangles_cloth_rest_positions, triangles_cloth_positions,  stretch, shear, thickness, dt)
triangles_cloth.addAttribute("baraff_witkin_energy", computed_attribute = baraff_witkin_energy)

from helpers import moving_energy
me = moving_energy(vertices_controlled_position, vertices_target_position, dt, moving_weight)
vertices_controlled.addAttribute("moving_energy", computed_attribute = me)

from helpers import point_point, point_edge, point_triangle, edge_edge
pp = point_point(pp_positions, dhat, kappa)
cloth.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa)
cloth.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa)
cloth.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa)
cloth.ee.addAttribute("edge_edge", computed_attribute = ee)



s0.addEnergy(inertia_free, projection_method = -1)
s0.addEnergy(bending_energy, projection_method = 2)
s0.addEnergy(baraff_witkin_energy, projection_method = 2)
s0.addEnergy(me, projection_method = -1)
s0.addEnergy(pp, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2)
s0.addMinimizeTarget([vertices_position])
##################################################################
## add ccd
##################################################################
before, _ = print_mem("Before")
mesh_indices = []
mesh_indices += [0] * positions_cloth.shape[0]
ccd = CCD(positions_cloth.shape[0], # the number of surface points
  positions_cloth.shape[0], # the number of total points
  max_ccd_pairs = 100000000,
  max_cd_pairs = 100000000,
  mesh_indices = mesh_indices
)

triangle_indices_all = []
surface_indices_all = []
triangle_indices_all.append((triangle_indices_cloth).astype(np.uint32))
surface_indices_all.append(np.array(range(positions_cloth.shape[0]), dtype = np.uint32))
triangle_indices_all = np.vstack(triangle_indices_all, dtype = np.uint32)
surface_indices_all = np.hstack(surface_indices_all).astype(np.uint32)

edge_indices_all = []
edge_indices_all.append((edge_indices_cloth).astype(np.uint32))
edge_indices_all = np.vstack(edge_indices_all, dtype = np.uint32)

surface_indices_gpu = gpuarray.to_gpu(surface_indices_all.flatten())
edge_indices_gpu = gpuarray.to_gpu(edge_indices_all.flatten())
triangle_indices_gpu = gpuarray.to_gpu(triangle_indices_all.flatten())

position_gpu = gpuarray.to_gpu(cloth.vertices["position"].value.get()) # basically copy it out

ccd.init_faces(position_gpu, triangle_indices_gpu, surface_indices_gpu, triangle_indices_all.shape[0])
ccd.init_edges(position_gpu, position_gpu, edge_indices_gpu, edge_indices_all.shape[0])
after, _ = print_mem("After")
print(f"Memory used by CCD: {(before - after) / 1e6:.2f} MB")

##################################################################
## plot the bunnies
##################################################################
import pyvista as pv
if SHOW_RENDER:
  plotter = pv.Plotter(window_size=[3840, 2160])
  all_vertices_computed = cloth.vertices["position"].compute().value.get().reshape((-1, 3))
  triangles = triangle_indices_all
  cloth_triangles = triangle_indices_cloth
  cells_cloth = np.hstack([np.full((cloth_triangles.shape[0], 1), 3), cloth_triangles])
  cloth_vertices_computed = all_vertices_computed
  cloth_poly = pv.PolyData(cloth_vertices_computed, cells_cloth)
  plotter.add_mesh(cloth_poly, color = "pink", opacity = 0.5)


  plotter.camera_position = [(0, 2, 6),
  (0.0, 0.0, 0.0),
  (0, 1, 0)
  ]
  plotter.show(interactive_update=True)

position_copy = cloth.vertices["position"].compute().value.copy()

# do a tmp one to initialize the bounding box size
ccd.cd(position_copy, DHAT_VALUE)
scene_diag_sqrt = math.sqrt(ccd.get_scene_size_faces())
print("scene_diag_sqrt", scene_diag_sqrt)


for i in range(200):
  vertices_target_position.updateValue(compute_target(i))
  vertices_last_position.updateValue(cloth.vertices["position"].value, deepCopy = True)
  inner_iteration = 0
  while True:
    print("==================================================================")
    print(f"At iteration {i}, inner iteration {inner_iteration}")
    result = s0.minimizeEnergy(tolerance = 1e-3)
    print("==================================================================")
    energies_before = s0.computeTotalEnergy()
    # we perform CCD here
    # first we get the rotation and translation
    d_pos = result[0]
    step_taken = 1.0
    # copy the current position and current affine matrix
    position_copy.set(cloth.vertices["position"].compute().value.copy())

    # now update the value
    vertices["position"].updateValue(vertices["position"].value - d_pos, deepCopy = True)

    # compute the new positions for the entire scene
    new_positions = cloth.vertices["position"].compute().value
    # now we compute the new direction, remember it's the negative we need to put in
    direction_copy = position_copy - new_positions
    max_movement = gpuarray.max(abs(direction_copy)).get()
    target_movement = 1e-2 * DT_VALUE
    print("max movement is:", max_movement, ", target movement is:", target_movement)
    if max_movement < target_movement:
      print(f"Iteration {inner_iteration} exited with max movement: {max_movement}")
      break

    # check for the largest step size we can take
    ccd.ccd(position_copy, DHAT_VALUE, direction_copy, 1.0)
    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction_copy)
    print("largest step we can take is", largest_step)
    # here we will take this step and check for the collision sets
    substep = 1
    step_taken = largest_step
    while substep <= 8:
      vertices["position"].updateValue(position_copy - d_pos * step_taken, deepCopy = True)

      # perform collision detection
      ccd.cd(cloth.vertices["position"].compute().value, DHAT_VALUE) # perform collision detection
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      print("The separated counts are", ccd.separated_counts)
      cloth.pp.updateNumInstances(pp_count)
      cloth.pe.updateNumInstances(pe_count)
      cloth.pt.updateNumInstances(pt_count)
      cloth.ee.updateNumInstances(ee_count)
      if pp_count > 0:
        pp2v.updateConnectivity(ccd.pp[:2 * pp_count])
      if pe_count > 0:
        pe2v.updateConnectivity(ccd.pe[:3 * pe_count])
      if pt_count > 0:
        pt2v.updateConnectivity(ccd.pt[:4 * pt_count])
      if ee_count > 0:
        ee2v.updateConnectivity(ccd.ee[:4 * ee_count])
      new_energies = s0.computeTotalEnergy()
      # print("==================================================================")
      # print(f"energy comparison: {new_energies} vs {energies_before}")
      # print("==================================================================")

      mem_current, _ = print_mem("Current")
      print(f"Memory used total: {(mem_start - mem_current) / 1e6:.2f} MB")

      if new_energies <= energies_before:
        break
      step_taken = step_taken / 2.0
      substep += 1
    # print("step taken is", step_taken)
    # print("substep is", substep)
    if SHOW_RENDER:
      all_vertices_computed = cloth.vertices["position"].compute().value.get().reshape((-1, 3))
      cloth_poly.points = all_vertices_computed
      plotter.render()
      plotter.update()

    inner_iteration += 1
  new_velocities_free = (vertices["position"].compute().value - vertices_last_position.value) / DT_VALUE
  vertices_velocity.updateValue(new_velocities_free, deepCopy = True)

  # plotter.render()
  # plotter.update()
  # plotter.screenshot(f"outputs/frame_{i:04d}.png")
  # save the mesh obj file
  # cloth_poly.save(f"meshes/cloth_{i:04d}.obj")

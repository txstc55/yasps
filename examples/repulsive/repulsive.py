from yasps import scene
import numpy as np
from helpers import inertia, point_point, point_edge, point_triangle, edge_edge, length_energy, smooth_energy, repulsive, radius_energy
import pycuda.gpuarray as gpuarray

DT_VALUE = 0.01
DHAT_VALUE = 15.0

TARGET_RADIUS = 100.0
NUM_LINE_POINTS = 40000
TARGET_LENGTH = 100000.0
SMOOTH_WEIGHTS = [0.1, 0.2, 0.4, 0.2, 0.1]
EDGE_LENGTH_PENALTY = 10.0
SMOOTH_PENALTY = 100.0
KAPPA = 20000000.0
ALPHA = 3.0
BETA = 6.0
REPULSIVE_WEIGHT = 0.0
RADIUS_PENALTY = 100000000000.0
MASS_SCALE = 0.000001


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
f.close()


sphere_vertices = []
f = open("../data/smoothing_result.obj", 'r')
for line in f:
  if line.startswith('v '):
    sphere_vertices.append([float(x) for x in line.strip().split()[1:]])

sphere_vertices = np.array(sphere_vertices, dtype=np.float64)
x_max = np.max(sphere_vertices[:, 0])
x_min = np.min(sphere_vertices[:, 0])
y_max = np.max(sphere_vertices[:, 1])
y_min = np.min(sphere_vertices[:, 1])
z_max = np.max(sphere_vertices[:, 2])
z_min = np.min(sphere_vertices[:, 2])
center = np.array([(x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0])
sphere_vertices -= center

for i in range(sphere_vertices.shape[0]):
  direction = sphere_vertices[i, :] / np.linalg.norm(sphere_vertices[i, :])
  sphere_vertices[i, :] = direction * TARGET_RADIUS
print(sphere_vertices)

#####################################################
# Compute the weight for each vertex
#####################################################
mass = np.zeros(bunny_vertices.shape[0], dtype=np.float64)
for i in range(bunny_faces.shape[0]):
  v0 = bunny_vertices[bunny_faces[i, 0], :]
  v1 = bunny_vertices[bunny_faces[i, 1], :]
  v2 = bunny_vertices[bunny_faces[i, 2], :]
  area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
  mass[bunny_faces[i, 0]] += area / 3.0
  mass[bunny_faces[i, 1]] += area / 3.0
  mass[bunny_faces[i, 2]] += area / 3.0

mass_on_sphere = np.zeros(sphere_vertices.shape[0], dtype=np.float64)
for i in range(bunny_faces.shape[0]):
  v0 = sphere_vertices[bunny_faces[i, 0], :]
  v1 = sphere_vertices[bunny_faces[i, 1], :]
  v2 = sphere_vertices[bunny_faces[i, 2], :]
  area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
  mass_on_sphere[bunny_faces[i, 0]] += area / 3.0
  mass_on_sphere[bunny_faces[i, 1]] += area / 3.0
  mass_on_sphere[bunny_faces[i, 2]] += area / 3.0

fraction = mass / mass_on_sphere

mass = fraction / np.max(fraction)
mass = mass ** 0.1

print("Total mass of bunny:", np.sum(mass))
print("Max mass: ", np.max(mass))
print("Min mass: ", np.min(mass))

#####################################################
# Port the weight to a map
#####################################################
from PIL import Image

def spherical_uv(vertices):
  v = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
  x, y, z = v[:,0], v[:,1], v[:,2]

  theta = np.arctan2(z, x)              # [-pi, pi]
  phi   = np.arcsin(y)                  # [-pi/2, pi/2]

  u = (theta + np.pi) / (2*np.pi)       # [0,1]
  v = (phi + np.pi/2) / np.pi           # [0,1]
  return u, v

def weight_to_rgb(w):
  w = np.clip(w, 0.0, 1.0)
  rgb = np.stack([w, w, w], axis=-1)    # grayscale
  return (rgb * 255).astype(np.uint8)

def rasterize_triangle(p0, p1, p2, w0, w1, w2, img_w, img_h, img):
  # Bounding box
  minx = int(max(min(p0[0], p1[0], p2[0]), 0))
  maxx = int(min(max(p0[0], p1[0], p2[0]), img_w-1))
  miny = int(max(min(p0[1], p1[1], p2[1]), 0))
  maxy = int(min(max(p0[1], p1[1], p2[1]), img_h-1))

  cross_x = False

  if maxx - minx > 0.5 * img_w:
    cross_x = True
    if p0[0] < img_w * 0.5:
      p0[0] += img_w * 1.0
    if p1[0] < img_w * 0.5:
      p1[0] += img_w * 1.0
    if p2[0] < img_w * 0.5:
      p2[0] += img_w * 1.0
    minx = int(min(p0[0], p1[0], p2[0]))
    maxx = int(max(p0[0], p1[0], p2[0]))


  # Triangle edges for barycentric
  v0 = p1 - p0
  v1 = p2 - p0
  denom = v0[0]*v1[1] - v1[0]*v0[1]
  if abs(denom) < 1e-12:
    return  # Degenerate

  for y in range(miny, maxy+1):
    for x in range(minx, maxx+1):
      if x < img_w:
        p = np.array([x, y]) - p0

        # Compute barycentric coordinates
        a = (p[0]*v1[1] - v1[0]*p[1]) / denom
        b = (v0[0]*p[1] - p[0]*v0[1]) / denom
        c = 1.0 - a - b

        if a >= 0 and b >= 0 and c >= 0:
          w = a * w1 + b * w2 + c * w0
          img[y, x] = max(img[y, x], w)

  if cross_x:
    p0[0] -= img_w * 1.0
    p1[0] -= img_w * 1.0
    p2[0] -= img_w * 1.0
    minx = int(min(p0[0], p1[0], p2[0]))
    maxx = int(max(p0[0], p1[0], p2[0]))
    # Triangle edges for barycentric
    v0 = p1 - p0
    v1 = p2 - p0
    denom = v0[0]*v1[1] - v1[0]*v0[1]
    if abs(denom) < 1e-12:
      return  # Degenerate

    for y in range(miny, maxy+1):
      for x in range(minx, maxx+1):
        if x >= 0:
          p = np.array([x, y]) - p0

          # Compute barycentric coordinates
          a = (p[0]*v1[1] - v1[0]*p[1]) / denom
          b = (v0[0]*p[1] - p[0]*v0[1]) / denom
          c = 1.0 - a - b

          if a >= 0 and b >= 0 and c >= 0:
            w = a * w1 + b * w2 + c * w0
            img[y, x] = max(img[y, x], w)



def bake_sphere_uv(vertices, faces, weights, img_w=1024, img_h=1024):
  u, v = spherical_uv(vertices)

  # Convert UV → pixel coords
  px = (u * (img_w-1)).astype(np.float64)
  py = (v * (img_h-1)).astype(np.float64)

  # Working buffer of weights
  img = np.zeros((img_h, img_w), dtype=np.float64)

  # Rasterize each triangle
  for f in faces:
    i0, i1, i2 = f

    p0 = np.array([px[i0], py[i0]])
    p1 = np.array([px[i1], py[i1]])
    p2 = np.array([px[i2], py[i2]])

    w0, w1, w2 = weights[i0], weights[i1], weights[i2]

    rasterize_triangle(p0, p1, p2, w0, w1, w2, img_w, img_h, img)

  # Convert to color image
  rgb = weight_to_rgb(img)
  return rgb


# img = bake_sphere_uv(sphere_vertices, bunny_faces, mass, img_w = 5000, img_h = 5000)
# Image.fromarray(img).save("baked_weights.png")
img = Image.open("baked_weights.png").convert("L")   # "L" = 8-bit grayscale
arr = np.array(img, dtype=np.float32) / 255.0

def sample_weights_bilinear(points, weight_tex):
  """
  points:     (N,3) array of query points on sphere
  weight_tex: (H,W) array of weights in [0,1]
  returns:    (N,) array of sampled weights (bilinear)
  """
  H, W = weight_tex.shape

  u, v = spherical_uv(points)

  # continuous pixel coords
  px = u * (W - 1)
  py = v * (H - 1)

  # keep inside [0, W-1-eps], [0, H-1-eps]
  eps = 1e-6
  px = np.clip(px, 0, W - 1 - eps)
  py = np.clip(py, 0, H - 1 - eps)

  x0 = np.floor(px).astype(int)
  y0 = np.floor(py).astype(int)
  x1 = np.clip(x0 + 1, 0, W - 1)
  y1 = np.clip(y0 + 1, 0, H - 1)

  sx = px - x0  # fractional part in x
  sy = py - y0  # fractional part in y

  I00 = weight_tex[y0, x0]
  I10 = weight_tex[y0, x1]
  I01 = weight_tex[y1, x0]
  I11 = weight_tex[y1, x1]

  # bilinear interpolation
  w = (1 - sx) * (1 - sy) * I00 \
    +      sx  * (1 - sy) * I10 \
    + (1 - sx) *      sy  * I01 \
    +      sx  *      sy  * I11

  return w

weights = sample_weights_bilinear(sphere_vertices, arr)
print(weights)
print(mass)
# exit()

#####################################################
# Create lines
#####################################################
import math
import random
line_points = []
line_edges = []
line_2_neighbors = []

# Parameters controlling curvature
wave_cycles = 40        # number of wiggles around the loop
amp         = 0.8     # small deviation (~5.7 degrees)
for i in range(NUM_LINE_POINTS):
  # t in [0,1)
  t = i / NUM_LINE_POINTS
  # Base great-circle (longitude-like)
  theta = 2.0 * math.pi * (t - 0.5)  # [-pi, pi)
  # Make phi periodic: use sin(2*pi * wave_cycles * t)
  phi = amp * math.sin(2.0 * math.pi * wave_cycles * t) * math.sin(theta) * math.sin(theta + 2.5) * math.cos(theta * 4.0)
  # Spherical to Cartesian
  x = math.cos(theta) * math.cos(phi)
  y = math.sin(phi)
  z = math.sin(theta) * math.cos(phi) + random.random() * 0.0001
  # Normalize just to be safe
  norm = math.sqrt(x*x + y*y + z*z)
  x /= norm
  y /= norm
  z /= norm
  line_points.append([x * TARGET_RADIUS,
                      y * TARGET_RADIUS,
                      z * TARGET_RADIUS])
  line_edges.append([i, (i + 1) % NUM_LINE_POINTS])
  line_2_neighbors.append([(i - 2) % NUM_LINE_POINTS, (i - 1) % NUM_LINE_POINTS, i, (i + 1) % NUM_LINE_POINTS, (i + 2) % NUM_LINE_POINTS])

line_points = np.array(line_points, dtype=np.float64)
line_edges = np.array(line_edges, dtype=np.uint32)
line_2_neighbors = np.array(line_2_neighbors, dtype=np.uint32)

edge_pairs = []
for i in range(NUM_LINE_POINTS):
  current_edge_start =  i
  current_edge_end =  (i + 1) % NUM_LINE_POINTS
  for k in range(i + 2, NUM_LINE_POINTS - 2, 2000):
    next_edge_start =  k % NUM_LINE_POINTS
    next_edge_end =  (k + 1) % NUM_LINE_POINTS
    if next_edge_end == current_edge_start or next_edge_start == current_edge_end or next_edge_end == current_edge_end or next_edge_start == current_edge_start:
      break
    edge_pairs.append([current_edge_start, current_edge_end, next_edge_start, next_edge_end])
edge_pairs = np.array(edge_pairs, dtype=np.uint32)
print(f"There are {edge_pairs.shape[0]} edge pairs in the loop")

# get initial masses
line_masses = sample_weights_bilinear(line_points, arr)
line_masses = (line_masses + 0.5) ** 25 * 910.01


#####################################################
# Create the scene and line mesh
#####################################################
s0 = scene("scene0")
dt = s0.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
dhat = s0.addConstant("dhat", rows = 1, cols = 1)
dhat.updateValue([DHAT_VALUE])
kappa = s0.addConstant("kappa", rows = 1, cols = 1)
kappa.updateValue([KAPPA])


lines = s0.addMesh("lines")
target_length = s0.addConstant("target_length", rows = 1, cols = 1)
target_length.updateValue([TARGET_LENGTH])
edge_length_penalty = s0.addConstant("edge_length_penalty", rows = 1, cols = 1)
edge_length_penalty.updateValue([EDGE_LENGTH_PENALTY])
smooth_penalty = s0.addConstant("smooth_penalty", rows = 1, cols = 1)
smooth_penalty.updateValue([SMOOTH_PENALTY])
smooth_weights = s0.addConstant("smooth_weights", rows = 5, cols = 1)
smooth_weights.updateValue(np.array(SMOOTH_WEIGHTS, dtype=np.float64).flatten())
alpha = s0.addConstant("alpha", rows = 1, cols = 1)
alpha.updateValue([ALPHA])
beta = s0.addConstant("beta", rows = 1, cols = 1)
beta.updateValue([BETA])
repulsive_weight = s0.addConstant("repulsive_weight", rows = 1, cols = 1)
repulsive_weight.updateValue([REPULSIVE_WEIGHT])
radius_penalty = s0.addConstant("radius_penalty", rows = 1, cols = 1)
radius_penalty.updateValue([RADIUS_PENALTY])
mass_scale = s0.addConstant("mass_scale", rows = 1, cols = 1)
mass_scale.updateValue([MASS_SCALE])

lv = lines.addPrimitive("vertices", numInstances = NUM_LINE_POINTS)
lvp = lv.addAttribute("position", rows = 3, cols = 1)
lvp.updateValue(line_points.flatten())
lvlp = lv.addConstant("last_position", rows = 3, cols = 1)
lvlp.updateValue(line_points.flatten())
lvv = lv.addAttribute("velocity", rows = 3, cols = 1)
lvv.updateValue(np.zeros((NUM_LINE_POINTS, 3), dtype=np.float64).flatten())
lvm = lv.addAttribute("mass", rows = 1, cols = 1)
lvm.updateValue(line_masses.flatten())

#####################################################
# Create edges
#####################################################
le = lines.addPrimitive("edges", numInstances = line_edges.shape[0])
le2v = le.addConnectivity("le2v", lv, line_edges, 2)
lep = le.addAttribute("positions", through = le2v, source = lvp)
lep.resize(2, 3)
lem = le.addAttribute("masses", through = le2v, source = lvm)
lem.resize(2, 1)

#####################################################
# Create edges 2 neighbors
#####################################################
le2 = lines.addPrimitive("edge_2_neighbors", numInstances = line_2_neighbors.shape[0])
le22v = le2.addConnectivity("le22v", lv, line_2_neighbors, 5)
le2p = le2.addAttribute("positions", through = le22v, source = lvp)
le2p.resize(5, 3)

#####################################################
# Add edge pairs
#####################################################
lepair = lines.addPrimitive("edge_pairs", numInstances = edge_pairs.shape[0])
lep2v = lepair.addConnectivity("lep2v", lv, edge_pairs, 4)
lepair_positions = lepair.addAttribute("positions", through = lep2v, source = lvp)
lepair_positions.resize(4, 3)

#####################################################
# Create collision pairs
#####################################################
lines.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
lines.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
lines.addPrimitive("pt", numInstances = 0, isDynamic = True) # # for point triangle collision
lines.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision
pp2v = lines.pp.addConnectivity("pp2v", lines.vertices, [], 2)
pe2v = lines.pe.addConnectivity("pe2v", lines.vertices, [], 3)
pt2v = lines.pt.addConnectivity("pt2v", lines.vertices, [], 4)
ee2v = lines.ee.addConnectivity("ee2v", lines.vertices, [], 4)
pp_positions = lines.pp.addAttribute("positions", through = pp2v, source = lines.vertices["position"])
pe_positions = lines.pe.addAttribute("positions", through = pe2v, source = lines.vertices["position"])
pt_positions = lines.pt.addAttribute("positions", through = pt2v, source = lines.vertices["position"])
ee_positions = lines.ee.addAttribute("positions", through = ee2v, source = lines.vertices["position"])
pp_masses = lines.pp.addAttribute("masses", through = pp2v, source = lvm)
pe_masses = lines.pe.addAttribute("masses", through = pe2v, source = lvm)
pt_masses = lines.pt.addAttribute("masses", through = pt2v, source = lvm)
ee_masses = lines.ee.addAttribute("masses", through = ee2v, source = lvm)


#####################################################
# Add energies
#####################################################

# inertia
inertia_energy = inertia(lvlp, lvv, dt, lvp, lvm)
lv.addAttribute("inertia_energy", computed_attribute = inertia_energy)

# add target length
length_energy_compute = length_energy(lep, target_length, edge_length_penalty, lem, dt, mass_scale)
le.addAttribute("length_energy", computed_attribute = length_energy_compute)

# add smoothing energy
smooth_energy_compute = smooth_energy(le2p, smooth_weights, smooth_penalty, dt)
le2.addAttribute("smooth_energy", computed_attribute = smooth_energy_compute)

repulsive_energy = repulsive(lepair_positions, alpha, beta, repulsive_weight, dt)
lepair.addAttribute("repulsive_energy", computed_attribute = repulsive_energy)

# radius penalty
radius_energy_compute = radius_energy(lvp, TARGET_RADIUS, radius_penalty, dt)
lv.addAttribute("radius_energy", computed_attribute = radius_energy_compute)

# collision
pp = point_point(pp_positions, dhat, kappa, pp_masses)
lines.pp.addAttribute("point_point", computed_attribute = pp)
pe = point_edge(pe_positions, dhat, kappa, pe_masses)
lines.pe.addAttribute("point_edge", computed_attribute = pe)
pt = point_triangle(pt_positions, dhat, kappa, pt_masses)
lines.pt.addAttribute("point_triangle", computed_attribute = pt)
ee = edge_edge(ee_positions, dhat, kappa, ee_masses)
lines.ee.addAttribute("edge_edge", computed_attribute = ee)


# meet target length
s0.addEnergy(inertia_energy, projection_method = 1)
s0.addEnergy(smooth_energy_compute, projection_method = 1)
s0.addEnergy(length_energy_compute, projection_method = 1)
# s0.addEnergy(repulsive_energy, projection_method = 2)
s0.addEnergy(radius_energy_compute, projection_method = 1)
s0.addEnergy(pp, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pe, dynamic_instances = True, projection_method = 2)
s0.addEnergy(pt, dynamic_instances = True, projection_method = 2)
s0.addEnergy(ee, dynamic_instances = True, projection_method = 2)

s0.addMinimizeTarget([lvp])

import pycuda.gpuarray as gpuarray
def compute_total_energy():
  total_energy = 0.0
  total_energy += gpuarray.sum(inertia_energy.compute().value).get()
  total_energy += gpuarray.sum(length_energy_compute.compute().value).get()
  total_energy += gpuarray.sum(smooth_energy_compute.compute().value).get()
  # total_energy += gpuarray.sum(repulsive_energy.compute().value).get()
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

ccd = CCD(line_points.shape[0], # the number of surface points
  line_points.shape[0], # the number of total points
  max_cd_pairs = 10000000,
  max_ccd_pairs = 10000000,

)

position_gpu = gpuarray.to_gpu(lvp.compute().value.get()) # basically copy it out
indices_gpu = gpuarray.to_gpu(np.array(list(range(bunny_vertices.shape[0]))))
faces_gpu = gpuarray.to_gpu(np.array([], dtype=np.uint32))
edges_gpu = gpuarray.to_gpu(line_edges.flatten())

ccd.init_faces(position_gpu, faces_gpu, indices_gpu, 0)
ccd.init_edges(position_gpu, position_gpu, edges_gpu, line_edges.shape[0])

#####################################################
# Visualization
#####################################################
import pyvista as pv
# first we add bunny
plotter = pv.Plotter(window_size = [3840, 2160])

cells_loop = np.hstack([np.full((line_edges.shape[0], 1), 2), line_edges])
loop_poly = pv.PolyData(line_points, lines = cells_loop)
plotter.add_mesh(loop_poly, color='red', line_width=3)

cells_sphere = np.hstack([np.full((bunny_faces.shape[0], 1), 3), bunny_faces])
sphere_poly = pv.PolyData(sphere_vertices, cells_sphere)
plotter.add_mesh(sphere_poly, color='white', opacity=0.5)

plotter.show(interactive_update=True)
position_copy = lvp.compute().value.copy()
direction = gpuarray.to_gpu(np.zeros(line_points.flatten().shape, dtype=np.float64))

total = 0
for i in range(200):
  lvlp.updateValue(lvp.compute().value, deepCopy = True)
  inner_iteration = 0
  while True:
    lvlp.updateValue(lvp.compute().value, deepCopy = True)
    print("==================================================================")
    print(f"At iteration {i}, inner iteration {inner_iteration}")
    result = s0.minimizeEnergy(tolerance = 1e-6)
    print("==================================================================")
    energy_before = compute_total_energy()
    d_p = result[0].get().reshape(-1, 3)
    updated_value = (
      1.0 * d_p +
      0.0 * np.roll(d_p, shift=-1, axis=0) +  # Next point in the same loop
      0.0 * np.roll(d_p, shift=1, axis=0)     # Previous point in the same loop
    )


    position_copy.set(lvp.compute().value.get())
    direction.set(-d_p.flatten())

    # ccd.reset()
    # ccd.ccd_edges(position_copy, DHAT_VALUE, direction, 1.0)


    largest_step = ccd.compute_largest_step_size(0.8, position_copy, direction)
    substep = 1
    step_taken = largest_step
    while substep <= 4:
      lvp.updateValue(position_copy + step_taken * direction, deepCopy = True)
      ccd.reset()
      ccd.cd_edges(lvp.value, DHAT_VALUE)
      pp_count, pe_count, pt_count, ee_count = ccd.separated_counts
      lines.pp.updateNumInstances(pp_count)
      lines.pe.updateNumInstances(pe_count)
      lines.pt.updateNumInstances(pt_count)
      lines.ee.updateNumInstances(ee_count)
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

    loop_points = lvp.compute().value.get().reshape((-1, 3))
    updated_value = (
      0.8 * loop_points +
      0.1 * np.roll(loop_points, shift=-1, axis=0) +  # Next point in the same loop
      0.1 * np.roll(loop_points, shift=1, axis=0)     # Previous point in the same loop
    )
    # loop_points = updated_value / np.linalg.norm(updated_value, axis=1, keepdims=True) * TARGET_RADIUS
    lvp.updateValue(updated_value.flatten(), deepCopy = True)
    line_masses = sample_weights_bilinear(updated_value, arr)
    line_masses = (line_masses + 0.5) ** 25
    print(line_masses)
    lvm.updateValue(line_masses.flatten() * 910.01)
    loop_poly.points = lvp.compute().value.get().reshape((-1, 3))
    loop_poly.save(f"outputs/loop_collision_density_new_total_{total:06d}.obj")
    total += 1

    plotter.update()
    plotter.render()
    inner_iteration += 1
    max_movement = gpuarray.max(abs(direction)).get()
    if max_movement < 1e-4 or inner_iteration >= 200:
      break

  # re normalize points to sphere

  # loop_points = loop_points / np.linalg.norm(loop_points, axis=1, keepdims=True) * TARGET_RADIUS
  # lvp.updateValue(loop_points.flatten(), deepCopy = True)
  # DHAT_VALUE += 0.000000
  # TARGET_LENGTH += 1.0 / 200 * TARGET_RADIUS / NUM_LINE_POINTS
  # dhat.updateValue([DHAT_VALUE])
  # target_length.updateValue([TARGET_LENGTH])

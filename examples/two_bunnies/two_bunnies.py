from yasps import scene
from helpers import extract_surface_triangles, stable_neo_hookean, inertia, extract_edges_from_triangles
import numpy as np
import sys
sys.path.append('../ccd')  # or an absolute path
from ccd import CCD
import pycuda.gpuarray as gpuarray
DT_VALUE = 0.01 # for time step
DELTA_VALUE = 0.001 # for collision detection
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
tet_indices_second = tet_indices.copy()

f = open("../data/bunny.node", 'r')
f.readline()
position = []
for line in f:
  position.append([float(x) for x in line.split()[1:]])
f.close()
position = np.array(position, dtype = np.float64)
position_second = position.copy()

position = position - np.array([5.0, 0.0, 0.0])
position_second = position_second + np.array([5.0, 0.0, 0.0])
tet_indices_second = tet_indices_second + position.shape[0]

position = np.concatenate((position, position_second), axis = 0)
tet_indices = np.concatenate((tet_indices, tet_indices_second), axis = 0)
surface_triangle_indices = extract_surface_triangles(tet_indices)
edge_indices = extract_edges_from_triangles(surface_triangle_indices)

surface_indices = list(set(surface_triangle_indices.flatten().tolist()))

##################################################
## create the mesh with primitives and attributes
##################################################
s0 = scene("scene0")
dt = s0.addAttribute("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])
delta = s0.addAttribute("delta", rows = 1, cols = 1)
delta.updateValue([DELTA_VALUE])

bunnies = s0.addMesh("bunnies")
bunnies.addPrimitive("vertices", numInstances = position.shape[0])
bunnies.addPrimitive("tets", numInstances = tet_indices.shape[0])
bunnies.addPrimitive("surfaceTriangles", numInstances = surface_triangle_indices.shape[0])
bunnies.addPrimitive("pp", numInstances = 0, isDynamic = True) # for point point collision
bunnies.addPrimitive("pe", numInstances = 0, isDynamic = True) # for point edge collision
bunnies.addPrimitive("pt", numInstances = 0, isDynamic = True) # for point triangle collision
bunnies.addPrimitive("ee", numInstances = 0, isDynamic = True) # for edge edge collision

bunnies.vertices.addAttribute("rest_position", rows = 3, cols = 1)
bunnies.vertices["rest_position"].updateValue(position)
bunnies.vertices.addAttribute("position", rows = 3, cols = 1)
bunnies.vertices["position"].updateValue(position)

bunnies.vertices.addAttribute("last_position", rows = 3, cols = 1)
bunnies.vertices["last_position"].updateValue(position)
bunnies.vertices.addAttribute("velocity", rows = 3, cols = 1)
velocities = np.zeros_like(position, dtype=np.float64)
velocities[:position.shape[0] //2, 0] = 4
velocities[position.shape[0] // 2:, 0] = -4
bunnies.vertices["velocity"].updateValue(velocities)

bunnies.vertices.addAttribute("mass", rows = 1, cols = 1)
bunnies.vertices["mass"].updateValue(np.ones((position.shape[0], 1), dtype=np.float64) * 0.01)

mu = bunnies.addAttribute("mu", rows = 1, cols = 1) # for stable neo hookean
lam = bunnies.addAttribute("lam", rows = 1, cols = 1) # for stable neo hookean
mu.updateValue([2000.0])
lam.updateValue([1000.0])

##################################################
## add connectivities, and attributes
##################################################
tet2v = bunnies.tets.addConnectivity("tet2v", bunnies.vertices, tet_indices, 4)
tri2v = bunnies.surfaceTriangles.addConnectivity("tri2v", bunnies.vertices, surface_triangle_indices, 3)
pp2v = bunnies.pp.addConnectivity("pp2v", bunnies.vertices, [], 2)
pe2v = bunnies.pe.addConnectivity("pe2v", bunnies.vertices, [], 3)
pt2v = bunnies.pt.addConnectivity("pt2v", bunnies.vertices, [], 4)
ee2v = bunnies.ee.addConnectivity("ee2v", bunnies.vertices, [], 4)
tet_positions = bunnies.tets.addAttribute("positions", through = tet2v, source = bunnies.vertices["position"])
tet_rest_positions = bunnies.tets.addAttribute("rest_positions", through = tet2v, source = bunnies.vertices["rest_position"])
tri_positions = bunnies.surfaceTriangles.addAttribute("positions", through = tri2v, source = bunnies.vertices["position"])
pp_positions = bunnies.pp.addAttribute("positions", through = pp2v, source = bunnies.vertices["position"])
pe_positions = bunnies.pe.addAttribute("positions", through = pe2v, source = bunnies.vertices["position"])
pt_positions = bunnies.pt.addAttribute("positions", through = pt2v, source = bunnies.vertices["position"])
ee_positions = bunnies.ee.addAttribute("positions", through = ee2v, source = bunnies.vertices["position"])

##################################################
# construct ccd
##################################################
ccd = CCD(position.shape[0])
surface_indices_gpu = gpuarray.to_gpu(position[surface_indices].astype(np.uint32))
ccd.init_faces(bunnies.vertices["position"].value, tri2v.value, surface_indices_gpu, surface_triangle_indices.shape[0])



##################################################
## add energy to the scene
##################################################
snh = stable_neo_hookean(tet_rest_positions, tet_positions, mu, lam, dt)
snh_energy = bunnies.tets.addAttribute("stable_neo_hookean", computed_attribute = snh)
inertia = inertia(bunnies.vertices["last_position"], bunnies.vertices["velocity"], dt, bunnies.vertices["position"], bunnies.vertices["mass"])
inertia_energy = bunnies.vertices.addAttribute("inertia", computed_attribute = inertia)

s0.addEnergy(snh_energy)
s0.addEnergy(inertia_energy)
s0.addMinimizeTarget([bunnies.vertices["position"]])

##################################################
## plot the scene
##################################################
import pyvista as pv
triangles = np.array(surface_triangle_indices)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
bunny_poly = pv.PolyData(position, cells)

plotter = pv.Plotter()
plotter.add_mesh(bunny_poly, color='blue')
plotter.camera_position = [(0, 0, 20), (0, 0, 0), (0, 1, 0)]


plotter.show(interactive_update=True)

for i in range(20000):
  bunnies.vertices["last_position"].updateValue(bunnies.vertices["position"].value.get())
  result = s0.minimizeEnergy()
  d_position = result[0]
  bunnies.vertices["position"].updateValue((bunnies.vertices["position"].value - d_position))

  new_velocities = (bunnies.vertices["position"].value - bunnies.vertices["last_position"].value) / DT_VALUE
  bunnies.vertices["velocity"].updateValue(new_velocities)
  new_positions = bunnies.vertices["position"].value.get().reshape(-1, 3)
  bunny_poly.points = new_positions
  plotter.render()
  plotter.update()

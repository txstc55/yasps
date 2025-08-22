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
NUM_LOOP_POINTS = 2000

######################################################
# Read the bunny mesh
######################################################
v_bunny = []
f_bunny = []
with open("../data/bunny_uv.obj", "r") as obj_file:
  for line in obj_file:
    if line.startswith("vt "):
      parts = line.split()
      v_bunny.append([float(parts[1]), float(parts[2]), 0.0])
    elif line.startswith("f "):
      parts = line.split()
      f_bunny.append([int(parts[1].split("/")[1]) - 1, int(parts[2].split("/")[1]) - 1, int(parts[3].split("/")[1]) - 1])

v_bunny = np.array(v_bunny, dtype=np.float64)
f_bunny = np.array(f_bunny, dtype=np.uint32)
# get the bounding box of the uv coordinates
min_x = np.min(v_bunny[:, 0])
max_x = np.max(v_bunny[:, 0])
min_y = np.min(v_bunny[:, 1])
max_y = np.max(v_bunny[:, 1])

# center the bunny
v_bunny[:, 0] = v_bunny[:, 0] - (min_x + max_x) / 2.0
v_bunny[:, 1] = v_bunny[:, 1] - (min_y + max_y) / 2.0

# now scale it so its in the -1 to 1 range
scale_x = 1.0 / ((max_x - min_x) / 2.0)
scale_y = 1.0 / ((max_y - min_y) / 2.0)

v_bunny[:, 0] = v_bunny[:, 0] * scale_x
v_bunny[:, 1] = v_bunny[:, 1] * scale_y

min_x = np.min(v_bunny[:, 0])
max_x = np.max(v_bunny[:, 0])
min_y = np.min(v_bunny[:, 1])
max_y = np.max(v_bunny[:, 1])
print("Bunny UV bounding box: ", min_x, max_x, min_y, max_y)


######################################################
# Read the bunny mesh
######################################################











import pyvista as pv
# first we add bunny
cells = np.hstack([np.full((f_bunny.shape[0], 1), 3), f_bunny])
bunny_poly = pv.PolyData(v_bunny, cells)
plotter = pv.Plotter()
plotter.add_mesh(bunny_poly, show_edges=True, color='cyan', opacity = 0.6)
plotter.camera_position = [(0, 0, 5.0),
 (0.0, 0.0, 0.0),
 (0.0, 1.0, 0.0)]
plotter.show()

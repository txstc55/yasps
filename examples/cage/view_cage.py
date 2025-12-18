import numpy as np

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
print(bunny_vertices.shape)
# open weights.npz
cage_npz = np.load("weights.npz")
points = cage_npz['grid_points_pos']
cages = cage_npz['boxes_grid_point_indices']
v2g = cage_npz['vertex_grid_point_indices']
v2w = cage_npz['vertex_weights']
print(points.shape)
print(cages.shape)
print(v2g.shape)
print(v2w.shape)


# 12 edges of a box in your corner order:
# 0:000 1:100 2:010 3:110 4:001 5:101 6:011 7:111
edge_pairs = np.array([
  [0, 1], [2, 3], [4, 5], [6, 7],  # x edges
  [0, 2], [1, 3], [4, 6], [5, 7],  # y edges
  [0, 4], [1, 5], [2, 6], [3, 7],  # z edges
], dtype=np.int32)

# Build all edges for all cages, then unique them
edges = cages[:, edge_pairs].reshape(-1, 2)   # (Nb*12, 2)

tet5 = np.array([
  [0, 1], [1, 3], [3, 7], [7, 0],
  [0, 3], [3, 2], [2, 7], [7, 0],
  [0, 2], [2, 6], [6, 7], [7, 0],
  [0, 6], [6, 4], [4, 7], [7, 0],
  [0, 4], [4, 5], [5, 7], [7, 0],
  [0, 5], [5, 1], [1, 7], [7, 0],
], dtype=np.int32)
edges = cages[:, tet5].reshape(-1, 2)   # (Nb*12, 2)
# exit()


import pyvista as pv
plotter = pv.Plotter(window_size=[3840, 2160])
bunny_mesh = pv.PolyData(bunny_vertices, np.hstack((np.full((bunny_faces.shape[0], 1), 3), bunny_faces)).astype(np.uint32))
# make it transparent
plotter.add_mesh(bunny_mesh, color='white', opacity=0.5)

# plot cages
cells_loop = np.hstack([np.full((edges.shape[0], 1), 2), edges])
loop_poly = pv.PolyData(points, lines = cells_loop)
plotter.add_mesh(loop_poly, color='red', line_width=3)

plotter.show()

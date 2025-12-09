import numpy as np
bunny_vertices = []
bunny_faces = []
################################################
# Read original bunny mesh
################################################
f = open("../data/bunny_small.obj", 'r')
for line in f:
  if line.startswith('v '):
    bunny_vertices.append([float(x) for x in line.strip().split()[1:]])
  if line.startswith('f '):
    bunny_faces.append([int(x.split('//')[0]) - 1 for x in line.strip().split()[1:]])
f.close()
bunny_vertices = np.array(bunny_vertices, dtype=np.float64)
bunny_faces = np.array(bunny_faces, dtype=np.int32)
################################################
# Read the sphere parameterization
################################################
sphere_vertices = []
f = open("../data/smoothing_result.obj", 'r')
for line in f:
  if line.startswith('v '):
    sphere_vertices.append([float(x) for x in line.strip().split()[1:]])
f.close()

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
  sphere_vertices[i, :] = direction

################################################
# Read the curve result
################################################
curve_vertices = []
curve_edges = []
f = open("./outputs/loop_000000.obj", 'r')
for line in f:
  if line.startswith('v '):
    curve_vertices.append([float(x) for x in line.strip().split()[1:]])
    curve_edges.append([len(curve_vertices) - 1, len(curve_vertices)])
f.close()
curve_vertices = np.array(curve_vertices, dtype=np.float64)
for i in range(curve_vertices.shape[0]):
  direction = curve_vertices[i, :] / np.linalg.norm(curve_vertices[i, :])
  curve_vertices[i, :] = direction
curve_edges = np.array(curve_edges, dtype=np.int32)
curve_edges[-1, -1] = 0  # close the loop

################################################
# Do the mapping
################################################
def closest_point_on_triangle_vec(P, A, B, C):
    """
    Vectorized closest point on triangle for many points.

    P: (K,3) array of query points
    A,B,C: (3,) triangle vertices

    Returns:
      Q: (K,3) closest points on triangle
      bary: (K,3) barycentric coords (w0,w1,w2)
    """
    K = P.shape[0]

    AB = B - A
    AC = C - A

    AP = P - A  # (K,3)
    d1 = np.einsum('ij,j->i', AP, AB)
    d2 = np.einsum('ij,j->i', AP, AC)

    Q = np.empty_like(P)
    bary = np.empty((K, 3))
    remaining = np.ones(K, dtype=bool)

    # Region A
    mask = remaining & (d1 <= 0.0) & (d2 <= 0.0)
    if np.any(mask):
        Q[mask] = A
        bary[mask] = np.array([1.0, 0.0, 0.0])
        remaining[mask] = False

    BP = P - B
    d3 = np.einsum('ij,j->i', BP, AB)
    d4 = np.einsum('ij,j->i', BP, AC)

    # Region B
    mask = remaining & (d3 >= 0.0) & (d4 <= d3)
    if np.any(mask):
        Q[mask] = B
        bary[mask] = np.array([0.0, 1.0, 0.0])
        remaining[mask] = False

    # Edge AB
    vc = d1 * d4 - d3 * d2
    mask = remaining & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(mask):
        v = d1[mask] / (d1[mask] - d3[mask])
        Q[mask] = A + np.outer(v, AB)
        bary[mask, 0] = 1.0 - v
        bary[mask, 1] = v
        bary[mask, 2] = 0.0
        remaining[mask] = False

    CP = P - C
    d5 = np.einsum('ij,j->i', CP, AB)
    d6 = np.einsum('ij,j->i', CP, AC)

    # Region C
    mask = remaining & (d6 >= 0.0) & (d5 <= d6)
    if np.any(mask):
        Q[mask] = C
        bary[mask] = np.array([0.0, 0.0, 1.0])
        remaining[mask] = False

    # Edge AC
    vb = d5 * d2 - d1 * d6
    mask = remaining & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(mask):
        w = d2[mask] / (d2[mask] - d6[mask])
        Q[mask] = A + np.outer(w, AC)
        bary[mask, 0] = 1.0 - w
        bary[mask, 1] = 0.0
        bary[mask, 2] = w
        remaining[mask] = False

    # Edge BC
    va = d3 * d6 - d5 * d4
    mask = remaining & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if np.any(mask):
        w = (d4[mask] - d3[mask]) / ((d4[mask] - d3[mask]) + (d5[mask] - d6[mask]))
        Q[mask] = B + np.outer(w, (C - B))
        bary[mask, 0] = 0.0
        bary[mask, 1] = 1.0 - w
        bary[mask, 2] = w
        remaining[mask] = False

    # Face interior for the rest
    if np.any(remaining):
        va_r = va[remaining]
        vb_r = vb[remaining]
        vc_r = vc[remaining]
        denom = 1.0 / (va_r + vb_r + vc_r)
        v = vb_r * denom
        w = vc_r * denom
        u = 1.0 - v - w

        Q[remaining] = (
            u[:, None] * A +
            v[:, None] * B +
            w[:, None] * C
        )
        bary[remaining, 0] = u
        bary[remaining, 1] = v
        bary[remaining, 2] = w

    return Q, bary



def map_curve_to_bunny(curve_vertices, sphere_vertices, bunny_vertices, bunny_faces):
    """
    curve_vertices: (K,3) points on unit sphere
    sphere_vertices: (N,3)
    bunny_vertices: (N,3)
    bunny_faces: (F,3)

    Returns:
      mapped: (K,3) points on bunny
    """
    K = curve_vertices.shape[0]
    F = bunny_faces.shape[0]

    best_dist2 = np.full(K, np.inf)
    best_face = np.full(K, -1, dtype=np.int32)
    best_bary = np.zeros((K, 3), dtype=np.float64)

    P = curve_vertices  # (K,3)

    for fi in range(F):
        print(f"Mapping to face {fi+1}/{F}", end='\r')
        i0, i1, i2 = bunny_faces[fi]
        A = sphere_vertices[i0]
        B = sphere_vertices[i1]
        C = sphere_vertices[i2]

        Q, bary = closest_point_on_triangle_vec(P, A, B, C)
        dist2 = np.einsum('ij,ij->i', Q - P, Q - P)

        mask = dist2 < best_dist2
        if not np.any(mask):
            continue

        best_dist2[mask] = dist2[mask]
        best_face[mask] = fi
        best_bary[mask] = bary[mask]

    # Now apply barycentric coords to bunny mesh
    v0 = bunny_vertices[bunny_faces[best_face, 0]]
    v1 = bunny_vertices[bunny_faces[best_face, 1]]
    v2 = bunny_vertices[bunny_faces[best_face, 2]]

    mapped = (
        best_bary[:, 0:1] * v0 +
        best_bary[:, 1:1+1] * v1 +  # or best_bary[:,1:2]
        best_bary[:, 2:3] * v2
    )

    return mapped



mapped_curve_vertices = map_curve_to_bunny(
  curve_vertices,
  sphere_vertices,
  bunny_vertices,
  bunny_faces
)

#####################################################
# Visualization
#####################################################
import pyvista as pv
# first we add bunny
plotter = pv.Plotter(window_size = [3840, 2160])

cells_loop = np.hstack([np.full((curve_edges.shape[0], 1), 2), curve_edges])
loop_poly = pv.PolyData(mapped_curve_vertices, lines = cells_loop)
plotter.add_mesh(loop_poly, color='red', line_width=3)
loop_poly.save("./final_mapped_curve_000.obj")
plotter.show()

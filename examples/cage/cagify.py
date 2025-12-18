#!/usr/bin/env python3
"""
FFD lattice cage with EXACTLY 8 weights per vertex, keeping only "used" cells.
Saves exactly what you asked:

1) grid_points_pos: (Nu,3) positions of USED grid/control points
2) boxes_grid_point_indices: (Nb,8) each row = 8 indices (into grid_points_pos) that form one used box (cell)
3) per-vertex:
   - vertex_grid_point_indices: (N,8) indices (into grid_points_pos)
   - vertex_weights: (N,8) weights

Also shows only used boxes in preview and renders mesh half-transparent.

Dependencies:
  pip install numpy trimesh open3d

Usage:
  python ffd_used_cells_export.py --mesh input.obj --out cage_data.npz
  python ffd_used_cells_export.py --mesh input.obj --out cage_data.npz --nx 16 --ny 16 --nz 16 --surface_samples 80000
  python ffd_used_cells_export.py --mesh input.obj --out cage_data.npz --no_preview
"""

import argparse
import sys
import numpy as np

try:
    import trimesh
except Exception as e:
    raise ImportError("Need trimesh: pip install trimesh") from e

try:
    import open3d as o3d
except Exception as e:
    raise ImportError("Need open3d: pip install open3d") from e


# ---------------- IO ----------------

def load_tri_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {path}")
    if m.faces is None or len(m.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    if m.faces.shape[1] != 3:
        m = m.triangulate()
    m.process(validate=True)
    if len(m.vertices) == 0 or len(m.faces) == 0:
        raise ValueError("Loaded mesh is empty after processing.")
    return m


# ---------------- Open3D helpers ----------------

def to_o3d_mesh(V: np.ndarray, F: np.ndarray) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(V, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(F, dtype=np.int32)),
    )
    m.compute_vertex_normals()
    return m


# ---------------- Lattice ----------------
# IMPORTANT: These match ctrl.reshape(-1,3) with default C-order:
# flattened index for ctrl[i,j,k] is ((i*ny)+j)*nz + k
# => k fastest, then j, then i.

def lattice_index(i: np.ndarray, j: np.ndarray, k: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return k + nz * (j + ny * i)

def lattice_unindex(idx: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    i = idx // (ny * nz)
    rem = idx - i * (ny * nz)
    j = rem // nz
    k = rem - j * nz
    return np.stack([i, j, k], axis=1).astype(np.int32)


def build_lattice(bmin: np.ndarray, bmax: np.ndarray, nx: int, ny: int, nz: int):
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)
    if np.any(bmax <= bmin):
        raise ValueError("Invalid bbox for lattice.")

    xs = np.linspace(bmin[0], bmax[0], nx, dtype=np.float64)
    ys = np.linspace(bmin[1], bmax[1], ny, dtype=np.float64)
    zs = np.linspace(bmin[2], bmax[2], nz, dtype=np.float64)

    ctrl = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)  # (nx,ny,nz,3)
    ctrl_pos = ctrl.reshape(-1, 3)  # C-order: k fastest, then j, then i

    dx = (bmax[0] - bmin[0]) / (nx - 1)
    dy = (bmax[1] - bmin[1]) / (ny - 1)
    dz = (bmax[2] - bmin[2]) / (nz - 1)
    return ctrl_pos, dx, dy, dz


def point_to_cell_ijk(P: np.ndarray, bmin: np.ndarray, dx: float, dy: float, dz: float,
                      nx: int, ny: int, nz: int):
    P = np.asarray(P, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)

    u = (P[:, 0] - bmin[0]) / dx
    v = (P[:, 1] - bmin[1]) / dy
    w = (P[:, 2] - bmin[2]) / dz

    u = np.clip(u, 0.0, (nx - 1) - 1e-9)
    v = np.clip(v, 0.0, (ny - 1) - 1e-9)
    w = np.clip(w, 0.0, (nz - 1) - 1e-9)

    i = np.floor(u).astype(np.int32)
    j = np.floor(v).astype(np.int32)
    k = np.floor(w).astype(np.int32)

    i = np.clip(i, 0, nx - 2)
    j = np.clip(j, 0, ny - 2)
    k = np.clip(k, 0, nz - 2)
    return i, j, k


# ---------------- FFD weights (8-corner) ----------------

def compute_ffd_top8(mesh_V: np.ndarray, bmin: np.ndarray, bmax: np.ndarray,
                     nx: int, ny: int, nz: int):
    V = np.asarray(mesh_V, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)

    dx = (bmax[0] - bmin[0]) / (nx - 1)
    dy = (bmax[1] - bmin[1]) / (ny - 1)
    dz = (bmax[2] - bmin[2]) / (nz - 1)
    eps = 1e-12
    if dx <= eps or dy <= eps or dz <= eps:
        raise ValueError("Degenerate lattice dimensions; increase bbox or resolution.")

    u = (V[:, 0] - bmin[0]) / dx
    v = (V[:, 1] - bmin[1]) / dy
    wcoord = (V[:, 2] - bmin[2]) / dz

    u = np.clip(u, 0.0, (nx - 1) - 1e-9)
    v = np.clip(v, 0.0, (ny - 1) - 1e-9)
    wcoord = np.clip(wcoord, 0.0, (nz - 1) - 1e-9)

    i0 = np.floor(u).astype(np.int32)
    j0 = np.floor(v).astype(np.int32)
    k0 = np.floor(wcoord).astype(np.int32)

    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    k1 = np.clip(k0 + 1, 0, nz - 1)

    fu = u - i0
    fv = v - j0
    fw = wcoord - k0

    one_fu = 1.0 - fu
    one_fv = 1.0 - fv
    one_fw = 1.0 - fw

    w000 = one_fu * one_fv * one_fw
    w100 = fu     * one_fv * one_fw
    w010 = one_fu * fv     * one_fw
    w110 = fu     * fv     * one_fw
    w001 = one_fu * one_fv * fw
    w101 = fu     * one_fv * fw
    w011 = one_fu * fv     * fw
    w111 = fu     * fv     * fw

    N = V.shape[0]
    idx = np.empty((N, 8), dtype=np.int32)
    ww = np.empty((N, 8), dtype=np.float64)

    # Corner order:
    # 000,100,010,110,001,101,011,111
    idx[:, 0] = lattice_index(i0,     j0,     k0,     nx, ny, nz)
    idx[:, 1] = lattice_index(i1,     j0,     k0,     nx, ny, nz)
    idx[:, 2] = lattice_index(i0,     j1,     k0,     nx, ny, nz)
    idx[:, 3] = lattice_index(i1,     j1,     k0,     nx, ny, nz)
    idx[:, 4] = lattice_index(i0,     j0,     k1,     nx, ny, nz)
    idx[:, 5] = lattice_index(i1,     j0,     k1,     nx, ny, nz)
    idx[:, 6] = lattice_index(i0,     j1,     k1,     nx, ny, nz)
    idx[:, 7] = lattice_index(i1,     j1,     k1,     nx, ny, nz)

    ww[:, 0] = w000
    ww[:, 1] = w100
    ww[:, 2] = w010
    ww[:, 3] = w110
    ww[:, 4] = w001
    ww[:, 5] = w101
    ww[:, 6] = w011
    ww[:, 7] = w111

    ww /= (ww.sum(axis=1, keepdims=True) + 1e-18)
    return idx, ww


def check_weights(idx: np.ndarray, w: np.ndarray, n_ctrl: int):
    if idx.shape != w.shape or idx.shape[1] != 8:
        raise ValueError("idx and w must be (N,8)")
    if not np.all(np.isfinite(w)):
        raise ValueError("w contains NaN/Inf")
    if np.min(idx) < 0 or np.max(idx) >= n_ctrl:
        raise ValueError("idx out of range")
    err = np.max(np.abs(w.sum(axis=1) - 1.0))
    if err > 1e-6:
        raise ValueError(f"Row sums not 1 (max err {err})")


# ---------------- Used cell selection (surface + scan fill) ----------------

def fill_inside_from_surface(surface: np.ndarray) -> np.ndarray:
    ni, nj, nk = surface.shape
    inside = np.zeros_like(surface, dtype=bool)

    for j in range(nj):
        for k in range(nk):
            line = surface[:, j, k]
            hit = np.flatnonzero(line)
            if hit.size < 2:
                continue

            breaks = np.where(np.diff(hit) > 1)[0]
            starts = np.concatenate(([hit[0]], hit[breaks + 1]))
            ends = np.concatenate((hit[breaks], [hit[-1]]))
            nr = starts.size
            if nr < 2:
                continue

            for r in range(0, nr - 1, 2):
                e0 = int(ends[r])
                s1 = int(starts[r + 1])
                if s1 > e0 + 1:
                    inside[e0 + 1:s1, j, k] = True

    return inside


def compute_used_cells(mesh: trimesh.Trimesh,
                       bmin: np.ndarray, bmax: np.ndarray,
                       nx: int, ny: int, nz: int,
                       surface_samples: int = 50000):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    bmin = np.asarray(bmin, dtype=np.float64)
    bmax = np.asarray(bmax, dtype=np.float64)

    dx = (bmax[0] - bmin[0]) / (nx - 1)
    dy = (bmax[1] - bmin[1]) / (ny - 1)
    dz = (bmax[2] - bmin[2]) / (nz - 1)

    ni, nj, nk = nx - 1, ny - 1, nz - 1
    surface = np.zeros((ni, nj, nk), dtype=bool)

    # Cells containing mesh vertices
    iv, jv, kv = point_to_cell_ijk(V, bmin, dx, dy, dz, nx, ny, nz)
    surface[iv, jv, kv] = True

    # Cells containing surface samples
    if surface_samples > 0:
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, int(surface_samples))
            pts = np.asarray(pts, dtype=np.float64)
            isamp, jsamp, ksamp = point_to_cell_ijk(pts, bmin, dx, dy, dz, nx, ny, nz)
            surface[isamp, jsamp, ksamp] = True
        except Exception as e:
            print(f"[WARN] surface sampling failed ({e}); proceeding with vertices only.", file=sys.stderr)

    inside = fill_inside_from_surface(surface)
    used = surface | inside
    return used, surface, inside


# ---------------- Build boxes + compact grid points ----------------

def used_cells_to_corners_old(used_cells_ijk: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    c = np.asarray(used_cells_ijk, dtype=np.int32)
    i = c[:, 0]
    j = c[:, 1]
    k = c[:, 2]

    p000 = lattice_index(i,     j,     k,     nx, ny, nz)
    p100 = lattice_index(i + 1, j,     k,     nx, ny, nz)
    p010 = lattice_index(i,     j + 1, k,     nx, ny, nz)
    p110 = lattice_index(i + 1, j + 1, k,     nx, ny, nz)
    p001 = lattice_index(i,     j,     k + 1, nx, ny, nz)
    p101 = lattice_index(i + 1, j,     k + 1, nx, ny, nz)
    p011 = lattice_index(i,     j + 1, k + 1, nx, ny, nz)
    p111 = lattice_index(i + 1, j + 1, k + 1, nx, ny, nz)

    corners_old = np.stack([p000, p100, p010, p110, p001, p101, p011, p111], axis=1).astype(np.int32)
    return corners_old


def compact_grid_points_and_boxes(ctrl_pos_full: np.ndarray,
                                  boxes_corners_old: np.ndarray,
                                  nx: int, ny: int, nz: int):
    """
    Inputs:
      ctrl_pos_full: (Nfull,3) all grid points
      boxes_corners_old: (Nb,8) old indices into ctrl_pos_full

    Outputs:
      grid_points_pos: (Nu,3) positions of USED grid points
      boxes_grid_point_indices: (Nb,8) indices into grid_points_pos
      old_to_new: (Nfull,) mapping (unused -> -1)
      new_to_old: (Nu,) old index for each used point
      grid_points_ijk: (Nu,3) integer coords for each used point
    """
    used_old = np.unique(boxes_corners_old.reshape(-1)).astype(np.int32)
    n_full = ctrl_pos_full.shape[0]
    old_to_new = np.full(n_full, -1, dtype=np.int32)
    old_to_new[used_old] = np.arange(len(used_old), dtype=np.int32)

    grid_points_pos = np.asarray(ctrl_pos_full[used_old], dtype=np.float64)
    boxes_grid_point_indices = old_to_new[boxes_corners_old].astype(np.int32)
    if np.any(boxes_grid_point_indices < 0):
        raise RuntimeError("Internal error: box references discarded point.")

    new_to_old = used_old
    grid_points_ijk = lattice_unindex(new_to_old, nx, ny, nz)
    return grid_points_pos, boxes_grid_point_indices, old_to_new, new_to_old, grid_points_ijk


# ---------------- Preview (used boxes only + transparent mesh) ----------------

def make_boundary_lines_from_boxes(boxes: np.ndarray) -> np.ndarray:
    """
    boxes: (Nb,8) indices into grid_points_pos with corner order:
      000,100,010,110,001,101,011,111

    Draw only the boundary faces (internal shared faces removed).
    Returns edges: (E,2)
    """
    b = np.asarray(boxes, dtype=np.int32)

    face_defs = np.asarray([
        [0, 2, 6, 4],  # x-min
        [1, 3, 7, 5],  # x-max
        [0, 1, 5, 4],  # y-min
        [2, 3, 7, 6],  # y-max
        [0, 1, 3, 2],  # z-min
        [4, 5, 7, 6],  # z-max
    ], dtype=np.int32)

    boundary = {}
    for cell in b:
        for fd in face_defs:
            face = tuple(int(x) for x in cell[fd])
            key = tuple(sorted(face))
            if key in boundary:
                del boundary[key]
            else:
                boundary[key] = face

    edges = []
    for face in boundary.values():
        a, c, d, e = face
        edges.append([a, c])
        edges.append([c, d])
        edges.append([d, e])
        edges.append([e, a])

    edges = np.asarray(edges, dtype=np.int32)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def preview(mesh_V, mesh_F, grid_points_pos, boxes_grid_point_indices, mesh_alpha: float):
    mesh_o3d = to_o3d_mesh(mesh_V, mesh_F)

    edges = make_boundary_lines_from_boxes(boxes_grid_point_indices)
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(grid_points_pos, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(edges),
    )
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(grid_points_pos, dtype=np.float64)))

    try:
        import open3d.visualization.rendering as rendering  # noqa: F401

        mat_mesh = o3d.visualization.rendering.MaterialRecord()
        mat_mesh.shader = "defaultLitTransparency"
        mat_mesh.base_color = [0.8, 0.8, 0.8, float(mesh_alpha)]
        mat_mesh.base_roughness = 1.0
        mat_mesh.base_metallic = 0.0

        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.base_color = [0.9, 0.1, 0.1, 1.0]
        mat_line.line_width = 2.0

        mat_pc = o3d.visualization.rendering.MaterialRecord()
        mat_pc.shader = "defaultUnlit"
        mat_pc.base_color = [0.9, 0.1, 0.1, 1.0]
        mat_pc.point_size = 4.0

        o3d.visualization.draw(
            [
                {"name": "mesh", "geometry": mesh_o3d, "material": mat_mesh},
                {"name": "boxes", "geometry": lineset, "material": mat_line},
                {"name": "grid_points", "geometry": pc, "material": mat_pc},
            ],
            title="Mesh (transparent) + USED boxes. Close to continue.",
            width=1280,
            height=720,
            show_ui=True,
        )
    except Exception:
        mesh_o3d.paint_uniform_color([0.85, 0.85, 0.85])
        lineset.paint_uniform_color([0.9, 0.1, 0.1])
        pc.paint_uniform_color([0.9, 0.1, 0.1])
        o3d.visualization.draw_geometries([mesh_o3d, lineset, pc],
                                          window_name="Preview",
                                          width=1280, height=720)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Input mesh file (obj/ply/stl/...).")
    ap.add_argument("--out", required=True, help="Output .npz (grid_points, boxes, per-vertex idx+weights).")
    ap.add_argument("--nx", type=int, default=12, help="Grid points in X (>=2).")
    ap.add_argument("--ny", type=int, default=12, help="Grid points in Y (>=2).")
    ap.add_argument("--nz", type=int, default=12, help="Grid points in Z (>=2).")
    ap.add_argument("--pad", type=float, default=0.05, help="Pad fraction of bbox diagonal.")
    ap.add_argument("--surface_samples", type=int, default=50000, help="Surface samples to seed boundary cells.")
    ap.add_argument("--mesh_alpha", type=float, default=0.5, help="Mesh transparency in preview (0..1).")
    ap.add_argument("--no_preview", action="store_true", help="Disable GUI preview.")
    args = ap.parse_args()

    nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("--nx/--ny/--nz must be >= 2")

    mesh = load_tri_mesh(args.mesh)
    mesh_V = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_F = np.asarray(mesh.faces, dtype=np.int32)

    if not mesh.is_watertight:
        print("[WARN] Mesh is not watertight; inside fill from surface-hits may be unreliable.", file=sys.stderr)

    bmin = mesh_V.min(axis=0)
    bmax = mesh_V.max(axis=0)
    diag = np.linalg.norm(bmax - bmin)
    pad = float(args.pad) * float(diag)
    bmin = bmin - pad
    bmax = bmax + pad

    # Full grid
    ctrl_pos_full, dx, dy, dz = build_lattice(bmin, bmax, nx, ny, nz)

    # Used cells
    used_mask, surface_mask, inside_mask = compute_used_cells(
        mesh=mesh,
        bmin=bmin, bmax=bmax,
        nx=nx, ny=ny, nz=nz,
        surface_samples=int(args.surface_samples),
    )
    used_cells_ijk = np.argwhere(used_mask).astype(np.int32)
    if used_cells_ijk.shape[0] == 0:
        raise RuntimeError("No used cells found. Increase --surface_samples or raise nx/ny/nz.")

    # Boxes as 8-corner indices (old/full-grid)
    boxes_corners_old = used_cells_to_corners_old(used_cells_ijk, nx, ny, nz)  # (Nb,8)

    # Compact grid points to only those used by boxes
    grid_points_pos, boxes_grid_point_indices, old_to_new, new_to_old, grid_points_ijk = compact_grid_points_and_boxes(
        ctrl_pos_full=ctrl_pos_full,
        boxes_corners_old=boxes_corners_old,
        nx=nx, ny=ny, nz=nz
    )

    # Per-vertex weights + indices (old/full-grid), then map to compact grid point indices
    vertex_idx_old, vertex_weights = compute_ffd_top8(mesh_V, bmin, bmax, nx, ny, nz)
    vertex_grid_point_indices = old_to_new[vertex_idx_old].astype(np.int32)
    if np.any(vertex_grid_point_indices < 0):
        bad = int(np.sum(vertex_grid_point_indices < 0))
        raise RuntimeError(
            f"{bad} mesh vertices reference grid points that were discarded. "
            "This means some vertex fell into a cell that wasn't marked used (increase surface_samples / resolution)."
        )

    check_weights(vertex_grid_point_indices, vertex_weights, n_ctrl=len(grid_points_pos))

    if not args.no_preview:
        preview(
            mesh_V=mesh_V,
            mesh_F=mesh_F,
            grid_points_pos=grid_points_pos,
            boxes_grid_point_indices=boxes_grid_point_indices,
            mesh_alpha=float(np.clip(args.mesh_alpha, 0.0, 1.0)),
        )

    # Save EXACTLY what you requested
    np.savez_compressed(
        args.out,
        grid_points_pos=grid_points_pos.astype(np.float64),                      # (Nu,3)
        boxes_grid_point_indices=boxes_grid_point_indices.astype(np.int32),     # (Nb,8)
        vertex_grid_point_indices=vertex_grid_point_indices.astype(np.int32),   # (N,8)
        vertex_weights=vertex_weights.astype(np.float64),                       # (N,8)

        # extra metadata (handy)
        mesh_path=args.mesh,
        bbox_min=np.asarray(bmin, dtype=np.float64),
        bbox_max=np.asarray(bmax, dtype=np.float64),
        nx=np.int32(nx), ny=np.int32(ny), nz=np.int32(nz),
        dx=np.float64(dx), dy=np.float64(dy), dz=np.float64(dz),
        grid_points_old_index=new_to_old.astype(np.int32),                      # (Nu,) optional
        grid_points_ijk=grid_points_ijk.astype(np.int32),                       # (Nu,3) optional
        cells_used_ijk=used_cells_ijk.astype(np.int32),                         # (Nb,3) optional
        surface_cell_count=np.int32(int(np.count_nonzero(surface_mask))),
        inside_cell_count=np.int32(int(np.count_nonzero(inside_mask))),
        surface_samples=np.int32(int(args.surface_samples)),
    )

    print("Done.")
    print(f"Saved: {args.out}")
    print(f"Used grid points: {grid_points_pos.shape[0]}")
    print(f"Used boxes: {boxes_grid_point_indices.shape[0]}")
    print(f"Mesh vertices: {mesh_V.shape[0]}")
    print(f"Surface cells: {int(np.count_nonzero(surface_mask))} | Inside cells: {int(np.count_nonzero(inside_mask))}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

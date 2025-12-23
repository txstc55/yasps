import os
import re
import struct
import numpy as np

BASE_BLOCKS_PER_N = 19193
BS = 3

FMT_BCOO = 1
FMT_BCSR = 2
FMT_COO  = 3
FMT_CSR  = 4

MAGIC8 = b"SPMVv2\x00\x00"  # EXACTLY 8 bytes


# ---------------------------
# Helpers
# ---------------------------
def infer_n_from_filename(path: str) -> int:
    m = re.search(r"_0*([0-9]+)\.npz$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot infer N from filename: {path}")
    return int(m.group(1))

def write_header(f, fmt: int, is_full: int, block_size: int, rows: int, cols: int, nnz: int):
    # 44 bytes total
    f.write(struct.pack("<8sIIIQQQ", MAGIC8, fmt, is_full, block_size,
                        int(rows), int(cols), int(nnz)))

def load_upper_blocks(npz_path: str):
    d = np.load(npz_path)
    blocks = d["blocks"]  # (K,3,3)
    pos = d["block_positions"]  # (K,2) scalar top-left corners
    pos = pos.flatten()
    pos_len = pos.shape[0]
    pos = pos[: int(pos_len / 1.5)]


    blocks = blocks.astype(np.float64)
    blocks = blocks.reshape(-1,3,3)
    pos = pos.astype(np.uint32)
    pos = pos.reshape(-1,2)
    print(blocks.shape, pos.shape)

    if blocks.ndim != 3 or blocks.shape[1:] != (3,3):
        raise ValueError(f"{npz_path}: blocks must be (K,3,3), got {blocks.shape}")
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(f"{npz_path}: block_positions must be (K,2+), got {pos.shape}")

    row = pos[:, 0].astype(np.uint32)
    col = pos[:, 1].astype(np.uint32)
    if np.any(row % 3 != 0) or np.any(col % 3 != 0):
        raise ValueError(f"{npz_path}: block_positions not multiples of 3 (expected top-left of 3x3)")

    brow = row // 3
    bcol = col // 3
    blocks = blocks.astype(np.float64, copy=False)
    return blocks, brow, bcol

def make_full_from_upper(blocks, brow, bcol):
    diag = (brow == bcol)
    off  = ~diag

    blocks_diag = blocks[diag]
    brow_diag = brow[diag]
    bcol_diag = bcol[diag]

    blocks_off = blocks[off]
    brow_off = brow[off]
    bcol_off = bcol[off]

    blocks_mirror = np.transpose(blocks_off, (0,2,1))
    brow_mirror = bcol_off
    bcol_mirror = brow_off

    blocks_full = np.concatenate([blocks_diag, blocks_off, blocks_mirror], axis=0)
    brow_full   = np.concatenate([brow_diag, brow_off, brow_mirror], axis=0)
    bcol_full   = np.concatenate([bcol_diag, bcol_off, bcol_mirror], axis=0)
    return blocks_full, brow_full, bcol_full

def validate_block_indices(brow, bcol, nb, tag=""):
    if brow.size == 0:
        raise ValueError(f"{tag}: no blocks")
    rmin, rmax = int(brow.min()), int(brow.max())
    cmin, cmax = int(bcol.min()), int(bcol.max())
    if rmin < 0 or cmin < 0 or rmax >= nb or cmax >= nb:
        raise ValueError(
            f"{tag}: block indices out of range. "
            f"brow[min,max]=[{rmin},{rmax}] bcol[min,max]=[{cmin},{cmax}] nb={nb}"
        )

def validate_scalar_indices(rows, cols, M, tag=""):
    if rows.size == 0:
        raise ValueError(f"{tag}: no scalar entries")
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    if rmin < 0 or cmin < 0 or rmax >= M or cmax >= M:
        raise ValueError(
            f"{tag}: scalar indices out of range. "
            f"row[min,max]=[{rmin},{rmax}] col[min,max]=[{cmin},{cmax}] M={M}"
        )

def coo_sum_duplicates(rows, cols, vals):
    """
    Sort by (row,col) then sum duplicates.
    Returns (rows_u, cols_u, vals_u) sorted by (row,col).
    """
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    # boundaries where key changes
    change = np.ones(rows.shape[0], dtype=bool)
    change[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], rows.shape[0]]

    rows_u = rows[starts]
    cols_u = cols[starts]
    vals_u = np.empty(starts.shape[0], dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        vals_u[i] = vals[s:e].sum()
    return rows_u, cols_u, vals_u

def coo_to_csr(M, rows, cols, vals):
    """
    Build CSR from COO (duplicates already OK; we will sum duplicates anyway).
    Returns (row_ptr, col_ind, val) with sorted columns within each row.
    """
    rows_u, cols_u, vals_u = coo_sum_duplicates(rows, cols, vals)

    row_ptr = np.zeros(M + 1, dtype=np.uint32)
    np.add.at(row_ptr, rows_u + 1, 1)
    row_ptr = np.cumsum(row_ptr, dtype=np.uint32)

    # Already globally sorted by (row,col), so within-row columns are sorted.
    col_ind = cols_u.astype(np.uint32, copy=False)
    val_out = vals_u.astype(np.float64, copy=False)

    # sanity
    if row_ptr[0] != 0 or row_ptr[-1] != col_ind.size:
        raise ValueError("CSR row_ptr invalid (not matching nnz)")
    if np.any(col_ind < 0) or np.any(col_ind >= M):
        raise ValueError("CSR col_ind out of range")
    if np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("CSR row_ptr not monotone")

    return row_ptr, col_ind, val_out

def bcoo_to_bcsr(nb, brow, bcol, blocks):
    """
    Convert block COO (3x3 blocks) to block CSR with duplicates summed.
    Returns (row_ptr, col_ind, data) where data is (nnzb,3,3).
    """
    order = np.lexsort((bcol, brow))
    r = brow[order]
    c = bcol[order]
    B = blocks[order]

    change = np.ones(r.shape[0], dtype=bool)
    change[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], r.shape[0]]

    r_u = r[starts]
    c_u = c[starts]
    B_u = np.empty((starts.shape[0], 3, 3), dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        B_u[i] = B[s:e].sum(axis=0)

    row_ptr = np.zeros(nb + 1, dtype=np.uint32)
    np.add.at(row_ptr, r_u + 1, 1)
    row_ptr = np.cumsum(row_ptr, dtype=np.uint32)

    col_ind = c_u.astype(np.uint32, copy=False)
    data = B_u

    # sanity
    if row_ptr[0] != 0 or row_ptr[-1] != col_ind.size:
        raise ValueError("BCSR row_ptr invalid (not matching nnz blocks)")
    if np.any(col_ind < 0) or np.any(col_ind >= nb):
        raise ValueError("BCSR col_ind out of range")
    if np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("BCSR row_ptr not monotone")

    return row_ptr, col_ind, data

def expand_blocks_to_scalar_coo(brow, bcol, blocks):
    """
    Expand block COO (K,3,3) → scalar COO (K*9,)
    """
    K = blocks.shape[0]
    r0 = (brow * BS).astype(np.uint32)
    c0 = (bcol * BS).astype(np.uint32)

    rr = np.array([0,0,0,1,1,1,2,2,2], dtype=np.uint32)
    cc = np.array([0,1,2,0,1,2,0,1,2], dtype=np.uint32)

    rows = (r0[:, None] + rr[None, :]).reshape(-1)
    cols = (c0[:, None] + cc[None, :]).reshape(-1)
    vals = blocks.reshape(K, 9).reshape(-1).astype(np.float64, copy=False)
    return rows, cols, vals


# ---------------------------
# Writers
# ---------------------------
def write_bcoo(path, is_full, nb, brow, bcol, blocks):
    K = blocks.shape[0]
    validate_block_indices(brow, bcol, nb, tag=f"BCOO {'full' if is_full else 'upper'}")
    with open(path, "wb") as f:
        write_header(f, FMT_BCOO, is_full, 3, nb, nb, K)
        np.asarray(brow, dtype=np.uint32).tofile(f)
        np.asarray(bcol, dtype=np.uint32).tofile(f)
        np.asarray(blocks, dtype=np.float64).reshape(-1).tofile(f)  # K*9

def write_bcsr(path, is_full, nb, row_ptr, col_ind, data):
    nnzb = data.shape[0]
    with open(path, "wb") as f:
        write_header(f, FMT_BCSR, is_full, 3, nb, nb, nnzb)
        np.asarray(row_ptr, dtype=np.uint32).tofile(f)
        np.asarray(col_ind, dtype=np.uint32).tofile(f)
        np.asarray(data, dtype=np.float64).reshape(-1).tofile(f)  # nnzb*9

def write_coo(path, is_full, M, rows, cols, vals):
    nnz = vals.size
    validate_scalar_indices(rows, cols, M, tag=f"COO {'full' if is_full else 'upper'}")
    with open(path, "wb") as f:
        write_header(f, FMT_COO, is_full, 1, M, M, nnz)
        np.asarray(rows, dtype=np.uint32).tofile(f)
        np.asarray(cols, dtype=np.uint32).tofile(f)
        np.asarray(vals, dtype=np.float64).tofile(f)

def write_csr(path, is_full, M, row_ptr, col_ind, vals):
    nnz = vals.size
    with open(path, "wb") as f:
        write_header(f, FMT_CSR, is_full, 1, M, M, nnz)
        np.asarray(row_ptr, dtype=np.uint32).tofile(f)
        np.asarray(col_ind, dtype=np.uint32).tofile(f)
        np.asarray(vals, dtype=np.float64).tofile(f)


# ---------------------------
# Main export for one NPZ
# ---------------------------
def export_all_formats(npz_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    N = infer_n_from_filename(npz_path)
    nb = N * BASE_BLOCKS_PER_N
    M  = nb * 3

    blocks_u, brow_u, bcol_u = load_upper_blocks(npz_path)
    validate_block_indices(brow_u, bcol_u, nb, tag=f"{os.path.basename(npz_path)} upper")

    blocks_f, brow_f, bcol_f = make_full_from_upper(blocks_u, brow_u, bcol_u)
    validate_block_indices(brow_f, bcol_f, nb, tag=f"{os.path.basename(npz_path)} full")

    stem = os.path.splitext(os.path.basename(npz_path))[0]

    def export_variant(tag, is_full, blocks, brow, bcol):
        # ---- Block formats ----
        bcoo_path = os.path.join(out_dir, f"{stem}_{tag}_BCOO.bin")
        write_bcoo(bcoo_path, is_full, nb, brow, bcol, blocks)

        row_ptr_b, col_ind_b, data_b = bcoo_to_bcsr(nb, brow, bcol, blocks)
        bcsr_path = os.path.join(out_dir, f"{stem}_{tag}_BCSR.bin")
        write_bcsr(bcsr_path, is_full, nb, row_ptr_b, col_ind_b, data_b)

        # ---- Scalar formats ----
        rows_s, cols_s, vals_s = expand_blocks_to_scalar_coo(brow, bcol, blocks)
        order = np.lexsort((cols_s, rows_s))   # primary rows, secondary cols
        rows_s = rows_s[order]
        cols_s = cols_s[order]
        vals_s = vals_s[order]
        # canonicalize COO: sort by (row,col) and sum duplicates
        rows_s, cols_s, vals_s = coo_sum_duplicates(rows_s, cols_s, vals_s)
        if not np.all(rows_s[:-1] <= rows_s[1:]):
            raise ValueError("COO rows are not sorted; cuSPARSE assumes row-sorted COO.")
        validate_scalar_indices(rows_s, cols_s, M, tag=f"{stem} {tag} scalar")

        coo_path = os.path.join(out_dir, f"{stem}_{tag}_COO.bin")
        write_coo(coo_path, is_full, M, rows_s, cols_s, vals_s)

        row_ptr_s, col_ind_s, vals_u = coo_to_csr(M, rows_s, cols_s, vals_s)
        csr_path = os.path.join(out_dir, f"{stem}_{tag}_CSR.bin")
        write_csr(csr_path, is_full, M, row_ptr_s, col_ind_s, vals_u)

        print(f"[OK] {stem}_{tag}: "
              f"BCOO blocks={blocks.shape[0]} | "
              f"BCSR blocks(nnzb)={data_b.shape[0]} | "
              f"COO nnz={vals_s.size} | "
              f"CSR nnz={vals_u.size}")

    export_variant("upper", 0, blocks_u, brow_u, bcol_u)
    export_variant("full",  1, blocks_f, brow_f, bcol_f)


def export_many(npz_paths, out_dir):
    for p in npz_paths:
        export_all_formats(p, out_dir)


if __name__ == "__main__":
    # Example:
    # export_all_formats("hessian_blocks_01.npz", "./bin_out_v2")

    import glob
    paths = sorted(glob.glob("hessian_blocks_*.npz"))
    export_many(paths, out_dir="./bin_out_v2")

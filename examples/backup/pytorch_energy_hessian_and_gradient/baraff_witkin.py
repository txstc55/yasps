import torch
import numpy as np
import time

# ----------------------------
# Setup
# ----------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
torch.set_default_dtype(torch.float64)

# Optional: reduce fragmentation (can help on long runs)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ----------------------------
# 1) Base triangle data (one triangle)
# ----------------------------
position = np.array([[0, 0, 1],
                     [1, 0, 2],
                     [0, 0, 0]], dtype=np.float64)

center = np.mean(position, axis=0)
position -= center

rest_tri = torch.tensor(position, device=device)          # (3,3)
curr_tri = (rest_tri * 1.54).clone()                      # (3,3)

stretchS = torch.tensor(2000.0, device=device)
shearS   = torch.tensor(1000.0, device=device)

# ----------------------------
# 2) Energy definitions (batched over T)
# ----------------------------
def compute_rest_shape(rest_triangle_pos):
    """
    rest_triangle_pos: (T,3,3)
    returns M: (T,2,2)
    """
    T = rest_triangle_pos.shape[0]
    v0 = rest_triangle_pos[:, 0, :]
    v1 = rest_triangle_pos[:, 1, :]
    v2 = rest_triangle_pos[:, 2, :]
    v01 = v1 - v0
    v02 = v2 - v0

    normal = torch.cross(v01, v02, dim=1)
    normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-8)

    target = rest_triangle_pos.new_tensor([0.0, 1.0, 0.0]).unsqueeze(0).expand(T, -1)
    vec = torch.cross(normal, target, dim=1)
    cos = torch.sum(normal * target, dim=1)

    eye = torch.eye(3, dtype=rest_triangle_pos.dtype, device=rest_triangle_pos.device)
    rotation = eye.unsqueeze(0).expand(T, -1, -1)

    zero = torch.zeros(T, 1, device=rest_triangle_pos.device, dtype=rest_triangle_pos.dtype)
    cross_vec = torch.stack([
        torch.cat([zero, -vec[:, 2:3], vec[:, 1:2]], dim=1),
        torch.cat([vec[:, 2:3], zero, -vec[:, 0:1]], dim=1),
        torch.cat([-vec[:, 1:2], vec[:, 0:1], zero], dim=1)
    ], dim=1)

    denom = (1.0 + cos).unsqueeze(1).unsqueeze(2) + 1e-8
    cross_vec_sq = torch.bmm(cross_vec, cross_vec)
    rotation = rotation + cross_vec + cross_vec_sq / denom

    rotate_uv0 = torch.bmm(rotation, v0.unsqueeze(2)).squeeze(2)
    rotate_uv1 = torch.bmm(rotation, v1.unsqueeze(2)).squeeze(2)
    rotate_uv2 = torch.bmm(rotation, v2.unsqueeze(2)).squeeze(2)

    uv0 = rotate_uv0[:, [0, 2]]
    uv1 = rotate_uv1[:, [0, 2]]
    uv2 = rotate_uv2[:, [0, 2]]

    uv1m0 = uv1 - uv0
    uv2m0 = uv2 - uv0

    M = torch.stack([
        torch.stack([uv1m0[:, 0], uv2m0[:, 0]], dim=1),
        torch.stack([uv1m0[:, 1], uv2m0[:, 1]], dim=1)
    ], dim=1)  # (T,2,2)
    return M

def baraff_witkin_energy(x_init, x, stretchS, shearS, thickness=0.01, dt=1.0):
    """
    x_init, x: (T,3,3)
    returns energies: (T,)
    """
    x10 = x[:, 1, :] - x[:, 0, :]
    x20 = x[:, 2, :] - x[:, 0, :]

    F = torch.stack([x10, x20], dim=1).transpose(1, 2)   # (T,3,2)

    M = compute_rest_shape(x_init)                        # (T,2,2)
    M_inv = torch.linalg.inv(M)                           # (T,2,2)
    F = torch.bmm(F, M_inv)                               # (T,3,2)

    F_TF = torch.bmm(F.transpose(1, 2), F)                # (T,2,2)

    I6 = F_TF[:, 0, 1]
    shear_energy = I6 ** 2

    I5u = F[:, :, 0].norm(dim=1)
    I5v = F[:, :, 1].norm(dim=1)
    stretch_energy = (I5u - 1.0) ** 2 + (I5v - 1.0) ** 2

    v01 = x_init[:, 1, :] - x_init[:, 0, :]
    v02 = x_init[:, 2, :] - x_init[:, 0, :]
    area = thickness * torch.cross(v01, v02, dim=1).norm(dim=1) / 2.0

    return (stretchS * stretch_energy + shearS * shear_energy) * area * (dt ** 2)  # (T,)

# ----------------------------
# 3) Bench sizes
# ----------------------------
N  = 20000   # energy/grad throughput over N
Nh = 20000       # Hessians over Nh only (keep small; try 1k~20k)

# ----------------------------
# 4) Build batches
# ----------------------------
# NOTE: N=2M float64 tensors are BIG.
# x_batch (N,3,3) is 2e6*9*8 ~ 144MB
# r_batch same ~ 144MB
# noise same ~ 144MB
# plus intermediates (lots). Expect multi-GB usage.

r_batch = rest_tri.unsqueeze(0).expand(N, -1, -1).contiguous()

# Create per-instance variation
gen = torch.Generator(device=device)
gen.manual_seed(0)
noise = 1e-3 * torch.randn((N, 3, 3), device=device, generator=gen)
x_batch = (curr_tri.unsqueeze(0).expand(N, -1, -1).contiguous() + noise).requires_grad_(True)

# Sub-batch for Hessians
r_h = r_batch[:Nh].contiguous()
x_h = x_batch[:Nh].contiguous()

# ----------------------------
# 5) Compiled functions
# ----------------------------
def energy_per_instance(r, x):
    return baraff_witkin_energy(r, x, stretchS, shearS)  # (T,)

def total_energy(r, x):
    return energy_per_instance(r, x).sum()               # scalar

compiled_energy_per_instance = torch.compile(energy_per_instance, mode="max-autotune")
compiled_total_energy        = torch.compile(total_energy, mode="max-autotune")

# ----------------------------
# 6) Per-instance Hessians correctly (Nh,9,9) using torch.func (vectorized)
# ----------------------------
from torch.func import vmap, hessian

def single_energy_flat(r_single, x_flat):
    # r_single: (3,3), x_flat: (9,)
    x = x_flat.view(3, 3)
    # Wrap to (1,3,3) so we reuse the batched energy
    return baraff_witkin_energy(r_single.unsqueeze(0), x.unsqueeze(0), stretchS, shearS).sum()

# Hessian of single instance wrt x_flat -> (9,9)
single_hess = hessian(single_energy_flat, argnums=1)

# Vectorize over instances: (Nh,3,3) and (Nh,9) -> (Nh,9,9)
batched_hess = vmap(single_hess, in_dims=(0, 0))

# ----------------------------
# 7) Timing helper
# ----------------------------
def time_ms(fn, iters):
    torch.cuda.synchronize()
    t0 = time.time()
    out = None
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) * 1e3 / iters, out

# ----------------------------
# 8) Warmup
# ----------------------------
with torch.no_grad():
    E = compiled_energy_per_instance(r_batch, x_batch)
torch.cuda.synchronize()

G = torch.autograd.grad(compiled_total_energy(r_batch, x_batch), x_batch, create_graph=True)[0]
torch.cuda.synchronize()

# Hessian warmup on small Nh
x_flat_h = x_h.reshape(Nh, 9)
H = batched_hess(r_h, x_flat_h)
torch.cuda.synchronize()

print("Warmup done.")
print("E shape:", E.shape, "G shape:", G.shape, "H shape:", H.shape)

# ----------------------------
# 9) Benchmarks
# ----------------------------
energy_ms, _ = time_ms(lambda: compiled_energy_per_instance(r_batch, x_batch), iters=10)

grad_ms, _ = time_ms(
    lambda: torch.autograd.grad(compiled_total_energy(r_batch, x_batch), x_batch, create_graph=False)[0],
    iters=10
)

# Hessian is very expensive; fewer iters
hess_ms, H_out = time_ms(lambda: batched_hess(r_h, x_h.reshape(Nh, 9)), iters=10)

print(f"\n=== Parallel independent-instance benchmark (PyTorch) ===")
print(f"Energy   over N={N:,}   : {energy_ms:.3f} ms / call   (output (N,))")
print(f"Grad     over N={N:,}   : {grad_ms:.3f} ms / call     (output (N,3,3))")
print(f"Hessian  over Nh={Nh:,} : {hess_ms:.3f} ms / call    (output (Nh,9,9))")
print("Hessian output shape:", H_out.shape)

print(f"\nPer-instance:")
print(f"Energy  : {energy_ms * 1e6 / N:.3f} ns / instance")
print(f"Grad    : {grad_ms  * 1e6 / N:.3f} ns / instance")
print(f"Hessian : {hess_ms  * 1e6 / Nh:.3f} ns / instance  (Nh only)")

import torch
import numpy as np

# ============================
# Setup
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_default_dtype(torch.float64)

# ============================
# Input (one tet)
# ============================
rest_position_np = np.array([
    [0.8168341,  -0.57334204, -0.28303627],
    [0.9386871,  -0.60033335, -0.10815139],
    [0.93494301, -0.73053988, -0.27055541],
    [1.00758576, -0.58084046, -0.37639869]
], dtype=np.float64)

current_position_np = np.array([
    [0.8168341,   0.34925004, -0.53558849],
    [0.9386871,   0.40100103,  0.4596684 ],
    [0.93494301,  0.44387604,  0.64020529],
    [1.00758576, -0.53429907,  0.43997285]
], dtype=np.float64)

rest_tet = torch.tensor(rest_position_np, device=device)      # (4,3)
curr_tet = torch.tensor(current_position_np, device=device)   # (4,3)

mu  = torch.tensor(2000.0, device=device)
lam = torch.tensor(1000.0, device=device)

# ============================
# Rest precompute (B^-1 and volume)
# ============================
def precompute_rest_invariants(r):
    """
    r: (T,4,3)
    returns:
      IB:  (T,3,3)  inverse rest matrix
      vol: (T,)     rest volume
    """
    x0, x1, x2, x3 = r[:, 0, :], r[:, 1, :], r[:, 2, :], r[:, 3, :]
    X0 = x1 - x0
    X1 = x2 - x0
    X2 = x3 - x0
    B  = torch.stack([X0, X1, X2], dim=2)            # (T,3,3)
    detB = torch.linalg.det(B)
    vol  = detB / 6.0                                # (T,)
    IB   = torch.linalg.inv(B)                       # (T,3,3)
    return IB, vol

# ============================
# Stable Neo-Hookean using precomputed rest invariants
# ============================
def stable_neo_hookean_energy_precomp(x, IB, vol, mu, lam):
    """
    x:   (T,4,3)    (requires_grad=True)
    IB:  (T,3,3)    (constant, detached)
    vol: (T,)       (constant, detached)
    returns: (T,)
    """
    y0, y1, y2, y3 = x[:, 0, :], x[:, 1, :], x[:, 2, :], x[:, 3, :]
    Y0 = y1 - y0
    Y1 = y2 - y0
    Y2 = y3 - y0
    F  = torch.stack([Y0, Y1, Y2], dim=2)            # (T,3,3)

    FI = torch.bmm(F, IB)
    J  = torch.linalg.det(FI)
    IC = torch.sum(FI * FI, dim=(1, 2))
    I3 = IC + 1.0

    shift = 1.0 + 0.75 * (mu / lam)
    return vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * torch.log(I3) + 0.5 * lam * (J - shift) ** 2)

# ============================
# Benchmark sizes
# ============================
N  = 79_935
Nh = 79_935     # keep Hessians smaller; (79,935,12,12) is huge + slow

# ============================
# Build independent batches
# ============================
r_batch = rest_tet.unsqueeze(0).expand(N, -1, -1).contiguous()

gen = torch.Generator(device=device)
gen.manual_seed(0)
noise = 1e-3 * torch.randn((N, 4, 3), device=device, generator=gen)
x_batch = (curr_tet.unsqueeze(0).expand(N, -1, -1).contiguous() + noise).requires_grad_(True)

# Precompute + detach rest invariants (IMPORTANT)
with torch.no_grad():
    IB_batch, vol_batch = precompute_rest_invariants(r_batch)
IB_batch = IB_batch.detach()
vol_batch = vol_batch.detach()

# Hessian subset
x_h = x_batch[:Nh].contiguous()
IB_h = IB_batch[:Nh].contiguous()
vol_h = vol_batch[:Nh].contiguous()
x_flat_h = x_h.reshape(Nh, 12)

# ============================
# Timing helper (CUDA events)
# ============================
def cuda_time_ms(fn, iters=50):
    out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out

# ============================
# Energy benchmark (N)
# ============================
def energy_call():
    return stable_neo_hookean_energy_precomp(x_batch, IB_batch, vol_batch, mu, lam)

energy_ms, E = cuda_time_ms(energy_call, iters=50)
print(f"Energy: {energy_ms:.3f} ms/call, E shape={tuple(E.shape)}")

# ============================
# Gradient benchmark (N)
# ============================
def grad_call():
    total = stable_neo_hookean_energy_precomp(x_batch, IB_batch, vol_batch, mu, lam).sum()
    g = torch.autograd.grad(total, x_batch, create_graph=False)[0]
    return g

grad_ms, G = cuda_time_ms(grad_call, iters=50)
print(f"Grad:   {grad_ms:.3f} ms/call, G shape={tuple(G.shape)}")

# ============================
# Hessian benchmark (Nh) - per-instance 12x12 blocks
# ============================
from torch.func import vmap, hessian

def single_energy_flat(IB_single, vol_single, x_flat):
    """
    IB_single:  (3,3)
    vol_single: () or (1,) scalar
    x_flat:     (12,)
    returns scalar energy
    """
    x = x_flat.reshape(4, 3)
    e = stable_neo_hookean_energy_precomp(
        x.unsqueeze(0),
        IB_single.unsqueeze(0),
        vol_single.reshape(1),
        mu, lam
    )
    return e[0]

single_hess = hessian(single_energy_flat, argnums=2)  # Hessian wrt x_flat -> (12,12)
batched_hess = vmap(single_hess, in_dims=(0, 0, 0))   # (Nh,12,12)

def hess_call():
    return batched_hess(IB_h, vol_h, x_flat_h)

hess_ms, H = cuda_time_ms(hess_call, iters=10)
print(f"Hess:   {hess_ms:.3f} ms/call, H shape={tuple(H.shape)}")

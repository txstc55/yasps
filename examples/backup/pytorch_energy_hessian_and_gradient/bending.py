import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_default_dtype(torch.float64)

# ============================
# Base tet (one instance)
# ============================
position = np.array([[0, 0, 1],
                     [1, 0, 2],
                     [0, 0, 0],
                     [0, 2, 0]], dtype=np.float64)
position -= position.mean(axis=0)

rest_tet0 = torch.tensor(position, device=device)          # (4,3)
curr_tet0 = (rest_tet0 * 1.54).clone()                     # (4,3)

bendStiff = torch.tensor(1000.0, device=device)

# ============================
# Bending energy pieces (batched)
# ============================
def angle(v, w, axis, eps=1e-8):
    # v,w,axis: (T,3)
    cross_vw = torch.cross(v, w, dim=1)                  # (T,3)
    cross_dot_axis = torch.sum(cross_vw * axis, dim=1)   # (T,)
    axis_norm = torch.norm(axis, dim=1) + eps            # (T,)
    numerator = cross_dot_axis / axis_norm               # (T,)

    denom = torch.sum(v * w, dim=1) + torch.norm(v, dim=1) * torch.norm(w, dim=1)  # (T,)
    return 2.0 * torch.atan2(numerator, denom)

def edgeTheta_from_x(x):
    """
    x: (T,4,3)
    returns theta: (T,)
    """
    q0, q1, q2, q3 = x[:,0,:], x[:,1,:], x[:,2,:], x[:,3,:]
    n0 = torch.cross(q0 - q2, q1 - q2, dim=1)
    n1 = torch.cross(q1 - q3, q0 - q3, dim=1)
    axis = q1 - q0
    return angle(n0, n1, axis)

def bending_energy_precomp(x, t_init, edge_len, bendStiff):
    """
    x: (T,4,3) requires grad
    t_init: (T,) detached constant
    edge_len: (T,) detached constant
    """
    t = edgeTheta_from_x(x)                 # (T,)
    return bendStiff * (t - t_init)**2 * edge_len

# ============================
# Benchmark sizes
# ============================
N  = 104_065   # energy + grad over N independent tets
Nh = 104_065     # Hessian blocks over Nh only (keep smaller)

# ============================
# Build independent batches
# ============================
# Same rest tet for all instances is fine; independence means no coupling.
r_batch = rest_tet0.unsqueeze(0).expand(N, -1, -1).contiguous()

gen = torch.Generator(device=device)
gen.manual_seed(0)
noise = 1e-3 * torch.randn((N, 4, 3), device=device, generator=gen)
x_batch = (curr_tet0.unsqueeze(0).expand(N, -1, -1).contiguous() + noise).requires_grad_(True)

# ============================
# Precompute rest-only constants (DETACHED)
# ============================
with torch.no_grad():
    t_init_batch = edgeTheta_from_x(r_batch)                      # (N,)
    edge_len_batch = torch.norm(r_batch[:,1,:] - r_batch[:,0,:], dim=1)  # (N,)

t_init_batch = t_init_batch.detach()
edge_len_batch = edge_len_batch.detach()

# Hessian subset
x_h = x_batch[:Nh].contiguous()
t_init_h = t_init_batch[:Nh].contiguous()
edge_len_h = edge_len_batch[:Nh].contiguous()
x_flat_h = x_h.reshape(Nh, 12)

# ============================
# torch.compile forward path (optional)
# ============================
def energy_per_tet(x, t_init, edge_len):
    return bending_energy_precomp(x, t_init, edge_len, bendStiff)  # (T,)

def total_energy(x, t_init, edge_len):
    return energy_per_tet(x, t_init, edge_len).sum()

compiled_energy_per_tet = torch.compile(energy_per_tet, mode="max-autotune")
compiled_total_energy   = torch.compile(total_energy, mode="max-autotune")

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
# Energy benchmark
# ============================
energy_ms, E = cuda_time_ms(lambda: compiled_energy_per_tet(x_batch, t_init_batch, edge_len_batch), iters=50)
print(f"Energy: {energy_ms:.3f} ms/call, E shape={tuple(E.shape)}")

# ============================
# Gradient benchmark (per tet gradient via grad(sum))
# ============================
def grad_call():
    total = compiled_total_energy(x_batch, t_init_batch, edge_len_batch)
    g = torch.autograd.grad(total, x_batch, create_graph=False)[0]  # (N,4,3)
    return g

grad_ms, G = cuda_time_ms(grad_call, iters=50)
print(f"Grad:   {grad_ms:.3f} ms/call, G shape={tuple(G.shape)}")

# ============================
# Hessian benchmark (Nh) - per-tet 12x12 blocks, vectorized
# ============================
from torch.func import vmap, hessian

def single_energy_flat(t_init_single, edge_len_single, x_flat):
    x = x_flat.reshape(4, 3)  # (4,3)
    e = bending_energy_precomp(
        x.unsqueeze(0),                    # (1,4,3)
        t_init_single.reshape(1),          # (1,)
        edge_len_single.reshape(1),        # (1,)
        bendStiff
    )
    return e[0]

single_hess = hessian(single_energy_flat, argnums=2)     # (12,12)
batched_hess = vmap(single_hess, in_dims=(0, 0, 0))      # (Nh,12,12)

hess_ms, H = cuda_time_ms(lambda: batched_hess(t_init_h, edge_len_h, x_flat_h), iters=10)
print(f"Hess:   {hess_ms:.3f} ms/call, H shape={tuple(H.shape)}")

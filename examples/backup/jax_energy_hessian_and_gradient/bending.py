import os
os.environ["JAX_ENABLE_X64"] = "True"

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
from functools import partial

print("JAX default backend:", jax.default_backend())

# ============================================================
# 1) Base tet (one instance)
# ============================================================
position = np.array([[0, 0, 1],
                     [1, 0, 2],
                     [0, 0, 0],
                     [0, 2, 0]], dtype=np.float64)

center = np.mean(position, axis=0)
position -= center

rest_tet = jnp.array(position, dtype=jnp.float64)          # (4,3)
curr_tet = (rest_tet * 1.54).astype(jnp.float64)           # (4,3)

# sizes
N  = 104_065      # energy + grad over N
Nh = 104_065          # Hessian blocks over Nh only (keep small)

bending_stiff = jnp.array(1000.0, dtype=jnp.float64)

# ============================================================
# 2) Geometry helpers (single instance)
# ============================================================
def angle_single(v, w, axis, eps=1e-8):
  # v,w,axis: (3,)
  cross_vw = jnp.cross(v, w)                 # (3,)
  cross_dot_axis = jnp.dot(cross_vw, axis)   # scalar
  axis_norm = jnp.linalg.norm(axis) + eps

  numerator = cross_dot_axis / axis_norm

  dot_vw = jnp.dot(v, w)
  denom = dot_vw + (jnp.linalg.norm(v) * jnp.linalg.norm(w))

  return 2.0 * jnp.arctan2(numerator, denom)

def edgeTheta_single(q0, q1, q2, q3):
  n0 = jnp.cross(q0 - q2, q1 - q2)
  n1 = jnp.cross(q1 - q3, q0 - q3)
  axis = q1 - q0
  return angle_single(n0, n1, axis)

def bending_single(x, x_init, bendStiff):
  """
  x, x_init: (4,3)
  returns scalar
  """
  x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
  t = edgeTheta_single(x0, x1, x2, x3)

  r0, r1, r2, r3 = x_init[0], x_init[1], x_init[2], x_init[3]
  t_init = edgeTheta_single(r0, r1, r2, r3)

  delta_t_sq = (t - t_init) ** 2
  edge_len = jnp.linalg.norm(r1 - r0)
  return bendStiff * delta_t_sq * edge_len

# ============================================================
# 3) Batched energy/grad/hess for independent instances
# ============================================================

# per-instance energy: (N,)
batched_energy = jit(jax.vmap(bending_single, in_axes=(0, 0, None)))

# per-instance gradient wrt x: (N,4,3)
batched_grad = jit(jax.vmap(grad(bending_single, argnums=0), in_axes=(0, 0, None)))

# per-instance Hessian wrt x_flat (12,) -> (12,12): batched gives (Nh,12,12)
def bending_single_flat(x_flat, x_init, bendStiff):
  return bending_single(x_flat.reshape(4,3), x_init, bendStiff)

single_hess = jax.hessian(bending_single_flat, argnums=0)
batched_hess = jit(jax.vmap(single_hess, in_axes=(0, 0, None)))

# ============================================================
# 4) Create N independent instances
# ============================================================
key = jax.random.PRNGKey(0)

r_batch = jnp.broadcast_to(rest_tet[None, :, :], (N, 4, 3))         # (N,4,3) same rest ok
noise = 1e-3 * jax.random.normal(key, (N, 4, 3), dtype=jnp.float64) # make x differ
x_batch = jnp.broadcast_to(curr_tet[None, :, :], (N, 4, 3)) + noise # (N,4,3)

# Hessian subset
r_h = r_batch[:Nh]
x_h = x_batch[:Nh]
x_flat_h = x_h.reshape(Nh, 12)

# ============================================================
# 5) Warmup (compile)
# ============================================================
E = batched_energy(x_batch, r_batch, bending_stiff); jax.block_until_ready(E)
G = batched_grad(x_batch, r_batch, bending_stiff);   jax.block_until_ready(G)
H = batched_hess(x_flat_h, r_h, bending_stiff);      jax.block_until_ready(H)

print("Warmup done.")
print("E shape:", E.shape, "G shape:", G.shape, "H shape:", H.shape)

# ============================================================
# 6) Timing
# ============================================================
def time_ms(fn, iters):
  # run once to ensure ready
  out = fn()
  jax.block_until_ready(out)
  t0 = time.time()
  for _ in range(iters):
    out = fn()
  jax.block_until_ready(out)
  t1 = time.time()
  return (t1 - t0) * 1e3 / iters

energy_ms = time_ms(lambda: batched_energy(x_batch, r_batch, bending_stiff), iters=50)
grad_ms   = time_ms(lambda: batched_grad(x_batch, r_batch, bending_stiff),   iters=50)
hess_ms   = time_ms(lambda: batched_hess(x_flat_h, r_h, bending_stiff),      iters=10)

print(f"\n=== Parallel independent-instance benchmark (JAX) ===")
print(f"Energy   over N={N:,}   : {energy_ms:.3f} ms / call   (output (N,))")
print(f"Grad     over N={N:,}   : {grad_ms:.3f} ms / call     (output (N,4,3))")
print(f"Hessian  over Nh={Nh:,} : {hess_ms:.3f} ms / call    (output (Nh,12,12))")

print(f"\nPer-instance:")
print(f"Energy  : {energy_ms * 1e6 / N:.3f} ns / instance")
print(f"Grad    : {grad_ms  * 1e6 / N:.3f} ns / instance")
print(f"Hessian : {hess_ms  * 1e6 / Nh:.3f} ns / instance  (Nh only)")

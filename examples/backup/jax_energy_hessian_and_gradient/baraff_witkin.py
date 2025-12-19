# ============================================================
# JAX benchmark: N independent triangle instances in parallel
# - Computes per-instance energy:   (N,)
# - Computes per-instance gradient: (N,3,3)
# - Computes per-instance Hessian:  (Nh,9,9)  (use smaller Nh!)
# ============================================================

import os
os.environ["JAX_ENABLE_X64"] = "True"  # set before importing jax

import time
import numpy as np
import jax
import jax.numpy as jnp

print("JAX default backend:", jax.default_backend())

# -----------------------------
# 1) Initialize one base triangle
# -----------------------------
position = np.array([[0, 0, 1],
                     [1, 0, 2],
                     [0, 0, 0]], dtype=np.float64)

center = np.mean(position, axis=0)
position -= center

rest_tri = jnp.array(position, dtype=jnp.float64)          # (3,3)
curr_tri = (rest_tri * 1.54).astype(jnp.float64)           # (3,3)

stretchS = jnp.array(2000.0, dtype=jnp.float64)
shearS   = jnp.array(1000.0, dtype=jnp.float64)

# -----------------------------
# 2) Rest-shape helper (single triangle)
# -----------------------------
def compute_rest_shape_single(rest_triangle_pos):
  """
  rest_triangle_pos: (3,3)
  returns M: (2,2)
  """
  v0 = rest_triangle_pos[0, :]
  v1 = rest_triangle_pos[1, :]
  v2 = rest_triangle_pos[2, :]

  v01 = v1 - v0
  v02 = v2 - v0

  normal = jnp.cross(v01, v02)
  normal = normal / (jnp.linalg.norm(normal) + 1e-8)

  target = jnp.array([0.0, 1.0, 0.0], dtype=rest_triangle_pos.dtype)

  vec = jnp.cross(normal, target)
  cos_val = jnp.dot(normal, target)

  # skew-symmetric matrix for vec
  vx, vy, vz = vec[0], vec[1], vec[2]
  cross_vec = jnp.array([[0.0, -vz,  vy],
                         [ vz, 0.0, -vx],
                         [-vy,  vx, 0.0]], dtype=rest_triangle_pos.dtype)

  I = jnp.eye(3, dtype=rest_triangle_pos.dtype)
  cross_vec_sq = cross_vec @ cross_vec
  rotation = I + cross_vec + cross_vec_sq / (1.0 + cos_val + 1e-8)

  ru0 = rotation @ v0
  ru1 = rotation @ v1
  ru2 = rotation @ v2

  idx = jnp.array([0, 2], dtype=jnp.int32)
  uv0 = jnp.take(ru0, idx, axis=0)
  uv1 = jnp.take(ru1, idx, axis=0)
  uv2 = jnp.take(ru2, idx, axis=0)

  uv1m0 = uv1 - uv0
  uv2m0 = uv2 - uv0

  M = jnp.array([[uv1m0[0], uv2m0[0]],
                 [uv1m0[1], uv2m0[1]]], dtype=rest_triangle_pos.dtype)
  return M

# -----------------------------
# 3) Baraff-Witkin energy (single triangle -> scalar)
# -----------------------------
def baraff_witkin_energy_single(x_init, x, stretchS, shearS, thickness=0.01, dt=1.0):
  """
  x_init, x: (3,3)
  returns scalar energy
  """
  x10 = x[1, :] - x[0, :]
  x20 = x[2, :] - x[0, :]

  # F: (3,2)
  F = jnp.stack([x10, x20], axis=1)  # (3,2)

  M = compute_rest_shape_single(x_init)     # (2,2)
  M_inv = jnp.linalg.inv(M)                # (2,2)

  F_def = F @ M_inv                        # (3,2)

  F_TF = F_def.T @ F_def                   # (2,2)
  I6 = F_TF[0, 1]
  shear_energy = I6 ** 2

  I5u = jnp.linalg.norm(F_def[:, 0])
  I5v = jnp.linalg.norm(F_def[:, 1])
  stretch_energy = (I5u - 1.0) ** 2 + (I5v - 1.0) ** 2

  v01 = x_init[1, :] - x_init[0, :]
  v02 = x_init[2, :] - x_init[0, :]
  area = thickness * jnp.linalg.norm(jnp.cross(v01, v02)) / 2.0

  return (stretchS * stretch_energy + shearS * shear_energy) * area * (dt ** 2)

# -----------------------------
# 4) Batched versions for N independent instances
# -----------------------------
# energies: (N,)
batched_energy = jax.jit(jax.vmap(baraff_witkin_energy_single, in_axes=(0, 0, None, None)))

# per-instance gradients wrt x: (N,3,3)
batched_grad = jax.jit(
  jax.vmap(jax.grad(baraff_witkin_energy_single, argnums=1), in_axes=(0, 0, None, None))
)

# Hessian per instance wrt x_flat (9,) -> (9,9): batch gives (Nh,9,9)
def energy_single_flat(x_init, x_flat, stretchS, shearS):
  return baraff_witkin_energy_single(x_init, x_flat.reshape(3, 3), stretchS, shearS)

batched_hess = jax.jit(
  jax.vmap(jax.hessian(energy_single_flat, argnums=1), in_axes=(0, 0, None, None))
)

# -----------------------------
# 5) Create N independent instances (vary x a bit so they’re truly independent)
# -----------------------------
N  = 20000   # big throughput test for energy/grad
Nh = 20000      # Hessian batch size (KEEP SMALL; (Nh,9,9) blows up quickly)

key = jax.random.PRNGKey(0)

# Same rest shape for all instances is fine; "independent" means no coupling between instances.
r_batch = jnp.broadcast_to(rest_tri[None, :, :], (N, 3, 3))

# Make each instance slightly different (prevents degenerate compiler CSE and mimics real batches)
noise = 1e-3 * jax.random.normal(key, (N, 3, 3), dtype=jnp.float64)
x_batch = jnp.broadcast_to(curr_tri[None, :, :], (N, 3, 3)) + noise

# Smaller batch for Hessian
r_batch_h = r_batch[:Nh]
x_batch_h = x_batch[:Nh]
x_flat_h  = x_batch_h.reshape(Nh, 9)

# -----------------------------
# 6) Warmup (compile)
# -----------------------------
E = batched_energy(r_batch, x_batch, stretchS, shearS)
jax.block_until_ready(E)

G = batched_grad(r_batch, x_batch, stretchS, shearS)
jax.block_until_ready(G)

H = batched_hess(r_batch_h, x_flat_h, stretchS, shearS)
jax.block_until_ready(H)

print("Warmup done.")
print("E shape:", E.shape, "G shape:", G.shape, "H shape:", H.shape)

# -----------------------------
# 7) Timing utilities
# -----------------------------
def time_it(fn, iters=10):
  # run once to ensure compiled and ready
  out = fn()
  jax.block_until_ready(out)
  t0 = time.time()
  for _ in range(iters):
    out = fn()
  jax.block_until_ready(out)
  t1 = time.time()
  return (t1 - t0) * 1e3 / iters, out

# -----------------------------
# 8) Benchmarks
# -----------------------------
energy_ms, _ = time_it(lambda: batched_energy(r_batch, x_batch, stretchS, shearS), iters=10)
grad_ms,  _  = time_it(lambda: batched_grad(r_batch, x_batch, stretchS, shearS), iters=10)

# Hessian is expensive; use fewer iters
hess_ms,  _  = time_it(lambda: batched_hess(r_batch_h, x_flat_h, stretchS, shearS), iters=3)

print(f"\n=== Parallel independent-instance benchmark ===")
print(f"Energy   over N={N:,}   : {energy_ms:.3f} ms / call   (output (N,))")
print(f"Grad     over N={N:,}   : {grad_ms:.3f} ms / call     (output (N,3,3))")
print(f"Hessian  over Nh={Nh:,} : {hess_ms:.3f} ms / call    (output (Nh,9,9))")

# Optional: also report per-instance cost (useful for comparing GPUs/CPUs)
print(f"\nPer-instance:")
print(f"Energy  : {energy_ms * 1e6 / N:.3f} ns / instance")
print(f"Grad    : {grad_ms  * 1e6 / N:.3f} ns / instance")
print(f"Hessian : {hess_ms  * 1e6 / Nh:.3f} ns / instance  (Nh only)")

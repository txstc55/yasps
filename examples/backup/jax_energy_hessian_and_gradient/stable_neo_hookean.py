import os
os.environ["JAX_ENABLE_X64"] = "True"

import time
import numpy as np
import jax
import jax.numpy as jnp

print("JAX default backend:", jax.default_backend())

# ============================================================
# 1) Setup: one base tet
# ============================================================
position = jnp.array([[0, 0, 1],
                      [1, 0, 2],
                      [0, 0, 0],
                      [0, 2, 0]], dtype=jnp.float64)
position = position - jnp.mean(position, axis=0)

rest_tet = position                        # (4,3)
curr_tet = (rest_tet * 1.54).astype(jnp.float64)

mu  = jnp.array(2000.0, dtype=jnp.float64)
lam = jnp.array(1000.0, dtype=jnp.float64)

N  = 79_935      # energy + grad over N independent tets
Nh = 79_935       # Hessian blocks over Nh only (keep small)

# ============================================================
# 2) Stable Neo-Hookean energy
# ============================================================
def stable_neo_hookean_energy(current_tet_pos, rest_tet_pos, mu, lam):
  """
  current_tet_pos, rest_tet_pos: (T,4,3)
  returns (T,)
  """
  x0 = rest_tet_pos[:, 0, :]
  x1 = rest_tet_pos[:, 1, :]
  x2 = rest_tet_pos[:, 2, :]
  x3 = rest_tet_pos[:, 3, :]
  X0 = x1 - x0
  X1 = x2 - x0
  X2 = x3 - x0
  B = jnp.stack([X0, X1, X2], axis=2)      # (T,3,3)
  detB = jnp.linalg.det(B)
  vol = detB / 6.0
  IB = jnp.linalg.inv(B)

  y0 = current_tet_pos[:, 0, :]
  y1 = current_tet_pos[:, 1, :]
  y2 = current_tet_pos[:, 2, :]
  y3 = current_tet_pos[:, 3, :]
  Y0 = y1 - y0
  Y1 = y2 - y0
  Y2 = y3 - y0
  F = jnp.stack([Y0, Y1, Y2], axis=2)      # (T,3,3)
  FI = jnp.einsum('tij,tjk->tik', F, IB)

  J  = jnp.linalg.det(FI)
  IC = jnp.sum(FI * FI, axis=(1, 2))
  I3 = IC + 1.0

  shift = 1.0 + 0.75 * (mu / lam)
  energy_per_tet = vol * (
      0.5 * mu * (IC - 3.0)
      - 0.5 * mu * jnp.log(I3)
      + 0.5 * lam * (J - shift)**2
  )
  return energy_per_tet

# ============================================================
# 3) Build N independent instances
#    (same rest is OK; x differs -> independent grads/hessians)
# ============================================================
key = jax.random.PRNGKey(0)

r_batch = jnp.broadcast_to(rest_tet[None, :, :], (N, 4, 3))
noise   = 1e-3 * jax.random.normal(key, (N, 4, 3), dtype=jnp.float64)
x_batch = jnp.broadcast_to(curr_tet[None, :, :], (N, 4, 3)) + noise

# Hessian subset
r_h = r_batch[:Nh]
x_h = x_batch[:Nh]
x_flat_h = x_h.reshape(Nh, 12)

# ============================================================
# 4) Batched energy/grad/hess (correct semantics)
# ============================================================

# Energy: (N,)
energy_fn = jax.jit(lambda x, r: stable_neo_hookean_energy(x, r, mu, lam))

# Grad per tet: (N,4,3) via vmap(grad(single))
def single_energy(x_single, r_single):
  # x_single, r_single: (4,3) -> scalar
  return stable_neo_hookean_energy(x_single[None, :, :], r_single[None, :, :], mu, lam)[0]

grad_single = jax.grad(single_energy, argnums=0)
grad_fn = jax.jit(jax.vmap(grad_single, in_axes=(0, 0)))   # (N,4,3)

# Hessian per tet block: (Nh,12,12)
def single_energy_flat(x_flat, r_single):
  return single_energy(x_flat.reshape(4, 3), r_single)

hess_single = jax.hessian(single_energy_flat, argnums=0)   # (12,12)
hess_fn = jax.jit(jax.vmap(hess_single, in_axes=(0, 0)))   # (Nh,12,12)

# ============================================================
# 5) Warmup (compile)
# ============================================================
E = energy_fn(x_batch, r_batch); jax.block_until_ready(E)
G = grad_fn(x_batch, r_batch);   jax.block_until_ready(G)
H = hess_fn(x_flat_h, r_h);      jax.block_until_ready(H)

print("Warmup done.")
print("E shape:", E.shape, "G shape:", G.shape, "H shape:", H.shape)

# ============================================================
# 6) Timing
# ============================================================
def time_ms(fn, iters):
  out = fn()
  jax.block_until_ready(out)
  t0 = time.time()
  for _ in range(iters):
    out = fn()
  jax.block_until_ready(out)
  t1 = time.time()
  return (t1 - t0) * 1e3 / iters

energy_ms = time_ms(lambda: energy_fn(x_batch, r_batch), iters=50)
grad_ms   = time_ms(lambda: grad_fn(x_batch, r_batch),   iters=50)
hess_ms   = time_ms(lambda: hess_fn(x_flat_h, r_h),      iters=10)

print(f"\n=== Parallel independent-instance benchmark (JAX) ===")
print(f"Energy   over N={N:,}   : {energy_ms:.3f} ms / call   (output (N,))")
print(f"Grad     over N={N:,}   : {grad_ms:.3f} ms / call     (output (N,4,3))")
print(f"Hessian  over Nh={Nh:,} : {hess_ms:.3f} ms / call    (output (Nh,12,12))")
print(f"\nPer-instance:")
print(f"Energy  : {energy_ms * 1e6 / N:.3f} ns / instance")
print(f"Grad    : {grad_ms  * 1e6 / N:.3f} ns / instance")
print(f"Hessian : {hess_ms  * 1e6 / Nh:.3f} ns / instance  (Nh only)")

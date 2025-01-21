import os
os.environ['JAX_ENABLE_X64'] = 'True'  # Enable double precision

import time
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from functools import partial

########################################
# Setup: Constants, Data
########################################
print("JAX default backend:", jax.default_backend())

position = jnp.array([[0, 0, 1],
                      [1, 0, 2],
                      [0, 0, 0],
                      [0, 2, 0]], dtype=jnp.float64)

center = jnp.mean(position, axis=0)
position = position - center

tet_indices = jnp.array([[0, 1, 2, 3]], dtype=jnp.int64)
mu_np = 2000.0
lam_np = 1000.0
NUM_TETS = 79935  # We'll treat as static in the JIT call

# Single tet
tet_id = 0
rest_single_tet = position[tet_indices[tet_id]][None, ...]    # shape (1,4,3)
current_single_tet = (position[tet_indices[tet_id]] * 1.54)[None, ...]  # shape (1,4,3)

mu = jnp.array(mu_np, dtype=jnp.float64)
lam = jnp.array(lam_np, dtype=jnp.float64)

########################################
# 1) Define stable Neo-Hookean
########################################
def stable_neo_hookean_energy(current_tet_pos, rest_tet_pos, mu, lam):
    """
    current_tet_pos, rest_tet_pos: (T, 4, 3)
    Returns energy_per_tet: (T,)
    """
    x0 = rest_tet_pos[:, 0, :]
    x1 = rest_tet_pos[:, 1, :]
    x2 = rest_tet_pos[:, 2, :]
    x3 = rest_tet_pos[:, 3, :]
    X0 = x1 - x0
    X1 = x2 - x0
    X2 = x3 - x0
    B = jnp.stack([X0, X1, X2], axis=2)  # (T,3,3)
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
    F = jnp.stack([Y0, Y1, Y2], axis=2)
    FI = jnp.einsum('tij,tjk->tik', F, IB)

    J = jnp.linalg.det(FI)
    IC = jnp.sum(FI * FI, axis=(1, 2))
    I3 = IC + 1.0

    shift = 1.0 + 0.75 * (mu / lam)
    energy_per_tet = vol * (
        0.5 * mu * (IC - 3.0)
        - 0.5 * mu * jnp.log(I3)
        + 0.5 * lam * (J - shift)**2
    )
    return energy_per_tet

########################################
# 2) Batched energy with JIT
########################################
@partial(jax.jit, static_argnums=(4,))
def batch_tet_energy(current_tet_pos, rest_tet_pos, mu, lam, num_iterations):
    """
    Repeat a single tet `num_iterations` times and multiply each energy by a scalar.
    Sum the total energy.
    """
    x = jnp.tile(current_tet_pos, [num_iterations, 1, 1])   # shape (num_iterations, 4, 3)
    r = jnp.tile(rest_tet_pos, [num_iterations, 1, 1])
    multipliers = jnp.arange(1, num_iterations + 1, dtype=jnp.float64)
    energy_per_tet = stable_neo_hookean_energy(x, r, mu, lam)  # (num_iterations,)
    totalE = jnp.sum(energy_per_tet * multipliers)
    return totalE

########################################
# 3) JIT-compiled Gradient
########################################
@jax.jit
def batch_tet_energy_grad(x):
    """
    x: shape (1,4,3)
    returns gradient of batch_tet_energy w.r.t. x
    """
    return grad(lambda y: batch_tet_energy(y, rest_single_tet, mu, lam, NUM_TETS))(x)

########################################
# 4) JIT-compiled Hessian
########################################
# Option A: Flatten/unflatten inside a wrapper
def energy_wrapper(x_flat):
    x_reshaped = x_flat.reshape((1,4,3))
    return batch_tet_energy(x_reshaped, rest_single_tet, mu, lam, NUM_TETS)

@jax.jit
def batch_tet_energy_hessian(x_flat):
    """
    x_flat: shape (12,)
    returns Hessian of batch_tet_energy w.r.t x_flat
    """
    # Compose jacfwd(jacrev(energy_wrapper))
    return jacfwd(jacrev(energy_wrapper))(x_flat)


###################################################
# Warm-up calls
###################################################
# 1) Forward pass warm-up
for _ in range(10):
    _ = batch_tet_energy(current_single_tet, rest_single_tet, mu, lam, NUM_TETS)

# 2) Gradient warm-up
_ = batch_tet_energy_grad(current_single_tet)

# 3) Hessian warm-up
x0 = current_single_tet.reshape(-1)  # shape (12,)
_ = batch_tet_energy_hessian(x0)

jax.block_until_ready(_)

###################################################
# Timing the gradient
###################################################
t0 = time.time()
for _ in range(100):
    g_val = batch_tet_energy_grad(current_single_tet)
jax.block_until_ready(g_val)
t1 = time.time()
print(f"Gradient time: {(t1 - t0)*1e3/100:.4f} ms (for {NUM_TETS} tets)")
print("Gradient shape:", g_val.shape)  # (1,4,3)

###################################################
# Timing the Hessian
###################################################
t0 = time.time()
for _ in range(100):
    h_val = batch_tet_energy_hessian(x0)
jax.block_until_ready(h_val)
t1 = time.time()
print(f"Hessian time: {(t1 - t0)*1e3/100:.4f} ms (for {NUM_TETS} tets)")
print("Hessian shape:", h_val.shape)   # (12,12)

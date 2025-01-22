import os
os.environ['JAX_ENABLE_X64'] = 'True'  # Enable float64 before importing JAX

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
from functools import partial

print("JAX default backend:", jax.default_backend())

########################################
# Step 1: initialize data
########################################
position = np.array([[0, 0, 1],
                     [1, 0, 2],
                     [0, 0, 0],
                     [0, 2, 0]], dtype=np.float64)

tet_indices = np.array([[0, 1, 2, 3]], dtype=np.int64)

center = np.mean(position, axis=0)
position -= center

# Convert to jax arrays
position_j = jnp.array(position)
tets_j     = jnp.array(tet_indices)

# single tet
tet_id = 0
rest_tet_pos = position_j[tets_j[tet_id]]          # shape (4,3)
rest_tet_pos = rest_tet_pos[jnp.newaxis, ...]      # shape (1,4,3)
current_tet_pos = rest_tet_pos * 1.54              # shape (1,4,3)

NUM_TETS = 158800

########################################
# Step 2: define repulsive_energy
########################################
def repulsive_energy(points, alpha=2.0, beta=4.5, epsilon=1e-8):
  """
  points: shape (T,4,3)
  Returns: shape (T,) repulsive energies
  """
  p0 = points[:,0,:]  # (T,3)
  p1 = points[:,1,:]
  p2 = points[:,2,:]
  p3 = points[:,3,:]

  # T01
  delta01 = p1 - p0
  T01_norm = jnp.linalg.norm(delta01, axis=1, keepdims=True) + epsilon
  T01 = delta01 / T01_norm

  # T23
  delta23 = p3 - p2
  T23_norm = jnp.linalg.norm(delta23, axis=1, keepdims=True) + epsilon
  T23 = delta23 / T23_norm

  # We'll accumulate in a single array "r"
  r = jnp.zeros((points.shape[0],), dtype=points.dtype)

  # pairs = [ (T01, p0 - p2), (T01, p0 - p3), (T01, p1 - p2), (T01, p1 - p3),
  #           (T23, p2 - p0), (T23, p2 - p1), (T23, p3 - p0), (T23, p3 - p1) ]
  def compute_term(Ti, pj):
    dot_val = jnp.sum(Ti * pj, axis=1)        # (T,)
    dist    = jnp.linalg.norm(pj, axis=1) + epsilon
    return (dot_val**alpha) / (dist**beta)

  # We can sum them up directly
  r += compute_term(T01, (p0 - p2))
  r += compute_term(T01, (p0 - p3))
  r += compute_term(T01, (p1 - p2))
  r += compute_term(T01, (p1 - p3))
  r += compute_term(T23, (p2 - p0))
  r += compute_term(T23, (p2 - p1))
  r += compute_term(T23, (p3 - p0))
  r += compute_term(T23, (p3 - p1))

  # Add the 1 / ||p0 - p1||^2 term
  p0_p1 = p0 - p1
  p0_p1_norm_sq = jnp.sum(p0_p1 * p0_p1, axis=1) + epsilon
  r += 1.0 / p0_p1_norm_sq

  return r

########################################
# Step 3: define batch function
########################################
@partial(jit, static_argnums=(1,))  # num_iterations is 2nd argument
def batch_tet_energy(current_tet_pos, num_iterations):
  """
  current_tet_pos: shape (1,4,3)
  """
  # tile to (num_iterations,4,3)
  x = jnp.tile(current_tet_pos, [num_iterations,1,1])
  multipliers = jnp.arange(1, num_iterations+1, dtype=jnp.float64)
  energies = repulsive_energy(x) * multipliers
  return jnp.sum(energies)

# Warm-up (compile the function)
_ = batch_tet_energy(current_tet_pos, NUM_TETS)
jax.block_until_ready(_)

########################################
# Step 4: compiled gradient
########################################
@jax.jit
def batch_tet_grad(x):
  return grad(lambda z: batch_tet_energy(z, NUM_TETS))(x)

# warm-up grad
g_val = batch_tet_grad(current_tet_pos)
jax.block_until_ready(g_val)

# Timing gradient
t0 = time.time()
for _ in range(100):
  g_val = batch_tet_grad(current_tet_pos)
# jax.block_until_ready(g_val)
t1 = time.time()

grad_time_ms = (t1 - t0)*1e3 / 100.0
print(f"Gradient computation time for {NUM_TETS} tets: {grad_time_ms:.5f} ms")
print("Gradient shape:", g_val.shape)  # (1,4,3)

########################################
# Step 5: compiled Hessian
########################################
def energy_wrapper(x_flat):
  # x_flat -> (1,4,3)
  x_reshaped = x_flat.reshape((1,4,3))
  return batch_tet_energy(x_reshaped, NUM_TETS)

@jax.jit
def hessian_fn(x_flat):
  # Hessian via jacfwd(jacrev)
  return jacfwd(jacrev(energy_wrapper))(x_flat)

x0 = current_tet_pos.reshape(-1)  # shape (12,)

# warm-up Hessian
_ = hessian_fn(x0)
jax.block_until_ready(_)

# timing Hessian
t0 = time.time()
for _ in range(100):
  h_val = hessian_fn(x0)
# jax.block_until_ready(h_val)
t1 = time.time()

hess_time_ms = (t1 - t0) * 1000.0 / 100.0
print(f"Hessian computation time for {NUM_TETS} tets: {hess_time_ms:.5f} ms")
print("Hessian shape:", h_val.shape)  # (12,12)

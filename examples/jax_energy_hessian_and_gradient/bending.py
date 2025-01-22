import os
os.environ['JAX_ENABLE_X64'] = 'True'  # Optionally enable 64-bit floats programmatically

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

# Convert to JAX arrays
position_j = jnp.array(position)
tets_j = jnp.array(tet_indices)

# We pick the first tet
tet_id = 0

rest_position_tet = position_j[tets_j[tet_id]]    # shape (4, 3)
rest_position_tet = rest_position_tet[jnp.newaxis, ...]   # shape (1, 4, 3)
current_position_tet = (rest_position_tet * 1.54)          # shape (1, 4, 3)

# Some constants
NUM_TETS = 104065
bending_stiff = 1000.0

########################################
# Define angle, edgeTheta, bending
########################################

def angle(v, w, axis, eps=1e-8):
  """
  v, w, axis: shape (T, 3)
  Returns: shape (T,)
  """
  cross_vw = jnp.cross(v, w, axis=1)              # cross product along dim=1
  cross_dot_axis = jnp.sum(cross_vw * axis, axis=1)
  axis_norm = jnp.linalg.norm(axis, axis=1) + eps

  numerator = cross_dot_axis / axis_norm

  # denominator from your PyTorch code:
  #  sum(v * w) + ||v|| * ||w||
  dot_vw = jnp.sum(v * w, axis=1)
  norm_v = jnp.linalg.norm(v, axis=1)
  norm_w = jnp.linalg.norm(w, axis=1)
  denominator = dot_vw + (norm_v * norm_w)

  theta = 2.0 * jnp.arctan2(numerator, denominator)
  return theta

def edgeTheta(q0, q1, q2, q3):
  """
  Given a tet's 4 points (T,3), compute the angle.
  Each is shape (T,3).
  Returns: shape (T,)
  """
  n0 = jnp.cross(q0 - q2, q1 - q2, axis=1)  # shape (T,3)
  n1 = jnp.cross(q1 - q3, q0 - q3, axis=1)
  axis = (q1 - q0)                          # shape (T,3)
  theta = angle(n0, n1, axis)
  return theta

def bending(x, x_init, bendStiff):
  """
  x, x_init: (T,4,3)
  returns bending energy shape (T,)
  """
  x0 = x[:, 0, :]
  x1 = x[:, 1, :]
  x2 = x[:, 2, :]
  x3 = x[:, 3, :]
  t = edgeTheta(x0, x1, x2, x3)

  x_init0 = x_init[:, 0, :]
  x_init1 = x_init[:, 1, :]
  x_init2 = x_init[:, 2, :]
  x_init3 = x_init[:, 3, :]
  t_init = edgeTheta(x_init0, x_init1, x_init2, x_init3)

  delta_t_sq = (t - t_init)**2
  edge_len = jnp.linalg.norm(x_init1 - x_init0, axis=1)  # shape (T,)

  bend_energy = bendStiff * delta_t_sq * edge_len
  return bend_energy  # (T,)

########################################
# Batch function
########################################
@partial(jit, static_argnums=(3,))
def batch_tet_energy(curr_tet_pos, rest_tet_pos, bend_stiff, num_iterations):
  """
  Repeat a single tet num_iterations times, multiply each energy by a scalar,
  and sum them up.
  curr_tet_pos, rest_tet_pos: shape (1,4,3)
  """
  # tile to shape (num_iterations, 4, 3)
  x = jnp.tile(curr_tet_pos, [num_iterations, 1, 1])
  r = jnp.tile(rest_tet_pos, [num_iterations, 1, 1])
  multipliers = jnp.arange(1, num_iterations + 1, dtype=jnp.float64)

  # compute bending
  bend_energies = bending(x, r, bend_stiff)  # shape (num_iterations,)
  total_energy = jnp.sum(bend_energies * multipliers)
  return total_energy

# Warm-up call to compile
_ = batch_tet_energy(current_position_tet, rest_position_tet, bending_stiff, NUM_TETS)

########################################
# Gradient test
########################################
grad_fn = jit(
    lambda x: grad(lambda cpos: batch_tet_energy(cpos, rest_position_tet, bending_stiff, NUM_TETS))(x)
)

# Warm-up the gradient
g_val = grad_fn(current_position_tet)
_ = jax.block_until_ready(g_val)  # force eval

# Time the gradient
start_time = time.time()
for _ in range(100):
  g_val = grad_fn(current_position_tet)
_ = jax.block_until_ready(g_val)
end_time = time.time()

grad_time_ms = (end_time - start_time)*1000.0 / 100.0
print(f"Gradient computation time for {NUM_TETS} tets: {grad_time_ms:.5f} ms")
print("Gradient shape:", g_val.shape)  # Should be (1,4,3)

########################################
# Hessian test
########################################
# We'll flatten (1,4,3) -> (12,) for Hessian
def energy_wrapper(x_flat):
  x_reshaped = x_flat.reshape((1,4,3))
  return batch_tet_energy(x_reshaped, rest_position_tet, bending_stiff, NUM_TETS)

@jit
def hessian_fn(x_flat):
  # Hessian via jacfwd(jacrev)
  return jacfwd(jacrev(energy_wrapper))(x_flat)

x0 = current_position_tet.reshape(-1)  # shape (12,)

# Warm up
_ = hessian_fn(x0)
_ = jax.block_until_ready(_)

start_time = time.time()
for _ in range(100):
  h_val = hessian_fn(x0)
_ = jax.block_until_ready(h_val)
end_time = time.time()

hess_time_ms = (end_time - start_time)*1000.0 / 100.0
print(f"Hessian computation time for {NUM_TETS} tets: {hess_time_ms:.5f} ms")
print("Hessian shape:", h_val.shape)  # Should be (12,12)

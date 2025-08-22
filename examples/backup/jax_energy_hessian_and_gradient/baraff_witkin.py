import os
os.environ["JAX_ENABLE_X64"] = "True"  # Enable float64 precision before importing jax

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev
from functools import partial

print("JAX default backend:", jax.default_backend())

########################################
# 1) Initialize data
########################################
position = np.array([[0, 0, 1],
                     [1, 0, 2],
                     [0, 0, 0],
                     [0, 2, 0]], dtype=np.float64)

triangle_indices = np.array([[0, 1, 2]], dtype=np.int64)

center = np.mean(position, axis=0)
position -= center

# Convert to JAX arrays
position_j = jnp.array(position)
triangles_j = jnp.array(triangle_indices)

# We pick the first (and only) triangle
triangle_id = 0
rest_triangle_pos = position_j[triangles_j[triangle_id]]        # shape (3,3)
rest_triangle_pos = rest_triangle_pos[jnp.newaxis, ...]         # shape (1,3,3)
current_triangle_pos = rest_triangle_pos * 1.54                 # shape (1,3,3)

# Constants
NUM_TRIANGLES = 2000000
stretchS_val = 2000.0
shearS_val = 1000.0

stretchS = jnp.array(stretchS_val, dtype=jnp.float64)
shearS  = jnp.array(shearS_val,  dtype=jnp.float64)

########################################
# 2) Define compute_rest_shape
########################################
def compute_rest_shape(rest_triangle_pos):
  """
  rest_triangle_pos: (T, 3, 3)
  Returns M: (T, 2, 2)
    for each triangle, we produce a 2x2 rest shape matrix M
  """
  T = rest_triangle_pos.shape[0]

  v0 = rest_triangle_pos[:, 0, :]  # (T,3)
  v1 = rest_triangle_pos[:, 1, :]
  v2 = rest_triangle_pos[:, 2, :]
  v01 = v1 - v0
  v02 = v2 - v0

  # Compute normal
  normal = jnp.cross(v01, v02, axis=1)  # (T,3)
  normal_norm = jnp.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
  normal = normal / normal_norm  # (T,3)

  # target = (0, 1, 0)
  target = jnp.array([0.0, 1.0, 0.0], dtype=rest_triangle_pos.dtype)
  target = jnp.tile(target[None, :], [T, 1])  # (T,3)

  # axis = cross(normal, target)
  vec = jnp.cross(normal, target, axis=1)  # (T,3)

  # cos angle between normal and target
  cos_val = jnp.sum(normal * target, axis=1)  # (T,)

  # Construct skew-symmetric cross_vec for each triangle
  zero = jnp.zeros((T,1))
  cross_vec = jnp.stack([
    jnp.concatenate([zero, -vec[:, 2:3],  vec[:, 1:2]], axis=1),  # row0
    jnp.concatenate([ vec[:, 2:3], zero, -vec[:, 0:1]], axis=1),  # row1
    jnp.concatenate([-vec[:, 1:2], vec[:, 0:1], zero], axis=1)    # row2
  ], axis=1)  # (T,3,3)

  # rotation = I + cross_vec + cross_vec^2 / (1+cos)
  I = jnp.tile(jnp.eye(3)[None, :, :], [T, 1, 1])   # (T,3,3)
  cross_vec_sq = jnp.einsum('tij,tjk->tik', cross_vec, cross_vec)  # (T,3,3)
  denominator = (1.0 + cos_val + 1e-8)[:, None, None]
  rotation = I + cross_vec + cross_vec_sq / denominator  # (T,3,3)

  # Rotate v0, v1, v2
  rotate_uv0 = jnp.einsum('tij,tj->ti', rotation, v0)  # (T,3)
  rotate_uv1 = jnp.einsum('tij,tj->ti', rotation, v1)
  rotate_uv2 = jnp.einsum('tij,tj->ti', rotation, v2)

  # Keep only x,z => shape (T,2)
  uv0 = rotate_uv0[:, [0,2]]
  uv1 = rotate_uv1[:, [0,2]]
  uv2 = rotate_uv2[:, [0,2]]

  uv1_minus_uv0 = uv1 - uv0
  uv2_minus_uv0 = uv2 - uv0

  # M = [[ uv1x, uv2x ], [ uv1y, uv2y ]]
  M = jnp.stack([
    jnp.stack([uv1_minus_uv0[:,0], uv2_minus_uv0[:,0]], axis=1),
    jnp.stack([uv1_minus_uv0[:,1], uv2_minus_uv0[:,1]], axis=1)
  ], axis=1)  # (T,2,2)
  return M

########################################
# 3) Define Baraff-Witkin energy
########################################
def baraff_witkin_energy(x_init, x, stretchS, shearS, thickness=0.01, dt=0.0):
  """
  x_init, x : (T,3,3)
  returns (T,) containing energy for each triangle
  """
  # Current edge vectors
  x10 = x[:,1,:] - x[:,0,:]   # (T,3)
  x20 = x[:,2,:] - x[:,0,:]   # (T,3)
  # F shape: (T,3,2) = [x10, x20]^T
  # We'll stack them along axis=1 => shape (T,2,3),
  # then transpose => (T,3,2). Let's do it directly:
  F = jnp.stack([x10, x20], axis=1)  # (T,2,3)
  F = jnp.swapaxes(F, 1, 2)         # (T,3,2)

  # Get rest shape (T,2,2) and invert
  M = compute_rest_shape(x_init)    # (T,2,2)
  M_inv = jnp.linalg.inv(M)         # (T,2,2)

  # Deformation gradient F : (T,3,2) x (T,2,2) => (T,3,2)
  # batch matmul with bmm or einsum
  F_def = jnp.einsum('tij,tjk->tik', F, M_inv)  # (T,3,2)

  # F^T * F => (T,2,2)
  F_TF = jnp.einsum('tji, tjk->tik', F_def, F_def)

  I6 = F_TF[:, 0, 1]  # off-diagonal => shear
  shear_energy = I6**2

  # Norm of columns => (T,)
  I5u = jnp.linalg.norm(F_def[:,:,0], axis=1)  # (T,)
  I5v = jnp.linalg.norm(F_def[:,:,1], axis=1)
  stretch_energy = (I5u - 1.0)**2 + (I5v - 1.0)**2

  # area for each triangle in rest config
  v01 = x_init[:,1,:] - x_init[:,0,:]
  v02 = x_init[:,2,:] - x_init[:,0,:]
  cross_v01_v02 = jnp.cross(v01, v02, axis=1)
  area = thickness * jnp.linalg.norm(cross_v01_v02, axis=1) / 2.0

  # total = (stretchS * stretch_energy + shearS * shear_energy) * area * (dt^2)
  energy = (stretchS * stretch_energy + shearS * shear_energy) * area * (dt**2)
  return energy

########################################
# 4) Batch function with JIT
########################################
@partial(jax.jit, static_argnums=(4,))
def batch_triangle_energy(curr_triangle_pos, rest_triangle_pos, stretch_s, shear_s, num_iterations=NUM_TRIANGLES):
  """
  Repeat a single triangle `num_iterations` times,
  multiply each energy by [1..num_iterations] scaling,
  sum the total.
  """
  x = jnp.tile(curr_triangle_pos, [num_iterations, 1, 1])   # (num_iterations, 3, 3)
  r = jnp.tile(rest_triangle_pos, [num_iterations, 1, 1])   # (num_iterations, 3, 3)
  multipliers = jnp.arange(1, num_iterations+1, dtype=jnp.float64)

  energies = baraff_witkin_energy(r, x, stretch_s, shear_s)
  return jnp.sum(energies * multipliers)

########################################
# Warm-up
########################################
_ = batch_triangle_energy(current_triangle_pos, rest_triangle_pos, stretchS, shearS, NUM_TRIANGLES)
jax.block_until_ready(_)

########################################
# 5) Gradient test
########################################
# We'll define a small function that returns gradient
@jax.jit
def batch_triangle_grad(x):
  return grad(lambda cpos:
          batch_triangle_energy(cpos, rest_triangle_pos, stretchS, shearS, NUM_TRIANGLES)
          )(x)

# Warm up
g_val = batch_triangle_grad(current_triangle_pos)
jax.block_until_ready(g_val)

# Time the gradient
t0 = time.time()
for _ in range(100):
  g_val = batch_triangle_grad(current_triangle_pos)
# jax.block_until_ready(g_val)
t1 = time.time()

grad_time_ms = (t1 - t0)*1e3 / 100.0
print(f"Gradient computation time for {NUM_TRIANGLES} triangles: {grad_time_ms:.5f} ms")
print("Gradient shape:", g_val.shape)  # (1,3,3)

########################################
# 6) Hessian test
########################################
# Flatten from (1,3,3) -> (9,)
def energy_wrapper(x_flat):
  x_reshaped = x_flat.reshape((1,3,3))
  return batch_triangle_energy(x_reshaped, rest_triangle_pos, stretchS, shearS, NUM_TRIANGLES)

@jax.jit
def hessian_fn(x_flat):
  # Hessian = jacfwd(jacrev(energy_wrapper))
  return jacfwd(jacrev(energy_wrapper))(x_flat)

x0 = current_triangle_pos.reshape(-1)  # shape (9,)

# Warm up
_ = hessian_fn(x0)
jax.block_until_ready(_)

# Time
t0 = time.time()
for _ in range(100):
  h_val = hessian_fn(x0)
# jax.block_until_ready(h_val)
t1 = time.time()

hess_time_ms = (t1 - t0) * 1000 / 100.0
print(f"Hessian computation time for {NUM_TRIANGLES} triangles: {hess_time_ms:.5f} ms")
print("Hessian shape:", h_val.shape)  # (9,9)

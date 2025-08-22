import torch
import torch.autograd.functional as F_autograd
import numpy as np

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

########################################
# Step 1: initialize data
########################################
position = np.array([[0, 0, 1], [1, 0, 2], [0, 0, 0], [0, 2, 0]], dtype = np.float64)
triangle_indices = np.array([[0, 1, 2]])
center = np.mean(position, axis=0)
position -= center

########################################
# Initialize Tensors on GPU
########################################

rest_position = torch.tensor(position, dtype=torch.float64, device=device, requires_grad=False)
current_position = (rest_position.clone() * 1.54).requires_grad_(True).to(device)

triangles = torch.tensor(triangle_indices, dtype=torch.int64, device=device)

T = triangles.shape[0]

# Gather the trianglerahedron vertices positions
# Shape: (T, 3, 3)
rest_triangle_pos = rest_position[triangles]       # T x 3 x 3
current_triangle_pos = current_position[triangles] # T x 3 x 3

# Ensure all tensors are on the correct device
rest_triangle_pos = rest_triangle_pos.to(device)
current_triangle_pos = current_triangle_pos.to(device)

########################################
# Define the baraff-witkin energy
########################################

def compute_rest_shape(rest_triangle_pos):
  # Number of triangles
  T = rest_triangle_pos.shape[0]

  # Compute edge vectors
  v0 = rest_triangle_pos[:, 0, :]  # (T, 3)
  v1 = rest_triangle_pos[:, 1, :]  # (T, 3)
  v2 = rest_triangle_pos[:, 2, :]  # (T, 3)
  v01 = v1 - v0  # (T, 3)
  v02 = v2 - v0  # (T, 3)
  # Compute normal vectors
  normal = torch.cross(v01, v02, dim=1)  # (T, 3)
  normal_norm = normal.norm(dim=1, keepdim=True) + 1e-8  # (T, 1) Adding epsilon to avoid division by zero
  normal = normal / normal_norm  # (T, 3)
  # Define target vector [0, 1, 0]
  target = torch.tensor([0.0, 1.0, 0.0], dtype=rest_triangle_pos.dtype, device=rest_triangle_pos.device)
  target = target.unsqueeze(0).expand(T, -1)  # (T, 3)
  # Compute vector orthogonal to normal and target
  vec = torch.cross(normal, target, dim=1)  # (T, 3)
  # Compute cosine of angle between normal and target
  cos = torch.sum(normal * target, dim=1)  # (T,)
  # Initialize rotation matrices as identity
  rotation = torch.eye(3, dtype=rest_triangle_pos.dtype, device=rest_triangle_pos.device).unsqueeze(0).expand(T, -1, -1)  # (T, 3, 3)
  # Create skew-symmetric cross_vec matrices
  # For each T, cross_vec is [[0, -vec_z, vec_y],
  #                          [vec_z, 0, -vec_x],
  #                          [-vec_y, vec_x, 0]]
  zero = torch.zeros(T, 1, device=rest_triangle_pos.device)
  cross_vec = torch.stack([
      torch.cat([zero, -vec[:, 2].unsqueeze(1), vec[:, 1].unsqueeze(1)], dim=1),
      torch.cat([vec[:, 2].unsqueeze(1), zero, -vec[:, 0].unsqueeze(1)], dim=1),
      torch.cat([-vec[:, 1].unsqueeze(1), vec[:, 0].unsqueeze(1), zero], dim=1)
  ], dim=1)  # (T, 3, 3)
  # Update rotation matrices: R = I + cross_vec + (cross_vec @ cross_vec) / (1 + cos)
  denominator = (1.0 + cos).unsqueeze(1).unsqueeze(2) + 1e-8  # (T, 1, 1)
  cross_vec_sq = torch.bmm(cross_vec, cross_vec)  # (T, 3, 3)
  rotation = rotation + cross_vec + cross_vec_sq / denominator  # (T, 3, 3)
  # Apply rotation to v0, v1, v2
  rotate_uv0 = torch.bmm(rotation, v0.unsqueeze(2)).squeeze(2)  # (T, 3)
  rotate_uv1 = torch.bmm(rotation, v1.unsqueeze(2)).squeeze(2)  # (T, 3)
  rotate_uv2 = torch.bmm(rotation, v2.unsqueeze(2)).squeeze(2)  # (T, 3)
  # Project to UV (take x and z coordinates)
  uv0 = rotate_uv0[:, [0, 2]]  # (T, 2)
  uv1 = rotate_uv1[:, [0, 2]]  # (T, 2)
  uv2 = rotate_uv2[:, [0, 2]]  # (T, 2)
  # Compute uv1 - uv0 and uv2 - uv0
  uv1_minus_uv0 = uv1 - uv0  # (T, 2)
  uv2_minus_uv0 = uv2 - uv0  # (T, 2)
  # Construct M matrix for each triangle: [[uv1_m0, uv2_m0], [uv1_m1, uv2_m1]]
  M = torch.stack([
      torch.stack([uv1_minus_uv0[:, 0], uv2_minus_uv0[:, 0]], dim=1),
      torch.stack([uv1_minus_uv0[:, 1], uv2_minus_uv0[:, 1]], dim=1)
  ], dim=1)  # (T, 2, 2)
  return M  # (T, 2, 2)

def baraff_witkin_energy(x_init, x, stretchS, shearS, thickness = 0.01, dt = 0.):
  # Compute edge vectors in current configuration
  x10 = x[:, 1, :] - x[:, 0, :]  # (T, 3)
  x20 = x[:, 2, :] - x[:, 0, :]  # (T, 3)
  # Construct F matrix: [ [x10], [x20] ] for each triangle
  F = torch.stack([x10, x20], dim=1)  # (T, 2, 3)
  # Transpose F to shape (T, 3, 2)
  F = F.transpose(1, 2)  # (T, 3, 2)
  # Compute rest shape matrix M and its inverse
  M = compute_rest_shape(x_init)  # (T, 2, 2)
  M_inv = torch.linalg.inv(M)     # (T, 2, 2)
  # Deformation gradient: F @ M_inv
  F = torch.bmm(F, M_inv)  # (T, 3, 2)
  # Compute F^T @ F
  F_TF = torch.bmm(F.transpose(1, 2), F)  # (T, 2, 2)
  # Compute I6 = [1, 0] * F_TF * [0, 1]^T = F_TF[:, 0, 1]
  I6 = F_TF[:, 0, 1]  # (T,)
  # Compute shear_energy = I6^2
  shear_energy = I6 ** 2  # (T,)
  # Compute I5u = ||F[:, :, 0]|| (Euclidean norm)
  I5u = F[:, :, 0].norm(dim=1)  # (T,)
  # Compute I5v = ||F[:, :, 1]|| (Euclidean norm)
  I5v = F[:, :, 1].norm(dim=1)  # (T,)
  # Compute stretch_energy = (I5u - 1)^2 + (I5v - 1)^2
  stretch_energy = (I5u - 1.0) ** 2 + (I5v - 1.0) ** 2  # (T,)
  # Compute area for each triangle in rest configuration
  v01 = x_init[:, 1, :] - x_init[:, 0, :]  # (T, 3)
  v02 = x_init[:, 2, :] - x_init[:, 0, :]  # (T, 3)
  cross = torch.cross(v01, v02, dim=1)       # (T, 3)
  area = thickness * cross.norm(dim=1) / 2.0  # (T,)
  # Compute total energy
  # energy = (stretchS * stretch_energy + shearS * shear_energy) * area * dt^2
  energy = (stretchS * stretch_energy + shearS * shear_energy) * area * (dt ** 2)  # (T,)
  return energy  # (T,)

stretchS = torch.tensor(2000.0, dtype=torch.float64, device=device, requires_grad=False)
shearS = torch.tensor(1000.0, dtype=torch.float64, device=device, requires_grad=False)

# Suppose we pick the first triangle
triangle_id = 0

# Extract the single triangle positions (size: 1x3x3)
rest_single_triangle = rest_triangle_pos[triangle_id:triangle_id+1]
current_single_triangle = current_triangle_pos[triangle_id:triangle_id+1].clone().detach().requires_grad_(True)

NUM_TRIANGLES = 20000
# Move multipliers to device inside the function to ensure they are on GPU
def batch_triangle_energy(current_triangle_pos, rest_triangle_pos, stretch_s, shear_s, num_iterations=NUM_TRIANGLES):
  x = current_triangle_pos.repeat(num_iterations, 1, 1)  # Shape: (num_iterations, 3, 3)
  rest = rest_triangle_pos.repeat(num_iterations, 1, 1)  # Shape: (num_iterations, 3, 3)
  multipliers = torch.arange(1, num_iterations + 1, dtype=torch.float64, device=device)  # Shape: (num_iterations,)
  energies = baraff_witkin_energy(x, rest, stretch_s, shear_s) * multipliers  # Shape: (num_iterations,)
  totalE = energies.sum()  # Scalar
  return totalE

# Compile the batch energy function using torch.compile
compiled_batch_triangle_energy = torch.compile(batch_triangle_energy)

# -------------------------
# Gradient Computation
# -------------------------

import time

# Warm-up GPU
for _ in range(10):
  _ = compiled_batch_triangle_energy(current_single_triangle, rest_single_triangle, stretchS, shearS)
_ = torch.autograd.grad(
  compiled_batch_triangle_energy(current_single_triangle, rest_single_triangle, stretchS, shearS),
  current_single_triangle,
  create_graph=True
)

torch.cuda.synchronize()  # Ensure all operations are complete before timing
start_time = time.time()
grad = None
for _ in range(100):
  # Compute gradient
  grad = torch.autograd.grad(
    compiled_batch_triangle_energy(current_single_triangle, rest_single_triangle, stretchS, shearS),
    current_single_triangle,
    create_graph=True
  )

torch.cuda.synchronize()  # Wait for the gradient computation to finish
end_time = time.time()

grad_time = end_time - start_time
print(f"Gradient computation time for {NUM_TRIANGLES} triangles: {grad_time* 1000.0 / 100.0:.5f} ms")
# print(grad)

# -------------------------
# Hessian Computation
# -------------------------

# Define a wrapper function for Hessian computation
def energy_wrapper(x_flat):
  # x_flat: (12,) -> reshape to (1, 3, 3)
  x_reshaped = x_flat.view(1, 3, 3)
  return compiled_batch_triangle_energy(x_reshaped, rest_single_triangle, stretchS, shearS, num_iterations=NUM_TRIANGLES)

# Prepare the input for Hessian computation
x_flat = current_single_triangle.view(-1).clone().detach().requires_grad_(True).to(device)  # Shape: (12,)

# Warm-up GPU for Hessian computation
for _ in range(10):
  _ = energy_wrapper(x_flat).backward()

# just in case anything needs to be cached
_ =  F_autograd.hessian(energy_wrapper, x_flat)

torch.cuda.synchronize()  # Ensure all operations are complete before timing
start_time = time.time()
hessian = None
for _ in range(100):
  # Compute Hessian
  hessian = F_autograd.hessian(energy_wrapper, x_flat)

torch.cuda.synchronize()  # Wait for the Hessian computation to finish
end_time = time.time()

hess_time = end_time - start_time
print(f"Hessian computation time for {NUM_TRIANGLES} triangles: {hess_time*1000 / 100.0:.5f} ms")
print("Hessian shape:", hessian.shape)   # Should be (12, 12)
# print(hessian.detach().cpu().numpy())

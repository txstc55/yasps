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
tet_indices = np.array([[0, 1, 2, 3]])
center = np.mean(position, axis=0)
position -= center

########################################
# Initialize Tensors on GPU
########################################

rest_position = torch.tensor(position, dtype=torch.float64, device=device, requires_grad=False)
current_position = (rest_position.clone() * 1.54).requires_grad_(True).to(device)

tets = torch.tensor(tet_indices, dtype=torch.int64, device=device)

T = tets.shape[0]

# Gather the tetrahedron vertices positions
# Shape: (T, 4, 3)
rest_tet_pos = rest_position[tets]       # T x 4 x 3
current_tet_pos = current_position[tets] # T x 4 x 3

# Ensure all tensors are on the correct device
rest_tet_pos = rest_tet_pos.to(device)
current_tet_pos = current_tet_pos.to(device)

########################################
# Define the stable neo-hookean energy
########################################

def repulsive_energy(points, alpha = 2.0, beta = 4.5, epsilon=1e-8):
  # Extract points
  p0 = points[:, 0, :]  # (T, 3)
  p1 = points[:, 1, :]  # (T, 3)
  p2 = points[:, 2, :]  # (T, 3)
  p3 = points[:, 3, :]  # (T, 3)

  # Compute T01 and T23
  delta01 = p1 - p0  # (T, 3)
  delta23 = p3 - p2  # (T, 3)
  T01_norm = torch.norm(delta01, dim=1, keepdim=True) + epsilon  # (T, 1)
  T23_norm = torch.norm(delta23, dim=1, keepdim=True) + epsilon  # (T, 1)
  T01 = delta01 / T01_norm  # (T, 3)
  T23 = delta23 / T23_norm  # (T, 3)

  # Compute pairwise differences
  pairs = [
    (T01, p0 - p2),
    (T01, p0 - p3),
    (T01, p1 - p2),
    (T01, p1 - p3),
    (T23, p2 - p0),
    (T23, p2 - p1),
    (T23, p3 - p0),
    (T23, p3 - p1)
  ]
  # Initialize repulsive energy
  r = torch.zeros(points.shape[0], dtype=points.dtype, device=points.device)  # (T,)
  for Ti, pj in pairs:
    dot_product = torch.sum(Ti * pj, dim=1)  # (T,)
    distance = torch.norm(pj, dim=1) + epsilon  # (T,)
    term = (dot_product.pow(alpha) / distance.pow(beta))  # (T,)
    r += term  # (T,)

  # Add inverse of squared distance between p0 and p1
  p0_p1 = p0 - p1  # (T, 3)
  p0_p1_norm_sq = torch.sum(p0_p1 * p0_p1, dim=1) + epsilon  # (T,)
  r += 1.0 / p0_p1_norm_sq  # (T,)
  return r  # (T,)

# Suppose we pick the first tet
tet_id = 0

# Extract the single tet positions (size: 1x4x3)
rest_single_tet = rest_tet_pos[tet_id:tet_id+1]
current_single_tet = current_tet_pos[tet_id:tet_id+1].clone().detach().requires_grad_(True)

NUM_TETS = 158800


# Move multipliers to device inside the function to ensure they are on GPU
def batch_tet_energy(current_tet_pos, num_iterations=NUM_TETS):
  x = current_tet_pos.repeat(num_iterations, 1, 1)  # Shape: (num_iterations, 4, 3)
  multipliers = torch.arange(1, num_iterations + 1, dtype=torch.float64, device=device)  # Shape: (num_iterations,)
  energies = repulsive_energy(x) * multipliers  # Shape: (num_iterations,)
  totalE = energies.sum()  # Scalar
  return totalE

# Compile the batch energy function using torch.compile
compiled_batch_tet_energy = torch.compile(batch_tet_energy)

# -------------------------
# Gradient Computation
# -------------------------

import time

# Warm-up GPU
for _ in range(10):
  _ = compiled_batch_tet_energy(current_single_tet)
_ = torch.autograd.grad(
  compiled_batch_tet_energy(current_single_tet),
  current_single_tet,
  create_graph=True
)

torch.cuda.synchronize()  # Ensure all operations are complete before timing
start_time = time.time()
grad = None
for _ in range(100):
  # Compute gradient
  grad = torch.autograd.grad(
    compiled_batch_tet_energy(current_single_tet),
    current_single_tet,
    create_graph=True
  )

torch.cuda.synchronize()  # Wait for the gradient computation to finish
end_time = time.time()

grad_time = end_time - start_time
print(f"Gradient computation time for {NUM_TETS} tets: {grad_time* 1000.0 / 100.0:.5f} ms")
# print(grad)

# -------------------------
# Hessian Computation
# -------------------------

# Define a wrapper function for Hessian computation
def energy_wrapper(x_flat):
  # x_flat: (12,) -> reshape to (1, 4, 3)
  x_reshaped = x_flat.view(1, 4, 3)
  return compiled_batch_tet_energy(x_reshaped, num_iterations=NUM_TETS)

# Prepare the input for Hessian computation
x_flat = current_single_tet.view(-1).clone().detach().requires_grad_(True).to(device)  # Shape: (12,)

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
print(f"Hessian computation time for {NUM_TETS} tets: {hess_time*1000 / 100.0:.5f} ms")
print("Hessian shape:", hessian.shape)   # Should be (12, 12)
# print(hessian.detach().cpu().numpy())

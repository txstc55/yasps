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

def angle(v, w, axis, eps=1e-8):
  # Compute cross product v x w
  cross = torch.cross(v, w, dim=1)  # Shape: (T, 3)
  # Compute (v x w) · axis
  cross_dot_axis = torch.sum(cross * axis, dim=1)  # Shape: (T,)
  # Compute ||axis||
  axis_norm = torch.norm(axis, dim=1) + eps  # Shape: (T,)
  # Compute numerator for atan2
  numerator = cross_dot_axis / axis_norm  # Shape: (T,)
  # Compute denominator for atan2
  denominator = torch.sum(v * w, dim=1) + torch.norm(v, dim=1) * torch.norm(w, dim=1)  # Shape: (T,)
  # Compute theta
  theta = 2.0 * torch.atan2(numerator, denominator)  # Shape: (T,)
  return theta

def edgeTheta(q0, q1, q2, q3):
  # Compute normals
  n0 = torch.cross(q0 - q2, q1 - q2, dim=1)  # Shape: (T, 3)
  n1 = torch.cross(q1 - q3, q0 - q3, dim=1)  # Shape: (T, 3)
  # Compute axis
  axis = q1 - q0  # Shape: (T, 3)
  # Compute angle
  theta = angle(n0, n1, axis)  # Shape: (T,)
  return theta

def bending(x, x_init, bendStiff):
  # Extract vertices from current positions
  x0 = x[:, 0, :]  # Shape: (T, 3)
  x1 = x[:, 1, :]  # Shape: (T, 3)
  x2 = x[:, 2, :]  # Shape: (T, 3)
  x3 = x[:, 3, :]  # Shape: (T, 3)
  # Compute current bending angle
  t = edgeTheta(x0, x1, x2, x3)  # Shape: (T,)
  # Extract vertices from rest positions
  x_init0 = x_init[:, 0, :]  # Shape: (T, 3)
  x_init1 = x_init[:, 1, :]  # Shape: (T, 3)
  x_init2 = x_init[:, 2, :]  # Shape: (T, 3)
  x_init3 = x_init[:, 3, :]  # Shape: (T, 3)
  # Compute rest bending angle
  t_init = edgeTheta(x_init0, x_init1, x_init2, x_init3)  # Shape: (T,)
  # Compute (t - t_init)^2
  delta_t_sq = (t - t_init) ** 2  # Shape: (T,)
  # Compute ||x_init1 - x_init0||
  norm_x_init1_x_init0 = torch.norm(x_init1 - x_init0, dim=1)  # Shape: (T,)
  # Compute bending energy
  bend_energy = bendStiff * delta_t_sq * norm_x_init1_x_init0  # Shape: (T,)
  return bend_energy  # Shape: (T,)

bending_stiff = 1000.0
# Suppose we pick the first tet
tet_id = 0
# Extract the single tet positions (size: 1x4x3)
rest_single_tet = rest_tet_pos[tet_id:tet_id+1]
current_single_tet = current_tet_pos[tet_id:tet_id+1].clone().detach().requires_grad_(True)

NUM_TETS = 104065


# Move multipliers to device inside the function to ensure they are on GPU
def batch_tet_energy(current_tet_pos, rest_tet_pos, bending_stiff, num_iterations=NUM_TETS):
  x = current_tet_pos.repeat(num_iterations, 1, 1)  # Shape: (num_iterations, 4, 3)
  rest = rest_tet_pos.repeat(num_iterations, 1, 1)  # Shape: (num_iterations, 4, 3)
  multipliers = torch.arange(1, num_iterations + 1, dtype=torch.float64, device=device, requires_grad = False)  # Shape: (num_iterations,)
  energies = bending(x, rest, bending_stiff) * multipliers  # Shape: (num_iterations,)
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
  _ = compiled_batch_tet_energy(current_single_tet, rest_single_tet, bending_stiff)
_ = torch.autograd.grad(
  compiled_batch_tet_energy(current_single_tet, rest_single_tet, bending_stiff),
  current_single_tet,
  create_graph=True
)

torch.cuda.synchronize()  # Ensure all operations are complete before timing
start_time = time.time()
grad = None
for _ in range(100):
  # Compute gradient
  grad = torch.autograd.grad(
    compiled_batch_tet_energy(current_single_tet, rest_single_tet, bending_stiff),
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
  return compiled_batch_tet_energy(x_reshaped, rest_single_tet, bending_stiff, num_iterations=NUM_TETS)

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
print(f"Hessian computation time for {NUM_TETS} tets: {hess_time * 1000 / 100.0:.5f} ms")
print("Hessian shape:", hessian.shape)   # Should be (12, 12)
# print(hessian.detach().cpu().numpy())

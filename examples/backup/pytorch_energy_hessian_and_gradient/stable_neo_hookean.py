import torch
import torch.autograd.functional as F_autograd
import numpy as np
import time

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

########################################
# Step 1: initialize data
########################################
rest_position_np = np.array([
    [0.8168341,  -0.57334204, -0.28303627],
    [0.9386871,  -0.60033335, -0.10815139],
    [0.93494301, -0.73053988, -0.27055541],
    [1.00758576, -0.58084046, -0.37639869]
], dtype=np.float64)

current_position_np = np.array([
    [0.8168341,   0.34925004, -0.53558849],
    [0.9386871,   0.40100103,  0.4596684 ],
    [0.93494301,  0.44387604,  0.64020529],
    [1.00758576, -0.53429907,  0.43997285]
], dtype=np.float64)

tet_indices = np.array([[0, 1, 2, 3]])

########################################
# Initialize Tensors on GPU
########################################
rest_position = torch.tensor(rest_position_np, dtype=torch.float64, device=device, requires_grad=False)
current_position = torch.tensor(current_position_np, dtype=torch.float64, device=device, requires_grad=True)
tets = torch.tensor(tet_indices, dtype=torch.int64, device=device)

rest_tet_pos = rest_position[tets]       # Shape: (T, 4, 3)
current_tet_pos = current_position[tets] # Shape: (T, 4, 3)

########################################
# Stable Neo-Hookean Energy
########################################
def stable_neo_hookean_energy(current_tet_pos, rest_tet_pos, mu, lam):
    x0 = rest_tet_pos[:, 0, :]
    x1 = rest_tet_pos[:, 1, :]
    x2 = rest_tet_pos[:, 2, :]
    x3 = rest_tet_pos[:, 3, :]
    X0 = x1 - x0
    X1 = x2 - x0
    X2 = x3 - x0
    B = torch.stack([X0, X1, X2], dim=2)
    detB = torch.linalg.det(B)
    vol = detB / 6.0
    IB = torch.linalg.inv(B)

    y0 = current_tet_pos[:, 0, :]
    y1 = current_tet_pos[:, 1, :]
    y2 = current_tet_pos[:, 2, :]
    y3 = current_tet_pos[:, 3, :]
    Y0 = y1 - y0
    Y1 = y2 - y0
    Y2 = y3 - y0
    F = torch.stack([Y0, Y1, Y2], dim=2)

    FI = torch.bmm(F, IB)
    J = torch.linalg.det(FI)
    IC = torch.sum(FI * FI, dim=[1, 2])
    I3 = IC + 1.0
    shift = 1.0 + 0.75 * (mu / lam)

    energy = vol * (0.5 * mu * (IC - 3.0) - 0.5 * mu * torch.log(I3) + 0.5 * lam * (J - shift) ** 2)
    return energy

########################################
# Batch Energy Function
########################################
def batch_tet_energy(current_tet_pos, rest_tet_pos, mu, lam, num_iterations=1):
    x = current_tet_pos.repeat(num_iterations, 1, 1)
    rest = rest_tet_pos.repeat(num_iterations, 1, 1)
    multipliers = torch.arange(1, num_iterations + 1, dtype=torch.float64, device=device)
    energies = stable_neo_hookean_energy(x, rest, mu, lam) * multipliers
    print("Energy check")
    print(energies[0])
    return energies.sum()

########################################
# Prepare Inputs for Gradient/Hessian
########################################
mu = torch.tensor(2000.0, dtype=torch.float64, device=device)
lam = torch.tensor(1000.0, dtype=torch.float64, device=device)

rest_single_tet = rest_tet_pos[0:1]
current_single_tet = current_tet_pos[0:1].clone().detach().requires_grad_(True)
NUM_TETS = 1

########################################
# Gradient Computation
########################################
torch.cuda.synchronize()
start_time = time.time()

for _ in range(1):
    grad = torch.autograd.grad(
        batch_tet_energy(current_single_tet, rest_single_tet, mu, lam, num_iterations=NUM_TETS),
        current_single_tet,
        create_graph=True
    )

torch.cuda.synchronize()
end_time = time.time()

print(f"Gradient computation time for {NUM_TETS} tets: {(end_time - start_time) * 1000 / 100:.5f} ms")
print("Gradient:", grad[0].detach().cpu().numpy())

########################################
# Hessian Computation
########################################
def energy_wrapper(x_flat):
    x_reshaped = x_flat.view(1, 4, 3)
    return batch_tet_energy(x_reshaped, rest_single_tet, mu, lam, num_iterations=NUM_TETS)

x_flat = current_single_tet.view(-1).clone().detach().to(device).requires_grad_(True)

torch.cuda.synchronize()
start_time = time.time()
hessian = F_autograd.hessian(energy_wrapper, x_flat)
torch.cuda.synchronize()
end_time = time.time()

print(f"Hessian computation time for {NUM_TETS} tets: {(end_time - start_time) * 1000:.5f} ms")
print("Hessian shape:", hessian.shape)
print(hessian.detach().cpu().numpy())

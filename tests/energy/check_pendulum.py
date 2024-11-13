from sympy import symbols, Matrix, diff, lambdify, sqrt, hessian, simplify, Function
import numpy as np

# Number of segments
N = 5  # Adjust this value as needed

# Define symbols for w_i and z_i
w = [symbols(f'w_{i}') for i in range(N)]
z = [symbols(f'z_{i}') for i in range(N)]

# Length of each segment
length = [1.0 for _ in range(N)]

# Target position
target = Matrix([5.0, 5.0])  # Replace with your desired target coordinates

# Function to compute rotation matrix from w and z
def rotation_matrix(w_i, z_i):
    return Matrix([
        [w_i ** 2 - z_i ** 2, -2 * w_i * z_i],
        [2 * w_i * z_i, w_i ** 2 - z_i ** 2]
    ])

# Initialize lists to store rotation matrices and end positions
R_local = [None] * N
R_global = [None] * N
e = [None] * N  # End point positions

# For segment 0 (the first segment)
R_local[0] = rotation_matrix(w[0], z[0])
R_global[0] = R_local[0]
e[0] = R_global[0] * Matrix([0, -length[0]])

# For segments 1 to N-1
for i in range(1, N):
    R_local[i] = rotation_matrix(w[i], z[i])
    R_global[i] = R_global[i - 1] * R_local[i]
    e[i] = R_global[i] * Matrix([0, -length[i]]) + e[i - 1]

# Compute the energy (norm of the difference)
E = sqrt((e[N - 1][0] - target[0])**2 + (e[N - 1][1] - target[1])**2)

# Variables for differentiation
vars = w + z  # Concatenate lists of w_i and z_i

# Compute gradient
grad_E = [diff(E, var) for var in vars]

# Compute Hessian
Hessian_E = hessian(E, vars)

# Substitute numerical values for initial w_i and z_i
# Initial numerical values (adjust as needed)
w_num = [np.cos(0) for _ in range(N)]
z_num = [np.sin(0) for _ in range(N)]

# Modify specific initial values (if any)
w_num[0] = np.cos(np.pi / 8)
z_num[0] = np.sin(np.pi / 8)
w_num[1] = np.cos(-np.pi / 8)
z_num[1] = np.sin(-np.pi / 8)
w_num[2] = np.cos(-np.pi / 8)
z_num[2] = np.sin(-np.pi / 8)
# The rest remain at initial 0

# Create a dictionary for substitutions
subs_dict = {}
for i in range(N):
    subs_dict[w[i]] = w_num[i]
    subs_dict[z[i]] = z_num[i]

# Evaluate the energy, gradient, and Hessian numerically
E_val = E.evalf(subs=subs_dict)
grad_E_val = [expr.evalf(subs=subs_dict) for expr in grad_E]
Hessian_E_val = Hessian_E.evalf(subs=subs_dict)

# Print the results
print("Energy (E):", E_val)
print("\nGradient of Energy:")
for var, val in zip(vars, grad_E_val):
    print(f"dE/d{var} =", val)

print("\nHessian of Energy:")
print(Hessian_E_val)

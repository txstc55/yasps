import sympy as sp
import numpy as np

# Define symbolic variables
mu, lam, vol = sp.symbols('mu lam vol')
# Create symbolic variables for tet_positions (X)
x0_x, x0_y, x0_z = sp.symbols('x0_x x0_y x0_z')
x1_x, x1_y, x1_z = sp.symbols('x1_x x1_y x1_z')
x2_x, x2_y, x2_z = sp.symbols('x2_x x2_y x2_z')
x3_x, x3_y, x3_z = sp.symbols('x3_x x3_y x3_z')

# Positions of the nodes in the tetrahedron
row0 = sp.Matrix([x0_x, x0_y, x0_z])
row1 = sp.Matrix([x1_x, x1_y, x1_z])
row2 = sp.Matrix([x2_x, x2_y, x2_z])
row3 = sp.Matrix([x3_x, x3_y, x3_z])

# Compute x0, x1, x2 as differences
x0 = row1 - row0  # Edge vector from node 0 to node 1
x1 = row2 - row0  # Edge vector from node 0 to node 2
x2 = row3 - row0  # Edge vector from node 0 to node 3

# Assemble the deformation gradient F
F = sp.Matrix.vstack(x0.T, x1.T, x2.T)

# Define IB as a symbolic 3x3 matrix (inverse of the rest configuration matrix)
IB_symbols = {}
for i in range(3):
    for j in range(3):
        key = f'IB_{i+1}{j+1}'
        IB_symbols[key] = sp.Symbol(key)
IB = sp.Matrix(3, 3, lambda i, j: IB_symbols[f'IB_{i+1}{j+1}'])

# Compute FI (modified deformation gradient)
FI = F.T * IB

# Compute determinant J of FI
J = FI.det()

# Compute the first invariant IC
IC = (FI.T * FI).trace()

# Compute I3 (non-standard term as per your function)
I3 = IC + 1.0

# Define the energy function exactly as given
energy = 0.5 * mu * (IC - 3.0) - 0.5 * mu * sp.log(I3) + 0.5 * lam * ((J - (1.0 + 0.75 * mu / lam))**2)
# Multiply by volume
energy = energy * vol

# Compute the gradient of the energy with respect to the positions (X)
positions = [x0_x, x0_y, x0_z,
             x1_x, x1_y, x1_z,
             x2_x, x2_y, x2_z,
             x3_x, x3_y, x3_z]

grad_energy = [sp.diff(energy, pos) for pos in positions]

# For substituting numerical values, create a dictionary
subs_dict = {
    lam: 1000.0,    # Example value for lambda
    mu: 2000.0,     # Example value for mu
    vol: 32.0 / 3.0,    # Example value for volume
    x0_x: 2.0, x0_y: 0.0, x0_z: 0.0,
    x1_x: 2.0, x1_y: 2.0, x1_z: 0.0,
    x2_x: 0.0, x2_y: 2.0, x2_z: 0.0,
    x3_x: 0.0, x3_y: 2.0, x3_z: 2.0,
}

# Add IB components to subs_dict using symbols
subs_dict.update({
    IB_symbols['IB_11']: 0.25, IB_symbols['IB_12']: 0.25, IB_symbols['IB_13']: 0.0,
    IB_symbols['IB_21']: -0.25, IB_symbols['IB_22']: 0.0, IB_symbols['IB_23']: -0.25,
    IB_symbols['IB_31']: 0.0, IB_symbols['IB_32']: -0.0, IB_symbols['IB_33']: 0.25
})

# Substitute numerical values into the energy
energy_numeric = energy.subs(subs_dict)

# Substitute numerical values into the gradient
grad_energy_numeric = [expr.subs(subs_dict) for expr in grad_energy]

# Evaluate to floating point numbers
energy_value = energy_numeric.evalf()
grad_energy_values = [expr.evalf() for expr in grad_energy_numeric]

# Output the energy and gradient
print("Energy Value:")
print(energy_value)
print("\nGradient of Energy with respect to positions:")
for var, val in zip(positions, grad_energy_values):
    print(f"dEnergy/d{var} = {val}")

# Compute the Hessian matrix H
H = sp.hessian(energy, positions)

# Substitute numerical values into the Hessian
H_numeric = H.subs(subs_dict)

# Evaluate to floating point numbers
H_numeric_eval = H_numeric.evalf()

# Convert to numpy array
H_numpy = np.array(H_numeric_eval).astype(np.float64)

# Output the Hessian matrix
print("\nHessian Matrix:")
print(H_numpy)
print("Shape of Hessian:", H_numpy.shape)

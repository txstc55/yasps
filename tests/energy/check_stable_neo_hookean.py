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
H = sp.hessian(energy, positions)

vols = [10.66666667,10.66666667,-10.66666667,-10.66666667,-10.66666667,10.66666667]
IBS = np.array([[0.25, 0., 0.  ],
 [ 0. ,   0.25, -0.25],
 [ 0.,    0. ,   0.25],
 [ 0.25 , 0.25,  0.  ],
 [-0.25 , 0. ,  -0.25],
 [ 0. ,  -0.,    0.25],
 [ 0.25 ,-0. ,  -0.  ],
 [-0.  , -0.25 , 0.25],
 [-0.  ,  0.25 ,-0.  ],
 [ 0.25, -0. ,   0.25],
 [-0.25 ,-0.25 , 0.  ],
 [-0. ,   0.25 ,-0.  ],
 [-0.  ,  0.25 ,-0.25],
 [ 0.25 ,-0.  ,  0.25],
 [-0.25 ,-0.  , -0.  ],
 [ 0.  , -0.25 , 0.25],
 [ 0.25 , 0.25 ,-0.  ],
 [-0.25 , 0. ,   0.  ]]).flatten()

tet_positions = [0.,0.,0.,2.,0.,0.,
  0.,2.,0.,0.,2.,2.,
  2.,0.,0.,2.,1.87938524,0.68404029,
  0.,2.,0.,0.,2.,2.,
  0.,0.,0.,2.,0.,0.,
  0.,0.,2.,0.,2.,2.,
  2.,0.,0.,2., -0.68404029,1.87938524,
  0.,0.,2.,0.,2.,2.,
  2.,0.,0.,2.,1.87938524 , 0.68404029,
  2.,1.19534495 , 2.56342553 , 0.,2.,2.,
  2.,0.,0.,2.    ,     -0.68404029 , 1.87938524,
  2.,1.19534495 , 2.56342553 , 0.,2.,2.]

indices = [[0, 1, 3, 7], [1, 2, 3, 7], [0, 1, 4, 7], [1, 5, 4, 7], [1, 2, 6, 7], [1, 5, 6, 7]]
# indices = [[0, 1, 3, 7]]
sum_mat = np.zeros((24, 24), dtype = np.float64)
energy_sum = 0
gradient_sum = np.zeros(24, dtype = np.float64)
for i in range(len(indices)):
# for i in range(0, 2):
  indices_i = indices[i]
  # For substituting numerical values, create a dictionary
  subs_dict = {
      lam: 1000.0,    # Example value for lambda
      mu: 2000.0,     # Example value for mu
      vol: vols[i],    # Example value for volume
      x0_x: tet_positions[i * 12 + 0], x0_y: tet_positions[i * 12 + 1], x0_z: tet_positions[i * 12 + 2],
      x1_x: tet_positions[i * 12 + 3], x1_y: tet_positions[i * 12 + 4], x1_z: tet_positions[i * 12 + 5],
      x2_x: tet_positions[i * 12 + 6], x2_y: tet_positions[i * 12 + 7], x2_z: tet_positions[i * 12 + 8],
      x3_x: tet_positions[i * 12 + 9], x3_y: tet_positions[i * 12 + 10], x3_z: tet_positions[i * 12 + 11],
  }

  # Add IB components to subs_dict using symbols
  subs_dict.update({
      IB_symbols['IB_11']: IBS[i * 9 + 0], IB_symbols['IB_12']: IBS[i * 9 + 1], IB_symbols['IB_13']: IBS[i * 9 + 2],
      IB_symbols['IB_21']: IBS[i * 9 + 3], IB_symbols['IB_22']: IBS[i * 9 + 4], IB_symbols['IB_23']: IBS[i * 9 + 5],
      IB_symbols['IB_31']: IBS[i * 9 + 6], IB_symbols['IB_32']: IBS[i * 9 + 7], IB_symbols['IB_33']: IBS[i * 9 + 8]
  })

  # Substitute numerical values into the energy
  energy_numeric = energy.subs(subs_dict)
  # Substitute numerical values into the gradient
  grad_energy_numeric = [expr.subs(subs_dict) for expr in grad_energy]
  # Evaluate to floating point numbers
  energy_value = energy_numeric.evalf()
  grad_energy_values = np.array([expr.evalf() for expr in grad_energy_numeric], dtype=np.float64)
  # Output the energy and gradient
  # print("Energy Value:")
  # print(energy_value)
  energy_sum += energy_value
  # print("\nGradient of Energy with respect to positions:")
  # for var, val in zip(positions, grad_energy_values):
  #   print(f"dEnergy/d{var} = {val}")
  for j in range(4):
    ind = indices_i[j]
    gradient_sum[ind * 3 : (ind * 3 + 3)] += grad_energy_values[j * 3 : (j * 3 + 3)]


  # Substitute numerical values into the Hessian
  H_numeric = H.subs(subs_dict)
  # Evaluate to floating point numbers
  H_numeric_eval = H_numeric.evalf()
  # Convert to numpy array
  H_numpy = np.array(H_numeric_eval).astype(np.float64)
  # Output the Hessian matrix
  # print("\nHessian Matrix:")
  # print(H_numpy)

  # now we need to assemble the hessian
  for j in range(4):
    for k in range(4):
      block = H_numpy[j*3:(j+1)*3, k*3:(k+1)*3]
      pos_x = indices_i[j]
      pos_y = indices_i[k]
      sum_mat[pos_x * 3 : (pos_x * 3 + 3), pos_y * 3 : (pos_y * 3 + 3)] += block
print("\nEnergy Sum:")
print(energy_sum)
print("\nGradient Sum:")
print(gradient_sum)
print("\nSum of Hessian Matrices:")
print(sum_mat)

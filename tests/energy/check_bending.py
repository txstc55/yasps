import sympy as sp
import numpy as np

# Define symbolic variables for x positions (current configuration)
x_vars = sp.symbols('x0_x x0_y x0_z x1_x x1_y x1_z x2_x x2_y x2_z x3_x x3_y x3_z')
x0 = sp.Matrix([x_vars[0], x_vars[1], x_vars[2]])
x1 = sp.Matrix([x_vars[3], x_vars[4], x_vars[5]])
x2 = sp.Matrix([x_vars[6], x_vars[7], x_vars[8]])
x3 = sp.Matrix([x_vars[9], x_vars[10], x_vars[11]])

# Define symbolic variables for x_init positions (initial configuration)
x_init_vars = sp.symbols('x0i_x x0i_y x0i_z x1i_x x1i_y x1i_z x2i_x x2i_y x2i_z x3i_x x3i_y x3i_z')
x0i = sp.Matrix([x_init_vars[0], x_init_vars[1], x_init_vars[2]])
x1i = sp.Matrix([x_init_vars[3], x_init_vars[4], x_init_vars[5]])
x2i = sp.Matrix([x_init_vars[6], x_init_vars[7], x_init_vars[8]])
x3i = sp.Matrix([x_init_vars[9], x_init_vars[10], x_init_vars[11]])

# Define bending stiffness as a symbolic variable
bendStiff = sp.symbols('bendStiff')

# Compute normals and axis for current positions
n0 = (x0 - x2).cross(x1 - x2)
n1 = (x1 - x3).cross(x0 - x3)
axis = x1 - x0

# Compute normals and axis for initial positions
n0i = (x0i - x2i).cross(x1i - x2i)
n1i = (x1i - x3i).cross(x0i - x3i)
axisi = x1i - x0i

# Compute theta for current positions
axis_norm = sp.sqrt(axis.dot(axis))
n0_cross_n1_dot_axis = (n0.cross(n1)).dot(axis)
n0_dot_n1 = n0.dot(n1)
theta = 2 * sp.atan2(n0_cross_n1_dot_axis / axis_norm, n0_dot_n1)

# Compute theta for initial positions
axisi_norm = sp.sqrt(axisi.dot(axisi))
n0i_cross_n1i_dot_axisi = (n0i.cross(n1i)).dot(axisi)
n0i_dot_n1i = n0i.dot(n1i)
theta_init = 2 * sp.atan2(n0i_cross_n1i_dot_axisi / axisi_norm, n0i_dot_n1i)

# Compute the edge length in the initial configuration
edge_length_init = sp.sqrt((x1i - x0i).dot(x1i - x0i))

# Compute the bending energy
bend_energy = bendStiff * (theta - theta_init)**2 * edge_length_init

# Compute the gradient of the bending energy with respect to x variables
x_symbols = x_vars  # list of x variables
grad_bend_energy = [sp.diff(bend_energy, var) for var in x_symbols]

# Create lambdified functions for bend_energy and grad_bend_energy
all_symbols = x_vars + x_init_vars + (bendStiff,)
modules = {'ImmutableMatrix': np.array, 'MutableDenseMatrix': np.array, 'sqrt': np.sqrt, 'atan2': np.arctan2, 'pi': np.pi, 'sin': np.sin, 'cos': np.cos, 'Abs': np.abs}

bend_energy_func = sp.lambdify(all_symbols, bend_energy, modules)
grad_bend_energy_func = sp.lambdify(all_symbols, grad_bend_energy, modules)

# Now define a function to compute the energy and gradient given numerical inputs
def compute_bend_energy_and_gradient(x_values, x_init_values, bend_stiff_value):
    """
    Compute the bending energy and its gradient with respect to x variables.

    Parameters:
    - x_values: numpy array of shape (4, 3) for x0, x1, x2, x3
    - x_init_values: numpy array of shape (4, 3) for x0i, x1i, x2i, x3i
    - bend_stiff_value: scalar value of bending stiffness

    Returns:
    - energy: scalar bending energy
    - gradient: numpy array of shape (12,) representing the gradient
    """
    x_flat = x_values.flatten()
    x_init_flat = x_init_values.flatten()
    inputs = np.concatenate([x_flat, x_init_flat, [bend_stiff_value]])
    energy = bend_energy_func(*inputs)
    gradient = np.array(grad_bend_energy_func(*inputs))
    return energy, gradient

# Example usage:
if __name__ == '__main__':
    # Define numerical values for x and x_init
    x_values = np.array([-48.787, 54.564, 35.221000000000004, -48.769, 55.932, 35.168, -49.384, 55.898, 34.367000000000004, -48.144, 54.592, 36.01]).reshape(4, 3)
    x_init_values = np.array([-97.574, 109.128, 70.44200000000001, -97.538, 111.864, 70.336, -98.768, 111.796, 68.73400000000001, -96.288, 109.184, 72.02]).reshape(4, 3)
    bend_stiff_value = 1.0

    energy, gradient = compute_bend_energy_and_gradient(x_values, x_init_values, bend_stiff_value)
    print('Bending Energy:', energy)
    print('Gradient:', gradient)

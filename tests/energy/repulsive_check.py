import sympy as sp
import numpy as np

# Step 2: Define symbols
p00, p01, p02 = sp.symbols('p00 p01 p02')
p10, p11, p12 = sp.symbols('p10 p11 p12')
p20, p21, p22 = sp.symbols('p20 p21 p22')
p30, p31, p32 = sp.symbols('p30 p31 p32')

# Step 3: Points as symbolic vectors
p0 = sp.Matrix([p00, p01, p02])
p1 = sp.Matrix([p10, p11, p12])
p2 = sp.Matrix([p20, p21, p22])
p3 = sp.Matrix([p30, p31, p32])

# Step 4: Constants
alpha = sp.S(2.0)
beta = sp.S(4.5)
repulsive_weight = sp.S(1.0)

# Step 5: Unit tangent vectors
d01 = p1 - p0
d23 = p3 - p2

norm_d01 = sp.sqrt(d01.dot(d01))
norm_d23 = sp.sqrt(d23.dot(d23))

T01 = d01 / norm_d01
T23 = d23 / norm_d23

# Step 6: Compute the energy function
terms = [
    (T01, p0 - p2),
    (T01, p0 - p3),
    (T01, p1 - p2),
    (T01, p1 - p3),
    (T23, p2 - p0),
    (T23, p2 - p1),
    (T23, p3 - p0),
    (T23, p3 - p1)
]

r = sp.S(0)
for T, s in terms:
    numerator = (T.dot(s))**alpha
    denominator = (sp.sqrt(s.dot(s)))**beta
    r += numerator / denominator

E = r * repulsive_weight / 4.0

# Step 7: Compute the gradient
variables = [p00, p01, p02, p10, p11, p12, p20, p21, p22, p30, p31, p32]
gradient = [sp.diff(E, var) for var in variables]

# Step 8: Substitute numeric values
numeric_values = {
    p00:  0.99882565, p01:  0.0,          p02:  0.04844924,
    p10:  0.30869284, p11:  0.95005887,   p12: -0.04579168,
    p20: -0.8077216,  p21:  0.58684409,   p22:  0.05656707,
    p30: -0.80634554, p31: -0.58584433,   p32:  0.08119908
}

# Step 9: Evaluate the gradient numerically
gradient_values = [grad_expr.subs(numeric_values).evalf() for grad_expr in gradient]
gradient_matrix = np.array(gradient_values, dtype=np.float64).reshape(4, 3)

# Step 10: Display the gradient
print("Gradient Matrix:")
print(gradient_matrix)

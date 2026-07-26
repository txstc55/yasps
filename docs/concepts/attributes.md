# Attributes

An attribute is a per-instance symbolic tensor. If its owner has \(N\)
instances and the attribute has `rows=r`, `cols=c`, its numerical buffer
contains \(Nrc\) scalars.

## Data attributes

`addAttribute()` without `computed_attribute`, `through`, or `source` creates
mutable data:

```python
position = vertices.addAttribute("position", rows=3, cols=1)
position.updateValue(positions)
```

Data attributes can be optimization targets.

`updateValue()` accepts a NumPy array, backend `GPUArray`, or a value
convertible to a NumPy array. Values are flattened. The implementation does
not comprehensively validate that the supplied scalar count matches
`numInstances * rows * cols`; validate input shapes in application code.

Use `deepCopy=True` when the new value is another device array and later
mutation must not alias it:

```python
last_position.updateValue(position.value, deepCopy=True)
```

## Constants

`addConstant()` creates mutable numerical data that symbolic
differentiation treats as constant:

```python
mass = vertices.addConstant("mass", rows=1, cols=1)
mass.updateValue(np.full(vertices.numInstances, 0.1))
```

“Constant” means excluded from differentiation, not immutable. Time steps,
material parameters, previous positions, and velocities are commonly updated
between solves.

`expression.asConstant()` stops derivatives through an expression while
preserving its numerical dependence when evaluated.

## Computed attributes

Arithmetic builds an expression graph:

```python
displacement = position - rest_position
squared_distance = displacement.dot(displacement)
```

Name an expression by attaching it to its owner:

```python
kinetic = vertices.addAttribute(
  "kinetic",
  computed_attribute=0.5 * mass * velocity.dot(velocity),
)
```

Named attributes are reusable graph roots and are required for energies.
Intermediate expressions do not need names.

## Materializing a value

```python
computed = squared_distance.compute()
device_value = computed.value
host_value = device_value.get()
```

`compute()` evaluates all instances and stores a flattened device result.
YASPS does not materialize every intermediate graph node.

Evaluation uses generated CUDA/Eigen kernels.

## Per-instance indexing

Attribute indexing addresses the symbolic per-instance tensor, not the
instance dimension:

```python
x = position[0]
first_row = matrix.row(0)
second_column = matrix.col(1)
entry = matrix[1, 2]
```

For `position` shaped 3×1, `position[0]` is the x component for every
instance.

## Building a matrix

```python
from yasps.attribute import attribute

matrix = attribute.to_array(
  [
    a00, a01, a02,
    a10, a11, a12,
    a20, a21, a22,
  ],
  rows=3,
  cols=3,
)
```

Children are listed in row-major logical order.

## `resize()` versus `reshape()`

These similarly named methods have different contracts:

- `resize(rows, cols)` returns a **new symbolic expression** that reinterprets
  each instance.
- `reshape(rows, cols)` mutates the attribute's shape metadata and returns
  `None`.

Use `resize()` in computations:

```python
joined_matrix = joined_flat.resize(3, 3)
```

Treat `reshape()` as advanced metadata mutation.

## Shape and broadcasting rules

YASPS supports scalar multiplication and scalar-style broadcast addition and
subtraction where implemented. Non-scalar `*` means matrix multiplication,
not elementwise multiplication.

```python
scaled = 2.0 * vector
matrix_product = left * right
```

Use explicit scalar expressions or construct an array for elementwise
operations.

## Symbolic equality

Python `==` on attributes is used internally for structural/hash equality and
does not create a numerical comparison node. Use:

```python
condition = left.eq(right)
different = left.neq(right)
```

Public numerical ordering currently provides `>` and `>=`. See [operator
support](../reference/operator-support.md).

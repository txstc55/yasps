# Writing computations

YASPS expressions look like numerical Python but construct symbolic graphs.
The graph is evaluated for every instance of its inferred owner.

## A vector computation

```python
displacement = position - rest_position
squared_norm = displacement.dot(displacement)
length = squared_norm.sqrt()
```

`displacement` is a 3×1 expression and `length` is scalar per instance.

Attach reusable or externally requested results:

```python
vertices.addAttribute("length", computed_attribute=length)
```

## Matrix construction and operations

```python
from yasps.attribute import attribute

matrix = attribute.to_array(
  [
    c0[0], c1[0], c2[0],
    c0[1], c1[1], c2[1],
    c0[2], c1[2], c2[2],
  ],
  rows=3,
  cols=3,
)

determinant = matrix.determinant()
inverse = matrix.inverse()
gram = matrix.transpose() * matrix
trace = gram.trace()
```

`*` performs matrix multiplication when both operands are non-scalar. YASPS
does not interpret it as Hadamard multiplication.

## Vectors

```python
normal = edge0.cross(edge1)
area_twice = normal.norm()
alignment = direction.dot(normal)
```

`cross()` requires three-component vectors. `dot()` requires equally sized
vectors.

## Branching

Use a symbolic condition and `attribute.select()`:

```python
smoothed = attribute.select(
  squared_speed > threshold,
  squared_speed.sqrt(),
  low_speed_approximation,
)
```

Both branches must have the same per-instance shape and compatible lineage.
The condition is symbolic; Python `if` cannot branch per instance.

## Constants in an expression

Python floats become scalar symbolic constants:

```python
energy = 0.5 * stiffness * displacement.dot(displacement)
```

Use an owner constant when a value changes during the simulation:

```python
stiffness = mesh.addConstant("stiffness")
stiffness.updateValue(1000.0)
```

## Stopping a derivative

```python
frozen_basis = current_basis.asConstant()
energy = function_of(position, frozen_basis)
```

The basis is still evaluated numerically but differentiation does not follow
through it. This is useful for lagged quantities. It is not a copy: if the
underlying expression changes, a later computation sees the new value.

## Materialization and device arrays

```python
result = expression.compute().value
host = result.get()
```

Keep values on the device when possible:

```python
position.updateValue(position.value - delta, deepCopy=True)
```

Calling `.get()` synchronizes and exposes a NumPy value. Dynamic collision
topology is an intentional host boundary in the current examples; routine
attribute arithmetic need not be.

## Lineage errors

If an expression combines unrelated primitive attributes directly, YASPS
cannot infer an owner:

```python
# Invalid when left and right are unrelated primitive types:
bad = left_position - right_position
```

Make the topological relation explicit:

- gather through `JOIN`;
- combine corresponding child attributes through `UNION`; or
- lift shared parameters to a common mesh/scene constant.

## Naming computed results

An expression can remain unnamed until it becomes:

- a reusable attribute;
- a JOIN source;
- a value requested by application code; or
- an energy.

Energy roots must be named:

```python
energy = tets.addAttribute("elastic_energy", computed_attribute=energy_expr)
simulation.addEnergy(energy)
```

Use valid identifier names. Generated CUDA symbols and internal full names are
derived from them.

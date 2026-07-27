---
title: Attributes and expressions
description: The public symbolic expression syntax, shapes, values, and materialization behavior in YASPS.
permalink: /attributes/
---

<p class="eyebrow">Core syntax</p>

# Attributes and expressions

The `attribute` class is YASPS's user-facing symbolic value. Every attribute has a per-instance shape, a correspondence in the scene hierarchy, an operator, and zero or more children.

## Creating leaves

Create differentiable data with `addAttribute`:

```python
x = vertices.addAttribute("position", rows=3, cols=1)
x.updateValue(initial_positions)
```

Create mutable nondifferentiable data with `addConstant`:

```python
mass = vertices.addConstant("mass", rows=1, cols=1)
mass.updateValue(vertex_masses)
```

Bind a symbolic expression to a reusable name:

```python
displacement = x - rest_position
vertices.addAttribute("displacement", computed_attribute=displacement)
```

Named attributes, JOIN nodes, UNION nodes, and requested outputs act as modular boundaries for generated code.

## Constructing matrices

Use `attribute.to_array` with row-major elements:

```python
from yasps import attribute

gravity = attribute.to_array(
    [0.0, -9.8, 0.0],
    rows=3,
    cols=1,
)

F = attribute.to_array(
    [
        e0[0], e0[1], e0[2],
        e1[0], e1[1], e1[2],
        e2[0], e2[1], e2[2],
    ],
    rows=3,
    cols=3,
)
```

Convenience constructors:

```python
zero = attribute.zeros(3, 3)
identity = attribute.identity(3)
literal = attribute(float_value=2.0)
```

## Indexing and shape operations

```python
scalar = matrix_attribute[1, 2]
flat_scalar = matrix_attribute[5]
row = matrix_attribute.row(1)
column = matrix_attribute.col(2)
transposed = matrix_attribute.transpose()
reshaped_expression = joined.resize(3, 3)
```

`resize(rows, cols)` returns a symbolic reshape and requires the same element count. `reshape(rows, cols)` mutates the Python object's shape metadata in place; prefer `resize` in ordinary symbolic expressions.

## Arithmetic

| Syntax | Meaning |
| --- | --- |
| `a + b`, `a - b` | Same-shape addition/subtraction; a scalar broadcasts |
| `a * b` | Matrix product, or scalar multiplication if either operand is scalar |
| `a / scalar` | Division by a scalar attribute or Python float |
| `a.pow(p)` | Scalar power |
| `-a` | Negation |
| `a.add_explicit(b)` | Element-expanded addition |
| `a.sub_explicit(b)` | Element-expanded subtraction |
| `a.mul_explicit(b)` | Element-expanded matrix/scalar multiplication |
| `a.div_explicit(b)` | Element-expanded scalar division |

The explicit forms construct individual scalar nodes. They are mainly useful when controlling symbolic simplification or working around a generated matrix expression.

## Scalar functions

The following methods require a scalar unless noted:

```python
y = x.sqrt()
y = x.log()
y = x.sin()
y = x.cos()
y = x.atan2(other)
y = x.abs()
```

`pow`, `sqrt`, `log`, `sin`, and `cos` perform constant folding when all inputs are numeric literals.

## Vector and matrix functions

```python
dot = u.dot(v)                 # vectors of equal flattened size
cross = u.cross(v)             # exactly three elements
length = u.norm()              # row or column vector
trace = A.trace()              # square matrix
determinant = A.determinant()  # square matrix
inverse = A.inverse()          # square matrix
```

`dot` currently expands into scalar operations. `cross` similarly builds its three scalar components explicitly.

## Conditions and selection

Conditions are symbolic scalar attributes:

```python
condition = distance >= threshold
result = attribute.select(condition, active_energy, zero_energy)
```

The true and false values must have the same shape and compatible lineage. Explicit equality methods are available as `a.eq(b)` and `a.neq(b)`.

## PSD projection

`A.spd(method)` creates a symbolic projection of a square matrix:

- `0`: leave eigenvalues unchanged;
- `1`: replace eigenvalues with their absolute values;
- `2`: clamp negative eigenvalues to zero.

Energy registration normally controls projection for you. Direct `spd` use is an advanced building block.

## Treating a subexpression as constant

`expression.asConstant()` preserves its numerical computation but makes its derivative zero:

```python
frozen_normal = normal.asConstant()
```

This is different from materializing the expression on the CPU; it remains part of generated device code.

## Materialization

Ordinary expression construction does not execute:

```python
energy_expression = 0.5 * displacement.dot(displacement)
```

Call `compute()` only when you need the values:

```python
energy_gpu = energy_expression.compute().value
energy_cpu = energy_gpu.get()
```

The first call generates and compiles a global kernel. The output buffer is cached and reused where possible. Intermediate symbolic attributes are evaluated inside the generated kernel rather than stored as full arrays.

## Lineage errors

If two attributes belong to unrelated primitives, arithmetic raises an error. Do not copy their numerical arrays together to bypass this: that would sever differentiation. Define the relation with [JOIN]({{ '/join/' | relative_url }}) or combine heterogeneous attributes with [UNION]({{ '/union/' | relative_url }}).

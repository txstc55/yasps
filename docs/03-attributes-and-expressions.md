---
title: Attributes and expressions
chapter: "03"
description: Construct data, constants, computed expressions, JOINs, UNIONs, matrices, and materialized outputs in YASPS.
permalink: /attributes/
next_url: /join/
next_label: Connectivity and JOIN
---

<p class="eyebrow">Core syntax · chapter 03</p>

# Attributes and expressions

`attribute` is the symbolic value type used everywhere in YASPS. It describes one `rows × cols` value per instance of its correspondence. The same class represents stored data, constants, computed expressions, JOIN results, UNION results, and requested outputs.

## Choose the right construction

| Need | Construction | Differentiable? | Storage |
| --- | --- | --- | --- |
| Degrees of freedom or mutable input | `owner.addAttribute(name, rows, cols)` | Yes | Uploaded GPU data |
| Mutable history/material/control value | `owner.addConstant(name, rows, cols)` | No | Uploaded GPU data |
| Reusable symbolic expression | `owner.addAttribute(name, computed_attribute=expr)` | Follows children | Fused unless materialized |
| Gather through topology | `primitive.addAttribute(name, through=..., source=...)` | Follows source path | Symbolic JOIN |
| Stack heterogeneous populations | `union.addAttribute(name)` | Follows child paths | Symbolic UNION |
| Numeric literal or matrix | `attribute(float_value=...)`, `to_array(...)` | No independent storage | Inlined into expression |

## Data attributes

Create data through the scene, mesh, primitive, or union that owns its instance index:

```python
position = vertices.addAttribute(
  "position",
  rows=3,
  cols=1,
)
```

### Parameters

```python
owner.addAttribute(
  name,
  computed_attribute=None,
  rows=1,
  cols=1,
)
```

| Parameter | Meaning |
| --- | --- |
| `name` | Unique name in `owner.attributes`; used in generated identifiers |
| `computed_attribute` | Leave `None` for a stored DATA leaf |
| `rows` | Rows in each instance value |
| `cols` | Columns in each instance value |

The expected flattened value count is `owner.numInstances * rows * cols`.

### Uploading values

```python
position.updateValue(initial_positions)
```

`updateValue(value, deepCopy=False)` accepts:

| Input | Behavior |
| --- | --- |
| NumPy array or array-like value | Flattens, converts to float64, allocates/uploads a PyCUDA array |
| PyCUDA `GPUArray`, `deepCopy=False` | Makes the attribute reference that device array |
| PyCUDA `GPUArray`, `deepCopy=True` | Reuses or allocates owned storage and copies values |

Use `deepCopy=True` when the input is a temporary expression or aliases a reusable solver buffer:

```python
position.updateValue(
  position.value - alpha * direction,
  deepCopy=True,
)
```

The implementation does not currently reject an incorrectly sized upload at this boundary. Supply exactly the flattened size expected by the correspondence and per-instance shape.

## Constant attributes

```python
mass = vertices.addConstant(
  "mass",
  rows=1,
  cols=1,
)
mass.updateValue(vertex_masses)
```

### Parameters

```python
owner.addConstant(name, rows=1, cols=1)
```

| Parameter | Meaning |
| --- | --- |
| `name` | Unique attribute name on the owner |
| `rows`, `cols` | Per-instance shape |

CONSTANT means “derivative is zero,” not “immutable.” Previous positions, velocities, masses, rest geometry, material parameters, timestep, barrier distance, and runtime controls are all commonly constants even though the application updates them.

## Computed attributes

Arithmetic creates unnamed expression nodes:

```python
displacement = position - rest_position
quadratic_expression = 0.5 * stiffness * displacement.dot(displacement)
```

Bind a reusable node to its correspondence:

```python
quadratic = vertices.addAttribute(
  "quadratic",
  computed_attribute=quadratic_expression,
)
```

### Why name expressions

Named nodes provide:

- stable generated identifiers;
- reusable symbolic boundaries;
- a place to request `compute()`;
- the required named scalar passed to `addEnergy`;
- natural JIT kernel boundaries when an expression is explicitly materialized.

On a primitive, if a computed constant matrix has no correspondence, `addAttribute` assigns the primitive. If an expression belongs to a different correspondence, the primitive path copies the symbolic wrapper and retargets its correspondence; ordinary modeling should still respect lineage rather than relying on that fallback.

## Primitive `addAttribute` modes

The full primitive signature is:

```python
primitive.addAttribute(
  name,
  computed_attribute=None,
  rows=1,
  cols=1,
  through=None,
  source=None,
  operation=None,
)
```

| Parameter | Used when | Meaning |
| --- | --- | --- |
| `name` | Always | New unique name on the primitive |
| `computed_attribute` | Computed mode | Expression to bind; this mode takes priority |
| `rows`, `cols` | Data mode | Per-instance shape for a DATA leaf |
| `through` | JOIN mode | Outgoing connectivity whose `fromPrimitive` is this primitive |
| `source` | Fixed JOIN | Explicit attribute on the connectivity target |
| `operation` | Variable JOIN | `"SUM"` or `"AVERAGE"` when `through.dimension == 0` |

The selection order is computed expression, then JOIN, then new DATA leaf. Do not provide arguments from multiple modes.

## JOIN attributes

Fixed-arity JOIN gathers a target attribute:

```python
tet_position = tets.addAttribute(
  "position",
  through=tet2vertex,
  source=vertices["position"],
)
```

If the connectivity dimension is `k` and source shape is `r × c`, the result shape is `k × (r*c)`.

When `source` is omitted, YASPS looks up an attribute with the new `name` on the target primitive:

```python
tet_position = tets.addAttribute(
  "position",
  through=tet2vertex,
)
```

For variable-arity connectivity (`dimension=0`), provide `operation="SUM"` or `"AVERAGE"` and use implicit same-name lookup. See [Connectivity and JOIN]({{ '/join/' | relative_url }}?v={{ site.time | date: '%s' }}).

## UNION attributes

Create a union on a mesh, then query a same-named child attribute:

```python
all_vertices = collision.addPrimitiveUnion(
  "vertices",
  [soft_vertices, affine_vertices],
)
all_position = all_vertices.addAttribute("position")
```

Every child must expose `position` with identical shape. The result retains the route to each child rather than becoming an eager concatenation.

The full union signature is:

```python
union.addAttribute(
  name,
  computed_attribute=None,
  rows=0,
  cols=0,
)
```

| Form | Meaning |
| --- | --- |
| `addAttribute(name)` | UNION same-named child attributes |
| `addAttribute(name, computed_attribute=expr)` | Bind a union-owned expression |
| `addAttribute(name, rows=r, cols=c)` | Create union-owned DATA rather than stack children |

See [Primitive unions]({{ '/union/' | relative_url }}?v={{ site.time | date: '%s' }}) for ordering, nested unions, and sparse symbolic matrices.

## Literal and matrix construction

```python
from yasps import attribute

scalar = attribute(float_value=2.0)
zero = attribute.zeros(3, 3)
identity = attribute.identity(3)

gravity = attribute.to_array(
  [0.0, -9.8, 0.0],
  rows=3,
  cols=1,
)
```

`attribute.to_array(children, rows, cols)` uses row-major order. `rows * cols` must equal `len(children)`. Python floats and ints become literal nodes; other elements must be attributes with compatible lineage.

## Shape, indexing, and views

```python
scalar = A[1, 2]
flat_scalar = A[5]
row = A.row(1)
column = A.col(2)
transposed = A.transpose()
reshaped = joined.resize(3, 3)
```

| Member | Constraint | Result |
| --- | --- | --- |
| `a[i]` | `i < rows*cols` | Flattened scalar access |
| `a[row, col]` | Indices in bounds | Matrix scalar access |
| `row(index)` | `index < rows` | `1 × cols` expression |
| `col(index)` | `index < cols` | `rows × 1` expression |
| `transpose()` | Any shape | Shape becomes `cols × rows` |
| `resize(rows, cols)` | Same element count | New symbolic reshape |
| `reshape(rows, cols)` | Same element count | Mutates this Python object's metadata |

Prefer `resize` inside expressions. `reshape` changes the object in place and returns `None`.

## Arithmetic

| Syntax | Shape rule | Meaning |
| --- | --- | --- |
| `a + b`, `a - b` | Equal shapes or one scalar | Add/subtract; scalar broadcasts |
| `a * b` | `a.cols == b.rows`, or one scalar | Matrix product or scalar multiplication |
| `a / b` | `b` scalar | Division |
| `a.pow(p)` | Both scalar | Power |
| `-a` | Any shape | Negation |

Element-expanded variants are `add_explicit`, `sub_explicit`, `mul_explicit`, and `div_explicit`. They construct scalar components directly and are useful for derivative simplification or generated-code control.

The ordinary operators simplify zero, identity, and literal cases before creating a symbolic node.

## Scalar functions

```python
root = x.sqrt()
log_x = x.log()
sin_x = x.sin()
cos_x = x.cos()
angle = y.atan2(x)
magnitude = x.abs()
power = x.pow(2.0)
```

`sqrt`, `log`, `sin`, `cos`, `atan2`, and `pow` require scalar inputs. Literal inputs are folded in Python. `log(0)` and division by a symbolic zero are rejected.

## Vector and matrix functions

```python
dot = u.dot(v)
cross = u.cross(v)
length = u.norm()
trace = A.trace()
determinant = A.determinant()
inverse = A.inverse()
```

| Member | Constraint |
| --- | --- |
| `dot(other)` | Both row/column vectors with equal flattened size |
| `cross(other)` | Exactly three elements each |
| `norm()` | Row or column vector |
| `trace()` | Square matrix |
| `determinant()` | Square matrix |
| `inverse()` | Square matrix; scalar inverse becomes `1/a` |

`dot` and `cross` currently expand to scalar operations.

## Conditions and selection

```python
condition = distance >= threshold
selected = attribute.select(
  condition,
  active_energy,
  zero_energy,
)
```

`eq`, `neq`, `>`, and `>=` produce scalar condition attributes. The true and false branches passed to `select` must share shape and compatible lineage.

## Projection and derivative control

### `spd`

```python
projected = A.spd(spd_method=2)
```

| Value | Eigenvalue policy |
| ---: | --- |
| `0` | No numerical change |
| `1` | Absolute value |
| `2` | Clamp negative values to zero |

`A` must be square. An integer method is stored in a generated scene constant; an attribute method may be used when it has compatible heritage.

### `asConstant`

```python
frozen_normal = normal.asConstant()
```

This keeps numerical evaluation in the generated expression but prevents differentiation through the wrapped value.

## Materialization and execution

Expression construction does not launch work:

```python
energy_expression = 0.5 * displacement.dot(displacement)
```

Request a full per-instance output with:

```python
energy_gpu = energy_expression.compute().value
energy_cpu = energy_gpu.get()
```

On first compute, YASPS generates a device expression, creates a global kernel, allocates an output buffer, and JIT-compiles. Later calls reuse the generated kernel and enlarge storage only when required. Computation still occurs on the GPU; `.get()` is the host synchronization and transfer.

## Public introspection

Useful read-only properties include:

| Property | Meaning |
| --- | --- |
| `name`, `fullName`, `fullNameWithHash` | User and generated identifiers |
| `rows`, `cols`, `size` | Per-instance shape |
| `correspondance` | Owning hierarchy object |
| `through` | Connectivity for JOIN-like nodes |
| `children`, `operator` | Symbolic graph structure |
| `value` | Current PyCUDA buffer |
| `isDynamic` | Whether correspondence is a dynamic primitive |
| `isZero`, `isIdentity`, `isFloatMat` | Symbolic simplification classification |
| `hash` | Structural identity used for caches |

`deviceKernel`, `globalKernel`, `generate_code`, `disable_array_access`, `setName`, and `setAsIntermediate` are exposed for generator work. Normal model construction should use hierarchy methods instead.

## Lineage errors

If operands belong to unrelated primitives, arithmetic raises an error. Define the relationship with JOIN or UNION. Copying numerical buffers to bypass lineage would remove the differentiation and sparse-index path that YASPS needs.

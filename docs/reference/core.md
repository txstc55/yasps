---
title: Core API reference
description: Signatures and properties for scenes, meshes, primitives, connectivity, unions, and attributes.
permalink: /reference/core/
---

<p class="eyebrow">Reference</p>

# Core API reference

All classes on this page are importable from the package root:

```python
from yasps import scene, mesh, primitive, primitiveUnion, connectivity, attribute
```

Construct hierarchy objects through their parent (`scene.addMesh`, `mesh.addPrimitive`, and so on) so names and lineage are registered correctly.

## `scene`

```python
scene(name)
```

| Member | Description |
| --- | --- |
| `addMesh(name) -> mesh` | Create and register a mesh |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1) -> attribute` | Create data or bind an expression at scene scope |
| `addConstant(name, rows=1, cols=1) -> attribute` | Create mutable nondifferentiable data |
| `addEnergy(e, targets=[], projection_method=1, save_intermediate=False, gradient_only=False, dynamic_instances=False, separate_hessian_jacobian=False)` | Register a named scalar energy |
| `addMinimizeTarget(targets)` | Set the ordered global target list and generate derivatives |
| `minimizeEnergy(tolerance=1e-3, maxIterations=20000)` | Assemble and solve; return per-target GPU views |
| `computeTotalEnergy() -> float` | Sum all active energy instances |
| `ignoreEnergies(energies)` | Replace the current ignored-energy list |
| `scene["name"]` | Retrieve an attribute |
| `scene["a", "b"]` | Construct a row vector from several attributes |

Read-only properties: `name`, `scene`, `type`, `attributes`, `minimizer`, `numInstances` (always one), `numMeshes`, `fullName`, `gradient`, `gradientSegments`, and `diagonal`.

`energies` exists as a dictionary property, but current registration is owned by `scene.minimizer`; inspect `minimizer.energies` and `minimizer.energiesDynamic` for the actual requests.

Scene names are globally unique for the lifetime of the Python process.

## `mesh`

Normally created with `scene.addMesh`.

```python
mesh(name, parent_scene)
```

| Member | Description |
| --- | --- |
| `addPrimitive(name, numInstances, isDynamic=False) -> primitive` | Create a fixed or dynamic instance population |
| `addPrimitiveUnion(name, primitives) -> primitiveUnion` | Stack primitives while preserving lineage |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1) -> attribute` | Create data or bind an expression at mesh scope |
| `addConstant(name, rows=1, cols=1) -> attribute` | Create mutable nondifferentiable data |
| `mesh["name"]` / `mesh["a", "b"]` | Retrieve or pack attributes |

Read-only properties: `name`, `scene`, `mesh`, `type`, `primitives`, `attributes`, `numInstances` (always one), `numPrimitives`, and `fullName`.

## `primitive`

Normally created with `mesh.addPrimitive`.

```python
primitive(name, parent_mesh, numInstances=0, isDynamic=False)
```

| Member | Description |
| --- | --- |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1, through=None, source=None, operation=None) -> attribute` | Create data, bind an expression, or construct a JOIN |
| `addConstant(name, rows=1, cols=1) -> attribute` | Create mutable nondifferentiable per-instance data |
| `addConnectivity(name, to, data, dimension) -> connectivity` | Add a same-mesh outgoing connectivity |
| `updateNumInstances(count)` | Change a dynamic primitive's instance count |
| `updateConnectivity(name, data, dimension)` | Update a named connectivity after checking arity |
| `primitive["name"]` | Retrieve an attribute |

Read-only properties: `isDynamic`, `name`, `mesh`, `scene`, `primitive`, `connectivities`, `type`, `numInstances`, `attributes`, `attributesNames`, `numConnectivities`, and `fullName`.

`operation` is required for `dimension=0` JOINs and accepts `"SUM"` or `"AVERAGE"`. That path currently uses the implicit same-name target lookup, so omit `source`.

## `primitiveUnion`

Normally created with `mesh.addPrimitiveUnion`.

```python
primitiveUnion(name, parent_mesh, primitives)
```

| Member | Description |
| --- | --- |
| `addAttribute(name, computed_attribute=None, rows=0, cols=0) -> attribute` | Union same-named child attributes, or bind a union-owned expression |
| `addConstant(name, rows=1, cols=1) -> attribute` | Create union-owned nondifferentiable data |
| `union["name"]` | Retrieve a union attribute |

Read-only properties: `name`, `mesh`, `scene`, `primitiveUnion`, `type`, `numInstances`, `attributes`, `attributesNames`, `fullName`, `code_generation_counts_name`, and `children_primitive_counts_gpu`.

When `computed_attribute` is omitted and `rows`/`cols` remain zero, every child must already expose the named attribute with an identical shape.

## `connectivity`

Normally created with `primitive.addConnectivity`.

```python
connectivity(name, from_primitive, to_primitive, value, dimension)
```

| Member | Description |
| --- | --- |
| `updateConnectivity(value)` | Replace/reuse the device index buffer |

Read-only properties: `name`, `fullName`, `fromPrimitive`, `toPrimitive`, `value`, `dimension`, `mesh`, `scene`, `type`, `compressedRows`, `code_generation_index_name`, and `code_generation_csr_name`.

`value` is a flattened uint32 GPU buffer. `compressedRows` contains CSR offsets for variable-arity connectivity.

## `attribute`

Users normally create leaves and named nodes through a hierarchy object. The constructor remains public:

```python
attribute(
  name="",
  rows=1,
  cols=1,
  correspondance=None,
  through=None,
  float_value=None,
  children=[],
  operator=DATA,
  index_value=None,
  is_constant=False,
  generate_code=True,
)
```

Use it directly mainly for numeric literals:

```python
two = attribute(float_value=2.0)
```

### Constructors and values

| Member | Description |
| --- | --- |
| `attribute.zeros(rows, cols)` | Symbolic constant zero matrix |
| `attribute.identity(rows)` | Symbolic identity matrix |
| `attribute.to_array(children, rows, cols)` | Row-major symbolic matrix construction |
| `updateValue(value, deepCopy=False)` | Set a leaf from NumPy or PyCUDA storage |
| `compute() -> attribute` | Generate/launch a global kernel and attach its output |
| `asConstant() -> attribute` | Preserve numeric evaluation while zeroing derivatives |

### Shape and indexing

| Member | Description |
| --- | --- |
| `a[i]`, `a[row, col]` | Extract a scalar |
| `row(index)`, `col(index)` | Extract a row or column |
| `transpose()` | Symbolic transpose |
| `resize(rows, cols)` | Return a same-element-count symbolic reshape |
| `reshape(rows, cols)` | Mutate this Python object's shape metadata |

### Operators

| Member | Description |
| --- | --- |
| `+`, `-`, unary `-` | Addition, subtraction, negation |
| `*` | Matrix or scalar multiplication |
| `/` | Scalar division |
| `pow`, `sqrt`, `log`, `sin`, `cos`, `atan2`, `abs` | Scalar functions |
| `trace`, `determinant`, `inverse` | Square-matrix functions |
| `norm`, `dot`, `dot_explicit`, `cross` | Vector operations |
| `eq`, `neq`, `>`, `>=` | Scalar conditions |
| `attribute.select(condition, true_attribute, false_attribute)` | Symbolic conditional |
| `spd(method=1)` | Square-matrix eigenvalue projection |
| `add_explicit`, `sub_explicit`, `mul_explicit`, `div_explicit` | Element-expanded variants |

### Introspection

Read-only properties include `name`, `fullName`, `fullNameWithHash`, `rows`, `cols`, `size`, `correspondance`, `through`, `children`, `float_value`, `index_value`, `operator`, `value`, `isDynamic`, `isZero`, `isIdentity`, `isFloatMat`, and `hash`.

Generator-facing mutable properties include `deviceKernel`, `globalKernel`, `generate_code`, and `disable_array_access`. `setName` and `setAsIntermediate` mutate code-generation metadata; prefer hierarchy methods unless extending the generator.

See [Attributes and expressions]({{ '/attributes/' | relative_url }}?v={{ site.time | date: '%s' }}) for semantics and examples.

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

| Constructor parameter | Type | Effect |
| --- | --- | --- |
| `name` | nonempty `str` | Global scene identifier and generated-name prefix. It must be unique for the lifetime of the Python process. |

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
Construction also creates the scene's `.yasps_tmp` and `.yasps_constant`
working directories and its owned `minimizer`.

### Scene construction methods

| Call | Parameters |
| --- | --- |
| `addMesh(name)` | `name`: unused valid Python identifier under this scene. |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1)` | `name`: scene attribute key; `computed_attribute`: expression to bind, or `None` for data; `rows`, `cols`: positive data shape when creating a leaf. |
| `addConstant(name, rows=1, cols=1)` | `name`: scene attribute key; `rows`, `cols`: positive constant-buffer shape. |
| `addMinimizeTarget(targets)` | `targets`: ordered unique, static `DATA` attributes forming the global vector layout. |

See [Energies and minimization]({{ '/optimization/' | relative_url }}?v={{ site.time | date: '%s' }})
for every energy and solver parameter.

## `mesh`

Normally created with `scene.addMesh`.

```python
mesh(name, parent_scene)
```

| Constructor parameter | Type | Effect |
| --- | --- | --- |
| `name` | `str` | Mesh identifier within `parent_scene`. Prefer `scene.addMesh` so it is validated and registered. |
| `parent_scene` | `scene` | Owning scene and code-generation namespace. |

| Member | Description |
| --- | --- |
| `addPrimitive(name, numInstances, isDynamic=False) -> primitive` | Create a fixed or dynamic instance population |
| `addPrimitiveUnion(name, primitives) -> primitiveUnion` | Stack primitives while preserving lineage |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1) -> attribute` | Create data or bind an expression at mesh scope |
| `addConstant(name, rows=1, cols=1) -> attribute` | Create mutable nondifferentiable data |
| `mesh["name"]` / `mesh["a", "b"]` | Retrieve or pack attributes |

Read-only properties: `name`, `scene`, `mesh`, `type`, `primitives`, `attributes`, `numInstances` (always one), `numPrimitives`, and `fullName`.

### Mesh construction methods

| Call | Parameters |
| --- | --- |
| `addPrimitive(name, numInstances, isDynamic=False)` | `name`: unused primitive identifier; `numInstances`: nonnegative initial count; `isDynamic`: whether `updateNumInstances` is permitted. |
| `addPrimitiveUnion(name, primitives)` | `name`: unused union identifier; `primitives`: ordered list of primitives or nested unions. |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1)` | `name`: mesh attribute key; `computed_attribute`: bound expression or `None`; `rows`, `cols`: data shape when no expression is supplied. |
| `addConstant(name, rows=1, cols=1)` | `name`: mesh attribute key; `rows`, `cols`: constant-buffer shape. |

## `primitive`

Normally created with `mesh.addPrimitive`.

```python
primitive(name, parent_mesh, numInstances=0, isDynamic=False)
```

| Constructor parameter | Type and default | Effect |
| --- | --- | --- |
| `name` | `str` | Primitive identifier within `parent_mesh`. |
| `parent_mesh` | `mesh` | Owning mesh. |
| `numInstances` | `int = 0` | Nonnegative number of indexed instances. |
| `isDynamic` | `bool = False` | Allows the instance count to change after construction. |

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

### Primitive construction methods

| Call | Parameters |
| --- | --- |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1, through=None, source=None, operation=None)` | `name`: new key; `computed_attribute`: expression-binding mode; `rows`, `cols`: data-leaf shape; `through`: JOIN connectivity; `source`: fixed-arity JOIN source attribute; `operation`: `"SUM"` or `"AVERAGE"` for variable arity. |
| `addConstant(name, rows=1, cols=1)` | `name`: new key; `rows`, `cols`: per-instance constant shape. |
| `addConnectivity(name, to, data, dimension)` | `name`: outgoing edge key; `to`: same-mesh target primitive; `data`: index rows; `dimension`: fixed arity, or zero for CSR input. |
| `updateNumInstances(count)` | `count`: new nonnegative count; valid only when `isDynamic=True`. |
| `updateConnectivity(name, data, dimension)` | `name`: existing connectivity key; `data`: replacement indices; `dimension`: must equal the stored arity. This is the primitive convenience forward to the connectivity object. |

## `primitiveUnion`

Normally created with `mesh.addPrimitiveUnion`.

```python
primitiveUnion(name, parent_mesh, primitives)
```

| Constructor parameter | Type | Effect |
| --- | --- | --- |
| `name` | `str` | Union identifier within `parent_mesh`. |
| `parent_mesh` | `mesh` | Owner of the logical stacked population. Children may originate from other meshes in the same scene. |
| `primitives` | ordered list | Child `primitive` or nested `primitiveUnion` objects. Their order determines contiguous instance ranges. |

| Member | Description |
| --- | --- |
| `addAttribute(name, computed_attribute=None, rows=0, cols=0) -> attribute` | Union same-named child attributes, or bind a union-owned expression |
| `addConstant(name, rows=1, cols=1) -> attribute` | Create union-owned nondifferentiable data |
| `union["name"]` | Retrieve a union attribute |

Read-only properties: `name`, `mesh`, `scene`, `primitiveUnion`, `type`, `numInstances`, `attributes`, `attributesNames`, `fullName`, `code_generation_counts_name`, and `children_primitive_counts_gpu`.

When `computed_attribute` is omitted and `rows`/`cols` remain zero, every child must already expose the named attribute with an identical shape.

### Union construction methods

| Call | Parameters |
| --- | --- |
| `addAttribute(name, computed_attribute=None, rows=0, cols=0)` | `name`: child lookup and new key in union mode; `computed_attribute`: bind a union-owned expression instead; positive `rows`, `cols`: create union-owned data rather than querying children. |
| `addConstant(name, rows=1, cols=1)` | `name`: new union-owned key; `rows`, `cols`: constant-buffer shape. |

## `connectivity`

Normally created with `primitive.addConnectivity`.

```python
connectivity(name, from_primitive, to_primitive, value, dimension)
```

| Constructor parameter | Type | Effect |
| --- | --- | --- |
| `name` | `str` | Connectivity identifier on the source primitive. |
| `from_primitive` | `primitive` | Source instance space. |
| `to_primitive` | `primitive` | Target instance space; must share the source mesh. |
| `value` | NumPy array or nested lists | Target indices, converted to a flattened uint32 GPU buffer. |
| `dimension` | nonnegative `int` | Fixed targets per source when positive; zero selects variable-arity CSR conversion. |

| Member | Description |
| --- | --- |
| `updateConnectivity(value)` | Replace/reuse the device index buffer |

Read-only properties: `name`, `fullName`, `fromPrimitive`, `toPrimitive`, `value`, `dimension`, `mesh`, `scene`, `type`, `compressedRows`, `code_generation_index_name`, and `code_generation_csr_name`.

`value` is a flattened uint32 GPU buffer. `compressedRows` contains CSR offsets for variable-arity connectivity.

`updateConnectivity(value)` replaces or reuses the flattened index buffer.
For fixed arity, the caller must keep its row count consistent with the current
source instance count. The method does not change `dimension`; the primitive
convenience method checks the supplied arity first. Variable-arity row offsets
are constructed initially and are not rebuilt by this update path.

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

| Constructor parameter | Type and default | Effect |
| --- | --- | --- |
| `name` | `str = ""` | Optional generated identifier. Unnamed nodes are expression temporaries. |
| `rows`, `cols` | `int = 1` | Symbolic matrix shape. |
| `correspondance` | hierarchy object or `None` | Scene, mesh, primitive, or union whose instance index evaluates this attribute. |
| `through` | `connectivity | None` | JOIN edge associated with this node. |
| `float_value` | `float | None` | Creates a numeric literal when supplied. |
| `children` | `list[attribute] = []` | Operand nodes for generator-created expressions. |
| `operator` | operator, default `DATA` | Node operation. Applications normally let arithmetic or hierarchy methods select it. |
| `index_value` | `int | None` | Index metadata for generated access operations. |
| `is_constant` | `bool = False` | Excludes this data node from differentiation. |
| `generate_code` | `bool = True` | Generator-facing switch for emitting this node. |

Use it directly mainly for numeric literals:

```python
two = attribute(float_value=2.0)
```

### Constructors and values

| Member | Description |
| --- | --- |
| `attribute.zeros(rows, cols)` | Symbolic constant zero matrix with positive `rows` and `cols` |
| `attribute.identity(rows)` | Square symbolic identity matrix with the requested dimension |
| `attribute.to_array(children, rows, cols)` | Row-major symbolic matrix; requires exactly `rows * cols` scalar children |
| `updateValue(value, deepCopy=False)` | Set a leaf from NumPy/list storage, or alias/copy a PyCUDA buffer according to `deepCopy` |
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

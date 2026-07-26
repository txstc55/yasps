# Meshes and primitives API

## Mesh

Create through a scene:

```python
mesh = simulation.addMesh("soft_body")
```

### Methods

| Method | Description |
| --- | --- |
| `addPrimitive(name, numInstances, isDynamic=False)` | Create an instance collection |
| `addPrimitiveUnion(name, primitives)` | Create a concatenated child view |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1)` | Add mesh data/computation |
| `addConstant(name, rows=1, cols=1)` | Add differentiably constant mesh data |
| `mesh[name]` | Get one attribute |
| `mesh[name1, name2]` | Construct a flattened symbolic array |

### Properties

`name`, `scene`, `fullName`, `primitives`, `attributes`, `numPrimitives`, and
`numInstances` (`1` for a mesh-level correspondence).

## Primitive

Create through a mesh:

```python
vertices = mesh.addPrimitive(
  "vertices",
  numInstances=100,
  isDynamic=False,
)
```

### Methods

#### `addAttribute(...)`

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

Modes:

- no extra arguments: mutable data;
- `computed_attribute=expr`: name/attach an expression;
- `through=connectivity, source=attribute`: fixed JOIN;
- `through=csr, source=attribute, operation="SUM"|"AVERAGE"`: reduced gather.

#### `addConstant(name, rows=1, cols=1)`

Adds mutable per-instance data excluded from differentiation.

#### `addConnectivity(name, to, data, dimension)`

Adds a same-mesh topology mapping owned by this primitive.

#### `updateNumInstances(count)`

Changes the active count of a primitive created with `isDynamic=True`.

#### `updateConnectivity(name, data, dimension)`

Updates a named connectivity through the primitive. Calling
`connectivity.updateConnectivity(data)` directly is usually clearer.

#### `primitive[name]`

Gets a named attribute.

### Properties

`name`, `mesh`, `scene`, `fullName`, `numInstances`, `isDynamic`,
`attributes`, `attributesNames`, `connectivities`, and `numConnectivities`.

## Primitive union

Create through a mesh:

```python
union = collision_mesh.addPrimitiveUnion(
  "vertices",
  [soft_vertices, rigid_vertices],
)
```

### Recommended methods

| Method | Description |
| --- | --- |
| `addAttribute(name)` | UNION matching named child attributes |
| `union[name]` | Get an already exposed union attribute |

`numInstances` is the sum of child counts. `children_primitive_counts_gpu`
provides device prefix metadata used by generated index routing.

The implementation also accepts raw dimensions or constants directly on a
union, but child-backed named views are the stable model pattern.

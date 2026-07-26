# Scenes, meshes, and primitives

YASPS organizes a simulation into three levels:

```text
scene
├── scene attributes and constants
├── mesh
│   ├── mesh attributes and constants
│   ├── primitive
│   │   ├── per-instance attributes
│   │   └── connectivities
│   └── primitive union
└── mesh
```

The hierarchy determines symbolic lineage and scope. It does **not** assign
physical semantics.

## Scene

A scene is the global simulation scope:

```python
from yasps.scene import scene

simulation = scene("scene0")
dt = simulation.addConstant("dt", rows=1, cols=1)
dt.updateValue(0.01)
```

It owns meshes, scene-level attributes, registered energies, minimization
targets, the global gradient/Hessian layout, and the solver.

Scene names must be non-empty and unique in the current Python process.
YASPS keeps a process-global scene registry, so tests that create many scenes
should use unique names.

## Mesh

A mesh groups primitive types and provides a lineage scope:

```python
soft_body = simulation.addMesh("soft_body")
youngs_modulus = soft_body.addConstant("youngs_modulus")
youngs_modulus.updateValue(10_000.0)
```

The word “mesh” is broader than a triangle or tetrahedral mesh. A mesh may
hold vertices, tetrahedra, affine bodies, collision pairs, deformation
gradients, or any other application-defined primitive type.

## Primitive

A primitive is a collection of instances:

```python
vertices = soft_body.addPrimitive("vertices", numInstances=1000)
tets = soft_body.addPrimitive("tets", numInstances=4000)
```

Every attribute on `vertices` has 1000 logical instances; every attribute on
`tets` has 4000. Per-instance shape is stored by the attribute itself.

Primitive names do not imply built-in behavior. YASPS does not automatically
give a primitive called `"vertices"` a position, mass, or neighborhood.
Applications add those explicitly.

## Dynamic primitives

Use a dynamic primitive when runtime topology changes:

```python
point_triangle = collision_mesh.addPrimitive(
  "point_triangle",
  numInstances=0,
  isDynamic=True,
)
```

Update the active count and connectivity together:

```python
point_triangle.updateNumInstances(contact_indices.shape[0])
if contact_indices.shape[0]:
  point_triangle_to_vertex.updateConnectivity(contact_indices)
```

Dynamic connectivity buffers may keep spare allocation internally, but their
public `value` exposes only the active prefix.

## Names and access

Names must be valid Python identifiers, must not be Python keywords, and must
not conflict with an existing method/property on the owner.

Objects are available through properties and dictionaries:

```python
soft_body.vertices
soft_body.primitives["vertices"]
vertices.attributes["position"]
vertices["position"]
```

Prefer explicit variables in library code. Property access is convenient in
short examples but makes accidental name collisions harder to see.

## Lineage rule

Ordinary symbolic expressions combine attributes only when they share a
lineage: one owner must be the same as, or an ancestor of, the other owner.

- A scene constant can participate in expressions anywhere in its scene.
- A mesh constant can participate in expressions on primitives in that mesh.
- Attributes on the same primitive can be combined directly.
- Attributes on unrelated primitives require `JOIN`.
- Attributes spanning heterogeneous primitive types require a primitive
  `UNION`.

The rule lets YASPS infer the number of expression instances and build the
correct derivative propagation path.

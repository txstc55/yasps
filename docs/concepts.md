---
title: Scene, mesh, and primitive model
description: Understand YASPS scenes, meshes, primitives, unions, lineage, and how model structure becomes generated execution.
permalink: /concepts/
next_url: /attributes/
next_label: Attributes and expressions
---

<p class="eyebrow">Core concepts · chapter 02</p>

# Scene, mesh, and primitive model

YASPS is not an eager tensor library. Python constructs a typed symbolic model; generated CUDA evaluates that model over instance populations. The hierarchy tells YASPS which index each value uses, while attributes describe what to compute at that index.

## The hierarchy at a glance

```text
scene
├── scene attributes and constants
└── mesh
    ├── mesh attributes and constants
    ├── primitive
    │   ├── per-instance attributes and constants
    │   └── outgoing connectivities
    └── primitiveUnion
        └── attributes stacked from child primitives
```

The four construction levels are distinct:

| Level | Represents | Instance count | Typical contents |
| --- | --- | ---: | --- |
| `scene` | One simulation problem | Always `1` | Global timestep, barrier distance, meshes, energies, minimizer |
| `mesh` | One ownership and lineage scope | Always `1` | Material constants, primitives, unions |
| `primitive` | A population of like instances | User supplied | Vertices, tetrahedra, bodies, contact pairs |
| `primitiveUnion` | Ordered logical stack of populations | Sum of children | A shared collision-facing vertex population |

## Scenes

A scene is the root of a model and owns exactly one minimizer.

```python
from yasps import scene

world = scene("world")
```

### Constructor

```python
scene(name)
```

| Parameter | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Nonempty global scene identifier. It contributes to every generated full name. |

Scene names must be unique for the lifetime of the Python process because `scene.scenes` is a class-wide registry. Construction also creates `.yasps_tmp` and `.yasps_constant` in the current working directory when absent.

### What a scene owns

- meshes created by `addMesh`;
- scene-scoped data and constants;
- energy registrations forwarded to its minimizer;
- the ordered global minimization targets;
- assembled gradient, solution, and diagonal views.

```python
dt = world.addConstant("dt")
gravity_scale = world.addConstant("gravity_scale")
```

Scene attributes have one instance, so generated primitive kernels can treat them as shared values.

### Scene construction members

| Member | Parameters | Result |
| --- | --- | --- |
| `addMesh(name)` | Valid, unused Python identifier | Creates a mesh, stores it, and exposes it as `world.<name>` |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1)` | Name plus either shape or expression | Creates scene data or binds a computed expression |
| `addConstant(name, rows=1, cols=1)` | Name and per-instance shape | Creates mutable data excluded from differentiation |
| `scene["name"]` | Attribute name | Retrieves a named scene attribute |
| `scene["a", "b"]` | Tuple of names | Packs the named values into a row attribute |

The energy and minimization members are covered later on this page and in [Energies and minimization]({{ '/optimization/' | relative_url }}?v={{ site.time | date: '%s' }}).

## Meshes

A mesh is a namespace and lineage boundary inside one scene. It does not need to be a geometric triangle mesh: a collision workspace or a collection of affine bodies is also a mesh.

```python
soft = world.addMesh("soft")
collision = world.addMesh("collision")
```

### Construction

```python
world.addMesh(name)
```

| Parameter | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Unique valid Python identifier within the scene. It may not collide with a mesh member. |

Use `scene.addMesh`; constructing `mesh(name, parent_scene)` directly bypasses registration on the scene.

### What belongs at mesh scope

Put a value on the mesh when every primitive below it should see the same instance:

```python
mu = soft.addConstant("mu")
lam = soft.addConstant("lambda")
```

Mesh-level values can legally interact with attributes on any child primitive of that mesh. Two sibling primitives still cannot interact directly; they need JOIN or UNION.

### Mesh construction members

| Member | Parameters | Result |
| --- | --- | --- |
| `addPrimitive(name, numInstances, isDynamic=False)` | Name, count, dynamic flag | Creates a primitive population |
| `addPrimitiveUnion(name, primitives)` | Name and ordered child list | Creates a logical stacked population |
| `addAttribute(name, computed_attribute=None, rows=1, cols=1)` | Name plus expression or shape | Creates/binds mesh-scoped data |
| `addConstant(name, rows=1, cols=1)` | Name and shape | Creates mesh-scoped nondifferentiable data |
| `mesh["name"]` / `mesh["a", "b"]` | Name or tuple | Retrieves or packs mesh attributes |

## Primitives

A primitive is a population of instances evaluated by one generated kernel index. YASPS assigns no geometric meaning to its name.

```python
vertices = soft.addPrimitive(
  "vertices",
  numInstances=num_vertices,
)
tets = soft.addPrimitive(
  "tets",
  numInstances=num_tets,
)
```

### Constructor parameters

```python
mesh.addPrimitive(name, numInstances, isDynamic=False)
```

| Parameter | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Unique valid Python identifier within the mesh |
| `numInstances` | Yes | Number of rows in this population; each attribute stores this many per-instance values |
| `isDynamic` | No | Permits `updateNumInstances`; intended for runtime-changing contact populations |

The flattened storage required by an attribute on this primitive is:

```text
numInstances × rows × cols
```

### Fixed primitives

Vertices, elements, and affine bodies are normally fixed. Their count participates in generated buffer sizes and sparse layouts:

```python
position = vertices.addAttribute("position", rows=3, cols=1)
rest_position = vertices.addConstant("rest_position", rows=3, cols=1)
```

### Dynamic primitives

Contact candidates commonly change count:

```python
pairs = collision.addPrimitive(
  "pairs",
  numInstances=0,
  isDynamic=True,
)
```

Only dynamic primitives accept:

```python
pairs.updateNumInstances(new_count)
```

The count must be nonnegative. Dynamic attributes cannot be global minimization targets.

### Primitive construction members

| Member | Parameters | Result |
| --- | --- | --- |
| `addAttribute(...)` | Data, computed, or JOIN arguments | Adds a per-instance symbolic value |
| `addConstant(name, rows=1, cols=1)` | Name and shape | Adds per-instance nondifferentiable data |
| `addConnectivity(name, to, data, dimension)` | Directed target, indices, arity | Adds an outgoing topology relation |
| `updateNumInstances(count)` | Nonnegative count | Changes a dynamic population |
| `updateConnectivity(name, data, dimension)` | Existing name, indices, unchanged arity | Updates topology through the primitive |
| `primitive["name"]` | Attribute name | Retrieves a named attribute |

The several `addAttribute` construction modes are separated in [Attributes and expressions]({{ '/attributes/' | relative_url }}?v={{ site.time | date: '%s' }}).

## Primitive unions

A primitive union presents several populations as one ordered population without erasing where each value came from.

```python
all_vertices = collision.addPrimitiveUnion(
  "vertices",
  [soft_vertices, affine_vertices, container_vertices],
)
all_position = all_vertices.addAttribute("position")
```

### Constructor parameters

```python
mesh.addPrimitiveUnion(name, primitives)
```

| Parameter | Required | Meaning |
| --- | --- | --- |
| `name` | Yes | Unique valid Python identifier within the owning mesh |
| `primitives` | Yes | Ordered list of `primitive` or nested `primitiveUnion` children |

`numInstances` is computed every time as the sum of child counts. Instance ranges follow list order and remain contiguous.

### Union attribute modes

```python
union.addAttribute(name, computed_attribute=None, rows=0, cols=0)
```

- With only `name`, every child must expose that name with identical `rows` and `cols`; the result is a symbolic UNION.
- With `computed_attribute`, the expression must belong to this union (or have no correspondence).
- With positive `rows` and `cols`, a new union-owned data attribute is created.
- `addConstant(name, rows, cols)` creates union-owned nondifferentiable data rather than stacking child constants.

For collision work, the first form is usually correct because it preserves each child's route to its own degrees of freedom.

## Attributes connect structure to computation

An attribute has:

- a per-instance shape (`rows × cols`);
- a correspondence (`scene`, `mesh`, `primitive`, or `primitiveUnion`);
- an operator such as DATA, CONSTANT, JOIN, UNION, or arithmetic;
- zero or more child attributes.

```python
position = vertices.addAttribute("position", rows=3, cols=1)
displacement = position - rest_position
```

`displacement` is only a Python graph. No full intermediate array is produced. When requested, YASPS emits a fused kernel over `vertices.numInstances`.

## Lineage determines legal expressions

YASPS must identify one current instance index for every leaf:

- values on the same correspondence may interact;
- scene values may interact with descendants in that scene;
- mesh values may interact with descendants in that mesh;
- unrelated primitive values must first be related by JOIN or UNION.

```python
# Invalid: tets and vertices use different instance indices.
bad = tets["volume"] + vertices["mass"]
```

This restriction is not cosmetic. It is what makes kernel arguments, topology traversal, differentiation paths, and sparse placement derivable from the symbolic expression.

## Connectivity and JOIN

A connectivity maps each source instance to target indices. JOIN gathers a target attribute into source index space:

```python
tet2v = tets.addConnectivity(
  "tet2v",
  to=vertices,
  data=tet_indices,
  dimension=4,
)
tet_position = tets.addAttribute(
  "position",
  through=tet2v,
  source=position,
)
```

A `3 × 1` vertex position through arity four becomes a `4 × 3` per-tet value. Differentiation follows the connectivity backward to place contributions into vertex blocks. See [Connectivity and JOIN]({{ '/join/' | relative_url }}?v={{ site.time | date: '%s' }}) for fixed and variable arity.

## Energies and the solve

An energy is a named scalar attribute evaluated once for each instance of its correspondence:

```python
elastic = tets.addAttribute(
  "elastic",
  computed_attribute=elastic_expression,
)
world.addEnergy(elastic, projection_method=2)
world.addMinimizeTarget([position])
```

`addMinimizeTarget` establishes the global vector layout and triggers symbolic differentiation. A solve assembles:

```text
H Δx = g
```

and returns target-aligned GPU views. It does not update state:

```python
direction = world.minimizeEnergy()[0]
position.updateValue(
  position.value - direction,
  deepCopy=True,
)
```

The application owns Newton iteration, line search, collision detection, CCD, timestep acceptance, and velocity updates.

## Static versus dynamic structure

Static energy terms build sparse coordinates once. Dynamic terms rebuild or refresh their coordinate compression when counts and connectivities change:

```python
pairs.updateNumInstances(pair_count)
if pair_count:
  pair2vertex.updateConnectivity(pair_indices)

world.addEnergy(
  contact_energy,
  dynamic_instances=True,
)
```

The symbolic contact expression remains fixed; the population and indices vary numerically.

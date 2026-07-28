---
title: Mental model
description: Understand YASPS scenes, lineage, symbolic attributes, JOIN, UNION, and generated execution.
permalink: /concepts/
next_url: /attributes/
next_label: Attributes and expressions
---

<p class="eyebrow">Core concepts</p>

# The YASPS mental model

YASPS is not an eager tensor library. An operation such as `a + b + c` constructs one symbolic computation graph. When the result is needed, YASPS emits a fused CUDA computation rather than materializing `a + b` as an intermediate array.

## The hierarchy

Every model starts with a hierarchy:

```text
scene
├── scene-level attributes and constants
└── mesh
    ├── mesh-level attributes and constants
    ├── primitive
    │   ├── per-instance attributes
    │   └── outgoing connectivities
    └── primitiveUnion
        └── unioned attributes
```

A primitive is a collection of instances: vertices, tetrahedra, affine bodies, collision pairs, or any application-defined entity. YASPS does not assign geometric meaning to a primitive name.

```python
world = scene("world")
bunny = world.addMesh("bunny")

vertices = bunny.addPrimitive("vertices", numInstances=n_vertices)
tets = bunny.addPrimitive("tets", numInstances=n_tets)
```

The names must be valid Python identifiers and must not collide with existing members. Created meshes and primitives are available both as return values and as attributes such as `world.bunny` or `bunny.vertices`.

## Attributes are per-instance expressions

For a primitive with `n` instances, an attribute declared with `rows=r, cols=c` represents a per-instance `r × c` value. Its numerical storage is flattened to `nrc` values.

```python
position = vertices.addAttribute("position", rows=3, cols=1)
```

The Python `attribute` object represents only the per-instance expression and its lineage. The generated global kernel broadcasts that expression over all instances of its correspondence.

There are two leaf kinds:

- `addAttribute` creates differentiable data and can be a minimization target.
- `addConstant` creates mutable numerical data whose derivative is always zero.

“Constant” therefore means constant with respect to differentiation, not immutable during the simulation.

## Lineage controls legal expressions

YASPS needs to know which instance index should be used when evaluating every leaf in an expression. It enforces this through lineage:

- attributes on the same primitive may interact;
- a scene or mesh attribute may interact with attributes below it;
- attributes on unrelated primitives may not interact directly.

For example, a tetrahedron attribute and a vertex attribute do not share an instance index. Their relationship must be made explicit using a connectivity and JOIN.

This rule is what lets YASPS infer kernel arguments and indexing without asking the user to write a CUDA launch interface.

## JOIN moves data along topology

A connectivity maps each source instance to one or more target instances. JOIN gathers a target attribute through that map and attaches the result to the source primitive.

```python
tet2v = tets.addConnectivity("tet2v", vertices, tet_indices, 4)
tet_positions = tets.addAttribute(
  "positions",
  through=tet2v,
  source=vertices["position"],
)
```

If `position` is `3 × 1` and the connectivity arity is four, `tet_positions` has per-instance shape `4 × 3`.

## UNION preserves heterogeneous origins

UNION stacks shape-compatible attributes from multiple primitives while preserving the symbolic route back to each source:

```python
collision_mesh = world.addMesh("collision")
all_vertices = collision_mesh.addPrimitiveUnion(
  "vertices",
  [soft_vertices, affine_vertices],
)
all_positions = all_vertices.addAttribute("position")
```

The children must all contain an attribute named `position` with the same shape. A downstream collision energy can now be written once against `all_positions`, even though some positions are direct degrees of freedom and others are computed from affine parameters.

## Energies are scalar attributes

An energy is a named scalar attribute evaluated once per instance of its correspondence:

```python
term = pairs.addAttribute("barrier", computed_attribute=barrier_expression)
world.addEnergy(term, dynamic_instances=True, projection_method=2)
```

YASPS sums those per-instance contributions conceptually when building the global gradient and Hessian.

## The solve returns a direction, not a timestep

`minimizeEnergy` assembles `H` and `g` and solves:

```text
H Δx = g
```

It returns one flattened GPU segment per minimization target, in the same order passed to `addMinimizeTarget`. YASPS does not apply the update and does not negate it:

```python
directions = world.minimizeEnergy()
x.updateValue(x.value - directions[0], deepCopy=True)
```

Your application remains responsible for Newton iteration, line search, continuous collision detection, timestep acceptance, and state updates such as position-to-velocity conversion.

## Static and dynamic structure

Static primitives keep a fixed instance count. Dynamic primitives are intended for collision pairs or other runtime-changing sets:

```python
pairs = mesh.addPrimitive("pairs", numInstances=0, isDynamic=True)
pair2v = pairs.addConnectivity("pair2v", vertices, [], 2)
```

At runtime, update both the count and the connectivity before evaluating the corresponding dynamic energy:

```python
pairs.updateNumInstances(pair_count)
if pair_count:
  pair2v.updateConnectivity(pair_indices)
```

Dynamic energies must be registered with `dynamic_instances=True`.

# Primitive unions

A primitive union concatenates multiple primitive types into one logical
index space while preserving each child's symbolic dependency path.

This is central to contact between different parameterizations. A soft
vertex may store position directly, while an affine-body vertex computes
position from a matrix and translation. Collision energy should still be
written once.

## Constructing a union

Both child primitives must be in the list supplied to a mesh:

```python
collision_mesh = simulation.addMesh("collision_mesh")
all_vertices = collision_mesh.addPrimitiveUnion(
  "vertices",
  [soft_vertices, affine_vertices],
)
```

The union instance count is the sum of child counts, in child-list order.
Indices `[0, soft_vertices.numInstances)` address the first child; the next
range addresses the second child.

## Unioning an existing named attribute

Every child must have the same named attribute with the same per-instance
shape:

```python
collision_position = all_vertices.addAttribute("position")
```

The result is a symbolic `UNION` whose numerical value is the stacked child
values. Derivatives are routed back to the appropriate child's parameters.

## Using a union in JOIN

Dynamic collision pairs can target the union:

```python
point_pairs = collision_mesh.addPrimitive(
  "point_pairs",
  numInstances=0,
  isDynamic=True,
)
pair_to_vertex = point_pairs.addConnectivity(
  "pair_to_vertex",
  to=all_vertices,
  data=[],
  dimension=2,
)
pair_position = point_pairs.addAttribute(
  "positions",
  through=pair_to_vertex,
  source=collision_position,
)
```

An energy written in terms of `pair_position` is independent of whether each
index refers to a soft or affine child.

## Recommended usage boundary

Treat a primitive union as a view over child attributes:

- create the physical data on child primitives;
- give matching concepts the same name and shape;
- call `union.addAttribute(name)` to expose the view;
- join or compute from that view.

The current implementation can create raw attributes/constants directly on a
union when dimensions are supplied, but those values do not represent a
normal child-backed UNION dependency. Avoid that path in portable model code.

## Why UNION is differentiable

Materializing a union is simple concatenation, but symbolic differentiation
does not forget origin. During sparse index construction, YASPS maps each
union instance back to a child range and then follows that child's JOIN/data
path. The final global gradient and Hessian therefore use the correct degrees
of freedom without generating one energy implementation for every
parameterization combination.

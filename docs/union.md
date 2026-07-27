---
title: Primitive unions
description: Combine shape-compatible attributes from heterogeneous primitives with UNION.
permalink: /union/
---

<p class="eyebrow">Core syntax</p>

# Primitive unions

UNION gives downstream code one logical primitive while retaining the symbolic route to several heterogeneous sources. Its common use is collision processing across soft-body vertices, affine-body vertices, or multiple meshes.

## Create a union

Each child must expose an attribute with the same name and per-instance shape:

```python
# Direct degrees of freedom on a soft body.
soft_position = soft_vertices.addAttribute("position", rows=3, cols=1)

# A computed position on an affine body.
affine_position = affine_vertices.addAttribute(
    "position",
    computed_attribute=translation + transform * rest_position,
)

collision = world.addMesh("collision")
all_vertices = collision.addPrimitiveUnion(
    "vertices",
    [soft_vertices, affine_vertices],
)
all_position = all_vertices.addAttribute("position")
```

`all_position` contains the soft vertices first and the affine vertices second. `all_vertices.numInstances` is the sum of the child counts.

The union may be owned by a different mesh from its child primitives. This is useful for building a dedicated collision mesh over simulation objects that otherwise retain separate topology and attributes.

## What is preserved

UNION is not a numerical concatenation that loses provenance. Each range in the union retains a path to its child attribute:

```text
union position
├── soft vertex position       → direct soft degrees of freedom
└── affine vertex position     → translation and transform parameters
```

A single collision energy written against `all_position` can therefore differentiate into different target parameterizations.

## Attribute rules

The usual form queries child attributes by name:

```python
all_velocity = all_vertices.addAttribute("velocity")
```

For this to succeed:

- every child has an attribute named `velocity`;
- every child version has identical `rows` and `cols`;
- the new name has not already been used on the union.

Child attributes may be data leaves or arbitrary named computed expressions. Sparse symbolic matrices with structurally zero entries are handled by unioning their nonzero elements and rebuilding the full shape.

You can also bind a computed attribute to the union:

```python
speed = all_velocity.norm()
all_vertices.addAttribute("speed", computed_attribute=speed)
```

`primitiveUnion.addConstant` exists, but it creates data owned by the union rather than unioning values from children. For heterogeneous parameters, prefer defining the constant on every child and using `addAttribute(name)` so lineage remains explicit.

## Nested unions

A union may contain primitives or other primitive unions:

```python
all_objects = collision.addPrimitiveUnion(
    "all_objects",
    [soft_union, affine_union, rigid_vertices],
)
```

Ordering remains recursive and deterministic: each child's instances occupy a contiguous range in the order supplied to `addPrimitiveUnion`.

## Dynamic children

The generated UNION code receives child counts so it can map a union index to the correct source. If child counts change, the counts buffer is regenerated when accessed. In practice, the dynamic collision *pairs* are usually a separate dynamic primitive, while the unioned simulation vertices remain fixed for the duration of a solve.

## When to use JOIN versus UNION

| Need | Operation |
| --- | --- |
| Gather target instances through explicit indices | [JOIN]({{ '/join/' | relative_url }}) |
| Stack complete primitives into one logical population | UNION |
| Reduce a variable number of neighbors | JOIN with `"SUM"` or `"AVERAGE"` |
| Make unrelated arrays interact without topology | Define either a JOIN or UNION; do not bypass lineage |

The mixed examples are the best complete references: [dropping_in_container_mixed](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container_mixed) and [two_bunnies_abd_soft](https://github.com/txstc55/yasps/tree/main/examples/two_bunnies_abd_soft).

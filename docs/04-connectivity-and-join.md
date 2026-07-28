---
title: Connectivity and JOIN
chapter: "04"
description: Define topology and gather attributes across primitives with fixed or variable-arity connectivity.
permalink: /join/
next_url: /union/
next_label: Primitive unions
---

<p class="eyebrow">Core syntax</p>

# Connectivity and JOIN

Attributes on different primitives do not share an instance index. A connectivity makes their relationship explicit, and JOIN gathers a target attribute into the source primitive's index space.

## Connectivity construction

```python
edge = source.addConnectivity(
  name,
  to=target,
  data=index_rows,
  dimension=arity,
)
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `name` | `str` | Unused connectivity identifier on `source`. |
| `to` | `primitive` | Target instance space. It must belong to the same mesh as `source`. |
| `data` | NumPy array or nested list | Target indices. Values are flattened and uploaded as uint32. |
| `dimension` | nonnegative `int` | Number of target indices per source instance. Zero selects variable-arity CSR conversion. |

The returned `connectivity` records the direction from `source` to `target`.
Its generated index name is used by JOIN kernels and its hash contributes to
the symbolic lineage.

## Fixed-arity connectivity

Suppose each tetrahedron references four vertices:

```python
tet2vertex = tetrahedra.addConnectivity(
  "tet2vertex",
  to=vertices,
  data=tet_indices,  # shape (num_tets, 4), integer-valued
  dimension=4,
)
```

The direction matters. `tetrahedra` is the source primitive, `vertices` is the target, and every tetrahedron owns four target indices.

Gather the vertex positions onto each tetrahedron:

```python
tet_positions = tetrahedra.addAttribute(
  "positions",
  through=tet2vertex,
  source=vertices["position"],
)
```

The fixed-arity JOIN mode uses:

```python
joined = source.addAttribute(
  name,
  through=edge,
  source=target_attribute,
)
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `name` | `str` | New attribute key on the connectivity's source primitive. |
| `through` | `connectivity` | Outgoing edge whose source must be this primitive. |
| `source` | `attribute | None` | Target attribute to gather. When omitted, YASPS looks up `edge.toPrimitive[name]`. |
| `computed_attribute` | must be `None` | Expression-binding and JOIN modes are mutually exclusive. |
| `rows`, `cols` | ignored by JOIN shape | Output shape is derived from connectivity arity and source shape. |
| `operation` | `None` for fixed arity | Reductions are reserved for `dimension=0`. |

For a source attribute with shape `r × c` and connectivity arity `k`, JOIN produces a `k × (rc)` attribute. A four-to-one JOIN of a `3 × 1` position therefore becomes a `4 × 3` per-tetrahedron matrix.

If the source attribute has the same name as the new attribute, `source` may be omitted:

```python
tet_positions = tetrahedra.addAttribute(
  "position",
  through=tet2vertex,
)
```

YASPS looks up `vertices["position"]`.

## Why JOIN is symbolic

JOIN does not eagerly construct a full gathered array. It contributes indexing code to the generated kernel. Derivatives follow that same path in reverse, allowing one local energy contribution to accumulate into the correct global vertex gradient and Hessian blocks.

This is the reason direct arithmetic between unrelated primitives is rejected:

```python
# Invalid: the two operands use different instance indices.
bad = tetrahedra["volume"] + vertices["mass"]

# Valid: first move mass into tetrahedron space.
tet_mass = tetrahedra.addAttribute(
  "vertex_mass",
  through=tet2vertex,
  source=vertices["mass"],
)
valid = tetrahedra["volume"] + tet_mass.row(0)
```

## Nested JOINs

JOINs may be chained. For example, a collision-pair primitive can gather faces, while those faces gather vertices. The differentiation path records every connectivity so the generated sparse indices still resolve to the original minimization targets.

Name useful gathered results with `addAttribute`. Besides improving readability, named nodes provide stable modular boundaries for code generation.

## Variable-arity connectivity

Use `dimension=0` for a list of neighbors whose length varies per source instance:

```python
vertex2face = vertices.addConnectivity(
  "vertex2face",
  to=faces,
  data=[
    [0, 3, 8],
    [0, 1],
    [],
  ],
  dimension=0,
)

incident_area = vertices.addAttribute(
  "area",
  through=vertex2face,
  operation="SUM",
)
```

The variable-arity JOIN parameters are:

| Parameter | Required value |
| --- | --- |
| `name` | Both the new source-side name and the implicit target-side attribute lookup. |
| `through` | A connectivity constructed with `dimension=0`. |
| `source` | Omit it in the current implementation. |
| `operation` | `"SUM"` or `"AVERAGE"`. |

The input is converted to a CSR-style flattened index array plus row offsets. A reduction is mandatory:

- `"SUM"` sums the gathered values;
- `"AVERAGE"` divides that sum by the number of neighbors.

The current variable-arity implementation applies the reduction through the implicit same-name lookup, so omit `source` and use the target attribute's name as shown above.

<div class="callout warning">
  <strong>Differentiation limitation.</strong> The current Hessian path rejects variable-arity JOINs. Use them for materialized computations such as normals or diagnostics, not inside an energy differentiated by <code>addMinimizeTarget</code>.
</div>

## Updating connectivity

For a dynamic, fixed-arity primitive, update the instance count and connectivity together:

```python
pairs.updateNumInstances(num_pairs)
if num_pairs:
  pair2vertex.updateConnectivity(pair_indices)
```

The object-level update signature is:

```python
pair2vertex.updateConnectivity(value)
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `value` | NumPy array, nested list, or PyCUDA `GPUArray` | Replaces the flattened device indices, reusing capacity when possible. |

The primitive convenience call is:

```python
pairs.updateConnectivity(
  name="pair2vertex",
  data=pair_indices,
  dimension=4,
)
```

Here `name` selects an existing connectivity, `data` is forwarded as the new
value, and `dimension` must match its stored arity.

The update path is designed for fixed arity. Construct variable-arity CSR connectivity up front; its row-offset metadata is not rebuilt by the same dynamic update path.

## Connectivity constraints

- Source and target primitives must belong to the same mesh.
- `dimension` is the number of target indices per source instance; use zero only for CSR reductions.
- Connectivity data is converted to unsigned 32-bit indices.
- The number and ordering of connectivity rows must match the source primitive's instances.
- A JOIN must be added to the connectivity's source primitive.

See [Dynamic topology]({{ '/dynamic-scenes/' | relative_url }}?v={{ site.time | date: '%s' }}) for the collision-pair lifecycle and [How YASPS executes]({{ '/architecture/' | relative_url }}?v={{ site.time | date: '%s' }}) for how JOIN paths become sparse matrix indices.

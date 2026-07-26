# Connectivity and JOIN

A connectivity maps each instance of one primitive type to instances of
another primitive type.

## Fixed-arity connectivity

For two endpoints per edge:

```python
edge_to_vertex = edges.addConnectivity(
  "edge_to_vertex",
  to=vertices,
  data=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32),
  dimension=2,
)
```

The connectivity is owned by `edges` (`fromPrimitive`) and targets
`vertices` (`toPrimitive`). Both must belong to the same mesh.

`dimension=2` means every edge references exactly two vertices.

## JOIN

JOIN gathers a target attribute through a connectivity and attaches the
result to the source primitive:

```python
edge_position = edges.addAttribute(
  "position",
  through=edge_to_vertex,
  source=vertices["position"],
)
```

If the source position is 3×1 and connectivity dimension is 2, the joined
attribute is 2×3 per edge. Each row is one gathered position.

```python
edge_vector = edge_position.row(1) - edge_position.row(0)
length = edge_vector.norm()
```

When the requested name exists on the target primitive, `source` can be
omitted:

```python
edge_position = edges.addAttribute(
  "position",
  through=edge_to_vertex,
)
```

Supplying `source` explicitly is clearer when multiple attributes have
similar names.

## Chained joins

A joined or computed attribute can be named on an intermediate primitive and
joined again. This is how YASPS propagates derivatives through multiple
topological levels, for example:

```text
affine body parameters
  -> vertex positions
  -> tetrahedron positions
  -> tetrahedron energy
```

The differentiation system emits the block chain rule across those JOIN
boundaries.

## Variable-arity connectivity

Set `dimension=0` for a list-of-lists adjacency represented internally as
CSR:

```python
vertex_to_edge = vertices.addConnectivity(
  "vertex_to_edge",
  to=edges,
  data=[[0], [0, 1], [1, 2], [2]],
  dimension=0,
)
```

A variable-arity gather requires reduction:

```python
vertex_force = vertices.addAttribute(
  "force",
  through=vertex_to_edge,
  source=edges["force"],
  operation="SUM",
)

vertex_average = vertices.addAttribute(
  "average_force",
  through=vertex_to_edge,
  source=edges["force"],
  operation="AVERAGE",
)
```

`SUM` and `AVERAGE` work for numerical computation. The current symbolic
differentiator does not support variable-arity reduction paths as generally
as fixed-arity JOIN; do not use them inside an energy without a focused test.

## Dynamic connectivity

Collision pairs commonly change each Newton iteration:

```python
pairs.updateNumInstances(indices.shape[0])
if indices.shape[0]:
  pair_to_vertex.updateConnectivity(indices)
```

Call `updateNumInstances()` before evaluating joined attributes or
differentiating the active numerical term. Fixed-arity connectivity exposes
only the active prefix even when its internal allocation is larger.

## Common mistakes

- Reversing `from` and `to`: call `addConnectivity()` on the primitive whose
  instances own the index tuples.
- Supplying flat data with the wrong arity.
- Joining across different meshes. Use a primitive union under a shared mesh
  when heterogeneous data needs a common representation.
- Treating JOIN output rows as columns. A dimension-\(k\) gather stacks
  \(k\) source tensors and flattens their trailing per-instance shape.

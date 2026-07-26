# Dynamic collision terms

Collision energy differs from static mesh energy because pair counts and
connectivity change at runtime.

## Dynamic pair primitive

```python
point_triangle = collision_mesh.addPrimitive(
  "point_triangle",
  numInstances=0,
  isDynamic=True,
)
point_triangle_to_vertex = point_triangle.addConnectivity(
  "point_triangle_to_vertex",
  to=collision_vertices,
  data=[],
  dimension=4,
)
positions = point_triangle.addAttribute(
  "positions",
  through=point_triangle_to_vertex,
  source=collision_vertices["position"],
)
```

Write one scalar energy for each active pair and register it with
`dynamic_instances=True`.

## Updating active pairs

```python
point_triangle.updateNumInstances(pt_indices.shape[0])
if pt_indices.shape[0]:
  point_triangle_to_vertex.updateConnectivity(pt_indices)
```

The next numerical Hessian assembly recomputes the dynamic term's sparse
coordinates and active values.

## CCD flow

A Newton trial for the dropping example follows:

1. Keep the current all-vertex positions.
2. Apply the proposed unconstrained direction.
3. Run swept broad phase and continuous collision detection.
4. Limit the feasible step.
5. Apply a trial fraction of the direction.
6. Run discrete contact classification at the trial position.
7. Update PP, PE, PT, and EE primitives/connectivities.
8. Evaluate total energy and backtrack if necessary.

The CUDA collision helper lives in `examples/ccd/ccd.py`.

## Contact encodings

After feature classification, YASPS separates contacts into:

| Type | Connectivity width | Meaning |
| --- | ---: | --- |
| PP | 2 | point–point |
| PE | 3 | point–edge |
| PT | 4 | point–triangle |
| EE | 4 | edge–edge |

## Friction timing

Friction uses the contact set from the previous accepted frame:

1. copy current position to `last_position`;
2. update friction primitives from prior contacts;
3. compute closest coordinates, tangent bases, and lagged normal forces;
4. keep that friction topology fixed during the current frame's Newton loop.

Barrier contact topology still changes during the line search.

## Fixed geometry and filtering

The detector supports per-vertex body types and mesh identifiers. The default
zero arrays allow self-collision subject to shared-feature filtering.
Applications with multiple bodies can supply mesh identifiers to omit
unwanted same-body candidates.

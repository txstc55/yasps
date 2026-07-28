---
title: Dynamic topology
description: Update runtime instance counts and fixed-arity connectivity for collision and contact energies.
permalink: /dynamic-scenes/
next_url: /tutorials/mixed-separation/
next_label: Assemble the mixed-body capstone
---

<p class="eyebrow">Simulation integration</p>

# Dynamic topology

Collision candidates and contact stencils change while the simulation runs. YASPS supports this with dynamic primitives and dynamic energy terms while keeping the symbolic energy expression fixed.

## Define the dynamic primitive once

Create the primitive and its fixed-arity connectivity during setup:

```python
collision = world.addMesh("collision")

pairs = collision.addPrimitive(
  "pairs",
  numInstances=0,
  isDynamic=True,
)

pair2vertex = pairs.addConnectivity(
  "pair2vertex",
  to=all_vertices,
  data=[],
  dimension=4,
)
```

The expression can be built even when there are no active pairs:

```python
pair_position = pairs.addAttribute(
  "position",
  through=pair2vertex,
  source=all_vertices["position"],
)

barrier_expression = build_barrier(pair_position)
barrier = pairs.addAttribute(
  "barrier",
  computed_attribute=barrier_expression,
)

world.addEnergy(
  barrier,
  projection_method=2,
  dynamic_instances=True,
  separate_hessian_jacobian=True,
)
```

`dynamic_instances=True` is essential. It routes the term through the dynamic sparse-index and numerical-assembly paths.

## Update every iteration

After collision detection produces a new two-dimensional index array (or an equivalent device buffer):

```python
num_pairs = len(pair_indices)
pairs.updateNumInstances(num_pairs)

if num_pairs:
  pair2vertex.updateConnectivity(pair_indices)
```

Keep the instance count and connectivity consistent before calling `minimizeEnergy`, `computeTotalEnergy`, or materializing an expression that depends on the pairs.

The zero-count guard matters because the fixed-arity update path infers shape information from nonempty CPU inputs. Leaving the old buffer allocated is safe: kernels use the updated instance count and ignore stale capacity.

## GPU-resident indices

`connectivity.updateConnectivity` accepts a PyCUDA `GPUArray`. It may copy into retained capacity or allocate a new device buffer as needed. Keeping collision output on the GPU can therefore avoid a CPU round trip, provided the data is unsigned-index compatible and already laid out at the expected fixed arity.

## What changes at runtime

For a dynamic energy, YASPS recomputes the term's active sparse coordinates, block placements, and numerical contributions. Static terms keep their precomputed structure. Both sets are merged into one active Hessian before the PCG solve.

The symbolic graph itself is not rebuilt. Create attributes and register energies once, then update only:

- dynamic primitive counts;
- connectivity indices;
- leaf attribute or constant values.

## Ownership boundary

YASPS does not perform broad-phase collision detection, narrow-phase candidate construction, continuous collision detection, or line search. The examples build those pieces around the symbolic objective. A typical frame is:

```text
predict state
→ detect candidates
→ update dynamic primitive/connectivity
→ solve with YASPS
→ compute collision-free step
→ line search and apply update
→ update velocity/state
```

For a full implementation, start with [dropping_in_container](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container). The mixed and separation variants add UNION-based heterogeneous bodies and more specialized contact terms.

## Current limitations

- Dynamic updates are intended for fixed-arity connectivity.
- Variable-arity (`dimension=0`) JOINs are not supported by the Hessian differentiation path.
- Changing a static primitive's topology after minimization-target registration is outside the intended lifecycle.
- A dynamic term with no active instances contributes nothing, but the global system still needs sufficient static terms or constraints to be solvable.

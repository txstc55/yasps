---
title: Differentiation
description: Use the local autodiff engine and the topology-aware second-order differentiator directly.
permalink: /advanced/differentiation/
---

<p class="eyebrow">Advanced API</p>

# Differentiation

YASPS has two related differentiation layers:

- `autodiff` applies local symbolic derivative rules to attribute expressions;
- `differentiator` follows JOIN/UNION paths, builds global second-order terms, and prepares sparse-index generators.

Most users should register energies with a minimizer. Use these classes when developing new symbolic operators, inspecting generated derivatives, or composing a custom assembly pipeline.

## Local symbolic derivative

```python
from yasps import autodiff

engine = autodiff()
d_energy_dx = engine.diff(energy_expression, x)
```

The result is another `attribute` graph. It can be named and computed like any other expression:

```python
d_energy_dx = points.addAttribute(
  "energy_gradient_local",
  computed_attribute=d_energy_dx,
)
values = d_energy_dx.compute().value
```

This layer differentiates expression operators. It does not perform topology-aware global accumulation or produce a sparse matrix.

## Topology-aware second derivative

```python
from yasps import differentiator

builder = differentiator()
H = builder.diff2(
  source=[energy],
  target1=[position],
  target2=[position],
  local_targets=[],
  projection_method=2,
  save_intermediate=False,
  separate_hessian_jacobian=False,
  dynamic_instances=False,
)
```

`source` may be one scalar attribute or a list of scalar attributes. A list is differentiated term by term and the resulting `hessian` objects are added.

`target1` and `target2` must currently contain the same attributes in the same order. General mixed second-order Jacobians are not implemented.

## Global and local targets

The global target list defines vector offsets and matrix dimensions:

```python
global_targets = [soft_position, affine_matrix, translation]
```

The optional local list limits which of those targets are differentiated for this term:

```python
H_soft = builder.diff2(
  [soft_energy],
  global_targets,
  global_targets,
  local_targets=[soft_position],
)
```

`H_soft` still uses the full global layout, but generates derivatives and sparse coordinates only for `soft_position`. Every local target should therefore appear in the global list.

## Materialize a Hessian

```python
H.compute()

gradient_object = H.gradient
gradient_gpu = gradient_object.value
segments = gradient_object.gradient_segments

static_blocks = H.blocks_flattened
static_positions = H.block_positions
```

On first use, `compute()` creates the sparse index mapping and JIT-compiles numerical assembly kernels. On later calls, static topology is reused; dynamic terms refresh their coordinates.

Direct `H.compute()` fills diagonal blocks but does not build the inverse block diagonal that the standard minimizer prepares for PCG. Use a `minimizer` if you intend to call the stock solver.

## Projection and separation controls

`projection_method`, `save_intermediate`, `separate_hessian_jacobian`, and `dynamic_instances` have the same meaning as in `addEnergy`.

`separate_hessian_jacobian=True` is intended for paths where a local inner Hessian and a sparse outer Jacobian can be generated and applied separately. It can reduce generated expression size and avoid dense symbolic expansion, but it adds another kernel stage and placement-reordering metadata. Benchmark it for the relevant energy; it is not universally faster.

`save_intermediate=True` marks reusable derivative expressions for materialization. That may reduce repeated computation at the cost of extra buffers and launches.

## Current limitations

- `differentiator.diff1(...)` is declared but currently has no implementation.
- `diff2` accepts only identical left and right target lists.
- Every source term must be scalar.
- Dynamic attributes cannot themselves be minimization targets.
- Variable-arity JOIN paths are rejected.

Use `autodiff().diff(...)` for a local first derivative, and use `gradient(wrt, hessian)` to expose the gradient assembled alongside a second-order term.

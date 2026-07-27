---
title: Advanced API reference
description: Signatures for minimization, differentiation, block-sparse numerical objects, and legacy energy assembly.
permalink: /reference/advanced/
---

<p class="eyebrow">Reference</p>

# Advanced API reference

These symbols are exported at the package root:

```python
from yasps import (
    minimizer, differentiator, autodiff, path,
    hessian, gradient, solver, matrix, vector, energy,
)
```

## `minimizer`

```python
minimizer()
```

| Member | Return / effect |
| --- | --- |
| `addEnergy(e, targets=[], projection_method=1, save_intermediate=False, gradient_only=False, dynamic_instances=False, separate_hessian_jacobian=False)` | Store one symbolic energy request |
| `addEnergies(energies)` | Add terms with default options |
| `addWrt(wrt)` | Define global target layout |
| `generateHessianAndGradient()` | Build symbolic derivatives if dirty |
| `computeNumericValue()` | Assemble and return the active `hessian` |
| `computeSolution(tolerance=1e-3, maxIterations=20000)` | Assemble, solve, return target segments |
| `computeHessianAndGradient(tolerance=1e-3, maxIterations=20000)` | Assemble, solve, return solver status |
| `computeTotalEnergy() -> float` | Sum nonignored energies |
| `ignoreEnergies(energies)` | Replace ignored-term list |

Properties: `solutionSegments`, `gradient`, `gradientSegments`, `energies`, `energiesDynamic`, `wrt`, and `diagonal`.

## `differentiator`

```python
differentiator()
```

```python
H = differentiator().diff2(
    source,
    target1,
    target2,
    local_targets=[],
    projection_method=1,
    save_intermediate=False,
    separate_hessian_jacobian=False,
    dynamic_instances=False,
)
```

`target1` and `target2` must currently be identical. `diff1(source, global_targets, local_targets=[], dynamic_instances=False)` is declared but unimplemented.

## `autodiff`

```python
derivative = autodiff().diff(current, wrt)
```

Produces a local symbolic derivative expression. It does not create global sparse indices.

## `path`

```python
path(global_targets, local_targets=[])
```

`getRoots(attribute, parent_path)` discovers valid target routes. `getPathDict()` finalizes dictionaries consumed by index generation and `wrt` exposes the effective target subset. This is generator-facing API; inspect [How YASPS executes]({{ '/architecture/' | relative_url }}) before using it directly.

## `hessian`

```python
hessian(wrt, local_targets=[], dynamic_instances=False)
```

| Member | Description |
| --- | --- |
| `H0 + H1` | Combine terms with the same global targets |
| `getSparseIndices()` | Build static coordinate lookup |
| `getSparseIndicesDynamic()` | Build initial dynamic lookup |
| `getSparseIndicesDynamicAgain()` | Refresh dynamic lookup |
| `compute(local_gradient=None) -> hessian` | Assemble all active numeric terms |

Important properties: `wrt`, `local_targets`, `dynamic_instances`, `gradient`, `diagonal`, `diagonal_blocks`, `diagonal_blocks_inverse`, `diagonal_blocks_start`, `diagonal_blocks_start_cpu`, `diagonal_blocks_local_sizes`, `gradient_segments_start`, `gradient_segments_start_cpu`, and `hash`.

The following property families are generator metadata: `sources`, `global_gradients`, `global_hessians`, `global_jacobians`, `global_inner_hessians`, `project_entire_hessian`, `projection_methods`, `separate_hessian_jacobian`, `intermediate_compute_pairs`, `merged_hessian_and_gradient_attributes`, `hessian_and_gradient_kernels`, `indices_kernels`, `block_indices_gpu`, `placement_reorder_kernels`, and each corresponding `_dynamic` property. Separate-Jacobian nonzero-position metadata is exposed through the `global_jacobian_*` properties.

## `gradient`

```python
gradient(wrt, hessian=None)
```

Extends `vector`. `compute()` delegates assembly to its parent Hessian. Properties: `hessian`, `wrt`, `gradient_segments_start`, `gradient_segments_start_cpu`, `gradient_segments`, `wrt_start_indices`, and `gradient_sizes`.

## `solver`

```python
solver()
```

| Member | Description |
| --- | --- |
| `computeSolution(active_hessian, wrt, gradient_object, initial_guess, tolerance=1e-3, maxIterations=20000)` | Run block-preconditioned CG and return a status code |
| `reset()` | Discard kernel association and work buffers |
| `solution` | Flattened solution GPUArray |

The Hessian's inverse diagonal blocks must already be populated. The minimizer normally owns this lifecycle.

## `vector`

```python
vector(size)
```

Properties: `size`, `value`. Methods: `updateValue(new_value)` and `resize(new_size)`. Operators: vector `+`, `-`, unary `-`, and scalar `*`.

## `matrix`

```python
matrix(rows=0, cols=0)
```

Properties: `rows`, `cols`, plus static `block_dimensions`, `blocks_flattened`, `blocks_start_indices`, `block_positions`, `block_counts` and their `_dynamic` counterparts. Compatibility aliases are `blockDimensions`, `blocksFlattened`, `blocksStartIndices`, `blockPositions`, and `blockCounts`.

`setDimensions(rows, cols)` updates logical dimensions. `matVecProduct`, `matVecProductInPlace`, and therefore `matrix * vector` are placeholders rather than working sparse multiplication.

## `energy` legacy class

```python
energy(
    energy,
    targets=[],
    projection_method=1,
    save_intermediate=False,
    gradient_only=False,
    separate_hessian_jacobian=False,
    dynamic_instances=False,
)
```

This class contains an older parallel path for derivative generation, sparse indices, and `computeHessianAndGradient`. The current `scene` uses `minimizer` plus `differentiator` instead. It remains exported for compatibility and research comparison; new integrations should not mix its internal buffers with a current minimizer.

Exposed inspection/assembly methods include `getSparseIndices`, `getSparseIndicesAgain`, `computeIndices`, `generateHessianAndGradient`, and `computeHessianAndGradient`. Properties include `energy`, `indices`, `block_indices_gpu`, `gradient_only`, sparse coordinate outputs, and `hash`.

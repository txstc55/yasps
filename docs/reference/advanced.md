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

The constructor has no parameters. It initializes empty request and target
registries plus a reusable solver.

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

### Minimizer parameters

| Call | Parameters |
| --- | --- |
| `addEnergy(...)` | `e`: named scalar energy; `targets=[]`: local target subset; `projection_method=1`: local SPD policy; `save_intermediate=False`: derivative-reuse hint; `gradient_only=False`: unsupported when true; `dynamic_instances=False`: dynamic structural path; `separate_hessian_jacobian=False`: split inner Hessian and outer Jacobian stages. |
| `addEnergies(energies)` | `energies`: list registered one by one with default options. |
| `addWrt(wrt)` | `wrt`: ordered unique static `DATA` targets. |
| `computeSolution(tolerance=1e-3, maxIterations=20000)` | `tolerance`: PCG residual tolerance; `maxIterations`: iteration cap. It assembles before solving. |
| `computeHessianAndGradient(tolerance=1e-3, maxIterations=20000)` | Same solver parameters; despite its name, it assembles and solves, returning status. |
| `ignoreEnergies(energies)` | `energies`: complete replacement ignore list; `[]` restores every request. |

`computeNumericValue()` is the assembly-only method. It also computes inverse
diagonal blocks and returns the active Hessian. There is no minimizer method
that solves this already assembled value without reassembly; use `solver`
directly for that case.

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

| Parameter | Effect |
| --- | --- |
| `source` | Scalar attribute or list of scalar attributes to differentiate and combine. |
| `target1` | Ordered global row targets. |
| `target2` | Ordered global column targets; currently must match `target1`. |
| `local_targets=[]` | Optional path-discovery subset. |
| `projection_method=1` | Local Hessian projection: `-1` none, `0` no-op, `1` absolute eigenvalues, `2` clamp negative values. |
| `save_intermediate=False` | Allows selected derivative intermediates to be retained. |
| `separate_hessian_jacobian=False` | Separates inner energy-Hessian and outer Jacobian generation. |
| `dynamic_instances=False` | Generates dynamic sparse-index metadata. |

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

| Parameter | Effect |
| --- | --- |
| `global_targets` | Ordered global differentiation targets. |
| `local_targets=[]` | Optional effective subset for the current energy. |

`getRoots(attribute, parent_path)` discovers valid target routes. `getPathDict()` finalizes dictionaries consumed by index generation and `wrt` exposes the effective target subset. This is generator-facing API; inspect [How YASPS executes]({{ '/architecture/' | relative_url }}?v={{ site.time | date: '%s' }}) before using it directly.

## `hessian`

```python
hessian(wrt, local_targets=[], dynamic_instances=False)
```

| Constructor parameter | Effect |
| --- | --- |
| `wrt` | Nonempty ordered static target list defining global rows, columns, and block sizes. |
| `local_targets=[]` | Optional local subset associated with the generated terms. |
| `dynamic_instances=False` | Chooses dynamic rather than static term storage. |

| Member | Description |
| --- | --- |
| `H0 + H1` | Combine terms with the same global targets |
| `getSparseIndices()` | Build static coordinate lookup |
| `getSparseIndicesDynamic()` | Build initial dynamic lookup |
| `getSparseIndicesDynamicAgain()` | Refresh dynamic lookup |
| `compute(local_gradient=None)` | Assemble all active numeric terms; optionally write the gradient into a compatible supplied object |

Important properties: `wrt`, `local_targets`, `dynamic_instances`, `gradient`, `diagonal`, `diagonal_blocks`, `diagonal_blocks_inverse`, `diagonal_blocks_start`, `diagonal_blocks_start_cpu`, `diagonal_blocks_local_sizes`, `gradient_segments_start`, `gradient_segments_start_cpu`, and `hash`.

The following property families are generator metadata: `sources`, `global_gradients`, `global_hessians`, `global_jacobians`, `global_inner_hessians`, `project_entire_hessian`, `projection_methods`, `separate_hessian_jacobian`, `intermediate_compute_pairs`, `merged_hessian_and_gradient_attributes`, `hessian_and_gradient_kernels`, `indices_kernels`, `block_indices_gpu`, `placement_reorder_kernels`, and each corresponding `_dynamic` property. Separate-Jacobian nonzero-position metadata is exposed through the `global_jacobian_*` properties.

## `gradient`

```python
gradient(wrt, hessian=None)
```

| Constructor parameter | Effect |
| --- | --- |
| `wrt` | Ordered target list defining flattened segment sizes. |
| `hessian=None` | Optional parent; `compute()` delegates to it when present. |

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

### Solver parameters

| Parameter | Effect |
| --- | --- |
| `active_hessian` | Current assembled static/dynamic blocks, coordinates, diagonal, and inverse diagonal blocks. |
| `wrt` | Ordered target list used for instance counts and block sizes. |
| `gradient_object` | Right-hand side and segment-offset provider. |
| `initial_guess` | Flattened `GPUArray` with the exact gradient length. |
| `tolerance=1e-3` | Residual tolerance. |
| `maxIterations=20000` | Iteration cap. |

The return is a status code; negative values denote nonconvergence. The best
available flattened result remains in `solution`.

## `vector`

```python
vector(size)
```

`size` is a nonnegative flattened element count. The constructor allocates a
zero buffer. `updateValue(new_value)` accepts a same-size NumPy array,
`GPUArray`, or `vector`. `resize(new_size)` changes size metadata but does not
reallocate storage.

Properties: `size`, `value`. Methods: `updateValue(new_value)` and `resize(new_size)`. Operators: vector `+`, `-`, unary `-`, and scalar `*`.

## `matrix`

```python
matrix(rows=0, cols=0)
```

`rows` and `cols` set logical dimensions only. The constructor does not create
a populated general sparse matrix.

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

| Constructor parameter | Effect |
| --- | --- |
| `energy` | Named scalar legacy energy expression. |
| `targets=[]` | Local differentiation targets. |
| `projection_method=1` | Local SPD projection policy. |
| `save_intermediate=False` | Derivative-reuse hint. |
| `gradient_only=False` | Legacy gradient-only request mode. |
| `separate_hessian_jacobian=False` | Split derivative-generation stages. |
| `dynamic_instances=False` | Dynamic sparse-structure path. |

This class contains an older parallel path for derivative generation, sparse indices, and `computeHessianAndGradient`. The current `scene` uses `minimizer` plus `differentiator` instead. It remains exported for compatibility and research comparison; new integrations should not mix its internal buffers with a current minimizer.

Exposed inspection/assembly methods include `getSparseIndices`, `getSparseIndicesAgain`, `computeIndices`, `generateHessianAndGradient`, and `computeHessianAndGradient`. Properties include `energy`, `indices`, `block_indices_gpu`, `gradient_only`, sparse coordinate outputs, and `hash`.

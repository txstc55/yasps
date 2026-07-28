---
title: Low-level exports
description: A map of code-generation, CUDA-kernel, context, helper, and operator symbols exported by YASPS.
permalink: /reference/internal/
---

<p class="eyebrow">Reference</p>

# Low-level exports

YASPS exports several implementation classes from `yasps.__init__` because its compiled subsystems compose through Python-visible objects. They are useful when extending the generator, but their constructor protocols and generated-buffer layouts are less stable than the scene and minimizer APIs.

## Symbolic operator layer

### `operator`

```python
operator(name, operator_type, commutative)
```

Properties are `name`, `type`, and `commutative`; `copy()` returns an equivalent operator. Type values mean unary function (`0`), infix binary (`1`), function syntax (`2`), or special handling (`3`).

The `attribute` module defines operator singletons including `ADD`, `SUB`, `MUL`, `DIV`, `POW`, `ATAN2`, `NEG`, `SIN`, `COS`, `ABS`, `LOG`, `SELECT`, `SQRT`, comparisons, `DATA`, `CONSTANT`, `ASCONSTANT`, `ARRAY`, `JOIN`, `SUM`, `AVERAGE`, `UNION`, `SPD`, and matrix/indexing operators. They are implementation details rather than package-root imports.

### Wildcard helper exports

The package root re-exports public names from three modules:

- `helper`: `extract_block`, `prune_duplicate_functions`, `timed`, and `DEBUG_TIME`;
- `attributeHelper`: `hashAttribute`, `attribute2str`, and `checkHeritage`;
- `attributeOperations`: `add`, `add_explicitly`, `sub`, `sub_explicitly`, `mul`, `mul_explicitly`, `div`, `div_explicitly`, `pow_op`, `sqrt_op`, `sin_op`, `cos_op`, and `log_op`.

Prefer `attribute` methods in application code. These functions are primarily derivative/code-generation building blocks.

## Expression code generation

### `codeGenerator`

```python
codeGenerator(input_attribute)
```

`generateCode()` traverses the symbolic graph, creates dependency order, assigns intermediate names, and emits Eigen/CUDA expressions for every supported operator. `getIntermediateName(attribute)` exposes the chosen local name.

### `deviceKernel`

Represents a generated device function plus its data, connectivity, primitive-union, dependency, and EVD requirements. Important properties are `kernelString`, `kernelHeader`, `kernelDatas`, `kernelConnectivity`, `kernelPrimitiveUnions`, `dependents`, `allEvdSizes`, and `attributeName`.

### `globalKernel`

Wraps an `attribute` in a global CUDA launch. `compute(output)` launches into a supplied GPU buffer; `kernelString` and `kernel` expose the generated source and loaded callable.

## Sparse-index kernels

### `gradientIndicesKernel`

Consumes the topology path dictionaries and emits local target indices, block coordinates, block dimensions, permutation metadata, and compressed groupings. `computeIndices(wrt_start_indices)` populates its output buffers.

### `coordinateCompressionKernel`

Merges coordinate streams from energy terms. `compressCoordinatesAndDimensions()` produces unique coordinates/dimensions and per-term `lookupArrays`; `updateCoordinates(...)` replaces dynamic inputs.

### `placementReorderKernel`

Reorders lookup placements used by the separated Hessian/Jacobian path. Call `generateKernel(...)` before `reorderPlacementIndices(...)`; inspect `reordered_lookups` afterward.

## Hessian assembly kernels

### `hessianAndGradientKernel`

Builds and launches the fused numerical assembly kernel for one term. `generateKernel(...)`, `compute(...)`, and `kernelString` are its primary surface.

The following classes provide source fragments selected by that wrapper:

- `hessianKernelHeader` — shared includes, dependency functions, and declarations;
- `hessianKernelHost` — generated host-side launch code;
- `hessianKernelFullProject` — full local Hessian projection variant;
- `hessianKernelNoProject` — non-full-projection variant;
- `hessianKernelSeparateJacobian` — separated local Hessian/Jacobian generation and stored multiplied blocks.

These fragment classes expose `kernelString`; `hessianKernelSeparateJacobian` also exposes `stored_multiplied_blocks` and `dependents`.

## Solver kernels

### `diagonalBlockInverseKernel`

Generates inverse routines for the set of target block sizes. `computeDiagonalBlockInverse(diagonal_blocks, diagonal_blocks_inverse)` fills the PCG preconditioner.

### `solverKernel`

Owns generated block-sparse PCG code. `updateBlockDimensions(...)` refreshes supported static/dynamic categories and `computeSolution(...)` operates on the complete matrix-buffer protocol.

Use the higher-level `solver` unless you are changing this protocol.

## CUDA `context`

```python
context()
```

`useDefaultContext()` returns to the context that was active when the class initialized. `useNamedContext(name)` creates or activates a named context on CUDA device 0.

This helper manipulates the PyCUDA context stack globally. Do not switch contexts while GPUArray objects or loaded modules from another context are in use.

## Package-root export inventory

For completeness, `yasps.__init__` exposes:

```text
scene, mesh, primitive, primitiveUnion, connectivity, attribute, operator,
deviceKernel, codeGenerator, globalKernel, gradientIndicesKernel,
hessianAndGradientKernel, coordinateCompressionKernel,
diagonalBlockInverseKernel, solverKernel, solver, vector, matrix, gradient,
hessian, energy, minimizer, autodiff, path, differentiator, context,
hessianKernelHost, hessianKernelHeader, hessianKernelFullProject,
hessianKernelNoProject, placementReorderKernel,
hessianKernelSeparateJacobian
```

The [core reference]({{ '/reference/core/' | relative_url }}?v={{ site.time | date: '%s' }}) and [advanced reference]({{ '/reference/advanced/' | relative_url }}?v={{ site.time | date: '%s' }}) document the application-facing members from that inventory.

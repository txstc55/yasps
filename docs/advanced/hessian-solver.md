---
title: Hessian and solver
description: Inspect YASPS block-sparse Hessians, gradient layouts, vectors, matrices, and the PCG solver.
permalink: /advanced/hessian-solver/
---

<p class="eyebrow">Advanced API</p>

# Hessian and solver

The public `hessian`, `gradient`, `matrix`, `vector`, and `solver` classes expose the numerical structures used by the minimizer. They are useful for inspection and experiments, but they form an internal block-sparse protocol rather than a general-purpose linear-algebra library.

## Obtain a populated Hessian

The meaningful way to obtain one is through differentiation:

```python
from yasps import differentiator

H = differentiator().diff2(
  [energy],
  targets,
  targets,
  projection_method=2,
)
H.compute()
```

The public constructor:

```python
from yasps import hessian
H = hessian(targets)
```

creates only an empty symbolic container with the correct global dimensions. It has no energy terms or index kernels until generator metadata is attached, so constructing one directly is rarely useful.

Hessians with the same global targets may be combined:

```python
H_total = H_elastic + H_contact
H_total.compute()
```

Addition concatenates their static and dynamic symbolic terms. Target count, ordering, and hashes must match.

## Block-sparse representation

YASPS groups blocks by dimensions. Static and dynamic energy terms use parallel storage:

| Property | Contents |
| --- | --- |
| `block_dimensions` | Flattened `(rows, cols)` pairs for static block categories |
| `blocks_start_indices` | Start of each category in flattened numerical storage |
| `block_counts` | Number of blocks in each category |
| `block_positions` | Global coordinates of static blocks |
| `blocks_flattened` | Static block values |
| `*_dynamic` counterparts | The same fields for runtime-changing terms |

The base `matrix` also exposes camelCase aliases such as `blockDimensions` and `blocksFlattened` for compatibility.

Hessian-specific buffers include:

- `diagonal` — scalar diagonal accumulation used by solver kernels;
- `diagonal_blocks` — un-inverted dense diagonal blocks;
- `diagonal_blocks_inverse` — preconditioner blocks, populated by the minimizer;
- `diagonal_blocks_start` and `diagonal_blocks_start_cpu` — target-block offsets;
- `gradient` — the `gradient` object filled during assembly.

Properties such as `global_hessians`, `global_jacobians`, `indices_kernels`, `block_indices_gpu`, `placement_reorder_kernels`, and their dynamic forms are code-generator metadata. They are public because the Cython subsystems exchange them, but applications should treat their exact structure as unstable.

## Gradient layout

```python
from yasps import gradient

g = gradient(targets, H)
g.compute()

g.value                       # one flattened GPUArray
g.gradient_segments           # per-target GPUArray views
g.gradient_segments_start     # uint32 offsets on the GPU
g.gradient_segments_start_cpu # offsets as a Python list
g.gradient_sizes              # flattened length of each target
g.wrt                         # target attributes
```

`g.compute()` asks its parent Hessian to assemble into it. If no parent was supplied, the method returns without changing the zero vector.

## Direct solver call

The safest direct solver experiment starts from a minimizer because it prepares the diagonal-block inverse:

```python
import pycuda.gpuarray as gpuarray
from yasps import minimizer, solver

engine = minimizer()
engine.addEnergy(energy, projection_method=2)
engine.addWrt(targets)
H = engine.computeNumericValue()

initial_guess = gpuarray.zeros_like(engine.gradient)
pcg = solver()
status = pcg.computeSolution(
  H,
  engine.wrt,
  H.gradient,
  initial_guess,
  tolerance=1e-3,
  maxIterations=20000,
)

flat_direction = pcg.solution
```

`status < 0` indicates nonconvergence. The solution is one flattened buffer; use `H.gradient.gradient_segments_start_cpu` to construct target-aligned views.

`solver.reset()` discards its compiled solver-kernel association and all work buffers. Ordinary repeated solves reuse those allocations.

## `vector`

`vector(size)` owns a float64 PyCUDA buffer and supports:

```python
v.updateValue(numpy_array_or_gpuarray_or_vector)
w = v + other
w = v - other
w = -v
w = 2.0 * v
```

Operands must have equal size. Arithmetic allocates a new result vector.

## Base `matrix` warning

`matrix` is a storage base class. Its `matVecProduct` and `matVecProductInPlace` methods currently validate dimensions but do not implement the block-sparse multiplication. Consequently, `matrix * vector` does not perform the numerical operation an application might expect.

The actual Hessian-vector products used by PCG are implemented by generated `solverKernel` code operating directly on the block buffers. Do not use the base `matrix` as a standalone sparse-matrix API.

## Stability expectations

The scene and minimizer APIs are the supported composition layer. Direct inspection of numerical buffers is useful and intentional; mutating generator metadata or block layouts requires keeping the Hessian, coordinate compression, diagonal preconditioner, and solver protocols in sync.

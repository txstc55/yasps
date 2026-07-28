---
title: Hessian and solver
description: Inspect YASPS block-sparse Hessians, gradient layouts, vectors, matrices, and the PCG solver.
permalink: /advanced/hessian-solver/
---

<p class="eyebrow">Advanced API</p>

# Hessian and solver

The public `differentiator`, `hessian`, `gradient`, `matrix`, `vector`, and
`solver` classes expose the numerical protocol beneath the minimizer. This is a
generated block-sparse system specialized to YASPS attributes—not a general
sparse linear-algebra library.

## Generate a Hessian with `diff2`

The useful way to construct a populated Hessian is second-order
differentiation:

```python
from yasps import differentiator

H = differentiator().diff2(
  source=[elasticity, contact],
  target1=targets,
  target2=targets,
  local_targets=[],
  projection_method=2,
  save_intermediate=False,
  separate_hessian_jacobian=False,
  dynamic_instances=False,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `source` | scalar `attribute` or `list[attribute]` | One or more scalar per-instance energies. Multiple sources produce Hessians that are symbolically added. |
| `target1` | `list[attribute]` | Ordered global row targets. |
| `target2` | `list[attribute]` | Ordered global column targets. It must match `target1` exactly; mixed second-order Jacobians are not implemented. |
| `local_targets` | `list[attribute] = []` | Optional target subset used for path discovery for every source in this call. |
| `projection_method` | `int = 1` | Local SPD projection policy: `-1`, `0`, `1`, or `2`. |
| `save_intermediate` | `bool = False` | Allows generated derivative intermediates to be retained for reuse. |
| `separate_hessian_jacobian` | `bool = False` | Separates inner energy-Hessian and outer Jacobian stages. |
| `dynamic_instances` | `bool = False` | Generates the dynamic sparse-index path for changing source-instance counts. |

Every source must be scalar. Passing an empty source list raises `ValueError`.
`differentiator.diff1` is declared in the current package but not implemented;
use the gradient attached to a computed Hessian for first-order values.

## Direct `hessian` construction

```python
from yasps import hessian

H_empty = hessian(
  wrt=targets,
  local_targets=[],
  dynamic_instances=False,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `wrt` | `list[attribute]` | Ordered global targets and block layout. At least one target and a positive global size are required. |
| `local_targets` | `list[attribute] = []` | Optional local subset recorded with the container. |
| `dynamic_instances` | `bool = False` | Selects dynamic rather than static term storage. |

The constructor creates dimensions and empty storage only. It has no energy
terms, index kernels, or placement metadata until the differentiator supplies
them. Direct construction is therefore useful primarily for generator work,
not ordinary assembly.

Dynamic attributes are not valid global Hessian targets.

## Add symbolic Hessian terms

```python
H_total = H_elastic + H_contact
```

Addition concatenates static and dynamic symbolic term metadata. Both operands
must have the same target count, order, and attribute hashes. It does not add
the current numerical buffers on the host; call `compute()` on the combined
container to assemble its current values.

This is also how the minimizer creates an active Hessian from all nonignored
energy requests.

## Assemble a Hessian

```python
H.compute()
```

or, to direct gradient output into an existing compatible layout:

```python
H.compute(local_gradient=g)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `local_gradient` | `gradient | None = None` | Optional global gradient object that receives the assembled values. Without one, the Hessian owns or creates its gradient. |

The no-return method performs numerical setup and assembly:

1. on first use, generate static sparse indices and coordinate compression;
2. initialize dynamic sparse metadata when dynamic terms exist;
3. on later dynamic assemblies, refresh changing coordinates;
4. launch placement/reorder and numerical kernels;
5. accumulate Hessian block values, the scalar diagonal, dense diagonal blocks,
   and gradient values.

Calling `H.compute()` directly does **not** construct the inverse diagonal
blocks required by the stock PCG preconditioner. `minimizer.computeNumericValue`
adds that step. Use the minimizer to prepare a Hessian before a direct
`solver.computeSolution` call.

## Sparse-index lifecycle

The public methods below exist mainly for the generator protocol:

| Method | Role |
| --- | --- |
| `getSparseIndices()` | Builds static global block coordinates and compression metadata. |
| `getSparseIndicesDynamic()` | Initializes dynamic coordinate and placement structures. |
| `getSparseIndicesDynamicAgain()` | Refreshes dynamic coordinates after runtime topology changes. |

Ordinary applications should call `compute()` and let it select the correct
path. Calling these manually out of order can desynchronize positions, reorder
maps, block counts, and numerical buffers.

## Block-sparse representation

YASPS groups blocks by their `(rows, cols)` dimensions. Static and dynamic
energy terms use parallel storage:

| Property | Contents |
| --- | --- |
| `block_dimensions` | Flattened dimension pairs for static block categories. |
| `blocks_start_indices` | Start of each static category in flattened numerical storage. |
| `block_counts` | Number of static blocks in each category. |
| `block_positions` | Global row/column coordinates of static blocks. |
| `blocks_flattened` | Static numerical block entries. |
| `block_dimensions_dynamic` | Dimension pairs for dynamic categories. |
| `blocks_start_indices_dynamic` | Starts for dynamic numerical categories. |
| `block_counts_dynamic` | Counts for dynamic categories. |
| `block_positions_dynamic` | Current global coordinates of dynamic blocks. |
| `blocks_flattened_dynamic` | Dynamic numerical block entries. |

The `matrix` base also exposes camelCase aliases such as `blockDimensions` and
`blocksFlattened` for compatibility.

### Diagonal and preconditioner storage

| Property | Contents |
| --- | --- |
| `diagonal` | Scalar diagonal accumulation used by solver kernels. |
| `diagonal_blocks` | Current dense diagonal blocks before inversion. |
| `diagonal_blocks_inverse` | Dense inverse blocks used by the preconditioner. |
| `diagonal_blocks_start` | GPU offsets for the target-aligned diagonal blocks. |
| `diagonal_blocks_start_cpu` | The same offsets as a Python list. |
| `gradient` | The associated assembled `gradient` object. |

The minimizer builds an inversion kernel specialized to the distinct target
element sizes, target instance counts, and block offsets. It reuses that kernel
while the target layout stays stable and recomputes the inverse values after
each numerical assembly.

### Generator metadata

Properties including `global_hessians`, `global_jacobians`, `indices_kernels`,
`block_indices_gpu`, `placement_reorder_kernels`, and their dynamic forms are
public because the Cython subsystems exchange them. Their shapes are part of
the code-generation implementation rather than a stable application data
model.

## Construct and inspect a gradient

```python
from yasps import gradient

g = gradient(
  wrt=targets,
  hessian=H,
)
g.compute()
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `wrt` | `list[attribute]` | Ordered target list used to allocate the flattened vector and segment views. |
| `hessian` | `hessian | None = None` | Optional parent whose `compute` method will assemble into this gradient. |

`g.compute()` takes no parameters and returns `None`. With a parent Hessian it
calls `H.compute(g)`; without one it leaves the initialized zero vector
unchanged.

| Property | Contents |
| --- | --- |
| `value` | One flattened `GPUArray`. |
| `gradient_segments` | One flattened view per target. |
| `gradient_segments_start` | Segment offsets in a GPU uint32 buffer. |
| `gradient_segments_start_cpu` | Segment offsets as a Python list. |
| `gradient_sizes` | Flattened element count for every target. |
| `wrt` | Ordered target list. |

Segment sizes are `numInstances × rows × cols`. Segments alias `value`; later
assemblies refill the same storage.

## Prepare a direct solver call

The safest source of a direct-solve Hessian is a minimizer, because it assembles
both numerical values and inverse diagonal blocks:

```python
import pycuda.gpuarray as gpuarray
from yasps import minimizer, solver

engine = minimizer()
engine.addEnergy(
  energy,
  targets=targets,
  projection_method=2,
)
engine.addWrt(targets)

H = engine.computeNumericValue()
initial_guess = gpuarray.zeros_like(engine.gradient)
```

If `H` is `None`, every energy is ignored and there is no active system to
solve.

## Call the PCG solver

```python
pcg = solver()
status = pcg.computeSolution(
  active_hessian=H,
  wrt=engine.wrt,
  gradient_object=H.gradient,
  initial_guess=initial_guess,
  tolerance=1e-3,
  maxIterations=20000,
)

flat_direction = pcg.solution
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `active_hessian` | assembled `hessian` | Supplies static/dynamic blocks, coordinates, scalar diagonal, and inverse diagonal blocks. |
| `wrt` | `list[attribute]` | Ordered targets used to derive instance counts and per-instance block sizes. |
| `gradient_object` | `gradient` | Supplies the right-hand side, target offsets, and flattened length. |
| `initial_guess` | `GPUArray` | Flattened initial direction with exactly the gradient layout. |
| `tolerance` | `float = 1e-3` | Residual tolerance passed to the generated PCG kernel. |
| `maxIterations` | `int = 20000` | Iteration cap. |

The solver owns reusable work vectors and one generated `solverKernel`
association. On each call it clears the work vectors and its solution, updates
the block-dimension list, then invokes the kernel.

`status < 0` means the solve did not converge. Exact negative codes are not a
documented API. `pcg.solution` still contains the best result produced.

The solution is a single flattened buffer. Use
`H.gradient.gradient_segments_start_cpu`, or the known target segment sizes, to
construct target-aligned views.

## Reset solver state

```python
pcg.reset()
```

This no-argument method returns `None`. It discards the compiled solver-kernel
association, all PCG work buffers, and the solution buffer. Repeated solves
with a stable layout should not call `reset`; they reuse those allocations.

Reset after changing to an incompatible target or block-dimension layout.
`minimizer.addWrt` resets its owned solver automatically.

## Empty-system behavior

If the gradient has zero elements, `computeSolution` returns `0`. If no active
Hessian is supplied, it returns `0` and leaves or clears the solution as
appropriate. A normal nonempty solve requires a non-`None` Hessian.

## The `vector` class

```python
from yasps import vector

v = vector(size=number_of_scalars)
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `size` | nonnegative `int` | Allocates a zero-filled PyCUDA buffer of that length. |

The current CUDA implementation stores float64 values.

```python
v.updateValue(numpy_array_or_gpuarray_or_vector)
w = v + other
w = v - other
w = -v
w = 2.0 * v
```

`updateValue(new_value)` requires exactly `size` elements. NumPy and non-float64
GPU inputs are converted to float64. Binary operands must have equal sizes, and
arithmetic creates a new result vector.

`resize(new_size)` updates the recorded size only; it does not reallocate the
underlying GPU buffer. Treat it as low-level metadata, not a safe general
resize operation.

## The base `matrix` class

```python
from yasps import matrix

M = matrix(
  rows=0,
  cols=0,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `rows` | `int = 0` | Recorded row count. |
| `cols` | `int = 0` | Recorded column count. |

`matrix` is a storage base. Its `matVecProduct` and `matVecProductInPlace`
methods currently validate dimensions but do not implement block-sparse
multiplication. Consequently, `matrix * vector` is not a usable numerical
Hessian-vector product.

PCG multiplication is generated by `solverKernel` and operates directly on the
Hessian's static and dynamic block buffers.

## Safe inspection versus mutation

Reading numerical blocks, gradients, diagonals, and segment offsets is useful
and intentional. Mutating block positions, counts, reorder metadata, or
generator kernel lists requires updating every related component:

```text
sparse coordinates
  → compression and placement maps
  → numerical block ordering
  → diagonal-block preconditioner
  → solver block-dimension protocol
```

For custom numerical experiments, preserve those structures and replace only
the solve that consumes an assembled minimizer Hessian.

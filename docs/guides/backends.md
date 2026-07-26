# CUDA and Metal backends

> This page is intentionally excluded from the published documentation while
> the Metal backend remains under validation.

YASPS exposes one symbolic scene interface and generates backend-specific
GPU programs from the same attribute and derivative graphs. Metal is not an
eager graph evaluator: it follows the CUDA architecture by specializing
source for computations, derivative indices, Hessian/gradient assembly,
matrix projection, sparse solving, and collision detection.

## Backend selection

At import, `yasps.backend`:

1. honors `YASPS_BACKEND=cuda|metal`;
2. otherwise selects usable CUDA/PyCUDA first;
3. otherwise selects MLX Metal on Apple silicon; and
4. raises if neither backend is available.

```python
from yasps.backend import backend_info, backend_name, is_cuda, is_metal
```

`backend_info()` reports the selected backend, platform, device, and primary
real dtype.

## Generated paths

| Operation | CUDA implementation | Metal implementation |
| --- | --- | --- |
| Attribute computation | `codeGenerator.pyx` + CUDA/Eigen | `codeGeneratorMetal.pyx` + generated MSL |
| Raw derivative indices | `gradientIndicesKernel.pyx` | `gradientIndicesKernelMetal.py` |
| Coordinate compression | `coordinateCompressionKernel.pyx` | `coordinateCompressionKernelMetal.py` |
| Placement reordering | `placementReorderKernel.pyx` | generated `MetalPlacementReorder` kernel |
| Hessian/gradient values | `hessianAndGradientKernel.pyx` | `hessianAndGradientKernelMetal.py` |
| Block inverse | `diagonalBlockInverseKernel.pyx` | `diagonalBlockInverseKernelMetal.py` |
| PCG | generated CUDA/C++ | generated MSL driven by a compiled C++ extension |
| CCD | generated CUDA/GIPC helper code | generated MSL in `ccdMetal.py` |

The existing Cython classes remain the dispatch points. Metal counterparts
live beside their CUDA equivalents under the same logical `kernel` and
`codeGenerator` directories.

## JIT module reuse

`MetalProgram` recursively compiles named attributes, JOIN/UNION boundaries,
and requested roots into reusable MSL helper modules. A module records:

- its generated function;
- the data/connectivity/union resources it consumes;
- dependencies on other generated modules; and
- a structural cache key.

Root kernels include the dependency modules in topological order. Generated
source is persisted under `.yasps_tmp/metal/` for inspection, and MLX JIT
compiles the resulting specialization. Repeated subexpressions and named
attributes reuse cached modules within the process.

The Metal API exposed by MLX does not provide CUDA-style relocatable object
linking, so modular reuse occurs at generated MSL function/source level
rather than by linking `.o` files.

## Precision

| Backend | Primary real dtype |
| --- | --- |
| CUDA | float64 |
| Metal | float32 |

The Metal facade maps incoming float64 numerical arrays and float64 allocation
requests to float32. Integer connectivity retains its integer dtype.

Float16 is not used for core simulation. It cannot safely represent common
IPC distances around \(10^{-6}\), and it is inadequate for determinant,
Jacobi EVD, sparse accumulation, and PCG residual calculations.

## Metal linear algebra

`yasps/yasps/kernel/metalLinalg.metal` supplies generated kernels with:

- statically sized determinant and inverse operations;
- cyclic symmetric Jacobi eigendecomposition;
- Hessian eigenvalue projection;
- small-block spectral inversion; and
- helpers shared by generated attribute and Hessian programs.

Large batches of 12-32 dimensional absolute-eigenvalue projections may use a
threadgroup Jacobi kernel. Smaller or incompatible terms retain the
thread-local generated path.

## Sparse solve

The Metal solver keeps the sparse block matrix, block-Jacobi preconditioner,
vectors, reductions, and PCG scalar recurrence on Metal. Python specializes
and caches the generated MSL program. The compiled extension under
`kernel/Solver/metalExtension/` owns the host control loop and dispatches
32-iteration GPU recurrence chunks, avoiding a Python dispatch per PCG
iteration.

## CCD

`examples/ccd/ccd.py` selects `ccdMetal.py` on Metal. Generated kernels build
a hashed uniform grid, emit face and edge candidates, classify discrete
features, and compute additive continuous-collision step sizes.

The default query uses a capacity-aware atomic append buffer and reuses
separate capacities for face/edge and discrete/continuous work. Set
`YASPS_METAL_CCD_APPEND=0` to retain the deterministic count-scan-write
fallback. No candidate is silently dropped: overflow grows the buffer or
raises at the configured maximum.

## Device arrays and synchronization

`yasps.backend.gpuarray` implements the subset of PyCUDA `GPUArray` used by
YASPS. On Metal, numerical arithmetic stays in MLX arrays. `.get()` and
`.item()` are explicit host boundaries; `.gpudata` and `.ptr` raise because
raw CUDA pointers have no Metal equivalent.

Use:

```python
from yasps.backend import synchronize

synchronize()
```

only for timing, diagnostics, or external consumers. Keep full-array `.get()`
calls outside numerical inner loops.

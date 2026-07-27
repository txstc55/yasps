# CUDA/Metal Separation Audit

Date: 2026-07-27

## Scope

The audit compared `codex/metal` with the pre-port merge base:

`60c02090476d6a6a0e716e72ce1bd1aabda1ae53`

It covered every modified pre-existing runtime/code-generation file, with
particular attention to:

- generated CUDA/C++ source;
- PCG convergence and failure behavior;
- sparse-index and coordinate batching;
- buffer allocation and reset ordering;
- Hessian/Jacobian code generation;
- CUDA package installation; and
- the evaluation examples' CUDA defaults.

New Metal-only modules, shaders, tests, and recorded evaluation artifacts
cannot execute on the CUDA backend and were excluded from CUDA semantic
comparison.

## Findings and corrections

### Generated CUDA PCG had changed

Commit `3e1fcca` changed the CUDA solver from the original RHS-relative
preconditioned tolerance to an initial-residual-relative tolerance. It also
added an identity-preconditioner fallback, changed the non-SPD check from
`h_alpha < 0` to `h_alpha <= 0`, added non-finite exit paths, and changed the
compiled-kernel cache key.

Those changes have been removed from the CUDA branch. CUDA again uses the
original:

- `delta0 = b * P^-1 * b`;
- `relativeTolerance = threshold * delta0`;
- block-Jacobi preconditioning on every iteration;
- negative-only non-SPD test;
- return codes and resource cleanup;
- generated-kernel cache name and JSON schema; and
- `d_p1_b` allocation/reset/use.

The optimized Metal PCG remains in the `is_metal()` branch.

### A Metal Eigen workaround leaked into CUDA source

Separate-Jacobian generation had changed CUDA `Eigen::Map<Matrix<...>>` to
`Eigen::Map<const Matrix<...>>`. The qualifier is now selected only for Metal;
the CUDA string is the original text.

### Metal scratch work leaked into CUDA object paths

Metal solver reductions, sparse scans, padded coordinate buffers, and padded
gradient-index buffers were being allocated or sized during CUDA object
construction/reallocation. They are now Metal-only.

CUDA retains the original:

- solver buffer clear list and order;
- Hessian buffer clear list and order;
- coordinate-compression clear list and order;
- gradient-index buffer capacities;
- gradient-index clear list and order; and
- one-dispatch-at-a-time PyCUDA calls.

Metal keeps its batched clears, padded sort/scan buffers, and dispatch batches.

### Backend and build defaults were not fully separated

- `real_dtype` is now `float64` on CUDA and `float32` on Metal.
- PyCUDA remains an automatic dependency on non-macOS platforms.
- Cython keeps the original 16 workers off macOS; macOS uses serial
  Cythonization because multiprocessing spawn recursively executes
  `setup.py`.
- Metal translation modules are imported only inside Metal code-generation
  branches.

### Example preprocessing had changed on CUDA

The container examples had replaced the original
`np.vstack(..., dtype=np.uint32)` path globally. That replacement is now used
only on Metal; CUDA retains the original preprocessing call. Frame counts,
solver tolerances, projection methods, and unlimited inner-loop defaults also
remain at their original CUDA values unless the user explicitly supplies an
evaluation environment override.

## Verification

Two sensitive generated CUDA source regions are byte-identical to the merge
base:

| Generated CUDA source | SHA-256 |
|---|---|
| PCG solver/CUDA wrapper | `d7fd72959a5c39dcd89b987e174f221c40a89fff9092ad8b2c8a2a361e9717e3` |
| Gradient-index generator | `76f523e3425fa28f0451428a0acb19263c7a05d2269cc124b1d5e794d3b453c5` |

Permanent regression tests in `tests/test_cuda_separation.py` enforce those
snapshots and the backend-specific dtype, reset, allocation, and Eigen
branches.

Verification results:

- Cython/native extension build: passed.
- Metal plus separation tests: `28 passed`.
- `git diff --check`: passed.

This Apple Silicon host has neither `nvcc` nor PyCUDA, so it cannot execute a
CUDA kernel. CUDA verification here is therefore source-level and
control-flow-level, including exact generated-source comparison. A CUDA host
should still run the existing numerical examples before merge as the final
hardware confirmation.

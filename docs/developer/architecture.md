# Generated backend architecture

> This developer page is intentionally excluded from the published
> documentation while Metal validation is in progress.

YASPS keeps one backend-independent symbolic frontend and two generated GPU
backends. Python constructs expression and topology graphs; it does not
interpret every operation numerically.

## Pipeline

```text
scene / mesh / primitive / attribute / connectivity / union
                              |
                              v
                    symbolic attribute DAG
                              |
                symbolic first/second derivatives
                              |
              generated raw sparse block coordinates
                              |
                  GPU coordinate compression
                              |
        generated local gradient/Hessian evaluation + assembly
                              |
            generated block inverse + sparse PCG solve
```

CUDA specializes CUDA/C++ with Eigen. Metal specializes MSL, allocates its
buffers through MLX, and uses a small compiled C++ extension where an eager
host dispatch loop materially reduces overhead.

## Dispatch remains in the old structure

The existing classes remain authoritative:

- `attribute/attribute.pyx` dispatches root computation;
- `kernel/Coordinate/gradientIndicesKernel.pyx` dispatches raw derivative
  indices;
- `kernel/Coordinate/coordinateCompressionKernel.pyx` dispatches compression;
- `kernel/Coordinate/placementReorderKernel.pyx` dispatches lookup reorder;
- `kernel/Hessian/hessianAndGradientKernel.pyx` dispatches numerical assembly;
- `kernel/Solver/diagonalBlockInverseKernel.pyx` dispatches block inversion;
- `kernel/Solver/solverKernel.pyx` dispatches the sparse solve; and
- `examples/ccd/ccd.py` dispatches collision detection.

Their Metal implementations sit beside them as
`codeGeneratorMetal.pyx`, `*KernelMetal.py`, `ccdMetal.py`, and
`metalLinalg.metal`. The CUDA paths are preserved.

## Generated attribute modules

`MetalProgram` traverses an attribute root and emits one MSL helper function
for each reusable boundary. Reusable boundaries include:

- the requested root;
- named computed attributes;
- JOIN and UNION nodes; and
- staged large SPD projections.

Each module records only the data, connectivity, and union-prefix resources
it consumes. Dependencies are emitted in topological order and cached by a
structural key. This mirrors CUDA's modular compilation intent, although MLX
accepts source rather than relocatable Metal object files.

Generated sources are written to `.yasps_tmp/metal/` with stable hashes.
Those files are diagnostic cache artifacts and are ignored by Git.

## Index generation

`MetalIndexPipeline` generates MSL for the derivative graph's raw global
indices and local sizes. Generated follow-up kernels:

1. compress duplicate local target indices;
2. build gradient-size histograms;
3. compact and group instances;
4. emit sparse upper-triangular block coordinates; and
5. route primitive-union children through prefix counts.

`MetalCoordinateCompressor` then sorts and deduplicates global coordinates,
builds lookup tables, and groups block dimensions. Hierarchical prefix scans
stay on Metal; only compact sizes/totals needed for specialization cross to
the host.

## Hessian and gradient assembly

Symbolic differentiation still produces local gradient and Hessian
expressions. `MetalHessianProgram` reuses `MetalProgram` modules, specializes
one MSL assembly kernel for each layout, and atomically accumulates:

- global gradient entries;
- sparse Hessian blocks;
- scalar diagonal values; and
- block-diagonal values for the preconditioner.

Named intermediate modules prevent repeated source expansion. Large repeated
12-32 dimensional absolute-value SPD projections can be materialized by a
threadgroup Jacobi kernel and consumed as a packed generated resource.

## Linear algebra

`kernel/metalLinalg.metal` provides statically sized determinant, inverse,
cyclic symmetric Jacobi EVD, eigenvalue projection, and block inverse helpers.
The Metal backend uses float32 because Apple GPU shaders do not provide
general float64 arithmetic.

Projection methods retain CUDA semantics:

| Method | Eigenvalue treatment |
| ---: | --- |
| `-1` | Skip projection |
| `0` | Keep unchanged |
| `1` | Replace by absolute value |
| `2` | Clamp below zero |

## PCG

`solverKernelMetal.py` generates and caches MSL specialized to the active
block dimensions:

- symmetric block-sparse SpMV;
- block-Jacobi application;
- compensated reductions;
- vector initialization/update; and
- a GPU-resident alpha/beta/status recurrence.

`kernel/Solver/metalExtension/` compiles a Darwin arm64 extension that owns
the eager PCG host loop. It dispatches 32-iteration recurrence chunks, checks
status/residuals at chunk boundaries, and restores the best finite iterate at
the float32 residual floor. This replaces the earlier Python-per-iteration
prototype without moving sparse arithmetic off the GPU.

## CCD

The Metal counterpart in `examples/ccd/` follows the CUDA component boundary:
topology and device arrays enter `CCD`, generated kernels perform numerical
work, and only compact counts/step scalars drive host control flow.

The broad phase:

1. generates swept or static element AABBs;
2. chooses a uniform grid from scene extent and activation gap;
3. counts and scans cell references;
4. generates and sorts 64-bit cell keys; and
5. queries face and edge candidates with duplicate/shared-feature filtering.

The default query atomically appends into a reusable capacity buffer in one
candidate traversal. Overflow reports the required size and retries; it never
silently truncates. `YASPS_METAL_CCD_APPEND=0` retains a
count-scan-write reference path.

Generated narrow-phase code classifies PP, PE, PT, and EE pairs. Continuous
queries feed generated additive time-of-impact kernels and a GPU minimum
reduction.

## Adding an operator

An operator is complete only when it has:

1. a public graph construction method and shape rules;
2. symbolic first/second derivatives;
3. CUDA generation;
4. Metal MSL emission in `MetalProgram._emit`;
5. scalar/matrix/batched numerical tests; and
6. derivative tests where applicable.

Do not add an eager Metal-only evaluation fallback. Missing operations should
fail source generation clearly until both symbolic and generated numerical
semantics are implemented.

## Validation order

Use increasing scope:

1. generated attribute operators;
2. matrix inverse/determinant/SPD projection;
3. raw indices, JOIN/UNION routing, and compression;
4. sparse assembly against a dense reference;
5. block inverse and PCG residuals;
6. isolated CCD feature and swept-step cases;
7. small complete scenes;
8. ordinary versus separate-Jacobian scene comparison;
9. large mixed/contact scenes; and
10. PyVista rendering in a separate or explicitly controlled phase.

Keep interactive PyVista rendering disabled during performance runs on the
same Apple GPU. The exact scene wrappers expose compute-only environment
controls without changing their default material or solver settings.

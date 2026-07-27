# Metal low-level solver and Hessian optimization

This pass preserves preconditioned conjugate gradient, generated symbolic
Hessian expressions, 12×12 local stable-Neo-Hookean projection, sparse block
assembly, and float32 arithmetic. It changes only Metal execution and
lowering details.

## Controlled partial-ABD result

The controlled workload is one 960×540 partial-ABD frame, one nonlinear
solve, and a warm shader cache on an Apple M2 Max. Both versions converged in
22 CG iterations without solver errors.

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Dominant Hessian GPU time | 52.36 ms | 32.60 ms | −37.7% |
| Dominant Hessian dispatch wall time | 79.01 ms | 59.15 ms | −25.1% |
| Total solver time | 14.43 ms | 9.75 ms | −32.5% |
| Solver time per CG iteration | 0.656 ms | 0.443 ms | −32.5% |
| Iteration-batch wall time | 11.07 ms | 6.84 ms | −38.3% |

“After” is the median of three warm runs. The before trace is
`/tmp/yasps_lowlevel_before.6DoIpJ`; the accepted after traces are
`/tmp/yasps_lowlevel_directout2_r1.0aiIec`,
`/tmp/yasps_lowlevel_directout2_r2.uo9lD3`, and
`/tmp/yasps_lowlevel_directout2_r3.02YE3P`.

## Solver changes

PCG formerly submitted and synchronously waited for two command buffers per
iteration: denominator/SpMV, then solution/residual/preconditioner. The Metal
path now:

1. reduces the denominator;
2. conditionally computes alpha and updates solution/residual on GPU;
3. applies the existing block-Jacobi preconditioner;
4. reduces the new preconditioned residual;
5. waits once and performs the same host convergence/error checks.

All dispatches in the iteration use one compute encoder with explicit buffer
barriers. The runtime also caches per-pipeline argument encoders and argument
buffers; a per-pipeline mutex makes that reuse safe across callers.

The recurrence, relative tolerance, identity-preconditioner fallback, and
non-positive-denominator handling are unchanged.

## Hessian changes

Generated Metal wrappers previously zeroed a 49×48 local output before a
device function overwrote all 2,352 entries. The device function itself then
filled a second 49×48 matrix and copied it to the wrapper.

The Metal translators now:

- omit the wrapper zero-fill;
- statically prove that every output slot is assigned and the temporary is
  never otherwise read;
- write those generated expressions directly to the caller buffer when the
  proof succeeds;
- retain the old initialized temporary/copy path for partial or nested
  outputs.

The 12×12 cyclic Jacobi eigensolver and projection rule remain intact.
Projection reconstruction computes one triangle and mirrors it, preserving
the symmetric result while avoiding duplicate dot products.

## First-pass longer validation

A 24-frame partial-ABD run completed in 29.08 seconds:

- 24/24 frames generated;
- 90 successful nonlinear solves;
- 14,157 CG iterations;
- zero solver errors and zero 0-iteration exits;
- dominant Hessian average: 29.44 ms GPU per solve;
- solver average: 0.393 ms per CG iteration.

Against the corresponding retained pre-optimization frames, sampled frames
0, 5, 10, 15, 20, and 23 changed at most 0.014% of pixels; mean absolute RGB
error stayed below 0.0003 on a 0–255 scale.

A five-soft-bunny container smoke run also completed cleanly with 81 CG
iterations and no zero-iteration or solver-error exit.

## Second low-level pass

The follow-up profile concentrated on host/GPU synchronization and CPU
dispatch preparation. It retained the same generated expressions, PCG
recurrence, collision predicates, and BVH layout.

### GPU dependency chains

- A complete LBVH construction is one ordered Metal batch. Leaf boxes are
  written directly to their final source buffer, and the internal-node AABB
  pass uses one 32-lane SIMD group per node instead of one serial thread
  walking the node's leaf range.
- Coordinate sorting, prefix scans, compaction, sparse-index sorting, and
  sparse-layout resets are dependency-ordered batches.
- Generic multi-level reductions and diagonal-block inversions submit their
  dependent kernels together.
- Continuous collision detection submits reset, face BVH/query, and edge
  BVH/query as one command buffer.
- Discrete collision detection retains the two required host-visible pair
  counts but folds reset, BVH construction, query, separation, and the
  inter-phase counter clear into three submissions instead of nine.

On the controlled one-frame workload:

| Measurement | Before this pass | After | Change |
|---|---:|---:|---:|
| Continuous collision detection | 7.22 ms | 5.47 ms | −24.2% |
| Discrete collision detection | 7.95 ms | 7.41 ms | −6.8% |
| Collision candidates | 2,651 | 2,651 | unchanged |

The dedicated accepted continuous traces are
`/tmp/yasps_lowlevel_ccdfullbatch1.Nxchso`,
`/tmp/yasps_lowlevel_ccdfullbatch2.lK5aIZ`, and
`/tmp/yasps_lowlevel_ccdfullbatch3.tbmn5h`. The accepted discrete traces are
`/tmp/yasps_lowlevel_cddiscretebatch1.CD6KZm`,
`/tmp/yasps_lowlevel_cddiscretebatch2.cl6Tf3`, and
`/tmp/yasps_lowlevel_cddiscretebatch3.9D8aYC`.

### Solver orchestration

PCG scalar state, alpha/beta, denominator checks, and convergence state now
remain on GPU for four complete iterations at a time. A `MetalBatch` encodes
the fixed argument/scalar layout once per solve and reuses it for subsequent
four-iteration chunks. The solver wrapper no longer clears five vectors that
the Metal implementation immediately overwrites.

Three controlled 22-iteration solves took 6.46, 6.82, and 7.64 ms, for a
6.82 ms median. That is 8.7% below the preceding 7.47 ms median and 30.1%
below the first-pass 9.75 ms result.

### Fully overwritten outputs

Generated global attribute kernels write every component for every live
instance. The Metal path therefore skips the legacy pre-dispatch output
clear. CUDA keeps its existing behavior. This removed 363 submissions from
the 24-frame, one-solve-per-frame profile without changing the generated
symbolic computation.

## Current longer validation

The final 24-frame 960×540 partial-ABD run is retained at
`/tmp/yasps_lowlevel_final24.dbx0hw`:

- 24/24 frames generated;
- 93 successful nonlinear solves;
- 14,983 CG iterations;
- zero solver errors or fallback exits;
- solver host-visible time: 3,148.40 ms total, or 0.210 ms per CG iteration;
- 8,399 recorded kernel submissions.

The preceding comparable run took 20.61 seconds, spent 4,519.71 ms in the
solver (0.306 ms per iteration), and recorded 11,930 submissions. The current
run took 20.51 seconds while performing 1.5% more CG iterations. Solver time
per iteration fell 31.4%, and submissions fell 29.6%. End-to-end time changed
only slightly because the dominant generated Hessian remained GPU-bound and
ran slower in the final thermally warm trace.

Against the preceding retained frames, sampled frames 0, 5, 10, 15, 20, and
23 changed at most 0.0085% of pixels. Mean absolute channel error was at most
0.00024 on a 0–255 scale.

Broader validation also completed:

- separate-Jacobian partial ABD: 2 frames, 22/23 CG iterations;
- five-soft-bunny container: 1 frame, 81 CG iterations;
- five-bunny mixed container: 1 frame, 50 CG iterations, active
  point/edge/triangle contacts;
- five-bunny mixed-separation container: 1 frame, 50 CG iterations.

Each smoke run generated its requested frame and completed without a solver
error.

## Rejected experiments

The following measured regressions are not present in the source:

- eight-iteration PCG chunks increased GPU and wall time;
- batching argument-buffer Hessian groups did not improve the dominant
  kernel and added runtime complexity;
- an LDL-based positive-definite fast path before the 12×12 Jacobi
  projection increased dominant Hessian time;
- caching static grouping metadata removed small host copies but produced no
  controlled timing improvement.

## Verification

The Metal suite has 24 passing tests. It includes:

- cross-dispatch buffer-dependency coverage for the single-encoder batch;
- solver convergence and singular-preconditioner handling;
- proof/fallback tests for direct generated output lowering;
- a 12×12 projected-Hessian comparison against `numpy.linalg.eigh`.

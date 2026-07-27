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

## Longer validation

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

## Verification

The Metal suite has 24 passing tests. It includes:

- cross-dispatch buffer-dependency coverage for the single-encoder batch;
- solver convergence and singular-preconditioner handling;
- proof/fallback tests for direct generated output lowering;
- a 12×12 projected-Hessian comparison against `numpy.linalg.eigh`.

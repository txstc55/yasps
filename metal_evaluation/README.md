# YASPS Metal evaluation

This directory records the end-to-end Metal evaluation performed on an
Apple M2 Max on 2026-07-26. Every simulation uses float32, a warm shader
cache, 24 timesteps, and the example's normal nonlinear stopping rule.
No nonlinear iteration cap is active in these final runs.

Rendered variants use 960×540 off-screen frames encoded as 12 fps H.264
videos. Kernel timings come from `MTLCommandBuffer` GPU start/end times,
so compilation, Python logging, OBJ output, and rendering are excluded.
End-to-end wall time includes all of those activities.

## Results

| Variant | Configuration | Wall (s) | GPU total (ms) | GPU/frame (ms) | Nonlinear solves | CG iterations |
|---|---|---:|---:|---:|---:|---:|
| Partial ABD | 1 bunny | 85.73 | 12,667.26 | 527.80 | 99 | 16,263 |
| Partial ABD, separate Jacobian | 1 bunny | 144.19 | 13,936.35 | 580.68 | 91 | 14,414 |
| Container drop | 1 soft bunny | 32.58 | 2,691.43 | 112.14 | 55 | 3,956 |
| Container drop, mixed | 1 soft + 1 affine | 22.17 | 1,407.69 | 58.65 | 25 | 43 |
| Container drop, mixed separation | 1 soft + 1 affine | 69.40 | 7,421.69 | 309.24 | 71 | 4,088 |
| Container drop, no save | 1 soft bunny | 7.29 | 715.73 | 29.82 | 25 | 80 |

The no-save wall time is not directly comparable to rendered variants:
it intentionally omits PyVista, screenshots, and mesh output.

### GPU time by stage

All values are milliseconds accumulated across 24 timesteps.

| Variant | Fused compute | Sparse indices | Hessian | CG solver | CCD | Array/runtime |
|---|---:|---:|---:|---:|---:|---:|
| Partial ABD | 37.52 | 157.59 | 5,691.57 | 2,839.48 | 3,791.97 | 149.13 |
| Partial ABD, separate Jacobian | 26.51 | 321.64 | 5,717.19 | 3,260.55 | 4,543.41 | 67.04 |
| Container drop | 7.82 | 198.87 | 579.99 | 639.30 | 1,216.91 | 48.54 |
| Container drop, mixed | 7.39 | 386.96 | 698.70 | 27.80 | 234.65 | 52.19 |
| Container drop, mixed separation | 34.00 | 270.06 | 1,796.92 | 1,167.95 | 4,064.54 | 88.21 |
| Container drop, no save | 2.74 | 282.90 | 251.68 | 12.86 | 122.24 | 43.31 |

The stage groups correspond to the port's major subsystems:

- Fused compute: source-translated symbolic global functions.
- Sparse indices: generated index kernels, coordinate compression,
  sorting, grouping, lookup, and placement reorder.
- Hessian: fused projected Hessian and gradient assembly.
- CG solver: block sparse matvec, block Jacobi, dot/reduction, and
  diagonal block inversion.
- CCD: LBVH construction, Morton sorting, traversal, separation, and
  ACCD step computation.
- Array/runtime: generic Metal GPUArray operations.

### Hessian mode audit

The three code-generation paths remain distinct:

- Entire-Hessian projection compiles one entry kernel for each observed
  compressed gradient size. It materializes that size's global square
  Hessian and projects it.
- Direct implicit projection compiles one maximum-child specialization.
  It evaluates the generated global Hessian/gradient expression on the
  fly. Metal now dispatches all runtime size groups together because
  this path does not need a size-uniform local matrix.
- Separate H/J projection also compiles one maximum-child
  specialization. Its main term materializes a 12×12 local Hessian,
  56 packed Jacobian values, and the global gradient for the partial-ABD
  stable term. Independently compiled, non-inlined helpers compute each
  upper child-block `JᵀHᵢⱼJ`; their live temporary requirements are
  therefore not additive. This path retains its five runtime size
  groups because collapsing unlike active block patterns was slower.

In a controlled four-step, one-nonlinear-solve-per-step profile,
collapsing the direct implicit groups reduced its stable-term Hessian
GPU time from 248.39 ms to 208.48 ms (16.1%) and dispatch wall time from
368.21 ms to 255.05 ms (30.7%). Applying the same grouping to the
separate path increased GPU time from 253.64 ms to 294.00 ms, so it was
not retained.

The full 24-step run does not show a separate-H/J speedup on the M2 Max.
The stable-term kernel averages 55.76 ms per nonlinear solve in direct
mode and 61.44 ms in separate mode. The smaller local H/J computation is
offset by five group dispatches, uncompressed local placement, and
additional atomic block assembly; sparse-index work is also 321.64 ms
versus 157.59 ms. These are measured Metal results, not an assumption
that CUDA helper register use adds across independent helpers.

### CG submission audit

CG still uses the same generated block sparse matvec, block-Jacobi,
dot/reduction, and vector-update kernels. The Metal runtime now encodes
each dependency-safe phase into one command buffer, with separate
compute encoders preserving ordering. Host synchronization remains only
where CG needs a reduced scalar for `alpha`, the new residual, or its
convergence test.

In the controlled four-step profile this reduced solver submissions
from 1,620 individual dispatch-and-wait calls to 194 phase batches.
Recorded solver dispatch wall time fell from 729.46 ms to 252.95 ms in
direct mode and from 841.23 ms to 477.73 ms in separate mode. No
preconditioner, stopping rule, or CG recurrence was replaced.

Exact per-kernel call counts, GPU/wall timings, top kernels, solver
statistics, and collision candidate statistics are in
[summary.json](summary.json). The flat table is in
[summary.csv](summary.csv). Run [summarize.py](summarize.py) to rebuild
both files from the retained logs and per-kernel timing records.

## Visual records

| Variant | Video | Frames | Log | Exact kernel timings |
|---|---|---|---|---|
| Partial ABD | [MP4](one_bunny_partial_abd/evaluation/one_bunny_partial_abd.mp4) | [PNGs](one_bunny_partial_abd/evaluation/frames) | [log](one_bunny_partial_abd/evaluation/run.log) | [JSON](one_bunny_partial_abd/evaluation/kernel_timings.json) |
| Partial ABD, separate Jacobian | [MP4](one_bunny_partial_abd_separate_jacobian/evaluation/one_bunny_partial_abd_separate_jacobian.mp4) | [PNGs](one_bunny_partial_abd_separate_jacobian/evaluation/frames) | [log](one_bunny_partial_abd_separate_jacobian/evaluation/run.log) | [JSON](one_bunny_partial_abd_separate_jacobian/evaluation/kernel_timings.json) |
| Container drop | [MP4](dropping_in_container/evaluation/dropping_in_container.mp4) | [PNGs](dropping_in_container/evaluation/frames) | [log](dropping_in_container/evaluation/run.log) | [JSON](dropping_in_container/evaluation/kernel_timings.json) |
| Container drop, mixed | [MP4](dropping_in_container_mixed/evaluation/dropping_in_container_mixed.mp4) | [PNGs](dropping_in_container_mixed/evaluation/frames) | [log](dropping_in_container_mixed/evaluation/run.log) | [JSON](dropping_in_container_mixed/evaluation/kernel_timings.json) |
| Container drop, mixed separation | [MP4](dropping_in_container_mixed_separation/evaluation/dropping_in_container_mixed_separation.mp4) | [PNGs](dropping_in_container_mixed_separation/evaluation/frames) | [log](dropping_in_container_mixed_separation/evaluation/run.log) | [JSON](dropping_in_container_mixed_separation/evaluation/kernel_timings.json) |
| Container drop, no save | — | — | [log](dropping_in_container_no_save/evaluation/run.log) | [JSON](dropping_in_container_no_save/evaluation/kernel_timings.json) |

Every video has 24 frames, a 960×540 frame size, and a two-second
duration. The two fully converged partial-ABD implementations also
agree closely after 24 timesteps: the final meshes have a maximum
per-coordinate difference of 2.20×10⁻⁴ and RMS difference of
1.75×10⁻⁵.

## Float32 convergence note

The original partial-ABD example uses a 1×10⁻⁴ maximum-gradient
criterion suited to CUDA double precision. On Metal float32, one
validation frame reduced the residual from 4.62 to 0.0172 and then
1.89×10⁻⁴, but stricter runs eventually oscillated around a
2–3×10⁻⁴ arithmetic floor. The Metal default is therefore 1×10⁻³,
while CUDA retains the original 1×10⁻⁴ value. It can be overridden
with `YASPS_GRADIENT_TOLERANCE`.

The clean convergence check is retained under
`one_bunny_partial_abd/converged_validation`.

## Reproduction

Run:

```sh
./metal_evaluation/run_evaluation.sh
```

The script runs all six variants sequentially, records logs and Metal
kernel timings, encodes the five videos, and regenerates the summaries.

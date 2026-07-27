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
| Partial ABD | 1 bunny | 128.93 | 12,083.93 | 503.50 | 92 | 14,702 |
| Partial ABD, separate Jacobian | 1 bunny | 200.65 | 12,274.91 | 511.45 | 91 | 14,444 |
| Container drop | 1 soft bunny | 50.42 | 4,199.81 | 174.99 | 55 | 3,958 |
| Container drop, mixed | 1 soft + 1 affine | 17.11 | 1,276.93 | 53.21 | 25 | 43 |
| Container drop, mixed separation | 1 soft + 1 affine | 64.80 | 6,943.20 | 289.30 | 71 | 4,090 |
| Container drop, no save | 1 soft bunny | 7.49 | 692.06 | 28.84 | 25 | 80 |

The no-save wall time is not directly comparable to rendered variants:
it intentionally omits PyVista, screenshots, and mesh output.

### GPU time by stage

All values are milliseconds accumulated across 24 timesteps.

| Variant | Fused compute | Sparse indices | Hessian | CG solver | CCD | Array/runtime |
|---|---:|---:|---:|---:|---:|---:|
| Partial ABD | 22.73 | 143.89 | 5,504.64 | 2,469.85 | 3,887.58 | 55.24 |
| Partial ABD, separate Jacobian | 22.82 | 318.43 | 5,966.92 | 2,682.81 | 3,224.74 | 59.18 |
| Container drop | 10.58 | 199.78 | 491.07 | 881.84 | 2,564.88 | 51.65 |
| Container drop, mixed | 6.87 | 375.01 | 640.44 | 23.80 | 177.92 | 52.90 |
| Container drop, mixed separation | 19.66 | 266.72 | 1,957.25 | 742.46 | 3,887.97 | 69.14 |
| Container drop, no save | 3.08 | 291.01 | 227.85 | 14.89 | 117.63 | 37.60 |

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
per-coordinate difference of 1.80×10⁻⁴ and RMS difference of
1.43×10⁻⁵.

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

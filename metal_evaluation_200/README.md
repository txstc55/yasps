# YASPS Metal 200-frame evaluation

This artifact records the requested Metal evaluation on an Apple M2 Max.
Every simulation uses float32, a warm shader cache, 200 timesteps, five
bunnies for each container workload, solver diagnostics, and a maximum of
25 nonlinear iterations per frame. Rendered variants are 960×540 and their
frames are encoded as 12 fps H.264 videos.

The iteration cap is important: a completed frame is not necessarily a
converged frame. The tables therefore report threshold-converged and capped
frames separately. Kernel time is measured from Metal command-buffer GPU
timestamps; wall time also includes Python/C++ orchestration, synchronous
GPU waits, diagnostics, rendering, and mesh output.

## Results

| Variant | Configuration | Wall (s) | Wall/frame (s) | GPU/frame (ms) | Converged | Capped | Solves | CG iterations | Solver failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Partial ABD | 1 bunny | 3,597.55 | 17.99 | 1,830.83 | 160 | 40 | 2,465 | 352,107 | 0 |
| Partial ABD, separate Jacobian | 1 bunny | 7,483.93 | 37.42 | 3,071.57 | 138 | 62 | 3,017 | 553,314 | 24 |
| Container drop | 5 soft | 1,986.40 | 9.93 | 3,543.98 | 78 | 122 | 3,910 | 342,030 | 0 |
| Container drop, mixed | 4 soft + 1 affine | 2,783.84 | 13.92 | 5,354.81 | 12 | 188 | 4,954 | 241,686 | 0 |
| Container drop, mixed separation | 4 soft + 1 affine | 1,532.00 | 7.66 | 2,945.33 | 110 | 90 | 3,357 | 247,410 | 0 |
| Container drop, no save | 5 soft | 2,730.08 | 13.65 | 5,248.90 | 3 | 197 | 4,993 | 273,595 | 0 |

All 22,672 successful linear solves used at least one CG iteration. There
were no false zero-iteration successes, nonfinite residual exits, or
preconditioner fallbacks. The separate-Jacobian partial variant additionally
reported 24 genuine non-positive CG denominator exits. Those codes are
preserved in `summary.json`; they are not zero-iteration or NaN successes.

The no-save program follows its own simulation path and reached the
nonlinear cap on 197 frames, versus 122 for the rendered soft-container
program. Its wall-time difference is therefore not a rendering-cost
measurement.

### GPU time per frame by stage

All values are milliseconds.

| Variant | Fused compute | Sparse indices | Hessian | CG solver | CCD | Array/runtime |
|---|---:|---:|---:|---:|---:|---:|
| Partial ABD | 7.51 | 28.78 | 769.13 | 553.29 | 458.95 | 13.17 |
| Partial ABD, separate Jacobian | 9.88 | 38.57 | 1,077.75 | 963.75 | 966.16 | 15.46 |
| Container drop | 8.72 | 311.13 | 743.30 | 658.68 | 1,796.92 | 25.24 |
| Container drop, mixed | 16.69 | 403.13 | 1,234.78 | 446.56 | 3,220.09 | 33.55 |
| Container drop, mixed separation | 7.83 | 96.89 | 829.66 | 449.88 | 1,543.15 | 17.91 |
| Container drop, no save | 12.57 | 548.34 | 911.21 | 541.05 | 3,203.43 | 32.30 |

## Hessian audit

The generated partial-ABD stable-Neo-Hookean shader projects a **12×12**
local Hessian (`spd_projection_inplace<12>`). It does not run a 48×48 EVD.
The generated `YaspsMatrix<49,48>` is the chained 48-DOF Hessian-plus-
gradient output that must be assembled into the global sparse system.

For the direct partial variant, the dominant stable Hessian kernel averaged
53.99 ms GPU time per nonlinear solve, close to 55.76 ms in the earlier
24-frame evaluation. The observed ~72 ms samples are individual high calls,
not a new mean regression. A controlled one-frame projection toggle measured
49.61 ms with the 12×12 projection and 38.98 ms without it: local projection
costs about 10.63 ms, while about 39 ms remains in chain-rule expansion,
materialization, and atomic assembly.

The older optimized container graph exposes deformation gradient `F` as a
9-DOF attribute and averaged about 10.42 ms for its stable Hessian kernel.
That graph is not equivalent to the current partial helper, whose projection
boundary is the tet's 12 position DOFs.

## CG/SpMV audit

The long partial trace initially showed about 2.00 ms wall time for the CG
denominator phase and 2.75 ms for the residual phase per iteration, while
their Metal GPU work was only about 0.213 ms and 0.097 ms. The shader SpMV
was not taking 3 ms. The gap came from two synchronous command-buffer waits
per iteration plus host-side buffer lookup.

The Objective-C++ Metal runtime previously linearly scanned every allocation
to resolve each sliced buffer view. Replacing that scan with ordered
`upper_bound` interval lookup made a 16-dispatch microbenchmark essentially
allocation-count independent: at 5,000 live allocations its median fell
from 0.847 ms to 0.264 ms. A one-frame partial profile after the fix measured
0.287 ms wall for the denominator phase and 0.214 ms for the residual phase,
with 0.629 ms total printed solver time per CG iteration.

Python/Cython still orchestrates PCG and Objective-C++ still encodes,
commits, and synchronously waits for the scalar reductions. Combining a
whole iteration into fewer GPU/host synchronization points remains the next
possible optimization, but the C++→Metal call itself is not the measured
3 ms bottleneck.

## Visual records

| Variant | Video | Frames | Log | Kernel timings |
|---|---|---|---|---|
| Partial ABD | [MP4](one_bunny_partial_abd/evaluation/one_bunny_partial_abd.mp4) | [PNGs](one_bunny_partial_abd/evaluation/frames) | [log](one_bunny_partial_abd/evaluation/run.log) | [JSON](one_bunny_partial_abd/evaluation/kernel_timings.json) |
| Partial ABD, separate Jacobian | [MP4](one_bunny_partial_abd_separate_jacobian/evaluation/one_bunny_partial_abd_separate_jacobian.mp4) | [PNGs](one_bunny_partial_abd_separate_jacobian/evaluation/frames) | [log](one_bunny_partial_abd_separate_jacobian/evaluation/run.log) | [JSON](one_bunny_partial_abd_separate_jacobian/evaluation/kernel_timings.json) |
| Container drop | [MP4](dropping_in_container/evaluation/dropping_in_container.mp4) | [PNGs](dropping_in_container/evaluation/frames) | [log](dropping_in_container/evaluation/run.log) | [JSON](dropping_in_container/evaluation/kernel_timings.json) |
| Container drop, mixed | [MP4](dropping_in_container_mixed/evaluation/dropping_in_container_mixed.mp4) | [PNGs](dropping_in_container_mixed/evaluation/frames) | [log](dropping_in_container_mixed/evaluation/run.log) | [JSON](dropping_in_container_mixed/evaluation/kernel_timings.json) |
| Container drop, mixed separation | [MP4](dropping_in_container_mixed_separation/evaluation/dropping_in_container_mixed_separation.mp4) | [PNGs](dropping_in_container_mixed_separation/evaluation/frames) | [log](dropping_in_container_mixed_separation/evaluation/run.log) | [JSON](dropping_in_container_mixed_separation/evaluation/kernel_timings.json) |
| Container drop, no save | — | — | [log](dropping_in_container_no_save/evaluation/run.log) | [JSON](dropping_in_container_no_save/evaluation/kernel_timings.json) |

Each video was independently checked with `ffprobe`: 960×540, 12 fps,
exactly 200 frames, and 16.667 seconds. Start, middle, and final frames were
also visually inspected. The partial variants are visually coincident at
frame 199; only 1.06% of pixels differ, with mean absolute RGB difference
0.061 on a 0–255 scale.

Exact per-kernel call counts and timings, solver error codes, line-search
statistics, and collision-candidate statistics are in
[summary.json](summary.json). [summary.csv](summary.csv) is the flat table.

## Reproduction

```sh
YASPS_EVAL_OUTPUT_ROOT="$PWD/metal_evaluation_200" \
YASPS_EVAL_FRAMES=200 \
YASPS_EVAL_SOFT_BUNNIES=5 \
YASPS_EVAL_MIXED_BUNNIES=5 \
YASPS_EVAL_MAX_INNER_ITERATIONS=25 \
YASPS_EVAL_SOLVER_DIAGNOSTICS=1 \
./metal_evaluation/run_evaluation.sh
```

The final Metal regression suite result was `20 passed`.

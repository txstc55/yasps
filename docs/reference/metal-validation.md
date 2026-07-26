# Metal validation

This page records the end-to-end acceptance run used to validate the Apple
Metal backend. It separates measured behavior from the feature claims in the
compatibility table and documents a solver failure discovered by inspecting
the first five-bunny render.

## Corrected acceptance run

The five-bunny dropping-in-a-container example completed all 500 requested
frames on 25 July 2026.

| Item | Result |
| --- | --- |
| Backend | Metal through MLX |
| Device | Apple M2 Max (`applegpu_g14s`) |
| Operating system | macOS 26.5.2, arm64 |
| Numerical dtype | float32 |
| Deformable vertices | 95,965 |
| Tetrahedra | 399,675 |
| Collision triangles | 104,170 |
| Collision edges | 156,257 |
| Completed/saved frames | 500/500 |
| Accepted Newton updates | 5,986 |
| Newton updates per frame | minimum 2, median 12, mean 11.972, maximum 20 |
| Numerical simulation time | 5,092.910 s (84 min 52.910 s) |
| End-to-end process time | 5,096.18 s |
| Frame time | minimum 1.001 s, median 10.171 s, p95 18.196 s, maximum 19.647 s |
| Newton update time | median 0.838 s, p95 0.943 s, maximum 1.084 s |
| Hessian/numeric evaluation | median 111.90 ms, p95 155.62 ms, maximum 226.53 ms |
| Maximum resident set | 1,050,591,232 bytes |
| Peak process footprint | 6,687,526,248 bytes |

The command was run from the repository root:

```bash
YASPS_BACKEND=metal .venv/bin/python \
  examples/dropping_in_container/dropping_in_container_metal.py \
  --num-bunnies 5 \
  --steps 500 \
  --max-newton 20 \
  --solver-iterations 20000 \
  --save-meshes \
  --output-directory \
    examples/dropping_in_container/meshes_metal_5_bunnies_500_fixed
```

`/usr/bin/time -l` wrapped the command for process-level timing and memory
measurements. The numerical source baseline was commit
[`567c861`](https://github.com/txstc55/yasps/commit/567c861).

### Performance over the trajectory

The frame cost tracks physical and contact work rather than output encoding:

| Simulation frames | Frames | Total | Mean/frame | Median/frame | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–139 | 140 | 1,728.596 s | 12.347 s | 15.698 s | 19.647 s |
| 140–249 | 110 | 1,883.603 s | 17.124 s | 17.392 s | 18.305 s |
| 250–399 | 150 | 1,269.794 s | 8.465 s | 7.943 s | 16.568 s |
| 400–499 | 100 | 210.905 s | 2.109 s | 1.874 s | 4.357 s |

The first two intervals contain the densest impact and rearrangement work. The
last interval is cheaper because the bunnies are approaching equilibrium, but
it is not a repeated-frame shortcut: the final frame still accepted a
`0.0271142`-speed Newton update followed by a `0.000958145` update.

## Why the first movie stopped after three seconds

The first 500-frame process did finish, but its physical state did not.
Simulation frame 140 was the last changed OBJ. Frames 141 through 499 were
byte-for-byte identical. Rendering every second simulation frame placed the
first frozen movie frame at 2.96 seconds, matching the visible stop.

The old log showed the precise transition:

- frame 139 used all 20 Newton updates and remained dynamically active;
- frame 140 failed at Newton update 8;
- the next 359 frames each returned a zero direction;
- 360 `scene.minimizeEnergy: got error code -1` messages were emitted; and
- the apparent 29-minute total was artificially short because the frozen tail
  cost only about 0.825 seconds/frame.

Decoded video motion also dropped sharply at 2.96 seconds. The mean adjacent
pixel change fell from about `0.57` before the transition to `0.00063` after
it; the remaining tiny values are H.264 prediction/quantization drift.
PyVista rendered every requested input and the MP4 contained all 269 encoded
frames, so neither rendering nor encoding caused the freeze.

### Numerical root cause

One edge-edge barrier pair reached the activation distance with a finite
energy and gradient. Its geometric denominator was
`b·b = 2.3205016e-10`. The generic quotient-rule expression

```text
(dA * B - A * dB) / B²
```

was then differentiated a second time. The resulting expression formed
`(b·b)^4 = 2.899529e-39`, below the smallest normal float32 value
`1.1754944e-38`. Metal fast math flushed that intermediate to zero.

The consequences were measurable:

- 90 NaNs in the dynamic edge-edge Hessian;
- four NaN block-Jacobi inverse blocks;
- a NaN preconditioned residual at PCG iteration zero; and
- a zero solution reused by every later frame.

CUDA uses double precision and does not underflow at this scale.

### Fix and regression

The shared `autodiff.pyx` implementation now uses the algebraically equivalent
Metal specialization

```text
dA / B - (A / B) * (dB / B)
```

which avoids creating the fourth-power denominator. The CUDA specialization
retains its original double-precision expression. The exact four-vertex
edge-edge configuration is preserved in a regression test that requires the
gradient, sparse Hessian, and block-Jacobi inverse to remain finite.

Reconstructing the old frame-141 state after the fix produced zero Hessian
NaNs and a finite Newton direction with maximum magnitude `0.0185227`. In the
fresh run, frame 141 continued with accepted speeds `0.289345`, `0.286132`,
and further nonzero updates instead of freezing.

## Collision and optimization evidence

The logged contact tuple is ordered point-point, point-edge, point-triangle,
and edge-edge. Every contact type became active during the run.

| Contact type | First active frame | Maximum active pairs |
| --- | ---: | ---: |
| Point-point | 43 | 221 |
| Point-edge | 37 | 1,851 |
| Point-triangle | 37 | 2,500 |
| Edge-edge | 37 | 4,330 |

The maximum combined active set was 8,626 pairs. The final Newton update
retained `(26, 461, 1372, 812)` contacts, so all four dynamic barrier and
friction paths remained active late in the run.

Continuous collision detection reduced the accepted step below one in 3,241
Newton updates; the smallest accepted step was `0.0156229`. No Newton proposal
needed the configured displacement cap. Of 5,986 logged updates, 138 appeared
uphill at the printed precision; the maximum difference was
`6.0e-8`, within the driver's float32 acceptance slack.

The corrected run contained none of the following:

- Metal PCG nonconvergence;
- wrapper error codes;
- NaNs;
- CCD candidate-capacity overflows;
- line-search stalls;
- runtime exceptions; or
- tracebacks.

All 500 OBJ files have distinct SHA-256 hashes. There are no identical
adjacent frames and no duplicated mesh state anywhere in the corrected
sequence.

## Corrected PyVista render

The saved surfaces were rendered separately so VTK is excluded from the
numerical timing:

```bash
.venv/bin/python \
  examples/dropping_in_container/render_metal.py \
  --input-directory \
    examples/dropping_in_container/meshes_metal_5_bunnies_500_fixed \
  --num-bunnies 5 \
  --stride 2 \
  --video \
    examples/dropping_in_container/artifacts/yasps_metal_5_bunnies_500_frames_corrected.mp4 \
  --screenshot \
    examples/dropping_in_container/artifacts/yasps_metal_5_bunnies_final_corrected.png
```

Rendering took 19.62 seconds. The movie contains 251 sampled simulation frames
followed by an 18-frame hold. Every PyVista frame is explicitly rendered
before it is passed to the encoder.

| Artifact | Metadata | SHA-256 |
| --- | --- | --- |
| Corrected MP4 | H.264, 1600×912, 24 fps, 269 frames, 11.209 s, 1,173,558 bytes | `0a632e073d046eb0a1210240733c1b0f57c0a565f4d850999f360510f4ac9ed8` |
| Corrected final PNG | RGB, 1600×912, 549,428 bytes | `9a788b7353711fe6eab9b9fad57a6adf01a27d106f036c7b7099e0dae2e26e37` |

The MP4 and PNG are generated artifacts and are intentionally ignored by Git.
Run the commands above to reproduce them.

## Automated checks

The backend and Metal suite reports:

```text
44 passed
```

The tests cover backend selection and array behavior, attribute evaluation,
derivatives and matrix operators, Jacobi eigendecomposition and PSD
projection, sparse coordinate compression and assembly, block inversion, PCG
success and failure states, all four collision pair types, broad-phase growth,
conservative continuous collision steps, and the float32 contact-Hessian
regression.

The documentation also builds successfully with:

```bash
mkdocs build --strict --clean
```

## Scope

This run establishes that the complete example executes on a large Apple
Metal scene with elasticity, inertia, gravity, friction, sparse Hessian
assembly, eigenspectrum projection, PCG, and dynamic PP/PE/PT/EE collision
terms. It is not a CUDA-versus-Metal performance comparison. CUDA could not be
compiled or executed on the Apple-silicon validation host, so the CUDA path
was preserved and reviewed but not revalidated there.

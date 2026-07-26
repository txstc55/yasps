# Metal JIT validation

> This page is intentionally excluded from the published documentation while
> validation and CUDA comparison continue.

## Validation host

| Item | Value |
| --- | --- |
| Date | 25-26 July 2026 |
| Device | Apple M2 Max (`applegpu_g14s`) |
| Operating system | macOS 26.5.2, arm64 |
| Backend | MLX custom Metal kernels |
| Real dtype | float32 |
| Branch | `metal` |

## Automated suite

```text
30 passed
```

The suite covers:

- generated scalar/vector/matrix attribute programs;
- named-module reuse, JOIN, and UNION;
- inverse, determinant, EVD, and SPD projection;
- hierarchical scan and generated sparse indices;
- coordinate compression and separate-Jacobian placement reorder;
- Hessian/gradient assembly;
- diagonal block inversion;
- generated sparse PCG and failure states;
- discrete face/edge classification;
- continuous swept candidates and step size;
- atomic CCD capacity growth; and
- the two-pass zero-pair fallback.

The public documentation also passes:

```bash
mkdocs build --strict --clean
```

Excluded Metal pages are absent from the built files and search index.

## Exact-setting scene launchers

Every launcher sets the backend and executes its original source with
`runpy`, so scene constants and solver settings are not duplicated:

- `brazil_nuts/brazil_nuts_metal.py`;
- `one_bunny_partial_abd/one_bunny_partial_abd_metal.py`;
- `one_bunny_many_cloths/one_bunny_many_cloths_metal.py`;
- `one_bunny_partial_abd_separate_jacobian/`
  `one_bunny_partial_abd_separate_jacobian_metal.py`;
- `teaser/teaser_metal.py`; and
- `dropping_in_container/dropping_in_container_metal.py`.

Only backend precision changes from CUDA float64 to Metal float32.
Frame/display/save environment controls bound validation work but default to
the original run behavior.

## Multi-frame results

| Scene | Run | Result |
| --- | --- | --- |
| Dropping in container | 3 frames | clean; continuous CCD mostly 8.4-9.9 ms |
| Partial ABD bunny | 3 frames | clean; accepted-state float32 floor handling |
| Partial ABD, separate Jacobian | 3 frames | clean; energies agree with ordinary path at micro-scale |
| Bunny with many cloths | 3 frames | clean; total reported simulation time 1.128 s |
| Teaser | 3 frames | clean; PP, PE, PT, and EE all active |
| Brazil nuts | 3 full frames | clean; 9, 9, and 11 nonlinear iterations; 93.914 s simulation time |

No final run emitted a traceback, non-finite numerical warning, or negative
solver error code.

## Performance evidence

Measurements are Apple-host comparisons between Metal implementation
variants, not CUDA-versus-Metal results.

### PCG recurrence

On the many-cloths system, moving alpha/beta/status recurrence to Metal and
dispatching it from the compiled C++ driver in 32-iteration chunks reduced a
233-iteration recurrence from roughly 117 ms to 36.6 ms.

### Batched EVD

For 8,192 random 12x12 symmetric matrices, the generated threadgroup Jacobi
path measured roughly 2.37-2.78 ms versus 5.2-7.9 ms for the sequential
generated projection. The many-cloths 12x12 tetrahedral term improved from
about 29.8 ms to 19.8 ms.

The batched path is intentionally conservative: dimensions 12-32, at least
1,024 instances, absolute-eigenvalue projection method 1, and bounded staged
storage. Smaller 9x9 bending terms remain on the local path because blanket
staging regressed them.

### CCD

Brazil profiling found the previous count-scan-write edge query traversed the
same hashed-cell candidates twice at about 110-145 ms per traversal. The
generated capacity-aware append query performs one traversal at about
127-132 ms for continuous edges and avoids an unused continuous pair buffer.

Steady-state Brazil continuous CCD measured about 113-165 ms plus roughly
27-30 ms for the generated step reduction. Discrete detection was about
140-176 ms in the three-frame run.

## Correctness issues found during validation

### Float32 PCG floor

Long contact systems can reach a finite residual floor without satisfying a
double-oriented tolerance. The C++ driver now:

- replaces the residual exactly every 32 iterations;
- tracks the best finite iterate;
- detects sustained lack of float32 progress; and
- restores the best improved iterate as a successful floor termination.

Non-finite values and non-positive curvature remain errors.

### Lazy CCD count scan

Chaining a long custom count query directly into the hierarchical scan
intermittently produced a corrupt terminal total on teaser-scale edge arrays.
The two-pass reference path now evaluates the count kernel before scanning,
putting the long traversal in its own command buffer. The single-pass append
path does not require this scan.

### Interactive GPU watchdog

Running PyVista interactive rendering during long Metal CCD workloads can
trigger macOS's `Impacting Interactivity` command-buffer watchdog. Compute
benchmarks therefore disable display and saving. A separate one-frame
off-screen PyVista export completed and produced a valid bunny/container
image.

## Remaining comparison work

CUDA cannot be executed on this Apple-silicon host. A fair cross-backend
benchmark still needs an NVIDIA machine using:

- the same source commit;
- identical scene constants and launchers;
- warm JIT caches;
- separate construction, first-use, and steady-state timing;
- matching frame/inner-iteration ranges; and
- trajectory/contact/energy checks in addition to wall time.

The Metal pages remain unpublished until that validation is complete.

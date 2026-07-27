# Metal Hessian bottleneck investigation

## Result

The dominant Hessian kernel is primarily projection-bound, not
assembly-bound.

The controlled workload is the 79,935-thread stable-Neo-Hookean Hessian term
from one-bunny partial ABD on an Apple M2 Max. Shader caches were warm and
projection/no-projection runs were alternated to reduce temperature bias.

| Phase | Approximate GPU time | Share |
|---|---:|---:|
| Symbolic and local Hessian computation | 10.3 ms | 32% |
| 12×12 EVD/PSD projection | 14.9 ms | 46% |
| Sparse blocks, diagonal blocks, and gradient assembly | 7.2 ms | 22% |

The phase split used four versions of the same generated kernel:

1. full computation, projection, and assembly;
2. projection method zero, preserving computation and assembly;
3. full projection with Hessian block assembly temporarily compiled out;
4. projection method zero with block assembly temporarily compiled out.

The temporary assembly switch was removed after measurement. The retained
phase traces are:

- full:
  `/tmp/yasps_hessian_phase_p2.ejpCgK`,
  `/tmp/yasps_hessian_phase_p2.tqhW7w`,
  `/tmp/yasps_hessian_phase_p2.kwq4Hg`;
- no EVD:
  `/tmp/yasps_hessian_phase_p0.p0lvLT`,
  `/tmp/yasps_hessian_phase_p0.d3cN6F`,
  `/tmp/yasps_hessian_phase_p0.tq6VTV`;
- no assembly:
  `/tmp/yasps_hessian_phase_noassembly_p0.3Tlfrz`,
  `/tmp/yasps_hessian_phase_noassembly_p2.mZMQOM`,
  `/tmp/yasps_hessian_phase_noassembly_p0.UWGKif`,
  `/tmp/yasps_hessian_phase_noassembly_p2.rKdl1x`,
  `/tmp/yasps_hessian_phase_noassembly_p0.I4v75y`,
  `/tmp/yasps_hessian_phase_noassembly_p2.Qb2JAo`.

## Accepted EVD optimization

The Metal Jacobi projection now:

- uses at most eight cyclic sweeps instead of twelve;
- skips rotations at the final `1e-6` convergence precision;
- copies the twelve projected eigenvalues into a small array before
  reconstruction, ending the 144-float work-matrix lifetime earlier and
  reducing simultaneous thread-local storage pressure.

The reconstruction and negative-eigenvalue rules are unchanged.

Three controlled one-frame traces measured 33.39, 29.66, and 29.98 ms GPU,
with a 29.98 ms median. The preceding median was 31.67 ms, a 5.3% improvement.

Across a thermally sustained 24-frame run, the dominant kernel averaged
29.79 ms GPU per solve. Comparable preceding runs averaged 34.34–35.71 ms,
so the sustained reduction was 13.2–16.6%. The accepted 24-frame trace is
`/tmp/yasps_hessian_evd_candidate24.zgXpCj`.

## Correctness

- The Metal suite has 24 passing tests.
- Projection is checked against `numpy.linalg.eigh` on symmetric 12×12
  matrices at `1e-3`, `1`, and `1e3` scales, plus a prescribed mixed
  eigenspectrum.
- The largest observed absolute error in those checks was 0.00342 for a
  matrix whose entries reached 1,651; relative Frobenius error was
  1.56e-6.
- The 24-frame partial-ABD candidate and baseline both used exactly 566 CG
  iterations.
- Sampled rendered frames changed at most 0.0243% of pixels, with maximum
  mean absolute channel error 0.000544 on a 0–255 scale.
- A four-frame separate-Jacobian run and an active-contact five-bunny mixed
  container run completed without solver errors.

## Rejected variants

- A conventional symmetry-preserving Jacobi update increased the dominant
  kernel to 32.66–36.16 ms because its additional symmetric stores were
  unfavorable on this GPU.
- Six- and seven-sweep caps were slower than eight after Metal compilation.
- A `1e-5` convergence or per-rotation cutoff was numerically acceptable in
  the unit case but slower in the real generated kernel.


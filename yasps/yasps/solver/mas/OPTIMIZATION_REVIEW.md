# MAS CUDA optimization review

2026-09-11. This is a source/resource-usage audit supported by the saved
frame-42 timings, not a new Nsight counter profile or a claim of measured
speedups. No optimizations below have been implemented in this change.

## Measured starting point

On the RTX 4090, the first fixed frame-42 system measured 0.830 ms per CG
iteration, 0.514 ms per standalone FP64 SpMV and 0.243 ms per standalone MAS
application. The standalone operations are not an additive decomposition of
CG: CG fuses curvature/dot reductions and workspace clearing differently.
These measurements preceded removal of the old tiny-pivot workaround.

Across the 91-Newton FP64 replay, all local inversions together took 0.145 s
out of 637.100 s frame time; all numerical preconditioner updates took 0.530 s.
Optimize repeated SpMV and application before inversion-only arithmetic.
Hessian/gradient assembly (313.716 s) is outside the MAS solver and will not
be improved by a MAS-only kernel change.

The current solver already has shape specialization/unrolling, grouped
inverse kernels, persistent workspaces, static/dynamic SpMV pairing, segmented
row reductions, fused auxiliary work/dot products, and device-controlled CG
graphs. The question is where those choices waste work on this workload.

After removing the unit-divisor workaround and restoring the default pivot
tolerance, all 26 CPU/GPU regression tests pass. A fresh frame-42 fixed-system
check completed five 256-iteration CG budgets without breakdown: median
0.793 ms/CG, standalone SpMV 0.494 ms and application 0.252 ms. This confirms
the cleanup did not reintroduce the early inverse/application failure; these
are capped timing samples, not converged solves or an isolated speedup claim.
The independently recomputed relative residual after that short budget is
1.819; this is not a substitute for replaying a converged full solve/frame.
The application GPU inertia-mass minimum remains 1e-8.

## Highest-priority experiments

1. **Skip empty dynamic-capacity CTAs.**

   `cuda_runtime.py` grows `dynamic_block_launch_capacity` by approximately
   1.5x to avoid recapturing graphs. In `cuda/block_spmv.cu`,
   `yasps_mas_specialized_fine_block_spmv_pair` predicates multiplication on
   live counts, but unused tail threads still enter segmented and block
   curvature reductions. Add a block-uniform exit only when the entire CTA
   is past both live matrix work and any fused auxiliary work. Preserve all
   synchronization requirements. Measure actual versus launched CTAs and
   same-matrix CG cost; do not assume a 1.5x allocation means a 1.5x speedup.

2. **Batch tiny inverse applications and reduce padding.**

   The saved explicit-hierarchy bank audit contains 163,733 domains;
   155,883 (95.21%) are 3x3 with an 8x8 storage stride. The active
   `yasps_mas_apply_exact_domain_warp` uses a warp per domain and computes
   rows in `row = lane; row < N; row += 32`, so only three row lanes do useful
   arithmetic for these banks. Batch multiple 3x3 matrices within a warp,
   potentially with a SoA layout for coalesced loads.

   Those 3x3 banks occupy 79,812,096 bytes per FP64 arena, versus 11,223,576
   bytes when tightly stored. This is a storage opportunity, not proof the
   current apply reads all 64 padded entries: it loops over active entries.
   Across all bank sizes there are 18,142,144 padded versus 9,165,450 active
   matrix entries. Restriction, correction storage, alignment and generated
   offsets must be updated consistently if padding changes.

3. **Fold certified duplicate singleton paths.**

   In the saved frame, 31,164 singleton banks equal `1e8 * I` at each of
   levels 0 through 4, corresponding to inactive grid nodes with the
   application mass floor. Repeated restriction/application/collection for
   identical domains can potentially become one operation. This must
   preserve all level weights, duplicate-level factors, normalization,
   fine-level weights and numerical contributions. Do not drop a node merely
   because its current residual is zero or its physical P2G mass is zero.
   Dynamic numerical coupling can exist outside the partition graph.

   A narrower experiment is to precompute uniquely owned restriction
   destinations: singleton parents can use direct stores instead of segmented
   reductions and FP64 atomics. This can also remove corresponding clears;
   the current frame layout clears 485,362 coarse doubles per CG iteration.
   Shared destinations must retain the correct reduction.

4. **Build a compact operator once per Newton for repeated SpMV.**

   The measured first system supplies 2,957,287 dynamic blocks, including
   duplicate contributions from bounded Hessian batches. An earlier sparse
   first-state audit obtained 5,910,354 scalar nonzeros after expanding,
   summing duplicates and eliminating zeros. Its RHS hash differs, so this
   is motivation to measure compaction, not an exact compression ratio for
   the timed matrix.

   A GPU-built block-CSR/BSR operator could sum duplicates once per numerical
   update and use row-owned output instead of repeated atomics. Freeze/reuse
   structural maps where valid, but refresh every changing value and dynamic
   contact pattern. Include construction time, extra memory and changed
   summation order in validation. With hundreds or thousands of CG iterations,
   a modest setup cost may amortize well.

## Further hot-kernel opportunities

- **Matrix loads and transpose atomics.** The fine-pair 3x3 kernel assigns
  successive blocks to successive lanes. Corresponding coefficient loads
  therefore have a 72-byte lane stride in the native AoS layout. The row pass
  combines contributions with segmented reduction; the symmetric transpose
  pass issues separate output atomics per block column. Test block-SoA or
  cooperative loads and row-owned/sorted transpose traversal. Source load
  counts do not establish DRAM traffic; measure sectors, cache hits and atomic
  contention. Sorted-transpose kernels exist but are not selected by the
  current production dispatch.

- **Curvature reduction.** CG SpMV accumulates one double atomic per CTA into
  one global scalar. Compare otherwise identical calls with/without this
  reduction. If costly, use block partials followed by a parallel final
  reduction. The existing `prepare_iteration_partials` loops over all partials
  in one thread; enabling that path unchanged is not a suitable replacement
  for thousands of CTAs.

- **Launch tuning, with an existing correctness prerequisite.** The shape
  heuristic chooses eight warps per CTA for 3x3/4x4 SpMV. The saved 3x3
  fine/fine-pair binary uses 56 registers/thread, 64 shared bytes and no local
  memory/stack spill. Test 128 versus 256 threads with occupancy/stall data.
  Before tuning inverse thread counts, resolve the existing mismatch between
  GROUPS derived from `threads_per_block` and packed launches hardcoded from
  96, and include group/thread configuration in the exact-kernel cache key.
  The default 96-thread configuration is internally consistent; do not change
  this setting blindly.

- **Avoid blind fusion.** Current restriction/warp-apply binaries use 40
  registers/thread without local memory; fused collection/dots uses 48
  registers and 256 shared bytes. The disabled fine-fused alternative uses
  124 registers and 1536 shared bytes. Fewer launches can reduce occupancy or
  worsen scattered accesses. The recurrence/restriction fusion is deliberately
  disabled because restriction order scrambles recurrence-vector accesses.

- **Graph control overhead.** The conditional loop launches a separate
  single-thread condition update through a child graph. Folding that into
  the final update kernel may help, but is lower priority. Preserve exact
  iteration-budget, convergence and restart semantics.

## Validation order

Start with unused-CTA removal and a tiny-bank apply specialization, separately.
For each, use identical H, RHS and inverse banks; time warmed SpMV, application,
CG iterations and the complete solve. Check independent FP64 `b - A*x`,
preconditioner quadratic forms, iteration counts and memory. Include mixed
block sizes, empty dynamic buffers, changed live counts, inactive grid nodes,
large/padded banks and hierarchy rebuilds. Then test full frame 42 without
changing its physics, tolerance or iteration limits. Kernel speed alone is
not enough if numerical convergence gets worse.

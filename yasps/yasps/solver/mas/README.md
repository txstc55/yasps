# MAS hierarchy and precision

The public solver facade supports:

```python
solver.rebuildHierarchy(block_positions, block_dimensions, num_blocks)
```

Both inputs must be flat integer GPU arrays of length `2 * num_blocks`:

- `block_positions`: global scalar `(row_start, column_start)` pairs.
- `block_dimensions`: `(rows, columns)` for each individual block, not each shape category.

Include every variable, including isolated variables through their diagonal
blocks. Offsets and dimensions must describe a contiguous DOF layout.
Duplicate and reversed block coordinates are accepted. No numerical Hessian
values are needed or inserted into the operator. The existing CPU METIS
partitioner reads the downloaded graph metadata.

If no hierarchy exists, the first solve builds one from the numerical
Hessian's static sparsity. Otherwise the hierarchy is reused until an explicit
rebuild or reset; there is no automatic static-sparsity comparison. A changed
variable layout requires rebuilding. Static operator coordinates must stay
fixed between rebuilds; dynamic coordinates and all numerical values may change.

Rebuilding discards hierarchy-dependent numerical maps, inverse banks,
workspaces, graphs and the borrowed solution. The next solve builds numerical
state from its actual Hessian. Supplying a different numerical matrix refreshes
its numerical state without silently replacing an explicit hierarchy.

Local inverse storage and the complete preconditioner application are FP64.
Historical internal names containing `mixed` are retained for compatibility;
optional mixed SpMV storage is a separate feature. Inversion never replaces a
small or zero pivot with one: invalid pivots follow the existing failure and
Cholesky-fallback paths. The default pivot tolerance is `1e-12`. Any physical
mass floor belongs in the application, not this solver.

A reported numeric failure consumes that submission and forces a fresh rebuild
at the next update, even with a lagged-preconditioner interval. This prevents
one failure from being rethrown before subsequent Hessians can be assembled.
PCG still rejects invalid banks until a rebuild passes validation.

The default Gauss-Jordan path requests a Cholesky retry for nonfinite output
or suspicious Schur pivots, including a relative pivot below `1e-8` or a
pivot spread above `1e7`. These are retry criteria, not diagonal shifts.
Cholesky factors the original bank after diagonal equilibration,
`B = D^-1/2 A_local D^-1/2`, where `D = diag(A_local)`, and undoes the scaling
on its inverse. No fallback adds anything to the diagonal. Small positive
normalized pivots remain valid if their physical pivots exceed the configured
tolerance; nonpositive/nonfinite pivots are rejected. Gauss-Jordan tests signed
pivots, so a negative pivot cannot pass merely because its magnitude is large.

Non-positive CG curvature requests a restart with a freshly computed `b-Ax`,
not only a direction reset from the old recurrence residual. At most eight
such restarts share the original iteration budget. Catastrophic growth of
`r^T M^-1 r` beyond `1e16` times its original RHS reference reports divergence
instead of running until floating-point overflow. Exhausting the iteration
budget is reported explicitly, not as an unspecified non-SPD error.
Likewise, 1,024 iterations without a 1% improvement in the best preconditioned
residual report stagnation immediately, without a restart. The public MAS
code is `-5`, and the current iterate remains available to the caller. This
does not declare convergence or relax the tolerance. Oscillations with
continuing progress remain permitted.

The public YASPS solver returns `-8` if local block inversion fails after its
fallback. `statistics.breakdown` preserves the failure details, `iterations`
is zero, and `solution` is empty because no valid iterate exists for that call.
The hierarchy is retained, and a subsequent solve rebuilds the failed numeric
state normally. This translates the internal `LocalInverseError`; unrelated
input-validation and CUDA exceptions still propagate.

CG success uses its existing squared preconditioned residual criterion,
not a Euclidean relative-residual threshold. The reported relative residual
normally uses the CG recurrence. Use an independent SpMV when an actual
`b - A*x` residual audit is needed.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s yasps/tests -p 'test_mas_*.py' -v
```

Tests cover hierarchy replacement/reuse, actual numerical matrices, FP64
storage/application, strict pivot rejection, cross-warp inverse synchronization,
numeric failure recovery, and clearing coarse
workspaces larger than the fine vector. The saved frame-42
bank regression is optional; synthetic numerical tests do not need scene assets.

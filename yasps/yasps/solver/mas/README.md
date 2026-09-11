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

CG success uses its existing squared preconditioned residual criterion,
not a Euclidean relative-residual threshold. The reported relative residual
normally uses the CG recurrence. Use an independent SpMV when an actual
`b - A*x` residual audit is needed.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s yasps/tests -p 'test_mas_*.py' -v
```

Tests cover hierarchy replacement/reuse, actual numerical matrices, FP64
storage/application, strict pivot rejection, cross-warp inverse synchronization
and clearing coarse workspaces larger than the fine vector. The saved frame-42
bank regression is optional; synthetic numerical tests do not need scene assets.

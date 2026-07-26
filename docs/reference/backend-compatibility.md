# Backend compatibility

> This page is intentionally excluded from the published documentation while
> Metal validation is in progress.

## Feature matrix

| Feature | CUDA | Metal |
| --- | :---: | :---: |
| Automatic selection | yes | yes |
| Attribute graph JIT | generated CUDA/Eigen | generated MSL |
| Named module reuse | generated objects/functions | cached generated MSL functions |
| Fixed JOIN | yes | yes |
| Primitive UNION | yes | yes |
| Dynamic connectivity | yes | yes |
| Symbolic gradient/Hessian | yes | yes |
| Sparse index generation | generated CUDA | generated MSL |
| Coordinate compression | CUDA/Thrust | generated MSL sort/scan/compress |
| Hessian/gradient assembly | generated CUDA | generated MSL atomics |
| Hessian PSD projection | CUDA/Eigen | generated Jacobi EVD |
| Block-Jacobi PCG | generated CUDA/C++ | generated MSL + compiled C++ driver |
| Discrete CCD | generated CUDA/GIPC | generated hashed-grid MSL |
| Continuous CCD step | CUDA ACCD | generated MSL conservative advancement |
| Primary real dtype | float64 | float32 |

## Matrix operations

Generated Metal attribute programs support the public YASPS matrix surface:
matrix multiplication, transpose, determinant, inverse, dot/cross/norm,
row/column access, and SPD projection for statically known dimensions.

The current regression and scene suite exercises 1x1 through 3x3 inverse and
determinant work, local projection blocks including 4, 6, 9, and 12
dimensions, and a batched 12x12 EVD path. Larger statically sized projections
are subject to Metal thread-local/threadgroup resource limits and require a
focused test before use.

## Array compatibility

The Metal facade implements allocation, `*_like`, copying, live slices,
reshape/ravel, dtype conversion, arithmetic, comparisons, and GPU
sum/min/max. It intentionally does not emulate raw CUDA addresses.

Small host reads remain where control flow requires a size, convergence
status, pair count, or line-search scalar. Bulk computation, indexing,
assembly, solving, and CCD remain GPU operations.

## Numerical expectations

Metal results must be judged with float32 tolerances rather than CUDA
double-precision machine epsilon. Useful validation includes:

- finite-difference gradients/Hessians at \(10^{-4}\)-scale tolerances;
- local matrix reconstruction and eigenvalue checks;
- residual and energy decrease rather than bitwise directions;
- contact-set validity and collision-free accepted steps; and
- multi-frame trajectories with no NaN, overflow, or repeated frozen state.

The exact material models and solver settings can remain shared while
backend-specific convergence behavior differs at the float32 residual floor.

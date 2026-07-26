# Backend compatibility

## Feature matrix

| Feature | CUDA | Metal |
| --- | :---: | :---: |
| Automatic selection | yes | yes |
| Attribute graph computation | generated CUDA/Eigen | MLX/Metal |
| Fixed JOIN | yes | yes |
| Primitive UNION | yes | yes |
| Dynamic connectivity | yes | yes |
| Symbolic gradient/Hessian | yes | yes |
| Sparse coordinate compression | CUDA/Thrust | MLX/Metal |
| Sparse Hessian assembly | generated CUDA | custom Metal atomics |
| Hessian PSD projection | CUDA/Eigen | custom Metal Jacobi EVD |
| Block-Jacobi PCG | generated CUDA | custom Metal sparse kernels |
| CCD broad/narrow phase | CUDA LBVH | Metal AABB/Morton traversal |
| Continuous collision step | CUDA ACCD | Metal conservative advancement |
| Primary real dtype | float64 | float32 |

## Metal matrix support

Attribute evaluation supports:

- determinant: 1×1, 2×2, 3×3;
- inverse: 1×1, 2×2, 3×3;
- ordinary matrix multiplication for compatible static shapes.

Symmetric Jacobi eigendecomposition/Hessian projection currently supports
local square sizes:

```text
1, 2, 3, 4, 6, 9, 12
```

These cover the degrees-of-freedom blocks and local Hessians used by the
repository's examples, including the 12×12 tetrahedral and contact terms.

## Array compatibility

The backend facade implements the PyCUDA-like operations YASPS uses:

- allocation and `*_like`;
- `.get()`, `.set()`, `.fill()`, `.copy()`;
- slicing with live parent updates;
- reshape/ravel/astype;
- arithmetic and comparisons;
- sum/min/max.

Code that requires a raw CUDA pointer is not portable. Metal deliberately
raises for `.gpudata` and `.ptr`.

## Numerical expectations

Use backend-appropriate tolerances. Float32 Metal should not be compared to
CUDA double at machine epsilon. For local matrix and solver tests, tolerances
around \(10^{-5}\)–\(10^{-4}\) are typical; model-scale validation should
compare energies, trajectories, and constraint safety.

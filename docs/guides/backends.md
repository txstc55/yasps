# CUDA and Metal backends

YASPS exposes one symbolic scene interface and specializes numerical
execution by backend.

## Auto detection

At import, YASPS:

1. honors `YASPS_BACKEND=cuda|metal` when set;
2. otherwise probes for a usable CUDA device/PyCUDA;
3. otherwise probes for Apple silicon MLX Metal;
4. raises if neither backend is usable.

```python
from yasps.backend import backend_info, backend_name, is_cuda, is_metal
```

## Shared high-level specialization

Backend branches live beside the existing CUDA path in the corresponding
shared file:

| Shared component | CUDA path | Metal specialization |
| --- | --- | --- |
| `attribute.pyx` | generated CUDA/Eigen evaluation | MLX graph evaluator |
| `gradientIndicesKernel.pyx` | generated CUDA/Thrust | JOIN/UNION Metal indexing |
| `coordinateCompressionKernel.pyx` | CUDA sorting/compression | MLX/Metal compression |
| `hessian.pyx` | generated Hessian kernels | Metal atomic assembly |
| `minimizer.pyx` | CUDA block inverse | Metal eigensolver block inverse |
| `solver.pyx` | generated CUDA CG | Metal block-sparse PCG |
| `examples/ccd/ccd.py` | `CudaCCD` | `MetalCCD` |

Low-level Metal kernels are grouped under `yasps/yasps/metal/` so the Jacobi
eigensolver, sparse SpMV, assembly, and CCD code form one dedicated library
rather than being duplicated across public interface files.

## Precision

| Backend | Real dtype | Rationale |
| --- | --- | --- |
| CUDA | float64 | Existing generated Eigen/CUDA implementation |
| Metal | float32 | Supported and accurate enough for IPC/PCG/eigensolve |

Metal maps incoming NumPy float64 data to float32. Integer connectivity keeps
the required integer dtype.

Float16 is not used for core simulation. Although the request permits it,
half precision is unsafe for `dhat` values around \(10^{-6}\), determinants,
Jacobi rotations, sparse accumulation, and PCG residuals.

## Metal linear algebra

The dedicated Metal library provides:

- symmetric Jacobi eigendecomposition for local matrix sizes used by YASPS;
- Hessian eigenvalue projection;
- 1×1, 2×2, and 3×3 determinant/inverse evaluation;
- diagonal block spectral inverse;
- upper-triangular block-sparse symmetric SpMV;
- block-Jacobi PCG.

The eigensolver is a custom MSL kernel because MLX's high-level
`linalg.eigh` path is not a GPU Metal implementation for this use.

## Device array facade

`yasps.backend.gpuarray` presents the subset of PyCUDA's `GPUArray` API used
by YASPS:

```python
from yasps.backend import gpuarray

x = gpuarray.to_gpu(host_values)
y = gpuarray.zeros_like(x)
maximum = gpuarray.max(abs(x))
```

Metal arrays preserve live slice/view behavior expected by cached solution
segments even though MLX slicing is functionally copy-oriented.

Raw CUDA pointers (`gpudata`, `ptr`) intentionally raise on Metal.

## Synchronization

```python
from yasps.backend import synchronize

synchronize()
```

Use synchronization for timing boundaries or before interacting with an
external consumer. `.get()` and scalar `.item()` also synchronize the needed
value.

## Performance notes

- Reuse scenes and static sparsity structures.
- Avoid `.get()` in inner numerical loops.
- Keep dynamic contact host transfer limited to compact index topology.
- Let the CCD retry capacity rather than choosing an enormous default.
- Benchmark after warm-up; Metal compiles custom kernels lazily.

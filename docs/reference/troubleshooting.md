# Troubleshooting

## CUDA/PyCUDA is unavailable

Verify that PyCUDA can initialize and see a device.

## A Cython edit has no effect

Rebuild:

```bash
python -m pip install -e ./yasps --no-build-isolation
```

## `module object is not callable`

When running from a repository layout that shadows the installed package, use
explicit class imports:

```python
from yasps.scene import scene
from yasps.attribute import attribute
```

Internal factories use explicit class imports as well.

## Shape/reshape errors after dynamic contact changes

Update the dynamic primitive count before connectivity:

```python
pairs.updateNumInstances(indices.shape[0])
if indices.shape[0]:
  pair_to_vertex.updateConnectivity(indices)
```

Make sure the index width matches connectivity dimension.

## The returned direction increases energy

YASPS returns the solution of \(H\Delta x=g\). Apply `x - delta`, not
`x + delta`.

Also verify that every energy is scalar and that constants/rest data are
created with `addConstant()` rather than as optimization targets.

## A matrix energy is wrong only on nonsymmetric inputs

Test `inverse @ matrix` and determinant gradients on a nonsymmetric matrix.
Symmetric matrices can hide an accidental transpose in an inverse/cofactor
implementation.

## CUDA kernel compilation fails

Inspect the first failing `nvcc` command. Common causes:

- hard-coded architecture does not match the GPU;
- Eigen include path differs;
- CUDA runtime libraries are not on the link path;
- the checkout predates the bundled GIPC CCD helper headers;
- stale `.yasps_tmp` objects were built with another toolkit.

## Performance numbers vary on the first run

The CUDA backend performs one-time work:

- symbolic differentiation and sparsity setup;
- generated CUDA compilation;
- allocator/cache warm-up.

Measure steady-state iterations separately from construction and first use.

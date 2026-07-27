---
title: Getting started
description: Install the CUDA implementation of YASPS and run a small symbolic minimization.
permalink: /getting-started/
---

<p class="eyebrow">Start here</p>

# Getting started

This documentation describes the CUDA implementation on the repository's `main` branch. YASPS is research software: expect a source build, an NVIDIA development environment, and JIT compilation the first time a new symbolic computation is used.

## Requirements

The current code assumes:

- Linux or a Linux-like CUDA development environment;
- an NVIDIA GPU, driver, and CUDA Toolkit with `nvcc`;
- Python, Cython, NumPy, and PyCUDA;
- a C++17 compiler;
- Eigen 3 headers available at `/usr/include/eigen3`;
- support for the `sm_89` CUDA target used by the generated build commands.

> The `sm_89` target is currently hard-coded in several JIT paths. On a different GPU architecture, update the `-arch=sm_89` flags in the kernel generators before expecting compilation to work.

## Install from the repository

Clone the project, enter its Python package directory, and install it:

```bash
git clone https://github.com/txstc55/yasps.git
cd yasps/yasps

python -m pip install --upgrade pip setuptools wheel Cython numpy pycuda
python -m pip install .
```

On Debian or Ubuntu, Eigen and the host build tools are normally installed with:

```bash
sudo apt-get install build-essential python3-dev libeigen3-dev
```

The repository also contains `yasps/install.sh`, but its final step invokes a maintainer-specific notification command and expects a local `fcm_token`. For a normal installation, use the direct Python commands above.

## Verify CUDA and the package

```bash
python - <<'PY'
import pycuda.autoinit
import pycuda.driver as cuda
from yasps import scene

print("CUDA device:", cuda.Device(0).name())
print("YASPS import: OK")
PY
```

Importing YASPS initializes PyCUDA, so this check must run where a CUDA device is visible.

## A complete quadratic example

This model creates four scalar degrees of freedom and minimizes

```text
E(x) = ½ Σᵢ (xᵢ − xᵢ*)²
```

```python
import numpy as np
from yasps import scene

model = scene("quadratic_demo")
mesh = model.addMesh("model")
dofs = mesh.addPrimitive("dofs", numInstances=4)

x = dofs.addAttribute("x", rows=1, cols=1)
x.updateValue(np.array([4.0, -2.0, 7.0, 1.0], dtype=np.float64))

target = dofs.addConstant("target", rows=1, cols=1)
target.updateValue(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64))

dx = x - target
quadratic_expression = 0.5 * dx * dx
quadratic = dofs.addAttribute(
    "quadratic",
    computed_attribute=quadratic_expression,
)

model.addEnergy(quadratic, projection_method=2)
model.addMinimizeTarget([x])

# YASPS solves H * direction = gradient.
direction = model.minimizeEnergy(tolerance=1e-8)[0]
x.updateValue(x.value - direction, deepCopy=True)

print(x.value.get())
```

The important sequence is:

1. Build a scene hierarchy.
2. Add data attributes and constants.
3. Build an unnamed symbolic expression.
4. Bind the expression to a name with `addAttribute`.
5. Register the scalar per-instance energy.
6. Register the data attributes to minimize against.
7. Ask for a solve and apply the returned direction.

The first solve generates and compiles derivative, assembly, and solver kernels. Later solves reuse the generated code as long as the symbolic structure and relevant sparse layout remain compatible.

## Working with GPU values

Attribute values are flattened `pycuda.gpuarray.GPUArray` objects:

```python
gpu_values = x.value
host_values = x.value.get()
```

`updateValue` accepts NumPy arrays, Python values convertible to NumPy, or PyCUDA arrays:

```python
x.updateValue(host_array)
x.updateValue(other_gpu_array)                  # may reuse the supplied GPU storage
x.updateValue(other_gpu_array, deepCopy=True)   # copy into owned storage
```

Call `expression.compute()` to materialize an otherwise symbolic expression:

```python
values = quadratic_expression.compute().value.get()
```

## Where to go next

- Read the [mental model]({{ '/concepts/' | relative_url }}) before combining attributes from different primitives.
- Learn the [attribute expression syntax]({{ '/attributes/' | relative_url }}).
- Use [Connectivity and JOIN]({{ '/join/' | relative_url }}) for mesh topology.
- Use [Primitive unions]({{ '/union/' | relative_url }}) for mixed parameterizations.
- Follow [Energies and minimization]({{ '/optimization/' | relative_url }}) for Newton and contact loops.
- Put the entire frontend together in the [five-bunny mixed-separation walkthrough]({{ '/tutorials/mixed-separation/' | relative_url }}).

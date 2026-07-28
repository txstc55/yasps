---
title: Troubleshooting
description: Diagnose installation, JIT compilation, topology, GPU buffer, and solver problems.
permalink: /troubleshooting/
---

<p class="eyebrow">Common failure modes</p>

# Troubleshooting

## Installation and JIT

### `nvcc` or CUDA headers are not found

Confirm the CUDA toolkit is installed and `nvcc --version` works in the same environment that runs Python. YASPS compiles generated CUDA C++ at runtime, so having only a CUDA runtime library is insufficient.

### Eigen headers are not found

The build and generated commands expect Eigen under `/usr/include/eigen3`. Install Eigen there or update the include paths in `yasps/setup.py` and the kernel/code-generator compile commands.

### `nvcc fatal: Unsupported gpu architecture 'compute_89'`

The current source hard-codes `sm_89` in JIT compile commands. Use a toolkit that supports it, or change every `-arch=sm_89` occurrence to an architecture supported by your GPU and CUDA toolkit, then remove stale generated artifacts.

```bash
rg --fixed-strings -- '-arch=sm_89' yasps/yasps
```

### The first call is much slower

That is expected for a new symbolic graph. The first `compute()` or minimization generates source and invokes the compiler. Later calls reuse caches in `.yasps_tmp` and `.yasps_constant`.

If generated code is stale after changing code-generation rules, stop all processes using the project and remove only those two cache directories inside `yasps/`. They are rebuildable, but the next run will compile again.

### Import creates or selects a CUDA context unexpectedly

Several runtime modules import `pycuda.autoinit`, which creates a context on import. Set the desired CUDA device before importing YASPS, or initialize and manage a compatible PyCUDA context explicitly in the application.

## Scene construction

### A name is rejected or collides

Scene, mesh, primitive, connectivity, and attribute names become Python members and generated identifiers. Use valid Python identifiers and unique names within their owner. Scene names are registered globally in the process, so constructing a second live scene with the same name can also fail.

### “No connection to the attribute”

The operands have incompatible lineage. Gather through a [connectivity and JOIN]({{ '/join/' | relative_url }}?v={{ site.time | date: '%s' }}), or stack shape-compatible sources with [UNION]({{ '/union/' | relative_url }}?v={{ site.time | date: '%s' }}).

### JOIN has the wrong shape

For fixed arity `k`, a source with per-instance shape `r × c` becomes `k × (rc)`. Index its rows or call `resize` before matrix algebra if you need another shape with the same number of entries.

### Variable connectivity asks for an operation

For `dimension=0`, pass `operation="SUM"` or `operation="AVERAGE"` to `primitive.addAttribute`. Do not place that reduced JOIN on a differentiated energy path; variable-arity Hessian paths are not implemented.

## Values and buffers

### `updateValue` reports the wrong size

Flattened input length must equal `correspondance.numInstances * rows * cols`. YASPS uses double-precision (`np.float64`) numerical storage on the CUDA implementation.

### Updating from another GPU expression changes later

The source and destination may alias or the source may be backed by a reusable output buffer. Pass `deepCopy=True` when applying a solution or preserving a computed state:

```python
position.updateValue(new_position_gpu, deepCopy=True)
```

### A dynamic term still uses old pairs

Update the dynamic primitive's count and connectivity before every assembly. For an empty set, update the count to zero and skip `updateConnectivity` on an empty CPU array.

## Differentiation and minimization

### “energy attribute must have a name”

Bind the expression to its primitive before registering it:

```python
energy = elements.addAttribute("elasticity", computed_attribute=expression)
world.addEnergy(energy)
```

### A minimization target is rejected

Targets must be unique, nondynamic `DATA` attributes. A computed attribute, JOIN result, UNION result, or constant cannot be a direct target. Minimize the underlying data parameters instead.

### A local energy target is missing

Every attribute passed in `addEnergy(..., targets=[...])` must also appear in `addMinimizeTarget([...])`.

### The update moves uphill

YASPS solves `H Δx = g`, not `H Δx = -g`. Apply `x - alpha * direction`.

### The solver does not converge

Check, in this order:

1. the update sign and target ordering;
2. unconstrained rigid modes or otherwise singular targets;
3. empty/invalid dynamic connectivity;
4. local Hessian projection choice;
5. scale differences between energy terms;
6. CG tolerance and iteration limit.

The minimizer prints a warning and returns the best solution found when the solver reports nonconvergence.

### `gradient_only=True` fails

The flag is exposed but not implemented in the Hessian-based minimizer. Register a normal energy, or use symbolic `autodiff().diff` for a low-level local derivative.

## Getting generated-source evidence

Generated `.cu`, C++, shared-library, and cache filenames contain hashes derived from the symbolic graph. When reporting a JIT failure, keep:

- the complete compiler command and stderr;
- the generated source file named in the error;
- the scene/attribute names;
- CUDA toolkit, driver, GPU model, Python, and compiler versions.

This usually makes a code-generation issue reproducible without the full simulation dataset.

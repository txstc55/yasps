---
title: Direct minimizer use
description: Control symbolic differentiation, numerical assembly, energy selection, and solution beneath the scene façade.
permalink: /advanced/minimizer/
---

<p class="eyebrow">Advanced API</p>

# Direct minimizer use

Every `scene` owns one `minimizer`. The scene methods forward to it, so direct
use does not select a different algorithm—it exposes the lifecycle that the
scene normally performs in one call.

Use this layer to isolate code-generation time, assemble without solving,
inspect the active Hessian, switch energy subsets, or connect an assembled
system to a custom solver.

## Lifecycle at a glance

```text
addEnergy / addEnergies
          │
          ▼
       addWrt                 target validation and vector layout
          │
          ▼
generateHessianAndGradient    symbolic path discovery and diff2
          │
          ▼
   computeNumericValue        sparse setup/refresh, H/g assembly,
          │                   inverse diagonal-block construction
          ▼
solver.computeSolution        solve the already assembled system
```

`generateHessianAndGradient` is lazy if omitted. In contrast,
`minimizer.computeSolution` always calls numerical assembly before solving.

## Construct a minimizer

```python
from yasps import minimizer

engine = minimizer()
```

The constructor takes no parameters. It creates an empty energy registry, an
empty target layout, a reusable low-level solver, and empty numerical buffers.
A scene-created minimizer behaves identically:

```python
engine = world.minimizer
```

## Add one energy

```python
engine.addEnergy(
  e,
  targets=[],
  projection_method=1,
  save_intermediate=False,
  gradient_only=False,
  dynamic_instances=False,
  separate_hessian_jacobian=False,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `e` | scalar named `attribute` | Energy whose instances are summed into the objective. An unnamed attribute is rejected. |
| `targets` | `list[attribute] = []` | Local differentiation targets for this energy. All must later occur in `addWrt`. |
| `projection_method` | `int = 1` | Local Hessian projection: `-1` none, `0` no-op, `1` absolute eigenvalues, `2` clamp negative eigenvalues. |
| `save_intermediate` | `bool = False` | Lets derivative generation retain selected intermediate attributes for reuse. |
| `gradient_only` | `bool = False` | Not implemented by the Hessian minimizer; `True` raises `NotImplementedError`. |
| `dynamic_instances` | `bool = False` | Stores this request in the dynamic list and enables later structural refresh. |
| `separate_hessian_jacobian` | `bool = False` | Separates the local energy-Hessian stage from the outer Jacobian stage. |

Adding a request marks symbolic differentiation dirty, clears the cached active
Hessian sum, and preserves the current target list. An equivalent duplicate
request raises `ValueError`.

## Add several default energies

```python
engine.addEnergies([elasticity, inertia, contact])
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `energies` | `list[attribute]` | Calls `addEnergy(item)` for each item with every option at its default. |

Use separate `addEnergy` calls when any term needs local targets, projection
selection, dynamic structure, or generation options.

## Define the global target layout

```python
engine.addWrt([
  position,
  affine_translation,
  affine_transform,
])
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `wrt` | `list[attribute]` | Ordered global unknowns. This order controls every gradient and solution segment. |

Validation rejects duplicate targets, non-`DATA` attributes, dynamic target
attributes, and any list missing a local target required by an energy request.

Calling `addWrt` rebuilds the empty gradient layout, resets solver and initial
guess buffers, and marks all differentiation dirty. Avoid calling it inside an
iteration unless the global unknown layout really changed.

## Generate symbolic derivatives

```python
engine.generateHessianAndGradient()
```

The method takes no parameters and returns `None`. For each request it:

1. discovers paths from the scalar energy to the global and local targets;
2. generates first- and second-order symbolic expressions;
3. creates sparse-index and placement metadata;
4. stores a static or dynamic `hessian` contribution.

It prints the autodiff duration. If no request or target changed since the last
generation, it returns immediately. Calling `computeNumericValue` or a solve
also triggers this step lazily, so the explicit call is mainly useful for
profiling and warm-up.

## Assemble numerical values

```python
active_hessian = engine.computeNumericValue()
```

The method takes no parameters. It requires a prior `addWrt`; otherwise it
raises `ValueError`.

The assembly sequence is:

1. lazily differentiate dirty requests;
2. combine every nonignored static and dynamic Hessian contribution;
3. initialize or refresh sparse coordinates;
4. launch generated kernels to fill block values, the scalar diagonal, and the
   global gradient;
5. build the diagonal-block inverse kernel on first use;
6. invert the current dense diagonal blocks into
   `active_hessian.diagonal_blocks_inverse`.

The return value is the active `hessian`. If every request is ignored, it
zeroes the global gradient and returns `None`.

### Static and dynamic assembly

Static contributions reuse their sparse coordinates once initialized. Dynamic
contributions keep separate block buffers and refresh coordinate compression
as the energy correspondence changes. Both are concatenated in the active
Hessian protocol seen by the generated solver kernel.

`dynamic_instances=True` belongs on the energy request; minimization targets
themselves remain static.

## Solve through the minimizer

```python
directions = engine.computeSolution(
  tolerance=1e-3,
  maxIterations=20000,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `tolerance` | `float = 1e-3` | Residual tolerance forwarded to PCG. |
| `maxIterations` | `int = 20000` | PCG iteration limit. |

This method calls `computeHessianAndGradient`, which first calls
`computeNumericValue` and then solves. Therefore:

```python
H = engine.computeNumericValue()
directions = engine.computeSolution()
```

performs two numerical assemblies. There is no public minimizer method named
“solve the already assembled active Hessian.” For that workflow, call the
low-level `solver.computeSolution` directly with `H`, as documented in
[Hessian and solver]({{ '/advanced/hessian-solver/' | relative_url }}?v={{ site.time | date: '%s' }}).

The return value is the target-aligned `solutionSegments` list. If the solver
returns a negative status, the method prints a warning and returns its best
available direction.

## Assemble and return solver status

```python
status = engine.computeHessianAndGradient(
  tolerance=1e-3,
  maxIterations=20000,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `tolerance` | `float = 1e-3` | PCG residual tolerance. |
| `maxIterations` | `int = 20000` | PCG iteration limit. |

Despite its name, this method both assembles and solves. It returns the raw
solver status rather than solution segments. A negative value denotes
nonconvergence; do not depend on undocumented exact status numbers.

Use `computeNumericValue()` for assembly only and `computeSolution()` when
target-segment output is more convenient than the raw status.

## Solve an already assembled Hessian

The minimizer prepares the preconditioner, after which the low-level solver can
consume the same numerical state without a second assembly:

```python
import pycuda.gpuarray as gpuarray
from yasps import solver

H = engine.computeNumericValue()

if H is not None:
  initial_guess = gpuarray.zeros_like(engine.gradient)
  pcg = solver()
  status = pcg.computeSolution(
    H,
    engine.wrt,
    H.gradient,
    initial_guess,
    tolerance=1e-3,
    maxIterations=20000,
  )
  flat_direction = pcg.solution
```

Keep `H`, `H.gradient`, the target list, and the initial guess layout together.
The solver protocol assumes the inverse diagonal blocks already correspond to
the current numerical Hessian.

## Result buffers and lifetimes

```python
engine.gradient
engine.gradientSegments
engine.solutionSegments
engine.diagonal
engine.wrt
```

| Property | Contents | Lifetime |
| --- | --- | --- |
| `gradient` | Flattened global gradient `GPUArray`. | Refilled by later assemblies. |
| `gradientSegments` | Per-target views into `gradient`. | Same storage as `gradient`. |
| `solutionSegments` | Per-target views into the low-level solver solution. | Refilled by later solves; rebuilt when layout changes. |
| `diagonal` | Flattened scalar Hessian diagonal. | Refilled by later assemblies. |
| `wrt` | Copy of the ordered target list. | Replaced by `addWrt`. |

Each segment length is:

```text
target.correspondance.numInstances × target.rows × target.cols
```

These are views, not snapshots. Use `.copy()` or an explicit deep copy when an
iteration must retain an old gradient or direction.

## Inspect registered requests

```python
static_requests = engine.energies
dynamic_requests = engine.energiesDynamic

for request in static_requests + dynamic_requests:
  print(
    request.energy.fullName,
    request.targets,
    request.projection_method,
    request.dynamic_instances,
  )
```

| Property | Contents |
| --- | --- |
| `energies` | Static `energyRequest` objects. |
| `energiesDynamic` | Dynamic `energyRequest` objects. |

`energyRequest` is an implementation object exposed through these properties,
but it is not exported at the package root. Treat its storage details as
inspection-oriented rather than a stable construction API.

## Ignore selected energies

```python
engine.ignoreEnergies([contact])
without_contact = engine.computeSolution()

engine.ignoreEnergies([])
with_everything = engine.computeSolution()
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `energies` | `list[attribute]` | Replaces the ignored hash list. Pass `[]` to reactivate all requests. |

The active Hessian sum is invalidated, but the differentiated per-energy
Hessians remain cached. This is the intended way to toggle terms without
repeating symbolic differentiation.

## Evaluate the objective

```python
value = engine.computeTotalEnergy()
```

This no-argument method computes every active energy, reduces its instances on
the GPU, transfers one scalar per request to the host, and returns their Python
`float` sum. A dynamic request with zero instances is skipped.

Every `.get()` is a host synchronization. Call this for line search,
acceptance, convergence, or diagnostics when that synchronization is useful;
it is not required for Hessian and gradient assembly.

## Reuse and invalidation rules

| Change | What is invalidated |
| --- | --- |
| `addEnergy` or `addEnergies` | Symbolic derivatives and active Hessian sum. |
| `addWrt` | Gradient layout, solver buffers, initial guess, derivatives, and active sum. |
| `ignoreEnergies` | Active Hessian sum only. |
| Dynamic primitive count | Dynamic sparse coordinates on the next assembly. |
| Attribute values only | Numerical blocks and gradient are recomputed; symbolic paths stay valid. |

For a steady topology, construct and differentiate once, then update data
buffers and repeatedly assemble/solve. That is the path that preserves JIT
kernel and allocation reuse.

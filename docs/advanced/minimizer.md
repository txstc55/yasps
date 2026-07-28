---
title: Direct minimizer use
description: Use the minimizer without the scene convenience methods and control assembly separately from solving.
permalink: /advanced/minimizer/
---

<p class="eyebrow">Advanced API</p>

# Direct minimizer use

Every `scene` owns a `minimizer`, and the scene's optimization methods are thin forwards to it. Advanced applications may use `minimizer` directly to separate symbolic generation, numerical assembly, energy selection, and solution.

## Build the request

```python
from yasps import minimizer

engine = minimizer()

engine.addEnergy(
  elasticity,
  targets=[position],
  projection_method=2,
)
engine.addEnergy(
  inertia,
  targets=[position],
  projection_method=-1,
)

engine.addWrt([position])
```

`addEnergy` stores a symbolic request. `addWrt` defines the global vector layout and validates that every energy-local target is present. The same restrictions as `scene.addMinimizeTarget` apply: targets must be unique, static `DATA` attributes.

Use `addEnergies([e0, e1, ...])` only when all terms use default registration options.

## Separate generation, assembly, and solution

```python
# Symbolic path discovery and derivative construction.
engine.generateHessianAndGradient()

# Numeric sparse-index setup/refresh and H/g assembly.
active_hessian = engine.computeNumericValue()

# PCG solve; returns target-aligned GPUArray views.
directions = engine.computeSolution(
  tolerance=1e-3,
  maxIterations=20000,
)
```

`generateHessianAndGradient()` is optional because later operations trigger it lazily. It is useful for isolating symbolic generation time from numerical execution time.

`computeNumericValue()` returns the active `hessian`, or `None` if every term is ignored. It also fills the global gradient and computes the inverse diagonal blocks used by the preconditioner.

`computeSolution()` performs assembly again before solving. If you already called `computeNumericValue()` only for inspection, expect another numerical assembly when you call it.

The method named `computeHessianAndGradient()` currently does more than its name suggests: it assembles and invokes the linear solver, then returns the solver status code. Prefer `computeNumericValue()` when you want assembly without a solve.

## Results and layout

```python
engine.gradient          # flattened GPUArray
engine.gradientSegments  # one GPUArray view per target
engine.solutionSegments  # one GPUArray view per target
engine.diagonal          # assembled flattened Hessian diagonal
engine.wrt               # target list in global order
```

The segment order always follows `addWrt`. Each segment length is:

```text
target.correspondance.numInstances * target.rows * target.cols
```

All segment objects are views into reusable global buffers. Copy them if they must survive a later assembly or solve.

## Select active energies

```python
engine.ignoreEnergies([contact])
without_contact = engine.computeSolution()

engine.ignoreEnergies([])
with_everything = engine.computeSolution()
```

Ignoring an energy invalidates the cached active sum but retains its differentiated term, so toggling does not rebuild the symbolic derivatives.

Inspect registration requests with:

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

`energyRequest` is an implementation type returned by these properties; it is not exported from `yasps.__init__`.

## Total energy

```python
value = engine.computeTotalEnergy()
```

This materializes every nonignored energy, sums its instances on the GPU, and transfers one scalar per term to the host. A dynamic term with zero instances is skipped.

Because this introduces synchronization and host transfers, use it for line search, convergence policy, or diagnostics rather than on every internal operation.

## Why use the scene façade by default

`scene.addEnergy`, `scene.addMinimizeTarget`, `scene.minimizeEnergy`, and the scene result properties preserve exactly the same minimizer behavior with less plumbing. Direct use is most useful when:

- one application owns several independent minimizers;
- symbolic generation must be benchmarked separately;
- the active Hessian needs to be inspected before solving;
- energy subsets are switched frequently;
- a custom linear-solve experiment needs the assembled YASPS structures.

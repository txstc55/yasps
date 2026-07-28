---
title: Energies and minimization
description: Register scalar energies, select targets, assemble derivatives, and integrate YASPS into a Newton loop.
permalink: /optimization/
next_url: /dynamic-scenes/
next_label: Dynamic contact topology
---

<p class="eyebrow">Core syntax</p>

# Energies and minimization

YASPS differentiates named scalar attributes, assembles a block-sparse Hessian
and gradient, and solves one linear system. Your application still owns the
outer Newton iteration: collision updates, line search, state acceptance, and
convergence tests.

## The optimization pipeline

The scene API is a thin façade over its `minimizer`:

```text
scene.addEnergy(...)
  └─ minimizer.addEnergy(...)             record symbolic requests

scene.addMinimizeTarget(...)
  ├─ minimizer.addWrt(...)                 define the global vector layout
  └─ minimizer.generateHessianAndGradient() discover paths and build derivatives

scene.minimizeEnergy(...)
  └─ minimizer.computeSolution(...)
     └─ minimizer.computeHessianAndGradient(...)
        ├─ minimizer.computeNumericValue() assemble H and g
        │  ├─ hessian.compute(...)
        │  └─ invert dense diagonal blocks
        └─ solver.computeSolution(...)     solve H dx = g with PCG
```

This distinction matters when profiling. Symbolic differentiation, numerical
assembly, diagonal-block inversion, and PCG are separate costs even though the
high-level call performs the last three together.

## Energy expressions

Each registered energy must be a scalar attribute: `rows == 1` and `cols == 1`.
It may have one value per primitive instance; the global objective is the sum
over those instances and over all active energy requests.

Name the final expression on its correspondence before registering it:

```python
displacement = position - target
quadratic = 0.5 * stiffness * displacement.dot(displacement)
quadratic = points.addAttribute(
  "quadratic",
  computed_attribute=quadratic,
)

world.addEnergy(quadratic)
```

The name forms a stable kernel boundary and supplies the identifier used by the
code generator. Registering an unnamed temporary raises `ValueError`.

## Register an energy

```python
world.addEnergy(
  energy_attribute,
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
| `energy_attribute` | scalar `attribute` | The named per-instance objective term to differentiate and sum. |
| `targets` | `list[attribute] = []` | Optional local subset of the global targets. Use it when this energy depends on only part of the global state. |
| `projection_method` | `int = 1` | Selects the local symmetric-positive-definite projection described below. |
| `save_intermediate` | `bool = False` | Allows the differentiator to materialize selected derivative intermediates for reuse. It trades memory and launches for recomputation. |
| `gradient_only` | `bool = False` | Reserved. Passing `True` currently raises `NotImplementedError`. |
| `dynamic_instances` | `bool = False` | Places the request in the dynamic structural path so sparse indices can be refreshed when the energy primitive's instance count changes. |
| `separate_hessian_jacobian` | `bool = False` | Generates inner energy-Hessian and outer Jacobian stages separately. This is useful when the outer Jacobian has exploitable within-block sparsity. |

Registering the same request twice raises `ValueError`. The duplicate check
includes the request options, not merely the energy object's Python identity.

### Local Hessian projection

| `projection_method` | Local operation |
| --- | --- |
| `-1` | Do not insert a projection operation. |
| `0` | Insert the projection machinery but leave eigenvalues unchanged. |
| `1` | Replace each local eigenvalue with its absolute value. |
| `2` | Clamp negative local eigenvalues to zero. |

Projection acts on a local energy Hessian before sparse assembly. It does not
guarantee that the final global system is nonsingular or well conditioned.
Convex inertia terms commonly use `-1`; nonlinear elasticity and contact terms
often use `1` or `2`.

### Local target subsets

Suppose one scene solves positions and affine body transforms, but each energy
touches only one group:

```python
world.addEnergy(
  soft_elasticity,
  targets=[soft_position],
)
world.addEnergy(
  affine_orthogonality,
  targets=[affine_transform],
)
```

The local list restricts path discovery and derivative generation for that
request. It does not change the global vector layout. Every local target must
later appear in `addMinimizeTarget`.

### Static and dynamic energy requests

Use the default static path when the energy correspondence keeps the same
instance count. Use `dynamic_instances=True` for contact pairs or other
runtime-changing correspondences:

```python
world.addEnergy(
  contact_barrier,
  targets=[position],
  projection_method=2,
  dynamic_instances=True,
)
```

Dynamic mode refreshes sparse coordinates during later assemblies. It does not
make the target itself dynamic: minimization targets must still be static data
attributes.

## Register global targets

After registering all energies, define the global unknown-vector layout:

```python
world.addMinimizeTarget([
  position,
  affine_translation,
  affine_transform,
])
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `target_list` | `list[attribute]` | Ordered differentiable data attributes. The order controls gradient and solution segmentation. |

Every target must:

- be a `DATA` attribute rather than a computed, JOIN, or UNION expression;
- have a fixed instance count;
- appear only once;
- include every attribute named in an energy's local `targets` list.

The flattened length of a target segment is:

```text
target.correspondance.numInstances × target.rows × target.cols
```

Calling `addMinimizeTarget` forwards to `minimizer.addWrt` and then explicitly
generates the symbolic derivatives. Construct topology and energies first so
this expensive generation happens once.

## High-level assembly and solve

```python
directions = world.minimizeEnergy(
  tolerance=1e-3,
  maxIterations=20000,
)
```

| Parameter | Type and default | Effect |
| --- | --- | --- |
| `tolerance` | `float = 1e-3` | Residual tolerance passed to the generated PCG solver. |
| `maxIterations` | `int = 20000` | Maximum PCG iterations for this solve. |

The call assembles the current numerical Hessian and gradient, rebuilds the
dense inverse diagonal blocks used by the preconditioner, and solves:

```text
H dx = g
```

It returns a `list[GPUArray]`, one flattened view per target in registration
order. A negative internal solver status prints a warning, but the method still
returns the best direction found.

YASPS does not negate or apply the direction. A Newton update therefore uses:

```python
directions = world.minimizeEnergy()
position.updateValue(
  position.value - directions[0],
  deepCopy=True,
)
```

The returned segments alias a reusable global solution buffer. Copy a segment
if it must survive another solve.

## A complete outer Newton loop

```python
for newton_iteration in range(max_newton_iterations):
  # Runtime-derived constants and dynamic contact correspondences.
  update_collision_topology()

  # Current H and g are assembled, then H dx = g is solved.
  directions = world.minimizeEnergy(
    tolerance=cg_tolerance,
    maxIterations=cg_iterations,
  )

  # The application chooses a feasible, energy-decreasing step.
  alpha_ccd = compute_collision_free_step(directions)
  alpha = backtracking_line_search(alpha_ccd, directions)

  # Apply x <- x - alpha dx.
  for target, direction in zip(targets, directions):
    target.updateValue(
      target.value - alpha * direction,
      deepCopy=True,
    )

  if converged(world.gradient):
    break
```

This separation lets each simulator choose its IPC barrier policy, feasibility
condition, convergence norm, timestep acceptance rule, and integrator.

## Manual assembly control

`scene.minimizeEnergy` is convenient, but it does not expose an
assembly-only mode. Reach through `world.minimizer` when you need one:

```python
engine = world.minimizer

# Assemble H and g and prepare the block-diagonal preconditioner.
active_hessian = engine.computeNumericValue()

gradient = engine.gradient
gradient_segments = engine.gradientSegments
inverse_blocks = active_hessian.diagonal_blocks_inverse
```

`computeNumericValue()` returns the active `hessian`, or `None` when every
energy is ignored. Calling `engine.computeSolution()` afterwards assembles
again; it does not merely solve the Hessian you just inspected. To solve an
already assembled Hessian, call the low-level `solver` as shown in
[Hessian and solver]({{ '/advanced/hessian-solver/' | relative_url }}?v={{ site.time | date: '%s' }}).

## Inspect the current problem

After an assembly or solve:

```python
engine = world.minimizer

gradient_gpu = world.gradient
gradient_segments = world.gradientSegments
solution_segments = world.solutionSegments
scalar_diagonal = world.diagonal
active_hessian = engine.computeNumericValue()
```

| Value | Meaning |
| --- | --- |
| `gradient` | One flattened global gradient buffer. |
| `gradientSegments` | Views of that gradient split in target order. |
| `solutionSegments` | Views of the latest PCG solution split in target order. |
| `diagonal` | The assembled scalar Hessian diagonal. |
| `active_hessian.diagonal_blocks_inverse` | Dense inverse diagonal blocks actually used by the preconditioner. |

The gradient and solution segments are views into reused buffers. Their Python
objects are convenient handles, not immutable snapshots.

## Select active energies

Temporarily exclude registered terms:

```python
world.ignoreEnergies([contact_energy])
directions_without_contact = world.minimizeEnergy()

world.ignoreEnergies([])
directions_with_everything = world.minimizeEnergy()
```

| Parameter | Type | Effect |
| --- | --- | --- |
| `energies` | `list[attribute]` | Replaces the ignore list with the hashes of these energy attributes. An empty list restores all terms. |

Changing the ignore list invalidates the cached active Hessian sum, but it
retains each request's differentiated Hessian. This makes toggling cheaper than
re-registering energies.

## Evaluate total energy

```python
objective = world.computeTotalEnergy()
```

The method computes every nonignored energy, reduces its instances on the GPU,
and transfers one scalar per energy request to the host. Dynamic requests with
zero instances are skipped.

This is a synchronization point. Use it deliberately for line search,
acceptance, or diagnostics rather than as a free inspection operation inside
every GPU stage.

For all minimizer methods and buffer-lifetime details, continue to
[Direct minimizer use]({{ '/advanced/minimizer/' | relative_url }}?v={{ site.time | date: '%s' }}).

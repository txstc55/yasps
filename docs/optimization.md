---
title: Energies and minimization
description: Register scalar energies, select targets, assemble derivatives, and integrate YASPS into a Newton loop.
permalink: /optimization/
---

<p class="eyebrow">Core syntax</p>

# Energies and minimization

YASPS turns named scalar attributes into sparse gradient and Hessian contributions. It assembles and solves the linear system; the application owns the surrounding optimization or timestep policy.

## Register a scalar energy

An energy expression must be scalar and should be named on its correspondence:

```python
displacement = position - target
quadratic = 0.5 * stiffness * displacement.dot(displacement)
quadratic = points.addAttribute("quadratic", computed_attribute=quadratic)

world.addEnergy(quadratic)
```

Naming is important because the code generator uses named attributes as kernel boundaries and identifiers.

The full registration signature is:

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

| Argument | Meaning |
| --- | --- |
| `targets` | Optional subset of the scene's global minimization targets that this energy depends on |
| `projection_method` | Local Hessian projection policy |
| `save_intermediate` | Materialize selected derivative intermediates for reuse |
| `gradient_only` | Reserved flag; the current minimizer raises `NotImplementedError` for it |
| `dynamic_instances` | Recompute structure for an energy whose primitive instance count changes |
| `separate_hessian_jacobian` | Generate the energy Hessian and outer Jacobian as separate stages |

### Projection methods

| Value | Behavior |
| --- | --- |
| `-1` | Skip insertion of the local SPD projection |
| `0` | Insert a no-op projection |
| `1` | Replace local eigenvalues with their absolute values |
| `2` | Clamp negative local eigenvalues to zero |

The default is `1`. Several examples use `-1` for convex inertia terms and a projected method for nonlinear elasticity or contact. Projection is applied to a local energy Hessian; it is not a promise that every assembled global system is well conditioned.

### Local target subsets

If the scene solves for several parameter groups but an energy depends on only some of them, declare that subset:

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

These local targets control differentiation for the energy while the global gradient and solution layout still follows all targets registered on the minimizer. Every local target must also appear in the global target list.

## Register minimization targets

After all energies have been added:

```python
world.addMinimizeTarget([
    position,
    affine_translation,
    affine_transform,
])
```

Each target must be a differentiable data attribute, and a target may appear only once. Registration triggers symbolic differentiation and preparation of the sparse structures, so do it once after the model topology is constructed.

## Solve

```python
directions = world.minimizeEnergy(
    tolerance=1e-3,
    maxIterations=20000,
)
```

The return value is a list of flattened PyCUDA `GPUArray` views, one per target in registration order. Internally YASPS solves:

$$
H\Delta x=g.
$$

It does not negate or apply the direction. A Newton-style update is therefore:

```python
directions = world.minimizeEnergy()
position.updateValue(
    position.value - directions[0],
    deepCopy=True,
)
```

`deepCopy=True` is useful when the source expression aliases a buffer that YASPS may reuse.

## A complete outer loop

YASPS deliberately leaves line search, continuous collision detection, and state acceptance to the application:

```python
for newton_iteration in range(max_newton_iterations):
    # 1. Update state-derived constants and dynamic collision pairs.
    update_collision_topology()

    # 2. Assemble H and g, then solve H dx = g.
    directions = world.minimizeEnergy(
        tolerance=cg_tolerance,
        maxIterations=cg_iterations,
    )

    # 3. Choose a safe step outside YASPS.
    alpha_ccd = compute_collision_free_step(directions)
    alpha = backtracking_line_search(alpha_ccd, directions)

    # 4. Apply x <- x - alpha dx.
    for target, direction in zip(targets, directions):
        target.updateValue(
            target.value - alpha * direction,
            deepCopy=True,
        )

    if converged(world.gradient):
        break
```

This separation lets a simulation choose its own IPC barrier policy, feasibility condition, convergence norm, and integrator.

## Inspecting the assembled problem

After a numerical assembly or solve:

```python
gradient_gpu = world.gradient
gradient_segments = world.gradientSegments
block_diagonal_inverse = world.diagonal
total_energy = world.computeTotalEnergy()
engine = world.minimizer
```

- `gradient` is the flattened global gradient buffer.
- `gradientSegments` follows the minimization-target layout.
- `diagonal` is the assembled flattened Hessian diagonal. The dense inverse diagonal blocks used by the preconditioner remain on the active `hessian`.
- `computeTotalEnergy()` launches each registered energy and returns a Python float.

Temporarily disable selected energy attributes with:

```python
world.ignoreEnergies([contact_energy])
```

The ignore list affects subsequent assembly and total-energy evaluation. Pass an empty list to restore all registered energies.

For direct control of assembly without the scene façade, continue to [Direct minimizer use]({{ '/advanced/minimizer/' | relative_url }}).

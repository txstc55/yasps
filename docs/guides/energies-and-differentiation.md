# Energies and differentiation

YASPS treats a named scalar attribute as one energy contribution per owner
instance. The scene sums all instances and all registered energy terms.

## Writing an energy

```python
offset = position - target
quadratic = vertices.addAttribute(
  "quadratic_energy",
  computed_attribute=0.5 * mass * offset.dot(offset),
)
```

An energy must:

- be scalar per instance (`rows=1`, `cols=1`);
- have a name;
- have a valid symbolic path to every intended minimization target.

## Registering energy

```python
simulation.addEnergy(quadratic, projection_method=2)
```

For a runtime-sized primitive:

```python
simulation.addEnergy(
  contact_energy,
  dynamic_instances=True,
  projection_method=2,
)
```

The optional `targets=[...]` argument records required minimization targets
for that energy. If used, every recorded target must later appear in
`addMinimizeTarget()`.

## Registering targets triggers differentiation

```python
simulation.addMinimizeTarget([position])
```

This call:

1. symbolically differentiates every registered energy;
2. propagates derivatives through JOIN and UNION paths;
3. computes local gradient/Hessian layouts;
4. prepares global sparse block coordinates.

For static topology, numerical solves reuse the structure. Dynamic energy
terms recompute their active sparse coordinates.

## Automatic symbolic differentiation

Most users should differentiate through the scene workflow:

```python
simulation.addEnergy(energy)
simulation.addMinimizeTarget([position])
delta = simulation.minimizeEnergy()
```

The generated graph contains symbolic local gradients and Hessians, which
YASPS turns into generated GPU kernels.

## Direct differentiation

For inspection or focused tests:

```python
from yasps.autodiff import autodiff

gradient_expression = autodiff().diff(energy_expression, position)
gradient_value = gradient_expression.compute().value.get()
```

Direct differentiation is a lower-level tool. It does not replace scene
registration, JOIN/UNION path propagation, sparse assembly, or target
segmentation.

## Hessian projection

`projection_method` controls local Hessian stabilization:

| Value | Behavior |
| ---: | --- |
| `-1` | Skip local Hessian projection work |
| `0` | Keep eigenvalues unchanged |
| `1` | Replace each eigenvalue by its absolute value |
| `2` | Clamp negative eigenvalues to zero |

Examples commonly use:

- `-1` for known-convex inertia;
- `1` for nonconvex elasticity;
- `2` for IPC contact and friction.

Projection uses the Eigen/CUDA path.

## Multiple energies

```python
simulation.addEnergy(elastic)
simulation.addEnergy(inertia, projection_method=-1)
simulation.addEnergy(contact, dynamic_instances=True, projection_method=2)
```

Overlapping terms accumulate into the same global gradient, diagonal blocks,
and upper-triangular sparse blocks.

## Ignoring terms temporarily

```python
simulation.ignoreEnergies([contact])
```

This changes the active numerical system without deleting the registered
term. Passing an empty list re-enables all terms:

```python
simulation.ignoreEnergies([])
```

## Differentiation boundaries and caveats

- Constants produce zero derivatives.
- `asConstant()` stops derivatives through its child.
- Fixed-arity JOIN and primitive UNION are first-class differentiated paths.
- Variable-arity `SUM`/`AVERAGE` work for computation but are not a
  general-purpose differentiated energy path.
- `gradient_only=True` exists in the signature but is not a supported
  user-facing minimization mode in the current implementation.
- The first-order helper path is incomplete compared with the normal
  second-order scene workflow.

Test new operator combinations with finite differences, especially matrix
inverse/determinant expressions and branch boundaries.

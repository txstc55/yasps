# Scene API

Import:

```python
from yasps.scene import scene
```

## Construction

```python
simulation = scene(name)
```

`name` must be a non-empty, process-unique identifier.

## Hierarchy methods

### `addMesh(name)`

Creates and returns a mesh owned by the scene.

### `addAttribute(name, computed_attribute=None, rows=1, cols=1)`

Creates scene-level mutable data or names a scene-level computed expression.

### `addConstant(name, rows=1, cols=1)`

Creates mutable scene-level data excluded from differentiation.

### `__getitem__(key)`

Looks up one named scene attribute. A tuple of names constructs one flattened
symbolic array.

## Energy methods

### `addEnergy(...)`

```python
simulation.addEnergy(
  energy,
  targets=[],
  projection_method=1,
  save_intermediate=False,
  gradient_only=False,
  dynamic_instances=False,
  separate_hessian_jacobian=False,
)
```

Important arguments:

| Argument | Meaning |
| --- | --- |
| `energy` | Named scalar attribute to sum |
| `targets` | Optional required target subset |
| `projection_method` | `-1`, `0`, `1`, or `2`; see projection guide |
| `dynamic_instances` | Rebuild active sparse topology numerically |
| `separate_hessian_jacobian` | Advanced generated-kernel strategy |

`save_intermediate` and `separate_hessian_jacobian` are advanced performance
controls. `gradient_only` is present but not a complete public solve mode.

### `addMinimizeTarget(targets)`

Registers an ordered list of mutable data attributes, runs symbolic
differentiation, and prepares Hessian/gradient structures.

Every target required by `addEnergy(..., targets=...)` must be included.
Targets must be unique by full name.

### `minimizeEnergy(tolerance=1e-3, maxIterations=20000)`

Computes current numerical derivatives, solves \(H\Delta x=g\), and returns
one flattened device-array segment per target.

### `computeTotalEnergy()`

Computes and host-reduces all active registered energies.

### `ignoreEnergies(energies)`

Sets the list of terms omitted from subsequent assembly and energy sums. Pass
`[]` to restore all terms.

## Diagnostic properties

| Property | Meaning |
| --- | --- |
| `gradient` | Flattened assembled gradient device array |
| `gradientSegments` | Gradient views in target order |
| `diagonal` | Flattened scalar Hessian diagonal |
| `minimizer` | Advanced minimizer object |
| `numMeshes` | Number of meshes |
| `fullName` | Scene name |

The `energies` property currently exposes a scene dictionary that
`addEnergy()` does not populate; do not use it to enumerate registered
minimizer energy requests.

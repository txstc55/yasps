# Solving and updating

After energies and targets are registered, `minimizeEnergy()` assembles and
solves the current Newton system.

## Solve

```python
segments = simulation.minimizeEnergy(
  tolerance=1e-4,
  maxIterations=20_000,
)
```

The solver uses block-sparse preconditioned conjugate gradient with generated
CUDA kernels and double-precision data.

## Result order and shape

Results follow target order:

```python
simulation.addMinimizeTarget([position, affine_matrix, translation])

d_position, d_affine, d_translation = simulation.minimizeEnergy()
```

Each segment is flattened exactly like the corresponding target data buffer.
Reshape only for host inspection:

```python
host_direction = d_position.get().reshape((-1, 3))
```

## Sign convention

YASPS solves:

\[
H\Delta x=g
\]

Apply:

```python
position.updateValue(position.value - d_position, deepCopy=True)
```

Do not add the returned segment unless your model intentionally changes the
sign of its gradient.

## Newton loop

```python
for iteration in range(max_newton):
  energy_before = simulation.computeTotalEnergy()
  direction = simulation.minimizeEnergy(tolerance=1e-4)[0]

  old_position = position.value.copy()
  step = 1.0
  accepted = False

  for _ in range(max_line_search):
    position.updateValue(old_position - step * direction, deepCopy=True)
    energy_after = simulation.computeTotalEnergy()
    if energy_after <= energy_before:
      accepted = True
      break
    step *= 0.5

  if not accepted:
    position.updateValue(old_position, deepCopy=True)
    break
```

Contact simulations update collision topology inside the line search; see
[dynamic collision terms](dynamic-collision-terms.md).

## Updating velocities

After accepting a frame:

```python
velocity.updateValue(
  (position.value - last_position.value) / dt_value,
  deepCopy=True,
)
```

Store `last_position` before the Newton loop.

## Inspecting assembled values

```python
gradient = simulation.gradient
diagonal = simulation.diagonal
gradient_segments = simulation.gradientSegments
```

These are primarily diagnostic. Sparse Hessian internals are managed by the
minimizer.

## Convergence

`tolerance` is the PCG criterion, not the outer Newton stopping test. Use a
separate physical criterion, such as accepted maximum speed:

```python
accepted_speed = max_abs_direction * step / dt_value
if accepted_speed < 1e-2:
  break
```

Always cap both PCG and Newton iterations.

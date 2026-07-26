# Quickstart

This example minimizes a three-vertex spring chain. It covers the complete
public workflow: hierarchy, data, connectivity, a joined attribute, a named
energy, differentiation, sparse assembly, PCG, and an update.

## 1. Create the hierarchy

```python
import numpy as np

from yasps.scene import scene

simulation = scene("spring_demo")
mesh = simulation.addMesh("mesh")
vertices = mesh.addPrimitive("vertices", numInstances=3)
edges = mesh.addPrimitive("edges", numInstances=2)
```

Primitive types carry no built-in geometric meaning. Here, `"vertices"` and
`"edges"` are names chosen by the application.

## 2. Add vertex data

```python
position = vertices.addAttribute("position", rows=3, cols=1)
position.updateValue(
  np.array(
    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
    dtype=np.float32,
  )
)
```

`position` has per-instance shape 3×1. Its device buffer contains three
instances, so `position.value` has nine scalar entries.

## 3. Describe edge topology and gather positions

```python
edge_to_vertex = edges.addConnectivity(
  "edge_to_vertex",
  to=vertices,
  data=np.array([[0, 1], [1, 2]], dtype=np.uint32),
  dimension=2,
)

edge_position = edges.addAttribute(
  "position",
  through=edge_to_vertex,
  source=position,
)
```

The joined `edge_position` has per-instance shape 2×3: one row for each
endpoint and three columns per vertex position.

## 4. Write and name an energy

```python
displacement = edge_position.row(0) - edge_position.row(1)
spring_energy = edges.addAttribute(
  "spring_energy",
  computed_attribute=0.5 * displacement.dot(displacement),
)
```

Expressions are symbolic. No numerical work occurs until `compute()` or a
scene solve needs a value.

Inspect the two per-edge contributions:

```python
print(spring_energy.compute().value.get())
```

## 5. Register, differentiate, and solve

```python
simulation.addEnergy(spring_energy, projection_method=2)
simulation.addMinimizeTarget([position])

energy_before = simulation.computeTotalEnergy()
delta = simulation.minimizeEnergy(tolerance=1e-6, maxIterations=100)[0]
position.updateValue(position.value - delta, deepCopy=True)
energy_after = simulation.computeTotalEnergy()

print(energy_before, energy_after)
```

`addMinimizeTarget()` performs the symbolic differentiation and prepares the
sparsity structure. The first solve performs remaining numerical setup; later
solves reuse the generated structure when topology is static.

## Important sign convention

The solver computes:

\[
H\Delta x = g
\]

where \(g\) is the assembled energy gradient. Apply the Newton direction by
subtracting the returned segment:

```python
position.updateValue(position.value - delta, deepCopy=True)
```

## Multiple targets

Pass targets in the order in which you want result segments:

```python
simulation.addMinimizeTarget([position, another_attribute])
position_delta, another_delta = simulation.minimizeEnergy()
```

Each result is a live device-array segment matching its target's flattened
layout.

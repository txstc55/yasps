# YASPS

YASPS—Yet Another Symbolic framework for Physical Simulation—is a Python
interface for describing mesh-based computations and energies symbolically.
YASPS differentiates those expressions, assembles sparse gradients and
Hessians, and solves the resulting Newton system on a GPU.

The current documented runtime uses NVIDIA CUDA with PyCUDA and generated
CUDA/Eigen kernels.

## The programming model

A YASPS program usually follows one chain:

1. Create a [scene](concepts/project-model.md), then add meshes and primitive
   types.
2. Store per-instance data in [attributes](concepts/attributes.md).
3. Describe topology with [connectivities](concepts/connectivity-and-join.md)
   and gather data with `JOIN`.
4. Combine heterogeneous primitive types with a [primitive
   union](concepts/primitive-unions.md).
5. Build named scalar energy attributes.
6. Register energies and minimization targets.
7. Ask the scene for a Newton direction, then apply the update.

```python
import numpy as np

from yasps.scene import scene

simulation = scene("quickstart")
mesh = simulation.addMesh("mesh")
vertices = mesh.addPrimitive("vertices", numInstances=3)

position = vertices.addAttribute("position", rows=3, cols=1)
position.updateValue(
  np.array(
    [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    dtype=np.float32,
  )
)

energy = vertices.addAttribute(
  "quadratic_energy",
  computed_attribute=0.5 * position.dot(position),
)
simulation.addEnergy(energy, projection_method=2)
simulation.addMinimizeTarget([position])

delta = simulation.minimizeEnergy(tolerance=1e-6)[0]
position.updateValue(position.value - delta, deepCopy=True)
```

`minimizeEnergy()` returns the solution of \(H\Delta x=g\). YASPS therefore
uses `position - delta` for the Newton update.

## Where to go next

- [Installation](getting-started/installation.md) covers CUDA requirements.
- [Quickstart](getting-started/quickstart.md) builds a complete spring system.
- [Attributes](concepts/attributes.md) explains shapes, constants, computed
  attributes, naming, and materialization.
- [Energies and differentiation](guides/energies-and-differentiation.md)
  explains symbolic derivatives and Hessian projection.
- [Dynamic collision terms](guides/dynamic-collision-terms.md) covers runtime
  contact connectivity and the collision-aware solve loop.
- [Limitations](reference/limitations.md) records sharp implementation edges
  that are easy to miss by reading only the paper.

The interface description is based on the YASPS paper, the public Cython API,
the differentiation and kernel implementations, and the repository examples.

import numpy as np
import pytest

from yasps.backend import is_metal
from yasps.scene import scene


pytestmark = pytest.mark.skipif(not is_metal, reason="requires Apple Metal")


def test_generated_block_sparse_pcg_end_to_end():
  simulation = scene("metal_sparse_pcg")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=2)
  edges = mesh.addPrimitive("edges", numInstances=1)
  values = vertices.addAttribute("values")
  initial = np.array([1.0, 3.0], np.float32)
  values.updateValue(initial)
  edge_vertices = edges.addConnectivity(
    "edge_vertices", vertices, np.array([[0, 1]], np.uint32), 2
  )
  joined = edges.addAttribute(
    "joined", through=edge_vertices, source=values
  )
  difference = joined[0] - joined[1]
  energy = edges.addAttribute(
    "energy",
    computed_attribute=(
      0.5 * difference * difference
      + 0.05 * (joined[0] * joined[0] + joined[1] * joined[1])
    ),
  )
  simulation.addEnergy(energy, projection_method=0)
  simulation.addMinimizeTarget([values])

  solution = simulation.minimizeEnergy(tolerance=1e-7, maxIterations=100)

  np.testing.assert_allclose(
    simulation.gradient.get(), np.array([-1.9, 2.3], np.float32), rtol=2e-6
  )
  np.testing.assert_allclose(solution[0].get(), initial, rtol=2e-5, atol=2e-5)

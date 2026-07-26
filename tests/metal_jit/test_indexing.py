import mlx.core as mx
import numpy as np
import pytest

from yasps.backend import is_metal
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.metal.scan import exclusive_scan, outer_indices
from yasps.scene import scene


pytestmark = pytest.mark.skipif(not is_metal, reason="requires Apple Metal")


def test_hierarchical_generated_prefix_scan():
  source = np.arange(5000, dtype=np.uint32) % 7
  values = mx.array(source)
  scanned = exclusive_scan(values)
  outer = outer_indices(values)
  mx.eval(scanned, outer)
  expected = np.zeros_like(source)
  expected[1:] = np.cumsum(source[:-1], dtype=np.uint32)
  np.testing.assert_array_equal(np.asarray(scanned), expected)
  np.testing.assert_array_equal(np.asarray(outer[:-1]), expected)
  assert int(outer[-1].item()) == int(np.sum(source, dtype=np.uint32))


def test_generated_join_indices_compress_duplicates_and_emit_coordinates():
  simulation = scene("joined_indices")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=3)
  edges = mesh.addPrimitive("edges", numInstances=2)
  position = vertices.addAttribute("position", rows=3, cols=1)
  position.updateValue(np.arange(9, dtype=np.float32))
  edge_vertices = edges.addConnectivity(
    "edge_vertices", vertices, [[0, 0], [1, 2]], 2
  )
  joined = edges.addAttribute(
    "joined_position", through=edge_vertices, source=position
  )
  energy = edges.addAttribute(
    "energy", computed_attribute=joined[0] * joined[0]
  )
  kernel = gradientIndicesKernel(
    {energy: [joined], joined: [position]},
    {},
    [position],
    [0],
    energy,
  )

  kernel.computeIndices([0])

  np.testing.assert_array_equal(
    kernel.outputIndices.get().reshape(2, 2),
    np.array([[2, 2], [5, 8]], dtype=np.uint32),
  )
  np.testing.assert_array_equal(
    kernel.outputPermutations.get().reshape(2, 2),
    np.array([[1, -1], [1, 4]], dtype=np.int16),
  )
  np.testing.assert_array_equal(
    kernel.outputUniqueGradientSizesCPU,
    np.array([3, 6], dtype=np.uint16),
  )
  np.testing.assert_array_equal(
    kernel.outputCompressedCoordinateCountsOuter.get(),
    np.array([0, 1, 4], dtype=np.uint32),
  )
  np.testing.assert_array_equal(
    kernel.outputCoordinates.get().reshape(-1, 2),
    np.array([[0, 0], [3, 3], [3, 6], [6, 6]], dtype=np.uint32),
  )


def test_generated_union_indices_shift_each_child():
  simulation = scene("union_indices")
  mesh = simulation.addMesh("mesh")
  first = mesh.addPrimitive("first", numInstances=2)
  second = mesh.addPrimitive("second", numInstances=2)
  first_value = first.addAttribute("value")
  second_value = second.addAttribute("value")
  first_value.updateValue([1, 2])
  second_value.updateValue([3, 4])
  combined = mesh.addPrimitiveUnion("combined", [first, second])
  combined_value = combined.addAttribute("value")
  energy = combined.addAttribute(
    "energy", computed_attribute=combined_value * combined_value
  )
  kernel = gradientIndicesKernel(
    {energy: [combined_value], combined_value: [first_value, second_value]},
    {first_value: [first_value], second_value: [second_value]},
    [first_value, second_value],
    [0, 2],
    energy,
  )

  kernel.computeIndices([0, 2])

  np.testing.assert_array_equal(
    kernel.outputIndices.get().reshape(4, 2),
    np.array([[2, 0], [3, 0], [4, 0], [5, 0]], dtype=np.uint32),
  )
  np.testing.assert_array_equal(
    kernel.outputSizes.get().reshape(4, 2),
    np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.uint16),
  )

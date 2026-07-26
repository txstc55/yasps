import numpy as np
import pytest

from yasps.backend import is_metal
from yasps import attribute
from yasps.differentiator import differentiator
from yasps.scene import scene


pytestmark = pytest.mark.skipif(not is_metal, reason="requires Apple Metal")


def test_generated_local_projection_assembles_duplicate_join_blocks():
  simulation = scene("local_projection_assembly")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=2)
  edges = mesh.addPrimitive("edges", numInstances=1)
  values = vertices.addAttribute("values")
  values.updateValue(np.array([2.0, 5.0], np.float32))
  edge_vertices = edges.addConnectivity(
    "edge_vertices", vertices, np.array([[0, 0]], np.uint32), 2
  )
  joined = edges.addAttribute(
    "joined", through=edge_vertices, source=values
  )
  summed = joined[0] + joined[1]
  energy = edges.addAttribute(
    "energy", computed_attribute=0.5 * summed * summed
  )
  assembled = differentiator().diff2(
    [energy], [values], [values], projection_method=2
  )

  assembled.compute()

  np.testing.assert_allclose(
    assembled.gradient.value.get(), np.array([8.0, 0.0], np.float32)
  )
  np.testing.assert_allclose(
    assembled.blocks_flattened.get(), np.array([4.0], np.float32), rtol=2e-6
  )
  np.testing.assert_allclose(
    assembled.diagonal_blocks.get(),
    np.array([4.0, 0.0], np.float32),
    rtol=2e-6,
  )


def test_generated_full_projection_uses_float32_eigensolver_and_assembly():
  simulation = scene("full_projection_assembly")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=3)
  samples = mesh.addPrimitive("samples", numInstances=2)
  values = vertices.addAttribute("values")
  values.updateValue(np.array([1.0, -2.0, 3.0], np.float32))
  squared = vertices.addAttribute(
    "squared", computed_attribute=values * values
  )
  sample_vertices = samples.addConnectivity(
    "sample_vertices", vertices, np.array([[0], [2]], np.uint32), 1
  )
  joined = samples.addAttribute(
    "joined", through=sample_vertices, source=squared
  )
  energy = samples.addAttribute("energy", computed_attribute=0.5 * joined)
  assembled = differentiator().diff2(
    [energy], [values], [values], projection_method=2
  )
  assert assembled.project_entire_hessian == [True]

  assembled.compute()

  np.testing.assert_allclose(
    assembled.gradient.value.get(), np.array([1.0, 0.0, 3.0], np.float32)
  )
  np.testing.assert_allclose(
    assembled.blocks_flattened.get(), np.array([1.0, 1.0], np.float32)
  )
  np.testing.assert_allclose(
    assembled.diagonal.get(), np.array([1.0, 0.0, 1.0], np.float32)
  )
  np.testing.assert_allclose(
    assembled.diagonal_blocks.get(), np.array([1.0, 0.0, 1.0], np.float32)
  )


def test_generated_separate_jacobian_assembles_j_transpose_h_j():
  simulation = scene("separate_jacobian_assembly")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=2)
  samples = mesh.addPrimitive("samples", numInstances=2)
  values = vertices.addAttribute("values", rows=2, cols=1)
  values.updateValue(
    np.array([[1.0, 2.0], [-1.0, 4.0]], np.float32)
  )
  weights = attribute.to_array([2.0, 3.0], rows=1, cols=2)
  mapped = vertices.addAttribute(
    "mapped", computed_attribute=weights.mul_explicit(values)
  )
  sample_vertices = samples.addConnectivity(
    "sample_vertices", vertices, np.array([[0], [1]], np.uint32), 1
  )
  joined = samples.addAttribute(
    "joined", through=sample_vertices, source=mapped
  )
  energy = samples.addAttribute(
    "energy", computed_attribute=0.5 * joined * joined
  )
  assembled = differentiator().diff2(
    [energy],
    [values],
    [values],
    projection_method=2,
    separate_hessian_jacobian=True,
  )
  assert assembled.project_entire_hessian == [False]

  assembled.compute()

  np.testing.assert_allclose(
    assembled.gradient.value.get().reshape(2, 2),
    np.array([[16.0, 24.0], [20.0, 30.0]], np.float32),
  )
  expected_block = np.array([[4.0, 6.0], [6.0, 9.0]], np.float32)
  np.testing.assert_allclose(
    assembled.blocks_flattened.get().reshape(2, 2, 2),
    np.broadcast_to(expected_block, (2, 2, 2)),
  )
  np.testing.assert_allclose(
    assembled.diagonal_blocks.get().reshape(2, 2, 2),
    np.broadcast_to(expected_block, (2, 2, 2)),
  )

import numpy as np
import pytest

from yasps.backend import is_metal
from yasps.scene import scene


pytestmark = pytest.mark.skipif(not is_metal, reason="requires Apple Metal")


def _primitive(name, count):
  simulation = scene(name)
  mesh = simulation.addMesh("mesh")
  return mesh.addPrimitive("items", numInstances=count)


def test_symbolic_graph_is_jit_compiled_as_reusable_modules(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  primitive = _primitive("module_test", 3)
  values = primitive.addAttribute("values", rows=3, cols=1)
  values.updateValue(np.arange(1, 10, dtype=np.float32).reshape(3, 3))
  shifted = primitive.addAttribute("shifted", computed_attribute=values + 2.0)
  scaled = primitive.addAttribute("scaled", computed_attribute=shifted * 3.0)

  scaled.compute()

  np.testing.assert_allclose(
    scaled.value.get().reshape(3, 3),
    (np.arange(1, 10, dtype=np.float32).reshape(3, 3) + 2.0) * 3.0,
  )
  program = scaled.globalKernel.program
  assert len(program.modules) == 2
  assert program.source.count("yasps_module_") == 1
  assert program.header.count("METAL_FUNC void yasps_module_") == 2
  assert len(list((tmp_path / ".yasps_tmp/metal").glob("*.metal"))) == 2

  reused = primitive.addAttribute("reused", computed_attribute=shifted - 1.0)
  reused.compute()
  shifted_module = next(
    module for module in program.modules if module.key == shifted.hash
  )
  assert shifted_module.name in reused.globalKernel.program.header


def test_generated_float32_matrix_kernels_and_spd_projection():
  primitive = _primitive("linalg_test", 2)
  matrices = primitive.addAttribute("matrices", rows=2, cols=2)
  source = np.array(
    [[[4.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [2.0, -1.0]]],
    dtype=np.float32,
  )
  matrices.updateValue(source)

  inverse = primitive.addAttribute(
    "inverse", computed_attribute=matrices.inverse()
  )
  determinant = primitive.addAttribute(
    "determinant", computed_attribute=matrices.determinant()
  )
  projected = primitive.addAttribute(
    "projected", computed_attribute=matrices.spd(2)
  )
  inverse.compute()
  determinant.compute()
  projected.compute()

  np.testing.assert_allclose(
    inverse.value.get().reshape(2, 2, 2),
    np.linalg.inv(source),
    rtol=2e-5,
    atol=2e-5,
  )
  np.testing.assert_allclose(
    determinant.value.get(), np.linalg.det(source), rtol=2e-5, atol=2e-5
  )
  symmetric = 0.5 * (source + source.swapaxes(-1, -2))
  eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
  expected = (eigenvectors * np.maximum(eigenvalues, 0.0)[:, None, :])
  expected = expected @ eigenvectors.swapaxes(-1, -2)
  np.testing.assert_allclose(
    projected.value.get().reshape(2, 2, 2),
    expected,
    rtol=3e-5,
    atol=3e-5,
  )


def test_generated_batched_spd_uses_threadgroup_jacobi(
  tmp_path, monkeypatch
):
  monkeypatch.chdir(tmp_path)
  count = 1024
  dimension = 12
  primitive = _primitive("batched_spd", count)
  matrices = primitive.addAttribute(
    "matrices", rows=dimension, cols=dimension
  )
  rng = np.random.default_rng(20260725)
  source = rng.normal(
    size=(count, dimension, dimension)
  ).astype(np.float32)
  source = 0.5 * (source + source.swapaxes(1, 2))
  matrices.updateValue(source)
  projected = primitive.addAttribute(
    "projected", computed_attribute=matrices.spd(1)
  )

  projected.compute()

  eigenvalues, eigenvectors = np.linalg.eigh(source)
  expected = eigenvectors * np.abs(eigenvalues)[:, None, :]
  expected = expected @ eigenvectors.swapaxes(1, 2)
  np.testing.assert_allclose(
    projected.value.get().reshape(source.shape),
    expected,
    rtol=8e-5,
    atol=3e-5,
  )
  program = projected.globalKernel.program
  assert program._staged_spd
  assert "yasps_staged_resources" in program.resource_input_names
  assert list(
    (tmp_path / ".yasps_tmp/metal").glob("yasps_batched_spd_12_*.metal")
  )


def test_generated_join_and_dynamic_sum_kernels():
  simulation = scene("topology_test")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=4)
  edges = mesh.addPrimitive("edges", numInstances=2)
  values = vertices.addAttribute("values", rows=2, cols=1)
  values.updateValue(
    np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)
  )
  edge_vertices = edges.addConnectivity(
    "edge_vertices",
    vertices,
    np.array([[0, 2], [1, 3]], dtype=np.uint32),
    2,
  )
  joined = edges.addAttribute(
    "joined", through=edge_vertices, source=values
  )
  joined.compute()
  np.testing.assert_array_equal(
    joined.value.get().reshape(2, 4),
    np.array([[1, 10, 3, 30], [2, 20, 4, 40]], dtype=np.float32),
  )

  vertex_edges = vertices.addConnectivity(
    "vertex_edges", edges, [[0], [0, 1], [1], []], 0
  )
  weights = edges.addAttribute("weights")
  weights.updateValue(np.array([5, 7], dtype=np.float32))
  summed = vertices.addAttribute(
    "summed", through=vertex_edges, source=weights, operation="SUM"
  )
  summed.compute()
  np.testing.assert_array_equal(
    summed.value.get(), np.array([5, 12, 7, 0], dtype=np.float32)
  )


def test_generated_triangle_normal_kernel():
  simulation = scene("normal_test")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=4)
  triangles = mesh.addPrimitive("triangles", numInstances=2)
  positions = vertices.addAttribute("position", rows=3, cols=1)
  source = np.array(
    [
      [0.0, 0.0, 0.0],
      [2.0, 0.0, 0.0],
      [0.0, 3.0, 0.0],
      [0.0, 0.0, 4.0],
    ],
    dtype=np.float32,
  )
  positions.updateValue(source)
  triangle_vertices = triangles.addConnectivity(
    "triangle_vertices",
    vertices,
    np.array([[0, 1, 2], [0, 3, 1]], dtype=np.uint32),
    3,
  )
  triangle_positions = triangles.addAttribute(
    "positions", through=triangle_vertices, source=positions
  )
  edge_01 = triangle_positions.row(1) - triangle_positions.row(0)
  edge_02 = triangle_positions.row(2) - triangle_positions.row(0)
  unnormalized = edge_01.cross(edge_02)
  normals = triangles.addAttribute(
    "normal", computed_attribute=unnormalized / unnormalized.norm()
  )

  normals.compute()

  expected = np.cross(
    source[[1, 3]] - source[[0, 0]],
    source[[2, 1]] - source[[0, 0]],
  )
  expected /= np.linalg.norm(expected, axis=1, keepdims=True)
  np.testing.assert_allclose(
    normals.value.get().reshape(2, 3), expected, rtol=2e-6, atol=2e-6
  )


def test_generated_resource_packing_avoids_metal_buffer_limit():
  primitive = _primitive("resource_packing", 1)
  sources = []
  for index in range(40):
    source = primitive.addAttribute(f"source_{index}")
    source.updateValue(np.array([index + 1], dtype=np.float32))
    sources.append(source)
  expression = sources[0]
  for source in sources[1:]:
    expression = expression + source
  total = primitive.addAttribute("total", computed_attribute=expression)

  total.compute()

  assert total.value.get()[0] == pytest.approx(sum(range(1, 41)))
  program = total.globalKernel.program
  assert program.resource_input_names == [
    "yasps_float_resources",
    "yasps_resource_offsets",
  ]
  assert len(program.input_names) == 3


def test_emission_cache_does_not_alias_equivalent_negations():
  primitive = _primitive("negation_identity_cache", 2)
  values = primitive.addAttribute("values", rows=3, cols=1)
  source = np.array(
    [[1.25, -2.5, 3.75], [-4.0, 5.0, -6.0]], dtype=np.float32
  )
  values.updateValue(source)
  restored = primitive.addAttribute(
    "restored", computed_attribute=-(-values)
  )

  restored.compute()

  np.testing.assert_array_equal(restored.value.get().reshape(2, 3), source)

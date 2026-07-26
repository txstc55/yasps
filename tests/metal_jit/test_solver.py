import numpy as np
import pytest

from yasps.backend import is_metal
from yasps.kernel.Compute.compensatedSumMetal import (
  compensated_dot,
  compensated_sum,
)
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


def test_generated_spmv_accumulates_high_valence_rows_across_threadgroups():
  simulation = scene("metal_sparse_pcg_star")
  mesh = simulation.addMesh("mesh")
  vertex_count = 70
  vertices = mesh.addPrimitive("vertices", numInstances=vertex_count)
  edges = mesh.addPrimitive("edges", numInstances=vertex_count - 1)
  values = vertices.addAttribute("values")
  initial = np.linspace(-2.0, 3.0, vertex_count, dtype=np.float32)
  values.updateValue(initial)
  edge_indices = np.column_stack(
    (
      np.zeros(vertex_count - 1, dtype=np.uint32),
      np.arange(1, vertex_count, dtype=np.uint32),
    )
  )
  edge_vertices = edges.addConnectivity(
    "edge_vertices", vertices, edge_indices, 2
  )
  joined = edges.addAttribute(
    "joined", through=edge_vertices, source=values
  )
  difference = joined[0] - joined[1]
  energy = edges.addAttribute(
    "energy",
    computed_attribute=(
      0.5 * difference * difference
      + 0.01 * (joined[0] * joined[0] + joined[1] * joined[1])
    ),
  )
  simulation.addEnergy(energy, projection_method=0)
  simulation.addMinimizeTarget([values])

  solution = simulation.minimizeEnergy(tolerance=1e-7, maxIterations=500)

  np.testing.assert_allclose(
    solution[0].get(), initial, rtol=2e-4, atol=2e-4
  )


def test_metal_solver_reuses_previous_solution_as_initial_guess(
  monkeypatch, capsys
):
  monkeypatch.setenv("YASPS_SOLVER_TRACE", "1")
  simulation = scene("metal_sparse_pcg_warm_start")
  mesh = simulation.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=2)
  edges = mesh.addPrimitive("edges", numInstances=1)
  values = vertices.addAttribute("values")
  values.updateValue(np.array([1.0, 3.0], np.float32))
  edge_vertices = edges.addConnectivity(
    "edge_vertices",
    vertices,
    np.array([[0, 1]], np.uint32),
    2,
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

  simulation.minimizeEnergy(tolerance=1e-7, maxIterations=100)
  capsys.readouterr()
  simulation.minimizeEnergy(tolerance=1e-7, maxIterations=100)

  assert "Metal PCG status 0" in capsys.readouterr().out


def test_compensated_sum_preserves_float32_cancellation_residual():
  import mlx.core as mx

  values = mx.array([1.0e8, 1.0, -1.0e8], dtype=mx.float32)

  assert compensated_sum(values) == 1.0


def test_compensated_dot_preserves_float32_cancellation_residual():
  import mlx.core as mx

  left = mx.array([1.0e8, 1.0, -1.0e8], dtype=mx.float32)
  right = mx.ones((3,), dtype=mx.float32)

  assert compensated_dot(left, right) == 1.0

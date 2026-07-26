from pathlib import Path
import sys

import numpy as np
import pytest

from yasps.backend import gpuarray, is_metal


pytestmark = pytest.mark.skipif(not is_metal, reason="requires Apple Metal")
sys.path.insert(
  0, str(Path(__file__).parents[2] / "examples" / "ccd")
)
from ccd import CCD  # noqa: E402


def _device(values, dtype):
  return gpuarray.to_gpu(np.asarray(values, dtype=dtype).reshape(-1))


def test_generated_discrete_face_and_edge_kernels_are_repeatable():
  face_vertices = _device(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.2, 0.2, 0.1],
    ],
    np.float32,
  )
  faces = _device([[0, 1, 2]], np.uint32)
  surface = _device([0, 1, 2, 3], np.uint32)
  face_ccd = CCD(4, 4, max_cd_pairs=100, max_ccd_pairs=100)
  face_ccd.init_faces(face_vertices, faces, surface, 1)

  for _ in range(2):
    face_ccd.cd_faces(face_vertices, 0.04)
    assert face_ccd.separated_counts == [0, 0, 1, 0]
    np.testing.assert_array_equal(
      face_ccd.pt.get()[:4], np.array([3, 0, 1, 2], np.uint32)
    )

  edge_vertices = _device(
    [
      [-1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, -1.0, 0.05],
      [0.0, 1.0, 0.05],
    ],
    np.float32,
  )
  edges = _device([[0, 1], [2, 3]], np.uint32)
  edge_ccd = CCD(0, 4, max_cd_pairs=100, max_ccd_pairs=100)
  edge_ccd.init_edges(edge_vertices, edge_vertices, edges, 2)
  edge_ccd.cd_edges(edge_vertices, 0.01)
  assert edge_ccd.separated_counts == [0, 0, 0, 1]
  np.testing.assert_array_equal(
    edge_ccd.ee.get()[:4], np.array([0, 1, 2, 3], np.uint32)
  )


def test_nearly_parallel_edges_use_a_finite_endpoint_feature():
  vertices = _device(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0e-4, 1.0e-4],
      [1.0, 1.01e-4, 1.0e-4],
    ],
    np.float32,
  )
  edges = _device([[0, 1], [2, 3]], np.uint32)
  detector = CCD(0, 4, max_cd_pairs=100, max_ccd_pairs=100)
  detector.init_edges(vertices, vertices, edges, 2)

  detector.cd_edges(vertices, 1.0e-6)

  counts = detector.separated_counts
  assert sum(counts) > 0
  assert counts[3] == 0


def test_generated_swept_broad_phase_and_additive_step_kernel():
  positions = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.2, 0.2, 1.0],
    ],
    np.float32,
  )
  movement = np.zeros_like(positions)
  movement[3, 2] = 2.0
  vertices = _device(positions, np.float32)
  directions = _device(movement, np.float32)
  faces = _device([[0, 1, 2]], np.uint32)
  surface = _device([0, 1, 2, 3], np.uint32)
  detector = CCD(4, 4, max_cd_pairs=100, max_ccd_pairs=100)
  detector.init_faces(vertices, faces, surface, 1)

  detector.ccd(vertices, 1.0e-4, directions, 1.0)

  assert int(detector.cp_num.get()[0]) == 1
  np.testing.assert_array_equal(
    detector.collision_pairs_ccd.get(),
    np.array([[-4, 0, 1, 2]], np.int32),
  )
  assert detector.compute_largest_step_size(
    0.8, vertices, directions
  ) == pytest.approx(0.4, abs=2.0e-6)


def test_generated_append_query_grows_and_reuses_pair_capacity():
  positions = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.2, 0.2, 1.0],
      [0.3, 0.2, 1.0],
      [0.2, 0.3, 1.0],
    ],
    np.float32,
  )
  movement = np.zeros_like(positions)
  movement[3:, 2] = 2.0
  vertices = _device(positions, np.float32)
  directions = _device(movement, np.float32)
  faces = _device([[0, 1, 2]], np.uint32)
  surface = _device([0, 1, 2, 3, 4, 5], np.uint32)
  detector = CCD(6, 6, max_cd_pairs=100, max_ccd_pairs=100)
  detector.init_faces(vertices, faces, surface, 1)
  detector._pair_capacities[("faces", True)] = 1

  detector.ccd(vertices, 1.0e-4, directions, 1.0)

  assert detector._pair_capacities[("faces", True)] >= 3
  assert int(detector.cp_num.get()[0]) == 3
  np.testing.assert_array_equal(
    np.sort(detector.collision_pairs_ccd.get()[:, 0]),
    np.array([-6, -5, -4], np.int32),
  )


def test_two_pass_fallback_handles_an_empty_query(monkeypatch):
  monkeypatch.setenv("YASPS_METAL_CCD_APPEND", "0")
  vertices = _device(
    [
      [-1.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [-1.0, 2.0, 0.0],
      [1.0, 2.0, 0.0],
    ],
    np.float32,
  )
  directions = _device(np.zeros((4, 3)), np.float32)
  edges = _device([[0, 1], [2, 3]], np.uint32)
  detector = CCD(0, 4, max_cd_pairs=100, max_ccd_pairs=100)
  detector.init_edges(vertices, vertices, edges, 2)

  detector.ccd(vertices, 1.0e-4, directions, 1.0)

  assert int(detector.cp_num.get()[0]) == 0

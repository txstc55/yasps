from pathlib import Path
import sys

import numpy as np

from yasps.backend import gpuarray


_CCD_DIRECTORY = (
  Path(__file__).resolve().parents[2] / "examples" / "ccd"
)
sys.path.insert(0, str(_CCD_DIRECTORY))

from ccd import CCD  # noqa: E402
from ccd_metal import _MetalBVH  # noqa: E402


def _gpu(values, dtype):
  return gpuarray.to_gpu(np.asarray(values, dtype=dtype).reshape(-1))


def test_metal_lbvh_root_bounds_across_reduction_boundary():
  rng = np.random.default_rng(37)
  count = 257
  vertices = rng.normal(size=(count * 3, 3)).astype(np.float32)
  faces = np.arange(count * 3, dtype=np.uint32).reshape(count, 3)
  bvh = _MetalBVH(count, "face")

  bvh.construct(
    _gpu(vertices, np.float32),
    _gpu(faces, np.uint32),
  )

  root = bvh.boxes.get()[:32].view(np.float32)
  np.testing.assert_allclose(
    root[:3],
    vertices.max(axis=0),
    rtol=0,
    atol=1.0e-6,
  )
  np.testing.assert_allclose(
    root[4:7],
    vertices.min(axis=0),
    rtol=0,
    atol=1.0e-6,
  )


def test_metal_cd_classifies_point_triangle_and_edge_edge_pairs():
  vertices = _gpu(
    [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0.25, 0.25, 0.01],
    ],
    np.float32,
  )
  faces = _gpu([[0, 1, 2], [0, 2, 1]], np.uint32)
  surface = _gpu([3], np.uint32)
  face_ccd = CCD(
    1,
    4,
    max_cd_pairs=32,
    max_ccd_pairs=32,
  )
  face_ccd.init_faces(vertices, faces, surface, 2)

  face_ccd.cd_faces(vertices, 0.01)

  assert face_ccd.separated_counts == [0, 0, 2, 0]
  assert {
    tuple(pair)
    for pair in face_ccd.pt.get()[:8].reshape(-1, 4)
  } == {
    (3, 0, 1, 2),
    (3, 0, 2, 1),
  }

  edge_vertices = _gpu(
    [
      [-1, 0, 0],
      [1, 0, 0],
      [0, -1, 0.01],
      [0, 1, 0.01],
    ],
    np.float32,
  )
  edges = _gpu([[0, 1], [2, 3]], np.uint32)
  edge_ccd = CCD(
    0,
    4,
    max_cd_pairs=32,
    max_ccd_pairs=32,
  )
  edge_ccd.init_edges(
    edge_vertices,
    edge_vertices.copy(),
    edges,
    2,
  )

  edge_ccd.cd_edges(edge_vertices, 0.01)

  assert edge_ccd.separated_counts == [0, 0, 0, 1]
  np.testing.assert_array_equal(
    edge_ccd.ee.get()[:4],
    np.array([0, 1, 2, 3], dtype=np.uint32),
  )


def test_metal_swept_broad_phase_and_accd_step():
  vertices = _gpu(
    [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0.25, 0.25, 1],
    ],
    np.float32,
  )
  directions = _gpu(
    [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 2],
    ],
    np.float32,
  )
  faces = _gpu([[0, 1, 2], [0, 2, 1]], np.uint32)
  surface = _gpu([3], np.uint32)
  ccd = CCD(
    1,
    4,
    max_cd_pairs=32,
    max_ccd_pairs=32,
  )
  ccd.init_faces(vertices, faces, surface, 2)

  ccd.ccd(vertices, 1.0e-4, directions, 1.0)
  step = ccd.compute_largest_step_size(
    0.8,
    vertices,
    directions,
  )

  assert int(ccd.cp_num.get()[0]) == 2
  np.testing.assert_allclose(step, 0.4, rtol=1.0e-6)

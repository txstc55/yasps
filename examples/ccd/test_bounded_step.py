"""Small CUDA regressions for the bounded CCD narrow phase.

Run from the repository root:
  python -m unittest discover -s examples/ccd -p test_bounded_step.py -v

Requires PyCUDA, CUDA, and NVCC. Collision times below are analytic; no
reference CCD implementation or simulation assets are needed.
"""

import ctypes
from pathlib import Path
import unittest

import numpy as np
import pycuda.autoinit  # noqa: F401: establish the context before building CCD
import pycuda.gpuarray as gpuarray

import ccd


def _gpu(values, dtype):
  return gpuarray.to_gpu(np.ascontiguousarray(values, dtype=dtype).ravel())


def _pointer(array):
  return ctypes.c_void_p(int(array.gpudata) if array.size else None)


def _pair(kind, gap=1.0):
  if kind == "PT":
    positions = ((0, 0, gap), (-2, -2, 0), (2, -2, 0), (0, 2, 0))
    return np.array(positions, dtype=np.float64), [0], ((1, 2, 3),), (), ((-1, 0),)
  positions = ((-2, 0, 0), (2, 0, 0), (0, -2, gap), (0, 2, gap))
  return np.array(positions, dtype=np.float64), [2, 3], (), ((0, 1), (2, 3)), ((0, 1),)


class BoundedStepTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    _, library_path = ccd._build_libraries(Path(ccd.__file__).resolve().parent)
    cls.library = ctypes.CDLL(str(library_path))
    cls.bounded = cls.library.self_largestFeasibleStepSizeCompactWithBound
    cls.bounded.argtypes = [ctypes.c_double] + [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_double]
    cls.bounded.restype = ctypes.c_double

  def step(self, positions, displacement, faces=(), edges=(), candidates=(), bound=1.0):
    # Prescribe candidates to isolate narrow-phase/reduction behavior from BVH.
    # CCD's moveDir is subtracted, while displacement is physical motion.
    arrays = (
      _gpu(positions, np.float64), _gpu(candidates, np.int32),
      _gpu(faces, np.uint32), _gpu(edges, np.uint32),
      _gpu(-displacement, np.float64),
      gpuarray.empty(max(1, (len(candidates) + 255) // 256), np.float64),
    )
    step = self.bounded(0.8, *map(_pointer, arrays), len(candidates), bound)
    self.assertTrue(np.isfinite(step), f"nonfinite step: {step}")
    self.assertGreaterEqual(step, 0.0)
    self.assertLessEqual(step, bound)
    return step

  def test_analytic_approach_respects_bound_across_scales(self):
    # Contact at t=0.5; slackness=0.8 retains 20% of initial clearance.
    for kind in ("PT", "EE"):
      for scale in (1e-8, 1.0, 1e8):
        positions, moving, faces, edges, pairs = _pair(kind)
        positions *= scale
        displacement = np.zeros_like(positions)
        displacement[moving, 2] = -2 * scale
        for bound in (0.0, 0.1, 0.75, 1.0):
          with self.subTest(kind=kind, scale=scale, bound=bound):
            step = self.step(positions, displacement, faces, edges, pairs, bound)
            self.assertAlmostEqual(step, min(bound, 0.4), delta=1e-12)

  def test_no_motion_initial_contact_and_empty_candidates(self):
    for kind in ("PT", "EE"):
      for gap in (0.0, 1.0):
        positions, moving, faces, edges, pairs = _pair(kind, gap)
        for moving_pair in (False, True):
          with self.subTest(kind=kind, gap=gap, moving=moving_pair):
            displacement = np.zeros_like(positions)
            if moving_pair:
              displacement[moving, 2] = -2.0
            step = self.step(positions, displacement, faces, edges, pairs, 0.125)
            self.assertEqual(step, 0.0 if gap == 0.0 else 0.125)
        for bound in (0.0, 0.125, 1.0):
          self.assertEqual(self.step(positions, np.zeros_like(positions), bound=bound), bound)

  def test_grazing_pairs_make_safe_progress_and_keep_short_bound(self):
    for kind in ("PT", "EE"):
      with self.subTest(kind=kind):
        positions, moving, faces, edges, pairs = _pair(kind, 1e-12)
        displacement = np.zeros_like(positions)
        displacement[moving] = (0.25, 0.0, -2e-12)
        step = self.step(positions, displacement, faces, edges, pairs)
        self.assertGreater(step, 0.399)
        self.assertLessEqual(step, 0.4)
        self.assertEqual(self.step(positions, displacement, faces, edges, pairs, 1e-12), 1e-12)

  def test_near_parallel_edges_stop_before_interior_crossing(self):
    positions = np.array(((-0.5, 0, 0), (0.5, 0, 0),
                          (-0.5, -5e-12, 1e-14), (0.5, 5e-12, 1e-14)))
    displacement = np.zeros_like(positions)
    displacement[2:, 2] = -2e-14
    step = self.step(positions, displacement, edges=((0, 1), (2, 3)), candidates=((0, 1),))
    self.assertGreater(step, 0.0)
    self.assertLess(step, 0.5)

  def test_collapsed_and_parallel_edges_remain_conservative(self):
    for extent in (0.0, 2.0):
      with self.subTest(extent=extent):
        positions = np.array(((0, 0, 0), (extent, 0, 0), (0, 0, 1), (extent, 0, 1)), dtype=np.float64)
        displacement = np.zeros_like(positions)
        displacement[2:, 2] = -2.0
        step = self.step(positions, displacement, edges=((0, 1), (2, 3)), candidates=((0, 1),))
        self.assertLess(step, 0.5)

  def test_unresolved_pair_in_final_block_reduces_provisional_bound(self):
    for kind in ("PT", "EE"):
      with self.subTest(kind=kind):
        grazing, moving, _, _, _ = _pair(kind, 1e-8)
        approaching, _, _, _, _ = _pair(kind, 1.25e-5)
        positions = np.concatenate((grazing, approaching))
        displacement = np.zeros_like(positions)
        displacement[moving] = (0.25, 0.0, -0.0016)
        displacement[np.array(moving) + 4, 2] = -1.0
        # 512 completed pairs yield 1e-5. The last grazing pair must retain
        # its unresolved marker through both reductions and lower this to 5e-6.
        if kind == "PT":
          step = self.step(positions, displacement, faces=((1, 2, 3), (5, 6, 7)),
                           candidates=[(-5, 1)] * 512 + [(-1, 0)])
        else:
          step = self.step(positions, displacement, edges=((0, 1), (2, 3), (4, 5), (6, 7)),
                           candidates=[(2, 3)] * 512 + [(0, 1)])
        self.assertGreater(step, 4.9e-6)
        self.assertLessEqual(step, 5e-6)

  def test_public_api_uses_cached_sweep_bound(self):
    for kind in ("PT", "EE"):
      with self.subTest(kind=kind):
        positions, moving, faces, edges, _ = _pair(kind, 1e-12)
        displacement = np.zeros_like(positions)
        displacement[moving] = (0.25, 0.0, -2e-12)
        vertices, directions = _gpu(positions, np.float64), _gpu(-displacement, np.float64)
        is_pt = kind == "PT"
        mesh_ids = [2, 1, 1, 1] if is_pt else [1, 1, 2, 2]
        with ccd.CCD(int(is_pt), 4, max_cd_pairs=1, max_ccd_pairs=1,
                     mesh_indices=mesh_ids, print_timings=False) as detector:
          detector.init_faces(vertices, _gpu(faces, np.uint32), _gpu([0] if is_pt else [], np.uint32), len(faces))
          detector.init_edges(vertices, vertices, _gpu(edges, np.uint32), len(edges))
          for bound in (1e-12, 0.1, 0.75):
            self.assertEqual(detector.ccd(vertices, 4.0, directions, bound), 1)
            step = detector.compute_largest_step_size(0.8, vertices, directions)
            self.assertTrue(np.isfinite(step))
            self.assertGreaterEqual(step, 0.0)
            self.assertLessEqual(step, bound)
            if bound < 0.4:
              self.assertEqual(step, bound)
            else:
              self.assertGreater(step, 0.399)
              self.assertLessEqual(step, 0.4)


if __name__ == "__main__":
  unittest.main()

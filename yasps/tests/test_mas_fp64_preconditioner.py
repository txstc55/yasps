"""FP64 MAS storage/application regressions, including the frame-42 bank.

The generated 24-by-24 SPD case is portable. The saved scene bank is optional
and is never modified. CPU tests do not import PyCUDA or create a context.
Run CPU-only checks with ``python -m unittest discover -s yasps/tests
-p test_mas_fp64_preconditioner.py -k CPU -v``; omit ``-k CPU`` for CUDA tests.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_REPOSITORY = Path(__file__).resolve().parents[2]
_MAS_DIRECTORY = _REPOSITORY / "yasps" / "yasps" / "solver" / "mas"
_PACKAGE_NAME = "_yasps_fp64_preconditioner_test_mas"
_SPEC = importlib.util.spec_from_file_location(
  _PACKAGE_NAME, _MAS_DIRECTORY / "__init__.py",
  submodule_search_locations=[str(_MAS_DIRECTORY)],
)
_MAS = importlib.util.module_from_spec(_SPEC)
sys.modules[_PACKAGE_NAME] = _MAS
_SPEC.loader.exec_module(_MAS)
BlockSparseMatrixView = _MAS.BlockSparseMatrixView
MASSolver = _MAS.MASSolver

_SAVED_BANK = (
  _REPOSITORY / "examples" / "mpm_fem_affine_bunnies" / "outputs"
  / "frame_0042_first_solve_diagnostic_20260911" / "explicit"
  / "domain_011141_level_0.npz"
)


def _cholesky_inverse(matrix):
  factor = np.linalg.cholesky(matrix)
  inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(len(matrix))))
  return 0.5 * (inverse + inverse.T)


def _generated_bank():
  # One stiff direction gives an inverse eigenvalue below FP32 roundoff.
  # This is an SPD input, not an indefinite matrix that should be rejected.
  direction = np.ones(24, dtype=np.float64) / np.sqrt(24.0)
  matrix = np.eye(24) + (1e8 - 1.0) * np.outer(direction, direction)
  return matrix, direction, _cholesky_inverse(matrix)


def _saved_bank():
  if not _SAVED_BANK.is_file():
    raise unittest.SkipTest("Optional saved frame-42 local bank is unavailable")
  with np.load(_SAVED_BANK, allow_pickle=False) as data:
    matrix = data["actual_matrix"].copy()
    residual = data["residual_local"].copy()
    expected = data["expected_matrix"].copy()
  return matrix, residual, _cholesky_inverse(matrix), expected


def _heterogeneous_system():
  dimensions = np.array([3, 4, 3, 2], dtype=np.int64)
  offsets = np.cumsum(np.r_[0, dimensions[:-1]], dtype=np.int64)
  rng = np.random.default_rng(20260911)
  factor = rng.normal(size=(int(dimensions.sum()),) * 2)
  dense = factor @ factor.T + 4.0 * np.eye(int(dimensions.sum()))
  static, dynamic = [], []
  for row, rows in enumerate(dimensions):
    for col in range(row, len(dimensions)):
      cols = dimensions[col]
      block = dense[offsets[row]:offsets[row] + rows, offsets[col]:offsets[col] + cols]
      (static if row == col else dynamic).append(((row, col), block.copy()))
  view = BlockSparseMatrixView.from_blocks(dimensions, static, dynamic)
  pairs = [(row, col) for row in range(len(dimensions)) for col in range(row, len(dimensions))]
  coordinates = np.array([(offsets[row], offsets[col]) for row, col in pairs], dtype=np.int64).ravel()
  shapes = np.array([(dimensions[row], dimensions[col]) for row, col in pairs], dtype=np.int64).ravel()
  return view, dense, (coordinates, shapes, len(pairs))


class FP64PreconditionerCPUTests(unittest.TestCase):
  def test_generated_spd_inverse_exposes_fp32_storage_loss(self):
    matrix, residual, inverse = _generated_bank()
    self.assertGreater(np.linalg.eigvalsh(matrix)[0], 0.0)
    self.assertGreater(np.linalg.eigvalsh(inverse)[0], 0.0)
    self.assertGreater(float(residual @ inverse @ residual), 0.0)
    rounded = inverse.astype(np.float32).astype(np.float64)
    self.assertLess(float(residual @ rounded @ residual), 0.0)
    np.testing.assert_allclose(matrix @ inverse, np.eye(24), atol=3e-8)

  def test_saved_bank_has_independent_spd_cpu_reference(self):
    matrix, residual, inverse, expected = _saved_bank()
    self.assertEqual(matrix.shape, (24, 24))
    self.assertEqual(residual.shape, (24,))
    self.assertLess(np.linalg.norm(matrix - expected) / np.linalg.norm(expected), 1e-12)
    self.assertGreater(np.linalg.eigvalsh(inverse)[0], 0.0)
    self.assertGreater(float(residual @ inverse @ residual), 0.0)

  def test_heterogeneous_reference_is_spd_and_explicit_graph_is_multilevel(self):
    view, dense, topology = _heterogeneous_system()
    np.linalg.cholesky(dense)
    selected = MASSolver(
      inverse_backend="cpu_reference", max_domain_dofs=16,
      max_levels=3, target_nodes_per_partition=8,
    )
    selected.rebuild_hierarchy_from_blocks(*topology)
    self.assertGreater(selected.hierarchy.number_of_levels, 1)
    rhs = np.linspace(-0.8, 0.7, view.rows)
    selected.solve(view, rhs, tolerance=1e-18, max_iterations=2000)
    self.assertTrue(selected.statistics.converged, selected.statistics)
    np.testing.assert_allclose(selected.solution, np.linalg.solve(dense, rhs), rtol=1e-8, atol=1e-10)


class FP64PreconditionerGPUTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Keep all device initialization here so -k CPU remains CUDA-free.
    try:
      import pycuda.driver as cuda
    except ImportError as error:
      raise unittest.SkipTest(f"PyCUDA is unavailable: {error}") from error
    try:
      cuda.init()
      if cuda.Device.count() == 0:
        raise unittest.SkipTest("No CUDA device is available")
      import pycuda.autoinit  # noqa: F401
      import pycuda.gpuarray as gpuarray
    except (cuda.Error, RuntimeError) as error:
      raise unittest.SkipTest(f"CUDA is unavailable: {error}") from error
    cls.cuda, cls.gpuarray = cuda, gpuarray

  def setUp(self):
    self.solvers = []

  def tearDown(self):
    self.cuda.Context.synchronize()
    for selected in self.solvers:
      selected.reset()

  def make_solver(self, **options):
    # Keep the production inverse algorithm/precision defaults; never permit
    # the CPU fallback to make a CUDA regression appear to pass.
    selected = MASSolver(allow_cpu_fallback=False, **options)
    self.solvers.append(selected)
    return selected

  def assert_fp64_buffers(self, runtime):
    for name in (
      "matrices", "inverses", "mixed_inverses", "packed_residual",
      "packed_correction", "_preconditioned_output", "dynamic_edge_matrices",
      "dynamic_edge_inverses",
    ):
      with self.subTest(buffer=name):
        self.assertEqual(np.dtype(getattr(runtime, name).dtype), np.dtype(np.float64))

  def apply(self, runtime, residual, reuse_workspace):
    result = runtime.precondition(
      self.gpuarray.to_gpu(residual), reuse_workspace=reuse_workspace,
      stream=runtime._pcg_stream,
    )
    self.cuda.Context.synchronize()
    return result.get()

  def check_local_bank(self, matrix, residual, reference, *, padded_size=None, apply_rtol=2e-5):
    options = dict(max_levels=1, level_weights=(1.0,), adaptive_fine_level_weight=False)
    if padded_size is not None:
      options["cuda_fixed_inverse_bucket_size"] = padded_size
    selected = self.make_solver(**options)
    view = BlockSparseMatrixView.from_blocks([len(matrix)], [((0, 0), matrix)])
    # A zero iteration budget builds/validates the actual default GPU inverse
    # and initializes PCG without using a capped solve as accuracy evidence.
    selected.solve(view, self.gpuarray.to_gpu(residual), tolerance=1e-10, max_iterations=0)
    self.assertEqual(selected.statistics.execution_backend, "cuda")
    self.assertIsNone(selected.statistics.breakdown)
    runtime = selected._cuda_runtime
    self.assert_fp64_buffers(runtime)
    self.assertEqual(runtime.domain_count, 1)
    offset = int(runtime.matrix_offsets.get()[0])
    stride = int(runtime.padded_sizes.get()[0])
    stored = runtime.mixed_inverses.get()[offset:offset + stride * stride].reshape(stride, stride)
    inverse = stored[:len(matrix), :len(matrix)]
    self.assertTrue(np.isfinite(inverse).all())
    self.assertLess(np.linalg.norm(inverse - reference) / np.linalg.norm(reference), 2e-5)
    self.assertGreater(float(residual @ inverse @ residual), 0.0)
    expected = reference @ residual
    for reuse in (False, True):
      with self.subTest(reuse_workspace=reuse):
        actual = self.apply(runtime, residual, reuse)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(float(residual @ actual), 0.0)
        self.assertLess(np.linalg.norm(actual - expected) / np.linalg.norm(expected), apply_rtol)
    np.testing.assert_array_equal(view.static_values, matrix.ravel())

  def test_generated_ill_conditioned_bank_storage_and_apply_are_fp64(self):
    matrix, residual, reference = _generated_bank()
    # Cover both a full bank and an exact underfilled bank specialization.
    for padded in (None, 32):
      with self.subTest(padded_size=padded):
        self.check_local_bank(matrix, residual, reference, padded_size=padded)

  def test_saved_24_by_24_bank_has_positive_fp64_quadratic(self):
    matrix, residual, reference, _ = _saved_bank()
    # This condition~5e12 bank involves cancellation even in FP64; compare
    # the resulting vector to an independent Cholesky oracle, not FP32 data.
    self.check_local_bank(matrix, residual, reference, padded_size=32, apply_rtol=5e-3)

  def test_default_heterogeneous_conditional_pcg_converges_and_reuses_runtime(self):
    view, dense, topology = _heterogeneous_system()
    selected = self.make_solver(max_domain_dofs=16, max_levels=3, target_nodes_per_partition=8)
    selected.rebuild_hierarchy_from_blocks(*topology)
    self.assertGreater(selected.hierarchy.number_of_levels, 1)
    runtime = None
    for rhs_host in (np.linspace(-0.8, 0.7, view.rows), np.linspace(0.2, 1.1, view.rows)):
      selected.solve(view, self.gpuarray.to_gpu(rhs_host), tolerance=1e-10, max_iterations=2000)
      stats = selected.statistics
      self.assertEqual(stats.execution_backend, "cuda")
      self.assertTrue(stats.converged, stats)
      self.assertIsNone(stats.breakdown)
      self.assertGreater(stats.iterations, 0)
      self.assertLess(stats.iterations, 2000)
      current = selected._cuda_runtime
      self.assert_fp64_buffers(current)
      if runtime is not None:
        self.assertIs(current, runtime)
      runtime = current
      result = selected.solution.get()
      reference = np.linalg.solve(dense, rhs_host)
      self.assertLess(np.linalg.norm(dense @ result - rhs_host) / np.linalg.norm(rhs_host), 3e-5)
      np.testing.assert_allclose(result, reference, rtol=2e-4, atol=2e-6)


if __name__ == "__main__":
  unittest.main(verbosity=2)

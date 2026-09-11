"""Strict pivot rejection and synchronization checks for MAS GPU inverses.

No dense simulation matrix is built: test matrices are at most 64 by 64.
The CUDA sources are loaded from this checkout, not an installed package.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


ROOT = Path(__file__).resolve().parents[1] / "yasps" / "solver" / "mas" / "cuda"


class InverseValidationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.cache = tempfile.mkdtemp(prefix="yasps_pivot_cuda_")
    cls.modules = {}
    cls.modules["ordinary"] = SourceModule((ROOT / "local_inverse.cu").read_text(), options=["-std=c++17", "-DYASPS_MAS_INVERSE_SIZE=4", "-DYASPS_MAS_INVERSE_GROUPS=2"], no_extern_c=True, cache_dir=cls.cache)
    cls.modules["exact"] = SourceModule((ROOT / "local_inverse_exact.cu").read_text(), options=["-std=c++17", "-DYASPS_MAS_ACTIVE_SIZE=3", "-DYASPS_MAS_STORAGE_STRIDE=4", "-DYASPS_MAS_INVERSE_GROUPS=2"], no_extern_c=True, cache_dir=cls.cache)

  def inverse(self, A, variant, fallback=False, pivot_tolerance=1e-12):
    N = len(A)
    stored = np.eye(4)
    stored[:N, :N] = A
    original = gpuarray.to_gpu(stored.ravel())
    # Legacy "mixed" entry-point names now retain FP64 inverse storage.
    output = gpuarray.empty(16, np.float64)
    sizes = gpuarray.to_gpu(np.array([N], dtype=np.int32))
    status = gpuarray.zeros(1, np.int32)
    failure = gpuarray.zeros(1, np.int32)
    if variant == "exact":
      module, name = self.modules["exact"], "yasps_mas_inverse_gj_exact_strided"
      block, shared, runtime = 6, 2 * (3 * 3 + 3) * 8, 1
    elif variant == "packed":
      module, name = self.modules["ordinary"], "yasps_mas_inverse_gj_packed_specialized"
      block, shared, runtime = 8, 2 * (4 * 4 + 4) * 8, 1
    else:
      module = self.modules["ordinary"]
      name = "yasps_mas_inverse_gauss_jordan" + ("_mixed" if variant == "mixed" else "")
      block, shared, runtime = 96, (2 * 4 * 4 + 4) * 8, 4
    kernel = module.get_function(name)
    kernel(original, output, sizes, status, np.int32(runtime), np.float64(pivot_tolerance), failure, block=(block, 1, 1), grid=(1, 1, 1), shared=shared)
    cuda.Context.synchronize()
    first_status = int(status.get()[0])
    if fallback:
      if variant == "exact":
        name, block, shared = "yasps_mas_inverse_spd_exact_fallback", 32, 3 * 3 * 8
      else:
        name, block, shared = "yasps_mas_inverse_spd_mixed_fallback_specialized", 8, 2 * 4 * 4 * 8
      module.get_function(name)(original, output, sizes, status, np.int32(1), np.float64(pivot_tolerance), failure, block=(block, 1, 1), grid=(1, 1, 1), shared=shared)
      cuda.Context.synchronize()
    # Neither inversion path may overwrite the original Hessian data.
    np.testing.assert_array_equal(original.get(), stored.ravel())
    return output.get().reshape(4, 4)[:N, :N], int(status.get()[0]), first_status

  def test_spd_and_inertia_mass_floor(self):
    for variant in ("generic", "mixed", "packed", "exact"):
      n = 3 if variant == "exact" else 4
      rng = np.random.default_rng(1337)
      R = rng.normal(size=(n, n))
      spd = R @ R.T + np.eye(n)
      cases = [(np.eye(n) * 1e-8, np.eye(n) * 1e8), (spd, np.linalg.inv(spd))]
      for A, expected in cases:
        with self.subTest(variant=variant, diagonal=A.diagonal()):
          result, status, _ = self.inverse(A, variant)
          self.assertEqual(status, 0)
          np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)

  def test_explicit_zero_tolerance_does_not_replace_tiny_positive_pivot(self):
    for variant in ("generic", "mixed", "packed", "exact"):
      n = 3 if variant == "exact" else 4
      for pivot in (0.5e-16, 1e-16, 1e-15, 4e-14):
        A = np.eye(n) * pivot
        inverse, status, _ = self.inverse(A, variant, fallback=True, pivot_tolerance=0.0)
        self.assertEqual(status, 0)
        np.testing.assert_allclose(inverse, np.eye(n) / pivot, rtol=1e-6)

  def test_singular_small_and_nonfinite_pivots_are_rejected(self):
    for variant in ("generic", "mixed", "packed", "exact"):
      n = 3 if variant == "exact" else 4
      for bad in (0.0, 0.5e-16, 1e-13, np.nan, np.inf):
        with self.subTest(variant=variant, pivot=bad):
          A = np.eye(n)
          A[1, 1] = bad
          _, status, _ = self.inverse(A, variant, fallback=variant in ("packed", "exact"))
          self.assertNotEqual(status, 0)

  def test_cholesky_fallback_inverts_ill_scaled_spd(self):
    for variant in ("packed", "exact"):
      n = 3 if variant == "exact" else 4
      diagonal = np.array([1.0, 1e-9, 3.0, 2.0])[:n]
      result, status, first_status = self.inverse(np.diag(diagonal), variant, fallback=True)
      self.assertNotEqual(first_status, 0)
      self.assertEqual(status, 0)
      np.testing.assert_allclose(result, np.diag(1.0 / diagonal), rtol=1e-6)

  def test_negative_pivot_still_fails_fallback(self):
    for variant in ("packed", "exact"):
      n = 3 if variant == "exact" else 4
      diagonal = np.array([1.0, -1e-9, 3.0, 2.0])[:n]
      _, status, first_status = self.inverse(np.diag(diagonal), variant, fallback=True)
      self.assertNotEqual(first_status, 0)
      self.assertNotEqual(status, 0)

  def test_multiple_warps_read_pivot_before_column_overwrite(self):
    # 27-lane groups straddle warp boundaries; 33/64 also span warps.
    # Small one-warp matrices cannot reliably expose this shared-memory race.
    for n, padded, exact in [(24, 32, True), (27, 32, True), (33, 33, False), (64, 64, False)]:
      groups = max(1, 96 // n)
      rng = np.random.default_rng(42)
      # Saturate scheduling: seven matrices masked this race, while a bank
      # of 2,048 underfilled domains reproduced the coupled-scene failure.
      count = 2048 if n == 24 else 7
      R = rng.normal(size=(count, n, n))
      A = R @ R.transpose(0, 2, 1) + np.eye(n)
      stored = np.tile(np.eye(padded), (count, 1, 1))
      stored[:, :n, :n] = A
      if exact:
        options = [f"-DYASPS_MAS_ACTIVE_SIZE={n}", f"-DYASPS_MAS_STORAGE_STRIDE={padded}", f"-DYASPS_MAS_INVERSE_GROUPS={groups}"]
        source, name = "local_inverse_exact.cu", "yasps_mas_inverse_gj_exact_strided"
      else:
        options = [f"-DYASPS_MAS_INVERSE_SIZE={n}", f"-DYASPS_MAS_INVERSE_GROUPS={groups}"]
        source, name = "local_inverse.cu", "yasps_mas_inverse_gj_packed_specialized"
      module = SourceModule((ROOT / source).read_text(), options=["-std=c++17", *options], no_extern_c=True, cache_dir=self.cache)
      input_values = gpuarray.to_gpu(stored.ravel())
      output = gpuarray.empty(stored.size, np.float64)
      sizes = gpuarray.to_gpu(np.full(count, n, np.int32))
      status = gpuarray.zeros(count, np.int32)
      failure = gpuarray.zeros(1, np.int32)
      for repeat in range(3):
        module.get_function(name)(input_values, output, sizes, status, np.int32(count), np.float64(1e-12), failure, block=(groups * n, 1, 1), grid=((count + groups - 1) // groups, 1, 1), shared=groups * (n * n + n) * 8)
        inverse = output.get().reshape(count, padded, padded)[:, :n, :n]
        with self.subTest(size=n, repeat=repeat):
          np.testing.assert_array_equal(status.get(), 0)
          np.testing.assert_allclose(inverse, np.linalg.inv(A), atol=1e-6, rtol=1e-5)
          np.testing.assert_allclose(A @ inverse, np.broadcast_to(np.eye(n), A.shape), atol=1e-5)

  def test_clear_coarse_residual_larger_than_fine_vector(self):
    # Isolated fixed-grid nodes can persist through many hierarchy levels.
    # The direction launch covers fine DOFs, not this longer packed buffer.
    module = SourceModule((ROOT / "pcg.cu").read_text(), options=["-std=c++17"], no_extern_c=True, cache_dir=self.cache)
    count, packed_count = 64, 513
    preconditioned = gpuarray.to_gpu(np.full(count, 2.0))
    direction = gpuarray.to_gpu(np.ones(count))
    state = gpuarray.to_gpu(np.array([0, 1, 3, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float64))
    product = gpuarray.to_gpu(np.ones(count))
    packed = gpuarray.to_gpu(np.ones(packed_count, np.float64))
    module.get_function("yasps_mas_update_direction")(preconditioned, direction, state, product, packed, np.uint32(count), np.uint32(packed_count), block=(64, 1, 1), grid=(1, 1, 1))
    np.testing.assert_array_equal(direction.get(), 2.5)
    np.testing.assert_array_equal(product.get(), 0)
    np.testing.assert_array_equal(packed.get()[:count], 1)
    np.testing.assert_array_equal(packed.get()[count:], 0)


if __name__ == "__main__":
  unittest.main(verbosity=2)

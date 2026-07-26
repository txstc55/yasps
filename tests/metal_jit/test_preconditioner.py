import numpy as np
import pytest

from yasps.backend import gpuarray, is_metal
from yasps.diagonalBlockInverseKernel import diagonalBlockInverseKernel


pytestmark = pytest.mark.skipif(not is_metal, reason="requires Apple Metal")


def test_generated_mixed_size_diagonal_block_evd_inverse():
  values = np.array(
    [
      2.0,
      4.0,
      4.0,
      1.0,
      1.0,
      3.0,
      2.0,
      0.0,
      0.0,
      5.0,
    ],
    np.float32,
  )
  source = gpuarray.to_gpu(values)
  output = gpuarray.zeros(values.size, np.float64)
  inverse = diagonalBlockInverseKernel(
    {1, 2},
    [0, 2, 10],
    [2, 2],
    [1, 2],
    2,
  )

  inverse.computeDiagonalBlockInverse(source, output)

  expected = np.concatenate(
    [
      np.array([0.5, 0.25], np.float32),
      np.linalg.inv(values[2:6].reshape(2, 2)).ravel(),
      np.linalg.inv(values[6:10].reshape(2, 2)).ravel(),
    ]
  )
  np.testing.assert_allclose(output.get(), expected, rtol=3e-5, atol=3e-5)


def test_float32_preconditioner_suppresses_unresolved_inverse_modes():
  values = np.array([1.0e6, 0.0, 0.0, 1.0e-3], np.float32)
  source = gpuarray.to_gpu(values)
  output = gpuarray.zeros(4, np.float64)
  inverse = diagonalBlockInverseKernel({2}, [0, 4], [1], [2], 1)

  inverse.computeDiagonalBlockInverse(source, output)

  np.testing.assert_allclose(
    output.get(),
    np.array([1.0e-6, 0.0, 0.0, 1.0e-3], np.float32),
    rtol=5e-5,
    atol=1e-8,
  )

import numpy as np

from yasps.backend import gpuarray
from yasps.coordinateCompressionKernel import coordinateCompressionKernel


class _Wrt:
  def __init__(self, size):
    self.size = size


def test_coordinate_compression_matches_cuda_layout():
  coordinates = np.array(
    [4, 1, 2, 3, 4, 1, 0, 7, 2, 3],
    dtype=np.uint32
  )
  dimensions = np.array(
    [3, 3, 1, 2, 3, 3, 2, 1, 1, 2],
    dtype=np.uint16
  )
  kernel = coordinateCompressionKernel(
    [gpuarray.to_gpu(coordinates)],
    [gpuarray.to_gpu(dimensions)],
    [5],
    [_Wrt(1), _Wrt(2), _Wrt(3)]
  )

  kernel.compressCoordinatesAndDimensions()

  assert kernel.numUniqueCoordinates == 3
  assert kernel.numUniqueDimensions == 3
  assert kernel.totalBlockSize == 13
  np.testing.assert_array_equal(
    kernel.uniqueCoordinates.get()[:6],
    np.array([2, 3, 0, 7, 4, 1], dtype=np.uint32)
  )
  np.testing.assert_array_equal(
    kernel.uniqueDimensions.get()[:6],
    np.array([1, 2, 2, 1, 3, 3], dtype=np.uint16)
  )
  np.testing.assert_array_equal(
    kernel.uniqueDimensionsOuterIndices.get()[:4],
    np.array([0, 2, 4, 13], dtype=np.uint32)
  )
  np.testing.assert_array_equal(
    kernel.uniqueDimensionsBlockCounts.get()[:3],
    np.array([1, 1, 1], dtype=np.uint32)
  )
  np.testing.assert_array_equal(
    kernel.lookupArray.get(),
    np.array([4, 0, 4, 2, 0], dtype=np.uint32)
  )

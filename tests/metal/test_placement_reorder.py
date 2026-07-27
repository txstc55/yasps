import numpy as np

from yasps.backend import gpuarray
from yasps.placementReorderKernel import placementReorderKernel


class _Correspondence:
  numInstances = 1


class _Energy:
  fullName = "metal_placement_reorder_test"
  correspondance = _Correspondence()


class _GradientIndices:
  outputSizes = gpuarray.to_gpu(
    np.array([1, 1, 1, 1], dtype=np.uint16)
  )
  outputPermutations = gpuarray.to_gpu(
    np.array([1, 2, 3, 4], dtype=np.int16)
  )
  outputCompressedCoordinateCountsOuter = gpuarray.to_gpu(
    np.array([0, 10], dtype=np.uint32)
  )


def test_separate_jacobian_placement_reorder():
  reorder = placementReorderKernel()
  reorder.generateKernel([2, 2], 4, _Energy())
  lookups = gpuarray.to_gpu(
    np.arange(10, 20, dtype=np.uint32)
  )

  reorder.reorderPlacementIndices(_GradientIndices(), lookups)

  np.testing.assert_array_equal(
    reorder.reordered_lookups.get(),
    np.array(
      [10, 11, 14, 12, 13, 15, 16, 17, 18, 19],
      dtype=np.uint32
    )
  )

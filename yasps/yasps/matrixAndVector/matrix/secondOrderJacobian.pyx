from __future__ import annotations

from typing import List, Optional

import numpy as np
import pycuda.gpuarray as gpuarray

from yasps.attribute import attribute
from yasps.codeGenerator import codeGenerator
from yasps.coordinateCompressionKernel import coordinateCompressionKernel
from yasps.helper import timed
from yasps.matrix import matrix
from yasps.secondOrderJacobianIndicesKernel import secondOrderJacobianIndicesKernel
from yasps.secondOrderJacobianKernel import secondOrderJacobianKernel


class secondOrderJacobian(matrix):
  """Rectangular matrix placeholder for a mixed second derivative."""

  def __init__(
    self,
    row_wrt: List[attribute],
    column_wrt: List[attribute]
  ):
    self.__row_wrt = self.__validateTargets(row_wrt, "row_wrt")
    self.__column_wrt = self.__validateTargets(column_wrt, "column_wrt")
    self.__row_start_indices = self.__computeStartIndices(self.__row_wrt)
    self.__column_start_indices = self.__computeStartIndices(self.__column_wrt)

    super().__init__(
      self.__row_start_indices[-1],
      self.__column_start_indices[-1]
    )

    self.__indices_kernels: List[secondOrderJacobianIndicesKernel] = []
    self.__sources: List[attribute] = []
    self.__second_order_jacobians: List[attribute] = []
    self.__computation_kernels: List[Optional[secondOrderJacobianKernel]] = []


    self.__indices_kernels_dynamic: List[secondOrderJacobianIndicesKernel] = []
    self.__sources_dynamic: List[attribute] = []
    self.__second_order_jacobians_dynamic: List[attribute] = []
    self.__computation_kernels_dynamic: List[
      Optional[secondOrderJacobianKernel]
    ] = []

    self.__block_indices_gpu: List[gpuarray.GPUArray] = []
    self.__block_indices_gpu_dynamic: List[gpuarray.GPUArray] = []
    self.__compression_kernel: Optional[coordinateCompressionKernel] = None
    self.__compression_kernel_dynamic: Optional[coordinateCompressionKernel] = None
    self.__is_setup = False

  def __validateTargets(
    self,
    targets: List[attribute],
    name: str
  ) -> List[attribute]:
    if not isinstance(targets, list) or len(targets) == 0:
      raise ValueError(
        f"secondOrderJacobian.__init__: {name} must be a non-empty list."
      )

    result: List[attribute] = []
    seen = set()
    for target in targets:
      if not isinstance(target, attribute):
        raise TypeError(
          f"secondOrderJacobian.__init__: every {name} item must be an attribute."
        )
      if target.isDynamic:
        raise ValueError(
          f"secondOrderJacobian.__init__: {name} cannot contain dynamic attributes."
        )
      if target.hash in seen:
        raise ValueError(
          f"secondOrderJacobian.__init__: {name} contains a duplicate target."
        )
      seen.add(target.hash)
      result.append(target)
    return result

  def __computeStartIndices(self, targets: List[attribute]) -> List[int]:
    starts = [0]
    for target in targets:
      starts.append(
        starts[-1] + target.size * target.correspondance.numInstances
      )
    return starts

  @property
  def row_wrt(self) -> List[attribute]:
    return self.__row_wrt

  @property
  def column_wrt(self) -> List[attribute]:
    return self.__column_wrt

  @property
  def row_start_indices(self) -> List[int]:
    return self.__row_start_indices

  @property
  def column_start_indices(self) -> List[int]:
    return self.__column_start_indices

  @property
  def indices_kernels(self) -> List[secondOrderJacobianIndicesKernel]:
    return self.__indices_kernels

  @indices_kernels.setter
  def indices_kernels(
    self,
    value: List[secondOrderJacobianIndicesKernel]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError("secondOrderJacobian.indices_kernels: value must be a list.")
    if any(not isinstance(item, secondOrderJacobianIndicesKernel) for item in value):
      raise TypeError(
        "secondOrderJacobian.indices_kernels: every item must be a secondOrderJacobianIndicesKernel."
      )
    self.__indices_kernels = list(value)
    self.__is_setup = False

  @property
  def indices_kernels_dynamic(self) -> List[secondOrderJacobianIndicesKernel]:
    return self.__indices_kernels_dynamic

  @indices_kernels_dynamic.setter
  def indices_kernels_dynamic(
    self,
    value: List[secondOrderJacobianIndicesKernel]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.indices_kernels_dynamic: value must be a list."
      )
    if any(not isinstance(item, secondOrderJacobianIndicesKernel) for item in value):
      raise TypeError(
        "secondOrderJacobian.indices_kernels_dynamic: every item must be a secondOrderJacobianIndicesKernel."
      )
    self.__indices_kernels_dynamic = list(value)
    self.__is_setup = False

  @property
  def block_indices_gpu(self) -> List[gpuarray.GPUArray]:
    return self.__block_indices_gpu

  @block_indices_gpu.setter
  def block_indices_gpu(self, value: List[gpuarray.GPUArray]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.block_indices_gpu: value must be a list."
      )
    if any(not isinstance(item, gpuarray.GPUArray) for item in value):
      raise TypeError(
        "secondOrderJacobian.block_indices_gpu: every item must be a GPUArray."
      )
    self.__block_indices_gpu = list(value)

  @property
  def block_indices_gpu_dynamic(self) -> List[gpuarray.GPUArray]:
    return self.__block_indices_gpu_dynamic

  @block_indices_gpu_dynamic.setter
  def block_indices_gpu_dynamic(
    self,
    value: List[gpuarray.GPUArray]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.block_indices_gpu_dynamic: value must be a list."
      )
    if any(not isinstance(item, gpuarray.GPUArray) for item in value):
      raise TypeError(
        "secondOrderJacobian.block_indices_gpu_dynamic: every item must be a GPUArray."
      )
    self.__block_indices_gpu_dynamic = list(value)

  @property
  def second_order_jacobians(self) -> List[attribute]:
    return self.__second_order_jacobians

  @second_order_jacobians.setter
  def second_order_jacobians(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians: value must be a list."
      )
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians: every item must be an attribute."
      )
    self.__second_order_jacobians = list(value)
    self.__is_setup = False

  @property
  def second_order_jacobians_dynamic(self) -> List[attribute]:
    return self.__second_order_jacobians_dynamic

  @second_order_jacobians_dynamic.setter
  def second_order_jacobians_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians_dynamic: value must be a list."
      )
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians_dynamic: every item must be an attribute."
      )
    self.__second_order_jacobians_dynamic = list(value)
    self.__is_setup = False

  @property
  def computation_kernels(self) -> List[Optional[secondOrderJacobianKernel]]:
    return self.__computation_kernels

  @computation_kernels.setter
  def computation_kernels(
    self,
    value: List[Optional[secondOrderJacobianKernel]]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.computation_kernels: value must be a list."
      )
    if any(
      item is not None and not isinstance(item, secondOrderJacobianKernel)
      for item in value
    ):
      raise TypeError(
        "secondOrderJacobian.computation_kernels: every item must be a "
        "secondOrderJacobianKernel or None."
      )
    self.__computation_kernels = list(value)
    self.__is_setup = False

  @property
  def computation_kernels_dynamic(
    self
  ) -> List[Optional[secondOrderJacobianKernel]]:
    return self.__computation_kernels_dynamic

  @computation_kernels_dynamic.setter
  def computation_kernels_dynamic(
    self,
    value: List[Optional[secondOrderJacobianKernel]]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.computation_kernels_dynamic: value must be a "
        "list."
      )
    if any(
      item is not None and not isinstance(item, secondOrderJacobianKernel)
      for item in value
    ):
      raise TypeError(
        "secondOrderJacobian.computation_kernels_dynamic: every item must "
        "be a secondOrderJacobianKernel or None."
      )
    self.__computation_kernels_dynamic = list(value)
    self.__is_setup = False

  @property
  def sources(self) -> List[attribute]:
    return self.__sources

  @sources.setter
  def sources(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("secondOrderJacobian.sources: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.sources: every item must be an attribute."
      )
    self.__sources = list(value)

  @property
  def sources_dynamic(self) -> List[attribute]:
    return self.__sources_dynamic

  @sources_dynamic.setter
  def sources_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.sources_dynamic: value must be a list."
      )
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.sources_dynamic: every item must be an attribute."
      )
    self.__sources_dynamic = list(value)

  def __add__(self, other: secondOrderJacobian):
    if not isinstance(other, secondOrderJacobian):
      raise TypeError(f"secondOrderJacobian.__add__: unsupported operand type(s) for +: 'secondOrderJacobian' and '{type(other).__name__}'")
    if len(self.__row_wrt) != len(other.row_wrt):
      raise ValueError("secondOrderJacobian.__add__: row_wrt length mismatch.")
    for left, right in zip(self.__row_wrt, other.row_wrt):
      if left.hash != right.hash:
        raise ValueError("secondOrderJacobian.__add__: row_wrt mismatch.")
    if len(self.__column_wrt) != len(other.column_wrt):
      raise ValueError("secondOrderJacobian.__add__: column_wrt length mismatch.")
    for left, right in zip(self.__column_wrt, other.column_wrt):
      if left.hash != right.hash:
        raise ValueError("secondOrderJacobian.__add__: column_wrt mismatch.")
    result = secondOrderJacobian(
      self.__row_wrt,
      self.__column_wrt
    )
    result.indices_kernels = self.__indices_kernels + other.indices_kernels
    result.sources = self.__sources + other.sources
    result.second_order_jacobians = self.__second_order_jacobians + other.second_order_jacobians
    result.computation_kernels = self.__computation_kernels + other.computation_kernels
    result.sources_dynamic = self.__sources_dynamic + other.sources_dynamic
    result.indices_kernels_dynamic = self.__indices_kernels_dynamic + other.indices_kernels_dynamic
    result.second_order_jacobians_dynamic = self.__second_order_jacobians_dynamic + other.second_order_jacobians_dynamic
    result.computation_kernels_dynamic = self.__computation_kernels_dynamic + other.computation_kernels_dynamic
    result.block_indices_gpu = self.__block_indices_gpu + other.block_indices_gpu
    result.block_indices_gpu_dynamic = self.__block_indices_gpu_dynamic + other.block_indices_gpu_dynamic
    return result

  # the lookup array for sparse blocks
  def __alignLookups(
    self,
    compression_kernel: coordinateCompressionKernel,
    indices_kernels: List[secondOrderJacobianIndicesKernel]
  ) -> List[gpuarray.GPUArray]:
    compressed_lookups = compression_kernel.lookupArrays
    result: List[gpuarray.GPUArray] = []
    lookup_index = 0
    for item in indices_kernels:
      if item.numTotalCoordinates == 0:
        result.append(gpuarray.empty(0, dtype=np.uint32))
      else:
        result.append(compressed_lookups[lookup_index])
        lookup_index += 1
    return result

  @timed("secondOrderJacobian.getSparseIndices")
  def getSparseIndices(self):
    if len(self.__indices_kernels) == 0:
      return

    for item in self.__indices_kernels:
      item.computeIndices()

    self.__compression_kernel = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels],
      [x.outputBlockDimensions for x in self.__indices_kernels],
      [x.numTotalCoordinates for x in self.__indices_kernels],
      self.__row_wrt,
      self.__column_wrt
    )
    self.__compression_kernel.compressCoordinatesAndDimensions()
    self.__block_indices_gpu = self.__alignLookups(
      self.__compression_kernel,
      self.__indices_kernels
    )

    total_block_size = self.__compression_kernel.totalBlockSize
    if self.blocks_flattened.size < total_block_size:
      self.blocks_flattened = gpuarray.zeros(
        total_block_size,
        dtype=np.float64
      )

    num_unique_dimensions = self.__compression_kernel.numUniqueDimensions
    self.blocks_start_indices = (
      self.__compression_kernel.uniqueDimensionsOuterIndices.get().tolist()[
        :num_unique_dimensions + 1
      ]
    )
    self.block_positions = self.__compression_kernel.uniqueCoordinates
    self.block_counts = (
      self.__compression_kernel.uniqueDimensionsBlockCounts.get().tolist()
    )
    self.block_dimensions = (
      self.__compression_kernel.uniqueDimensions.get().tolist()[
        :num_unique_dimensions * 2
      ]
    )

  @timed("secondOrderJacobian.getSparseIndicesDynamic")
  def getSparseIndicesDynamic(self):
    if len(self.__indices_kernels_dynamic) == 0:
      return

    for item in self.__indices_kernels_dynamic:
      item.computeIndices()

    self.__compression_kernel_dynamic = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels_dynamic],
      [x.outputBlockDimensions for x in self.__indices_kernels_dynamic],
      [x.numTotalCoordinates for x in self.__indices_kernels_dynamic],
      self.__row_wrt,
      self.__column_wrt
    )
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()

    self.__block_indices_gpu_dynamic = self.__alignLookups(
      self.__compression_kernel_dynamic,
      self.__indices_kernels_dynamic
    )

    total_block_size = self.__compression_kernel_dynamic.totalBlockSize
    if self.blocks_flattened_dynamic.size < total_block_size:
      self.blocks_flattened_dynamic = gpuarray.zeros(
        total_block_size,
        dtype=np.float64
      )

    num_unique_dimensions = (
      self.__compression_kernel_dynamic.numUniqueDimensions
    )
    self.blocks_start_indices_dynamic = (
      self.__compression_kernel_dynamic.uniqueDimensionsOuterIndices.get(
      ).tolist()[:num_unique_dimensions + 1]
    )
    self.block_positions_dynamic = (
      self.__compression_kernel_dynamic.uniqueCoordinates
    )
    self.block_counts_dynamic = (
      self.__compression_kernel_dynamic.uniqueDimensionsBlockCounts.get(
      ).tolist()
    )
    self.block_dimensions_dynamic = (
      self.__compression_kernel_dynamic.uniqueDimensions.get().tolist()[
        :num_unique_dimensions * 2
      ]
    )

  @timed("secondOrderJacobian.getSparseIndicesDynamicAgain")
  def getSparseIndicesDynamicAgain(self):
    if len(self.__indices_kernels_dynamic) == 0:
      return
    if self.__compression_kernel_dynamic is None:
      self.getSparseIndicesDynamic()
      return

    for item in self.__indices_kernels_dynamic:
      item.computeIndices()

    self.__compression_kernel_dynamic.updateCoordinates(
      [x.outputCoordinates for x in self.__indices_kernels_dynamic],
      [x.outputBlockDimensions for x in self.__indices_kernels_dynamic],
      [x.numTotalCoordinates for x in self.__indices_kernels_dynamic]
    )
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()

    self.__block_indices_gpu_dynamic = self.__alignLookups(
      self.__compression_kernel_dynamic,
      self.__indices_kernels_dynamic
    )

    total_block_size = self.__compression_kernel_dynamic.totalBlockSize
    if self.blocks_flattened_dynamic.size < total_block_size:
      self.blocks_flattened_dynamic = gpuarray.zeros(
        total_block_size,
        dtype=np.float64
      )

    num_unique_dimensions = (
      self.__compression_kernel_dynamic.numUniqueDimensions
    )
    self.blocks_start_indices_dynamic = (
      self.__compression_kernel_dynamic.uniqueDimensionsOuterIndices.get(
      ).tolist()[:num_unique_dimensions + 1]
    )
    self.block_positions_dynamic = (
      self.__compression_kernel_dynamic.uniqueCoordinates
    )
    self.block_counts_dynamic = (
      self.__compression_kernel_dynamic.uniqueDimensionsBlockCounts.get(
      ).tolist()
    )
    self.block_dimensions_dynamic = (
      self.__compression_kernel_dynamic.uniqueDimensions.get().tolist()[
        :num_unique_dimensions * 2
      ]
    )

  def __ensureTermKernel(self, index: int, dynamic_term: bool) -> None:
    if dynamic_term:
      local_jacobians = self.__second_order_jacobians_dynamic
      indices_kernels = self.__indices_kernels_dynamic
      computation_kernels = self.__computation_kernels_dynamic
    else:
      local_jacobians = self.__second_order_jacobians
      indices_kernels = self.__indices_kernels
      computation_kernels = self.__computation_kernels

    if index >= len(local_jacobians) or index >= len(indices_kernels):
      raise ValueError(
        "secondOrderJacobian.__ensureTermKernel: symbolic term metadata is "
        "incomplete."
      )
    while len(computation_kernels) <= index:
      computation_kernels.append(None)
    if computation_kernels[index] is not None:
      return

    local_jacobian = local_jacobians[index]
    generator = codeGenerator(local_jacobian)
    generator.generateCode()
    kernel = secondOrderJacobianKernel(
      local_jacobian,
      dynamic_term=dynamic_term
    )
    kernel.generateKernel(
      indices_kernels[index].rowIndicesKernel.maxNumIndicesNeeded,
      indices_kernels[index].columnIndicesKernel.maxNumIndicesNeeded
    )
    computation_kernels[index] = kernel

  def __setupCompute(self) -> None:
    if self.__is_setup:
      return
    if len(self.__indices_kernels) != len(self.__second_order_jacobians):
      raise ValueError(
        "secondOrderJacobian.__setupCompute: static coordinate/value term "
        "counts differ."
      )
    if (
      len(self.__indices_kernels_dynamic)
      != len(self.__second_order_jacobians_dynamic)
    ):
      raise ValueError(
        "secondOrderJacobian.__setupCompute: dynamic coordinate/value term "
        "counts differ."
      )

    if len(self.__indices_kernels) > 0:
      self.getSparseIndices()
    if len(self.__indices_kernels_dynamic) > 0:
      self.getSparseIndicesDynamic()
    for index in range(len(self.__indices_kernels)):
      self.__ensureTermKernel(index, False)
    for index in range(len(self.__indices_kernels_dynamic)):
      self.__ensureTermKernel(index, True)
    self.__is_setup = True

  def __computeOneTerm(
    self,
    index: int,
    indices_kernels: List[secondOrderJacobianIndicesKernel],
    computation_kernels: List[Optional[secondOrderJacobianKernel]],
    lookups: List[gpuarray.GPUArray],
    blocks: gpuarray.GPUArray
  ) -> None:
    indices_kernel = indices_kernels[index]
    if indices_kernel.numTotalCoordinates == 0:
      return
    if index >= len(lookups):
      raise ValueError(
        "secondOrderJacobian.__computeOneTerm: sparse lookup is missing."
      )
    if index >= len(computation_kernels):
      raise ValueError(
        "secondOrderJacobian.__computeOneTerm: computation kernel is "
        "missing."
      )
    kernel = computation_kernels[index]
    if kernel is None:
      raise ValueError(
        "secondOrderJacobian.__computeOneTerm: computation kernel is None."
      )
    kernel.compute(indices_kernel, lookups[index], blocks)

  @timed("secondOrderJacobian.compute")
  def compute(self):
    if not self.__is_setup:
      self.__setupCompute()
    elif len(self.__indices_kernels_dynamic) > 0:
      self.getSparseIndicesDynamicAgain()

    self.blocks_flattened.fill(0)
    self.blocks_flattened_dynamic.fill(0)
    for index in range(len(self.__indices_kernels)):
      self.__computeOneTerm(
        index,
        self.__indices_kernels,
        self.__computation_kernels,
        self.__block_indices_gpu,
        self.blocks_flattened
      )
    for index in range(len(self.__indices_kernels_dynamic)):
      self.__computeOneTerm(
        index,
        self.__indices_kernels_dynamic,
        self.__computation_kernels_dynamic,
        self.__block_indices_gpu_dynamic,
        self.blocks_flattened_dynamic
      )
    return self

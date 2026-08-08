from __future__ import annotations

from typing import List

import numpy as np
import pycuda.gpuarray as gpuarray

from yasps.attribute import attribute
from yasps.secondOrderJacobian import secondOrderJacobian


class _sequentialRowIndicesKernel:
  """Provide one sequential row block for every source instance.

  A first-order Jacobian's output attribute is already stored sequentially,
  so its row coordinates do not require a graph traversal.  This adapter has
  the small interface consumed by the rectangular coordinate/value kernels.
  """

  def __init__(self, source: attribute):
    if source.size <= 0:
      raise ValueError(
        "_sequentialRowIndicesKernel: source size must be positive."
      )
    if source.size > np.iinfo(np.uint16).max:
      raise ValueError(
        "_sequentialRowIndicesKernel: source block size exceeds uint16."
      )
    self.__source = source
    self.__output_indices = gpuarray.empty(0, dtype=np.uint32)
    self.__output_sizes = gpuarray.empty(0, dtype=np.uint16)
    self.__output_permutations = gpuarray.empty(0, dtype=np.int16)
    self.__num_instances = 0

  @property
  def maxNumIndicesNeeded(self) -> int:
    return 1

  @property
  def numInstances(self) -> int:
    return self.__num_instances

  @property
  def outputIndices(self):
    return self.__output_indices

  @property
  def outputSizes(self):
    return self.__output_sizes

  @property
  def outputPermutations(self):
    return self.__output_permutations

  def computeIndices(self, unused_start_indices=None) -> None:
    num_instances = self.__source.correspondance.numInstances
    self.__num_instances = num_instances
    if num_instances == 0:
      self.__output_indices = gpuarray.empty(0, dtype=np.uint32)
      self.__output_sizes = gpuarray.empty(0, dtype=np.uint16)
      self.__output_permutations = gpuarray.empty(0, dtype=np.int16)
      return

    if self.__output_indices.size != num_instances:
      self.__output_indices = gpuarray.arange(
        num_instances,
        dtype=np.uint32
      )
      self.__output_sizes = gpuarray.empty(
        num_instances,
        dtype=np.uint16
      )
      self.__output_permutations = gpuarray.empty(
        num_instances,
        dtype=np.int16
      )

    self.__output_indices[:] = (
      gpuarray.arange(num_instances, dtype=np.uint32)
      * np.uint32(self.__source.size)
      + np.uint32(2)
    )
    self.__output_sizes.fill(np.uint16(self.__source.size))
    self.__output_permutations.fill(np.int16(1))


class firstOrderJacobian(secondOrderJacobian):
  """Sparse first derivative of a vector attribute with respect to targets.

  Rows follow the source attribute's ordinary instance-major storage. Columns
  follow the requested target attributes and use the same path-based sparse
  index generation and global coordinate compression as other YASPS matrices.
  """

  def __init__(self, row_source: attribute, wrt: List[attribute]):
    if not isinstance(row_source, attribute):
      raise TypeError(
        "firstOrderJacobian.__init__: row_source must be an attribute."
      )
    if row_source.correspondance is None:
      raise ValueError(
        "firstOrderJacobian.__init__: row_source must have a correspondance."
      )
    if row_source.isDynamic:
      raise ValueError(
        "firstOrderJacobian.__init__: dynamically sized row sources are not "
        "supported because matrix dimensions must remain fixed."
      )
    self.__row_source = row_source
    super().__init__([row_source], wrt)

  @property
  def row_source(self) -> attribute:
    return self.__row_source

  @property
  def wrt(self) -> List[attribute]:
    return self.column_wrt

  def createSequentialRowIndicesKernel(self):
    return _sequentialRowIndicesKernel(self.__row_source)

  def __add__(self, other: firstOrderJacobian):
    if not isinstance(other, firstOrderJacobian):
      raise TypeError(
        "firstOrderJacobian.__add__: the other operand must be a "
        "firstOrderJacobian."
      )
    if self.__row_source.size != other.row_source.size:
      raise ValueError(
        "firstOrderJacobian.__add__: source block sizes differ."
      )
    if (
      self.__row_source.correspondance
      is not other.row_source.correspondance
    ):
      raise ValueError(
        "firstOrderJacobian.__add__: source correspondances differ."
      )
    if len(self.wrt) != len(other.wrt):
      raise ValueError("firstOrderJacobian.__add__: wrt length mismatch.")
    for left, right in zip(self.wrt, other.wrt):
      if left.hash != right.hash:
        raise ValueError("firstOrderJacobian.__add__: wrt mismatch.")

    result = firstOrderJacobian(self.__row_source, self.wrt)
    result.indices_kernels = self.indices_kernels + other.indices_kernels
    result.sources = self.sources + other.sources
    result.second_order_jacobians = (
      self.second_order_jacobians + other.second_order_jacobians
    )
    result.computation_kernels = (
      self.computation_kernels + other.computation_kernels
    )
    result.indices_kernels_dynamic = (
      self.indices_kernels_dynamic + other.indices_kernels_dynamic
    )
    result.sources_dynamic = self.sources_dynamic + other.sources_dynamic
    result.second_order_jacobians_dynamic = (
      self.second_order_jacobians_dynamic
      + other.second_order_jacobians_dynamic
    )
    result.computation_kernels_dynamic = (
      self.computation_kernels_dynamic
      + other.computation_kernels_dynamic
    )
    result.block_indices_gpu = (
      self.block_indices_gpu + other.block_indices_gpu
    )
    result.block_indices_gpu_dynamic = (
      self.block_indices_gpu_dynamic + other.block_indices_gpu_dynamic
    )
    return result

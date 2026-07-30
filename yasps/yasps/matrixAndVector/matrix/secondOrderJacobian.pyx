from __future__ import annotations

from typing import List, Optional

import numpy as np
import pycuda.gpuarray as gpuarray

from yasps.attribute import attribute
from yasps.coordinateCompressionKernel import coordinateCompressionKernel
from yasps.helper import timed
from yasps.matrix import matrix
from yasps.secondOrderJacobianIndicesKernel import secondOrderJacobianIndicesKernel
from yasps.secondOrderJacobianKernel import secondOrderJacobianKernel
from yasps.vector import vector


class secondOrderJacobian(matrix):
  """
  Fully materialized mixed second derivative.

  The first target defines block rows and the second target defines block
  columns.  Unlike ``hessian``, this matrix is rectangular, asymmetric, and
  never projected.  Coordinate compression is optional and disabled by
  default so each generated occurrence owns one stored block.
  """

  def __init__(
    self,
    row_wrt: List[attribute],
    column_wrt: List[attribute],
    dynamic_instances: bool = False,
    compress_coordinates: bool = False
  ):
    self.__row_wrt = self.__validateTargets(row_wrt, "row_wrt")
    self.__column_wrt = self.__validateTargets(column_wrt, "column_wrt")
    self.__row_start_indices = self.__computeStartIndices(self.__row_wrt)
    self.__column_start_indices = self.__computeStartIndices(self.__column_wrt)
    super().__init__(
      self.__row_start_indices[-1],
      self.__column_start_indices[-1]
    )

    self.__dynamic_instances = bool(dynamic_instances)
    self.__compress_coordinates = bool(compress_coordinates)
    self.__indices_kernels: List[secondOrderJacobianIndicesKernel] = []
    self.__indices_kernels_dynamic: List[secondOrderJacobianIndicesKernel] = []
    self.__mixed_derivatives: List[attribute] = []
    self.__mixed_derivatives_dynamic: List[attribute] = []
    # The retained mixed derivative evaluates the chain rule directly:
    # J_row^T H_inner J_column + H_recursive.
    self.__row_outer_jacobians: List[attribute] = []
    self.__column_outer_jacobians: List[attribute] = []
    self.__inner_hessians: List[attribute] = []
    self.__recursive_mixed_terms: List[attribute] = []
    self.__row_outer_jacobians_dynamic: List[attribute] = []
    self.__column_outer_jacobians_dynamic: List[attribute] = []
    self.__inner_hessians_dynamic: List[attribute] = []
    self.__recursive_mixed_terms_dynamic: List[attribute] = []

    self.__lookups: List[gpuarray.GPUArray] = []
    self.__lookups_dynamic: List[gpuarray.GPUArray] = []
    self.__compression_kernel: Optional[coordinateCompressionKernel] = None
    self.__compression_kernel_dynamic: Optional[coordinateCompressionKernel] = None
    self.__numeric_kernel = secondOrderJacobianKernel()
    self.__is_setup = False

  def __validateTargets(self, targets, name):
    if not isinstance(targets, list) or len(targets) == 0:
      raise ValueError(f"secondOrderJacobian.__init__: {name} must be non-empty.")
    result = []
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

  def __computeStartIndices(self, targets):
    starts = [0]
    for target in targets:
      starts.append(
        starts[-1] + target.correspondance.numInstances * target.size
      )
    if starts[-1] <= 0:
      raise ValueError(
        "secondOrderJacobian.__init__: target space must have positive size."
      )
    return starts

  @property
  def row_wrt(self):
    return self.__row_wrt

  @property
  def column_wrt(self):
    return self.__column_wrt

  @property
  def row_start_indices(self):
    return self.__row_start_indices

  @property
  def column_start_indices(self):
    return self.__column_start_indices

  @property
  def compress_coordinates(self):
    return self.__compress_coordinates

  @property
  def row_outer_jacobians(self):
    return self.__row_outer_jacobians

  @property
  def column_outer_jacobians(self):
    return self.__column_outer_jacobians

  @property
  def inner_hessians(self):
    return self.__inner_hessians

  @property
  def recursive_mixed_terms(self):
    return self.__recursive_mixed_terms

  @property
  def indices_kernels(self):
    return self.__indices_kernels

  @property
  def indices_kernels_dynamic(self):
    return self.__indices_kernels_dynamic

  def addTerm(
    self,
    indices_kernel,
    mixed_derivative,
    row_outer_jacobian,
    column_outer_jacobian,
    inner_hessian,
    recursive_mixed_term,
    dynamic=False
  ):
    if not isinstance(indices_kernel, secondOrderJacobianIndicesKernel):
      raise TypeError(
        "secondOrderJacobian.addTerm: indices_kernel has the wrong type."
      )
    for value, name in [
      (mixed_derivative, "mixed_derivative"),
      (row_outer_jacobian, "row_outer_jacobian"),
      (column_outer_jacobian, "column_outer_jacobian"),
      (inner_hessian, "inner_hessian"),
      (recursive_mixed_term, "recursive_mixed_term")
    ]:
      if not isinstance(value, attribute):
        raise TypeError(
          f"secondOrderJacobian.addTerm: {name} must be an attribute."
        )
    if dynamic:
      self.__indices_kernels_dynamic.append(indices_kernel)
      self.__mixed_derivatives_dynamic.append(mixed_derivative)
      self.__row_outer_jacobians_dynamic.append(row_outer_jacobian)
      self.__column_outer_jacobians_dynamic.append(column_outer_jacobian)
      self.__inner_hessians_dynamic.append(inner_hessian)
      self.__recursive_mixed_terms_dynamic.append(recursive_mixed_term)
    else:
      self.__indices_kernels.append(indices_kernel)
      self.__mixed_derivatives.append(mixed_derivative)
      self.__row_outer_jacobians.append(row_outer_jacobian)
      self.__column_outer_jacobians.append(column_outer_jacobian)
      self.__inner_hessians.append(inner_hessian)
      self.__recursive_mixed_terms.append(recursive_mixed_term)
    self.__is_setup = False

  def __add__(self, other):
    if not isinstance(other, secondOrderJacobian):
      raise TypeError(
        "secondOrderJacobian.__add__: both operands must be secondOrderJacobian."
      )
    if [x.hash for x in self.__row_wrt] != [x.hash for x in other.row_wrt]:
      raise ValueError("secondOrderJacobian.__add__: row targets differ.")
    if [x.hash for x in self.__column_wrt] != [
      x.hash for x in other.column_wrt
    ]:
      raise ValueError("secondOrderJacobian.__add__: column targets differ.")
    if self.__compress_coordinates != other.compress_coordinates:
      raise ValueError(
        "secondOrderJacobian.__add__: coordinate compression modes differ."
      )
    result = secondOrderJacobian(
      self.__row_wrt,
      self.__column_wrt,
      self.__dynamic_instances or other.__dynamic_instances,
      self.__compress_coordinates
    )
    for owner in (self, other):
      for i, indices in enumerate(owner.__indices_kernels):
        result.addTerm(
          indices,
          owner.__mixed_derivatives[i],
          owner.__row_outer_jacobians[i],
          owner.__column_outer_jacobians[i],
          owner.__inner_hessians[i],
          owner.__recursive_mixed_terms[i],
          False
        )
      for i, indices in enumerate(owner.__indices_kernels_dynamic):
        result.addTerm(
          indices,
          owner.__mixed_derivatives_dynamic[i],
          owner.__row_outer_jacobians_dynamic[i],
          owner.__column_outer_jacobians_dynamic[i],
          owner.__inner_hessians_dynamic[i],
          owner.__recursive_mixed_terms_dynamic[i],
          True
        )
    return result

  def __applyCompressionMetadata(self, compression, dynamic=False):
    total_block_size = compression.totalBlockSize
    num_dimensions = compression.numUniqueDimensions
    starts = compression.uniqueDimensionsOuterIndices.get().tolist()[
      :num_dimensions + 1
    ]
    counts = compression.uniqueDimensionsBlockCounts.get().tolist()[
      :num_dimensions
    ]
    dimensions = compression.uniqueDimensions.get().tolist()[
      :num_dimensions * 2
    ]
    if dynamic:
      if self.blocks_flattened_dynamic.size < total_block_size:
        self.blocks_flattened_dynamic = gpuarray.zeros(
          total_block_size, dtype=np.float64
        )
      self.blocks_start_indices_dynamic = starts
      self.block_counts_dynamic = counts
      self.block_dimensions_dynamic = dimensions
      self.block_positions_dynamic = compression.uniqueCoordinates
    else:
      if self.blocks_flattened.size < total_block_size:
        self.blocks_flattened = gpuarray.zeros(
          total_block_size, dtype=np.float64
        )
      self.blocks_start_indices = starts
      self.block_counts = counts
      self.block_dimensions = dimensions
      self.block_positions = compression.uniqueCoordinates

  def __alignLookups(self, compression, kernels):
    compressed_lookups = compression.lookupArrays
    result = []
    current = 0
    for kernel in kernels:
      if kernel.numTotalCoordinates == 0:
        result.append(gpuarray.empty(0, dtype=np.uint32))
      else:
        result.append(compressed_lookups[current])
        current += 1
    return result

  @timed("secondOrderJacobian.getSparseIndices")
  def getSparseIndices(self):
    if len(self.__indices_kernels) == 0:
      return
    for kernel in self.__indices_kernels:
      kernel.computeIndices()
    self.__compression_kernel = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels],
      [x.outputBlockDimensions for x in self.__indices_kernels],
      [x.numTotalCoordinates for x in self.__indices_kernels],
      self.__row_wrt,
      self.__compress_coordinates,
      self.__column_wrt
    )
    self.__compression_kernel.compressCoordinatesAndDimensions()
    self.__lookups = self.__alignLookups(
      self.__compression_kernel, self.__indices_kernels
    )
    self.__applyCompressionMetadata(self.__compression_kernel, False)

  @timed("secondOrderJacobian.getSparseIndicesDynamic")
  def getSparseIndicesDynamic(self):
    if len(self.__indices_kernels_dynamic) == 0:
      return
    for kernel in self.__indices_kernels_dynamic:
      kernel.computeIndices()
    coordinates = [x.outputCoordinates for x in self.__indices_kernels_dynamic]
    dimensions = [
      x.outputBlockDimensions for x in self.__indices_kernels_dynamic
    ]
    counts = [x.numTotalCoordinates for x in self.__indices_kernels_dynamic]
    if self.__compression_kernel_dynamic is None:
      self.__compression_kernel_dynamic = coordinateCompressionKernel(
        coordinates,
        dimensions,
        counts,
        self.__row_wrt,
        self.__compress_coordinates,
        self.__column_wrt
      )
    else:
      self.__compression_kernel_dynamic.updateCoordinates(
        coordinates, dimensions, counts
      )
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()
    self.__lookups_dynamic = self.__alignLookups(
      self.__compression_kernel_dynamic, self.__indices_kernels_dynamic
    )
    self.__applyCompressionMetadata(self.__compression_kernel_dynamic, True)

  def __setup(self):
    self.getSparseIndices()
    self.getSparseIndicesDynamic()
    self.__is_setup = True

  @timed("secondOrderJacobian.compute")
  def compute(self):
    if not self.__is_setup:
      self.__setup()
    elif len(self.__indices_kernels_dynamic) > 0:
      self.getSparseIndicesDynamic()

    self.blocks_flattened.fill(0)
    self.blocks_flattened_dynamic.fill(0)
    for i, mixed_derivative in enumerate(self.__mixed_derivatives):
      if self.__indices_kernels[i].numTotalCoordinates == 0:
        continue
      mixed_derivative.compute()
      self.__numeric_kernel.assemble(
        mixed_derivative,
        self.__indices_kernels[i],
        self.__lookups[i],
        self.blocks_flattened
      )
    for i, mixed_derivative in enumerate(self.__mixed_derivatives_dynamic):
      if self.__indices_kernels_dynamic[i].numTotalCoordinates == 0:
        continue
      mixed_derivative.compute()
      self.__numeric_kernel.assemble(
        mixed_derivative,
        self.__indices_kernels_dynamic[i],
        self.__lookups_dynamic[i],
        self.blocks_flattened_dynamic
      )
    return self

  def __spmvRepresentation(
    self,
    positions,
    blocks,
    starts,
    counts,
    dimensions,
    x,
    output,
    transpose
  ):
    coordinate_offset = 0
    for category, count in enumerate(counts):
      h = dimensions[2 * category]
      w = dimensions[2 * category + 1]
      self.__numeric_kernel.spmvCategory(
        positions[2 * coordinate_offset:2 * (coordinate_offset + count)],
        blocks[starts[category]:starts[category + 1]],
        count,
        h,
        w,
        x,
        output,
        transpose
      )
      coordinate_offset += count

  def matVecProduct(self, x: vector):
    if not isinstance(x, vector) or x.size != self.cols:
      raise ValueError(
        "secondOrderJacobian.matVecProduct: input vector has wrong size."
      )
    output = vector(self.rows)
    self.matVecProductInPlace(x, output)
    return output

  def matVecProductInPlace(self, x: vector, output: vector):
    if x.size != self.cols or output.size != self.rows:
      raise ValueError(
        "secondOrderJacobian.matVecProductInPlace: vector size mismatch."
      )
    output.value.fill(0)
    self.__spmvRepresentation(
      self.block_positions,
      self.blocks_flattened,
      self.blocks_start_indices,
      self.block_counts,
      self.block_dimensions,
      x.value,
      output.value,
      False
    )
    self.__spmvRepresentation(
      self.block_positions_dynamic,
      self.blocks_flattened_dynamic,
      self.blocks_start_indices_dynamic,
      self.block_counts_dynamic,
      self.block_dimensions_dynamic,
      x.value,
      output.value,
      False
    )

  def transposeMatVecProduct(self, x: vector):
    if not isinstance(x, vector) or x.size != self.rows:
      raise ValueError(
        "secondOrderJacobian.transposeMatVecProduct: input has wrong size."
      )
    output = vector(self.cols)
    self.transposeMatVecProductInPlace(x, output)
    return output

  def transposeMatVecProductInPlace(self, x: vector, output: vector):
    if x.size != self.rows or output.size != self.cols:
      raise ValueError(
        "secondOrderJacobian.transposeMatVecProductInPlace: size mismatch."
      )
    output.value.fill(0)
    self.__spmvRepresentation(
      self.block_positions,
      self.blocks_flattened,
      self.blocks_start_indices,
      self.block_counts,
      self.block_dimensions,
      x.value,
      output.value,
      True
    )
    self.__spmvRepresentation(
      self.block_positions_dynamic,
      self.blocks_flattened_dynamic,
      self.blocks_start_indices_dynamic,
      self.block_counts_dynamic,
      self.block_dimensions_dynamic,
      x.value,
      output.value,
      True
    )

  def __addDenseRepresentation(
    self,
    dense,
    positions_gpu,
    blocks_gpu,
    starts,
    counts,
    dimensions
  ):
    if len(counts) == 0:
      return
    total_coordinates = sum(counts)
    positions = positions_gpu.get()[:2 * total_coordinates]
    blocks = blocks_gpu.get()
    coordinate_offset = 0
    for category, count in enumerate(counts):
      h = dimensions[2 * category]
      w = dimensions[2 * category + 1]
      data_offset = starts[category]
      for local_block in range(count):
        coordinate_index = coordinate_offset + local_block
        row = positions[2 * coordinate_index]
        col = positions[2 * coordinate_index + 1]
        begin = data_offset + local_block * h * w
        block = blocks[begin:begin + h * w].reshape(h, w)
        dense[row:row + h, col:col + w] += block
      coordinate_offset += count

  def toDense(self):
    dense = np.zeros((self.rows, self.cols), dtype=np.float64)
    self.__addDenseRepresentation(
      dense,
      self.block_positions,
      self.blocks_flattened,
      self.blocks_start_indices,
      self.block_counts,
      self.block_dimensions
    )
    self.__addDenseRepresentation(
      dense,
      self.block_positions_dynamic,
      self.blocks_flattened_dynamic,
      self.blocks_start_indices_dynamic,
      self.block_counts_dynamic,
      self.block_dimensions_dynamic
    )
    return dense

from __future__ import annotations
from yasps.backend import autoinit
import numpy as np
from yasps.backend import gpuarray
from typing import List


from yasps.vector import vector


class matrix:
  def __init__(self, rows: int = 0, cols: int = 0):
    if rows < 0 or cols < 0:
      raise ValueError("matrix.__init__: rows and cols must be non-negative.")
    self.__rows: int = rows
    self.__cols: int = cols

    # Block sparse representation, the static part
    self.__block_dimensions: List[int] = []  # record unique block dimensions (flattened)
    self.__blocks_flattened: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.float64)  # flattened block storage
    self.__blocks_start_indices: List[int] = []  # start index for each block size in the flattened storage
    self.__block_positions: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.uint32)  # coordinates for each block
    self.__block_counts: List[int] = []  # count for each block-size category

    # block sparse representation, the dynamic part
    self.__block_dimensions_dynamic: List[int] = []  # unique dynamic block dimensions (flattened)
    self.__blocks_flattened_dynamic: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.float64)  # flattened dynamic block storage
    self.__blocks_start_indices_dynamic: List[int] = []  # dynamic block category start indices
    self.__block_positions_dynamic: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.uint32)  # coordinates for dynamic blocks
    self.__block_counts_dynamic: List[int] = []  # count for each dynamic block-size category

  def __mul__(self, other: vector) -> vector:
    if not isinstance(other, vector):
      raise TypeError(f"matrix.__mul__: unsupported operand type(s) for *: 'matrix' and '{type(other).__name__}'")
    return self.matVecProduct(other)

  @property
  def rows(self) -> int:
    return self.__rows

  @property
  def cols(self) -> int:
    return self.__cols

  @property
  def block_dimensions(self) -> List[int]:
    """Static unique block sizes used by this matrix."""
    return self.__block_dimensions

  @block_dimensions.setter
  def block_dimensions(self, dims: List[int]) -> None:
    if not isinstance(dims, list):
      raise TypeError("matrix.block_dimensions: dims must be a list of int.")
    self.__block_dimensions = [int(x) for x in dims]

  @property
  def blocks_flattened(self) -> gpuarray.GPUArray:
    """Flattened static block storage buffer."""
    return self.__blocks_flattened

  @blocks_flattened.setter
  def blocks_flattened(self, values) -> None:
    if values is None:
      self.__blocks_flattened = gpuarray.empty(0, dtype=np.float64)
      return
    if isinstance(values, np.ndarray):
      values = gpuarray.to_gpu(np.asarray(values, dtype=np.float64).ravel())
    elif not isinstance(values, gpuarray.GPUArray):
      raise TypeError("matrix.blocks_flattened: values must be a numpy array or a GPUArray.")
    if values.dtype != np.float64:
      values = values.astype(np.float64)
    if values.ndim != 1:
      values = values.ravel()
    self.__blocks_flattened = values

  @property
  def blocks_start_indices(self) -> List[int]:
    """Start indices for each static block-size category."""
    return self.__blocks_start_indices

  @blocks_start_indices.setter
  def blocks_start_indices(self, starts: List[int]) -> None:
    if not isinstance(starts, list):
      raise TypeError("matrix.blocks_start_indices: starts must be a list of int.")
    self.__blocks_start_indices = [int(x) for x in starts]

  @property
  def block_positions(self) -> gpuarray.GPUArray:
    """Coordinates associated with each static block."""
    return self.__block_positions

  @block_positions.setter
  def block_positions(self, positions) -> None:
    if positions is None:
      self.__block_positions = gpuarray.empty(0, dtype=np.uint32)
      return
    if isinstance(positions, np.ndarray):
      positions = gpuarray.to_gpu(np.asarray(positions, dtype=np.uint32).ravel())
    elif not isinstance(positions, gpuarray.GPUArray):
      raise TypeError("matrix.block_positions: positions must be a numpy array or a GPUArray.")
    if positions.dtype != np.uint32:
      positions = positions.astype(np.uint32)
    if positions.ndim != 1:
      positions = positions.ravel()
    self.__block_positions = positions

  @property
  def block_counts(self) -> List[int]:
    """Number of blocks per static block-size category."""
    return self.__block_counts

  @block_counts.setter
  def block_counts(self, counts: List[int]) -> None:
    if not isinstance(counts, list):
      raise TypeError("matrix.block_counts: counts must be a list of int.")
    self.__block_counts = [int(x) for x in counts]

  @property
  def block_dimensions_dynamic(self) -> List[int]:
    """Dynamic unique block sizes used by this matrix."""
    return self.__block_dimensions_dynamic

  @block_dimensions_dynamic.setter
  def block_dimensions_dynamic(self, dims: List[int]) -> None:
    if not isinstance(dims, list):
      raise TypeError("matrix.block_dimensions_dynamic: dims must be a list of int.")
    self.__block_dimensions_dynamic = [int(x) for x in dims]

  @property
  def blocks_flattened_dynamic(self) -> gpuarray.GPUArray:
    """Flattened dynamic block storage buffer."""
    return self.__blocks_flattened_dynamic

  @blocks_flattened_dynamic.setter
  def blocks_flattened_dynamic(self, values) -> None:
    if values is None:
      self.__blocks_flattened_dynamic = gpuarray.empty(0, dtype=np.float64)
      return
    if isinstance(values, np.ndarray):
      values = gpuarray.to_gpu(np.asarray(values, dtype=np.float64).ravel())
    elif not isinstance(values, gpuarray.GPUArray):
      raise TypeError("matrix.blocks_flattened_dynamic: values must be a numpy array or a GPUArray.")
    if values.dtype != np.float64:
      values = values.astype(np.float64)
    if values.ndim != 1:
      values = values.ravel()
    self.__blocks_flattened_dynamic = values

  @property
  def blocks_start_indices_dynamic(self) -> List[int]:
    """Start indices for each dynamic block-size category."""
    return self.__blocks_start_indices_dynamic

  @blocks_start_indices_dynamic.setter
  def blocks_start_indices_dynamic(self, starts: List[int]) -> None:
    if not isinstance(starts, list):
      raise TypeError("matrix.blocks_start_indices_dynamic: starts must be a list of int.")
    self.__blocks_start_indices_dynamic = [int(x) for x in starts]

  @property
  def block_positions_dynamic(self) -> gpuarray.GPUArray:
    """Coordinates associated with each dynamic block."""
    return self.__block_positions_dynamic

  @block_positions_dynamic.setter
  def block_positions_dynamic(self, positions) -> None:
    if positions is None:
      self.__block_positions_dynamic = gpuarray.empty(0, dtype=np.uint32)
      return
    if isinstance(positions, np.ndarray):
      positions = gpuarray.to_gpu(np.asarray(positions, dtype=np.uint32).ravel())
    elif not isinstance(positions, gpuarray.GPUArray):
      raise TypeError("matrix.block_positions_dynamic: positions must be a numpy array or a GPUArray.")
    if positions.dtype != np.uint32:
      positions = positions.astype(np.uint32)
    if positions.ndim != 1:
      positions = positions.ravel()
    self.__block_positions_dynamic = positions

  @property
  def block_counts_dynamic(self) -> List[int]:
    """Number of blocks per dynamic block-size category."""
    return self.__block_counts_dynamic

  @block_counts_dynamic.setter
  def block_counts_dynamic(self, counts: List[int]) -> None:
    if not isinstance(counts, list):
      raise TypeError("matrix.block_counts_dynamic: counts must be a list of int.")
    self.__block_counts_dynamic = [int(x) for x in counts]

  # Backward-compatible camelCase aliases for matrix-wide representation attributes.
  @property
  def blockDimensions(self) -> List[int]:
    return self.__block_dimensions

  @blockDimensions.setter
  def blockDimensions(self, dims: List[int]) -> None:
    self.block_dimensions = dims

  @property
  def blocksFlattened(self) -> gpuarray.GPUArray:
    return self.__blocks_flattened

  @blocksFlattened.setter
  def blocksFlattened(self, values) -> None:
    self.blocks_flattened = values

  @property
  def blocksStartIndices(self) -> List[int]:
    return self.__blocks_start_indices

  @blocksStartIndices.setter
  def blocksStartIndices(self, starts: List[int]) -> None:
    self.blocks_start_indices = starts

  @property
  def blockPositions(self) -> gpuarray.GPUArray:
    return self.__block_positions

  @blockPositions.setter
  def blockPositions(self, positions) -> None:
    self.block_positions = positions

  @property
  def blockCounts(self) -> List[int]:
    return self.__block_counts

  @blockCounts.setter
  def blockCounts(self, counts: List[int]) -> None:
    self.block_counts = counts

  def matVecProduct(self, x: vector) -> vector:
    if x.size != self.__cols:
      raise ValueError("matrix.matVecProduct: input vector has wrong size.")
    result = vector(self.__rows)
    # Placeholder for real sparse-block matvec implementation.
    return result

  def matVecProductInPlace(self, x: vector, out: vector) -> None:
    if x.size != self.__cols:
      raise ValueError("matrix.matVecProductInPlace: input vector has wrong size.")
    if out.size != self.__rows:
      raise ValueError("matrix.matVecProductInPlace: output vector has wrong size.")
    # Placeholder for real sparse-block matvec implementation.
    pass

  def setDimensions(self, rows: int, cols: int) -> None:
    if rows < 0 or cols < 0:
      raise ValueError("matrix.setDimensions: rows and cols must be non-negative.")
    self.__rows = rows
    self.__cols = cols

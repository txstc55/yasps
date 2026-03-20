from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import List


from yasps.vector import vector


class matrix:
  def __init__(self, rows: int = 0, cols: int = 0):
    if rows < 0 or cols < 0:
      raise ValueError("matrix.__init__: rows and cols must be non-negative.")
    self.__rows: int = rows
    self.__cols: int = cols

    # Block sparse representation
    self.__blockDimensions: List[int] = []  # record unique block dimensions (flattened)
    self.__blocksFlattened: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)  # flattened block storage
    self.__blocksStartIndices: List[int] = []  # start index for each block size in the flattened storage
    self.__blockPositions: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.uint32)  # coordinates for each block
    self.__blockCounts: List[int] = []  # count for each block-size category

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
  def blockDimensions(self) -> List[int]:
    return self.__blockDimensions

  @blockDimensions.setter
  def blockDimensions(self, dims: List[int]) -> None:
    if not isinstance(dims, list):
      raise TypeError("matrix.blockDimensions: dims must be a list of int.")
    self.__blockDimensions = [int(x) for x in dims]

  @property
  def blocksFlattened(self) -> gpuarray.GPUArray:
    return self.__blocksFlattened

  @blocksFlattened.setter
  def blocksFlattened(self, values) -> None:
    if values is None:
      self.__blocksFlattened = gpuarray.empty(0, dtype = np.float64)
      return
    if isinstance(values, np.ndarray):
      values = gpuarray.to_gpu(np.asarray(values, dtype = np.float64).ravel())
    elif not isinstance(values, gpuarray.GPUArray):
      raise TypeError("matrix.blocksFlattened: values must be a numpy array or a GPUArray.")
    if values.dtype != np.float64:
      values = values.astype(np.float64)
    if values.ndim != 1:
      values = values.ravel()
    self.__blocksFlattened = values

  @property
  def blocksStartIndices(self) -> List[int]:
    return self.__blocksStartIndices

  @blocksStartIndices.setter
  def blocksStartIndices(self, starts: List[int]) -> None:
    if not isinstance(starts, list):
      raise TypeError("matrix.blocksStartIndices: starts must be a list of int.")
    self.__blocksStartIndices = [int(x) for x in starts]

  @property
  def blockPositions(self) -> gpuarray.GPUArray:
    return self.__blockPositions

  @blockPositions.setter
  def blockPositions(self, positions) -> None:
    if positions is None:
      self.__blockPositions = gpuarray.empty(0, dtype = np.uint32)
      return
    if isinstance(positions, np.ndarray):
      positions = gpuarray.to_gpu(np.asarray(positions, dtype = np.uint32).ravel())
    elif not isinstance(positions, gpuarray.GPUArray):
      raise TypeError("matrix.blockPositions: positions must be a numpy array or a GPUArray.")
    if positions.dtype != np.uint32:
      positions = positions.astype(np.uint32)
    if positions.ndim != 1:
      positions = positions.ravel()
    self.__blockPositions = positions

  @property
  def blockCounts(self) -> List[int]:
    return self.__blockCounts

  @blockCounts.setter
  def blockCounts(self, counts: List[int]) -> None:
    if not isinstance(counts, list):
      raise TypeError("matrix.blockCounts: counts must be a list of int.")
    self.__blockCounts = [int(x) for x in counts]

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

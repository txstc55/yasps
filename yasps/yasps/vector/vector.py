# cython: language_level=3
from __future__ import annotations
import pycuda.gpuarray as gpuarray
from typing import Union
import numpy as np
class vector:
  def __init__(self):
    self.__data: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__size: int = 0
    pass


  @property
  def size(self) -> int:
    return self.__size

  @property
  def data(self) -> gpuarray.GPUArray:
    return self.__data[:self.__size]  # return only the valid portion of the data


  def resize(self, new_size: int) -> None:
    if new_size < 0:
      raise ValueError("vector.resize: new_size cannot be negative.")
    if new_size <= self.__data.size:
      self.__size = new_size
      return
    self.__data = gpuarray.empty(new_size, dtype = np.float64)
    self.__size = new_size
    return

  def updateValue(self, new_value: Union[np.ndarray, gpuarray.GPUArray]) -> None:
    n = self.__size
    if isinstance(new_value, np.ndarray):
      arr = np.asarray(new_value, dtype=np.float64).ravel()   # flatten to 1D
      if arr.size != n:
          raise ValueError("vector.updateValue: new_value size must match vector size.")
      # host -> device copy into existing buffer (no allocation)
      self.__data[:n].set(arr)
    elif isinstance(new_value, gpuarray.GPUArray):
      if new_value.size != n:
        raise ValueError("vector.updateValue: new_value size must match vector size.")

      # If dtype matches, do direct device->device copy:
      if new_value.dtype == np.float64:
        self.__data[:n].set(new_value)  # device->device copy
      else:
        # cast then copy (this may allocate a temporary)
        tmp = new_value.astype(np.float64)
        self.__data[:n].set(tmp)
    else:
      raise TypeError("vector.updateValue: new_value must be either a numpy array or a GPU array.")


  def __add__(self, other: vector) -> vector:
    if self.__size != other.__size:
      raise ValueError("vector.__add__: vectors must have the same size.")

    result = vector()
    result.resize(self.__size)
    # in-place addition into preallocated memory
    result.__data[:] = self.__data[:self.__size] + other.__data[:other.__size]
    return result

  def __neg__(self) -> vector:
    result = vector()
    result.resize(self.__size)
    result.__data[:] = -self.__data[:self.__size]
    return result

  def __sub__(self, other: vector) -> vector:
    if self.__size != other.__size:
      raise ValueError("vector.__sub__: vectors must have the same size.")
    result = vector()
    result.resize(self.__size)
    result.__data[:] = self.__data[:self.__size] - other.__data[:other.__size]
    return result

  def __mul__(self, other: Union[int, float, vector]) -> vector:
    if isinstance(other, (int, float)):
      result = vector()
      result.resize(self.__size)
      result.__data[:] = self.__data[:self.__size] * other
      return result
    elif isinstance(other, vector):
      if self.__size != other.__size:
        raise ValueError("vector.__mul__: vectors must have the same size.")
      result = vector()
      result.resize(self.__size)
      result.__data[:] = self.__data[:self.__size] * other.__data[:other.__size]
      return result
    else:
      raise TypeError("vector.__mul__: unsupported operand type(s) for *: 'vector' and '{}'".format(type(other).__name__))

  def dot(self, other: vector) -> float:
    if self.__size != other.__size:
      raise ValueError("vector.dot: vectors must have the same size.")
    return gpuarray.dot(self.__data[:self.__size], other.__data[:other.__size]).get()

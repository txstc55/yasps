from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from numbers import Real
from typing import List, Tuple, Set, Optional

class vector:
  def __init__(self, size: int):
    assert size >= 0, f'vector.__init__: size must be grater than 0, current size: {size}'
    self.__size: int = size
    self.__value : gpuarray.GPUArray = gpuarray.zeros(size, np.float64) # zero array initialization
    pass

  @property
  def size(self) -> int:
    return self.__size

  @property
  def value(self) -> gpuarray.GPUArray:
    return self.__value

  @value.setter
  def value(self, new_value):
    assert new_value.size == self.__size, f'vector.value: size must match, old size: {self.__size}, new size: {new_value.size}'
    self.__value = new_value

  def updateValue(self, new_value) -> None:
    if isinstance(new_value, np.ndarray):
      arr = np.asarray(new_value, dtype=np.float64).ravel()
      if arr.size != self.__size:
        raise ValueError(f"vector.updateValue: size must match, vector size: {self.__size}, input size: {arr.size}")
      self.__value.set(arr)
      return

    if isinstance(new_value, gpuarray.GPUArray):
      if int(new_value.size) != self.__size:
        raise ValueError(f"vector.updateValue: size must match, vector size: {self.__size}, input size: {new_value.size}")
      if new_value.dtype == np.float64:
        self.__value.set(new_value)
      else:
        self.__value.set(new_value.astype(np.float64))
      return

    if isinstance(new_value, vector):
      if new_value.size != self.__size:
        raise ValueError(f"vector.updateValue: size must match, vector size: {self.__size}, input size: {new_value.size}")
      self.__value.set(new_value.value)
      return

    raise TypeError(
      "vector.updateValue: new_value must be numpy.ndarray, pycuda.gpuarray.GPUArray, or vector."
    )

  def resize(self, new_size):
    assert new_size >= 0, f'vector.resize: new size must be grater than 0, new size: {new_size}'
    self.__size = new_size

  def __add__(self, other: vector) -> vector:
    if not isinstance(other, vector):
      raise TypeError(f"vector.__add__: unsupported operand type(s) for +: 'vector' and '{type(other).__name__}'")
    if self.__size != other.__size:
      raise ValueError("vector.__add__: vector sizes must match.")
    result = vector(self.__size)
    result.value = self.__value + other.__value
    return result

  def __sub__(self, other: vector) -> vector:
    if not isinstance(other, vector):
      raise TypeError(f"vector.__sub__: unsupported operand type(s) for -: 'vector' and '{type(other).__name__}'")
    if self.__size != other.__size:
      raise ValueError("vector.__sub__: vector sizes must match.")
    result = vector(self.__size)
    result.value = self.__value - other.__value
    return result

  def __subtract__(self, other: vector) -> vector:
    return self.__sub__(other)

  def __neg__(self) -> vector:
    result = vector(self.__size)
    result.value = -self.__value
    return result

  def __mul__(self, scalar: Real) -> vector:
    if not isinstance(scalar, Real):
      raise TypeError(f"vector.__mul__: only scalar multiplication is supported, got '{type(scalar).__name__}'")
    result = vector(self.__size)
    result.value = self.__value * float(scalar)
    return result

  def __rmul__(self, scalar: Real) -> vector:
    return self.__mul__(scalar)

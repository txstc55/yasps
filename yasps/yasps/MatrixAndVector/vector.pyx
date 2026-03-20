from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
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

  def resize(self, new_size):
    assert new_size >= 0, f'vector.resize: new size must be grater than 0, new size: {new_size}'
    self.__size = new_size

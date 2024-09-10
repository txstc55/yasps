from __future__ import annotations
# from ..symbolic import symbolicMatrix
# from ..symbolic import symbolic
import numpy as np
import pycuda.gpuarray as gpuarray
class attribute:
  def __init__(self, name: str, rows: int = 1, cols: int = 1, correspondance = [], fetch_from = [], value = None):
    self.__value = None
    self.__name = name
    self.__n_rows = rows
    self.__n_cols = cols
    self.__correspondance = correspondance
    self.__fetch_from = fetch_from
    # update the value
    self.updateValue(value)


  @property
  def name(self)->str:
    return self.__name

  @property
  def rows(self)->int:
    return self.__n_rows

  @property
  def cols(self)->int:
    return self.__n_cols

  @property
  def correspondance(self):
    return self.__correspondance

  @property
  def fetch_from(self):
    return self.__fetch_from

  @property
  def value(self):
    print("here at value")
    if self.__value is None:
      raise ValueError("Attribute.value: value is None. Please call compute() first or manually update value.")
    return self.__value

  def updateValue(self, value):
    # check value array for gpu array conversion
    if isinstance(value, np.ndarray):
      # flatten the array and convert to gpu array
      self.__value = gpuarray.to_gpu(value.flatten())
    elif isinstance(value, gpuarray.GPUArray):
      self.__value = value.ravel()
    else:
      try:
        self.__value = gpuarray.to_gpu(np.array(value).flatten())
      except:
        self.__value = None
        raise ValueError("Attribute.updateValue: Invalid value type, cannot be converted to gpuarray")

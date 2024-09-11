from __future__ import annotations
# from ..symbolic import symbolicMatrix
# from ..symbolic import symbolic
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional
from typing import List
class attribute:
  def __init__(self, name: str, attribute: Optional['attribute'] = None, rows: int = 1, cols: int = 1, correspondance = [], fetch_from = [], value = None):
    if attribute is not None:
      # copy the attribute
      self.__value = attribute.value
      self.__name = attribute.name
      self.__n_rows = attribute.rows
      self.__n_cols = attribute.cols
      self.__correspondance = attribute.correspondance
      self.__fetch_from = attribute.fetch_from
      self.__children = attribute.children

    self.__value = None
    self.__name = name
    self.__n_rows = rows
    self.__n_cols = cols
    self.__correspondance = correspondance
    self.__fetch_from = fetch_from
    self.__children = []
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
  def children(self):
    return self.__children

  @property
  def value(self):
    print("here at value")
    if self.__value is None:
      raise ValueError("attribute.value: value is None. Please call compute() first or manually update value.")
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
        raise ValueError("attribute.updateValue: Invalid value type, cannot be converted to gpuarray")

  def concat(self, other: attribute) -> attribute:
    # check if the other attribute has the same correspondance
    if self.__correspondance != other.correspondance:
      raise ValueError("attribute.concat: attributes do not have the same correspondance.")
    return self

  @staticmethod
  def concatenate(attributes: List[attribute]) -> attribute:
    if len(attributes) == 0:
      raise ValueError("attribute.concatenate: attributes list is empty.")
    if len(attributes) == 1:
      return attributes[0]

    # sequesntially concatenate the results
    result = attributes[0].concat(attributes[1])
    for i in range(2, len(attributes)):
      result = result.concat(attributes[i])
    return result

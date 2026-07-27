# cython: language_level=3
from __future__ import annotations
import numpy as np
from yasps.backend import gpuarray
from itertools import accumulate
from typing import TYPE_CHECKING, List, Union
if TYPE_CHECKING:
  from yasps.primitive import primitive
  from yasps.mesh import mesh as ymesh
  from yasps.scene import scene as yscene

class connectivity:
  def __init__(self, name: str, from_primitive: primitive, to_primitive: primitive, value: Union[np.ndarray, List[List[int]]], dimension: int):
    # for example the connectivity between triangle and vertex
    # one triangle contains 3 vertices
    # so the dimension is 3
    # from is triangle
    # to is vertex
    self.__name: str = name
    self.__fromPrimitive: primitive = from_primitive
    self.__toPrimitive: primitive = to_primitive
    self.__value: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32)
    if dimension != 0:
      self.__value: gpuarray.GPUArray = gpuarray.to_gpu(np.array(value).flatten().astype(np.uint32))
      self.__dimension: int = dimension
      self.__compressedRows: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32)
    else:
      # if the dimension is 0
      # this means the connectivity is not fixed
      # for example the connectivity between vertex to triangle
      # one vertex may connect to unknown number of triangles
      # in this case, we will need to generate a CSR format for quick navigation
      flattened_list: List[int] = [item for sublist in value for item in sublist]
      self.__value: gpuarray.GPUArray = gpuarray.to_gpu(np.array(flattened_list).flatten().astype(np.uint32))
      # the value can only be a list of list
      lengths: List[int] = [len(x) for x in value]
      # Using accumulate to generate the prefix sum directly
      prefix_sum: List[int] = [0] + list(accumulate(lengths))
      self.__dimension: int = dimension
      self.__compressedRows: gpuarray.GPUArray = gpuarray.to_gpu(np.array(prefix_sum).astype(np.uint32))




  @property
  def name(self)->str:
    return self.__name

  @property
  def fullName(self)->str:
    return f"from_{self.fromPrimitive.fullName}_to_{self.toPrimitive.fullName}_through_{self.name}"

  @property
  def fromPrimitive(self)->primitive:
    return self.__fromPrimitive

  @property
  def toPrimitive(self)->primitive:
    return self.__toPrimitive

  @property
  def value(self)->gpuarray.GPUArray:
    return self.__value

  @property
  def dimension(self)->int:
    return self.__dimension

  @property
  def mesh(self)->ymesh:
    return self.fromPrimitive.mesh

  @property
  def scene(self)->yscene:
    return self.mesh.scene

  @property
  def type(self)->str:
    return "connectivity"

  @property
  def compressedRows(self)->gpuarray.GPUArray:
    return self.__compressedRows


  @property
  def code_generation_index_name(self) -> str:
    return f'{self.fullName}_global_indices'

  @property
  def code_generation_csr_name(self) -> str:
    return f'{self.fullName}_compressed_row_indices'

  # for updating connectivity when the primitive has dynamic count
  def updateConnectivity(self, value: Union[np.ndarray, List[List[int]], gpuarray.GPUArray]):
    ## check if we can reserve space by not reallocating
    oldGPUArraySize: int = int(self.__value.size)
    if isinstance(value, gpuarray.GPUArray):
      if oldGPUArraySize > int(value.size):
        # print("Old value shape", self.__value.shape)
        # print("new value shape", value.shape)
        self.__value[:value.size] = value
      else:
        # print("new shape is", value.shape)
        new_gpu_array = gpuarray.empty_like(value)
        new_gpu_array[:] = value  # Device-to-device copy
        self.__value = new_gpu_array
      return

    newCPUArray = np.array(value).flatten().astype(np.uint32)
    newGPUArraySize: int = newCPUArray.size
    # now we set the new value
    if oldGPUArraySize < newGPUArraySize:
      self.__value = gpuarray.to_gpu(newCPUArray)
    else:
      self.__value[:newGPUArraySize] = gpuarray.to_gpu(newCPUArray)
      self.__compressedRows = gpuarray.empty(0, dtype = np.uint32)
      self.__dimension = len(value[0])

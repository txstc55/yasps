from __future__ import annotations
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.primitive import primitive
  from yasps.mesh import mesh
  from yasps.scene import scene

class connectivity:
  def __init__(self, name: str, from_primitive: primitive, to_primitive: primitive, value: np.ndarray, dimension: int):
    # for example the connectivity between triangle and vertex
    # one triangle contains 3 vertices
    # so the dimension is 3
    # from is triangle
    # to is vertex
    self.__name: str = name
    self.__fromPrimitive: primitive = from_primitive
    self.__toPrimitive: primitive = to_primitive
    self.__value: gpuarray.GPUArray = gpuarray.to_gpu(value.flatten().astype(np.uint32))
    self.__dimension: int = dimension

  @property
  def name(self)->str:
    return self.__name

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
  def mesh(self)->mesh:
    return self.fromPrimitive.mesh

  @property
  def scene(self)->scene:
    return self.mesh.scene

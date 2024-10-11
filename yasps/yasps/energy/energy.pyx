# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Union, Tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.operator import operator
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.primitive import primitive
  from yasps.connectivity import connectivity
  from yasps.deviceKernel import deviceKernel
  from yasps.globalKernel import globalKernel
  from yasps.codeGenerator import codeGenerator
  from yasps.attribute import attribute

class energy:
  def __init__(self, energy: attribute):
    if energy.size != 1:
      raise ValueError("energy.__init__: energy must be size 1.")
    self.__energy: attribute = energy
    self.__root: List[attribute] = [] # check what data attribute we have
    self.__rootSize: List[int] = [] # check for each data attribute, the actual size. This is useful when gathering is needed. Optimizing on a tet which has root of vertex position actual have root size of 4 since we need 4 of them in derivative and hessian computation

    self.getRoots() # get the root attributes


  def getRoots(self) -> None:
    from yasps.attribute import GATHER, SUM, AVERAGE, DATA
    stack: List[attribute] = [self.__energy]
    multiplications: List[int] = [1]
    while stack:
      current = stack.pop()
      multiplication = multiplications.pop()
      if current.operator == GATHER or current.operator == SUM or current.operator == AVERAGE:
        if current.through.dimension == 0:
          raise ValueError("energy.getRoots: current.through.dimension is 0. Such operation is not supported in energy minimization.")
        multiplications.append(multiplication * current.through.dimension)
      elif current.operator == DATA:
        if current not in self.__root:
          self.__root.append(current)
          self.__rootSize.append(multiplication)
        else:
          index = self.__root.index(current)
          self.__rootSize[index] += multiplication
      else:
        for child in current.children:
          stack.append(child)
          multiplications.append(multiplication)
    # for scene and mesh attributes, the count should always be 1
    for i in range(len(self.__root)):
      if self.__root[i].correspondance.type == "scene" or self.__root[i].correspondance.type == "mesh":
        self.__rootSize[i] = 1


  def __hash__(self) -> int:
    return self.__energy.hash

  @property
  def hash(self) -> int:
    return self.__hash__()

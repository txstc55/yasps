# cython: language_level=3
from __future__ import annotations
from yasps.scene import scene
from yasps.mesh import mesh
from yasps.primitive import primitive
from yasps.connectivity import connectivity
from yasps.attribute import attribute
from typing import List, Set

class deviceKernel:
  def __init__(self, kernel_string: str, kernel_header: str, kernel_datas: Set[attribute], kernel_connectivity: Set[connectivity], dependents: Set[deviceKernel]):
    self.__kernelString: str = kernel_string
    self.__kernelHeader: str = kernel_header
    self.__kernelDatas: Set[attribute] = kernel_datas
    self.__kernelConnectivity: Set[connectivity] = kernel_connectivity
    self.__dependents: Set[deviceKernel] = dependents

  @property
  def kernelString(self)->str:
    return self.__kernelString

  @property
  def kernelDatas(self)->Set[attribute]:
    return self.__kernelDatas

  @property
  def kernelConnectivity(self)->Set[connectivity]:
    return self.__kernelConnectivity

  def __hash__(self) -> int:
    return hash(self.__kernelString)

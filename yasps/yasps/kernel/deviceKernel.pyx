# cython: language_level=3
from __future__ import annotations
from yasps.scene import scene
from yasps.mesh import mesh
from yasps.primitive import primitive
from yasps.connectivity import connectivity
from yasps.attribute import attribute
from typing import List

class deviceKernel:
  def __init__(self, kernel_string: str, kernel_header: str, kernel_datas: List[attribute], kernel_connectivity: List[connectivity], dependents: List[deviceKernel]):
    self.__kernelString: str = kernel_string
    self.__kernelHeader: str = kernel_header
    self.__kernelDatas: List[attribute] = sorted(set(kernel_datas), key = lambda x: x.fullName)
    self.__kernelConnectivity: List[connectivity] = sorted(set(kernel_connectivity), key = lambda x: x.fullName)
    self.__dependents: List[deviceKernel] = sorted(set(dependents), key = lambda x: x.kernelHeader)

  @property
  def kernelString(self)->str:
    return self.__kernelString

  @property
  def kernelDatas(self)->List[attribute]:
    return self.__kernelDatas

  @property
  def kernelConnectivity(self)->List[connectivity]:
    return self.__kernelConnectivity

  @property
  def kernelHeader(self)->str:
    return self.__kernelHeader

  @property
  def dependents(self)->List[deviceKernel]:
    return self.__dependents

  def __hash__(self) -> int:
    return hash(self.__kernelString)

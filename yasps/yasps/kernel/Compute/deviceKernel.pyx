# cython: language_level=3
from __future__ import annotations
from yasps.connectivity import connectivity
from yasps.attribute import attribute
from yasps.primitiveUnion import primitiveUnion
from typing import List, Set

class deviceKernel:
  def __init__(self, kernel_string: str, kernel_header: str, kernel_datas: List[attribute], kernel_connectivity: List[connectivity], kernel_union_primitives: List[primitiveUnion], dependents: List[deviceKernel], allEvdSizes: Set[int] = set(), attributeName: str = ""):
    self.__kernelString: str = kernel_string
    self.__kernelHeader: str = kernel_header
    self.__kernelDatas: List[attribute] = sorted(set(kernel_datas), key = lambda x: x.fullName) # all the data needed for the kernel
    self.__kernelConnectivity: List[connectivity] = sorted(set(kernel_connectivity), key = lambda x: x.fullName) # all the connectivity needed for the kernel
    self.__kernelPrimitiveUnions: List[primitiveUnion] = sorted(set(kernel_union_primitives), key = lambda x: x.fullName) # all the union primitives needed for the kernel
    self.__dependents: List[deviceKernel] = sorted(set(dependents), key = lambda x: x.kernelHeader)
    self.__allEvdSizes = allEvdSizes
    self.__attributeName = attributeName


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
  def kernelPrimitiveUnions(self)->List[primitiveUnion]:
    return self.__kernelPrimitiveUnions

  @property
  def kernelHeader(self)->str:
    return self.__kernelHeader

  @property
  def dependents(self)->List[deviceKernel]:
    return self.__dependents

  def __hash__(self) -> int:
    return hash(self.__kernelString)

  @property
  def allEvdSizes(self) -> Set[int]:
    return self.__allEvdSizes

  @property
  def attributeName(self) -> str:
    return self.__attributeName

  # allow directly modifying the kernelString if necessary
  @kernelString.setter
  def kernelString(self, new_kernel_string: str):
    self.__kernelString = new_kernel_string

  @kernelHeader.setter
  def kernelHeader(self, new_kernel_header: str):
    self.__kernelHeader = new_kernel_header

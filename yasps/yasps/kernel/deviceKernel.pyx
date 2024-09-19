from yasps.scene import scene
from yasps.mesh import mesh
from yasps.primitive import primitive
from yasps.connectivity import connectivity
from yasps.attribute import attribute
from typing import List, Set

class deviceKernel:
  def __init__(self, kernel_string: str, kernel_datas: Set[attribute], kernel_connectivity: Set[connectivity]):
    self.__kernelString: str = kernel_string
    self.__kernelDatas: Set[attribute] = kernel_datas
    self.__kernelConnectivity: Set[connectivity] = kernel_connectivity

  @property
  def kernelString(self)->str:
    return self.__kernelString

  @property
  def kernelDatas(self)->Set[attribute]:
    return self.__kernelDatas

  @property
  def kernelConnectivity(self)->Set[connectivity]:
    return self.__kernelConnectivity

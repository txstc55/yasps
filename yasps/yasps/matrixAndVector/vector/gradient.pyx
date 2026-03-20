from __future__ import annotations
from typing import Optional, Any, List

from yasps.vector import vector
from yasps.attribute import attribute
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray

class gradient(vector):
  def __init__(self, wrt: List[attribute], hessian: Optional[Any] = None):
    if wrt is None:
      wrt = []
    self.__wrt: List[attribute] = list(wrt)
    total_size = 0
    self.__gradientSizes = []
    for item in self.__wrt:
      if item.isDynamic:
        # for wrt let's disallow dynamic attributes
        raise ValueError("minimizer.__getGradientSize: wrt is a dynamic attributes.")
      self.__gradientSizes.append(item.size * item.correspondance.numInstances)
      total_size += self.__gradientSizes[-1]
    super().__init__(total_size)
    self.__hessian = hessian
    # initialize segments
    gradient_segment_start = [0]
    for size in self.__gradientSizes:
      gradient_segment_start.append(gradient_segment_start[-1] + size)
    self.__gradient_segments_start = gpuarray.to_gpu(np.array(gradient_segment_start, dtype = np.uint32))
    self.__gradient_segments_start_cpu = gradient_segment_start
    start = 0
    self.__wrtStartIndices = []
    self.__wrtStartIndices.append(start) # get where each data element starts
    # here we compute for each data, where does it reside in the fianl solution arrays
    self.__gradientSegments = []
    for size in self.__gradientSizes:
      self.__gradientSegments.append(self.__gradient[start:start + size])
      start += size
      self.__wrtStartIndices.append(start)


  @property
  def hessian(self) -> Optional[Any]:
    return self.__hessian

  @hessian.setter
  def hessian(self, parent: Optional[Any]) -> None:
    self.__hessian = parent

  def compute(self) -> None:
    pass # we will define how to compute the gradient later

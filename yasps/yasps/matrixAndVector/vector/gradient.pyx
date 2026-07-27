from __future__ import annotations
from typing import Optional, Any, List

from yasps.vector import vector
from yasps.attribute import attribute
from yasps.backend import autoinit
import numpy as np
from yasps.backend import gpuarray


class gradient(vector):
  def __init__(self, wrt: List[attribute], hessian: Optional[Any] = None):
    if wrt is None:
      wrt = []
    self.__wrt: List[attribute] = list(wrt)
    total_size = 0
    self.__gradient_sizes: List[int] = []
    for item in self.__wrt:
      if item.isDynamic:
        raise ValueError("gradient.__init__: wrt can not contain dynamic attributes.")
      self.__gradient_sizes.append(item.size * item.correspondance.numInstances)
      total_size += self.__gradient_sizes[-1]

    super().__init__(total_size)
    self.__hessian = hessian

    gradient_segment_start = [0]
    for size in self.__gradient_sizes:
      gradient_segment_start.append(gradient_segment_start[-1] + size)
    self.__gradient_segments_start = gpuarray.to_gpu(np.array(gradient_segment_start, dtype=np.uint32))
    self.__gradient_segments_start_cpu = gradient_segment_start

    start = 0
    self.__wrt_start_indices: List[int] = [0]
    self.__gradient_segments: List[gpuarray.GPUArray] = []
    for size in self.__gradient_sizes:
      self.__gradient_segments.append(self.value[start:start + size])
      start += size
      self.__wrt_start_indices.append(start)

  @property
  def hessian(self) -> Optional[Any]:
    return self.__hessian

  @hessian.setter
  def hessian(self, parent: Optional[Any]) -> None:
    self.__hessian = parent

  @property
  def wrt(self) -> List[attribute]:
    return self.__wrt

  @property
  def gradient_segments_start(self) -> gpuarray.GPUArray:
    return self.__gradient_segments_start

  @property
  def gradient_segments_start_cpu(self) -> List[int]:
    return self.__gradient_segments_start_cpu

  @property
  def gradient_segments(self) -> List[gpuarray.GPUArray]:
    return self.__gradient_segments

  @property
  def wrt_start_indices(self) -> List[int]:
    return self.__wrt_start_indices

  @property
  def gradient_sizes(self) -> List[int]:
    return self.__gradient_sizes

  def compute(self) -> None:
    if self.__hessian is None:
      return
    self.__hessian.compute(self)

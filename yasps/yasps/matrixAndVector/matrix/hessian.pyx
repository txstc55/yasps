from __future__ import annotations
from typing import Optional

from yasps.matrix import matrix
from yasps.gradient import gradient


class hessian(matrix):
  def __init__(self, rows: int, cols: int, grad: gradient):
    if not isinstance(grad, gradient):
      raise TypeError("hessian.__init__: gradient must be a yasps.gradient.gradient.")
    super().__init__(rows, cols)
    self.__gradient = grad

  @property
  def gradient(self) -> gradient:
    return self.__gradient

  @gradient.setter
  def gradient(self, value: gradient) -> None:
    if not isinstance(value, gradient):
      raise TypeError("hessian.gradient: value must be a yasps.gradient.gradient.")
    self.__gradient = value

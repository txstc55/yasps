from __future__ import annotations
from typing import Optional

from yasps.matrix import matrix
from yasps.gradient import gradient


class hessian(matrix):
  def __init__(self, rows: int, cols: int, gradient: gradType):
    if not isinstance(gradient, gradType):
      raise TypeError("hessian.__init__: gradient must be a yasps.gradient.gradient.")
    super().__init__(rows, cols)
    self.__gradient = gradient

  @property
  def gradient(self) -> gradType:
    return self.__gradient

  @gradient.setter
  def gradient(self, value: gradType) -> None:
    if not isinstance(value, gradType):
      raise TypeError("hessian.gradient: value must be a yasps.gradient.gradient.")
    self.__gradient = value

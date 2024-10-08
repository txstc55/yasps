# cython: language_level=3
from __future__ import annotations
from typing import Optional, List, Union, Tuple, Dict
from typing import TYPE_CHECKING
import yasps.attribute as ya


class autodiff:
  def __init__(self, source: ya.attribute, wrt: List[ya.attribute]):
    for item in wrt:
      if item.operator != ya.DATA:
        raise ValueError("autodiff: The wrt attribute must be a data attribute")
    self.__source: ya.attribute = source
    self.__wrt: List[ya.attribute] = wrt
    self.__results: List[ya.attribute] = []
    self.__seen_differentiations: Dict[Tuple[ya.attribute, ya.attribute], ya.attribute] = {}



  def __diff(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # we have done this differentiation before
    if (current, wrt) in self.__seen_differentiations:
      return self.__seen_differentiations[(current, wrt)]
    # if current.operator == ya.ADD:
    #   return self.__diff_add(current, wrt)
    # elif current.operator == ya.SUB:
    #   return self.__diff_sub(current, wrt)
    # elif current.operator == ya.MUL:
    #   return self.__diff_mul(current, wrt)
    # elif current.operator == ya.DIV:
    #   return self.__diff_div(current, wrt)
    # elif current.operator == ya.POW:
    #   return self.__diff_pow(current, wrt)
    # elif current.operator == ya.NEG:
    #   return self.__diff_neg(current, wrt)
    if current.operator == ya.DATA:
      result = self.__diff_data(current, wrt)
    elif current.operator == ya.ARRAY_ACCESS:
      self.__seen_differentiations[(current, wrt)] = self.__diff_data(current, wrt)
      result = self.__diff_array_access(current, wrt)
    else:
      raise ValueError("autodiff: The operator is not supported")
    self.__seen_differentiations[(current, wrt)] = self.__diff_data(current, wrt)
    return result

  def __diff_add(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    return dA.add_explicit(dB)
    # return ya.to_array([self.__diff(current.children[0], wrt) + self.__diff(current.children[1], wrt)], rows = current.rows, columns = current.cols, correspondance = current.correspondance)

  def __diff_sub(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return ya.to_array([self.__diff(current.children[0], wrt) - self.__diff(current.children[1], wrt)], rows = current.rows, columns = current.cols, correspondance = current.correspondance)

  def __diff_mul(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # we explicitly do the multiplication
    # first we differentiate the matrices
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    A = current.children[0]
    B = current.children[1]
    # then we multiply the matrices explicitly
    dAB = dA.explicit_mul(B)
    dBA = A.explicit_mul(dB)

    return dAB.explicit_add(dBA)

  # def __diff_log(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
  #   # check if the children is a determinant
  #   if current.children[0].operator == ya.DET:



  def __diff_data(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # when we differentiate a data attribute
    # there are only 1 and 0
    # because we are always differentiating wrt to an element of a data attribute
    # the result is always a ARRAY_ACCESS

    data_name = wrt.children[0].fullName
    current_name = current.fullName
    if current_name == data_name:
      # the current attribute is the wrt attribute
      # the result is 1 somewhere and 0 everywhere else
      result: List[ya.attribute] = []
      for i in range(current.size):
        if i == wrt.children[1].index_value:
          result.append(ya.attribute(float_value = 1.0))
        else:
          result.append(ya.attribute(float_value = 0.0))
      return ya.to_array(result, rows = current.rows, columns = current.cols)
    else:
      result: List[ya.attribute] = []
      for _ in range(current.size):
        result.append(ya.attribute(float_value = 0.0))
      return ya.to_array(result, rows = current.rows, columns = current.cols)

  def __diff_array_access(self, current: ya.attribute, wrt: ya.attribute):
    # differentiating an array access
    # is equal to differentiating every element of the array
    return self.__diff(current.children[0], wrt)

  def __diff_row(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # differentiating a row
    mat_diff = self.__diff(current.children[0], wrt)
    mat_cols = current.children[0].cols
    row_index = current.children[1].index_value
    return ya.to_array([mat_diff[mat_cols * row_index + i] for i in range(mat_cols)], rows = 1, cols = mat_cols)

  def __diff_col(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    mat_diff = self.__diff(current.children[0], wrt)
    mat_rows = current.children[0].rows
    col_index = current.children[1].index_value
    return ya.to_array([mat_diff[col_index + i * mat_rows] for i in range(mat_rows)], rows = mat_rows, cols = 1)

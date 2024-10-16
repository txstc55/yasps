# cython: language_level=3
from __future__ import annotations
from typing import Optional, List, Union, Tuple, Dict
from typing import TYPE_CHECKING
import yasps.attribute as ya


class autodiff:
  def __init__(self):
    self.__seen_differentiations: Dict[Tuple[ya.attribute, ya.attribute], ya.attribute] = {}



  def diff(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return self.__diff(current, wrt)

  def __diff(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # we have done this differentiation before
    if (current, wrt) in self.__seen_differentiations:
      return self.__seen_differentiations[(current, wrt)]
    if current.operator == ya.ADD:
      return self.__diff_add(current, wrt)
    elif current.operator == ya.SUB:
      return self.__diff_sub(current, wrt)
    elif current.operator == ya.MUL:
      return self.__diff_mul(current, wrt)
    # elif current.operator == ya.DIV:
    #   return self.__diff_div(current, wrt)
    # elif current.operator == ya.POW:
    #   return self.__diff_pow(current, wrt)
    # elif current.operator == ya.NEG:
    #   return self.__diff_neg(current, wrt)
    elif current.operator == ya.DATA:
      result = self.__diff_data(current, wrt)
    elif current.operator == ya.ARRAY_ACCESS:
      self.__seen_differentiations[(current, wrt)] = self.__diff_data(current, wrt)
      result = self.__diff_array_access(current, wrt)
    else:
      raise ValueError("autodiff: The operator is not supported")
    self.__seen_differentiations[(current, wrt)] = self.__diff_data(current, wrt)
    return result

  def __diff_add(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # dimension will match
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    return dA.add_explicit(dB)

  def __diff_sub(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return ya.to_array([self.__diff(current.children[0], wrt) - self.__diff(current.children[1], wrt)], rows = current.rows, columns = current.cols, correspondance = current.correspondance)

  def __diff_mul(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # we explicitly do the multiplication
    # first we differentiate the matrices
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    A = current.children[0]
    B = current.children[1]
    # dA is now a tensor, since we flatten the third dimension
    # dA is m by n by k, and we merged the first two dimension so it is m*n by k
    # since we need to multiply dA by B, we need to expand the first dimension of dA
    result = [None] * current.size * wrt.size
    for i in range(wrt.size):
      dAi = ya.attribute.to_array([dA[j, i] for j in range(A.size)], rows = A.rows, cols = A.cols)
      dAB = dAi.mul_explicit(B)
      dBi = ya.attribute.to_array([dB[j, i] for j in range(B.size)], rows = B.rows, cols = B.cols)
      AdB = A.mul_explicit(dBi)
      summation = dAB.add_explicit(AdB)
      for j in range(current.size):
        result[j * wrt.size + i] = summation[j]
    return ya.attribute.to_array(result, rows = current.size, cols = wrt.size)


    # # then we multiply the matrices explicitly
    # dAB = dA.mul_explicit(B)
    # dBA = A.mul_explicit(dB)

    # return dAB.explicit_add(dBA)

  # def __diff_log(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
  #   # check if the children is a determinant
  #   if current.children[0].operator == ya.DET:

  def __diff_gather(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # differentiating through a gathering is a bit different
    # because we will need to generate the jacobian for two things
    # first the children to wrt
    child = current.children[0]
    child_jacobian = ya.attribute.zeros(child.size, wrt.size)
    child_jacobian_name = f"d{current.fullName}_d{child.fullName}"
    if child_jacobian_name not in child.correspondance.attributes:
      child_jacobian = self.__diff(child, wrt)
      # once we get the child jacobian, we first add it as an attribute in case we need it later
      child.correspondance.attributes[child_jacobian_name] = child_jacobian
    else:
      child_jacobian = child.correspondance.attributes[child_jacobian_name]
    return child_jacobian


  def __diff_data(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # checking if the data is the same
    # if it is, we return identity
    # else we return zeros
    if current.fullName == wrt.fullName:
      return ya.attribute.identity(current.size)
    else:
      return ya.attribute.zeros(current.size, wrt.size)

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

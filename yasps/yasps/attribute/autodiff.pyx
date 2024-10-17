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
    result: ya.attribute
    # we have done this differentiation before
    if (current, wrt) in self.__seen_differentiations:
      result = self.__seen_differentiations[(current, wrt)]
    if current.operator == ya.ADD:
      result = self.__diff_add(current, wrt)
    elif current.operator == ya.SUB:
      result = self.__diff_sub(current, wrt)
    elif current.operator == ya.MUL:
      result = self.__diff_mul(current, wrt)
    elif current.operator == ya.DIV:
      result = self.__diff_div(current, wrt)
    # elif current.operator == ya.POW:
    #   return self.__diff_pow(current, wrt)
    # elif current.operator == ya.NEG:
    #   return self.__diff_neg(current, wrt)
    elif current.operator == ya.FLOAT:
      result = self.__diff_float(current, wrt)
    elif current.operator == ya.DATA:
      result = self.__diff_data(current, wrt)
    elif current.operator == ya.ARRAY_ACCESS:
      result = self.__diff_array_access(current, wrt)
    elif current.operator == ya.ROW:
      result = self.__diff_row(current, wrt)
    elif current.operator == ya.COL:
      result = self.__diff_col(current, wrt)
    else:
      raise ValueError(f"autodiff: The operator is not supported: {current.operator}")
    self.__seen_differentiations[(current, wrt)] = result
    return result

  def __diff_add(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # dimension will match
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    return dA.add_explicit(dB)

  def __diff_sub(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    return dA.sub_explicit(dB)

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

  def __diff_div(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    dA = self.__diff(current.children[0], wrt) # dA is now m by n by k
    dB = self.__diff(current.children[1], wrt) # dB is now m by n by k
    result = [None] * current.size * wrt.size
    for i in range(wrt.size):
      dAi = ya.attribute.to_array([dA[j, i] for j in range(current.size)], rows = current.rows, cols = current.cols)
      # because for division, B is always a singular value
      # dBi is then always a singular value
      dBi = dB[i]
      Ai = current.children[0]
      Bi = current.children[1]
      dAB = dAi.mul_explicit(Bi)
      AdB = Ai.mul_explicit(dBi)
      summation = dAB.sub_explicit(AdB).div_explicit(Bi * Bi)
      for j in range(current.size):
        result[j * wrt.size + i] = summation[j]
    return ya.attribute.to_array(result, rows = current.size, cols = wrt.size)

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
    mat_diff = self.__diff(current.children[0], wrt)
    ind = current.children[1].index_value
    return ya.attribute.to_array(mat_diff.children[ind * wrt.size: (ind + 1) * wrt.size], rows = 1, cols = wrt.size)

  def __diff_row(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # differentiating a row
    mat_diff = self.__diff(current.children[0], wrt)
    mat_cols = current.children[0].cols
    row_index = current.children[1].index_value
    return ya.attribute.to_array(mat_diff.children[row_index * mat_cols * wrt.size:(row_index + 1) * mat_cols * wrt.size], rows = mat_cols, cols = wrt.size)

  def __diff_col(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    mat_diff = self.__diff(current.children[0], wrt)
    mat_rows = current.children[0].rows
    col_index = current.children[1].index_value
    result = []
    for i in range(mat_rows):
      result += mat_diff.children[(i * mat_rows * wrt.size + col_index * wrt.size):(i * mat_rows * wrt.size + (col_index + 1) * wrt.size)]
    return ya.attribute.to_array(result, rows = mat_rows, cols = wrt.size)

  def __diff_float(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return ya.attribute.zeros(current.size, wrt.size)

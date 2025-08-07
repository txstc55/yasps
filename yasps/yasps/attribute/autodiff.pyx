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
    elif current.operator == ya.ABS:
      result = self.__diff_abs(current, wrt)
    elif current.operator == ya.ADD:
      result = self.__diff_add(current, wrt)
    elif current.operator == ya.SUB:
      result = self.__diff_sub(current, wrt)
    elif current.operator == ya.MUL:
      result = self.__diff_mul(current, wrt)
    elif current.operator == ya.DIV:
      result = self.__diff_div(current, wrt)
    elif current.operator == ya.POW:
      result = self.__diff_pow(current, wrt)
    elif current.operator == ya.ATAN2:
      result = self.__diff_atan2(current, wrt)
    elif current.operator == ya.SQRT:
      result = self.__diff_sqrt(current, wrt)
    elif current.operator == ya.LOG:
      result = self.__diff_log(current, wrt)
    elif current.operator == ya.SIN:
      result = self.__diff_sin(current, wrt)
    elif current.operator == ya.COS:
      result = self.__diff_cos(current, wrt)
    elif current.operator == ya.NORM:
      result = self.__diff_norm(current, wrt)
    elif current.operator == ya.NEG:
      result = self.__diff_neg(current, wrt)
    elif current.operator == ya.FLOAT:
      result = self.__diff_float(current, wrt)
    elif current.operator == ya.DATA:
      result = self.__diff_data(current, wrt)
    elif current.operator == ya.CONSTANT:
      result = self.__diff_constant(current, wrt)
    elif current.operator == ya.ARRAY_ACCESS:
      result = self.__diff_array_access(current, wrt)
    elif current.operator == ya.ARRAY:
      result = self.__diff_array(current, wrt)
    elif current.operator == ya.ROW:
      result = self.__diff_row(current, wrt)
    elif current.operator == ya.COL:
      result = self.__diff_col(current, wrt)
    elif current.operator == ya.JOIN:
      result = self.__diff_join(current, wrt)
    elif current.operator == ya.UNION:
      result = self.__diff_union(current, wrt)
    elif current.operator == ya.SELECT:
      result = self.__diff_select(current, wrt)
    elif current.operator == ya.DET:
      result = self.__diff_det(current, wrt)
    elif current.operator == ya.TRANSPOSE:
      result = self.__diff_transpose(current, wrt)
    elif current.operator == ya.INV:
      result = self.__diff_inv(current, wrt)
    elif current.operator == ya.RESIZE:
      result = self.__diff_resize(current, wrt)
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

  def __diff_abs(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    dA = self.__diff(current.children[0], wrt)
    return ya.attribute.to_array([dA[i].abs() for i in range(dA.size)], rows = dA.rows, cols = dA.cols)

  def __diff_mul(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # we explicitly do the multiplication
    # first we differentiate the matrices
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
    A = current.children[0]
    B = current.children[1]
    if A.size !=1 and B.size != 1:
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
    elif A.size == 1:
      result = []
      for i in range(B.size):
        Bi = B[i]
        dABi = dA.mul_explicit(Bi)
        for j in range(wrt.size):
          result.append(dABi[j])
      AdB = A.mul_explicit(dB)
      return AdB.add_explicit(ya.attribute.to_array(result, rows = current.size, cols = wrt.size))
    elif B.size == 1:
      result = []
      dAB = dA.mul_explicit(B)
      for i in range(A.size):
        Ai = A[i]
        AdBi = Ai.mul_explicit(dB)
        for j in range(wrt.size):
          result.append(AdBi[j])
      return dAB.add_explicit(ya.attribute.to_array(result, rows = current.size, cols = wrt.size))

  def __diff_div(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    dA = self.__diff(current.children[0], wrt)
    dB = self.__diff(current.children[1], wrt)
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

  def __diff_neg(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    current_diff = self.__diff(current.children[0], wrt)
    result = [-current_diff[i] for i in range(current_diff.size)]
    return ya.attribute.to_array(result, rows = current_diff.rows, cols = current_diff.cols)

  def __diff_pow(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # we know that current is size one
    # d/dx of f(x)^g(x) = f(x)^g(x) * (g'(x) * ln(f(x)) + g(x) * f'(x) / f(x))
    fx = current.children[0]
    gx = current.children[1]
    d_fx = self.__diff(fx, wrt)
    d_gx = self.__diff(gx, wrt)
    ln_fx = fx.log()
    d_gx_ln_fx = d_gx.mul_explicit(ln_fx) # we know ln_fx is a scalar
    gx_d_fx_fx = d_fx.mul_explicit(gx / fx) # we know gx and fx are both scalars
    result = current.mul_explicit(d_gx_ln_fx.add_explicit(gx_d_fx_fx))
    return result

  def __diff_atan2(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # d/dx of atan2(f(x), g(x)) = (g(x) * f'(x) - f(x) * g'(x)) / (f(x)^2 + g(x)^2)
    denominator = current.children[0] * current.children[0] + current.children[1] * current.children[1]
    d_fx = self.__diff(current.children[0], wrt)
    d_gx = self.__diff(current.children[1], wrt)
    result = (current.children[1].mul_explicit(d_fx) - current.children[0].mul_explicit(d_gx)) / denominator
    return result

  def __diff_sin(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    df = self.__diff(current.children[0], wrt)
    cosf = current.children[0].cos()
    return df.mul_explicit(cosf)

  def __diff_cos(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    df = self.__diff(current.children[0], wrt)
    sinf = current.children[0].sin()
    return df.mul_explicit(-sinf)


  def __diff_log(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    df = self.__diff(current.children[0], wrt)
    return df.div_explicit(current.children[0])

  def __diff_sqrt(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # rule is 1 / (2 * sqrt(dx))
    df = self.__diff(current.children[0], wrt) # the jacobian is 1 by n
    result = [0.5 * df[i] / current for i in range(wrt.size)]
    return ya.attribute.to_array(result, rows = 1, cols = wrt.size)




  def __diff_norm(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    df = self.__diff(current.children[0], wrt) # df is m by n
    result = [None] * wrt.size # because norm is a scalar
    for i in range(wrt.size):
      dfi = ya.attribute.to_array([df[j, i] for j in range(current.children[0].size)], rows = current.children[0].size, cols = 1)
      fdfi = current.children[0].dot_explicit(dfi)
      result[i] = fdfi / current
    return ya.attribute.to_array(result, rows = 1, cols = wrt.size)

  def __diff_select(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return ya.attribute.select(current.children[0], self.__diff(current.children[1], wrt), self.__diff(current.children[2], wrt))



  def __diff_join(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # ok so we never explicitly differentiate a join operation here
    # instead we deal it in energy.pyx
    # so whatever we have here is just placeholder
    # in the future we may need to move it back
    # but this also have some use when we do join on a join
    # because the path we generated, if we join a join, we differentiate the first join's child wrt the second join, which should lead to identity
    # so this is fine
    if wrt.operator != ya.JOIN:
      return ya.attribute.zeros(current.size, wrt.size)
    if current.fullName == wrt.fullName:
      return ya.attribute.identity(current.size)
    else:
      return ya.attribute.zeros(current.size, wrt.size)

  def __diff_union(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # same logic actually
    # we will never at energy.pyx encounter a case where we need to differentiate a union wrt a join, or differentiate a join wrt a union, since we always do diff node's child wrt next node, and if node's child is a union, that means the next node has to be itself
    if wrt.operator != ya.UNION:
      return ya.attribute.zeros(current.size, wrt.size)
    if current.fullName == wrt.fullName:
      return ya.attribute.identity(current.size)
    else:
      return ya.attribute.zeros(current.size, wrt.size)


  def __diff_data(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # checking if the data is the same
    # if it is, we return identity
    # else we return zeros
    if current.fullName == wrt.fullName:
      return ya.attribute.identity(current.size)
    else:
      return ya.attribute.zeros(current.size, wrt.size)

  def __diff_constant(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    if current.fullName == wrt.fullName:
      return ya.attribute.identity(current.size)
    else:
      return ya.attribute.zeros(current.size, wrt.size)

  def __diff_array(self, current: ya.attribute, wrt: ya.attribute):
    # print(f'Current: {str(current)}')
    # print(f'Wrt: {str(wrt)}')
    result = []
    for i in range(current.size):
      # print(f'Child {i} of current: {str(current.children[i])}')
      diff_result = self.__diff(current.children[i], wrt)
      # print(f'Diff result: {str(diff_result)}')
      result += [diff_result[i] for i in range(wrt.size)]
    return ya.attribute.to_array(result, rows = current.size, cols = wrt.size)

  def __diff_array_access(self, current: ya.attribute, wrt: ya.attribute):
    # differentiating an array access
    # is equal to differentiating every element of the array
    mat_diff = self.__diff(current.children[0], wrt)
    ind = current.children[1].index_value
    return ya.attribute.to_array([mat_diff[i] for i in range(ind * wrt.size, (ind + 1) * wrt.size)], rows = 1, cols = wrt.size)
    # return ya.attribute.to_array(mat_diff.children[ind * wrt.size: (ind + 1) * wrt.size], rows = 1, cols = wrt.size)

  def __diff_row(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # differentiating a row
    mat_diff = self.__diff(current.children[0], wrt)
    mat_cols = current.children[0].cols
    row_index = current.children[1].index_value
    return ya.attribute.to_array([mat_diff[row_index * mat_cols * wrt.size + i] for i in range(mat_cols * wrt.size)], rows = mat_cols, cols = wrt.size)
    # return ya.attribute.to_array(mat_diff.children[row_index * mat_cols * wrt.size:(row_index + 1) * mat_cols * wrt.size], rows = mat_cols, cols = wrt.size)

  def __diff_col(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    mat_diff = self.__diff(current.children[0], wrt)
    mat_rows = current.children[0].rows
    col_index = current.children[1].index_value
    result = []
    for i in range(mat_rows):
      result += [mat_diff[j] for j in range((i * mat_rows * wrt.size + col_index * wrt.size), (i * mat_rows * wrt.size + (col_index + 1) * wrt.size))]
      # result += mat_diff.children[(i * mat_rows * wrt.size + col_index * wrt.size):(i * mat_rows * wrt.size + (col_index + 1) * wrt.size)]
    return ya.attribute.to_array(result, rows = mat_rows, cols = wrt.size)

  def __diff_float(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return ya.attribute.zeros(current.size, wrt.size)


  def __diff_det(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    # differentiating det(f(x))
    # is equal to det(f(x)) * tr(f(x)^-1 * df/dx)
    f_inv = current.children[0].inverse()
    d_fx = self.__diff(current.children[0], wrt)
    result = [None] * current.size * wrt.size
    for i in range(wrt.size):
      mat = ya.attribute.to_array([d_fx[j * wrt.size + i] for j in range(current.children[0].size)], rows = current.children[0].rows, cols = current.children[0].cols)
      result[i] = current * (f_inv.mul_explicit(mat).trace())
    return ya.attribute.to_array(result, rows = current.size, cols = wrt.size)

  def __diff_transpose(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    d_fx = self.__diff(current.children[0], wrt)
    # ok now we need to reorient the matrix
    result = [None] * current.size * wrt.size
    for i in range(current.rows):
      for j in range(current.cols):
        ind = i * current.cols + j
        transposed_ind = j * current.rows + i
        for k in range(wrt.size):
          result[ind * wrt.size + k] = d_fx[transposed_ind * wrt.size + k]
    return ya.attribute.to_array(result, rows = current.size, cols = wrt.size)

  def __diff_inv(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    d_fx = self.__diff(current.children[0], wrt)
    result = [None] * current.size * wrt.size
    for i in range(wrt.size):
      d_fx_i = ya.attribute.to_array([d_fx[j * wrt.size + i] for j in range(current.children[0].size)], rows = current.children[0].rows, cols = current.children[0].cols)
      result_i = current.mul_explicit(d_fx_i).mul_explicit(current).mul_explicit(-1.0)
      for j in range(result_i.size):
        result[j * wrt.size + i] = result_i[j]
    return ya.attribute.to_array(result, rows = current.size, cols = wrt.size)

  def __diff_resize(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    return self.__diff(current.children[0], wrt)

  def __diff_spd(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
    raise ValueError("yasps.autodiff: Differentiation of spd is not supported yet")

  # def __diff_trace(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
  #   # differentiating trace(f(x))
  #   # is equal to trace(df/dx)
  #   d_fx = self.diff(current.children[0], wrt)
  #   result = [None] * wrt.size
  #   for i in range(wrt.size):
  #     trace_sum = 0.0
  #     for j in range(current.children[0].rows):
  #       trace_sum += d_fx[j * current.children[0].cols * wrt.size + j * wrt.size + i]
  #     result[i] = trace_sum
  #   return ya.attribute.to_array(result, rows = 1, cols = wrt.size)

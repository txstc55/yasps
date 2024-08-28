from __future__ import annotations
import pycuda
import numpy as np
from .operator import operator

class symbolic:
  # here we first define all the variables
  addition = operator("+", 1, True)
  subtraction = operator("-", 1, False)
  multiplication = operator("*", 1, True)
  division = operator("/", 1, False)
  power = operator("pow", 2, False)
  negation = operator("-", 0, False)
  absolute = operator("abs", 0, False)
  square_root = operator("sqrt", 0, False)
  sin = operator("sin", 0, False)
  cos = operator("cos", 0, False)
  tan = operator("tan", 0, False)
  asin = operator("asin", 0, False)
  acos = operator("acos", 0, False)
  atan = operator("atan", 2, False)
  select = operator("select", 2, False)
  eq = operator("==", 1, True)
  neq = operator("!=", 1, True)
  gt = operator(">", 1, True)
  gte = operator(">=", 1, True)
  lt = operator("<", 1, True)
  lte = operator("<=", 1, True)
  assign = operator("=", 1, False)
  number = operator("number", 3, False)
  variable = operator("variable", 3, False)
  intermediate = operator("intermediate", 3, False)

  def __init__(self, value, operator: operator = None, children:list[symbolic] = [], is_constant: bool = False, is_variable: bool = False):
    self.__children = children
    self.__value = value
    # if is_constant is false, and is_variable is false
    # it means that the value is changing, but user will set the value
    # if is_constant is true, it means that the value is constant
    # if is_variable is true, it means that the value is changing, and we want to optimize it
    self.__is_constant = is_constant
    self.__is_variable = is_variable
    self.__operator = operator

    if operator is None:
      if isinstance(value, str):
        self.__operator = symbolic.variable
      elif isinstance(value, float):
        self.__operator = symbolic.number
      elif isinstance(value, int):
        self.__operator = symbolic.number

  @property
  def children(self):
    return self.__children

  @property
  def value(self):
    return self.__value

  @property
  def is_number(self):
    return self.__operator == symbolic.number

  @property
  def operator(self):
    return self.__operator

  def __add__(self, other: symbolic | float):
    ## we always need to do type conversion
    ## if possible
    if (type(other) != symbolic):
        otherSym = symbolic(other)
        return self + otherSym

    if (self.is_number and other.is_number):
        return symbolic(self.value + other.value)
    ## deal with 0
    elif (self.value == 0):
        return other
    elif (other.value == 0):
        return self
    elif (self.value == 0 and other.value == 0):
        return 0

    return symbolic(None, operator=symbolic.addition, children=[self, other])


  def __str__(self):
    # print(self.operator.symbol)
    if self.operator == symbolic.number:
      return str(self.value)
    elif self.operator == symbolic.variable:
      return str(self.value)
    elif self.operator == symbolic.intermediate:
      return "intermediate_" + str(self.value)
    else:
      return self.operator.to_string(self.children)

  def to_string(self):
    return str(self)

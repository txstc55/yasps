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
  sine = operator("sin", 0, False)
  cosine = operator("cos", 0, False)
  tangent = operator("tan", 0, False)
  asine = operator("asin", 0, False)
  acosine = operator("acos", 0, False)
  atangent = operator("atan2", 2, False)
  select_op = operator("select", 2, False)
  eq = operator("==", 1, True)
  neq = operator("!=", 1, True)
  gt = operator(">", 1, True)
  ge = operator(">=", 1, True)
  lt = operator("<", 1, True)
  le = operator("<=", 1, True)
  assign = operator("=", 1, False)
  number = operator("number", 3, False)
  variable = operator("variable", 3, False)
  intermediate = operator("intermediate", 3, False)

  def __init__(self, value, operator: operator = None, children:list[symbolic] = [], is_constant: bool = False, is_variable: bool = False):
    # if is_constant is false, and is_variable is false
    # it means that the value is changing, but user will set the value
    # if is_constant is true, it means that the value is constant
    # if is_variable is true, it means that the value is changing, and we want to optimize it
    if type(value) == symbolic:
      self.__children = value.children
      self.__value = value.value
      self.__is_constant = value.is_constant
      self.__is_variable = value.is_variable
      self.__operator = value.operator
      return

    self.__children = children
    self.__value = value
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

  def is_number(self):
    return self.__operator == symbolic.number

  @property
  def operator(self):
    return self.__operator

  def is_constant(self):
    return self.__is_constant

  def is_variable(self):
    return self.__is_variable


  def __add__(self, other)->symbolic:
    ## we always need to do type conversion
    ## if possible
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self + otherSym

    # special cases
    if (self.is_number() and other.is_number()):
      return self.value + other.value
    ## deal with 0
    elif (self.value == 0):
      return other
    elif (other.value == 0):
      return self
    elif (self.value == 0 and other.value == 0):
      return symbolic(0)

    # check if both are addition
    if self.operator == symbolic.addition and other.operator == symbolic.addition:
      return symbolic(None, operator = symbolic.addition, children = self.children + other.children)
    elif self.operator == symbolic.addition:
      return symbolic(None, operator = symbolic.addition, children = self.children + [other])
    elif other.operator == symbolic.addition:
      return symbolic(None, operator = symbolic.addition, children = [self] + other.children)

    return symbolic(None, operator=symbolic.addition, children=[self, other])

  def __radd__(self, other)->symbolic:
    return self + other

  def __neg__(self)->symbolic:
    # special cases
    if self.is_number():
      return symbolic(-self.value)
    elif self.operator == symbolic.negation:
      return self.children[0]
    return symbolic(None, operator=symbolic.negation, children=[self])

  def __sub__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self - otherSym

    if (self.is_number() and other.is_number()):
      return symbolic(self.value - other.value)
    elif (self.value == 0):
      return -other
    elif (other.value == 0):
      return self
    elif (self.value == 0 and other.value == 0):
      return symbolic(0)
    elif other.operator == symbolic.negation:
      return self + other.children[0]

    return symbolic(None, operator=symbolic.subtraction, children=[self, other])

  def __rsub__(self, other)->symbolic:
    return symbolic(other) - self


  def __mul__(self, other)->symbolic:
    # conversion if necessary
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self * otherSym

    if (self.is_number() and self.value == -1):
      return -other
    elif (other.is_number() and other.value == -1):
      return -self
    # special cases
    if (self.is_number() and self.value == 0):
      return symbolic(0)
    if (other.is_number() and other.value == 0):
      return symbolic(0)
    elif (self.value == 1):
      return other
    elif (other.value == 1):
      return self

    # check if both are multiplication
    if self.operator == symbolic.multiplication and other.operator == symbolic.multiplication:
      return symbolic(None, operator = symbolic.multiplication, children = self.children + other.children)
    elif self.operator == symbolic.multiplication:
      return symbolic(None, operator = symbolic.multiplication, children = self.children + [other])
    elif other.operator == symbolic.multiplication:
      return symbolic(None, operator = symbolic.multiplication, children = [self] + other.children)
    return symbolic(None, operator = symbolic.multiplication, children = [self, other])

  def __rmul__(self, other)->symbolic:
    return self * other

  def __truediv__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self / otherSym
    if (self.is_number() and other.is_number()):
      return symbolic(self.value / other.value)

    if (self.value == 0):
      return symbolic(0)
    elif (other.value == 1):
      return self
    elif (other.value == 0):
      raise ZeroDivisionError

    div_sym = symbolic(None, operator = symbolic.division, children = [symbolic(1), other])
    return self * div_sym

  def __rtruediv__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return otherSym / self
    return other * (symbolic(1) / self)

  def __iadd__(self, other)->symbolic:
    return self + other

  def __pow__(self, exp)->symbolic:
    if (type(exp) != symbolic):
      expSym = symbolic(exp)
      return self ** expSym

    # special cases
    if (self.is_number() and exp.is_number()):
      return symbolic(self.value ** exp.value)
    if exp.is_number():
      if exp.value == 0:
        return symbolic(1)
      elif exp.value == 1:
        return self
      elif exp.value == 2:
        return self * self
    if self.is_number():
      if self.value == 0:
        return symbolic(0)
      elif self.value == 1:
        return symbolic(1)
    return symbolic(None, operator = symbolic.power, children = [self, exp])

  def __rpow__(self, other)->symbolic:
    return symbolic(other) ** self

  @classmethod
  def sin(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.sine, children = [operand])

  def sin(self)->symbolic:
    return symbolic(None, operator = symbolic.sine, children = [self])

  @classmethod
  def cos(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.cosine, children = [operand])
  def cos(self)->symbolic:
    return symbolic(None, operator = symbolic.cosine, children = [self])

  @classmethod
  def tan(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.tangent, children = [operand])
  def tan(self)->symbolic:
    return symbolic(None, operator = symbolic.tangent, children = [self])

  @classmethod
  def pow(cls, operand, other)->symbolic:
    return symbolic(None, operator = symbolic.power, children = [operand, other])
  def pow(self, other)->symbolic:
    return symbolic(None, operator = symbolic.power, children = [self, other])

  @classmethod
  def sqrt(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.sqrt, children = [operand])
  def sqrt(self)->symbolic:
    return symbolic(None, operator = symbolic.sqrt, children = [self])

  @classmethod
  def log(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.log, children = [operand], unary=True)
  def log(self)->symbolic:
    return symbolic(None, operator = symbolic.log, children = [self], unary=True)
  @classmethod
  def abs(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.absolute, children = [operand], unary=True)
  def abs(self)->symbolic:
    return symbolic(None, operator = symbolic.absolute, children = [self], unary=True)

  @classmethod
  def atan2(cls, operand1, operand2)->symbolic:
    return symbolic(None, operator = symbolic.atangent, children = [operand1, operand2])

  @classmethod
  def acos(cls, operand)->symbolic:
    return symbolic(None, operator = symbolic.acosine, children = [operand])
  def acos(self)->symbolic:
    return symbolic(None, operator = symbolic.acosine, children = [self])

  ## here are bunch of logic operators
  def __eq__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self == otherSym
    return symbolic(None, operator = symbolic.eq, children = [self, other])

  def __req__(self, other)->symbolic:
    return self == other

  def __ne__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self != otherSym
    return symbolic(None, operator = symbolic.neq, children = [self, other])

  def __rne__(self, other)->symbolic:
      return self != other

  def __lt__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self < otherSym
    return symbolic(None, operator = symbolic.lt, children = [self, other])

  def __le__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self <= otherSym
    return symbolic(None, operator = symbolic.le, children = [self, other])

  def __rlt__(self, other)->symbolic:
      return self >= other

  def __rle__(self, other)->symbolic:
      return self > other

  def __gt__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self > otherSym
    return symbolic(None, operator = symbolic.gt, children = [self, other])

  def __ge__(self, other)->symbolic:
    if (type(other) != symbolic):
      otherSym = symbolic(other)
      return self >= otherSym
    return symbolic(None, operator = symbolic.ge, children = [self, other])

  def __rgt__(self, other)->symbolic:
    return self <= other

  def __rge__(self, other)->symbolic:
    return self < other

  @staticmethod
  def select(condition, trueValue, falseValue):
    if (type(condition) != symbolic):
      raise TypeError("symbolic.select: Condition must be a symbolic type")
    if (type(trueValue) != symbolic):
      trueValueSym = symbolic(trueValue)
    else:
      trueValueSym = trueValue
    if (type(falseValue) != symbolic):
      falseValueSym = symbolic(falseValue)
    else:
      falseValueSym = falseValue
    return symbolic(None, operator = symbolic.select_op, children = [condition, trueValueSym, falseValueSym])


  def __str__(self)->str:
    # print(self.operator, self.value)
    if self.operator == symbolic.number:
      return str(self.value)
    elif self.operator == symbolic.variable:
      return str(self.value)
    elif self.operator == symbolic.intermediate:
      return "intermediate_" + str(self.value)
    else:
      return self.operator.to_string(self.children)

  def to_string(self)->str:
    return str(self)

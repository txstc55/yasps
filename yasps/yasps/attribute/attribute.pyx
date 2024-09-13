# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Union, Tuple
from typing import TYPE_CHECKING
from yasps.operator import operator
if TYPE_CHECKING:
  from yasps.operator import operator
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.primitive import primitive
  from yasps.connectivity import connectivity


ADD = operator("+", 1, True)
SUB = operator("-", 1, False)
MUL = operator("*", 1, True)
DIV = operator("/", 1, False)
POW = operator("pow", 1, False)
NEG = operator("-", 0, False)
SIN = operator("sin", 0, False)
COS = operator("cos", 0, False)
TAN = operator("tan", 0, False)
COT = operator("cot", 0, False)
ABS = operator("abs", 0, False)
SELECT = operator("select", 2, False)
SQRT = operator("sqrt", 0, False)
EQ = operator("==", 1, True)
NE = operator("!=", 1, True)
LT = operator("<", 1, False)
LE = operator("<=", 1, False)
GT = operator(">", 1, False)
GE = operator(">=", 1, False)
ASSIGN = operator("=", 1, False)
INDEX = operator("ijk", 3, False) # the index used for array access
FLOAT = operator("float", 3, False) # for float numbers
ARRAY_ACCESS = operator("access", 3, False) # for accessing an element in the array
DATA = operator("data", 3, False) # for directly accessing data
ARRAY = operator("array", 3, False) # for constructing an array
GATHER = operator("gather", 3, False) # for gathering data from one primitive to another


class attribute:
  def __init__(self, name: str = "", rows: int = 1, cols: int = 1, correspondance: Optional[Union[scene, mesh, primitive]] = None, through: Optional[connectivity] = None, float_value: Optional[float] = None, children: List[attribute] = [], operator: operator = DATA, index_value: Optional[int] = None):
    # by default, any attribute is a data access
    # which does the following:
    # given x the data, and id, return x + id * rows * cols
    self.__value: gpuarray.GPUArray = gpuarray.to_gpu(np.array([]))
    self.__name: str = name
    self.__n_rows: int = rows
    self.__n_cols: int = cols
    self.__correspondance: Optional[Union[scene, mesh, primitive]] = correspondance
    self.__through: Optional[connectivity] = through
    self.__children: List[attribute] = children
    self.__operator: operator = operator
    self.__float_value: float = 0.0
    self.__index_value: int = 0
    # first let's check if it is a constant value
    if float_value is not None:
      self.__float_value = float_value
      self.__operator = FLOAT
    elif index_value is not None:
      # next if it is an index value
      self.__index_value = index_value
      self.__operator = INDEX




  @property
  def name(self)->str:
    return self.__name

  @property
  def rows(self)->int:
    return self.__n_rows

  @property
  def cols(self)->int:
    return self.__n_cols

  @property
  def correspondance(self):
    return self.__correspondance

  @property
  def through(self):
    return self.__through

  @property
  def children(self):
    return self.__children

  @property
  def float_value(self):
    return self.__float_value

  @property
  def index_value(self):
    return self.__index_value

  @property
  def value(self):
    if self.__value is None:
      raise ValueError("attribute.value: value is None. Please call compute() first or manually update value.")
    return self.__value

  def reshape(self, rows, cols):
    if self.rows * self.cols != rows * cols:
      raise ValueError("attribute.reshape: new shape must have the same number of elements.")
    self.__n_rows = rows
    self.__n_cols = cols

  @property
  def operator(self):
    return self.__operator

  def setName(self, name):
    self.__name = name


  def updateValue(self, value: Union[np.ndarray, gpuarray.GPUArray], deepCopy = False):
    # check value array for gpu array conversion
    # let's worry about memory allocation later on when the size changes
    # TODO: CHECK FOR SIZE AND DO NOT ALLOCATE NEW MEMORY WHEN SIZE IS SMALLER
    if isinstance(value, np.ndarray):
      self.__value = gpuarray.to_gpu(value)

    elif isinstance(value, gpuarray.GPUArray):
      if deepCopy:
        self.__value = value.copy()
      else:
        self.__value = value.ravel()
    else:
      try:
        flattend_value = np.array(value).flatten()
        self.updateValue(flattend_value)
      except:
        raise ValueError("attribute.updateValue: Invalid value type, cannot be converted to gpuarray")

  # construct a new attribute from a list of attributes
  @staticmethod
  def to_array(children: List[attribute], rows: int, cols: int):
    if rows * cols != len(children):
      raise ValueError("attribute.to_array: number of elements must match the number of children.")
    # TODO: CHECK FOR CORRESPONDANCE
    return attribute(name = "", rows = rows, cols = cols, children = children, operator = ARRAY, correspondance = children[0].correspondance)

  # every attribute is actually a vector or a mat
  # so accessing them through [] operator returns an access attribute
  def __getitem__(self, index: Union[int, Tuple[int, int]]) -> attribute:
    if isinstance(index, int):
      if index >= self.rows * self.cols:
        raise ValueError("attribute.__getitem__: index out of range.")
      indexAttribute = attribute(operator = INDEX, index_value = index)
      return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
    elif isinstance(index, tuple):
      if len(index) != 2:
        raise ValueError("attribute.__getitem__: index must be a tuple of two integers.")
      if index[0] >= self.rows or index[1] >= self.cols:
        raise ValueError("attribute.__getitem__: index out of range.")
      indexAttribute = attribute(operator = INDEX, index_value = index[0] * self.cols + index[1])
      return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)

  def __str__(self)->str:
    if self.operator.type == 0:
      return f"{operator.name}({self.children[0]})"
    elif self.operator.type == 1:
      return f"({self.children[0]} {operator.name} {self.children[1]})"
    elif self.operator.type == 2:
      return f"{operator.name}({', '.join([str(child) for child in self.children])})"
    elif self.operator.type == 3:
      if self.operator == INDEX:
        return str(self.index_value)
      elif self.operator == FLOAT:
        return str(self.float_value)
      elif self.operator == ARRAY_ACCESS:
        return f"{self.children[0]}[{self.children[1]}]"
      elif self.operator == DATA:
        if self.correspondance is not None:
          return f"{self.correspondance.fullName}.data"
        else:
          raise ValueError("attribute.__str__: correspondance is None for a DATA attribute.")
      elif self.operator == ARRAY:
        # Construct the string without backslashes inside the f-string
        children_str = ',\n'.join([str(child) for child in self.children])
        return f"array(\n{children_str}\n)"
      elif self.operator == GATHER:
        if len(self.children) != 1:
          raise ValueError("attribute.__str__: GATHER operator must have one child.")
        if self.children[0].correspondance is None:
          raise ValueError("attribute.__str__: GATHER operator's first child must have a correspondance.")
        if self.through is None:
          raise ValueError("attribute.__str__: GATHER operator must have a through attribute.")
        return f"gather(\n{self.children[0].correspondance.fullName}.{self.name}\n->\n{self.through.fromPrimitive.fullName}.{self.name}"
      else:
        raise ValueError("attribute.__str__: unknown operator type.")
    else:
      raise ValueError("attribute.__str__: unknown operator type.")

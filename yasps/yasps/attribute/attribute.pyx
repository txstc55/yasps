# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Union, Tuple
from typing import TYPE_CHECKING
import pycuda.driver as cuda
import time

from yasps.operator import operator
if TYPE_CHECKING:
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.primitive import primitive
  from yasps.connectivity import connectivity
  from yasps.deviceKernel import deviceKernel
  from yasps.globalKernel import globalKernel



ADD = operator("+", 1, True)
SUB = operator("-", 1, False)
MUL = operator("*", 1, True)
DIV = operator("/", 1, False)
POW = operator("pow", 2, False)
ATAN2 = operator("atan2", 2, False)
NEG = operator("neg", 3, False)
SIN = operator("sin", 0, False)
COS = operator("cos", 0, False)
TAN = operator("tan", 0, False)
COT = operator("cot", 0, False)
ABS = operator("abs", 0, False)
LOG = operator("log", 0, False)
SELECT = operator("select", 2, False)
SQRT = operator("sqrt", 0, False)
EQ = operator("==", 1, True)
NEQ = operator("!=", 1, True)
LT = operator("<", 1, False)
LEQ = operator("<=", 1, False)
GT = operator(">", 1, False)
GEQ = operator(">=", 1, False)
ASSIGN = operator("=", 1, False)
INDEX = operator("ijk", 3, False) # the index used for array access
FLOAT = operator("float", 3, False) # for float numbers
ARRAY_ACCESS = operator("access", 3, False) # for accessing an element in the array
DATA = operator("data", 3, False) # for directly accessing data
ARRAY = operator("array", 3, False) # for constructing an array

TRANSPOSE = operator("transpose", 3, False) # for transposing a matrix
BROADCAST_ADD = operator("+", 3, False) # broadcast an add to all elements
BROADCAST_SUB = operator("-", 3, False) # broadcast a sub to all elements
INTERMEDIATE = operator("intermediate", 3, False) # for intermediate results
ROW = operator("row", 3, False) # for row access
COL = operator("col", 3, False) # for column access
CROSS = operator("cross", 3, False) # for cross product
NORM = operator("norm", 3, False) # for norm
DET = operator("det", 3, False) # determinant of the matrix
INV = operator("inverse", 3, False) # matrix inverse
DOT = operator("dot", 3, False) # dot product
SPD = operator("spd", 3, False) # spd projection
RESIZE = operator("resize", 3, False) # resize the matrix

JOIN = operator("join", 3, False) # for joining data from one primitive to another
SUM = operator("sum", 3, False) # for summation when the connectivity is unfixed
AVERAGE = operator("average", 3, False) # for averaging when the connectivity is unfixed
UNION = operator("union", 3, False) # for union of multiple attributes



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
    self.__is_intermediate = False # if this is an intermediate value, we will need to generate the code and compute it before it being promped to another computation
    # first let's check if it is a constant value
    if float_value is not None:
      self.__float_value = float_value
      self.__operator = FLOAT
    elif index_value is not None:
      # next if it is an index value
      self.__index_value = index_value
      self.__operator = INDEX
    self.__hash: int = 0
    self.__deviceKernel: Optional[deviceKernel] = None
    self.__globalKernel: Optional[globalKernel] = None
    self.__isFloatMat = None
    # self.__is_dynamic = is_dynamic


  ################################################
  ################################################
  #     ATTRIBUTE PROPERTY DEFINITION
  ################################################
  ################################################
  @property
  def isDynamic(self)->bool:
    # first check the correspondance
    corr = self.correspondance
    if corr is None:
      return False
    else:
      if corr.type == "primitive":
        return corr.isDynamic
      else:
        return False

  @property
  def name(self)->str:
    return self.__name

  @property
  def fullName(self)->str:
    if self.__name != "":
      if self.correspondance is not None:
        return self.correspondance.fullName + "_" + self.__name
      else:
        return self.__name
    else:
      if self.correspondance is not None:
        return self.correspondance.fullName + "_" + str(self.hash).replace("-", "_neg_")
      else:
        return 'attr_' + str(self.hash).replace("-", "_neg_")

  @property
  def rows(self)->int:
    return self.__n_rows

  @property
  def cols(self)->int:
    return self.__n_cols

  @property
  def correspondance(self) -> Union[scene, mesh, primitive]:
    return self.__correspondance



  @property
  def through(self) -> connectivity:
    return self.__through

  @property
  def children(self) -> List[attribute]:
    return self.__children

  @property
  def float_value(self) -> float:
    return self.__float_value

  @property
  def index_value(self) -> int:
    return self.__index_value

  @property
  def deviceKernel(self) -> Optional[deviceKernel]:
    return self.__deviceKernel

  @deviceKernel.setter
  def deviceKernel(self, deviceKernel: deviceKernel) -> None:
    self.__deviceKernel = deviceKernel

  @property
  def globalKernel(self) -> Optional[globalKernel]:
    return self.__globalKernel

  @globalKernel.setter
  def globalKernel(self, globalKernel: globalKernel) -> None:
    self.__globalKernel = globalKernel

  @property
  def value(self) -> gpuarray.GPUArray:
    if self.__value is None:
      raise ValueError("attribute.value: value is None. Please call compute() first or manually update value.")
    return self.__value

  # @value.setter
  # def value(self, newValue: Union[gpuarray.GPUArray, np.ndarray]) -> None:
  #   if isinstance(newValue, np.ndarray):
  #     self.__value = gpuarray.to_gpu(np.array(newValue.flatten(), dtype = np.float64).flatten())
  #   elif isinstance(newValue, gpuarray.GPUArray):
  #     self.__value = newValue.copy()
  #   else:
  #     raise ValueError("attribute.value: Invalid value type, can only be np.ndarray or gpuarray.GPUArray.")


  @property
  def size(self) -> int:
    return self.rows * self.cols


  def reshape(self, rows, cols):
    if self.rows * self.cols != rows * cols:
      raise ValueError("attribute.reshape: new shape must have the same number of elements.")
    self.__n_rows = rows
    self.__n_cols = cols

  def resize(self, rows, cols):
    if self.rows * self.cols != rows * cols:
      raise ValueError("attribute.resize: new shape must have the same number of elements.")
    if self.size == 1:
      return self
    return attribute(children = [self, attribute(index_value = rows), attribute(index_value = cols)], operator = RESIZE, rows = rows, cols = cols, correspondance = self.correspondance)

  @property
  def operator(self):
    return self.__operator

  def setName(self, name) -> None:
    self.__name = name

  @property
  def code_generation_data_name(self) -> str:
    return f'{self.fullName}_global_data'

  # for iszero and isidentity
  # 0 means not zero or not identity
  # 1 means zero or identity and singular value
  # 2 means zero or identity and mat value
  @property
  def isZero(self) -> int:
    if self.operator == FLOAT and self.float_value == 0.0:
      return 1
    elif self.operator == ARRAY:
      # check if all are zero
      for i in range(self.size):
        if self.children[i].isZero == 0:
          return 0
      if self.size == 1:
        return 1
      return 2
    return 0

  @property
  def isIdentity(self) -> int:
    if self.operator == FLOAT and self.float_value == 1.0:
      return 1
    elif self.operator == ARRAY:
      if self.rows != self.cols:
        return 0
      for i in range(self.rows):
        for j in range(self.rows):
          if i == j:
            if self.children[i * self.cols + j].isIdentity == 0:
              return 0
          else:
            if self.children[i * self.cols + j].isZero == 0:
              return 0
      if self.rows == 1:
        return 1
      return 2
    return 0

  @property
  def isFloatMat(self) -> bool:
    if self.__isFloatMat:
      return self.__isFloatMat
    # check if this is just a constant value array
    # because it doesnt need correspondance
    if self.operator == ARRAY:
      for i in range(self.size):
        if not (self.children[i].operator == FLOAT):
          self.__isFloatMat =  False
          return False
      self.__isFloatMat = True
      return True
    # # SPECIAL CASES FOR CHECKING IF A MAT IS FLOAT
    # if self.operator == FLOAT:
    #   return True
    # if self.operator == ARRAY_ACCESS:
    #   return self.children[0].isFloatMat
    if self.operator == NEG or self.operator == ABS or self.operator == ROW or self.operator == COL or self.operator == RESIZE or self.operator == TRANSPOSE or self.operator == RESIZE:
      self.__isFloatMat = self.children[0].isFloatMat
      return self.children[0].isFloatMat
    self.__isFloatMat = False
    return False

  ################################################
  ################################################
  #     SOME METHODS
  ################################################
  ################################################
  def setAsIntermediate(self) -> None:
    self.__is_intermediate = True


  @staticmethod
  def zeros(rows: int, cols: int) -> attribute:
    zeroArray: List[attribute] = [attribute(float_value = 0.0) for _ in range(rows * cols)]
    return attribute.to_array(zeroArray, rows, cols)

  @staticmethod
  def identity(rows: int) -> attribute:
    identityArray: List[attribute] = [attribute(float_value = 1.0) if i == j else attribute(float_value = 0.0) for i in range(rows) for j in range(rows)]
    return attribute.to_array(identityArray, rows, rows)

  def updateValue(self, value: Union[np.ndarray, gpuarray.GPUArray], deepCopy = False):
    # check value array for gpu array conversion
    # let's worry about memory allocation later on when the size changes
    # TODO: CHECK FOR SIZE AND DO NOT ALLOCATE NEW MEMORY WHEN SIZE IS SMALLER
    if isinstance(value, np.ndarray):
      self.__value = gpuarray.to_gpu(np.array(value.flatten(), dtype = np.float64).flatten())

    elif isinstance(value, gpuarray.GPUArray):
      if deepCopy:
        self.__value = value.copy().ravel()
      else:
        self.__value = value.ravel()
    else:
      try:
        flattend_value = np.array(value, dtype=np.float64).flatten()
        self.updateValue(flattend_value)
      except:
        raise ValueError("attribute.updateValue: Invalid value type, cannot be converted to gpuarray")

  @staticmethod
  def __check_heritage(a1: attribute, a2: attribute) -> attribute:
    from yasps.attributeHelper import checkHeritage
    return checkHeritage(a1, a2)


  # construct a new attribute from a list of attributes
  @staticmethod
  def to_array(children: List[Union[attribute, float, int]], rows: int, cols: int) -> attribute:
    if rows * cols != len(children):
      raise ValueError(f"attribute.to_array: number of elements must match the number of children. {rows} * {cols} != {len(children)}.")
    convertedChildren: List[attribute] = []
    for item in children:
      if isinstance(item, float):
        convertedChildren.append(attribute(float_value = item))
      elif isinstance(item, int):
        convertedChildren.append(attribute(float_value = float(item)))
      else:
        convertedChildren.append(item)
    if rows * cols == 1:
      # no need to create an array for a single element
      return convertedChildren[0]
    # let's get the correspondance
    youngest_child: attribute = convertedChildren[0]
    for i in range(1, len(convertedChildren)):
      youngest_child = attribute.__check_heritage(youngest_child, convertedChildren[i])
    return attribute(name = "", rows = rows, cols = cols, children = convertedChildren, operator = ARRAY, correspondance = youngest_child.correspondance)

  # every attribute is actually a vector or a mat
  # so accessing them through [] operator returns an access attribute
  def __getitem__(self, index: Union[int, Tuple[int, int]]) -> attribute:
    if isinstance(index, int):
      if index >= self.rows * self.cols:
        raise ValueError("attribute.__getitem__: index out of range.")
      if self.operator == TRANSPOSE:
        row = index // self.cols
        col = index % self.cols
        return self.children[0][col, row]
      if self.isFloatMat:
        return self.children[index]
      if self.operator == ARRAY:
        return self.children[index]
      if self.operator == SELECT:
        # for select, if the two values are the same
        # then we return it
        # otherwise we need to return another select
        true_value = self.children[1]
        false_value = self.children[2]
        if true_value[index].hash == false_value[index].hash:
          return true_value[index]
        else:
          return attribute.select(self.children[0], true_value[index], false_value[index])
      if self.operator == DATA:
        indexAttribute = attribute(operator = INDEX, index_value = index)
        return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
      if self.size == 1 and index == 0:
        return self
      indexAttribute = attribute(operator = INDEX, index_value = index)
      return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
    elif isinstance(index, tuple):
      if len(index) != 2:
        raise ValueError("attribute.__getitem__: index must be a tuple of two integers.")
      if index[0] >= self.rows or index[1] >= self.cols:
        raise ValueError(f"attribute.__getitem__: index out of range, acceessing index of {index} in a matrix of {self.rows}x{self.cols}.")
      if self.operator == TRANSPOSE:
        return self.children[0][index[1], index[0]]
      if self.isFloatMat:
        return self.children[index[0] * self.cols + index[1]]
      if self.operator == ARRAY:
        return self.children[index[0] * self.cols + index[1]]
      if self.operator == SELECT:
        # for select, if the two values are the same
        # then we return it
        # otherwise we need to return another select
        true_value = self.children[1]
        false_value = self.children[2]
        new_ind = index[0] * self.cols + index[1]
        if true_value[new_ind].hash == false_value[new_ind].hash:
          return true_value[new_ind]
        else:
          return attribute.select(self.children[0], true_value[new_ind], false_value[new_ind])
      if self.operator == DATA:
        indexAttribute = attribute(operator = INDEX, index_value = index[0] * self.cols + index[1])
        return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
      if self.size == 1 and index[0] == 0 and index[1] == 0:
        return self

      indexAttribute = attribute(operator = INDEX, index_value = index[0] * self.cols + index[1])
      return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
    else:
      raise ValueError("attribute.__getitem__: index must be either an integer or a tuple of two integers.")


  ################################################
  ################################################
  #     ATTRIBUTE STRING DEFINITION
  ################################################
  ################################################
  def __str__(self)->str:
    from yasps.attributeHelper import attribute2str
    return attribute2str(self)


  ################################################
  ################################################
  #     ATTRIBUTE OPERATION DEFINITION
  ################################################
  ################################################
  def row(self, index: int)->attribute:
    if self.size == 1:
      return self
    if index >= self.rows:
      raise ValueError("attribute.row: index out of range.")
    return attribute(children = [self, attribute(index_value = index)], operator = ROW, correspondance = self.correspondance, rows = 1, cols = self.cols)

  def col(self, index: int)->attribute:
    if self.size == 1:
      return self
    if index >= self.cols:
      raise ValueError("attribute.col: index out of range.")
    return attribute(children = [self, attribute(index_value = index)], operator = COL, correspondance = self.correspondance, rows = self.rows, cols = 1)

  ################################################
  # addition
  ################################################
  def __add__(self, other: Union[attribute, float])->attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self + other_attribute
    elif isinstance(other, attribute):
      from yasps.attributeOperations import add
      return add(self, other)
    raise ValueError("attribute.__add__: cannot add an attribute with a non-attribute.")

  def __radd__(self, other: float)->attribute:
    return self + other

  def add_explicit(self, other: attribute) -> attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self.add_explicit(other_attribute)
    elif isinstance(other, attribute):
      from yasps.attributeOperations import add_explicitly
      return add_explicitly(self, other)
    raise ValueError("attribute.add_explicit: cannot add an attribute with a non-attribute.")


  ################################################
  # subtraction
  ################################################
  def __sub__(self, other: Union[attribute, float])->attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self - other_attribute
    elif isinstance(other, attribute):
      from yasps.attributeOperations import sub
      return sub(self, other)
    raise ValueError("attribute.__sub__: cannot sub an attribute with a non-attribute.")

  def __rsub__(self, other: float)->attribute:
    return -self + other

  def sub_explicit(self, other: attribute) -> attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self.sub_explicit(other_attribute)
    elif isinstance(other, attribute):
      from yasps.attributeOperations import sub_explicitly
      return sub_explicitly(self, other)
    raise ValueError("attribute.sub_explicit: cannot sub an attribute with a non-attribute.")

  def __neg__(self)->attribute:
    if self.operator == FLOAT:
      return attribute(float_value = -self.float_value)
    if self.isFloatMat:
      return attribute(children = [-x for x in self.children], operator = ARRAY, correspondance = self.correspondance, rows = self.rows, cols = self.cols)
    return attribute(children = [self], operator = NEG, correspondance = self.correspondance, rows = self.rows, cols = self.cols)


  ################################################
  # multiplication
  ################################################
  def __mul__(self, other: Union[attribute, float])->attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self * other_attribute
    elif isinstance(other, attribute):
      from yasps.attributeOperations import mul
      return mul(self, other)
    raise ValueError("attribute.__mul__: cannot multiply an attribute with a non-attribute.")

  def __rmul__(self, other: float)->attribute:
    return self * other

  def mul_explicit(self, other) -> attribute:
    # explicitly multiply the elements out without using matrix operators
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self.mul_explicit(other_attribute)
    elif isinstance(other, attribute):
      from yasps.attributeOperations import mul_explicitly
      return mul_explicitly(self, other)
    raise ValueError("attribute.mul_explicit: cannot multiply an attribute with a non-attribute.")

  ################################################
  # division
  ################################################
  def __truediv__(self, other: Union[float, attribute]) -> attribute:
    if isinstance(other, float):
      return self * (1.0 / other)
    elif isinstance(other, attribute):
      from yasps.attributeOperations import div
      return div(self, other)
    raise ValueError(f"attribute.__truediv__: cannot divide an attribute by {type(other)}.")

  def __rtruediv__(self, other: float) -> attribute:
    if self.size == 1:
      return attribute(children = [attribute(float_value = other), self], operator = DIV, correspondance = self.correspondance, rows = self.rows, cols = self.cols)
    else:
      raise ValueError("attribute.__rtruediv__: cannot divide a non scalar")

  def div_explicit(self, other) -> attribute:
    if isinstance(other, float):
      return self * (1.0 / other)
    elif isinstance(other, attribute):
      from yasps.attributeOperations import div_explicitly
      return div_explicitly(self, other)
    raise ValueError(f"attribute.div_explicit: cannot divide an attribute by {type(other)}.")

  ################################################
  # all other things
  ################################################
  def pow(self, other: Union[float, attribute]) -> attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self.pow(other_attribute)
    elif isinstance(other, attribute):
      from yasps.attributeOperations import pow_op
      return pow_op(self, other)
    raise ValueError("attribute.pow: cannot raise an attribute to a non-attribute.")

  def sqrt(self) -> attribute:
    from yasps.attributeOperations import sqrt_op
    return sqrt_op(self)


  def log(self) -> attribute:
    from yasps.attributeOperations import log_op
    return log_op(self)

  def sin(self) -> attribute:
    from yasps.attributeOperations import sin_op
    return sin_op(self)

  def cos(self) -> attribute:
    from yasps.attributeOperations import cos_op
    return cos_op(self)

  def trace(self) -> attribute:
    if self.rows != self.cols:
      raise ValueError("attribute.trace: cannot compute trace of a non-square matrix.")
    result = self[0, 0]
    # print(f"At trace, {self.rows}, {self.cols}")
    for i in range(1, self.rows):
      result += self[i, i]
    return result

  def spd(self, spd_method: int = 1) -> attribute:
    import numpy as np
    # 0 for no projection
    # 1 for project negative eigen value to absolute value
    # 2 for project negative eigen value to 0
    if self.rows != self.cols:
      raise ValueError("attribute.spd: cannot compute spd projection of a non-square matrix.")
    if self.size == 1 and self.operator == FLOAT:
      if spd_method == 1:
        return attribute(float_value = abs(self.float_value))
      elif spd_method == 2:
        return attribute(float_value = max([0, self.float_value]))
    if self.isFloatMat:
      print("We are directly constructing positive definite matrix since input is constant value")
      # reconstruct the matrix in numpy
      mat = np.array([x.float_value for x in self.children], dtype = np.float64).reshape(self.rows, self.cols)
      # print("Directly projecting a float matrix")
      # print(mat)
      ev, evc = np.linalg.eig(mat)
      if spd_method == 1:
        ev = np.abs(ev)
      elif spd_method == 2:
        ev[ev < 0] = 0
      reconstructed_mat = evc @ np.diag(ev) @ evc.T
      m = [attribute(float_value = float(x)) for x in reconstructed_mat.flatten()]
      return attribute.to_array(m, self.rows, self.cols)
    if spd_method == 0:
      return self
    return attribute(children = [self, attribute(index_value = spd_method)], operator = SPD, correspondance = self.correspondance, rows = self.rows, cols = self.cols)



  def cross(self, other: attribute) -> attribute:
    if self.size != 3 or other.size != 3:
      raise ValueError("attribute.cross: cross product is only defined for 3D vectors.")
    # if self.rows != 3:
    #   return self.transpose().cross(other)
    # if other.rows != 3:
    #   return self.cross(other.transpose())
    u0 = self[0]
    u1 = self[1]
    u2 = self[2]
    v0 = other[0]
    v1 = other[1]
    v2 = other[2]
    return attribute.to_array([u1*v2 - u2*v1, u2*v0 - u0*v2, u0*v1 - u1*v0], rows = 3, cols = 1)

  def atan2(self, other: attribute) -> attribute:
    if self.size != 1 or other.size != 1:
      raise ValueError("attribute.atan2: atan2 is only defined for scalar attributes.")
    if self.operator == FLOAT and other.operator == FLOAT:
      import math
      return attribute(float_value = math.atan2(self.float_value, other.float_value))
    return attribute(children = [self, other], operator = ATAN2, correspondance = attribute.__check_heritage(self, other).correspondance, rows = self.rows, cols = self.cols)

  def abs(self) -> attribute:
    if self.operator == FLOAT:
      return attribute(float_value = abs(self.float_value))
    return attribute(children = [self], operator = ABS, correspondance = self.correspondance, rows = self.rows, cols = self.cols)

  # transpose operator
  def transpose(self)->attribute:
    if self.size == 0:
      return self
    if self.size == 1:
      return self
    else:
      if self.operator == TRANSPOSE:
        return self.children[0]
      else:
        return attribute(children = [self], operator = TRANSPOSE, correspondance = self.correspondance, rows = self.cols, cols = self.rows)

  def norm(self) -> attribute:
    if self.rows ==1 and self.cols == 1:
      return self # norm of self is self
    if self.rows == 1 or self.cols == 1:
      return attribute(children = [self], operator = NORM, correspondance = self.correspondance) # return the norm
    else:
      raise ValueError("attribute.norm: norm is only defined for vectors.")

  def inverse(self) -> attribute:
    if self.rows == 1 and self.cols == 1:
      return 1.0 / self
    else:
      if self.rows != self.cols:
        raise ValueError("attribute.inverse: cannot compute inverse of a non-square matrix.")
      return attribute(children = [self], operator = INV, correspondance = self.correspondance, rows = self.rows, cols = self.cols)

  def determinant(self) -> attribute:
    if self.rows == 1 and self.cols == 1:
      return self
    else:
      if self.rows != self.cols:
        raise ValueError("attribute.determinant: cannot compute determinant of a non-square matrix.")
      return attribute(children = [self], operator = DET, correspondance = self.correspondance, rows = 1, cols = 1)

  def dot(self, other: attribute) -> attribute:
    # just use dot explicit
    return self.dot_explicit(other)



  def dot_explicit(self, other: attribute) -> attribute:
    if self.size != other.size:
      raise ValueError("attribute.dot: cannot compute dot product of vectors of different sizes.")
    if not (self.rows == 1 or self.cols == 1):
      raise ValueError(f"attribute.dot: dot product is only defined for vectors, got dimension {self.rows}x{self.cols}.")
    if not (other.rows == 1 or other.cols == 1):
      raise ValueError(f"attribute.dot: dot product is only defined for vectors, got dimension {other.rows}x{other.cols}.")
    result = attribute(float_value = 0.0)
    for i in range(self.size):
      result += self[i] * other[i]
    return result

  def eq(self, other: attribute) -> attribute:
    if self.cols != other.cols or self.rows != other.rows:
      raise ValueError("attribute.eq: cannot compare attributes of different sizes.")
    return attribute(children = [self, other], operator = EQ, correspondance = attribute.__check_heritage(self, other).correspondance, rows = 1, cols = 1)

  def neq(self, other: attribute) -> attribute:
    if self.cols != other.cols or self.rows != other.rows:
      raise ValueError("attribute.neq: cannot compare attributes of different sizes.")
    return attribute(children = [self, other], operator = NEQ, correspondance = attribute.__check_heritage(self, other).correspondance, rows = 1, cols = 1)

  def __gt__(self, other: Union[attribute, float]) -> attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self > other_attribute
    elif isinstance(other, attribute):
      if self.cols != other.cols or self.rows != other.rows:
        raise ValueError("attribute.gt: cannot compare attributes of different sizes.")
      return attribute(children = [self, other], operator = GT, correspondance = attribute.__check_heritage(self, other).correspondance, rows = 1, cols = 1)
    else:
      raise ValueError("attribute.__gt__: other must be an attribute or a float.")

  def __ge__(self, other: Union[attribute, float]) -> attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self >= other_attribute
    elif isinstance(other, attribute):
      if self.cols != other.cols or self.rows != other.rows:
        raise ValueError("attribute.geq: cannot compare attributes of different sizes.")
      return attribute(children = [self, other], operator = GEQ, correspondance = attribute.__check_heritage(self, other).correspondance, rows = 1, cols = 1)
    else:
      raise ValueError("attribute.__ge__: other must be an attribute or a float.")

  @staticmethod
  def select(condition: attribute, true_attribute: attribute, false_attribute: attribute) -> attribute:
    # check dimension of true and false attribute
    if true_attribute.rows != false_attribute.rows or true_attribute.cols != false_attribute.cols:
      raise ValueError("attribute.select: true and false attributes must have the same dimension.")
    return attribute(children = [condition, true_attribute, false_attribute], operator = SELECT, correspondance = attribute.__check_heritage(true_attribute, false_attribute).correspondance, rows = true_attribute.rows, cols = true_attribute.cols)



  ################################################
  ################################################
  #     ATTRIBUTE HASH DEFINITION
  ################################################
  ################################################
  @property
  def hash(self)->int:
    if self.__hash != 0:
      return self.__hash
    else:
      from yasps.attributeHelper import hashAttribute
      return hashAttribute(self)

  def __hash__(self) -> int:
    return self.hash

  def __eq__(self, other) -> bool:
    return hash(self) == hash(other)


  ################################################
  ################################################
  #     ATTRIBUTE COMPUTATION DEFINITION
  ################################################
  ################################################
  def compute(self) -> attribute:
    start_compute = cuda.Event()
    end_compute = cuda.Event()
    start_compute.record()
    if self.__operator == DATA:
      # do nothing, its a data attribute
      return self
    if self.__globalKernel is None:
      from yasps.codeGenerator import codeGenerator
      from yasps.globalKernel import globalKernel
      start_generator = time.time()
      codegen: codeGenerator = codeGenerator(self) # this will generate the string for the device kernel and all of its descendants
      codegen.generateCode()
      end_generator = time.time()
      # print time in ms
      print(f"Code generation time: {(end_generator - start_generator) * 1000.0:.5f} ms")

      # now add the global kernel
      self.__globalKernel = globalKernel(self)

      # after we generate the kernel, we first check if our data is already allocated or if the size does not match
      assert self.__correspondance is not None # cannot be none
      assert self.__deviceKernel is not None # cannot be none
      assert self.__globalKernel is not None
      if self.__value is None or self.__value.size < self.__correspondance.numInstances * self.size:
        # reallocate a new pycuda array with the correct size
        self.__value = gpuarray.empty(self.__correspondance.numInstances * self.size, dtype=np.float64)

    assert self.__globalKernel is not None
    assert self.__value is not None
    self.__globalKernel.compute(self.__value)
    return self
    # print(self.value)


  ################################################
  ################################################
  #     AUTODIFF
  ################################################
  ################################################

  def __diff(self, wrt: List[attribute]) -> List[List[attribute]]:
    # we first check the operation of input
    # it can only be either an array access or a data
    for item in wrt:
      if item.operator != ARRAY_ACCESS and item.operator != DATA:
        raise ValueError(f"attribute.__diff: cannot differentiate with respect to {item.operator.name.upper()}, available operations are ARRAY_ACCESS and DATA")




    return []


  # def __diff_data(self, wrt: List[attribute]) -> List[List[attribute]]:
  #   # we check for any of the attribute, is it

# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Union, Tuple
from typing import TYPE_CHECKING
import hashlib # for hashing
from yasps.operator import operator
if TYPE_CHECKING:
  from yasps.operator import operator
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.primitive import primitive
  from yasps.connectivity import connectivity
  from yasps.deviceKernel import deviceKernel



ADD = operator("+", 1, True)
SUB = operator("-", 1, False)
MUL = operator("*", 1, True)
DIV = operator("/", 1, False)
POW = operator("pow", 2, False)
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
SCATTER = operator("scatter", 3, False) # for scattering data from one primitive to another
TRANSPOSE = operator("transpose", 3, False) # for transposing a matrix
BROADCAST_ADD = operator("+", 3, False) # broadcast an add to all elements
INTERMEDIATE = operator("intermediate", 3, False) # for intermediate results
ROW = operator("row", 3, False) # for row access
COL = operator("col", 3, False) # for column access


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
    self.__hash: int = 0
    self.__deviceKernel: Optional["deviceKernel"] = None



  @property
  def name(self)->str:
    return self.__name

  @property
  def fullName(self)->str:
    return self.correspondance.fullName + "_" + self.__name

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
  def value(self) -> gpuarray.GPUArray:
    if self.__value is None:
      raise ValueError("attribute.value: value is None. Please call compute() first or manually update value.")
    return self.__value

  @property
  def size(self) -> int:
    return self.rows * self.cols


  def reshape(self, rows, cols):
    if self.rows * self.cols != rows * cols:
      raise ValueError("attribute.reshape: new shape must have the same number of elements.")
    self.__n_rows = rows
    self.__n_cols = cols

  @property
  def operator(self):
    return self.__operator

  def setName(self, name) -> None:
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
        flattend_value = np.array(value, dtype=np.float64).flatten()
        self.updateValue(flattend_value)
      except:
        raise ValueError("attribute.updateValue: Invalid value type, cannot be converted to gpuarray")

  @staticmethod
  def __check_heritage(a1: attribute, a2: attribute)->attribute:
    # we check if two attribute are from the same line of blood
    # return the younger one always
    if a1.correspondance is None and a1.operator != FLOAT:
      raise ValueError("attribute.__check_heritage: a1 must have a correspondance since it is not a float value.")
    if a2.correspondance is None and a2.operator != FLOAT:
      raise ValueError("attribute.__check_heritage: a2 must have a correspondance since it is not a float value.")
    if a1.operator == FLOAT or a2.operator == FLOAT:
      return a1

    if a1.correspondance is None or a2.correspondance is None:
      raise ValueError("attribute.__check_heritage: correspondance should be set for both attributes.")
    else:
      if a1.correspondance.fullName == a2.correspondance.fullName:
        # same correspondance, we can return either one
        return a1
      if a1.correspondance.type == "scene":
        # a1 is a scene, we check if a2 is a child of a1
        if a2.correspondance.scene.fullName == a1.fullName:
          return a2
      if a2.correspondance.type == "scene":
        # same scenario
        if a1.correspondance.scene.fullName == a2.fullName:
          return a1
      # now we actually need to check the heritage
      if a1.correspondance.type == "mesh":
        if a2.correspondance.mesh.fullName == a1.correspondance.fullName:
          return a2
      if a2.correspondance.type == "mesh":
        if a1.correspondance.mesh.fullName == a2.correspondance.fullName:
          return a1
      # we dont need to check for primitives, sicen if they are the same
      # then we already checked it
      # if they are not the same, we raise error anyway
      raise ValueError("attribute.__check_heritage: attributes do not share the same heritage.")


  # construct a new attribute from a list of attributes
  @staticmethod
  def to_array(children: List[attribute], rows: int, cols: int):
    if rows * cols != len(children):
      raise ValueError("attribute.to_array: number of elements must match the number of children.")
    # let's get the correspondance
    youngest_child: attribute = children[0]
    for i in range(1, len(children)):
      youngest_child = attribute.__check_heritage(youngest_child, children[i])
    return attribute(name = "", rows = rows, cols = cols, children = children, operator = ARRAY, correspondance = youngest_child.correspondance)

  # every attribute is actually a vector or a mat
  # so accessing them through [] operator returns an access attribute
  def __getitem__(self, index: Union[int, Tuple[int, int]]) -> attribute:
    if isinstance(index, int):
      if index >= self.rows * self.cols:
        raise ValueError("attribute.__getitem__: index out of range.")
      if self.operator == ARRAY:
        return self.children[index]
      elif self.operator == DATA:
        indexAttribute = attribute(operator = INDEX, index_value = index)
        return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
      indexAttribute = attribute(operator = INDEX, index_value = index)
      return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
    elif isinstance(index, tuple):
      if len(index) != 2:
        raise ValueError("attribute.__getitem__: index must be a tuple of two integers.")
      if index[0] >= self.rows or index[1] >= self.cols:
        raise ValueError("attribute.__getitem__: index out of range.")
      if self.operator == ARRAY:
        return self.children[index[0] * self.cols + index[1]]
      elif self.operator == DATA:
        indexAttribute = attribute(operator = INDEX, index_value = index[0] * self.cols + index[1])
        return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
      indexAttribute = attribute(operator = INDEX, index_value = index[0] * self.cols + index[1])
      return attribute(children = [self, indexAttribute], operator = ARRAY_ACCESS, correspondance = self.correspondance)
    else:
      raise ValueError("attribute.__getitem__: index must be either an integer or a tuple of two integers.")

  # transpose operator
  def transpose(self)->attribute:
    if self.size == 0:
      return self
    else:
      if self.operator == TRANSPOSE:
        return self.children[0]
      else:
        return attribute(children = [self], operator = TRANSPOSE, correspondance = self.correspondance, rows = self.cols, cols = self.rows)

  def __str__(self)->str:
    if self.operator.type == 0:
      return f"{self.operator.name}({self.children[0]})"
    elif self.operator.type == 1:
      return f"({self.children[0]} {self.operator.name} {self.children[1]})"
    elif self.operator.type == 2:
      return f"{self.operator.name}({', '.join([str(child) for child in self.children])})"
    elif self.operator.type == 3:
      if self.operator == INDEX:
        return str(self.index_value)
      elif self.operator == FLOAT:
        return str(self.float_value)
      elif self.operator == ARRAY_ACCESS:
        return f"{self.children[0]}[{self.children[1]}]"
      elif self.operator == DATA:
        if self.correspondance is not None:
          return f"{self.correspondance.fullName}"
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
        return f"gather({self.through.toPrimitive.fullName}.{self.name}->{self.through.fromPrimitive.fullName}.{self.name})"
      elif self.operator == SCATTER:
        if len(self.children) != 1:
          raise ValueError("attribute.__str__: SCATTER operator must have one child.")
        if self.children[0].correspondance is None:
          raise ValueError("attribute.__str__: SCATTER operator's first child must have a correspondance.")
        if self.through is None:
          raise ValueError("attribute.__str__: SCATTER operator must have a through attribute.")
        return f"scatter({self.through.fromPrimitive.fullName}.{self.name}->{self.through.toPrimitive.fullName}.{self.name})"
      elif self.operator == ROW:
        if len(self.children) != 2:
          raise ValueError("attribute.__str__: ROW operator must have two children.")
        return f"{self.children[0]}.row({self.children[1]})"
      elif self.operator == COL:
        if len(self.children) != 2:
          raise ValueError("attribute.__str__: COL operator must have two children.")
        return f"{self.children[0]}.col({self.children[1]})"
      elif self.operator == TRANSPOSE:
        if len(self.children) != 1:
          raise ValueError("attribute.__str__: TRANSPOSE operator must have one child.")
        return f"{self.children[0]}.transpose()"
      elif self.operator == BROADCAST_ADD:
        print("At broadcast add")
        print(self.children[0])
        print(self.children[1])
        return f"{self.children[0]} + {self.children[1]}"
      else:
        raise ValueError("attribute.__str__: unknown operator type.")
    else:
      raise ValueError("attribute.__str__: unknown operator type.")


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

  # here we define the operator overloading
  def __add__(self, other: Union[attribute, float])->attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self + other_attribute
    elif isinstance(other, attribute):
      if other.operator == FLOAT:
        if other.float_value == 0:
          return self
      if self.size == 1 or other.size == 1:
        return attribute(children = [self, other], operator = ADD, correspondance = attribute.__check_heritage(self, other).correspondance, rows = max(self.rows, other.rows), cols = max(self.cols, other.cols))
      elif other.size == 1:
        return attribute(children = [self, other], operator = BROADCAST_ADD, correspondance = attribute.__check_heritage(self, other).correspondance, rows = self.rows, cols = self.cols)
      elif self.size == 1:
        return attribute(children = [other, self], operator = BROADCAST_ADD, correspondance = attribute.__check_heritage(self, other).correspondance, rows = other.rows, cols = other.cols)
      else:
        if self.rows == other.rows and self.cols == other.cols:
          return attribute(children = [self, other], operator = ADD, correspondance = attribute.__check_heritage(self, other).correspondance, rows = self.rows, cols = self.cols)
        else:
          raise ValueError("attribute.__add__: cannot add two attributes of different dimensions.")
    raise ValueError("attribute.__add__: cannot add an attribute with a non-attribute.")

  def __radd__(self, other: Union[attribute, float])->attribute:
    return self + other

  def __mul__(self, other: Union[attribute, float])->attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self * other_attribute
    elif isinstance(other, attribute):
      if other.operator == FLOAT:
        if other.float_value == 1:
          return self
        # else:
        #   return attribute(children = [self, other], operator = MUL, correspondance = self, rows = self.rows, cols = self.cols)
      if self.operator == FLOAT:
        if self.float_value == 1:
          return other
        # else:
        #   return attribute(children = [self, other], operator = MUL, correspondance = other, rows = other.rows, cols = other.cols)
      if self.size == 1 or other.size == 1:
        return attribute(children = [self, other], operator = MUL, correspondance = attribute.__check_heritage(self, other).correspondance, rows = max(self.rows, other.rows), cols = max(self.cols, other.cols))
      else:
        if self.cols == other.rows:
          return attribute(children = [self, other], operator = MUL, correspondance = attribute.__check_heritage(self, other).correspondance, rows = self.rows, cols = other.cols)
        else:
          raise ValueError(f"attribute.__mul__: dimension mismatch, cannot multiply {self.rows}x{self.cols} with {other.rows}x{other.cols}.")
    raise ValueError("attribute.__mul__: cannot multiply an attribute with a non-attribute.")

  def __rmul__(self, other: Union[attribute, float])->attribute:
    if isinstance(other, float):
      other_attribute = attribute(float_value = other)
      return self * other_attribute
    elif isinstance(other, attribute):
      return other * self
    raise ValueError("attribute.__rmul__: cannot multiply an attribute with a non-attribute.")

  @property
  def hash(self)->int:
    if self.__hash != 0:
      return self.__hash

    if self.operator == ADD or self.operator == BROADCAST_ADD:
      self.__hash = sum([child.hash for child in self.children])
    elif self.operator == MUL:
      self.__hash = self.children[0].hash * self.children[1].hash
    elif self.operator == SUB:
      self.__hash = self.children[0].hash - self.children[1].hash
    elif self.operator == DIV:
      division_string:str = f"{self.children[0].hash}/{self.children[1].hash}"
      self.__hash = int(hashlib.sha256(division_string.encode()).hexdigest(), 16)
    elif self.operator == POW:
      power_string:str = f"{self.children[0].hash}**{self.children[1].hash}"
      self.__hash = int(hashlib.sha256(power_string.encode()).hexdigest(), 16)
    elif self.operator == NEG:
      self.__hash = -self.children[0].hash
    elif self.operator == SIN:
      sin_string:str = f"sin({self.children[0].hash})"
      self.__hash = int(hashlib.sha256(sin_string.encode()).hexdigest(), 16)
    elif self.operator == COS:
      cos_string:str = f"cos({self.children[0].hash})"
      self.__hash = int(hashlib.sha256(cos_string.encode()).hexdigest(), 16)
    elif self.operator == TAN:
      tan_string:str = f"tan({self.children[0].hash})"
      self.__hash = int(hashlib.sha256(tan_string.encode()).hexdigest(), 16)
    elif self.operator == COT:
      cot_string:str = f"cot({self.children[0].hash})"
      self.__hash = int(hashlib.sha256(cot_string.encode()).hexdigest(), 16)
    elif self.operator == ABS:
      self.__hash = abs(self.children[0].hash)
    elif self.operator == SELECT:
      select_string:str = f"select({self.children[0].hash},{self.children[1].hash},{self.children[2].hash})"
      self.__hash = int(hashlib.sha256(select_string.encode()).hexdigest(), 16)
    elif self.operator == SQRT:
      sqrt_string:str = f"sqrt({self.children[0].hash})"
      self.__hash = int(hashlib.sha256(sqrt_string.encode()).hexdigest(), 16)
    elif self.operator == EQ:
      eq_string:str = f"{self.children[0].hash} == {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(eq_string.encode()).hexdigest(), 16)
    elif self.operator == NE:
      ne_string:str = f"{self.children[0].hash} != {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(ne_string.encode()).hexdigest(), 16)
    elif self.operator == GT:
      gt_string:str = f"{self.children[0].hash} > {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(gt_string.encode()).hexdigest(), 16)
    elif self.operator == GE:
      ge_string:str = f"{self.children[0].hash} >= {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(ge_string.encode()).hexdigest(), 16)
    elif self.operator == LT:
      lt_string:str = f"{self.children[0].hash} < {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(lt_string.encode()).hexdigest(), 16)
    elif self.operator == LE:
      le_string:str = f"{self.children[0].hash} <= {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(le_string.encode()).hexdigest(), 16)
    elif self.operator == ASSIGN:
      assign_string:str = f"{self.children[0].hash} = {self.children[1].hash}"
      self.__hash = int(hashlib.sha256(assign_string.encode()).hexdigest(), 16)
    elif self.operator == INDEX:
      return self.__index_value
    elif self.operator == ARRAY_ACCESS:
      array_access_string:str = f"{self.children[0].hash}[{self.children[1].hash}]"
      self.__hash = int(hashlib.sha256(array_access_string.encode()).hexdigest(), 16)
    elif self.operator == ARRAY:
      array_string:str = f"[{','.join([str(child.hash) for child in self.children])}]"
      self.__hash = int(hashlib.sha256(array_string.encode()).hexdigest(), 16)
    elif self.operator == FLOAT:
      float_str = str(self.float_value).encode()
      # Compute the SHA-256 hash
      hash_hex = hashlib.sha256(float_str).hexdigest()
      # Convert the hexadecimal hash to an integer
      hash_int = int(hash_hex, 16)
      self.__hash = hash_int
    elif self.operator == DATA:
      fullname = str(self)
      # Compute the SHA-256 hash and convert to an integer
      hash_hex = hashlib.sha256(fullname.encode()).hexdigest()
      hash_int = int(hash_hex, 16)
      self.__hash = hash_int
    elif self.operator == GATHER:
      if self.through is None:
        raise ValueError("attribute.hash: Gather operator must have a through attribute.")
      gather_string:str = f"gather({self.through.toPrimitive[self.name].hash}_through_{self.through})"
      self.__hash = int(hashlib.sha256(gather_string.encode()).hexdigest(), 16)
    elif self.operator == SCATTER:
      if self.through is None:
        raise ValueError("attribute.hash: Scatter operator must have a through attribute.")
      scatter_string:str = f"scatter({self.through.fromPrimitive[self.name].hash}, {self.through.toPrimitive[self.name].hash})"
      self.__hash = int(hashlib.sha256(scatter_string.encode()).hexdigest(), 16)
    elif self.operator == TRANSPOSE:
      transpose_string:str = f"transpose({self.children[0].hash})"
      self.__hash = int(hashlib.sha256(transpose_string.encode()).hexdigest(), 16)
    elif self.operator == ROW:
      row_string:str = f"{self.children[0].hash}.row({self.children[1].hash})"
      self.__hash = int(hashlib.sha256(row_string.encode()).hexdigest(), 16)
    elif self.operator == COL:
      col_string:str = f"{self.children[0].hash}.col({self.children[1].hash})"
      self.__hash = int(hashlib.sha256(col_string.encode()).hexdigest(), 16)
    return self.__hash

  def __hash__(self):
    return self.hash

  def __eq__(self, other) -> bool:
    return hash(self) == hash(other)

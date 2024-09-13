# cython: language_level=3
from __future__ import annotations
from typing import Dict, Union, Tuple, Optional
import keyword
import numpy as np
# a primitive may have its own attributes
# and connectivities to other primitives
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.attribute import attribute
  from yasps.connectivity import connectivity



class primitive:
  def __init__(self, name: str, parent_mesh: mesh, numInstances: int = 0):
    if name == "":
      raise ValueError("mesh.__init__: name cannot be empty.")
    if parent_mesh is None:
      raise ValueError("mesh.__init__: mesh cannot be None.")
    self.__name: str = name
    self.__mesh: mesh = parent_mesh
    self.__connectivities: Dict[str, primitive] = {}
    self.__attributes: Dict[str, attribute] = {}
    self.__numInstances: int = numInstances


  @property
  def name(self)->str:
    return self.__name

  @property
  def mesh(self)->mesh:
    return self.__mesh

  @property
  def scene(self)->scene:
    return self.mesh.scene

  @property
  def type(self)->str:
    return "primitive"

  @property
  def numInstances(self)->int:
    return self.__numInstances

  @property
  def attributes(self)->Dict[str, attribute]:
    return self.__attributes

  @property
  def numConnectivities(self)->int:
    return len(self.__connectivities)

  def isValidName(self, name: str)->bool:
    # Check if the name is a valid Python identifier (variable name)
    if not name.isidentifier():
      return False
    # Check if the name is a reserved Python keyword
    if name in keyword.kwlist:
      return False
    # Check if the name conflicts with existing scene attributes or methods
    if hasattr(self, name):
      return False
    return True

  def addConnectivity(self, name:str, to: primitive, data: np.ndarray) -> connectivity:
    if name in self.__connectivities:
      raise ValueError(f"primitive.addConnectivity: connectivity with name '{name}' already exists in primitive.")

    # check name is valid
    if not self.isValidName(name):
      raise ValueError(f"primitive.addConnectivity: '{name}' is not a valid name for a connectivity.")

    # check if the connectivity is the same as the mesh
    if to.mesh != self.mesh:
      raise ValueError(f"primitive.addConnectivity: connectivity '{name}' must connect to the same mesh.")

    from yasps import connectivity
    newConnectivity = connectivity(name = name, fromPrimitive = self, toPrimitive = to, data = data)
    self.__connectivities[name] = newConnectivity
    return newConnectivity


  def addAttribute(self, name: str, computed_attribute: Optional[attribute] = None, rows: int = 1, cols: int = 1, through: Optional[connectivity] = None) -> attribute:
    from yasps.attribute import attribute
    if name in self.__attributes:
      raise ValueError(f"primitive.addAttribute: attribute with name '{name}' already exists in primitive.")
    if computed_attribute != None:
      if computed_attribute.name != "":
        raise ValueError(f"primitive.addAttribute: the computed_attribute supplied already has a name '{computed_attribute.name}'. This indicates that the computed_attribute most likely is already set for another object.")
        self.__attributes[name] = computed_attribute
        computed_attribute.setName(name)
        return computed_attribute
    elif through is not None:
      # we will gather the attribute from another primitive
      # we will check if the name is inside the to primitive
      toPrimitive = through.toPrimitive
      if name not in toPrimitive.__attributes:
        raise ValueError(f"primitive.addAttribute: attribute with name '{name}' does not exist in primitive '{toPrimitive.name}'. The through construction is not successful.")
      newAttribute = attribute(name = name, correspondance = self, rows = through.dimension, cols = toPrimitive[name].rows * toPrimitive.cols, through = through, children = [toPrimitive[name]])
      print(f"The attribute {name} for primitive {self.name} now has dimension {newAttribute.rows}x{newAttribute.cols}")
      self.__attributes[name] = newAttribute
      return newAttribute
    else:
      newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols)
      self.__attributes[name] = newAttribute
      return newAttribute

  @property
  def fullName(self)->str:
    return f"{self.scene.name}_{self.mesh.name}_{self.name}"

  # accessing attribute by [] operator
  # user can get multiple attributes by passing a list of names
  def __getitem__(self, key: Union[str, Tuple[str]]) -> attribute:
    if isinstance(key, str):
      if key in self.__attributes:
        return self.__attributes[key]
      else:
        raise KeyError(f"primitive.__getitem__: attribute with name '{key}' not found in primitive.")
    elif isinstance(key, tuple):
      from yasps.attribute import attribute
      # first we check if all names are in the attributes
      for name in key:
        if name not in self.__attributes:
          raise KeyError(f"primitive.__getitem__: attribute with name '{name}' not found in primitive {self.fullName}.")
      # get the list of attributes first
      attributes = attribute.to_array([self.__attributes[name] for name in key], 1, sum([self.__attributes[name].cols * self.__attributes[name].rows for name in key]))
      return attributes
    else:
      raise KeyError(f"primitive.__getitem__: attribute with name '{key}' not found in primitive.")

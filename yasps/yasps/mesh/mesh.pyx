# cython: language_level=3
from __future__ import annotations
from typing import Dict, Union, Tuple
# a mesh may have primitives
# and its own attributes
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.scene import scene  # Only imported for type hints
  from yasps.primitive import primitive
  from yasps.attribute import attribute


import keyword
class mesh:
  def __init__(self, name: str, parent_scene: scene):

    if name == "":
      raise ValueError("mesh.__init__: name cannot be empty.")
    if parent_scene is None:
      raise ValueError("mesh.__init__: parent scene cannot be None.")
    self.__name: str = name
    self.__scene: scene = parent_scene
    self.__primitives: Dict[str, primitive] = {}
    self.__attributes: Dict[str, attribute] = {}


  @property
  def name(self)->str:
    return self.__name

  @property
  def scene(self)->scene:
    return self.__scene

  @property
  def type(self)->str:
    return "mesh"

  @property
  def attributes(self)->Dict[str, attribute]:
    return self.__attributes

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

  def addPrimitive(self, name:str, numInstances: int)->primitive:
    if name in self.__primitives:
      raise ValueError(f"mesh.addPrimitive: primitive with name '{name}' already exists in mesh.")

    # check name is valid
    if not self.isValidName(name):
      raise ValueError(f"mesh.addPrimitive: '{name}' is not a valid name for a primitive.")

    from yasps.primitive import primitive
    # add the mesh to the scene
    newPrimitive = primitive(name, self, numInstances)
    self.__primitives[name] = newPrimitive
    # add mesh as an attribute to the scene
    setattr(self, name, newPrimitive)
    return newPrimitive


  @property
  def numInstances(self)->int:
    return 1

  @property
  def numPrimitives(self)->int:
    return len(self.__primitives)


  def addAttribute(self, name, attribute: attribute = None, rows: int = 1, cols: int = 1):
    if name in self.__attributes:
      raise ValueError(f"mesh.addAttribute: attribute with name '{name}' already exists in mesh.")
    if attribute != None:
      if attribute.name != "":
        raise ValueError(f"mesh.addAttribute: the attribute supplied already has a name '{attribute.name}'. This indicates that the attribute most likely is already set for another object.")
      self.__attributes[name] = attribute
      attribute.setName(name)
      return attribute
    else:
      from yasps.attribute import attribute
      newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols)
      self.__attributes[name] = newAttribute
      return newAttribute

  # accessing attributes by name
  # user can get multiple attributes by passing a list of names
  def __getitem__(self, key: Union[str, Tuple[str]]) -> attribute:
    if isinstance(key, str):
      if key in self.__attributes:
        return self.__attributes[key]
      else:
        raise KeyError(f"mesh.__getitem__: attribute with name '{key}' not found in mesh.")
    elif isinstance(key, tuple):
      # get the list of attributes first
      attributes = [self[name] for name in key]
    else:
      raise KeyError(f"mesh.__getitem__: attribute with name '{key}' not found in mesh.")

  @property
  def fullName(self)->str:
    return f"{self.__scene.name}_{self.__name}"

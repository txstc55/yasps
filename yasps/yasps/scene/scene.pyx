from __future__ import annotations
# a scene can have meshes
# and its own attributes
from typing import Dict, Union, Tuple
from yasps.mesh import mesh
from yasps.attribute import attribute
import keyword
class scene:
  scenes: Dict[str, scene] = {}
  def __init__(self, name):
    # check if name exists
    if name == "":
      raise ValueError("scene.__init__: name cannot be empty.")
    if name in scene.scenes:
      raise ValueError(f"scene.__init__: scene with name '{name}' already exists in scenes.")
    self.__name = name
    # add self to the scenes dict for easy lookup
    scene.scenes[name] = self

    # scene has meshes and itself some attributes
    self.__meshes: Dict[str, mesh]= {}
    self.__attributes: Dict[str, attribute] = {}


  @property
  def name(self)->str:
    return self.__name

  @property
  def type(self)->str:
    return "scene"

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


  def addMesh(self, name: str) -> mesh:
    if name in self.__meshes:
      raise ValueError(f"scene.addMesh: mesh with name '{name}' already exists in scene.")

    # check name is valid
    if not self.isValidName(name):
      raise ValueError(f"scene.addMesh: '{name}' is not a valid name for a mesh.")

    # add the mesh to the scene
    newMesh = mesh(name)
    self.__meshes[name] = newMesh
    # add mesh as an attribute to the scene
    setattr(self, name, newMesh)
    return newMesh

  @attribute
  def numInstances(self):
    return 1

  @attribute
  def numMeshes(self):
    return len(self.__meshes)

  def addAttribute(self, name, attribute: attribute = None, rows: int = 1, cols: int = 1):
    if name in self.__attributes:
      raise ValueError(f"scene.addAttribute: attribute with name '{name}' already exists in scene.")
    if attribute != None:
      if attribute.name != "":
        raise ValueError(f"scene.addAttribute: the attribute supplied already has a name '{attribute.name}'. This indicates that the attribute most likely is already set for another object.")
      self.__attributes[name] = attribute
      attribute.setName(name)
      return attribute
    else:
      newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols)
      self.__attributes[name] = newAttribute
      return newAttribute

  # accessing attribute by [] operator
  # user can get multiple attributes by passing a list of names
  def __getitem__(self, key: Union[str, Tuple[str]]) -> attribute:
    if isinstance(key, str):
      if key in self.__attributes:
        return self.__attributes[key]
      else:
        raise KeyError(f"scene.__getitem__: attribute with name '{key}' not found in scene.")
    elif isinstance(key, tuple):
      # get the list of attributes first
      attributes = [self[name] for name in key]
    else:
      raise KeyError(f"scene.__getitem__: attribute with name '{key}' not found in scene.")

  def fullName(self):
    return self.name

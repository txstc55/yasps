from __future__ import annotations
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np


# a scene can have meshes
# and its own attributes
from typing import Dict, Union, Tuple, Optional, List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.mesh import mesh  # Only imported for type hints
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
    self.__energies: Dict[int, attribute] = {}
    from yasps.minimizer import minimizer
    self.__minimizer: minimizer = minimizer()




  @property
  def name(self)->str:
    return self.__name


  # return the scene of this scene
  @property
  def scene(self)->scene:
    return self

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

    from yasps.mesh import mesh
    # add the mesh to the scene
    newMesh = mesh(name, self)
    self.__meshes[name] = newMesh
    # add mesh as an attribute to the scene
    setattr(self, name, newMesh)
    return newMesh

  @property
  def numInstances(self)->int:
    return 1

  @property
  def numMeshes(self)->int:
    return len(self.__meshes)

  def addAttribute(self, name, computed_attribute: Optional[attribute] = None, rows: int = 1, cols: int = 1)->attribute:
    if name in self.__attributes:
      raise ValueError(f"scene.addAttribute: attribute with name '{name}' already exists in scene.")
    if computed_attribute is not None:
      if computed_attribute.name != "":
        raise ValueError(f"scene.addAttribute: the computed_attribute supplied already has a name '{computed_attribute.name}'. This indicates that the input_attribute most likely is already set for another object.")
      self.__attributes[name] = computed_attribute
      computed_attribute.setName(name)
      return computed_attribute
    else:
      from yasps.attribute import attribute
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
      from yasps.attribute import attribute
      # first we check if all names are in the attributes
      for name in key:
        if name not in self.__attributes:
          raise KeyError(f"primitive.__getitem__: attribute with name '{name}' not found in scene {self.fullName}.")
      attributes = attribute.to_array([self.__attributes[name] for name in key], 1, sum([self.__attributes[name].cols * self.__attributes[name].rows for name in key]))
      return attributes
    else:
      raise KeyError(f"scene.__getitem__: attribute with name '{key}' not found in scene.")

  @property
  def fullName(self) -> str:
    return self.name

  @property
  def energyes(self) -> Dict[int, attribute]:
    return self.__energies

  def addEnergy(self, e: attribute) -> None:
    from yasps.energy import energy
    newEnergy = energy(e)
    if e.name == "":
      raise ValueError("scene.addEnergy: energy attribute must have a name.")
    self.__minimizer.addEnergy(newEnergy)


  def minimizeEnergy(self, ):
    self.__minimizer.computeHessianAndGradient()
    return self.__minimizer.gradientSegments

  def addMinimizeTarget(self, target: List[attribute]):
    self.__minimizer.addWrt(target)
    self.__minimizer.generateHessianAndGradient()

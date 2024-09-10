from __future__ import annotations
# a scene can have meshes
# and its own attributes
from typing import Dict, Union, Tuple
from yasps.mesh import mesh
from yasps.attribute import attribute
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

    # add to scene
    self.__meshes: Dict[str, mesh]= {}
    self.__attributes: Dict[str, attribute] = {}


  @property
  def name(self)->str:
    return self.__name

  @property
  def type(self)->str:
    return "scene"

  def addMesh(self, name: str):
    if name in self.__meshes:
      raise ValueError(f"scene.addMesh: mesh with name '{name}' already exists in scene.")

    # add the mesh to the scene
    newMesh = mesh(name)
    self.__meshes[name] = newMesh
    return newMesh



  def addAttribute(self, name, attribute: attribute = None, value = None, dimension = None):
    if name in self.__attributes:
      raise ValueError(f"scene.addAttribute: attribute with name '{name}' already exists in scene.")
    if attribute != None:
      self.__attributes[name] = attribute
      return attribute
    else:
      if value == None and dimension == None:
        raise ValueError("scene.addAttribute: value and dimension cannot both be None if attribute is None.")
      if value != None:
        newAttribute = attribute(name, value = value, correspondance = [self])
        self.__attributes[name] = newAttribute
        return newAttribute
      elif dimension != None:
        newAttribute = attribute(name, dimension = dimension, correspondance = [self])
        self.__attributes[name] = newAttribute
        return newAttribute

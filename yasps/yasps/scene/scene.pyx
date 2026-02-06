from __future__ import annotations
import os


# a scene can have meshes
# and its own attributes
from typing import Dict, Union, Tuple, Optional, List, Set
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.mesh import mesh  # Only imported for type hints
  from yasps.attribute import attribute
  from yasps.minimizer import minimizer as yminimizer

import keyword
class scene:
  scenes: Dict[str, scene] = {}
  def __init__(self, name):
    # mkdir .yasps_tmp if it does not exist
    if not os.path.exists(".yasps_tmp"):
      os.makedirs(".yasps_tmp")
    if not os.path.exists(".yasps_constant"):
      os.makedirs(".yasps_constant")
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
    self.__minimizer: yminimizer = minimizer()
    self.__seen_pre_targets_full_names: Set[str] = set() # for recording partial tagets for any energy added, because maybe for some energy it doesnt want to optimize wrt all the targets supported in the end




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

  @property
  def minimizer(self) -> yminimizer:
    return self.__minimizer


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

  def addConstant(self, name, rows: int = 1, cols: int = 1) -> attribute:
    # add a constant attribute to the scene
    # constant attributes will never be considered to be any part of minimization
    if name in self.__attributes:
      raise ValueError(f"scene.addConstant: attribute with name '{name}' already exists in scene.")
    from yasps.attribute import attribute
    newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols, is_constant = True)
    self.__attributes[name] = newAttribute
    return newAttribute

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
  def energies(self) -> Dict[int, attribute]:
    return self.__energies

  def addEnergy(self, e: attribute, targets: List[attribute] = [], projection_method = 1, save_intermediate = False, gradient_only = False, dynamic_instances = False, separate_hessian_jacobian = False) -> None:
    # projection_method = 0 means no projection, 1 means project eigen value to absolute, 2 means project eigen value to max(e, 0)
    # save_intermediate = True means save intermediate results for gradient and hessian computation
    # gradient only means in the CG system we will not have the hessian
    if e.name == "":
      raise ValueError("scene.addEnergy: energy attribute must have a name.")
    # we add the names of the targes to the pre_targets_full_names set
    for t in targets:
      self.__seen_pre_targets_full_names.add(t.fullName)
    self.__minimizer.addEnergy(e, targets = targets, projection_method = projection_method, save_intermediate = save_intermediate, gradient_only = gradient_only, dynamic_instances = dynamic_instances, separate_hessian_jacobian = separate_hessian_jacobian)


  def minimizeEnergy(self, tolerance = 1e-3, maxIterations = 20000):
    error_code = self.__minimizer.computeHessianAndGradient(tolerance = tolerance, maxIterations = maxIterations)
    if error_code < 0:
      print("scene.minimizeEnergy: got error code", error_code)
      # return []
    return self.__minimizer.solutionSegments

  @property
  def gradientSegments(self):
    return self.__minimizer.gradientSegments

  @property
  def gradient(self):
    return self.__minimizer.gradient

  @property
  def diagonal(self):
    return self.__minimizer.diagonal

  def addMinimizeTarget(self, target: List[attribute]):
    # we check if the target matches the pre_targets_full_names set
    target_full_name_set = set([t.fullName for t in target])
    if len(target_full_name_set) != len(target):
      raise ValueError("scene.addMinimizeTarget: target contains duplicate attributes.")
    # check if the target full name set contains all the pre_targets_full_names
    if not self.__seen_pre_targets_full_names.issubset(target_full_name_set):
      missing = self.__seen_pre_targets_full_names - target_full_name_set
      raise ValueError(f"scene.addMinimizeTarget: target is missing attributes {missing} that are required by the energies added.")
    self.__minimizer.addWrt(target)
    self.__minimizer.generateHessianAndGradient()

  def computeTotalEnergy(self) -> float:
    return self.__minimizer.computeTotalEnergy()

  def ignoreEnergies(self, energies: List[attribute]) -> None:
    self.__minimizer.ignoreEnergies(energies)

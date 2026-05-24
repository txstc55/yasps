# cython: language_level=3
from __future__ import annotations
from typing import Dict, Union, Optional, List
import keyword
import numpy as np
# a primitive may have its own attributes
# and connectivities to other primitives
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.mesh import mesh
  from yasps.attribute import attribute
  from yasps.connectivity import connectivity
  from yasps.scene import scene as yscene



class primitive:
  def __init__(self, name: str, parent_mesh: mesh, numInstances: int = 0, isDynamic: bool = False):
    if name == "":
      raise ValueError("mesh.__init__: name cannot be empty.")
    if parent_mesh is None:
      raise ValueError("mesh.__init__: mesh cannot be None.")
    self.__name: str = name
    self.__mesh: mesh = parent_mesh
    self.__connectivities: Dict[str, connectivity] = {}
    self.__attributes: Dict[str, attribute] = {}
    self.__numInstances: int = numInstances
    self.__isDynamic: bool = isDynamic


  # check if has dynamic instances
  @property
  def isDynamic(self)->bool:
    return self.__isDynamic

  @property
  def name(self)->str:
    return self.__name

  # return the mesh of this primitive
  @property
  def mesh(self)->mesh:
    return self.__mesh

  # return the scene of this primitive
  @property
  def scene(self)-> yscene:
    return self.mesh.scene

  # return the primitive of this primitive
  @property
  def primitive(self)->primitive:
    return self

  @property
  def connectivities(self)->Dict[str, connectivity]:
    return self.__connectivities

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
  def attributesNames(self) -> List[str]:
    return list(self.__attributes.keys())

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

  def addConnectivity(self, name:str, to: primitive, data: Union[np.ndarray, List[List[int]]], dimension: int) -> connectivity:
    if name in self.__connectivities:
      raise ValueError(f"primitive.addConnectivity: connectivity with name '{name}' already exists in primitive.")

    # check name is valid
    if not self.isValidName(name):
      raise ValueError(f"primitive.addConnectivity: '{name}' is not a valid name for a connectivity.")

    # check if the connectivity is the same as the mesh
    if to.mesh != self.mesh:
      raise ValueError(f"primitive.addConnectivity: connectivity '{name}' must connect to the same mesh.")

    from yasps import connectivity
    newConnectivity = connectivity(name, self, to, data, dimension)
    self.__connectivities[name] = newConnectivity
    setattr(self, name, newConnectivity)
    return newConnectivity

  def updateNumInstances(self, numInstances: int) -> None:
    if not self.isDynamic:
      raise ValueError("primitive.updateNumInstances: primitive is not dynamic, cannot update number of instances.")
    if numInstances < 0:
      raise ValueError("primitive.updateNumInstances: number of instances cannot be negative.")
    self.__numInstances = numInstances

  def updateConnectivity(self, name: str, data: Union[np.ndarray, List[List[int]]], dimension: int) -> None:
    ## first check name
    if name not in self.__connectivities:
      raise ValueError(f"primitive.updateConnectivity: connectivity with name '{name}' does not exist in primitive.")
    if dimension != self.__connectivities[name].dimension:
      raise ValueError(f"primitive.updateConnectivity: connectivity with name '{name}' has dimension {self.__connectivities[name].dimension}, but the supplied data has dimension {dimension}.")
    self.__connectivities[name].updateConnectivity(data)


  def addConstant(self, name: str, rows: int = 1, cols: int = 1):
    if name in self.__attributes:
      raise ValueError(f"primitive.addConstant: attribute with name '{name}' already exists in the primitive.")
    from yasps.attribute import attribute
    newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols, is_constant = True)
    self.__attributes[name] = newAttribute
    return newAttribute

  def addAttribute(self, name: str, computed_attribute: Optional[attribute] = None, rows: int = 1, cols: int = 1, through: Optional[connectivity] = None, source: Optional[attribute] = None, operation: Optional[str] = None) -> attribute:
    from yasps.attribute import attribute
    from yasps.attribute import JOIN, SUM, AVERAGE
    if name in self.__attributes:
      raise ValueError(f"primitive.addAttribute: attribute with name '{name}' already exists in primitive.")

    if computed_attribute is not None:
      # if computed_attribute.name != "":
        # raise ValueError(f"primitive.addAttribute: the computed_attribute supplied already has a name '{computed_attribute.name}'. This indicates that the computed_attribute most likely is already set for another object. The supplied name is: '{name}'.")
      self.__attributes[name] = computed_attribute
      if computed_attribute.name == "":
        computed_attribute.setName(name)
      if computed_attribute.correspondance is None:
        # this is very likely a constant matrix
        # we need to set the correspondance to self
        computed_attribute._attribute__correspondance = self
        # print('Case 1')
        return computed_attribute
      if computed_attribute.correspondance != self:
        import copy
        copied_attribute = copy.deepcopy(computed_attribute)
        copied_attribute._attribute__correspondance = self
        self.__attributes[name] = copied_attribute
        # print('Case 2')
        return copied_attribute
      # print('Case 3')
      return computed_attribute
    elif through is not None:
      # we now check if this is a joining operation or scattering operation
      if through.fromPrimitive == self:
        # we will join the attribute from another primitive
        # we will check if the name is inside the to primitive
        toPrimitive = through.toPrimitive
        if source is not None:
          # first we check if the source attribute is in the fromPrimitive
          if source.correspondance != toPrimitive:
            # ok here are some cases
            # if the source is actually a mesh or scene attribute
            # and the primitive itself is a descendant
            # then we can add the attribute
            if source.correspondance.fullName == self.mesh.fullName or source.correspondance.fullName == self.scene.fullName:
              self.__attributes[name] = source
              if source.name == "":
                source.setName(name)
            else:
              raise ValueError(f"primitive.addAttribute: the primitive {self.fullName} has no connection to the attribute, whose correspondance is {source.correspondance.fullName}.")
          if through.dimension == 0 and operation is None:
            raise ValueError("primitive.addAttribute: an operation must be specified when the connectivity is not fixed. Available operations are: SUM and AVERAGE.")
          op = JOIN
          # for JOIN let's do special handling
          # we will try to only join the non-zero elements, then we will assemble the matrix
          source_element_is_nonzero = [1 for i in range(source.rows * source.cols)]
          for i in range(source.rows * source.cols):
            if source[i].isZero != 0: # this means it is a zero element:
              source_element_is_nonzero[i] = 0
          # we will only join the non-zero elements, so first we need to get the non-zero elements to another attribute
          if sum(source_element_is_nonzero) == len(source_element_is_nonzero) or sum(source_element_is_nonzero) == 0:
            # there's nothing we need to do, directly do the join
            pass
          else:
            # we create a new attribute on the source, then do the join again.
            non_zero_joined_children = []
            for i in range(source.rows * source.cols):
              if source_element_is_nonzero[i] == 1:
                non_zero_joined_children.append(source[i])
            nonzero_child_attribute: attribute
            nonzero_joined_attribute: attribute
            if f"{name}_non_zeros" in source.correspondance.attributes:
              nonzero_child_attribute = source.correspondance.attributes[f"{name}_non_zeros"]
            else:
              # print(f"Creating non-zero child attribute for source attribute {source.name} with name {name}_non_zeros, which has {sum(source_element_is_nonzero)} non-zero elements out of {len(source_element_is_nonzero)} total elements.")
              nonzero_child_attribute = source.correspondance.addAttribute(name = f"{name}_non_zeros", computed_attribute = attribute.to_array(non_zero_joined_children, rows = 1, cols = sum(source_element_is_nonzero)))
            if f"{name}_non_zeros" in self.attributes:
              nonzero_joined_attribute = self.attributes[f"{name}_non_zeros"]
            else:
              nonzero_joined_attribute = attribute(name = f"{name}_non_zeros", correspondance = self, rows = through.dimension, cols = sum(source_element_is_nonzero), through = through, children = [nonzero_child_attribute], operator = JOIN)
              self.__attributes[f"{name}_non_zeros"] = nonzero_joined_attribute
            # now we create the joined attribute as an array
            new_joined_attribute_children = []
            for i in range(through.dimension):
              non_zero_count = 0
              for j in range(source.rows * source.cols):
                if source_element_is_nonzero[j] == 1:
                  new_joined_attribute_children.append(nonzero_joined_attribute[i, non_zero_count])
                  non_zero_count += 1
                else:
                  new_joined_attribute_children.append(0.0)
            # create the new joined attribute
            newRows: int = through.dimension
            newCols: int = source.rows * source.cols
            newAttribute = self.addAttribute(name = name, computed_attribute = attribute.to_array(new_joined_attribute_children, rows = newRows, cols = newCols))
            return newAttribute


          # if the matrix is fully dense already, we just make the join attribute regardless.
          newRows: int = through.dimension
          newCols: int = source.rows * source.cols
          newAttribute = attribute(name = name, correspondance = self, rows = newRows, cols = newCols, through = through, children = [source], operator = op)
          # print(f"The attribute {name} for primitive {self.name} now has dimension {newRows}x{newCols}")
          self.__attributes[name] = newAttribute
          return newAttribute
        # the source is not set up
        # we automatically try to retrieve the attribute with the same name
        if name not in toPrimitive.attributes:
          raise ValueError(f"primitive.addAttribute: attribute with name '{name}' does not exist in primitive '{toPrimitive.name}'. The through construction is not successful for the joining operation.")
        if through.dimension == 0 and operation is None:
          raise ValueError("primitive.addAttribute: an operation must be specified when the connectivity is not fixed. Available operations are: SUM and AVERAGE.")
        op = JOIN
        newRows: int = through.dimension
        newCols: int = toPrimitive[name].rows * toPrimitive[name].cols
        if operation is not None:
          newRows = toPrimitive[name].rows
          newCols = toPrimitive[name].cols
          if operation == "SUM":
            op = SUM
          elif operation == "AVERAGE":
            op = AVERAGE
          else:
            raise ValueError(f"primitive.addAttribute: the operation '{operation}' is not valid. Available operations are: SUM and AVERAGE.")
        newAttribute = attribute(name = name, correspondance = self, rows = newRows, cols = newCols, through = through, children = [toPrimitive[name]], operator = op)
        # print(f"The attribute {name} for primitive {self.name} now has dimension {newRows}x{newCols}")
        self.__attributes[name] = newAttribute
        return newAttribute
      else:
        raise ValueError("primitive.addAttribute: the through construction must have the fromPrimitive as the primitive where the attribute is being added.")

    else:
      newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols)
      self.__attributes[name] = newAttribute
      return newAttribute

  @property
  def fullName(self)->str:
    return f"{self.scene.name}_{self.mesh.name}_{self.name}"

  # accessing attribute by [] operator
  # user can get multiple attributes by passing a list of names
  def __getitem__(self, key: str) -> attribute:
    if isinstance(key, str):
      if key in self.__attributes:
        return self.__attributes[key]
      else:
        raise KeyError(f"primitive.__getitem__: attribute with name '{key}' not found in primitive.")
    else:
      raise KeyError(f"primitive.__getitem__: attribute with name '{key}' not found in primitive.")

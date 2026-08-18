# cython: language_level=3
from __future__ import annotations
from typing import Dict, Union, List, Set, Optional
# import keyword
# import numpy as np
# a primitive may have its own attributes
# and connectivities to other primitives
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from yasps.mesh import mesh
  from yasps.attribute import attribute
  from yasps.scene import scene as yscene
  from yasps.primitive import primitive

import pycuda.gpuarray as gpuarray
import numpy as np

## the goal of primitive Union
## is to provide a way to access the two primitives stacked together
## and rename it to a new primitive
## think of an example where we have primitive A and B, A is vertices of a rigid body
## and B is vertices of soft body
## we can add this primitive union to make it so that we can just
## call A and B as vertices


class primitiveUnion:
  def __init__(self, name: str, parent_mesh: mesh, primitives: List[Union[primitive, primitiveUnion]]):
    if name == "":
      raise ValueError("mesh.__init__: name cannot be empty.")
    if parent_mesh is None:
      raise ValueError("mesh.__init__: mesh cannot be None.")
    self.__name: str = name
    self.__mesh: mesh = parent_mesh
    ## the primitives are now stacked in the order
    self.__primitives: List[Union[primitive, primitiveUnion]] = primitives
    self.__attributes: Dict[str, attribute] = {}
    self.__children_primitive_counts = None
    self.__children_primitive_counts_gpu = gpuarray.empty(0, dtype=np.uint32)

  # the name of the primitive union
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
  def primitiveUnion(self)->primitiveUnion:
    return self

  @property
  def type(self)->str:
    return "primitiveUnion"

  @property
  def numInstances(self)->int:
    ## the number of instances is the sum of all the instances of the primitives
    return sum([x.numInstances for x in self.__primitives])

  @property
  def attributes(self)->Dict[str, List[attribute]]:
    return self.__attributes

  @property
  def attributesNames(self) -> List[str]:
    return list(self.__attributes.keys())

  def __getitem__(self, key: str) -> attribute:
    if isinstance(key, str):
      if key in self.__attributes:
        return self.__attributes[key]
      else:
        raise KeyError(f"primitive.__getitem__: attribute with name '{key}' not found in primitive.")
    else:
      raise KeyError(f"primitive.__getitem__: attribute with name '{key}' not found in primitive.")

  @property
  def fullName(self)->str:
    return f"{self.scene.name}_{self.mesh.name}_{self.name}"

  @property
  def code_generation_counts_name(self)->str:
    return f"{self.fullName}_children_primitive_counts"

  @property
  def children_primitive_counts_gpu(self):
    counts = tuple(int(child.numInstances) for child in self.__primitives)
    if counts != self.__children_primitive_counts:
      counts_cpu = np.asarray(counts, dtype=np.uint32)
      if self.__children_primitive_counts_gpu.size == counts_cpu.size:
        self.__children_primitive_counts_gpu.set(counts_cpu)
      else:
        self.__children_primitive_counts_gpu = gpuarray.to_gpu(counts_cpu)
      self.__children_primitive_counts = counts
    return self.__children_primitive_counts_gpu

  def addConstant(self, name: str, rows: int = 1, cols: int = 1):
    if name in self.__attributes:
      raise ValueError(f"primitiveUnion.addConstant: attribute with name '{name}' already exists in the primitive union.")
    from yasps.attribute import attribute
    newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols, is_constant = True)
    self.__attributes[name] = newAttribute
    return newAttribute

  def addAttribute(self, name: str, computed_attribute: Optional[attribute] = None, rows = 0, cols = 0) -> attribute:
    # ok addint attribute for primitive union is different
    # we technically cannot add new attribute
    # instead we only query the children to see if that attribute exists
    # if there are any computed attribute, we need to check if they are of the same correspondance
    # unlike normal primitives, we do not allow new data to be initialized on primitiveUnion
    # neither do we allow join operator to be used on primitiveUnion
    if name in self.__attributes:
      raise ValueError(f"primitiveUnion.addAttribute: attribute with name '{name}' already exists in primitive.")
    if computed_attribute is None:
      if rows > 0 and cols > 0:
        from yasps.attribute import attribute
        newAttribute = attribute(name = name, correspondance = self, rows = rows, cols = cols)
        self.__attributes[name] = newAttribute
        return newAttribute
      # we check if the name exists for all of the primitives
      rows_size_check: Set[int] = set()
      cols_size_check: Set[int] = set()
      for child in self.__primitives:
        if name not in child.attributesNames:
          raise ValueError(f"primitiveUnion.addAttribute: attribute {name} does not exist in {child.fullName}")
        rows_size_check.add(child[name].rows)
        cols_size_check.add(child[name].cols)
      if len(rows_size_check) > 1:
        raise ValueError(f"primitiveUnion.addAttribute: attribute {name} has different rows size for the primitives in the union")
      if len(cols_size_check) > 1:
        raise ValueError(f"primitiveUnion.addAttribute: attribute {name} has different cols size for the primitives in the union")

      # now we are sure that the name exists in all attributes
      # we can create the new attribute
      from yasps.attribute import attribute
      from yasps.attribute import UNION
      # we only want to union the non-zero attributes
      # so we need to do nonzero check on all the children now
      all_children_attributes = [x[name] for x in self.__primitives]
      sample_attribute = all_children_attributes[0]
      source_element_is_nonzero = [0 for _ in range(sample_attribute.rows * sample_attribute.cols)]
      for i in range(sample_attribute.rows * sample_attribute.cols):
        for child_attribute in all_children_attributes:
          if child_attribute[i].isZero == 0: # this means it's a non-zero element
            source_element_is_nonzero[i] = 1
            break

      # ok now we need to check if it is fully nonzero
      if sum(source_element_is_nonzero) == len(source_element_is_nonzero) or sum(source_element_is_nonzero) == 0:
        # this means that all the elements are non-zero, we can just do a simple union
        pass
      else:
        nonzero_attribute_name = f"{name}_nonzero_for_union_with_{'__'.join([x.fullName for x in self.__primitives])}"
        # we can start creating a non-zero attribute for each children primitive
        for child in self.__primitives:
          child_attribute = child[name]
          nonzero_elements = []
          for i in range(child_attribute.rows * child_attribute.cols):
            if source_element_is_nonzero[i] == 1:
              nonzero_elements.append(child_attribute[i])
          nonzero_attribute = attribute.to_array(nonzero_elements, rows = 1, cols = sum(source_element_is_nonzero))
          child.addAttribute(nonzero_attribute_name, computed_attribute = nonzero_attribute)
        # now create the UNIONed attribut which only contains nonzero
        unioned_nonzero_attribute = attribute(name = nonzero_attribute_name, rows = 1, cols = sum(source_element_is_nonzero), correspondance = self, children = [x[nonzero_attribute_name] for x in self.__primitives], operator = UNION)
        # then we create the new array which contains the zero elements
        unioned_attribute_list = []
        nonzero_count = 0
        for i in range(len(source_element_is_nonzero)):
          if source_element_is_nonzero[i] == 1:
            unioned_attribute_list.append(unioned_nonzero_attribute[nonzero_count])
            nonzero_count += 1
          else:
            unioned_attribute_list.append(0.0)
        unioned_attribute = self.addAttribute(name = name, computed_attribute = attribute.to_array(unioned_attribute_list, rows = sample_attribute.rows, cols = sample_attribute.cols))
        return unioned_attribute

      # if the attribute is fully dense
      # we just do normal union
      unioned_attribute = attribute(name, rows = rows_size_check.pop(), cols = cols_size_check.pop(), correspondance = self, children = [x[name] for x in self.__primitives], operator = UNION)
      self.__attributes[name] = unioned_attribute
      return unioned_attribute
    else:
      # we are adding a computed attribute
      # we first check if the correspondance is self
      if computed_attribute.correspondance is None:
        # this is very likely a constant matrix
        # let's hack a bit to directly modify the correspondance
        computed_attribute._attribute__correspondance = self
        self.__attributes[name] = computed_attribute
        return computed_attribute
      if computed_attribute.correspondance.fullName != self.fullName:
        raise ValueError(f"primitiveUnion.addAttribute: computed attribute {name} correspondance does not match the primitive union")
      if computed_attribute.name != "":
        self.__attributes[name] = computed_attribute
        # raise ValueError(f"primitiveUnion.addAttribute: computed attribute {name} name must be empty")
        return computed_attribute
      else:
        computed_attribute.setName(name)
        self.__attributes[name] = computed_attribute
        return computed_attribute

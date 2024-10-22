# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Union, Tuple, Set, Dict
from typing import TYPE_CHECKING
from yasps.attribute import attribute
from yasps.autodiff import autodiff
if TYPE_CHECKING:
  from yasps.operator import operator
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.primitive import primitive
  from yasps.connectivity import connectivity
  from yasps.deviceKernel import deviceKernel
  from yasps.globalKernel import globalKernel
  from yasps.codeGenerator import codeGenerator

class energy:
  def __init__(self, energy: attribute):
    if energy.size != 1:
      raise ValueError("energy.__init__: energy must be size 1.")
    self.__energy: attribute = energy
    self.__paths: List[List[attribute]] = [] # how to get to the roots
    self.__roots: List[attribute] = []
    self.__roots, self.__paths = self.getRoots(energy, [energy]) # get the root attributes


  @property
  def roots(self) -> List[attribute]:
    return self.__roots


  def getRoots(self, att: attribute, parentPath: List[attribute]) -> Tuple[List[attribute], List[List[attribute]]]:
    from yasps.attribute import GATHER, SUM, AVERAGE, DATA
    stack: List[attribute] = [att]
    seenRoots: Set[attribute] = set([])
    roots: List[attribute] = []
    while stack:
      current = stack.pop()
      if current.operator == DATA:
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      elif current.operator == GATHER or current.operator == SUM or current.operator == AVERAGE:
        if current.through.dimension == 0:
          raise ValueError("energy.getRoots: att.through.dimension is 0. Such operation is not supported in energy minimization.")
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      else:
        stack.extend(current.children)
    # now we have the roots, which contains some gathering operation
    # we need to duplicate the roots for those gathering operation
    trueRoots: List[attribute] = []
    allPaths: List[List[attribute]] = []
    for root in roots:
      if root.operator != DATA:
        childrenRoots, childrenPaths = self.getRoots(root.children[0], [])
        trueRoots += childrenRoots * root.through.dimension
        for i in range(root.through.dimension):
          for childrenPath in childrenPaths:
            allPaths.append(parentPath + [root.row(i)] + childrenPath)
        # allPaths += childrenPaths * root.through.dimension
      else:
        trueRoots.append(root)
        allPaths.append(parentPath + [root])
    return trueRoots, allPaths

  def getHessianOffDiagonalBlockSizes(self, wrt: List[attribute]) -> List[Tuple[int, int]]:
    sizePairs = []
    for i in range(len(self.roots)):
      if self.roots[i] in wrt:
        for j in range(i+1, len(self.roots)):
          if self.roots[j] in wrt:
            sizePairs.append((self.roots[i].size, self.roots[j].size))
    sizePairs = list(set(sizePairs))
    return sizePairs

  def getSparseIndices(self, wrt: List[attribute], wrt_start_indices: List[int]):
    from yasps.attribute import ROW, DATA
    # we have the path, now determine which path to use since we have the wrt
    usedPaths: List[List[attribute]] = []
    for i in range(len(self.roots)):
      if self.roots[i] in wrt:
        usedPaths.append(self.__paths[i])


    indicesCPU: Dict[int, np.ndarray] = {} # the indices to cpu
    # now we get the indices of the paths by recursively go over the indices, first we transfer the indices to CPU
    for path in usedPaths:
      for att in path:
        if att.operator == ROW:
          if att.children[0].hash not in indicesCPU:
            indicesCPU[att.children[0].hash] = att.children[0].through.value.get().reshape(att.children[0].through.fromPrimitive.numInstances, att.children[0].through.dimension) # reshape to a 2D array with num instances, and dimension
    # now we recursively go over each path
    currentIndex = 0
    allIndices: List[np.uint32] = []
    for i in range(self.__energy.correspondance.numInstances):
      for path in usedPaths:
        currentIndex = i
        for node in path:
          if node.operator == ROW:
            # get the new index

            rowIndex = node.children[1].index_value
            currentIndex = indicesCPU[node.children[0].hash][currentIndex, rowIndex]
          elif node.operator == DATA:
            # because it is a data
            # and we have a starting position for the data
            # we will need to aggregate the starting index
            dataIndexInWrt = wrt.index(node)
            if node.correspondance.type == "primitive":
              currentIndex = wrt_start_indices[dataIndexInWrt] + currentIndex * node.size
              allIndices.append(currentIndex)
            else:
              currentIndex = wrt_start_indices[dataIndexInWrt]
              allIndices.append(np.uint32(currentIndex))
    print("All indices are: ", allIndices)
    return allIndices

  def generateHessianAndGradient(self, wrt: List[attribute]) -> None:
    differentiater = autodiff()
    # generate the symbolic code for gradient and hessian
    # first we check which path we need
    filteredPath: List[List[attribute]] = []
    for path in self.__paths:
      if path[-1] in wrt:
        filteredPath.append(path)

    # now we generate from the bottom up
    for path in filteredPath:
      # do it in reverse order
      for i in range(len(path) - 2, -1, -1):
        # we will compute the jacobian for each neighboring nodes
        differentiater.diff(path[i], path[i+1])

  # def __diff_gather(self, current: ya.attribute, wrt: ya.attribute) -> ya.attribute:
  #   # differentiating through a gathering is a bit different
  #   # because we will need to generate the jacobian for two things
  #   # first the children to wrt
  #   child = current.children[0]
  #   child_jacobian = ya.attribute.zeros(child.size, wrt.size)
  #   child_jacobian_name = f"d{current.fullName}_d{child.fullName}"
  #   if child_jacobian_name not in child.correspondance.attributes:
  #     child_jacobian = self.__diff(child, wrt)
  #     # once we get the child jacobian, we first add it as an attribute in case we need it later
  #     child.correspondance.addAttribute(child_jacobian_name, computed_attribute = child_jacobian)
  #   else:
  #     child_jacobian = child.correspondance.attributes[child_jacobian_name]
  #   # now we need to determine, in the jacobian, are there elements that aren't tied to a primitive
  #   skipped_indices = []

  #   for i in range(child_jacobian.size):
  #     if child_jacobian[i].correspondance is None or (child_jacobian[i].correspondance.type != "primitive") or child_jacobian[i].operator == ya.FLOAT:
  #       skipped_indices.append(i)
  #   # ok now we need to create new attributes for the non skipped indices
  #   for i in range(child_jacobian.size):
  #     if i in skipped_indices:
  #       continue
  #     # # we need to create a new attribute for the ith element
  #     # child.correspondance.addAttribute(f"{child_jacobian_name}_{i}", computed_attribute = child_jacobian.children[i])
  #     current.correspondance.addAttribute(f"{child_jacobian_name}_{i}", through = current.through, source = child.correspondance[child_jacobian_name][i])
  #   # now we need to assemble the jacobian matrix
  #   result = [None] * current.size * wrt.size * current.through.dimension
  #   # this will be a blocked matrix, where blocks are always on diagonal
  #   for i in range(current.through.dimension): # work on the block
  #     block_jacobian = [ya.attribute(float_value = 0.0)] * child_jacobian.size
  #     for j in range(child_jacobian.size):
  #       if j in skipped_indices:
  #         # direcltly get it
  #         block_jacobian[j] = child_jacobian[j]
  #       else:
  #         block_jacobian[j] = current.correspondance[f"{child_jacobian_name}_{j}"][i]
  #     # now we put the block jacobian back
  #     for j in range(child_jacobian.rows):
  #       for k in range(child_jacobian.cols):
  #         ind = j * child_jacobian.cols + k
  #         result_ind = (i * child_jacobian.size * current.through.dimension) + j * (wrt.size * current.through.dimension) + i * wrt.size + k
  #         result[result_ind] = block_jacobian[ind]
  #   return ya.attribute.to_array(result, rows = current.size, cols = wrt.size * current.through.dimension)


  def __hash__(self) -> int:
    return self.__energy.hash

  @property
  def hash(self) -> int:
    return self.__hash__()

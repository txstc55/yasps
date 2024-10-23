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
        for childrenPath in childrenPaths:
          allPaths.append(parentPath + [root] + childrenPath)
        # allPaths += childrenPaths * root.through.dimension
      else:
        trueRoots.append(root)
        allPaths.append(parentPath + [root])
    # print("All paths: ")
    # for path in allPaths:
    #   print([p.fullName for p in path])
    # print("All roots: ")
    # for root in trueRoots:
    #   print(root.fullName)
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
    from yasps.attribute import ROW, DATA, GATHER
    # we have the path, now determine which path to use since we have the wrt
    usedPaths: List[List[attribute]] = []
    # for i in range(len(self.roots)):
    #   if self.roots[i] in wrt:
    #     usedPaths.append(self.__paths[i])
    for path in self.__paths:
      if path[-1] in wrt:
        usedPaths.append(path)


    indicesCPU: Dict[int, np.ndarray] = {} # the indices to cpu
    # now we get the indices of the paths by recursively go over the indices, first we transfer the indices to CPU
    for path in usedPaths:
      for att in path:
        if att.operator == GATHER:
          if att.hash not in indicesCPU:
            indicesCPU[att.hash] = att.through.value.get().reshape(att.through.fromPrimitive.numInstances, att.through.dimension) # reshape to a 2D array with num instances, and dimension
    # now we recursively go over each path
    # we first recursively duplicate the path with rows

    currentIndex = 0
    allIndices: List[np.uint32] = []

    print("Used paths: ")
    for path in usedPaths:
      print([p.fullName for p in path])
    duplicatedPaths = []
    for path in usedPaths:
      duplicatedPaths += self.__duplicatePath(path)
    print("Duplicated paths: ")
    for path in duplicatedPaths:
      print([p.fullName for p in path])
    for i in range(self.__energy.correspondance.numInstances):
      for path in duplicatedPaths:
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


  def __duplicatePath(self, path: List[attribute]) -> List[List[attribute]]:
    from yasps.attribute import DATA, GATHER
    if path[0].operator == DATA:
      # we've reached the end
      # return itself
      return [path]
    elif path[0].operator == GATHER:
      # we need to duplicate the path
      # first we need to get the children paths
      childrenPaths = self.__duplicatePath(path[1:])
      duplicatedPaths = []
      for i in range(path[0].through.dimension):
        # duplicate each path
        for childrenPath in childrenPaths:
          duplicatedPaths.append([path[0].row(i)] + childrenPath)
      return duplicatedPaths
    else:
      # we are at top level
      childrenPaths = self.__duplicatePath(path[1:])
      return [[path[0]] + childrenPath for childrenPath in childrenPaths]



  def generateHessianAndGradient(self, wrt: List[attribute]) -> None:
    from yasps.attribute import DATA, GATHER
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
        # # we will compute the jacobian for each neighboring nodes
        # # print("Differentiating: ", path[i].fullName, path[i+1].fullName)
        # result = differentiater.diff(path[i], path[i+1])
        # # print(result)
        lead_node = path[i]
        follow_node = path[i+1]
        if lead_node.operator == GATHER:
          # we need to differentiate a gather's children wrt to the next node
          child_att = lead_node.children[0]
          child_att_correspondance = child_att.correspondance
          child_att_full_name = child_att.fullName
          follow_node_full_name = follow_node.fullName
          diff_att_name = f'd{child_att_full_name}_d{follow_node_full_name}'
          child_att_jacobian: attribute
          if diff_att_name not in child_att_correspondance.attributes:
            child_att_jacobian = differentiater.diff(child_att, follow_node)
            child_att_correspondance.addAttribute(diff_att_name, computed_attribute = child_att_jacobian)
            print(f"Diff {child_att_full_name} wrt {follow_node_full_name} done")
            print("Jacobian is: ")
            print(child_att_jacobian)
          else:
            child_att_jacobian = child_att_correspondance.attributes[diff_att_name]
        # now that we have the child jacobian, we need to differentiate the gather wrt to the child
        else:
          # it's a normal node, which is the energy
          # add the differentiation wrt to the next node
          diff_att_name = f'd{lead_node.fullName}_d{follow_node.fullName}'
          if diff_att_name not in lead_node.correspondance.attributes:
            diff_att = differentiater.diff(lead_node, follow_node)
            lead_node.correspondance.addAttribute(diff_att_name, computed_attribute = diff_att)
            print(f"Diff {lead_node.fullName} wrt {follow_node.fullName} done")


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

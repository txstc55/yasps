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




  def __hash__(self) -> int:
    return self.__energy.hash

  @property
  def hash(self) -> int:
    return self.__hash__()

# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Union, Tuple, Set, Dict
from typing import TYPE_CHECKING
from yasps.attribute import attribute
from yasps.autodiff import autodiff
import pycuda.driver as cuda
if TYPE_CHECKING:
  from yasps.operator import operator
  from yasps.scene import scene
  from yasps.mesh import mesh
  from yasps.primitive import primitive
  from yasps.connectivity import connectivity
  from yasps.deviceKernel import deviceKernel
  from yasps.globalKernel import globalKernel
  from yasps.codeGenerator import codeGenerator
  from yasps.hessianAndGradientKernel import hessianAndGradientKernel

class energy:
  def __init__(self, energy: attribute):
    if energy.size != 1:
      raise ValueError("energy.__init__: energy must be size 1.")
    self.__energy: attribute = energy
    self.__paths: List[List[attribute]] = [] # how to get to the roots
    self.__roots: List[attribute] = []
    self.__roots, self.__paths = self.getRoots(energy, [energy]) # get the root attributes
    self.__wrt: List[int] = [] # an energy can be minimized for different attributes, for safety let's save all histories
    self.__indices: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # save the indices
    self.__gradient_sizes: List[int] = [] # save the sizes of the gradient
    self.__gradient_sizes_gpu: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # save the sizes of the gradient
    self.__hessian: Optional[attribute] = None # save the hessian for each wrt input
    self.__gradient: Optional[attribute] = None # save the gradient for each wrt input
    self.__hessianAndGradientKernel: Optional[hessianAndGradientKernel] = None
    # we are dealing with f(g(x))
    # the hessian of f(g(x)) wrt x is given by:
    # H(f(g(x))) = Jg^T H(f(g(x))) Jg + sum over k d_k(g(x)) * d2(g_k(x))
    self.__jg: Optional[attribute] = None
    self.__hf: Optional[attribute] = None
    self.__d2g: List[attribute] = []
    self.__d2g_start_indices: List[int] = []


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
    self.__wrt = wrt
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

    # print("Used paths: ")
    # for path in usedPaths:
    #   print([p.fullName for p in path])
    duplicatedPaths = []
    for path in usedPaths:
      duplicatedPaths += self.__duplicatePath(path)
    # print("Duplicated paths: ")
    # for path in duplicatedPaths:
    #   print([p.fullName for p in path])
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
    # print("All indices are: ", allIndices)
    self.__indices = gpuarray.to_gpu(np.array(allIndices, dtype = np.uint32))
    self.__gradient_sizes = [x[-1].size for x in duplicatedPaths]
    # print("Corresponding sizes are: ", [x[-1].size for x in duplicatedPaths])
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

  def __generateGradient(self, wrt: List[attribute], differentiater: autodiff) -> None:
    from yasps.attribute import DATA, GATHER, FLOAT
    # generate the symbolic code for gradient and hessian
    # first we check which path we need
    filteredPath: List[List[attribute]] = []
    for path in self.__paths:
      if path[-1] in wrt:
        filteredPath.append(path)
    gradients: List[attribute] = []
    if len(filteredPath) == 0:
      gradients.append(attribute.zeros(1, sum([x.size for x in wrt])))
      return

    if f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}' in self.__energy.correspondance.attributes:
      # nothing we need to do, the gradient is already computed
      self.__gradient = self.__energy.correspondance.attributes[f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}']
    else:
      # now we generate from the bottom up
      for path in filteredPath:
        # do it in reverse order
        for i in range(len(path) - 2, -1, -1):
          # # we will compute the jacobian for each neighboring nodes
          lead_node = path[i]
          follow_node = path[i+1]
          if lead_node.operator == GATHER:
            # we need to differentiate a gather's children wrt to the next node
            child_att = lead_node.children[0]
            child_att_correspondance = child_att.correspondance
            child_att_full_name = child_att.fullName
            follow_node_full_name = follow_node.fullName
            diff_att_name = f'd_{child_att_full_name}_d_{follow_node_full_name}'
            child_diff: attribute
            if diff_att_name not in child_att_correspondance.attributes:
              child_diff = differentiater.diff(child_att, follow_node)
              if diff_att_name not in child_att_correspondance.attributes:
                child_att_correspondance.addAttribute(diff_att_name, computed_attribute = child_diff)
            else:
              child_diff = child_att_correspondance.attributes[diff_att_name]
          # now that we have the child jacobian, we need to differentiate the gather wrt to the child
          else:
            # it's a normal node, which is the energy
            # add the differentiation wrt to the next node
            diff_att_name = f'd_{lead_node.fullName}_d_{follow_node.fullName}'
            if diff_att_name not in lead_node.correspondance.attributes:
              diff_att = differentiater.diff(lead_node, follow_node)
              if diff_att_name not in lead_node.correspondance.attributes:
                lead_node.correspondance.addAttribute(diff_att_name, computed_attribute = diff_att)
        # ok now we have done the differentiation from node to node
        # we need to assemble the actual jacobian matrix for the gather operator
        if len(path) == 2:
          # there is no gather operation
          # the jacobian should be directly used
          derivative = path[0].correspondance[f'd_{path[0].fullName}_d_{path[1].fullName}']
          gradients.append(derivative)
        else:
          # we go from bottom up
          # and for the first one, we will deal with it separately
          for i in range(len(path) - 2, 0, -1):
            lead_att = path[i]
            next_att = path[i+1]
            data_node = path[-1]
            neighboring_jacobian = next_att.correspondance[f'd_{lead_att.children[0].fullName}_d_{next_att.fullName}']
            multiplied_jacobian: attribute
            if i == len(path) - 2:
              # just return the neighboring jacobian as the multiplied
              multiplied_jacobian = neighboring_jacobian
            else:
              # now we get the jacobian from the next att to the data node
              gather_path = path[i + 1: -1]
              gather_path_str = "_d_".join([x.fullName for x in gather_path])
              next_att_data_jacobian = next_att.correspondance[f'd_{gather_path_str}_d_{data_node.fullName}']
              multiplied_jacobian = neighboring_jacobian.mul_explicit(next_att_data_jacobian)
            # now we determine if it is needed to actually gather the entire thing by checking if everything has a correspondance and not a float
            skipped_indices: List[int] = []
            # TODO: there are elements that are repeated
            # we can reduce the amout of gather by taking out those elements
            # and only gather once
            for j in range(multiplied_jacobian.size):
              if multiplied_jacobian[j].correspondance is None or (multiplied_jacobian[j].correspondance.type != "primitive") or multiplied_jacobian[j].operator == FLOAT:
                skipped_indices.append(j)

            # ok now we need to create new attributes for the non skipped indices
            included_paths = [lead_att.children[0]] + path[i + 1: -1]
            included_path_str = "_d_".join([x.fullName for x in included_paths])
            # print("Lead attribute fullname", lead_att.fullName)
            # print("Included path str:", included_path_str)
            for j in range(multiplied_jacobian.size):
              if j in skipped_indices:
                continue
              # we need to create a new attribute for the ith element through gather
              if f"d_{included_path_str}_d_{data_node.fullName}_{j}" not in lead_att.correspondance.attributes:
                lead_att.correspondance.addAttribute(f"d_{included_path_str}_d_{data_node.fullName}_{j}", through = lead_att.through, source = multiplied_jacobian[j]) # we add the new gathering attribute and use it later on
            # now we have a new gather attribute which is the jacobian
            new_jacobian_children = [attribute(float_value = 0.0) for _ in range(multiplied_jacobian.size * lead_att.through.dimension * lead_att.through.dimension)]
            # ok, if the child jacobian is m by n, then the new jacobian has k by k blocks, each block is m by n
            # and only the diagonal blocks will have nonzero values
            # the jacobian is corresponding to per data
            # so there's no need to reorient them
            for j in range(multiplied_jacobian.size):
              m = multiplied_jacobian.rows # rows
              n = multiplied_jacobian.cols # cols
              k = lead_att.through.dimension # how many times we need to repeat the jacobian through gather
              child_jacobian_row = j // n # which row of the child jacobian
              child_jacobian_col = j % n # which col of the child jacobian
              for l in range(lead_att.through.dimension):
                leading_index = m * n * k * l # offset because we are doing jacobian by jacobian
                element_index = leading_index + child_jacobian_row * (n * k) + (n * l + child_jacobian_col)
                if j in skipped_indices:
                  new_jacobian_children[element_index] = multiplied_jacobian[j]
                else:
                  new_jacobian_children[element_index] = lead_att.correspondance[f"d_{included_path_str}_d_{data_node.fullName}_{j}"][l]
            new_jacobian = attribute.to_array(new_jacobian_children, rows = multiplied_jacobian.rows * lead_att.through.dimension, cols = multiplied_jacobian.cols * lead_att.through.dimension)
            # add the jacobian to the correspondance
            included_paths = path[i: -1]
            included_path_str = "_d_".join([x.fullName for x in included_paths])
            if f"d_{included_path_str}_d_{data_node.fullName}" not in lead_att.correspondance.attributes:
              lead_att.correspondance.addAttribute(f"d_{included_path_str}_d_{data_node.fullName}", computed_attribute = new_jacobian)

          # now we deal with the last pair
          children_path = path[1: -1]
          children_path_str = "_d_".join([x.fullName for x in children_path])
          neighboring_jacobian = path[0].correspondance[f'd_{path[0].fullName}_d_{path[1].fullName}']
          last_data_jacobian = path[1].correspondance[f'd_{children_path_str}_d_{path[-1].fullName}']
          final_jacobian = neighboring_jacobian.mul_explicit(last_data_jacobian)
          gradients.append(final_jacobian)
      gradients_assembled_children = []
      for gradient in gradients:
        for i in range(gradient.size):
          gradients_assembled_children.append(gradient[i])
      if f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}' not in self.__energy.correspondance.attributes:
        g = self.__energy.correspondance.addAttribute(f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}', computed_attribute = attribute.to_array(gradients_assembled_children, rows = self.__energy.size, cols = len(gradients_assembled_children)))
      self.__gradient = self.__energy.correspondance[f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}']

  def __generateHessianForParts(self, differentiater: autodiff, currentPaths: List[List[attribute]]) -> None:
    from yasps.attribute import DATA, GATHER, FLOAT
    # we need to recursively generate the hessian for parts
    # the input currentPaths all have the same first element
    # then we just need to find the hessian for this part of the path
    # here's something we need to know
    # if i have the currentpath, the next node is guaranteed to present in batches
    # which means for the next node, it's guranteed to be like [a, a, b, b, b, c]
    # same node will always be batched together

    # we first check if H of f(g) is already computed
    lead_node = currentPaths[0][0]
    child_att = lead_node.children[0]
    child_att_full_name = child_att.fullName
    # we need to get the hessian from child_att to data node
    # first, we construct the Hessian of f(g(x))
    # which is essentially the hessian from child_att to all of its children
    h_f_g_size = 0
    allChildren: List[attribute] = []
    for path in currentPaths:
      follow_node = path[1]
      if follow_node not in allChildren:
        allChildren.append(follow_node)
    for follow_node in allChildren:
      follow_node_full_name = follow_node.fullName
      # we know this is definitely computed
      diff_att_name = f'd_{child_att_full_name}_d_{follow_node_full_name}'
      diff_att = child_att.correspondance.attributes[diff_att_name]
      h_f_g_size += diff_att.size / child_att.size # because hessian is generated for each element of the child att
    h_f_g_children = [attribute(float_value = 0.0) for _ in range(h_f_g_size * h_f_g_size * child_att.size)] # a hessian for each element of the child att
    row_offset = 0
    for i in range(len(allChildren)):
      follow_node = allChildren[i]
      follow_node_full_name = follow_node.fullName
      diff_att_name = f'd_{child_att_full_name}_d_{follow_node_full_name}'
      col_offset = 0 # whenever we go to a new child, we need to reset the col_offset
      for j in range(i, len(allChildren)):
        diff_target_node = allChildren[j]
        diff_target_node_full_name = diff_target_node.fullName
        d2_name = f'd_{diff_att_name}_d_{diff_target_node_full_name}'
        # we have computed this hessian for sure
        d2_attribute = child_att.correspondance.attributes[d2_name]
        single_att_d2_size = d2_attribute / child_att.size
        for l in range(child_att.size):
          d2_attribute_partial = d2_attribute[l * single_att_d2_size: (l + 1) * single_att_d2_size]
          # the block has size follow_node.size * diff_target_node.size
          # we need to put this block into the hessian
          # the offset are row_offset and col_offset
          for m in range(follow_node.size):
            for n in range(diff_target_node.size):
              h_f_g_children[l * h_f_g_size * h_f_g_size + (row_offset + m) * h_f_g_size + col_offset + n] = d2_attribute_partial[m * diff_target_node.size + n]
              # make it symmetric
              h_f_g_children[l * h_f_g_size * h_f_g_size + (col_offset + n) * h_f_g_size + row_offset + m] = d2_attribute_partial[m * diff_target_node.size + n]
        col_offset += diff_target_node.size
      row_offset += follow_node.size
    # ok now we need to assemble J of g(x)
    # we have previously computed them supposedly
    j_g_x_col_size = 0
    for path in currentPaths:
      next_node = path[1]
      if next_node.operator == DATA:
        j_g_x_col_size += next_node.size
      else:
        data_node = path[-1]
        included_paths = path[1: -1]
        included_path_str = "_d_".join([x.fullName for x in included_paths])
        next_jacobian_name = f"d_{included_path_str}_d_{data_node.fullName}"
        next_jacobian = next_node.correspondance.attributes[next_jacobian_name]
        j_g_x_col_size += next_jacobian.cols
    j_g_x_children = [attribute(float_value = 0.0) for _ in range(h_f_g_size * j_g_x_col_size)] # allocate space for the jacobian
    passed_next_node = [currentPaths[0][1]] # check if we already passed this node
    row_offset = 0
    col_offset = 0
    for path in currentPaths:
      next_node = path[1]
      if next_node not in passed_next_node:
        passed_next_node.append(next_node)
        row_offset += passed_next_node[-1].size # we moved to the next node, row_offset updates
      if next_node.operator == DATA:
        # this is easy
        # we put the identity matrix inside
        for i in range(next_node.size):
          index = (row_offset + i) * j_g_x_col_size + col_offset + i
          j_g_x_children[index] = attribute(float_value = 1.0)
        col_offset += next_node.size
      else:
        # it's not easy
        # we need to get the jacobian now
        included_paths = path[2: -1]
        data_node = path[-1]
        included_path_str = "_d_".join([x.fullName for x in included_paths])
        next_jacobian_name = f"d_{included_path_str}_d_{data_node.fullName}"
        next_jacobian = next_node.correspondance.attributes[next_jacobian_name]
        for i in range(next_jacobian.rows):
          for j in range(next_jacobian.cols):
            index = (row_offset + i) * j_g_x_col_size + col_offset + j
            j_g_x_children[index] = next_jacobian[i * next_jacobian.cols + j]
        col_offset += next_jacobian.cols
    # now we can compute the first part of the hessian by doing the multiplication
    j_g_x = attribute.to_array(j_g_x_children, rows = h_f_g_size, cols = j_g_x_col_size)
    multiplication_result = [] # the multiplied out first term, we do this for each element of child_att
    for i in range(child_att.size):
      hessian_children = h_f_g_children[i * h_f_g_size * h_f_g_size: (i + 1) * h_f_g_size * h_f_g_size]
      hessian = attribute.to_array(hessian_children, rows = h_f_g_size, cols = h_f_g_size)
      multiplication_result.append(j_g_x.transpose().mul_explicit(hessian).mul_explicit(j_g_x))

    # we've finished the first part of the hessian now
    # now we need to compute the second part









  def __generateHessian(self, wrt: List[attribute], differentiater: autodiff) -> None:
    from yasps.attribute import DATA, GATHER, FLOAT
    # first we check which path we need
    filteredPath: List[List[attribute]] = []
    for path in self.__paths:
      if path[-1] in wrt:
        filteredPath.append(path)
    # check if hessian is already generated
    if f'd2_{self.__energy.fullName}_d2_{"__".join([x.fullName for x in wrt])}' in self.__energy.correspondance.attributes:
      self.__hessian = self.__energy.correspondance.attributes[f'd2_{self.__energy.fullName}_d2_{"__".join([x.fullName for x in wrt])}']
    else:
      # now we need to compute the hessian
      # the first step is mapping out the attribute to descendant from the paths
      descendant_lookup: Dict[int, List[attribute]] = {}
      for path in filteredPath:
        for i in range(len(path) - 1):
          parent = path[i]
          child = path[i + 1]
          if parent.hash not in descendant_lookup:
            descendant_lookup[parent.hash] = []
          if child not in descendant_lookup[parent.hash]:
            descendant_lookup[parent.hash].append(child)
      # once we get the descendant lookup, we can start computing the hessian
      # by iterating over the path
      for path in filteredPath:
        # do it in reverse order
        for i in range(len(path) - 2, -1, -1):
          # # we will compute the jacobian for each neighboring nodes
          lead_node = path[i]
          follow_node = path[i+1]
          lead_node_hash = lead_node.hash
          lead_node_descendants = descendant_lookup[lead_node_hash]
          if lead_node.operator == GATHER:
            child_att = lead_node.children[0]
            child_att_correspondance = child_att.correspondance
            child_att_full_name = child_att.fullName
            follow_node_full_name = follow_node.fullName
            # we know this is definitely computed
            diff_att_name = f'd_{child_att_full_name}_d_{follow_node_full_name}'
            diff_att = child_att_correspondance.attributes[diff_att_name]
            # now we differentiate wrt the other descendants
            for descendant in lead_node_descendants:
              # perform the second derivative
              double_diff_att_name = f'd_{diff_att_name}_d_{descendant.fullName}'
              if double_diff_att_name not in child_att_correspondance.attributes:
                # we differentiate the node wrt to the descendant
                double_diff_att = differentiater.diff(diff_att, descendant)
                child_att_correspondance.addAttribute(double_diff_att_name, computed_attribute = double_diff_att)
          else:
            # it's a normal node, which is the energy
            # add the differentiation wrt to the next node
            diff_att_name = f'd_{lead_node.fullName}_d_{follow_node.fullName}'
            diff_att = lead_node.correspondance.attributes[diff_att_name]
            for descendant in lead_node_descendants:
              double_diff_att_name = f'd_{diff_att_name}_d_{descendant.fullName}'
              if double_diff_att_name not in lead_node.correspondance.attributes:
                double_diff_att = differentiater.diff(diff_att, descendant)
                lead_node.correspondance.addAttribute(double_diff_att_name, computed_attribute = double_diff_att)


  def generateHessianAndGradient(self, wrt: List[attribute]) -> None:
    from yasps.attribute import DATA, GATHER, FLOAT
    differentiater = autodiff()
    # generate the symbolic code for gradient and hessian
    self.__generateGradient(wrt, differentiater)





  def computeHessianAndGradient(self, gradient_array: gpuarray.GPUArray):
    start_compute = cuda.Event()
    end_compute = cuda.Event()
    start_compute.record()
    if self.__gradient is None:
      # the gradient is 0, return the 0 array
      return
    if self.__hessianAndGradientKernel is None:
      from yasps.codeGenerator import codeGenerator
      from yasps.hessianAndGradientKernel import hessianAndGradientKernel
      codegen: codeGenerator = codeGenerator(self.__gradient)
      codegen.generateCode()
      # now add the global kernel
      self.__hessianAndGradientKernel = hessianAndGradientKernel(self.__gradient, self.__gradient_sizes)
      self.__gradient_sizes_gpu = gpuarray.to_gpu(np.array(self.__gradient_sizes, dtype = np.uint32))
    assert self.__hessianAndGradientKernel is not None
    # after we allocated, we invoke the kernel
    arguments: List[gpuarray.GPUArray] = [x.value for x in self.__gradient.deviceKernel.kernelDatas] + [x.value for x in self.__gradient.deviceKernel.kernelConnectivity]+ [x.compressedRows for x in self.__gradient.deviceKernel.kernelConnectivity if x.dimension == 0] + [self.__indices] + [self.__gradient_sizes_gpu] + [gradient_array]


    # finally call the kernel
    # time the execution
    start_call = cuda.Event()
    end_call = cuda.Event()
    start_call.record()
    self.__hessianAndGradientKernel.kernel(*arguments, np.uint32(self.__gradient.correspondance.numInstances), block=(32, 1, 1), grid=((self.__gradient.correspondance.numInstances + 32) // 32, 1, 1))
    # Record the end event
    end_call.record()
    # Wait for the end event to complete
    end_call.synchronize()
    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = start_call.time_till(end_call)
    end_compute.record()
    end_compute.synchronize()
    # print(f"Kernel execution time: {elapsed_time_ms:.5f} ms")
    # print(f"Total time: {start_compute.time_till(end_compute):.5f} ms")
    # print(f"Gradient is: {gradient_array.get()}")
    return self

  def __hash__(self) -> int:
    return self.__energy.hash

  @property
  def hash(self) -> int:
    return self.__hash__()

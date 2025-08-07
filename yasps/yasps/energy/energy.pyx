# cython: language_level=3
from __future__ import annotations
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Tuple, Set, Dict, Union
from typing import TYPE_CHECKING
from yasps.attribute import attribute
from yasps.autodiff import autodiff
from yasps.helper import extract_block
import math
# for multiprocessing
from yasps.helper import timed # for timing

if TYPE_CHECKING:
  from yasps.hessianAndGradientKernel import hessianAndGradientKernel

from yasps.gradientIndicesKernel import gradientIndicesKernel

class energy:
  def __init__(self, energy: attribute, projection_method = 1, save_intermediate = False, gradient_only = False):
    if energy.size != 1:
      raise ValueError("energy.__init__: energy must be size 1.")
    self.__energy: attribute = energy
    self.__paths: List[List[attribute]] = [] # how to get to the roots
    # self.__roots: List[attribute] = []

    self.__wrt: List[attribute] = [] # an energy can be minimized for different attributes, for safety let's save all histories
    self.__wrt_start_indices: List[int] = [] # the start indices of the wrt attributes, this is used to determine the indices of the gradient
    self.__indices_cpu: np.ndarray = np.array([]) # save the indices on cpu
    self.__block_indices_gpu: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # save the block indices, this is for hessian accumulation
    self.__gradient_sizes_cpu: List[int] = [] # save the sizes of the gradient, this is to determine for the gradient, how large it is for each segment
    self.__gradient_sizes_gpu: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # save the sizes of the gradient
    self.__hessian: Optional[attribute] = None # save the hessian for each wrt input
    self.__gradient: Optional[attribute] = None # save the gradient for each wrt input
    self.__hessianAndGradientKernel: Optional[hessianAndGradientKernel] = None
    self.__hessian_blocks_where_to_check: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # we have a flattened array which stores the blocks. The blocks are sorted by dimensions. We need to know which block we are in for each smaller blocks in the hessian
    self.__merged_hessian_and_gradient_attribute: Optional[attribute] = None
    self.__project_entire_hessian: bool = True
    self.__projection_method = projection_method # 0 for no projection, 1 for absolute, 2 for max(0, val)
    # self.__projection_method = 0
    self.__save_intermediate = save_intermediate # save intermediate gradient and hessian result
    self.__intermediate_compute_pairs: Dict[str, Tuple[attribute, attribute]] = {} # save the intermediate compute pairs
    self.__gradient_only: bool = gradient_only # only compute gradient
    self.__indices_kernel: Optional[gradientIndicesKernel] = None
    self.__path_dict: Dict[attribute, List[attribute]] = {}
    self.__unioned_child_to_its_children: Dict[attribute, List[attribute]] = {} # because of the way our path is constructed, the direct unioned attribute doesnt show up in the path. So we need to record its children in the path
    _, self.__paths = self.getRoots(energy, [energy]) # get the root attributes

  # @property
  # def roots(self) -> List[attribute]:
  #   return self.__roots

  # @property
  # def gradient_sizes_cpu(self) -> List[int]:
  #   return self.__gradient_sizes_cpu

  @property
  def indices(self) -> np.ndarray:
    return self.__indices_cpu

  def clearIndices(self) -> None:
    # clear the indices_cpu
    self.__indices_cpu = np.array([])

  @property
  def block_indices_gpu(self) -> gpuarray.GPUArray:
    return self.__block_indices_gpu


  @block_indices_gpu.setter
  def block_indices_gpu(self, block_indices_gpu: gpuarray.GPUArray):
    self.__block_indices_gpu = block_indices_gpu

  @property
  def hessian_blocks_where_to_check(self) -> gpuarray.GPUArray:
    return self.__hessian_blocks_where_to_check

  @hessian_blocks_where_to_check.setter
  def hessian_blocks_where_to_check(self, hessian_blocks_where_to_check: gpuarray.GPUArray):
    self.__hessian_blocks_where_to_check = hessian_blocks_where_to_check

  @property
  def gradient_only(self) -> bool:
    return self.__gradient_only

  @property
  def outputCoordinates(self):
    assert self.__indices_kernel is not None, "Indices kernel not initialized"
    return self.__indices_kernel.outputCoordinates

  @property
  def outputBlockDimensions(self):
    assert self.__indices_kernel is not None, "Indices kernel not initialized"
    return self.__indices_kernel.outputBlockDimensions

  @property
  def numTotalCoordinates(self):
    assert self.__indices_kernel is not None, "Indices kernel not initialized"
    return self.__indices_kernel.numTotalCoordinates


  def getRoots(self, att: attribute, parentPath: List[attribute]) -> Tuple[List[attribute], List[List[attribute]]]:
    from yasps.attribute import JOIN, SUM, AVERAGE, DATA, UNION
    stack: List[attribute] = [att]
    seenRoots: Set[attribute] = set([])
    roots: List[attribute] = []
    # we perform dfs to extract a path and its children
    while stack:
      current: attribute = stack.pop()
      if current.operator == DATA:
        ## we got to the bottom of this path
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      elif current.operator == JOIN or current.operator == SUM or current.operator == AVERAGE:
        if current.through.dimension == 0:
          raise ValueError("energy.getRoots: att.through.dimension is 0. Such operation is not supported in energy minimization.")
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      elif current.operator == UNION:
        # we add the union operator to roots
        roots.append(current)
        seenRoots.add(current)
      else:
        stack.extend(current.children)
    # now we have the roots, which contains some joining operation
    # we need to duplicate the roots for those joining operation
    trueRoots: List[attribute] = []
    allPaths: List[List[attribute]] = []
    for root in roots:
      if root.operator == JOIN:
        # this is a joining operation
        # ok so at join, we will need to check how the joined attribute
        # will lead us to the final wrt attribute
        # TODO:
        # when the joined attribute is actually a union attribute
        # we will need to separate the paths into different possibilities
        # come back later for this
        childrenRoots, childrenPaths = self.getRoots(root.children[0], [])
        trueRoots += childrenRoots * root.through.dimension
        for childrenPath in childrenPaths:
          allPaths.append(parentPath + [root] + childrenPath)
        # allPaths += childrenPaths * root.through.dimension
      elif root.operator == UNION:
        # at union operator, we will need to add all the possible children paths
        for child in root.children:
          if child not in self.__unioned_child_to_its_children:
            self.__unioned_child_to_its_children[child] = []
          childrenRoots, childrenPaths = self.getRoots(child, [])
          trueRoots += childrenRoots
          for childrenPath in childrenPaths:
            allPaths.append(parentPath + [root] + childrenPath)
            if childrenPath[0] not in self.__unioned_child_to_its_children[child]:
              self.__unioned_child_to_its_children[child].append(childrenPath[0])
      elif root.operator == DATA:
        trueRoots.append(root)
        allPaths.append(parentPath + [root])
      else:
        raise ValueError(f"energy.getRoots: operator {root.operator} is not supported.")
    return trueRoots, allPaths

  @timed("energy.getSparseIndices")
  def getSparseIndices(self, wrt: List[attribute], wrt_start_indices: List[int]):
    self.__wrt = wrt
    self.__wrt_start_indices = wrt_start_indices
    usedPaths: List[List[attribute]] = []
    # we now always differentiate wrt all the data attributes
    # note that this excludes the constant attributes
    # after differentiation, we will decide which part of the matrix to put back in
    # if we simply cut off an attribute here, it will not be the full hessian we are projecting
    # as the eigen value we get will just be wrong
    # for path in self.__paths:
    #   if path[-1] in wrt:
    #     usedPaths.append(path)
    if self.__indices_kernel is None:
      # construct the path dict and generate the kernel
      pathDict: Dict[attribute, List[attribute]] = {}
      # we convert the used path as a dictionary
      # by having the parent children relationship
      for path in usedPaths:
        if len(path) == 1:
          raise ValueError("energy.getSparseIndices: minimizing a data attribute as energy is not allowed.")
        for i in range(len(path) - 1):
          parent: attribute = path[i]
          child: attribute = path[i + 1]
          if parent not in pathDict:
            pathDict[parent] = []
          if child not in pathDict[parent]:
            pathDict[parent].append(child)
      self.__path_dict = pathDict
      self.__indices_kernel = gradientIndicesKernel(pathDict, self.__unioned_child_to_its_children, wrt, wrt_start_indices, self.__energy)
    assert self.__indices_kernel is not None
    self.__indices_kernel.computeIndices(wrt_start_indices) # actually compute the indices
    return

  @timed("energy.getSparseIndicesAgain")
  def getSparseIndicesAgain(self):
    assert self.__indices_kernel is not None
    self.__indices_kernel.computeIndices(self.__wrt_start_indices) # actually compute the indices


  def computeIndices(self) -> None:
    assert self.__indices_kernel is not None, "energy.computeIndices: Indices kernel not initialized"
    self.__indices_kernel.computeIndices(self.__wrt_start_indices)
    return

  def __generateGradientThroughPathDict(self, wrt: List[attribute], differentiater: autodiff) -> None:
    from yasps.attribute import JOIN, UNION
    # # we are generating the symbolic gradient through the path dict
    # # the path dict already contains only the used paths
    # gradients: List[attribute] = []
    # if len(self.__path_dict) == 0:
    #   # there is nothing to do
    #   gradients.append(attribute.zeros(1, sum([x.size for x in wrt])))
    #   return
    if f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}' in self.__energy.correspondance.attributes:
      # nothing we need to do, the gradient is already computed
      self.__gradient = self.__energy.correspondance.attributes[f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}']
      return

    # otherwise
    # we start the magic
    # we first generate the local gradient for each parent to child attribute
    # this is the neighboring jacobian
    for parent in self.__path_dict.keys():
      children = self.__path_dict[parent]
      # first of all, let's ignore the case where you union energy to be an energy
      # let's say the energy is always at least some computed attribute instead of a joined or unioned attribute
      # we now first generate neighboring jacobian
      if parent.operator != JOIN and parent.operator != UNION:
        self.__generateNeighborJacobianForEnergy(parent, children, differentiater)
      elif parent.operator == JOIN:
        self.__generateNeighborJacobianForJoin(parent, children, differentiater)
      elif parent.operator == UNION:
        self.__generateNeighborJacobianForUnion(parent, children, differentiater)
      else:
        raise ValueError(f"energy.__generateGradientThroughPathDict: operator {parent.operator} is not supported in path dict.")

    # we have now generated all the neighboring jacobians
    # we need to multiply recursively to get the final gradient
    # first_gradient_name = f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    final_gradient = self.__generateGradientThroughRecursion(self.__energy, wrt)
    self.__gradient = final_gradient
    return

  def __generateGradientThroughRecursion(self, current: attribute, wrt: List[attribute]) -> attribute:
    from yasps.attribute import JOIN, UNION, DATA, CONSTANT
    if current.operator == CONSTANT:
      raise ValueError(f"energy.__generateHessianThroughRecursion: CONSTANT attributes are not supposed to show up in the path dict")
    if current.operator == DATA:
      # we are at the bottom level
      # end the recursion and return identity
      return attribute.identity(current.size)
    gradient_attribute_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    if gradient_attribute_name in current.correspondance.attributes:
      return current.correspondance[gradient_attribute_name]


    # now we determine case by case
    if current.operator != JOIN and current.operator != UNION:
      return self.__gennerateGlobalJacobianForEnergy(current, wrt)
    elif current.operator == JOIN:
      result = self.__generateGlobalJacobianForJoin(current, wrt)
      # if self.__save_intermediate:
      #   if result.fullName not in self.__intermediate_compute_pairs:
      #     # we save the intermediate result as an evaluated data pair
      #     new_name = result.name + "_evaluated"
      #     evaluated_result = current.correspondance.addAttribute(new_name, rows = result.rows, cols = result.cols)
      #     self.__intermediate_compute_pairs[result.fullName] = (result, evaluated_result)
      #     return evaluated_result
      #   else:
      #     # we already have the intermediate result, we just return the evaluated result
      #     return self.__intermediate_compute_pairs[result.fullName][1]
      return self.__generateGlobalJacobianForJoin(current, wrt)
    elif current.operator == UNION:
      result = self.__generateGlobalJacobianForUnion(current, wrt)
      # if self.__save_intermediate:
      #   if result.fullName not in self.__intermediate_compute_pairs:
      #     # we save the intermediate result as an evaluated data pair
      #     new_name = result.name + "_evaluated"
      #     evaluated_result = current.correspondance.addAttribute(new_name, rows = result.rows, cols = result.cols)
      #     self.__intermediate_compute_pairs[result.fullName] = (result, evaluated_result)
      #     return evaluated_result
      #   else:
      #     # we already have the intermediate result, we just return the evaluated result
      #     return self.__intermediate_compute_pairs[result.fullName][1]
      return self.__generateGlobalJacobianForUnion(current, wrt)
    else:
      raise ValueError(f"energy.__generateGradientThroughRecursion: operator {current.operator} is not supported in path dict.")

  def __generateNeighborJacobianForEnergy(self, parent: attribute, children: List[attribute], differentiater) -> None:
    # we are at the top level
    # compute the gradient at this level wrt all its children
    local_gradient_name = f'd_{parent.fullName}_d_{"__".join([x.fullName for x in children])}'
    if not local_gradient_name in parent.correspondance.attributes:
      diff_energy_wrt_children_list: List[attribute] = []
      for child in children:
        result = differentiater.diff(parent, child)
        for i in range(result.size):
          diff_energy_wrt_children_list.append(result[i])
      # we now construct the local gradient within this level
      local_gradient = attribute.to_array(diff_energy_wrt_children_list, rows = 1, cols = len(diff_energy_wrt_children_list))
      parent.correspondance.addAttribute(f'd_{parent.fullName}_d_{"__".join([x.fullName for x in children])}', computed_attribute = local_gradient)
    if self.__gradient_only:
      return
    local_hessian_name = f'd2_{parent.fullName}_d2_{"__".join([x.fullName for x in children])}'
    if not local_hessian_name in parent.correspondance.attributes:
      local_gradient = parent.correspondance[local_gradient_name]
      # now the local hessian is simply the autodiff of the gradient wrt the children again
      double_diff_results = [0.0 for _ in range(local_gradient.size * local_gradient.size)]
      col_offset = 0
      for child in children:
        # we differentiate the local gradient wrt to each child in path_dict
        # this should give us the result of dimension local_gradient.size x child.size
        local_double_diff = differentiater.diff(local_gradient, child)
        # we put it in double_diff_results
        for i in range(local_double_diff.rows):
          for j in range(local_double_diff.cols):
            double_diff_results[i * local_gradient.size + col_offset + j] = local_double_diff[i, j]
        col_offset += child.size
      assert col_offset == local_gradient.size, f"energy.__generateNeighborJacobianForEnergy: col_offset {col_offset} is not equal to local_gradient size {local_gradient.size}"
      local_hessian = attribute.to_array(double_diff_results, rows = local_gradient.size, cols = local_gradient.size)
      if self.__save_intermediate and local_hessian.isZero == 0:
        if local_hessian_name not in parent.correspondance.attributes:
          # we first create the attribute that will just be data
          hessian_data_attribute = parent.correspondance.addAttribute(local_hessian_name, rows = local_hessian.rows, cols = local_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            # add the pair
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            parent.correspondance.addAttribute(new_name, computed_attribute = local_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (local_hessian, hessian_data_attribute)
      else:
        # print("------------------------------------------------------------------------------------")
        # print("Local hessian name in local energy hessian is", local_hessian_name)
        # print("------------------------------------------------------------------------------------")
        parent.correspondance.addAttribute(local_hessian_name, computed_attribute = local_hessian)
        # print("Energy local hessian check")
        # print(local_hessian.compute().value.get()[:(local_hessian.rows * local_hessian.cols)].reshape((local_hessian.rows, local_hessian.cols)))
        # exit()
    return

  def __gennerateGlobalJacobianForEnergy(self, current: attribute, wrt: List[attribute]):
    gradient_attribute_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    if gradient_attribute_name in current.correspondance.attributes:
      # we already have the gradient
      return current.correspondance[gradient_attribute_name]
    # we are at energy
    # we get the children
    # and we simply assemble the final gradient by
    # multiplying the local gradient
    # with the global jacobian produced by all of its children
    children = self.__path_dict[current]
    current_gradient: attribute = current.correspondance[f'd_{current.fullName}_d_{"__".join([x.fullName for x in children])}']
    children_global_jacobian_name = f'd_{"__".join([x.fullName for x in children])}_d_{"__".join([x.fullName for x in wrt])}'
    next_jacobian: attribute
    if children_global_jacobian_name in current.correspondance.attributes:
      next_jacobian = current.correspondance[children_global_jacobian_name]
    else:
      children_jacobian: List[attribute] = []
      for child in children:
        children_jacobian.append(self.__generateGradientThroughRecursion(child, wrt))
      # then we first assemble everything in order
      next_jacobian_children = []
      next_jacobian_rows = sum([x.size for x in children])
      next_jacobian_cols = sum([x.cols for x in children_jacobian])
      next_jacobian_children = [attribute(float_value = 0.0) for _ in range(next_jacobian_rows * next_jacobian_cols)]
      # now we fill the jacobian
      row_offset = 0
      col_offset = 0
      for item in children_jacobian:
        for i in range(item.rows):
          for j in range(item.cols):
            next_jacobian_children[(i + row_offset) * next_jacobian_cols + j + col_offset] = item[i, j]
        col_offset += item.cols
        row_offset += item.rows
      next_jacobian = attribute.to_array(next_jacobian_children, rows = next_jacobian_rows, cols = next_jacobian_cols)
      if children_global_jacobian_name not in current.correspondance.attributes:
        current.correspondance.addAttribute(children_global_jacobian_name, computed_attribute = next_jacobian)
    full_gradient = current_gradient.mul_explicit(next_jacobian)
    current.correspondance.addAttribute(gradient_attribute_name, computed_attribute = full_gradient)
    return full_gradient

  def __gennerateGlobalHessianForEnergy(self, current: attribute, wrt: List[attribute]) -> attribute:
    # here we go, we will start computing the global hessian for the energy
    global_hessian_name = f'd2_{current.fullName}_d2_{"__".join([x.fullName for x in wrt])}'
    if global_hessian_name in current.correspondance.attributes:
      return current.correspondance[global_hessian_name]

    # now we first construct the second part of the hessian
    children = self.__path_dict[current]
    local_gradient_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in children])}'
    local_gradient = current.correspondance[local_gradient_name]

    global_jacobian_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    global_jacobian = current.correspondance[global_jacobian_name]

    # now we first assemble everything in order
    second_part_hessian_array = [0.0 for _ in range(global_jacobian.cols * global_jacobian.cols)]
    # now for each child, we will fill the hessian
    block_offset = 0
    for child in children:
      child_global_hessian = self.__generateHessianThroughRecursion(child, wrt)
      # now the child global hessian is size of child.size x hessian_size x hessian_size
      child_size = child.size # this marks the number of global hessian we have
      hessian_size = child_global_hessian.size // child_size # this is the size of the hessian for each child
      assert hessian_size * child_size == child_global_hessian.size, f"energy.__gennerateGlobalHessianForEnergy: hessian size {hessian_size} * child size {child_size} is not equal to child global hessian size {child_global_hessian.size}"
      hessian_rows = int(math.sqrt(hessian_size))
      assert hessian_rows * hessian_rows == hessian_size, f"energy.__gennerateGlobalHessianForEnergy: hessian rows {hessian_rows} * hessian rows {hessian_rows} is not equal to hessian size {hessian_size}"
      hessian_cols = hessian_rows # since its a square matrix
      for i in range(hessian_rows):
        for j in range(hessian_cols):
          for k in range(child_size):
            second_part_hessian_array[(block_offset + i) * global_jacobian.cols + (block_offset + j)] += local_gradient[block_offset + k] * child_global_hessian[k * hessian_size + i * hessian_cols + j]
      block_offset += child_size # we move the block offset to the next child
    # ok now we have the second part of the hessian
    # we will now construct the global hessian
    second_part_hessian = attribute.to_array(second_part_hessian_array, rows = global_jacobian.cols, cols = global_jacobian.cols)

    # ok then we construct the first part of the hessian
    local_hessian_name = f'd2_{current.fullName}_d2_{"__".join([x.fullName for x in children])}'
    # print("------------------------------------------------------------------------------------")
    # print("Local hessian name in global energy hessian is", local_hessian_name)
    # print("------------------------------------------------------------------------------------")

    local_hessian = current.correspondance[local_hessian_name]
    if second_part_hessian.isZero > 0:
      local_hessian = local_hessian.spd(self.__projection_method)
      self.__project_entire_hessian = False
    else:
      self.__project_entire_hessian = True
    # print("------------------------------------------------------------------------------------")
    # print("need to project entire hessian", self.__project_entire_hessian)
    # print(str(second_part_hessian))
    # print("------------------------------------------------------------------------------------")
    # exit(0)

    # now we assemble the children jacobian
    children_jacobian_name = f'd_{"__".join([x.fullName for x in children])}_d_{"__".join([x.fullName for x in wrt])}'
    # it is guaranteed that we have this since the children jacobian is computed before the global hessian
    children_global_jacobian = current.correspondance[children_jacobian_name]

    final_hessian = children_global_jacobian.transpose().mul_explicit(local_hessian.mul_explicit(children_global_jacobian)).add_explicit(second_part_hessian)
    # print("------------------------------------------------------------------------------------")
    # print("Checking the local hessian")
    # print(local_hessian.compute().value.get()[:(local_hessian.rows * local_hessian.cols)].reshape((local_hessian.rows, local_hessian.cols)))
    # print("Checking the jacobian")
    # print(children_global_jacobian.compute().value.get()[:(children_global_jacobian.rows * children_global_jacobian.cols)].reshape((children_global_jacobian.rows, children_global_jacobian.cols)))
    # print("Hessian checking")
    # print(final_hessian.compute().value.get()[:(final_hessian.rows * final_hessian.cols)].reshape((final_hessian.rows, final_hessian.cols)))
    # # exit()
    # print("------------------------------------------------------------------------------------")
    current.correspondance.addAttribute(global_hessian_name, computed_attribute = final_hessian)
    return final_hessian

  def __generateNeighborJacobianForJoin(self, parent: attribute, children: List[attribute], differentiater) -> None:
    joined_child = parent.children[0] # for join operation we first determine the child being joined
    # we will differentiate the joined node wrt the children
    child_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in children])}'
    if child_jacobian_name not in joined_child.correspondance.attributes:
      child_diff_next_children: List[attribute] = []
      for child in children:
        # we differentiate the joined_child wrt to each child in path_dict
        diff_result = differentiater.diff(joined_child, child)
        child_diff_next_children.append(diff_result)
      # we now determine the size of the jacobian
      jacobian_num_rows = joined_child.size
      jacobian_num_cols = sum([x.size for x in children])
      merged_jacobian_list: List[Union[float, attribute]] = [attribute(float_value = 0.0) for _ in range(jacobian_num_rows * jacobian_num_cols)]
      # now we fill the jacobian of the joined child
      col_offset = 0
      for child in child_diff_next_children:
        for i in range(child.rows):
          for j in range(child.cols):
            merged_jacobian_list[i * jacobian_num_cols + col_offset + j] = child[i, j]
        col_offset += child.cols
      merged_jacobian = attribute.to_array(merged_jacobian_list, rows = jacobian_num_rows, cols = jacobian_num_cols)
      # now add it to the correspondance
      joined_child.correspondance.addAttribute(child_jacobian_name, computed_attribute = merged_jacobian)
    if self.__gradient_only:
      return
    local_hessian_name = f'd2_{joined_child.fullName}_d2_{"__".join([x.fullName for x in children])}'
    if local_hessian_name not in joined_child.correspondance.attributes:
      # compute the neighboring hessian
      # for hessian what we are doing is basically just stacking up the hessian
      # instead of making it a tensor
      num_hessians = joined_child.size
      hessian_num_cols = sum([x.size for x in children])
      hessian_num_rows = hessian_num_cols
      # now we first access the jacobian which we just computed
      joined_child_jacobian = joined_child.correspondance[child_jacobian_name]
      # ok now each row is technically a derivative
      # and what we can do is just differentiate the jacobian wrt the children again
      all_hessian_results = []
      for row in range(num_hessians):
        # each row is a derivative wrt all the children
        # we will now need to differentiate it wrt each child again
        for col in range(joined_child_jacobian.cols):
          current_item = joined_child_jacobian[row, col]
          for child in children:
            # we differentiate the current item wrt the child
            diff_result = differentiater.diff(current_item, child)
            for i in range(diff_result.rows):
              for j in range(diff_result.cols):
                all_hessian_results.append(diff_result[i, j])

      merged_hessian = attribute.to_array(all_hessian_results, rows = hessian_num_rows * num_hessians, cols = hessian_num_cols)
      if self.__save_intermediate and merged_hessian.isZero == 0:
        if local_hessian_name not in joined_child.correspondance.attributes:
          # we first create the attribute that will just be data
          hessian_data_attribute = joined_child.correspondance.addAttribute(local_hessian_name, rows = merged_hessian.rows, cols = merged_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            # add the pair
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            joined_child.correspondance.addAttribute(new_name, computed_attribute = merged_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (merged_hessian, hessian_data_attribute)
      else:
        joined_child.correspondance.addAttribute(local_hessian_name, computed_attribute = merged_hessian)
    return

  def __generateGlobalJacobianForJoin(self, current: attribute, wrt: List[attribute]):
    gradient_attribute_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    if gradient_attribute_name in current.correspondance.attributes:
      result = current.correspondance[gradient_attribute_name]
      return result
    # ok to evaluate the full jacobian at join
    # we first note that the previous operation doesn't produce the local jacobian for the node with the join operation
    # instead it produces the local jacobian for the joined child
    # so what we need to do is get the local jacobian for the joind child
    # then get the global jacobian of this child's children
    # then multiply the two we get the global jacobian for the joined child
    # we will then join that global jacobian to get the global jacobian of the joined node
    joined_child = current.children[0]
    joined_child_global_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    joined_child_global_jacobian: attribute
    if joined_child_global_jacobian_name in joined_child.correspondance.attributes:
      # we already have the jacobian
      joined_child_global_jacobian = joined_child.correspondance[joined_child_global_jacobian_name]
    else:
      next_children = self.__path_dict[current]
      joined_child_local_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in next_children])}'
      joined_child_local_jacobian = joined_child.correspondance[joined_child_local_jacobian_name]

      children_global_jacobian_name = f'd_{"__".join([x.fullName for x in next_children])}_d_{"__".join([x.fullName for x in wrt])}'
      children_global_jacobian: attribute
      if children_global_jacobian_name in joined_child.correspondance.attributes:
        children_global_jacobian = joined_child.correspondance[children_global_jacobian_name]
      else:
        next_children_jacobians = []
        for child in next_children:
          # we get the global jacobian for the child
          child_jacobian = self.__generateGradientThroughRecursion(child, wrt)
          next_children_jacobians.append(child_jacobian)
        # ok we now assemble the global jacobian for the children of the joined child
        children_global_jacobian_items = []
        children_global_jacobian_rows = sum([x.size for x in next_children])
        children_global_jacobian_cols = sum([x.cols for x in next_children_jacobians])
        children_global_jacobian_items = [attribute(float_value = 0.0) for _ in range(children_global_jacobian_rows * children_global_jacobian_cols)]
        # now we fill the jacobian
        row_offset = 0
        col_offset = 0
        for item in next_children_jacobians:
          for i in range(item.rows):
            for j in range(item.cols):
              children_global_jacobian_items[(i + row_offset) * children_global_jacobian_cols + j + col_offset] = item[i, j]
          col_offset += item.cols
          row_offset += item.rows
        children_global_jacobian = attribute.to_array(children_global_jacobian_items, rows = children_global_jacobian_rows, cols = children_global_jacobian_cols)
        if children_global_jacobian_name not in joined_child.correspondance.attributes:
          joined_child.correspondance.addAttribute(children_global_jacobian_name, computed_attribute = children_global_jacobian)

      # we will now multiply the local jacobian with the global jacobian
      child_global_jacobian = joined_child_local_jacobian.mul_explicit(children_global_jacobian)

      # now we first add the attribute
      if joined_child_global_jacobian_name in joined_child.correspondance.attributes:
        # we already have the jacobian
        joined_child_global_jacobian = joined_child.correspondance[joined_child_global_jacobian_name]
      else:
        joined_child_global_jacobian = joined_child.correspondance.addAttribute(joined_child_global_jacobian_name, computed_attribute = child_global_jacobian)

    if self.__save_intermediate:
      if joined_child_global_jacobian.fullName not in self.__intermediate_compute_pairs:
        # we save the intermediate result as an evaluated data pair
        new_name = joined_child_global_jacobian.name + "_evaluated"
        evaluated_result = joined_child_global_jacobian.correspondance.addAttribute(new_name, rows = joined_child_global_jacobian.rows, cols = joined_child_global_jacobian.cols)
        self.__intermediate_compute_pairs[joined_child_global_jacobian.fullName] = (joined_child_global_jacobian, evaluated_result)
        joined_child_global_jacobian = evaluated_result
      else:
        # we already have the intermediate result, we just return the evaluated result
        joined_child_global_jacobian = self.__intermediate_compute_pairs[joined_child_global_jacobian.fullName][1]

    # we then perform the join operation
    res = current.correspondance.addAttribute(gradient_attribute_name+"_unresized", through = current.through, source = joined_child_global_jacobian)
    # ok now we need to reorder them because the true jacobian is not stacked together
    assert (joined_child_global_jacobian.rows * current.through.dimension) == current.size, f"energy.__generateGlobalJacobianForJoin: joined child global jacobian rows {joined_child_global_jacobian.rows} * current.through.dimension {current.through.dimension} is not equal to current size {current.size}"
    actual_global_jacobian_rows = current.size
    actual_global_jacobian_cols = joined_child_global_jacobian.cols * current.through.dimension
    actual_global_jacobian_items = [attribute(float_value = 0.0) for _ in range(actual_global_jacobian_rows * actual_global_jacobian_cols)]
    # now we put the items in the right order
    for index in range(current.through.dimension):
      for i in range(joined_child_global_jacobian.rows):
        for j in range(joined_child_global_jacobian.cols):
          actual_global_jacobian_items[(index * joined_child_global_jacobian.rows + i) * actual_global_jacobian_cols + (index * joined_child_global_jacobian.cols) + j] = res[index, i * joined_child_global_jacobian.cols + j]

    actual_global_jacobian = attribute.to_array(actual_global_jacobian_items, rows = actual_global_jacobian_rows, cols = actual_global_jacobian_cols)
    current.correspondance.addAttribute(gradient_attribute_name, computed_attribute = actual_global_jacobian)
    return actual_global_jacobian

  def __generateGlobalHessianForJoin(self, current: attribute, wrt: List[attribute]) -> attribute:
    # ok so remember, for join, we have done the local hessian for the joined child
    # so what we need to do, is to compute the global hessian for the joined child
    # then accumulate it through the join operation
    # then rearrange it
    global_hessian_name = f'd2_{current.children[0].fullName}_d2_{"__".join([x.fullName for x in wrt])}'
    if global_hessian_name in current.correspondance.attributes:
      return current.correspondance[global_hessian_name]

    # first we need to get the global hessian for the joined child
    joined_child: attribute = current.children[0]

    # we now check if the joined child's global hessian has already been computed
    joined_child_global_hessian: attribute
    joined_child_global_hessian_name = f'd2_{joined_child.fullName}_d2_{"__".join([x.fullName for x in wrt])}'
    if joined_child_global_hessian_name in joined_child.correspondance.attributes:
      # we already have the global hessian for the joined child
      joined_child_global_hessian = joined_child.correspondance[joined_child_global_hessian_name]
    else:
      num_hessians = joined_child.size # how many hessians we have
      next_children = self.__path_dict[current]
      joined_child_local_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in next_children])}'
      joined_child_local_jacobian = joined_child.correspondance[joined_child_local_jacobian_name]
      joined_child_local_hessian_name = f'd2_{joined_child.fullName}_d2_{"__".join([x.fullName for x in next_children])}'
      joined_child_local_hessian = joined_child.correspondance[joined_child_local_hessian_name]

      next_children_global_jacobian_name = f'd_{"__".join([x.fullName for x in next_children])}_d_{"__".join([x.fullName for x in wrt])}'
      next_children_global_jacobian = joined_child.correspondance[next_children_global_jacobian_name]

      next_children_global_hessian: List[attribute] = []
      for child in next_children:
        # we get the global hessian for the child
        child_global_hessian = self.__generateHessianThroughRecursion(child, wrt)
        next_children_global_hessian.append(child_global_hessian)
      next_children_total_size = sum([x.size for x in next_children])
      local_hessian_size = next_children_total_size * next_children_total_size # what is the size of the local hessian (remember we have a tensor, this is the size of the first and second dimension)
      assert local_hessian_size * num_hessians == joined_child_local_hessian.size, f"energy.__generateGlobalHessianForJoin: local hessian size {local_hessian_size} * num hessians {num_hessians} is not equal to joined child local hessian size {joined_child_local_hessian.size}"
      local_hessian_rows = int(math.sqrt(local_hessian_size))
      assert local_hessian_rows * local_hessian_rows == local_hessian_size, f"energy.__generateGlobalHessianForJoin: local hessian rows {local_hessian_rows} * local hessian rows {local_hessian_rows} is not equal to local hessian size {local_hessian_size}"
      local_hessian_cols = local_hessian_rows # since its a square matrix
      joined_child_global_hessian_items = []
      for N in range(num_hessians):
        # we will construct the Nth hessian, because it's a tensor so we need to stack them up
        nth_joined_child_local_hessian = attribute.to_array([joined_child_local_hessian[i] for i in range(N * local_hessian_size, (N + 1) * local_hessian_size)], rows = local_hessian_rows, cols = local_hessian_cols)
        # now we construct the first part of the hessian, which is jacobian * local hessian * jacobian
        first_part_hessian = next_children_global_jacobian.transpose().mul_explicit(nth_joined_child_local_hessian.mul_explicit(next_children_global_jacobian))

        # get the gradient size
        local_gradient_size = joined_child_local_jacobian.size // num_hessians # this is the size of the gradient for each hessian
        assert local_gradient_size * num_hessians == joined_child_local_jacobian.size, f"energy.__generateGlobalHessianForJoin: local gradient size {local_gradient_size} * num hessians {num_hessians} is not equal to joined child local jacobian size {joined_child_local_jacobian.size}"

        # now we will need to add the second part of the hessian
        second_part_hessian_array = [0.0 for _ in range(next_children_global_jacobian.cols * next_children_global_jacobian.cols)]
        block_offset = 0
        index = 0
        for child in next_children:
          next_child_global_hessian = next_children_global_hessian[index]
          # now the child global hessian is size of child.size x hessian_size x hessian_size
          child_size = child.size # this marks the number of global hessian we have
          hessian_size = next_child_global_hessian.size // child_size # this is the size of the hessian for each child
          assert hessian_size * child_size == next_child_global_hessian.size, f"energy.__gennerateGlobalHessianForJoin: hessian size {hessian_size} * child size {child_size} is not equal to child global hessian size {next_child_global_hessian.size}"
          hessian_rows = int(math.sqrt(hessian_size))
          assert hessian_rows * hessian_rows == hessian_size, f"energy.__gennerateGlobalHessianForJoin: hessian rows {hessian_rows} * hessian rows {hessian_rows} is not equal to hessian size {hessian_size}"
          hessian_cols = hessian_rows # since its a square matrix
          for i in range(hessian_rows):
            for j in range(hessian_cols):
              for k in range(child_size):
                second_part_hessian_array[(block_offset + i) * next_children_global_jacobian.cols + (block_offset + j)] += joined_child_local_jacobian[N * joined_child_local_jacobian.cols + k + block_offset] * next_child_global_hessian[k * hessian_size + i * hessian_cols + j]
          block_offset += child_size # we move the block offset to the next child
          index += 1
        # ok now we have the second part of the hessian
        # we will now construct the final hessian
        second_part_hessian = attribute.to_array(second_part_hessian_array, rows = next_children_global_jacobian.cols, cols = next_children_global_jacobian.cols)
        final_hessian = first_part_hessian.add_explicit(second_part_hessian)

        # now we add the final hessian to the items
        for i in range(final_hessian.size):
          joined_child_global_hessian_items.append(final_hessian[i])
      # so now we have all the hessians stacked up
      # we perform the join operation
      # first we construct the joined child global hessian
      joined_child_global_hessian = attribute.to_array(joined_child_global_hessian_items, rows = num_hessians * next_children_global_jacobian.cols, cols = next_children_global_jacobian.cols)
      # now we add it to the correspondance
      joined_child.correspondance.addAttribute(joined_child_global_hessian_name, computed_attribute = joined_child_global_hessian)
      # finally we perform the join operation on the current node
    if self.__save_intermediate:
      if joined_child_global_hessian.fullName not in self.__intermediate_compute_pairs:
        # we save the intermediate result as an evaluated data pair
        new_name = joined_child_global_hessian.name + "_evaluated"
        evaluated_result = joined_child_global_hessian.correspondance.addAttribute(new_name, rows = joined_child_global_hessian.rows, cols = joined_child_global_hessian.cols)
        self.__intermediate_compute_pairs[joined_child_global_hessian.fullName] = (joined_child_global_hessian, evaluated_result)
        joined_child_global_hessian = evaluated_result
      else:
        # we already have the intermediate result, we just return the evaluated result
        joined_child_global_hessian = self.__intermediate_compute_pairs[joined_child_global_hessian.fullName][1]
    current.correspondance.addAttribute(global_hessian_name, through = current.through, source = joined_child_global_hessian)
    return current.correspondance[global_hessian_name]

  def __generateNeighborJacobianForUnion(self, parent: attribute, children: List[attribute], differentiater):
    unioned_children = parent.children
    # print("Unioned parent is")
    # print(parent.fullName)
    # for each unioned child, we will check what the jacobian is wrt that child's children
    for unioned_child in unioned_children:
      unioned_child_used_children: List[attribute] = []
      for child in children:
        if child in self.__unioned_child_to_its_children[unioned_child]:
          # ok we will add this child since its on the path
          unioned_child_used_children.append(child)
      # now we actually know what children are being used
      # we will need to generate the jacobian
      # check if there is anything needed
      if len(unioned_child_used_children) == 0:
        continue

      # check if the jacobian already exists
      child_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in unioned_child_used_children])}'
      if child_jacobian_name in unioned_child.correspondance.attributes:
        continue

      local_jacobians = []
      for child in unioned_child_used_children:
        diff_result = differentiater.diff(unioned_child, child)
        local_jacobians.append(diff_result)
      # ok we construct the jacobian now
      jacobian_num_rows = unioned_child.size
      jacobian_num_cols = sum([x.size for x in unioned_child_used_children])
      merged_jacobian_list = [attribute(float_value = 0.0) for _ in range(jacobian_num_rows * jacobian_num_cols)]
      # now we fill the jacobian of the joined child
      col_offset = 0
      for child in local_jacobians:
        for i in range(child.rows):
          for j in range(child.cols):
            merged_jacobian_list[i * jacobian_num_cols + col_offset + j] = child[i, j]
        col_offset += child.cols
      merged_jacobian = attribute.to_array(merged_jacobian_list, rows = jacobian_num_rows, cols = jacobian_num_cols)
      # now add it to the correspondance
      unioned_child.correspondance.addAttribute(child_jacobian_name, computed_attribute = merged_jacobian)
    if self.__gradient_only:
      return
    # now we compute the hessian for each unioned child
    for unioned_child in unioned_children:
      unioned_child_used_children: List[attribute] = []
      for child in children:
        if child in self.__unioned_child_to_its_children[unioned_child]:
          # ok we will add this child since its on the path
          unioned_child_used_children.append(child)
      # now we actually know what children are being used
      # we will need to generate the jacobian
      # check if there is anything needed
      if len(unioned_child_used_children) == 0:
        continue
      # check if the jacobian already exists
      child_hessian_name = f'd2_{unioned_child.fullName}_d2_{"__".join([x.fullName for x in unioned_child_used_children])}'
      if child_hessian_name in unioned_child.correspondance.attributes:
        continue
      child_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in unioned_child_used_children])}'
      # now we do the same thing as the join operator
      # we furst get the jacobian of the unioned child
      child_jacobian = unioned_child.correspondance[child_jacobian_name]
      num_hessians = unioned_child.size
      hessian_num_cols = sum([x.size for x in unioned_child_used_children])
      hessian_num_rows = hessian_num_cols
      all_hessian_results = []
      for row in range(num_hessians):
        for col in range(child_jacobian.cols):
          current_item = child_jacobian[row, col]
          for child in unioned_child_used_children:
            diff_result = differentiater.diff(current_item, child)
            for i in range(diff_result.rows):
              for j in range(diff_result.cols):
                all_hessian_results.append(diff_result[i, j])
      merged_hessian = attribute.to_array(all_hessian_results, rows = hessian_num_rows * num_hessians, cols = hessian_num_cols)

      if self.__save_intermediate and merged_hessian.isZero == 0:
        if child_hessian_name not in unioned_child.correspondance.attributes:
          # we first create the attribute that will just be data
          hessian_data_attribute = unioned_child.correspondance.addAttribute(child_hessian_name, rows = merged_hessian.rows, cols = merged_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            # add the pair
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            unioned_child.correspondance.addAttribute(new_name, computed_attribute = merged_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (merged_hessian, hessian_data_attribute)
      else:
        if child_hessian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(child_hessian_name, computed_attribute = merged_hessian)
    return

  def __generateGlobalJacobianForUnion(self, current: attribute, wrt: List[attribute]):
    gradient_attribute_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}_filled_for_{self.__energy.fullName}'
    # print("-----------------------------------------------------------")
    # print(f"At union, generating for {gradient_attribute_name}")
    # print("Checking children correspondance")
    # for child in current.children:
    #   print(f"Child {child.fullName} has correspondance: {child.correspondance.fullName}")
    # print("-----------------------------------------------------------")
    if gradient_attribute_name in current.correspondance.attributes:
      return current.correspondance[gradient_attribute_name]
    # gradient_attribute_name_unfilled = f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}'
    # the procedure for union is similar to join
    # except that we need to deal with each of the children separately
    # then perform the union again
    # and we also need to make sure the union attribute has the same dimension
    # so we also technically need to expand the global jacobian of each child
    # to make sure they have the same dimension
    unioned_children_global_jacobians: List[attribute] = []
    children_on_path: List[attribute] = self.__path_dict[current]
    for unioned_child in current.children:
      # print(f"We are at unioned child {unioned_child.fullName}")
      # first we determine if this global jacobian is already computed
      unioned_child_global_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in wrt])}'
      unioned_child_global_jacobian: attribute
      if unioned_child_global_jacobian_name in unioned_child.correspondance.attributes:
        # we already have the jacobian
        unioned_child_global_jacobian = unioned_child.correspondance[unioned_child_global_jacobian_name]
        unioned_children_global_jacobians.append(unioned_child_global_jacobian)
        # print(f"Unioned child already has a global jacobian, array length: {len(unioned_children_global_jacobians)}")
      else:
        # ok we first need to find out which children are being used
        used_children: List[attribute] = []
        for child in children_on_path:
          if child in self.__unioned_child_to_its_children[unioned_child]:
            # ok we will add this child since its on the path
            used_children.append(child)
        if len(used_children) == 0:
          unioned_children_global_jacobians.append(attribute.zeros(unioned_child.size, 1))
          # print(f"Unioned child has no need to compute jacobian, array length: {len(unioned_children_global_jacobians)}")
          continue

        used_children_global_jacobian_name = f'd_{"__".join([x.fullName for x in used_children])}_d_{"__".join([x.fullName for x in wrt])}'
        children_global_jacobian: attribute
        if used_children_global_jacobian_name in unioned_child.correspondance.attributes:
          # we already have the global jacobian for the children
          children_global_jacobian = unioned_child.correspondance[used_children_global_jacobian_name]
        else:
          # ok now we get the used children's global jacobian
          used_children_global_jacobians: List[attribute] = []
          for child in used_children:
            # we get the global jacobian for the child
            used_children_global_jacobian = self.__generateGradientThroughRecursion(child, wrt)
            used_children_global_jacobians.append(used_children_global_jacobian)

          # ok now we construct the global jacobian for the used children
          children_global_jacobian_items = []
          children_global_jacobian_rows = sum([x.size for x in used_children])
          children_global_jacobian_cols = sum([x.cols for x in used_children_global_jacobians])
          children_global_jacobian_items = [attribute(float_value = 0.0) for _ in range(children_global_jacobian_rows * children_global_jacobian_cols)]
          # now we fill the jacobian
          row_offset = 0
          col_offset = 0
          for item in used_children_global_jacobians:
            for i in range(item.rows):
              for j in range(item.cols):
                children_global_jacobian_items[(i + row_offset) * children_global_jacobian_cols + j + col_offset] = item[i, j]
            col_offset += item.cols
            row_offset += item.rows
          children_global_jacobian = attribute.to_array(children_global_jacobian_items, rows = children_global_jacobian_rows, cols = children_global_jacobian_cols)
          if used_children_global_jacobian_name not in unioned_child.correspondance.attributes:
            unioned_child.correspondance.addAttribute(used_children_global_jacobian_name, computed_attribute = children_global_jacobian)
        # we will now multiply the local jacobian with the global jacobian
        # get the child neighbor jacobian
        child_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in used_children])}'
        child_local_jacobian = unioned_child.correspondance[child_jacobian_name]
        unioned_child_global_jacobian = child_local_jacobian.mul_explicit(children_global_jacobian)
        if unioned_child_global_jacobian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(unioned_child_global_jacobian_name, computed_attribute = unioned_child_global_jacobian)
        unioned_children_global_jacobians.append(unioned_child.correspondance[unioned_child_global_jacobian_name])
        # print(f"Unioned child add a jacobian now, array length: {len(unioned_children_global_jacobians)}")
    if self.__save_intermediate:
      for (index, item) in enumerate(unioned_children_global_jacobians):
        if item.fullName not in self.__intermediate_compute_pairs:
          # we save the intermediate result as an evaluated data pair
          new_name = item.name + "_evaluated"
          evaluated_result = item.correspondance.addAttribute(new_name, rows = item.rows, cols = item.cols)
          self.__intermediate_compute_pairs[item.fullName] = (item, evaluated_result)
          unioned_children_global_jacobians[index] = evaluated_result
        else:
          # we already have the intermediate result, we just return the evaluated result
          unioned_children_global_jacobians[index] = self.__intermediate_compute_pairs[item.fullName][1]
    # ok now that we have all of the children's jacobian
    # what we need to do is to get the largest dimension
    # in reality, they all should have the same rows
    # only different columns
    max_cols = max([x.cols for x in unioned_children_global_jacobians])
    max_rows = max([x.rows for x in unioned_children_global_jacobians])
    # ok now we expand each of the unioned children
    for (index, jacobian) in enumerate(unioned_children_global_jacobians):
      # let's do an assert to see if the rows are the same
      assert jacobian.rows == max_rows, f"energy.__generateGlobalJacobianForUnion: jacobian rows {jacobian.rows} is not equal to max rows {max_rows}"
      if jacobian.name == gradient_attribute_name:
        assert jacobian.cols == max_cols, f"energy.__generateGlobalJacobianForUnion: jacobian cols {jacobian.cols} is not equal to max cols {max_cols}"
        assert jacobian.rows == max_rows, f"energy.__generateGlobalJacobianForUnion: jacobian rows {jacobian.rows} is not equal to max rows {max_rows}"
        continue
      expanded_jacobian_list = [0.0 for _ in range(max_rows * max_cols)]
      # now we fill the jacobian
      for i in range(jacobian.rows):
        for j in range(jacobian.cols):
          expanded_jacobian_list[i * max_cols + j] = jacobian[i, j]
      expanded_jacobian = attribute.to_array(expanded_jacobian_list, rows = max_rows, cols = max_cols)
      if not gradient_attribute_name in current.children[index].correspondance.attributes:
        # print(f"Adding attribute with name {gradient_attribute_name} to child {current.children[index].correspondance.fullName}")
        current.children[index].correspondance.addAttribute(gradient_attribute_name, computed_attribute = expanded_jacobian)

      # print("Checking correspondance again")
      # print(expanded_jacobian.correspondance.fullName)

      # well we have expanded the jacobian for each child
    #   # perform the union operation
    res = current.correspondance.addAttribute(gradient_attribute_name)
    return res

  def __generateGlobalHessianForUnion(self, current: attribute, wrt: List[attribute]) -> attribute:
    global_hessian_name = f'd2_{current.fullName}_d2_{"__".join([x.fullName for x in wrt])}_filled_for_{self.__energy.fullName}'
    # global_hessian_name_unfilled = f'd2_{current.fullName}_d2_{"__".join([x.fullName for x in wrt])}'
    if global_hessian_name in current.correspondance.attributes:
      return current.correspondance[global_hessian_name]
    # the union operator is a bit more complicated
    # what we need to do is to compute the global hessian for each unioned child
    # and then fill them with 0 because we are working on gpu, and need to preallocate the space
    unioned_children_global_hessians: List[attribute] = []
    children_on_path: List[attribute] = self.__path_dict[current]
    for unioned_child in current.children:
      # print(f"We are at unioned child {unioned_child.fullName}")
      # first we determine if this global hessian is already computed
      unioned_child_global_hessian_name = f'd2_{unioned_child.fullName}_d2_{"__".join([x.fullName for x in wrt])}'
      unioned_child_global_hessian: attribute
      if unioned_child_global_hessian_name in unioned_child.correspondance.attributes:
        # we already have the hessian
        unioned_child_global_hessian = unioned_child.correspondance[unioned_child_global_hessian_name]
        unioned_children_global_hessians.append(unioned_child_global_hessian)
        # print("=============================================================")
        # print("Unioned child already has a global hessian")
        # print(f"Dimensions are: {unioned_child_global_hessian.rows} x {unioned_child_global_hessian.cols}")
        # print("=============================================================")
      else:
        # ok we first need to find out which children are being used
        used_children: List[attribute] = []
        for child in children_on_path:
          if child in self.__unioned_child_to_its_children[unioned_child]:
            # ok we will add this child since its on the path
            used_children.append(child)
        if len(used_children) == 0:
          unioned_children_global_hessians.append(attribute.zeros(current.size, 1))
          continue

        # now we construct the global hessian for the unioned child
        num_hessians = unioned_child.size # how many hessians we have
        next_children = used_children
        unioned_child_local_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in next_children])}'
        unioned_child_local_jacobian = unioned_child.correspondance[unioned_child_local_jacobian_name]
        unioned_child_local_hessian_name = f'd2_{unioned_child.fullName}_d2_{"__".join([x.fullName for x in next_children])}'
        unioned_child_local_hessian = unioned_child.correspondance[unioned_child_local_hessian_name]

        next_children_global_jacobian_name = f'd_{"__".join([x.fullName for x in next_children])}_d_{"__".join([x.fullName for x in wrt])}'
        next_children_global_jacobian = unioned_child.correspondance[next_children_global_jacobian_name]

        next_children_global_hessians: List[attribute] = []
        for child in next_children:
          child_global_hessian = self.__generateHessianThroughRecursion(child, wrt)
          next_children_global_hessians.append(child_global_hessian)
        next_children_total_size = sum([x.size for x in next_children])
        local_hessian_size = next_children_total_size * next_children_total_size # what is the size of the local hessian (remember we have a tensor, this is the size of the first and second dimension)
        assert local_hessian_size * num_hessians == unioned_child_local_hessian.size, f"energy.__generateGlobalHessianForUnion: local hessian size {local_hessian_size} * num hessians {num_hessians} is not equal to unioned child local hessian size {unioned_child_local_hessian.size}"
        local_hessian_rows = int(math.sqrt(local_hessian_size))
        assert local_hessian_rows * local_hessian_rows == local_hessian_size, f"energy.__generateGlobalHessianForUnion: local hessian rows {local_hessian_rows} * local hessian rows {local_hessian_rows} is not equal to local hessian size {local_hessian_size}"
        local_hessian_cols = local_hessian_rows # since its a square matrix
        unioned_child_global_hessian_items = []
        for N in range(num_hessians):
          nth_joined_child_local_hessian = attribute.to_array([unioned_child_local_hessian[i] for i in range(N * local_hessian_size, (N + 1) * local_hessian_size)], rows = local_hessian_rows, cols = local_hessian_cols)
          first_part_hessian = next_children_global_jacobian.transpose().mul_explicit(nth_joined_child_local_hessian.mul_explicit(next_children_global_jacobian))

          # get the gradient size
          local_gradient_size = unioned_child_local_jacobian.size // num_hessians # this is the size of the gradient for each hessian
          assert local_gradient_size * num_hessians == unioned_child_local_jacobian.size, f"energy.__generateGlobalHessianForUnion: local gradient size {local_gradient_size} * num hessians {num_hessians} is not equal to unioned child local jacobian size {unioned_child_local_jacobian.size}"

          second_part_hessian_array = [0.0 for _ in range(next_children_global_jacobian.cols * next_children_global_jacobian.cols)]
          block_offset = 0
          for (index, child) in enumerate(next_children):
            next_child_global_hessian = next_children_global_hessians[index]
            # now the child global hessian is size of child.size x hessian_size x hessian_size
            child_size = child.size
            hessian_size = next_child_global_hessian.size // child_size # this is the size of the hessian for each child
            assert hessian_size * child_size == next_child_global_hessian.size, f"energy.__generateGlobalHessianForUnion: hessian size {hessian_size} * child size {child_size} is not equal to child global hessian size {next_child_global_hessian.size}"
            hessian_rows = int(math.sqrt(hessian_size))
            assert hessian_rows * hessian_rows == hessian_size, f"energy.__generateGlobalHessianForUnion: hessian rows {hessian_rows} * hessian rows {hessian_rows} is not equal to hessian size {hessian_size}"
            hessian_cols = hessian_rows # since its a square matrix
            for i in range(hessian_rows):
              for j in range(hessian_cols):
                for k in range(child_size):
                  second_part_hessian_array[(block_offset + i) * next_children_global_jacobian.cols + (block_offset + j)] += unioned_child_local_jacobian[N * unioned_child_local_jacobian.cols + k + block_offset] * next_child_global_hessian[k * hessian_size + i * hessian_cols + j]
            block_offset += child_size # we move the block offset to the next child
          second_part_hessian = attribute.to_array(second_part_hessian_array, rows = next_children_global_jacobian.cols, cols = next_children_global_jacobian.cols)
          final_hessian = first_part_hessian.add_explicit(second_part_hessian)
          for i in range(final_hessian.size):
            unioned_child_global_hessian_items.append(final_hessian[i])
        unioned_child_global_hessian = attribute.to_array(unioned_child_global_hessian_items, rows = num_hessians * next_children_global_jacobian.cols, cols = next_children_global_jacobian.cols)
        if unioned_child_global_hessian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(unioned_child_global_hessian_name, computed_attribute = unioned_child_global_hessian)
        # print("Unioned child name is")
        # print(unioned_child.fullName)
        assert unioned_child.correspondance[unioned_child_global_hessian_name].rows == unioned_child_global_hessian.rows, f"energy.__generateGlobalHessianForUnion: unioned child global hessian rows {unioned_child.correspondance[unioned_child_global_hessian_name].rows} is not equal to unioned child global hessian rows {unioned_child_global_hessian.rows}, name is {unioned_child_global_hessian_name}"
        assert unioned_child.correspondance[unioned_child_global_hessian_name].cols == unioned_child_global_hessian.cols, f"energy.__generateGlobalHessianForUnion: unioned child global hessian cols {unioned_child.correspondance[unioned_child_global_hessian_name].cols} is not equal to unioned child global hessian cols {unioned_child_global_hessian.cols}"
        unioned_children_global_hessians.append(unioned_child.correspondance[unioned_child_global_hessian_name])
        # print("-------------------------------------------------------------")
        # print("Unioned child does not have a global hessian")
        # print(f"Dimensions are: {unioned_child_global_hessian.rows} x {unioned_child_global_hessian.cols}")
        # print("-------------------------------------------------------------")
    # ok now that we have all of the children's hessian
    # what we need to do is to get the largest dimension
    # print("=========================================================")
    # print("All children global hessian dimensions")
    # for item in unioned_children_global_hessians:
    #   print(f"Rows: {item.rows}, Cols: {item.cols}, Size: {item.size}")
    # print("=========================================================")
    if self.__save_intermediate:
      for (index, item) in enumerate(unioned_children_global_hessians):
        if item.fullName not in self.__intermediate_compute_pairs:
          # we save the intermediate result as an evaluated data pair
          new_name = item.name + "_evaluated"
          evaluated_result = item.correspondance.addAttribute(new_name, rows = item.rows, cols = item.cols)
          self.__intermediate_compute_pairs[item.fullName] = (item, evaluated_result)
          unioned_children_global_hessians[index] = evaluated_result
        else:
          # we already have the intermediate result, we just return the evaluated result
          unioned_children_global_hessians[index] = self.__intermediate_compute_pairs[item.fullName][1]

    largest_cols = max([x.cols for x in unioned_children_global_hessians])
    largest_rows = max([x.rows for x in unioned_children_global_hessians])
    assert largest_rows % largest_cols == 0, f"energy.__generateGlobalHessianForUnion: largest rows {largest_rows} is not divisible by largest cols {largest_cols}"

    # now that we have the unioned children
    # we will need to expand each of the unioned children to the expected size
    for (index, unioned_child) in enumerate(current.children):
      expanded_hessian_items = [0.0 for _ in range(largest_rows * largest_cols)]
      num_hessians = unioned_child.size
      unexpanded_hessian = unioned_children_global_hessians[index]
      hessian_size = unexpanded_hessian.size // num_hessians
      assert hessian_size * num_hessians == unexpanded_hessian.size, f"energy.__generateGlobalHessianForUnion: hessian size {hessian_size} * num hessians {num_hessians} is not equal to unexpanded hessian size {unexpanded_hessian.size}"
      assert largest_rows // largest_cols == num_hessians, f"energy.__generateGlobalHessianForUnion: largest rows {largest_rows} // largest cols {largest_cols} is not equal to num hessians {num_hessians}, unioned child is {unioned_child.fullName}"
      for i in range(num_hessians):
        for j in range(unexpanded_hessian.cols):
          for k in range(unexpanded_hessian.cols):
            expanded_hessian_items[i * largest_cols * largest_cols + j * largest_cols + k] = unexpanded_hessian[i * hessian_size + j * unexpanded_hessian.cols + k]
      expanded_hessian = attribute.to_array(expanded_hessian_items, rows = largest_rows, cols = largest_cols)
      if self.__save_intermediate:
      # we first check if the name is in the unioned child's correspondance
        if global_hessian_name not in unioned_child.correspondance.attributes:
        # we create a data attribute with the same dimension
          hessian_data_attribute = unioned_child.correspondance.addAttribute(global_hessian_name, rows = expanded_hessian.rows, cols = expanded_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            # add the pair
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            unioned_child.correspondance.addAttribute(new_name, computed_attribute = expanded_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (expanded_hessian, hessian_data_attribute)
      else:
        if global_hessian_name not in unioned_child.correspondance.attributes:
          # we add the expanded hessian to the correspondance
          unioned_child.correspondance.addAttribute(global_hessian_name, computed_attribute = expanded_hessian)

    # now we add the expanded hessian to the correspondance
    res = current.correspondance.addAttribute(global_hessian_name)
    return res






  def __generateHessianThroughPathDict(self, wrt: List[attribute], differentiater: autodiff) -> None:
    # # we are generating the symbolic hessian through the path dict
    # # the path dict already contains only the used paths
    if f'd2_{self.__energy.fullName}_d2_{"__".join([x.fullName for x in wrt])}' in self.__energy.correspondance.attributes:
      # nothing we need to do, the gradient is already computed
      self.__hessian = self.__energy.correspondance.attributes[f'd2_{self.__energy.fullName}_d2_{"__".join([x.fullName for x in wrt])}']
      return
    # otherwise, the local hessian is already generated
    # we will just call the function that computes the global hessian
    self.__hessian = self.__generateHessianThroughRecursion(self.__energy, wrt)
    return

  def __generateHessianThroughRecursion(self, current: attribute, wrt: List[attribute]) -> attribute:
    from yasps.attribute import JOIN, UNION, DATA, CONSTANT
    if current.operator == CONSTANT:
      raise ValueError(f"energy.__generateHessianThroughRecursion: CONSTANT attributes are not supposed to show up in the path dict")
    if current.operator == DATA:
      # if its data, the second derivative is just zeros
      return attribute.zeros(current.size, current.size * current.size)
    hessian_attribute_name = f'd2_{current.fullName}_d2_{"__".join([x.fullName for x in wrt])}'
    if hessian_attribute_name in current.correspondance.attributes:
      return current.correspondance[hessian_attribute_name]
    if current.operator != JOIN and current.operator != UNION:
      return self.__gennerateGlobalHessianForEnergy(current, wrt)
    elif current.operator == JOIN:
      result = self.__generateGlobalHessianForJoin(current, wrt)
      # if self.__save_intermediate and result.isZero == 0:
      #   if result.fullName not in self.__intermediate_compute_pairs:
      #     # we save the intermediate result as an evaluated data pair
      #     new_name = result.name + "_evaluated"
      #     evaluated_result = current.correspondance.addAttribute(new_name, rows = result.rows, cols = result.cols)
      #     self.__intermediate_compute_pairs[result.fullName] = (result, evaluated_result)
      #     return evaluated_result
      #   else:
      #     # we already have the intermediate result, we just return the evaluated result
      #     return self.__intermediate_compute_pairs[result.fullName][1]
      return result
    elif current.operator == UNION:
      result = self.__generateGlobalHessianForUnion(current, wrt)
      # if self.__save_intermediate:
      #   if result.fullName not in self.__intermediate_compute_pairs:
      #     # we save the intermediate result as an evaluated data pair
      #     new_name = result.name + "_evaluated"
      #     evaluated_result = current.correspondance.addAttribute(new_name, rows = result.rows, cols = result.cols)
      #     self.__intermediate_compute_pairs[result.fullName] = (result, evaluated_result)
      #     return evaluated_result
      #   else:
      #     # we already have the intermediate result, we just return the evaluated result
      #     return self.__intermediate_compute_pairs[result.fullName][1]
      return self.__generateGlobalHessianForUnion(current, wrt)
    else:
      raise ValueError(f"energy.__generateHessianThroughRecursion: operator {current.operator} is not supported in path dict.")



  def generateHessianAndGradient(self, wrt: List[attribute]) -> None:
    differentiater = autodiff()
    # generate the symbolic code for gradient and hessian
    self.__generateGradientThroughPathDict(wrt, differentiater)
    assert self.__gradient is not None
    # print("----------------------------------------------------------------------------")
    # print("Gradient actual")
    # # print(str(self.__gradient))
    # print("Checking the gradient")
    # print(self.__gradient.compute().value.get()[:self.__gradient.size])
    # print("Checking energy")
    # print(self.__energy.compute().value.get()[0])
    # print("----------------------------------------------------------------------------")
    # exit(0)
    if not self.__gradient_only:
      # dont generate hessian if we only need gradient
      self.__generateHessianThroughPathDict(wrt, differentiater)
      # print("Hessian generated")
      assert self.__hessian is not None, "yasps.energy.generateHessianAndGradient: The hessian is not computed yet. Please call generateHessianAndGradient first."
      # print("----------------------------------------------------------------------------")
      # print("Computed hessian check")
      # computed_hessian = self.__hessian.compute().value.get()
      # print(computed_hessian[:(self.__hessian.rows * self.__hessian.cols)].reshape((self.__hessian.rows, self.__hessian.cols)))
      # print("----------------------------------------------------------------------------")
      # exit()




  @timed("energy.computeHessianAndGradient")
  def computeHessianAndGradient(self, gradient_array: gpuarray.GPUArray, hessian_blocks: gpuarray.GPUArray, diagonal: gpuarray.GPUArray):
    if self.__gradient is None:
      # the gradient is 0, return the 0 array
      return
    if self.__hessian is None and not self.__gradient_only:
      raise ValueError("yasps.energy.computeHessianAndGradient: The hessian is not computed yet. Please call generateHessianAndGradient first.")
    # assert self.__hessian is not None
    if self.__hessianAndGradientKernel is None:
      from yasps.hessianAndGradientKernel import hessianAndGradientKernel
      # we need to put the gradient and the hessian together
      # we know the graidient sizes square is the hessian size
      merged_hessian_and_gradient = []
      if not self.__gradient_only: # we need the hessian
        assert self.__hessian is not None
        for i in range(self.__hessian.rows):
          for j in range(self.__hessian.cols):
            merged_hessian_and_gradient.append(self.__hessian[i, j])
      for i in range(self.__gradient.size):
        merged_hessian_and_gradient.append(self.__gradient[i])
      # create the attribute for the merged hessian and gradient
      merged_hessian_rows: int
      if self.__gradient_only:
        merged_hessian_rows = 1
      else:
        assert self.__hessian is not None
        merged_hessian_rows = self.__hessian.rows + 1
      self.__merged_hessian_and_gradient_attribute = self.__energy.correspondance.addAttribute(f'hessian_and_gradient_d2_{self.__energy.fullName}_d2_{"__".join([x.fullName for x in self.__wrt])}', computed_attribute = attribute.to_array(merged_hessian_and_gradient, rows = merged_hessian_rows, cols = self.__gradient.size))


      from yasps.codeGenerator import codeGenerator
      codegen: codeGenerator = codeGenerator(self.__merged_hessian_and_gradient_attribute)
      codegen.generateCode() # this will give us the local kernel strings
      # now add the global kernel
      self.__hessianAndGradientKernel = hessianAndGradientKernel(self.__merged_hessian_and_gradient_attribute, self.__project_entire_hessian, self.__projection_method, self.__gradient_only)
    assert self.__hessianAndGradientKernel is not None
    assert self.__indices_kernel is not None
    self.__hessianAndGradientKernel.generateKernel(self.__indices_kernel.outputUniqueGradientSizesCPU.tolist(), self.__wrt)

    # print(f"There are {len(self.__intermediate_compute_pairs)} intermediate attributes")
    # make sure that we also compute the intermediate values
    for name, value in self.__intermediate_compute_pairs.items():
      value[0].compute()
      value[1].updateValue(value[0].value)
      # print(f"Computed intermediate attribute with name {name}, hash: {value[0].hash}")

    # assertion here
    # assert self.__hessianAndGradientKernel is not None
    assert self.__merged_hessian_and_gradient_attribute is not None
    # assert self.__indices_kernel is not None

    # after we allocated, we invoke the kernel
    counts_gpu = [x.children_primitive_counts_gpu for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelPrimitiveUnions] # get the children counts
    # print(f"counts gpu: {[x.get() for x in counts_gpu]}")
    arguments: List[gpuarray.GPUArray] = [x.value for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelDatas] + [x.value for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelConnectivity] + [x.compressedRows for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelConnectivity if x.dimension == 0] + counts_gpu

    self.__hessianAndGradientKernel.compute(arguments, self.__indices_kernel, self.__block_indices_gpu, gradient_array, hessian_blocks, diagonal)
    return self

  def __hash__(self) -> int:
    return self.__energy.hash

  @property
  def hash(self) -> int:
    return self.__hash__()

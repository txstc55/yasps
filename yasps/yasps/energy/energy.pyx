# cython: language_level=3
from __future__ import annotations
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import Optional, List, Tuple, Set, Dict
from typing import TYPE_CHECKING
from yasps.attribute import attribute
from yasps.autodiff import autodiff
from yasps.helper import extract_block
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
    self.__roots: List[attribute] = []
    self.__roots, self.__paths = self.getRoots(energy, [energy]) # get the root attributes
    self.__wrt: List[attribute] = [] # an energy can be minimized for different attributes, for safety let's save all histories
    self.__indices_cpu: np.ndarray = np.array([]) # save the indices on cpu
    self.__block_indices_gpu: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # save the block indices, this is for hessian accumulation
    self.__gradient_sizes_cpu: List[int] = [] # save the sizes of the gradient, this is to determine for the gradient, how large it is for each segment
    self.__gradient_sizes_gpu: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # save the sizes of the gradient
    self.__hessian: Optional[attribute] = None # save the hessian for each wrt input
    self.__gradient: Optional[attribute] = None # save the gradient for each wrt input
    self.__hessianAndGradientKernel: Optional[hessianAndGradientKernel] = None
    self.__hessian_blocks_where_to_check: gpuarray.GPUArray = gpuarray.to_gpu(np.array([])) # we have a flattened array which stores the blocks. The blocks are sorted by dimensions. We need to know which block we are in for each smaller blocks in the hessian
    self.__merged_hessian_and_gradient_attribute: Optional[attribute] = None
    self.__project_entire_hessian: bool = False
    self.__projection_method = projection_method # 0 for no projection, 1 for absolute, 2 for max(0, val)
    self.__save_intermediate = save_intermediate # save intermediate gradient and hessian result
    self.__intermediate_compute_pairs: Dict[str, Tuple[attribute, attribute]] = {} # save the intermediate compute pairs
    self.__gradient_only: bool = gradient_only # only compute gradient
    self.__indices_kernel: Optional[gradientIndicesKernel] = None


  # @property
  # def roots(self) -> List[attribute]:
  #   return self.__roots

  @property
  def gradient_sizes_cpu(self) -> List[int]:
    return self.__gradient_sizes_cpu

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
      elif current.opeartor == UNION:
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
          childrenRoots, childrenPaths = self.getRoots(child, [])
          trueRoots += childrenRoots
          for childrenPath in childrenPaths:
            allPaths.append(parentPath + [root] + childrenPath)
      elif root.operator == DATA:
        trueRoots.append(root)
        allPaths.append(parentPath + [root])
      else:
        raise ValueError(f"energy.getRoots: operator {root.operator} is not supported.")
    return trueRoots, allPaths


  @timed("energy.getSparseIndices")
  def getSparseIndices(self, wrt: List[attribute], wrt_start_indices: List[int]):
    self.__wrt = wrt
    # the wrt_start_indices, size and if type is primitive
    wrtStartIndicesAndSize: Dict[int, Tuple[int, int, bool]] = {x.hash: (wrt_start_indices[i], x.size, x.correspondance.type == "primitive") for i, x in enumerate(wrt)} # this maps from all the data being used, to where it should start in the gradient, its size and if it is a primitive (we can optimize for mesh or scene primitive, which doesnt really have any number of instances)
    # we have the path, now determine which path to use since we have the wrt
    usedPaths: List[List[attribute]] = []
    for path in self.__paths:
      if path[-1] in wrt:
        usedPaths.append(path)
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
      # print("Path dict check")
      # for key in pathDict.keys():
      #   print("Parent:", key.fullName)
      #   print("Children:", [child.fullName for child in pathDict[key]])
      # create a gradientIndicesKernel
      self.__indices_kernel = gradientIndicesKernel(pathDict, wrt, wrt_start_indices, self.__energy)

    assert self.__indices_kernel is not None
    self.__indices_kernel.computeIndices(wrt_start_indices) # actually compute the indices
    # now we recursively go over each path
    # we first recursively duplicate the path with rows
    duplicatedPaths = []
    for path in usedPaths:
      duplicatedPaths += self.__duplicatePathForOperation(path)

    self.__indices_cpu = self.__indices_kernel.outputIndices.get()
    self.__gradient_sizes_cpu = [wrtStartIndicesAndSize[x[-1][0]][1] for x in duplicatedPaths]
    return self.__indices_cpu


  def __duplicatePathForOperation(self, path: List[attribute]) -> List[List[Tuple[int, int]]]:
    # we duplicate the paths so that join operations are expanded
    # and we can later on use it to get the indices
    from yasps.attribute import DATA, JOIN
    # if it is just data, we return the hash and -1
    # if it is a row operator
    # instead we return the node hash and row index
    # if it is something else we return 0 and -2
    if path[0].operator == DATA:
      return [[(path[0].hash, -1)]]
    elif path[0].operator == JOIN:
      childrenPaths = self.__duplicatePathForOperation(path[1:])
      duplicatedPaths = []
      for i in range(path[0].through.dimension):
        # duplicate each path
        for childrenPath in childrenPaths:
          duplicatedPaths.append([(path[0].hash, i)] + childrenPath)
      return duplicatedPaths
    else:
      # we are at top level
      childrenPaths = self.__duplicatePathForOperation(path[1:])
      return childrenPaths

  def __generateGradient(self, wrt: List[attribute], differentiater: autodiff) -> None:
    from yasps.attribute import JOIN, FLOAT
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
        # print("generate gradient")
        # print(f'At path: {", ".join([x.fullName for x in path])}')
        # do it in reverse order
        for i in range(len(path) - 2, -1, -1):
          # # we will compute the jacobian for each neighboring nodes
          lead_node = path[i]
          follow_node = path[i+1]
          if lead_node.operator == JOIN:
            # we need to differentiate a join's children wrt to the next node
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
          # now that we have the child jacobian, we need to differentiate the join wrt to the child
          else:
            # it's a normal node, which is the energy
            # add the differentiation wrt to the next node
            diff_att_name = f'd_{lead_node.fullName}_d_{follow_node.fullName}'
            if diff_att_name not in lead_node.correspondance.attributes:
              diff_att = differentiater.diff(lead_node, follow_node)
              if diff_att_name not in lead_node.correspondance.attributes:
                lead_node.correspondance.addAttribute(diff_att_name, computed_attribute = diff_att)
        # ok now we have done the differentiation from node to node
        # we need to assemble the actual jacobian matrix for the join operator
        if len(path) == 2:
          # there is no join operation
          # the jacobian should be directly used
          derivative = path[0].correspondance[f'd_{path[0].fullName}_d_{path[1].fullName}']
          gradients.append(derivative)
        else:
          # we go from bottom up
          # and for the first one, we will deal with it separately
          for i in range(len(path) - 2, 0, -1):
            lead_att = path[i] # we know this has to be a join node
            next_att = path[i+1]
            data_node = path[-1]
            neighboring_jacobian = lead_att.children[0].correspondance[f'd_{lead_att.children[0].fullName}_d_{next_att.fullName}']
            multiplied_jacobian: attribute
            if i == len(path) - 2:
              # just return the neighboring jacobian as the multiplied
              multiplied_jacobian = neighboring_jacobian
            else:
              # now we get the jacobian from the next att to the data node
              join_path = path[i + 1: -1]
              join_path_str = "_d_".join([x.fullName for x in join_path])
              next_att_data_jacobian = next_att.correspondance[f'd_{join_path_str}_d_{data_node.fullName}']
              multiplied_jacobian = neighboring_jacobian.mul_explicit(next_att_data_jacobian)
            # now we determine if it is needed to actually join the entire thing by checking if everything has a correspondance and not a float
            skipped_indices: List[int] = []
            # TODO: there are elements that are repeated
            # we can reduce the amout of join by taking out those elements
            # and only join once
            for j in range(multiplied_jacobian.size):
              if multiplied_jacobian[j].correspondance is None or (multiplied_jacobian[j].correspondance.type != "primitive") or multiplied_jacobian[j].operator == FLOAT:
                skipped_indices.append(j)

            multiplied_jacobian_materialized: attribute = None
            # multiplied_jacobian_materialized_name: str = ""
            if self.__save_intermediate:
              # ok we want to materialize the jacobian to a data variable
              # and we can use it later
              materialized_jacobian_list = []
              for j in range(multiplied_jacobian.size):
                if j not in skipped_indices:
                  materialized_jacobian_list.append(multiplied_jacobian[j])
              if len(materialized_jacobian_list) > 0:
                materialized_jacobian = attribute.to_array(materialized_jacobian_list, rows = len(materialized_jacobian_list), cols = 1)
                # add the name
                included_paths = [lead_att.children[0]] + path[i + 1: -1]
                included_path_str = "_d_".join([x.fullName for x in included_paths])
                attribute_name = f"d_{included_path_str}_d_{data_node.fullName}_lead_materialized"
                # multiplied_jacobian_materialized_name = attribute_name # save the name for later
                if attribute_name not in materialized_jacobian.correspondance.attributes:
                  multiplied_jacobian_materialized = multiplied_jacobian.correspondance.addAttribute(attribute_name, rows = len(materialized_jacobian_list), cols = 1)
                  self.__intermediate_compute_pairs[attribute_name] = (multiplied_jacobian, multiplied_jacobian_materialized)


            # ok now we need to create new attributes for the non skipped indices
            included_paths = [lead_att.children[0]] + path[i + 1: -1]
            included_path_str = "_d_".join([x.fullName for x in included_paths])
            # print("Lead attribute fullname", lead_att.fullName)
            # print("Included path str:", included_path_str)
            sequential_count = 0 # for accessing the materialized jacobian sequentially
            for j in range(multiplied_jacobian.size):
              if j in skipped_indices:
                continue
              # we need to create a new attribute for the ith element through join
              if f"d_{included_path_str}_d_{data_node.fullName}_{j}" not in lead_att.correspondance.attributes:
                if not self.__save_intermediate:
                  lead_att.correspondance.addAttribute(f"d_{included_path_str}_d_{data_node.fullName}_{j}", through = lead_att.through, source = multiplied_jacobian[j]) # we add the new joining attribute and use it later on
                else:
                  assert multiplied_jacobian_materialized is not None
                  lead_att.correspondance.addAttribute(f"d_{included_path_str}_d_{data_node.fullName}_{j}", through = lead_att.through, source = multiplied_jacobian_materialized[sequential_count])
                  sequential_count += 1
            # now we have a new join attribute which is the jacobian
            new_jacobian_children = [attribute(float_value = 0.0) for _ in range(multiplied_jacobian.size * lead_att.through.dimension * lead_att.through.dimension)]
            # ok, if the child jacobian is m by n, then the new jacobian has k by k blocks, each block is m by n
            # and only the diagonal blocks will have nonzero values
            # the jacobian is corresponding to per data
            # so there's no need to reorient them
            for j in range(multiplied_jacobian.size):
              m = multiplied_jacobian.rows # rows
              n = multiplied_jacobian.cols # cols
              k = lead_att.through.dimension # how many times we need to repeat the jacobian through join
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
          # print(f"Neighboring jacobian size: {neighboring_jacobian.rows} x {neighboring_jacobian.cols}")
          # print(f"Neighboring jacobian: {str(neighboring_jacobian)}")
          # print(f"Last data jacobian size: {last_data_jacobian.rows} x {last_data_jacobian.cols}")
          # print(f"Last data jacobian: {str(last_data_jacobian)}")
          final_jacobian = neighboring_jacobian.mul_explicit(last_data_jacobian)
          gradients.append(final_jacobian)
      gradients_assembled_children = []
      for gradient in gradients:
        for i in range(gradient.size):
          gradients_assembled_children.append(gradient[i])
      gradient_assembled_attribute = attribute.to_array(gradients_assembled_children, rows = self.__energy.size, cols = len(gradients_assembled_children))
      if gradient_assembled_attribute.name != "":
        self.__gradient = gradient_assembled_attribute
        return
      elif f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}' not in self.__energy.correspondance.attributes:
        self.__energy.correspondance.addAttribute(f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}', computed_attribute = gradient_assembled_attribute)
      self.__gradient = self.__energy.correspondance[f'd_{self.__energy.fullName}_d_{"__".join([x.fullName for x in wrt])}']

  def __generateHessianForParts(self, differentiater: autodiff, currentPaths: List[List[attribute]]) -> None:
    from yasps.attribute import DATA, JOIN, FLOAT
    # we need to recursively generate the hessian for parts
    # the input currentPaths all have the same first element
    # then we just need to find the hessian for this part of the path
    # here's something we need to know
    # if i have the currentpath, the next node is guaranteed to present in batches
    # which means for the next node, it's guranteed to be like [a, a, b, b, b, c]
    # same node will always be batched together

    # we first check if H of f(g) is already computed
    lead_node: attribute = currentPaths[0][0]
    child_att: attribute
    # differentiate between the join canse and the top node, which is the energy
    if lead_node.operator != JOIN:
      child_att = lead_node
    else:
      child_att = lead_node.children[0]
    child_att_full_name: str = child_att.fullName
    # we first check if the differentiation has been done before
    all_datas = [x[-1] for x in currentPaths]
    h_g_name = f'd2_{child_att.fullName}_d_{"__".join([x.fullName for x in all_datas])}'
    if lead_node.operator != JOIN and h_g_name in child_att.correspondance.attributes:
      return
    # we need to get the hessian from child_att to data node
    # first, we construct the Hessian of f(g(x)) wrt g(x)
    # which is essentially the hessian from child_att to all of its children
    h_f_g_size: int = 0
    allChildren: List[attribute] = [] # all the following nodes
    # we first check and join all the g(x) nodes
    for path in currentPaths:
      follow_node: attribute = path[1]
      if follow_node not in allChildren:
        allChildren.append(follow_node)
    for follow_node in allChildren:
      h_f_g_size += follow_node.size # we just need to know for each follow node, its size and we can accumulate the hessian size
    h_f_g_children: List[attribute] = [attribute(float_value = 0.0) for _ in range(h_f_g_size * h_f_g_size * child_att.size)] # a hessian for each element of the child att
    row_offset: int = 0
    for i in range(len(allChildren)):
      follow_node: attribute = allChildren[i]
      follow_node_full_name: str = follow_node.fullName
      diff_att_name: str = f'd_{child_att_full_name}_d_{follow_node_full_name}' # ok so this is the jacobian of the first node wrt one of its children
      previous_children = allChildren[: i]
      col_offset = sum([x.size for x in previous_children]) # whenever we go to a new child, we need to reset the col_offset
      for j in range(i, len(allChildren)):
        diff_target_node: attribute = allChildren[j]
        diff_target_node_full_name: str = diff_target_node.fullName
        d2_name: str = f'd_{diff_att_name}_d_{diff_target_node_full_name}' # we have computed this jacobian or hessian
        d2_attribute: attribute = child_att.correspondance.attributes[d2_name]

        # let's say follow_node has size of K, and diff_target_node has size of M
        # and our lead_node(or child node, whatever, it's the first node in the current path)
        # has size of N, then this single_att_d2_size is M * N * K (might need to check the order later)
        single_att_d2_size: int = d2_attribute.size // child_att.size #
        assert single_att_d2_size * child_att.size == d2_attribute.size # make sure the size is integer division
        for l in range(child_att.size):
          d2_attribute_partial: List[attribute] = [d2_attribute[k] for k in range(l * single_att_d2_size, (l + 1) * single_att_d2_size)]
          # the block has size follow_node.size * diff_target_node.size
          # we need to put this block into the hessian
          # the offset are row_offset and col_offset
          # print(f"Row and column offset: {row_offset}, {col_offset}")
          for m in range(follow_node.size):
            for n in range(diff_target_node.size):
              h_f_g_children[l * h_f_g_size * h_f_g_size + (row_offset + m) * h_f_g_size + col_offset + n] = d2_attribute_partial[m * diff_target_node.size + n]
              # make it symmetric
              h_f_g_children[l * h_f_g_size * h_f_g_size + (col_offset + n) * h_f_g_size + row_offset + m] = d2_attribute_partial[m * diff_target_node.size + n]
        col_offset += diff_target_node.size
      row_offset += follow_node.size
    # ok now we need to assemble J of g(x)
    # we have previously computed them supposedly
    j_g_x_col_size: int = 0
    j_g_x_col_sizes: List[int] = []
    for path in currentPaths:
      next_node = path[1]
      if next_node.operator == DATA:
        j_g_x_col_size += next_node.size
        j_g_x_col_sizes.append(next_node.size)
      else:
        data_node = path[-1]
        included_paths: List[attribute] = path[1: -1]
        included_path_str: str = "_d_".join([x.fullName for x in included_paths])
        next_jacobian_name: str = f"d_{included_path_str}_d_{data_node.fullName}"
        next_jacobian: attribute = next_node.correspondance.attributes[next_jacobian_name]
        j_g_x_col_size += next_jacobian.cols
        j_g_x_col_sizes.append(next_jacobian.cols)
    j_g_x_children: List[attribute] = [attribute(float_value = 0.0) for _ in range(h_f_g_size * j_g_x_col_size)] # allocate space for the jacobian
    passed_next_node: List[attribute] = [currentPaths[0][1]] # check if we already passed this node
    row_offset: int = 0
    col_offset: int = 0
    for path in currentPaths:
      next_node: attribute = path[1]
      if next_node not in passed_next_node:
        row_offset += passed_next_node[-1].size # we moved to the next node, row_offset updates
        passed_next_node.append(next_node)
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
        included_paths = path[1: -1]
        data_node = path[-1]
        included_path_str = "_d_".join([x.fullName for x in included_paths])
        next_jacobian_name = f"d_{included_path_str}_d_{data_node.fullName}"
        # print(next_node.correspondance.attributes.keys())
        next_jacobian = next_node.correspondance.attributes[next_jacobian_name]
        for i in range(next_jacobian.rows):
          for j in range(next_jacobian.cols):
            index = (row_offset + i) * j_g_x_col_size + col_offset + j
            j_g_x_children[index] = next_jacobian[i * next_jacobian.cols + j]
        col_offset += next_jacobian.cols
    # now we can compute the first part of the hessian by doing the multiplication
    # but we defer it to last as we may need to do pd projection with some of the hessian


    # we've finished the first part of the hessian now
    # now we need to compute the second part
    # the second part is the sum of k
    # df/dg_k of g(x) times the hessian of g_k(x)
    # df/dg_k of g(x) we should have already computed
    all_df_dg: List[attribute] = [] # for each next node
    for follow_node in allChildren:
      follow_node_full_name = follow_node.fullName
      # we know this is definitely computed
      diff_att_name: str = f'd_{child_att_full_name}_d_{follow_node_full_name}'
      diff_att: attribute = child_att.correspondance.attributes[diff_att_name]
      all_df_dg.append(diff_att)

    # now, for each next node, we will need to have a permutation matrix
    # this is because when we have join, we cannot simply have on diagonal H0, H1, H2
    # instead, because we store the index by attribute, we need to put the same attribute at the same place
    # first, we will need to know for each of the follow node, their follow node and the size of the attribute
    attribute_sizes: List[List[int]] = [[]]
    corresponding_data_attributes: List[List[attribute]] = [[]]
    last_checked_attribute: attribute = currentPaths[0][1]
    for i in range(len(currentPaths)):
      path: List[attribute] = currentPaths[i]
      grandchildren_path: List[attribute] = path[2:]
      attribute_size = 1
      for item in grandchildren_path:
        if item.operator == JOIN:
          attribute_size *= item.through.dimension
        else:
          attribute_size *= item.size
      if path[1] != last_checked_attribute:
        last_checked_attribute = path[1]
        attribute_sizes.append([])
        corresponding_data_attributes.append([])
      attribute_sizes[-1].append(attribute_size)
      corresponding_data_attributes[-1].append(path[-1])
    # ok now we have for each next node, all the attributes and the sizes
    # we should know how to reorient them when we get them
    # now, for each of the next node, we will get the hessian
    h_g_full_mat_children = [[attribute(float_value = 0.0) for _ in range(j_g_x_col_size * j_g_x_col_size)] for _ in range(child_att.size)]
    mat_offset = 0 # the offset for the diagonal block
    for i in range(len(allChildren)):
      follow_node: attribute = allChildren[i]
      if follow_node.operator == DATA:
        # great, the hessian is 0, all we need to do is set the offset
        mat_offset += follow_node.size
      else:
        # first of all, we check if the hessian exists
        data_attributes: List[attribute] = corresponding_data_attributes[i]
        h_g_name = f'd2_{follow_node.children[0].fullName}_d_{"__".join([x.fullName for x in data_attributes])}'
        if h_g_name not in follow_node.children[0].correspondance.attributes:
          # ok here we finally do the recursion part
          # first we select all the paths that are relevant
          relevant_paths: List[List[attribute]] = []
          for path in currentPaths:
            if path[1] == follow_node:
              relevant_paths.append(path[1:])
          self.__generateHessianForParts(autodiff, relevant_paths)
        h_g: attribute = follow_node.children[0].correspondance.attributes[h_g_name] # h_g has size j_g_x_col_size * j_g_x_col_size * (number of elements for the follow node's child 0, because we have a hessian for each of the element)
        follow_node_child_size = follow_node.children[0].size # the number of blocks in h_g
        follow_node_dimension = follow_node.through.dimension # how many children is joined
        h_g_true_size = h_g.size // follow_node_child_size # for each h_g, what's the size
        assert h_g_true_size * follow_node_child_size == h_g.size # make sure the division is integer
        h_g_true_row_size = sum(attribute_sizes[i]) # how many rows in this hessian
        # let's do an assert here
        assert h_g_true_size == sum(attribute_sizes[i]) * sum(attribute_sizes[i])
        # now we check how many elements of h_g we actually need, because it's possible that many of them are constants
        skipped_indices = []
        for j in range(h_g.size):
          if h_g[j].operator == FLOAT:
            skipped_indices.append(j)

        second_term_hessian_before_join_materialized: attribute = None
        # multiplied_jacobian_materialized_name: str = ""
        if self.__save_intermediate:
          # ok we want to materialize the jacobian to a data variable
          # and we can use it later
          materialized_second_term_list = []
          for j in range(h_g.size):
            if j not in skipped_indices:
              materialized_second_term_list.append(h_g[j])
          if len(materialized_second_term_list) > 0:
            materialized_second_term = attribute.to_array(materialized_second_term_list, rows = len(materialized_second_term_list), cols = 1)
            # add the name
            attribute_name = f"{h_g_name}_nonconstant_materialized"
            # multiplied_jacobian_materialized_name = attribute_name # save the name for later
            if attribute_name not in h_g.correspondance.attributes:
              second_term_hessian_before_join_materialized = h_g.correspondance.addAttribute(attribute_name, rows = len(materialized_second_term_list), cols = 1)
              self.__intermediate_compute_pairs[attribute_name] = (materialized_second_term, second_term_hessian_before_join_materialized)

        sequential_count = 0 # for sequentially adding materialized attribute
        for j in range(h_g.size):
          if j not in skipped_indices:
            # we add the attribute
            if f'{h_g_name}_{j}' not in follow_node.correspondance.attributes:
              if not self.__save_intermediate:
                follow_node.correspondance.addAttribute(f'{h_g_name}_{j}', through = follow_node.through, source = h_g[j])
              else:
                assert second_term_hessian_before_join_materialized is not None
                follow_node.correspondance.addAttribute(f'{h_g_name}_{j}', through = follow_node.through, source = second_term_hessian_before_join_materialized[sequential_count])
                sequential_count += 1
        # now we actually need to accumulate the new matrix, which is going to be the size of
        # h_g_true_size * follow_node_child_size * follow_node_dimension
        expanded_second_term_hessians = [attribute(float_value = 0.0) for _ in range(h_g.size * follow_node_dimension)]
        for j in range(h_g.size):
          if j in skipped_indices:
            # we know exactly it is a float, put it in corresponding place
            for k in range(follow_node_dimension):
              expanded_second_term_hessians[k * h_g.size + j] = h_g[j]
          else:
            # we have the accumulated attribute, put it in
            for k in range(follow_node_dimension):
              expanded_second_term_hessians[k * h_g.size + j] = follow_node.correspondance.attributes[f'{h_g_name}_{j}'][k]
        # we have assembled the expanded
        # now, do the following for each of the element in child_att
        for j in range(child_att.size):
          # get the correct row elements of df_dg
          dfj_dg = [all_df_dg[i][k] for k in range(j * follow_node.size, (j + 1) * follow_node.size)]
          compressed_second_term_hessians: List[attribute] = [attribute(float_value = 0.0) for _ in range(h_g_true_size * follow_node_dimension)] # because for the child, let's say i have an attribute, that is accumulated 4 times, and the attribute itself has dimension 3, with the final data size of 8, then because each of the 3 elements are corresponding to the same 8 data attributes, we actually can accumulate them together for the 3 hessians, multiplied by the correct df_dg
          for k in range(follow_node_child_size * follow_node_dimension):
            nth_joined_element = k // follow_node_child_size # which joined index it is
            # nth_child_element = k % follow_node_child_size # which child it is
            selected_hessian = expanded_second_term_hessians[k * h_g_true_size : (k + 1) * h_g_true_size] # extract the hessian block
            selected_hessian = [dfj_dg[k] * selected_hessian[m] for m in range(len(selected_hessian))] # multiply by the correct dfj_dg
            # now we add it back
            for m in range(h_g_true_size):
              compressed_second_term_hessians[nth_joined_element * h_g_true_size + m] += selected_hessian[m]
          # ok now we reorient the children attributes
          # here's what we need to do
          # we have N hessians, N is the join dimension
          # and in each of the hessian, the blocks are sorted by data
          # now we need to put the same data in the same block
          # we have the mat_offset, which is the offset on both the row and col
          current_children_sizes: List[int] = attribute_sizes[i]
          for k in range(len(current_children_sizes)):
            block_row_size = current_children_sizes[k]
            block_row_start = sum(current_children_sizes[:k])
            for m in range(k, len(current_children_sizes)):
              block_col_size = current_children_sizes[m]
              block_col_start = sum(current_children_sizes[:m])
              for n in range(follow_node_dimension): # for each join
                # first we get the correct block
                block = extract_block(compressed_second_term_hessians, h_g_true_row_size * follow_node_dimension, h_g_true_row_size, h_g_true_row_size * n + block_row_start, block_col_start, block_row_size, block_col_size)
                # ok now we have the block, we need to know where to put it in the final matrix
                block_row_start_in_final_matrix = mat_offset + block_row_start * follow_node_dimension + block_row_size * n
                block_col_start_in_final_matrix = mat_offset + block_col_start * follow_node_dimension + block_col_size * n
                # now we put it in
                for p in range(block_row_size):
                  for q in range(block_col_size):
                    h_g_full_mat_children[j][(block_row_start_in_final_matrix + p) * j_g_x_col_size + block_col_start_in_final_matrix + q] = block[p * block_col_size + q]
        # now we set the mat offset
        mat_offset += h_g_true_row_size * follow_node_dimension
    # now for all h_g_full_mat, we need to make it symmetric
    for i in range(len(h_g_full_mat_children)):
      for j in range(j_g_x_col_size):
        for k in range(j):
          index = j * j_g_x_col_size + k
          transpose_index = k * j_g_x_col_size + j
          h_g_full_mat_children[i][index] = h_g_full_mat_children[i][transpose_index]
    # ok so now h_g_full_mat is constructed, we need to flatten it and give it the attribute name
    all_datas = [x[-1] for x in currentPaths]
    h_g_name = f'd2_{child_att.fullName}_d_{"__".join([x.fullName for x in all_datas])}'
    h_g_full_mats = [attribute.to_array(x, rows = j_g_x_col_size, cols = j_g_x_col_size) for x in h_g_full_mat_children]
    h_g_full_mat_children: List[attribute] = []
    # here we compute the first part of the hessian
    second_term_is_zero = False
    j_g_x = attribute.to_array(j_g_x_children, rows = h_f_g_size, cols = j_g_x_col_size)
    multiplication_result = [] # the multiplied out first term, we do this for each element of child_att
    for i in range(child_att.size):
      hessian_children = h_f_g_children[i * h_f_g_size * h_f_g_size: (i + 1) * h_f_g_size * h_f_g_size]
      hessian = attribute.to_array(hessian_children, rows = h_f_g_size, cols = h_f_g_size)
      # ok determine if we want to do hessian projection
      if lead_node.operator != JOIN:
        if h_g_full_mats[i].isZero > 0: # check if the second part is just zero matrix
          second_term_is_zero = True
          # the second term is zero, we can do hessian projection
          hessian = hessian.spd(self.__projection_method)
          # print("We can project the inner hessian")
      mul1 = hessian.mul_explicit(j_g_x)
      mul2 = j_g_x.transpose().mul_explicit(mul1)
      multiplication_result.append(mul2)
    for i in range(child_att.size):
      h_g_full_mats[i] = h_g_full_mats[i].add_explicit(multiplication_result[i])
      if (lead_node.operator != JOIN) and (not second_term_is_zero) and (self.__projection_method != 0):
        # # we need to project the whole matrix
        # h_g_full_mats[i] = h_g_full_mats[i].spd(0)
        # print("We project the entire hessian")
        self.__project_entire_hessian = True
      for j in range(h_g_full_mats[i].size):
        h_g_full_mat_children.append(h_g_full_mats[i][j])
    if lead_node.operator == JOIN:
      if h_g_name not in child_att.correspondance.attributes:
        # we only add the hessian if it is a join node
        # because for a non join node, we may have done projection
        child_att.correspondance.addAttribute(h_g_name, computed_attribute = attribute.to_array(h_g_full_mat_children, rows = h_g_full_mats[0].rows * child_att.size, cols = h_g_full_mats[0].cols))
    else:
      if f"{h_g_name}_projected" not in child_att.correspondance.attributes:
        child_att.correspondance.addAttribute(f"{h_g_name}_projected", computed_attribute = attribute.to_array(h_g_full_mat_children, rows = h_g_full_mats[0].rows * child_att.size, cols = h_g_full_mats[0].cols))
      # finally we assign the hessian
      self.__hessian = lead_node.correspondance.attributes[f"{h_g_name}_projected"]
      # if self.__hessian is not None:
        # print(f"Final hessian correspondance: {self.__hessian.correspondance}")

    # # this is our hessian now
    # if lead_node.operator != JOIN:
    #   self.__hessian = attribute.to_array(h_g_full_mat_children, rows = h_g_full_mats[0].rows * child_att.size, cols = h_g_full_mats[0].cols)




  def __generateHessian(self, wrt: List[attribute], differentiater: autodiff) -> None:
    from yasps.attribute import JOIN
    # first we check which path we need
    filteredPath: List[List[attribute]] = []
    for path in self.__paths:
      if path[-1] in wrt:
        filteredPath.append(path)
    # check if hessian is already generated
    all_datas = [x[-1] for x in filteredPath]
    h_g_name = f'd2_{self.__energy.fullName}_d_{"__".join([x.fullName for x in all_datas])}'

    if f'{h_g_name}_projected' in self.__energy.correspondance.attributes:
      self.__hessian = self.__energy.correspondance.attributes[f'{h_g_name}_projected']
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
          if lead_node.operator == JOIN:
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
    self.__generateHessianForParts(differentiater, filteredPath)


  def generateHessianAndGradient(self, wrt: List[attribute]) -> None:
    differentiater = autodiff()
    # generate the symbolic code for gradient and hessian
    self.__generateGradient(wrt, differentiater)
    if not self.__gradient_only:
      # dont generate hessian if we only need gradient
      self.__generateHessian(wrt, differentiater)




  @timed("energy.computeHessianAndGradient")
  def computeHessianAndGradient(self, gradient_array: gpuarray.GPUArray, hessian_blocks: gpuarray.GPUArray, diagonal: gpuarray.GPUArray):
    if self.__gradient is None:
      # the gradient is 0, return the 0 array
      return
    if self.__hessian is None and not self.__gradient_only:
      raise ValueError("yasps.energy.computeHessianAndGradient: The hessian is not computed yet. Please call generateHessianAndGradient first.")
    assert self.__hessian is not None
    if self.__hessianAndGradientKernel is None:
      from yasps.hessianAndGradientKernel import hessianAndGradientKernel
      # we need to put the gradient and the hessian together
      # we know the graidient sizes square is the hessian size
      merged_hessian_and_gradient = []
      if not self.__gradient_only: # we need the hessian
        for i in range(self.__hessian.rows):
          for j in range(self.__hessian.cols):
            merged_hessian_and_gradient.append(self.__hessian[i, j])
      for i in range(self.__hessian.rows):
        merged_hessian_and_gradient.append(self.__gradient[i])
      # create the attribute for the merged hessian and gradient
      self.__merged_hessian_and_gradient_attribute = self.__energy.correspondance.addAttribute(f'hessian_and_gradient_d2_{self.__energy.fullName}_d2_{"__".join([x.fullName for x in self.__wrt])}', computed_attribute = attribute.to_array(merged_hessian_and_gradient, rows = self.__hessian.rows + 1, cols = self.__hessian.cols))

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
    for _, value in self.__intermediate_compute_pairs.items():
      value[0].compute()
      value[1].updateValue(value[0].value)

    # assertion here
    assert self.__hessianAndGradientKernel is not None
    assert self.__merged_hessian_and_gradient_attribute is not None
    assert self.__indices_kernel is not None

    # after we allocated, we invoke the kernel
    arguments: List[gpuarray.GPUArray] = [x.value for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelDatas] + [x.value for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelConnectivity] + [x.compressedRows for x in self.__merged_hessian_and_gradient_attribute.deviceKernel.kernelConnectivity if x.dimension == 0]

    self.__hessianAndGradientKernel.compute(arguments, self.__indices_kernel, self.__block_indices_gpu, gradient_array, hessian_blocks, diagonal)
    return self

  def __hash__(self) -> int:
    return self.__energy.hash

  @property
  def hash(self) -> int:
    return self.__hash__()

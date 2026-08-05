from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math

from yasps.attribute import attribute, JOIN, UNION, DATA, CONSTANT
from yasps.autodiff import autodiff
from yasps.gradient import gradient
from yasps.hessian import hessian
from yasps.secondOrderJacobian import secondOrderJacobian
from yasps.path import path
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.secondOrderJacobianIndicesKernel import secondOrderJacobianIndicesKernel
from yasps.placementReorderKernel import placementReorderKernel

class differentiator:
  def __init__(self):
    self.__source: Optional[attribute] = None
    self.__path_dict: Dict[attribute, List[attribute]] = {}
    self.__unioned_child_to_its_children: Dict[attribute, List[attribute]] = {}
    self.__save_intermediate: bool = False
    self.__intermediate_compute_pairs: Dict[str, Tuple[attribute, attribute]] = {}
    self.__gradient_only: bool = False
    self.__projection_method: int = 1
    self.__global_jacobian: Optional[attribute] = None
    self.__global_inner_hessian: Optional[attribute] = None
    self.__project_entire_hessian: bool = False
    self.__separate_hessian_jacobian: bool = False
    self.__gradient: Optional[attribute] = None
    self.__hessian: Optional[attribute] = None
    self.__derivative_name_suffix: str = ""

    # those are the attributes that helps with the separation of jacobian and hessian
    # when jacobian is created, it is very likely that it is not only block sparse
    # but also each block is sparse
    # so what we do is store the nonzeor positions, and attributes so we can later on reconstruct the jacobian matrix's diagonal blocks
    self.__global_jacobian_block_nonzero_attributes: List[attribute] = []
    self.__global_jacobian_block_nonzero_local_positions: List[int] = []
    self.__global_jacobian_children_sizes: List[int] = []
    self.__global_jacobian_children_spans: List[int] = []
    # self.__local_hessian_reordered_array: List[attribute] = []

  def __resetDiffState(
    self,
    source: attribute,
    path_dict: Dict[attribute, List[attribute]],
    unioned_child_to_its_children: Dict[attribute, List[attribute]],
    gradient_only: bool,
    projection_method: int,
    save_intermediate: bool,
    separate_hessian_jacobian: bool,
    dynamic_instances: bool,
    generation_mode: str
  ) -> None:
    self.__source = source
    self.__path_dict = path_dict
    self.__unioned_child_to_its_children = unioned_child_to_its_children
    self.__save_intermediate = save_intermediate
    self.__intermediate_compute_pairs = {}
    self.__gradient_only = gradient_only
    self.__projection_method = projection_method
    self.__global_jacobian = None
    self.__global_inner_hessian = None
    self.__project_entire_hessian = False
    self.__separate_hessian_jacobian = separate_hessian_jacobian
    self.__gradient = None
    self.__hessian = None
    projection_flag = str(projection_method)
    if projection_flag.startswith("-"):
      projection_flag = "neg_" + projection_flag[1:]
    projection_flag = "".join(
      character if character.isalnum() else "_"
      for character in projection_flag
    )
    mode_flag = "".join(
      character if character.isalnum() else "_"
      for character in generation_mode
    )
    self.__derivative_name_suffix = (
      f'_spd_projection_{projection_flag}'
      f'_save_{int(save_intermediate)}'
      f'_separate_{int(separate_hessian_jacobian)}'
      f'_dynamic_{int(dynamic_instances)}'
      f'_mode_{mode_flag}'
    )
    if gradient_only:
      self.__derivative_name_suffix += '_gradient_only_1'


    self.__global_jacobian_block_nonzero_attributes = []
    self.__global_jacobian_block_nonzero_local_positions = []
    self.__global_jacobian_children_sizes = []
    self.__global_jacobian_children_spans = []
    # self.__local_hessian_reordered_array = []

  def __sameTargets(self, target1: List[attribute], target2: List[attribute]) -> bool:
    if len(target1) != len(target2):
      return False
    for left, right in zip(target1, target2):
      if left.hash != right.hash:
        return False
    return True

  # helper function for generating the current hessian name wrt the children
  def __hessianName(
    self,
    current: attribute,
    wrt: List[attribute],
    filled_for_source: bool = False
  ) -> str:
    result = (
      f'd2_{current.fullName}_d2_{"__".join([x.fullName for x in wrt])}'
      f'{self.__derivative_name_suffix}'
    )
    if filled_for_source:
      assert self.__source is not None
      result += f'_filled_for_{self.__source.fullName}'
    return result

  # helper function for generating the current gradient/jacobian name wrt the children
  def __gradientName(
    self,
    current: attribute,
    wrt: List[attribute],
    filled_for_source: bool = False
  ) -> str:
    result = (
      f'd_{current.fullName}_d_{"__".join([x.fullName for x in wrt])}'
      f'{self.__derivative_name_suffix}'
    )
    if filled_for_source:
      assert self.__source is not None
      result += f'_filled_for_{self.__source.fullName}'
    return result

  def __combinedGradientName(
    self,
    children: List[attribute],
    wrt: List[attribute]
  ) -> str:
    return (
      f'd_{"__".join([x.fullName for x in children])}'
      f'_d_{"__".join([x.fullName for x in wrt])}'
      f'{self.__derivative_name_suffix}'
    )

  # helper function. For jacobian, the path should never include bottom attributes which are not inside the actual target
  def __validateJacobianIndexPath(
    self,
    target_path: path,
    targets: List[attribute],
    role: str
  ) -> None:
    target_hashes = {target.hash for target in targets}
    for children in target_path.path_dict.values():
      for child in children:
        if (
          (child.operator == DATA or child.operator == CONSTANT)
          and child.hash not in target_hashes
        ):
          raise RuntimeError(
            f"differentiator.diff2: {role} index path contains terminal "
            f"attribute {child.fullName}, which is not in {role}_targets."
          )

  def diff1(self, source, global_targets: List[attribute], local_targets: List[attribute] = [], dynamic_instances = False) -> gradient:
    """Differentiate scalar source terms and return their summed gradient."""
    if not isinstance(source, list):
      source = [source]
    if len(source) == 0:
      raise ValueError("differentiator.diff1: source can not be empty.")
    if not isinstance(global_targets, list) or len(global_targets) == 0:
      raise ValueError("differentiator.diff1: global_targets can not be empty.")

    if len(source) == 1:
      parent_hessian = self.__diff2_hessian_single(
        source[0],
        global_targets,
        local_targets,
        -1,
        False,
        False,
        dynamic_instances,
        True
      )
    else:
      parent_hessian = self.__diff2_hessian_all(
        source,
        global_targets,
        local_targets,
        -1,
        False,
        False,
        dynamic_instances,
        True
      )

    result = gradient(global_targets, parent_hessian)
    parent_hessian.gradient = result
    return result

  def diff2(self, source: List[attribute], target1: List[attribute], target2: List[attribute], local_targets: List[attribute] = [], projection_method = 1, save_intermediate = False, separate_hessian_jacobian = False, dynamic_instances = False):
    if not isinstance(source, list):
      source = [source]
    if len(source) == 0:
      raise ValueError("differentiator.diff2: source can not be empty.")
    if self.__sameTargets(target1, target2):
      if len(source) == 1:
        return self.__diff2_hessian_single(source[0], target1, local_targets, projection_method, save_intermediate, separate_hessian_jacobian, dynamic_instances)
      return self.__diff2_hessian_all(source, target1, local_targets, projection_method, save_intermediate, separate_hessian_jacobian, dynamic_instances)

    if len(source) == 1:
      return self.__diff2_jacobian_single(source[0], target1, target2, dynamic_instances)
    return self.__diff2_jacobian_all(source, target1, target2, dynamic_instances)


  def __diff2_jacobian_all(
    self,
    source: List[attribute],
    row_targets: List[attribute],
    column_targets: List[attribute],
    dynamic_instances: bool = False
  ) -> secondOrderJacobian:
    total_jacobian: Optional[secondOrderJacobian] = None
    for item in source:
      current_jacobian = self.__diff2_jacobian_single(
        item,
        row_targets,
        column_targets,
        dynamic_instances
      )
      if total_jacobian is None:
        total_jacobian = current_jacobian
      else:
        total_jacobian = total_jacobian + current_jacobian
    assert total_jacobian is not None
    return total_jacobian

  #########################################################
  ## Second-order Jacobian differentiation, each
  ## differentiation will return a secondOrderJacobian
  ## matrix object.
  #########################################################
  def __diff2_jacobian_single(
    self,
    source: attribute,
    row_targets: List[attribute],
    column_targets: List[attribute],
    dynamic_instances: bool = False
  ) -> secondOrderJacobian:
    if source.size != 1:
      raise ValueError(
        "differentiator.__diff2_jacobian_single: source must be a scalar attribute."
      )

    # First differentiate the scalar source with respect to only the row
    # targets.  The resulting 1 x R gradient is reshaped in-place to R x 1 so
    # that differentiating it with respect to the column targets directly
    # produces the desired R x C second-order Jacobian.
    row_paths = path(row_targets)
    row_paths.getRoots(source, [source], row_targets)
    row_paths.getPathDict()
    self.__validateJacobianIndexPath(row_paths, row_targets, "row")

    self.__resetDiffState(
      source,
      row_paths.path_dict,
      row_paths.unioned_child_to_its_children,
      True,
      -1,
      False,
      False,
      dynamic_instances,
      "second_order_jacobian_row_gradient"
    )
    autodiff_engine = autodiff()
    self.__generateGradientThroughPathDict(
      row_targets,
      autodiff_engine
    )
    assert self.__gradient is not None
    row_gradient = self.__gradient
    row_gradient.reshape(row_gradient.size, 1)

    # Preserve the complete column occurrence layout from the original
    # scalar source, including columns whose mixed derivative is structurally
    # zero.  Re-rooting that compile-time path at the row gradient makes the
    # second pass differentiate the new vector-valued source while retaining
    # the exact column ordering used by the coordinate stream.
    column_paths = path(column_targets)
    column_paths.getRoots(source, [source], column_targets)
    column_paths.getPathDict()
    self.__validateJacobianIndexPath(column_paths, column_targets, "column")
    if source not in column_paths.path_dict:
      raise RuntimeError(
        "differentiator.__diff2_jacobian_single: the column path has no "
        "entry for the scalar source."
      )
    column_derivative_path_dict = dict(column_paths.path_dict)
    source_column_children = column_derivative_path_dict.pop(source)
    column_derivative_path_dict[row_gradient] = source_column_children

    # Differentiate the row gradient with respect to only the column targets.
    # This is a first-order Jacobian operation on a vector-valued source; no
    # combined Hessian or rectangular block selection is involved.
    self.__resetDiffState(
      row_gradient,
      column_derivative_path_dict,
      column_paths.unioned_child_to_its_children,
      True,
      -1,
      False,
      False,
      dynamic_instances,
      "second_order_jacobian_column_jacobian_row_major_complete_layout"
    )
    self.__generateGradientThroughPathDict(
      column_targets,
      autodiff()
    )
    assert self.__gradient is not None
    rectangular_jacobian = self.__gradient

    result = secondOrderJacobian(
      row_targets,
      column_targets
    )

    # Row and column coordinates are generated together by the rectangular
    # second-order Jacobian kernel.  Each stream therefore computes only its
    # indices and excludes every leaf that is not in its own target list.
    row_indices_kernel = gradientIndicesKernel(
      row_paths.path_dict,
      row_paths.unioned_child_to_its_children,
      row_targets,
      result.row_start_indices,
      source,
      no_local_permutation=True,
      generate_coordinates=False
    )
    column_indices_kernel = gradientIndicesKernel(
      column_paths.path_dict,
      column_paths.unioned_child_to_its_children,
      column_targets,
      result.column_start_indices,
      source,
      no_local_permutation=True,
      generate_coordinates=False
    )

    rectangular_indices_kernel = secondOrderJacobianIndicesKernel(
      row_indices_kernel,
      column_indices_kernel,
      result.row_start_indices,
      result.column_start_indices
    )

    if dynamic_instances:
      result.sources_dynamic = [source]
      result.indices_kernels_dynamic = [rectangular_indices_kernel]
      result.second_order_jacobians_dynamic = [rectangular_jacobian]
    else:
      result.sources = [source]
      result.indices_kernels = [rectangular_indices_kernel]
      result.second_order_jacobians = [rectangular_jacobian]
    return result

  def __diff2_hessian_all(self, source: List[attribute], global_targets: List[attribute], local_targets: List[attribute] = [], projection_method = 1, save_intermediate = False, separate_hessian_jacobian = False, dynamic_instances = False, gradient_only = False):
    total_hessian: Optional[hessian] = None
    for item in source:
      current_hessian = self.__diff2_hessian_single(item, global_targets, local_targets, projection_method, save_intermediate, separate_hessian_jacobian, dynamic_instances, gradient_only)
      if total_hessian is None:
        total_hessian = current_hessian
      else:
        total_hessian = total_hessian + current_hessian
    assert total_hessian is not None
    return total_hessian

  #########################################################
  ## Hessian differentiation, each differentiation
  ## will return us a Hessian matrix object
  #########################################################
  def __diff2_hessian_single(self, source: attribute, global_targets: List[attribute], local_targets: List[attribute] = [], projection_method = 1, save_intermediate = False, separate_hessian_jacobian = False, dynamic_instances = False, gradient_only = False) -> hessian:
    if source.size != 1:
      raise ValueError("differentiator.__diff2_hessian_single: source must be a scalar attribute.")

    hessian_local = hessian(global_targets, local_targets, dynamic_instances)
    wrt_start_indices = hessian_local.wrt_start_indices

    paths = path(global_targets, local_targets)
    paths.getRoots(source, [source])
    paths.getPathDict()

    indices_kernel = gradientIndicesKernel(
      paths.path_dict,
      paths.unioned_child_to_its_children,
      paths.wrt,
      wrt_start_indices,
      source,
      separate_hessian_jacobian
    )

    self.__resetDiffState(
      source,
      paths.path_dict,
      paths.unioned_child_to_its_children,
      gradient_only,
      projection_method,
      save_intermediate,
      separate_hessian_jacobian,
      dynamic_instances,
      "gradient" if gradient_only else "hessian"
    )
    autodiff_engine = autodiff()
    self.__generateGradientThroughPathDict(paths.wrt, autodiff_engine)
    assert self.__gradient is not None
    if not gradient_only:
      self.__generateHessianThroughPathDict(paths.wrt, autodiff_engine)
      assert self.__hessian is not None

    global_jacobian_block_nonzero_attributes = [list(self.__global_jacobian_block_nonzero_attributes)]
    global_jacobian_block_nonzero_local_positions = [list(self.__global_jacobian_block_nonzero_local_positions)]
    global_jacobian_children_sizes = [list(self.__global_jacobian_children_sizes)]
    global_jacobian_children_spans = [list(self.__global_jacobian_children_spans)]
    # local_hessian_reordered_array = [list(self.__local_hessian_reordered_array)]

    if not dynamic_instances:
      hessian_local.indices_kernels = [indices_kernel]
      hessian_local.global_gradients = [self.__gradient]
      hessian_local.global_hessians = [self.__hessian]
      hessian_local.gradient_only = [gradient_only]
      hessian_local.global_jacobians = [self.__global_jacobian]
      hessian_local.global_inner_hessians = [self.__global_inner_hessian]
      hessian_local.project_entire_hessian = [self.__project_entire_hessian]
      hessian_local.projection_methods = [projection_method]
      hessian_local.separate_hessian_jacobian = [separate_hessian_jacobian]
      hessian_local.intermediate_compute_pairs = [dict(self.__intermediate_compute_pairs)]
      hessian_local.merged_hessian_and_gradient_attributes = [None]
      hessian_local.hessian_and_gradient_kernels = [None]
      hessian_local.sources = [source]
      hessian_local.global_jacobian_block_nonzero_attributes = global_jacobian_block_nonzero_attributes
      hessian_local.global_jacobian_block_nonzero_local_positions = global_jacobian_block_nonzero_local_positions
      hessian_local.global_jacobian_children_sizes = global_jacobian_children_sizes
      hessian_local.global_jacobian_children_spans = global_jacobian_children_spans
      hessian_local.placement_reorder_kernels = [placementReorderKernel()]
      # hessian_local.local_hessian_reordered_array = local_hessian_reordered_array
    else:
      hessian_local.indices_kernels_dynamic = [indices_kernel]
      hessian_local.global_gradients_dynamic = [self.__gradient]
      hessian_local.global_hessians_dynamic = [self.__hessian]
      hessian_local.gradient_only_dynamic = [gradient_only]
      hessian_local.global_jacobians_dynamic = [self.__global_jacobian]
      hessian_local.global_inner_hessians_dynamic = [self.__global_inner_hessian]
      hessian_local.project_entire_hessian_dynamic = [self.__project_entire_hessian]
      hessian_local.projection_methods_dynamic = [projection_method]
      hessian_local.separate_hessian_jacobian_dynamic = [separate_hessian_jacobian]
      hessian_local.intermediate_compute_pairs_dynamic = [dict(self.__intermediate_compute_pairs)]
      hessian_local.merged_hessian_and_gradient_attributes_dynamic = [None]
      hessian_local.hessian_and_gradient_kernels_dynamic = [None]
      hessian_local.sources_dynamic = [source]
      hessian_local.global_jacobian_block_nonzero_attributes_dynamic = global_jacobian_block_nonzero_attributes
      hessian_local.global_jacobian_block_nonzero_local_positions_dynamic = global_jacobian_block_nonzero_local_positions
      hessian_local.global_jacobian_children_sizes_dynamic = global_jacobian_children_sizes
      hessian_local.global_jacobian_children_spans_dynamic = global_jacobian_children_spans
      hessian_local.placement_reorder_kernels_dynamic = [placementReorderKernel()]
      # hessian_local.local_hessian_reordered_array_dynamic = local_hessian_reordered_array
    return hessian_local

  def __generateGradientThroughPathDict(self, wrt: List[attribute], autodiff_engine: autodiff) -> None:
    assert self.__source is not None
    source_gradient_name = self.__gradientName(self.__source, wrt)
    cached_gradient: Optional[attribute] = None
    if source_gradient_name in self.__source.correspondance.attributes:
      cached_gradient = self.__source.correspondance.attributes[
        source_gradient_name
      ]

    # Even when the first derivative is cached, each Hessian configuration
    # has its own local second-derivative attributes.  Ensure those attributes
    # exist before reusing the gradient.
    for parent in self.__path_dict.keys():
      children = self.__path_dict[parent]
      if parent.operator != JOIN and parent.operator != UNION:
        self.__generateNeighborJacobianForEnergy(parent, children, autodiff_engine)
      elif parent.operator == JOIN:
        self.__generateNeighborJacobianForJoin(parent, children, autodiff_engine)
      elif parent.operator == UNION:
        self.__generateNeighborJacobianForUnion(parent, children, autodiff_engine)
      else:
        raise ValueError(f"differentiator.__generateGradientThroughPathDict: operator {parent.operator} is not supported in path dict.")

    if cached_gradient is not None:
      self.__gradient = cached_gradient
      return
    self.__gradient = self.__generateGradientThroughRecursion(self.__source, wrt)

  def __generateGradientThroughRecursion(self, current: attribute, wrt: List[attribute]) -> attribute:
    from yasps.attribute import JOIN, UNION, DATA, CONSTANT
    if current.operator == DATA or current.operator == CONSTANT:
      return attribute.identity(current.size)
    gradient_attribute_name = self.__gradientName(current, wrt)
    if gradient_attribute_name in current.correspondance.attributes:
      return current.correspondance[gradient_attribute_name]

    if current.operator != JOIN and current.operator != UNION:
      return self.__generateGlobalJacobianForEnergy(current, wrt)
    if current.operator == JOIN:
      return self.__generateGlobalJacobianForJoin(current, wrt)
    return self.__generateGlobalJacobianForUnion(current, wrt)

  def __generateNeighborJacobianForEnergy(self, parent: attribute, children: List[attribute], autodiff_engine: autodiff) -> None:
    # This computes d(parent) / d(children).  For the usual scalar energy the
    # result has one row.  A second-order Jacobian's second pass uses a
    # vector-valued row gradient, so retain one row per parent component.
    local_gradient_name = f'd_{parent.fullName}_d_{"__".join([x.fullName for x in children])}'
    if local_gradient_name not in parent.correspondance.attributes:
      local_gradient_cols = sum(child.size for child in children)
      diff_energy_wrt_children_list: List[attribute] = [
        attribute(float_value=0.0)
        for _ in range(parent.size * local_gradient_cols)
      ]
      column_offset = 0
      for child in children:
        result = autodiff_engine.diff(parent, child)
        if result.rows != parent.size or result.cols != child.size:
          raise RuntimeError(
            "differentiator.__generateNeighborJacobianForEnergy: local "
            f"Jacobian block has shape {result.rows}x{result.cols}, "
            f"expected {parent.size}x{child.size}."
          )
        for row in range(result.rows):
          for column in range(result.cols):
            diff_energy_wrt_children_list[
              row * local_gradient_cols + column_offset + column
            ] = result[row, column]
        column_offset += child.size
      if column_offset != local_gradient_cols:
        raise RuntimeError(
          "differentiator.__generateNeighborJacobianForEnergy: local "
          "Jacobian column assembly is inconsistent."
        )
      local_gradient = attribute.to_array(
        diff_energy_wrt_children_list,
        rows=parent.size,
        cols=local_gradient_cols
      )
      parent.correspondance.addAttribute(local_gradient_name, computed_attribute=local_gradient)

    if self.__gradient_only:
      return

    # this computes d2 energy / d children d children, where the children are the local children of the energy variable
    local_hessian_name = self.__hessianName(parent, children)
    if local_hessian_name not in parent.correspondance.attributes:
      local_gradient = parent.correspondance[local_gradient_name]
      double_diff_results = [0.0 for _ in range(local_gradient.size * local_gradient.size)]
      col_offset = 0
      for child in children:
        local_double_diff = autodiff_engine.diff(local_gradient, child)
        for i in range(local_double_diff.rows):
          for j in range(local_double_diff.cols):
            double_diff_results[i * local_gradient.size + col_offset + j] = local_double_diff[i, j]
        col_offset += child.size
      assert col_offset == local_gradient.size, f"differentiator.__generateNeighborJacobianForEnergy: col_offset {col_offset} is not equal to local gradient size {local_gradient.size}"
      local_hessian = attribute.to_array(double_diff_results, rows=local_gradient.size, cols=local_gradient.size)
      if self.__save_intermediate and local_hessian.isZero == 0:
        if local_hessian_name not in parent.correspondance.attributes:
          hessian_data_attribute = parent.correspondance.addAttribute(local_hessian_name, rows=local_hessian.rows, cols=local_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            parent.correspondance.addAttribute(new_name, computed_attribute=local_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (local_hessian, hessian_data_attribute)
      else:
        parent.correspondance.addAttribute(local_hessian_name, computed_attribute=local_hessian)

  def __generateGlobalJacobianForEnergy(self, current: attribute, wrt: List[attribute]):
    # first access the local gradient d energy / d children, where the children are the local children of the energy variable
    gradient_attribute_name = self.__gradientName(current, wrt)
    if gradient_attribute_name in current.correspondance.attributes:
      result = current.correspondance[gradient_attribute_name]
      self.__global_jacobian = result
      return result

    # we now access the d children /d x, where x is the global optimization variables
    # we will then multiply them to get the global gradient
    # we also assemble the global jacobian matrix here, which is d children /dx
    children = self.__path_dict[current]
    current_gradient: attribute = current.correspondance[f'd_{current.fullName}_d_{"__".join([x.fullName for x in children])}']
    children_global_jacobian_name = self.__combinedGradientName(
      children,
      wrt
    )
    if children_global_jacobian_name in current.correspondance.attributes:
      next_jacobian = current.correspondance[children_global_jacobian_name]
      self.__global_jacobian = next_jacobian
    else:
      children_jacobian: List[attribute] = []
      for child in children:
        children_jacobian.append(self.__generateGradientThroughRecursion(child, wrt))
      next_jacobian_rows = sum([x.size for x in children])
      next_jacobian_cols = sum([x.cols for x in children_jacobian])
      next_jacobian_children = [attribute(float_value=0.0) for _ in range(next_jacobian_rows * next_jacobian_cols)]
      row_offset = 0
      col_offset = 0
      for item in children_jacobian:
        for i in range(item.rows):
          for j in range(item.cols):
            next_jacobian_children[(i + row_offset) * next_jacobian_cols + j + col_offset] = item[i, j]
        col_offset += item.cols
        row_offset += item.rows
      next_jacobian = attribute.to_array(next_jacobian_children, rows=next_jacobian_rows, cols=next_jacobian_cols)
      if children_global_jacobian_name not in current.correspondance.attributes:
        current.correspondance.addAttribute(children_global_jacobian_name, computed_attribute=next_jacobian)
      self.__global_jacobian = next_jacobian
    full_gradient = current_gradient.mul_explicit(next_jacobian)
    current.correspondance.addAttribute(gradient_attribute_name, computed_attribute=full_gradient)
    return full_gradient

  def __generateGlobalHessianForEnergy(self, current: attribute, wrt: List[attribute]) -> attribute:
    # this function computes the global hessian
    # it is separated into two parts, the first part is the local hessian multiplied by the global jacobian, and the second part is the local gradient multiplied by the global hessian of the children
    # if the second part is completely 0, then we can project the inner local hessian of the first part
    global_hessian_name = self.__hessianName(current, wrt)
    if global_hessian_name in current.correspondance.attributes:
      return current.correspondance[global_hessian_name]

    children = self.__path_dict[current]
    local_gradient_name = f'd_{current.fullName}_d_{"__".join([x.fullName for x in children])}'
    local_gradient = current.correspondance[local_gradient_name]

    global_jacobian_name = self.__gradientName(current, wrt)
    global_jacobian = current.correspondance[global_jacobian_name]

    second_part_hessian_array = [0.0 for _ in range(global_jacobian.cols * global_jacobian.cols)]
    block_offset = 0
    block_sizes = [] # record the span of jacobian (the column)
    children_sizes = []
    for child in children:
      child_global_hessian = self.__generateHessianThroughRecursion(child, wrt)
      child_size = child.size
      hessian_size = child_global_hessian.size // child_size
      assert hessian_size * child_size == child_global_hessian.size, f"differentiator.__generateGlobalHessianForEnergy: hessian size {hessian_size} * child size {child_size} is not equal to child global hessian size {child_global_hessian.size}"
      hessian_rows = int(math.sqrt(hessian_size))
      if child.operator == JOIN:
        block_sizes += [hessian_rows] * child.through.dimension  # this tells us in the final Hessian (local), what span does this child cover
        children_sizes += [child.size // child.through.dimension] * child.through.dimension # this tells us in the final Hessian (global), what span does this child cover
      elif (
        child.operator == UNION
        or child.operator == DATA
        or child.operator == CONSTANT
      ):
        block_sizes.append(hessian_rows)
        children_sizes.append(child.size)
      assert hessian_rows * hessian_rows == hessian_size, f"differentiator.__generateGlobalHessianForEnergy: hessian rows {hessian_rows} is not equal to hessian size {hessian_size}"
      hessian_cols = hessian_rows
      for i in range(hessian_rows):
        for j in range(hessian_cols):
          for k in range(child_size):
            second_part_hessian_array[(block_offset + i) * global_jacobian.cols + (block_offset + j)] += local_gradient[block_offset + k] * child_global_hessian[k * hessian_size + i * hessian_cols + j]
      block_offset += child_size
    second_part_hessian = attribute.to_array(second_part_hessian_array, rows=global_jacobian.cols, cols=global_jacobian.cols)

    local_hessian_name = self.__hessianName(current, children)
    local_hessian = current.correspondance[local_hessian_name]

    if second_part_hessian.isZero > 0:
      if self.__projection_method >= 0:
        local_hessian = local_hessian.spd(self.__projection_method)
      self.__global_inner_hessian = local_hessian
      self.__project_entire_hessian = False
    else:
      self.__project_entire_hessian = True

    children_jacobian_name = self.__combinedGradientName(children, wrt)
    children_global_jacobian = current.correspondance[children_jacobian_name]
    self.__global_jacobian = children_global_jacobian

    # ok regardless of if we want to separate jacobian and hessian or not
    # let's just compute the following
    # because the jacobian matrix is blocked sparse
    # and even the blocks are sparse
    # what we will do is extract each block, then for each block we extract the non-zero entries
    assert sum(children_sizes) == children_global_jacobian.rows, f"differentiator.__generateGlobalHessianForEnergy: sum of children sizes {sum(children_sizes)} is not equal to children global jacobian rows {children_global_jacobian.rows}"
    assert sum(block_sizes) == children_global_jacobian.cols, f"differentiator.__generateGlobalHessianForEnergy: sum of block sizes {sum(block_sizes)} is not equal to children global jacobian cols {children_global_jacobian.cols}"
    row_offset = 0 # the offset of rows in the jacobian matrix
    col_offset = 0 # the offset of columns in the jacobian matrix
    nonzero_attributes_array = []
    nonzero_local_positions = []
    # print("Children sizes: ", children_sizes)
    # print("Block sizes: ", block_sizes)
    # print("Jacobian Size: ", (children_global_jacobian.rows, children_global_jacobian.cols))
    for i in range(len(children_sizes)):
      nonzero_counts = 0
      child_size = children_sizes[i]
      child_span = block_sizes[i]
      for i in range(child_size):
        for j in range(child_span):
          if children_global_jacobian[(row_offset + i), (col_offset + j)].isZero == 0:
            nonzero_counts += 1 # iszero == 0 means it's not zero, it's fucked up, i know
            nonzero_local_positions.append(row_offset + i)
            nonzero_local_positions.append(col_offset + j)
            nonzero_attributes_array.append(children_global_jacobian[(row_offset + i), (col_offset + j)])
      # the jacobian matrix is always block diagonal, so once we are done with one child, we can skip to the next diagonal block
      row_offset += child_size
      col_offset += child_span
    # print(f"Nonzero element count: {len(nonzero_attributes_array)} / {children_global_jacobian.size}")
    # print(f"True nonzero count: {sum(children_global_jacobian[i].isZero == 0 for i in range(children_global_jacobian.size))}")

    # # if we need to separate the jacobian and hessian
    # # we will need to do some kind of reordering
    # # this is because the jacobian is blocked diagonal
    # # so we want to process the matrices by order
    # local_hessian_reordered_array = []
    # row_offset = 0
    # for i in range(len(children_sizes)):
    #   child_size_i = children_sizes[i]
    #   col_offset = 0
    #   for j in range(i, len(children_sizes)):
    #     child_size_j = children_sizes[j]
    #     for m in range(child_size_i):
    #       for n in range(child_size_j):
    #         local_hessian_reordered_array.append(local_hessian[row_offset + m, col_offset + n])
    #     col_offset += child_size_j
    #   row_offset += child_size_i

    self.__global_jacobian_block_nonzero_attributes = nonzero_attributes_array
    self.__global_jacobian_block_nonzero_local_positions = nonzero_local_positions
    self.__global_jacobian_children_sizes = children_sizes
    self.__global_jacobian_children_spans = block_sizes
    # self.__local_hessian_reordered_array = local_hessian_reordered_array







    final_hessian = children_global_jacobian.transpose().mul_explicit(local_hessian.mul_explicit(children_global_jacobian)).add_explicit(second_part_hessian)
    current.correspondance.addAttribute(global_hessian_name, computed_attribute=final_hessian)
    return final_hessian

  def __generateNeighborJacobianForJoin(self, parent: attribute, children: List[attribute], autodiff_engine: autodiff) -> None:
    # computes the neighboring jacobian, as well as the neighboring hessian
    # this the this is computing d children / d grandchildren, where the children are the local children of the join variable, and the grandchildren are the local children of the child variable
    joined_child = parent.children[0]
    child_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in children])}'
    if child_jacobian_name not in joined_child.correspondance.attributes:
      child_diff_next_children: List[attribute] = []
      for child in children:
        diff_result = autodiff_engine.diff(joined_child, child)
        child_diff_next_children.append(diff_result)
      jacobian_num_rows = joined_child.size
      jacobian_num_cols = sum([x.size for x in children])
      merged_jacobian_list: List[attribute] = [attribute(float_value=0.0) for _ in range(jacobian_num_rows * jacobian_num_cols)]
      col_offset = 0
      for child in child_diff_next_children:
        for i in range(child.rows):
          for j in range(child.cols):
            merged_jacobian_list[i * jacobian_num_cols + col_offset + j] = child[i, j]
        col_offset += child.cols
      merged_jacobian = attribute.to_array(merged_jacobian_list, rows=jacobian_num_rows, cols=jacobian_num_cols)
      joined_child.correspondance.addAttribute(child_jacobian_name, computed_attribute=merged_jacobian)

    local_hessian_name = self.__hessianName(joined_child, children)
    if local_hessian_name not in joined_child.correspondance.attributes:
      num_hessians = joined_child.size
      hessian_num_cols = sum([x.size for x in children])
      hessian_num_rows = hessian_num_cols
      joined_child_jacobian = joined_child.correspondance[child_jacobian_name]
      all_hessian_results = []
      for row in range(num_hessians):
        for col in range(joined_child_jacobian.cols):
          current_item = joined_child_jacobian[row, col]
          for child in children:
            diff_result = autodiff_engine.diff(current_item, child)
            for i in range(diff_result.rows):
              for j in range(diff_result.cols):
                all_hessian_results.append(diff_result[i, j])

      merged_hessian = attribute.to_array(all_hessian_results, rows=hessian_num_rows * num_hessians, cols=hessian_num_cols)
      if self.__save_intermediate and merged_hessian.isZero == 0:
        if local_hessian_name not in joined_child.correspondance.attributes:
          hessian_data_attribute = joined_child.correspondance.addAttribute(local_hessian_name, rows=merged_hessian.rows, cols=merged_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            joined_child.correspondance.addAttribute(new_name, computed_attribute=merged_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (merged_hessian, hessian_data_attribute)
      else:
        joined_child.correspondance.addAttribute(local_hessian_name, computed_attribute=merged_hessian)

  def __generateGlobalJacobianForJoin(self, current: attribute, wrt: List[attribute]):
    gradient_attribute_name = self.__gradientName(current, wrt)
    if gradient_attribute_name in current.correspondance.attributes:
      result = current.correspondance[gradient_attribute_name]
      self.__global_jacobian = result
      return result

    joined_child = current.children[0]
    joined_child_global_jacobian_name = self.__gradientName(
      joined_child,
      wrt
    )
    if joined_child_global_jacobian_name in joined_child.correspondance.attributes:
      joined_child_global_jacobian = joined_child.correspondance[joined_child_global_jacobian_name]
    else:
      next_children = self.__path_dict[current]
      joined_child_local_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in next_children])}'
      joined_child_local_jacobian = joined_child.correspondance[joined_child_local_jacobian_name]

      children_global_jacobian_name = self.__combinedGradientName(
        next_children,
        wrt
      )
      if children_global_jacobian_name in joined_child.correspondance.attributes:
        children_global_jacobian = joined_child.correspondance[children_global_jacobian_name]
      else:
        next_children_jacobians = []
        for child in next_children:
          child_jacobian = self.__generateGradientThroughRecursion(child, wrt)
          next_children_jacobians.append(child_jacobian)

        children_global_jacobian_rows = sum([x.size for x in next_children])
        children_global_jacobian_cols = sum([x.cols for x in next_children_jacobians])
        children_global_jacobian_items = [attribute(float_value=0.0) for _ in range(children_global_jacobian_rows * children_global_jacobian_cols)]
        row_offset = 0
        col_offset = 0
        for item in next_children_jacobians:
          for i in range(item.rows):
            for j in range(item.cols):
              children_global_jacobian_items[(i + row_offset) * children_global_jacobian_cols + j + col_offset] = item[i, j]
          col_offset += item.cols
          row_offset += item.rows
        children_global_jacobian = attribute.to_array(children_global_jacobian_items, rows=children_global_jacobian_rows, cols=children_global_jacobian_cols)
        if children_global_jacobian_name not in joined_child.correspondance.attributes:
          joined_child.correspondance.addAttribute(children_global_jacobian_name, computed_attribute=children_global_jacobian)

      child_global_jacobian = joined_child_local_jacobian.mul_explicit(children_global_jacobian)
      if joined_child_global_jacobian_name in joined_child.correspondance.attributes:
        joined_child_global_jacobian = joined_child.correspondance[joined_child_global_jacobian_name]
      else:
        joined_child_global_jacobian = joined_child.correspondance.addAttribute(joined_child_global_jacobian_name, computed_attribute=child_global_jacobian)

    if self.__save_intermediate:
      if joined_child_global_jacobian.fullName not in self.__intermediate_compute_pairs:
        new_name = joined_child_global_jacobian.name + "_evaluated"
        if new_name not in joined_child_global_jacobian.correspondance.attributes:
          evaluated_result = joined_child_global_jacobian.correspondance.addAttribute(new_name, rows=joined_child_global_jacobian.rows, cols=joined_child_global_jacobian.cols)
          self.__intermediate_compute_pairs[joined_child_global_jacobian.fullName] = (joined_child_global_jacobian, evaluated_result)
          joined_child_global_jacobian = evaluated_result
      else:
        joined_child_global_jacobian = self.__intermediate_compute_pairs[joined_child_global_jacobian.fullName][1]

    res = current.correspondance.addAttribute(gradient_attribute_name + "_unresized", through=current.through, source=joined_child_global_jacobian)
    assert (joined_child_global_jacobian.rows * current.through.dimension) == current.size, f"differentiator.__generateGlobalJacobianForJoin: joined child global jacobian rows {joined_child_global_jacobian.rows} * current.through.dimension {current.through.dimension} is not equal to current size {current.size}"
    actual_global_jacobian_rows = current.size
    actual_global_jacobian_cols = joined_child_global_jacobian.cols * current.through.dimension
    actual_global_jacobian_items = [attribute(float_value=0.0) for _ in range(actual_global_jacobian_rows * actual_global_jacobian_cols)]
    for index in range(current.through.dimension):
      for i in range(joined_child_global_jacobian.rows):
        for j in range(joined_child_global_jacobian.cols):
          actual_global_jacobian_items[(index * joined_child_global_jacobian.rows + i) * actual_global_jacobian_cols + (index * joined_child_global_jacobian.cols) + j] = res[index, i * joined_child_global_jacobian.cols + j]

    actual_global_jacobian = attribute.to_array(actual_global_jacobian_items, rows=actual_global_jacobian_rows, cols=actual_global_jacobian_cols)
    current.correspondance.addAttribute(gradient_attribute_name, computed_attribute=actual_global_jacobian)
    self.__global_jacobian = actual_global_jacobian
    return actual_global_jacobian

  def __generateGlobalHessianForJoin(self, current: attribute, wrt: List[attribute]) -> attribute:
    global_hessian_name = self.__hessianName(current.children[0], wrt)
    if global_hessian_name in current.correspondance.attributes:
      return current.correspondance[global_hessian_name]

    joined_child = current.children[0]
    joined_child_global_hessian_name = self.__hessianName(
      joined_child,
      wrt
    )
    if joined_child_global_hessian_name in joined_child.correspondance.attributes:
      joined_child_global_hessian = joined_child.correspondance[joined_child_global_hessian_name]
    else:
      num_hessians = joined_child.size
      next_children = self.__path_dict[current]
      joined_child_local_jacobian_name = f'd_{joined_child.fullName}_d_{"__".join([x.fullName for x in next_children])}'
      joined_child_local_jacobian = joined_child.correspondance[joined_child_local_jacobian_name]
      joined_child_local_hessian_name = self.__hessianName(
        joined_child,
        next_children
      )
      joined_child_local_hessian = joined_child.correspondance[joined_child_local_hessian_name]

      next_children_global_jacobian_name = self.__combinedGradientName(
        next_children,
        wrt
      )
      next_children_global_jacobian = joined_child.correspondance[next_children_global_jacobian_name]

      next_children_global_hessian: List[attribute] = []
      for child in next_children:
        child_global_hessian = self.__generateHessianThroughRecursion(child, wrt)
        next_children_global_hessian.append(child_global_hessian)
      next_children_total_size = sum([x.size for x in next_children])
      local_hessian_size = next_children_total_size * next_children_total_size
      assert local_hessian_size * num_hessians == joined_child_local_hessian.size, f"differentiator.__generateGlobalHessianForJoin: local hessian size {local_hessian_size} * num hessians {num_hessians} is not equal to joined child local hessian size {joined_child_local_hessian.size}"
      local_hessian_rows = int(math.sqrt(local_hessian_size))
      assert local_hessian_rows * local_hessian_rows == local_hessian_size, f"differentiator.__generateGlobalHessianForJoin: local hessian rows {local_hessian_rows} is not equal to local hessian size {local_hessian_size}"
      local_hessian_cols = local_hessian_rows
      joined_child_global_hessian_items = []
      for N in range(num_hessians):
        nth_joined_child_local_hessian = attribute.to_array([joined_child_local_hessian[i] for i in range(N * local_hessian_size, (N + 1) * local_hessian_size)], rows=local_hessian_rows, cols=local_hessian_cols)
        first_part_hessian = next_children_global_jacobian.transpose().mul_explicit(nth_joined_child_local_hessian.mul_explicit(next_children_global_jacobian))
        local_gradient_size = joined_child_local_jacobian.size // num_hessians
        assert local_gradient_size * num_hessians == joined_child_local_jacobian.size, f"differentiator.__generateGlobalHessianForJoin: local gradient size {local_gradient_size} * num hessians {num_hessians} is not equal to joined child local jacobian size {joined_child_local_jacobian.size}"

        second_part_hessian_array = [0.0 for _ in range(next_children_global_jacobian.cols * next_children_global_jacobian.cols)]
        block_offset = 0
        index = 0
        for child in next_children:
          next_child_global_hessian = next_children_global_hessian[index]
          child_size = child.size
          hessian_size = next_child_global_hessian.size // child_size
          assert hessian_size * child_size == next_child_global_hessian.size, f"differentiator.__generateGlobalHessianForJoin: hessian size {hessian_size} * child size {child_size} is not equal to child global hessian size {next_child_global_hessian.size}"
          hessian_rows = int(math.sqrt(hessian_size))
          assert hessian_rows * hessian_rows == hessian_size, f"differentiator.__generateGlobalHessianForJoin: hessian rows {hessian_rows} is not equal to hessian size {hessian_size}"
          hessian_cols = hessian_rows
          for i in range(hessian_rows):
            for j in range(hessian_cols):
              for k in range(child_size):
                second_part_hessian_array[(block_offset + i) * next_children_global_jacobian.cols + (block_offset + j)] += joined_child_local_jacobian[N * joined_child_local_jacobian.cols + k + block_offset] * next_child_global_hessian[k * hessian_size + i * hessian_cols + j]
          block_offset += child_size
          index += 1
        second_part_hessian = attribute.to_array(second_part_hessian_array, rows=next_children_global_jacobian.cols, cols=next_children_global_jacobian.cols)
        final_hessian = first_part_hessian.add_explicit(second_part_hessian)
        for i in range(final_hessian.size):
          joined_child_global_hessian_items.append(final_hessian[i])

      joined_child_global_hessian = attribute.to_array(joined_child_global_hessian_items, rows=num_hessians * next_children_global_jacobian.cols, cols=next_children_global_jacobian.cols)
      joined_child.correspondance.addAttribute(joined_child_global_hessian_name, computed_attribute=joined_child_global_hessian)

    if self.__save_intermediate:
      if joined_child_global_hessian.fullName not in self.__intermediate_compute_pairs:
        new_name = joined_child_global_hessian.name + "_evaluated"
        if new_name not in joined_child_global_hessian.correspondance.attributes:
          evaluated_result = joined_child_global_hessian.correspondance.addAttribute(new_name, rows=joined_child_global_hessian.rows, cols=joined_child_global_hessian.cols)
          self.__intermediate_compute_pairs[joined_child_global_hessian.fullName] = (joined_child_global_hessian, evaluated_result)
          joined_child_global_hessian = evaluated_result
      else:
        joined_child_global_hessian = self.__intermediate_compute_pairs[joined_child_global_hessian.fullName][1]
    current.correspondance.addAttribute(global_hessian_name, through=current.through, source=joined_child_global_hessian)
    return current.correspondance[global_hessian_name]

  def __generateNeighborJacobianForUnion(self, parent: attribute, children: List[attribute], autodiff_engine: autodiff):
    unioned_children = parent.children
    for unioned_child in unioned_children:
      unioned_child_used_children: List[attribute] = []
      for child in children:
        if child in self.__unioned_child_to_its_children[unioned_child]:
          unioned_child_used_children.append(child)
      if len(unioned_child_used_children) == 0:
        continue

      child_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in unioned_child_used_children])}'
      if child_jacobian_name in unioned_child.correspondance.attributes:
        continue

      local_jacobians = []
      for child in unioned_child_used_children:
        diff_result = autodiff_engine.diff(unioned_child, child)
        local_jacobians.append(diff_result)
      jacobian_num_rows = unioned_child.size
      jacobian_num_cols = sum([x.size for x in unioned_child_used_children])
      merged_jacobian_list = [attribute(float_value=0.0) for _ in range(jacobian_num_rows * jacobian_num_cols)]
      col_offset = 0
      for child in local_jacobians:
        for i in range(child.rows):
          for j in range(child.cols):
            merged_jacobian_list[i * jacobian_num_cols + col_offset + j] = child[i, j]
        col_offset += child.cols
      merged_jacobian = attribute.to_array(merged_jacobian_list, rows=jacobian_num_rows, cols=jacobian_num_cols)
      unioned_child.correspondance.addAttribute(child_jacobian_name, computed_attribute=merged_jacobian)

    for unioned_child in unioned_children:
      unioned_child_used_children: List[attribute] = []
      for child in children:
        if child in self.__unioned_child_to_its_children[unioned_child]:
          unioned_child_used_children.append(child)
      if len(unioned_child_used_children) == 0:
        continue

      child_hessian_name = self.__hessianName(
        unioned_child,
        unioned_child_used_children
      )
      if child_hessian_name in unioned_child.correspondance.attributes:
        continue
      child_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in unioned_child_used_children])}'
      child_jacobian = unioned_child.correspondance[child_jacobian_name]
      num_hessians = unioned_child.size
      hessian_num_cols = sum([x.size for x in unioned_child_used_children])
      hessian_num_rows = hessian_num_cols
      all_hessian_results = []
      for row in range(num_hessians):
        for col in range(child_jacobian.cols):
          current_item = child_jacobian[row, col]
          for child in unioned_child_used_children:
            diff_result = autodiff_engine.diff(current_item, child)
            for i in range(diff_result.rows):
              for j in range(diff_result.cols):
                all_hessian_results.append(diff_result[i, j])
      merged_hessian = attribute.to_array(all_hessian_results, rows=hessian_num_rows * num_hessians, cols=hessian_num_cols)

      if self.__save_intermediate and merged_hessian.isZero == 0:
        if child_hessian_name not in unioned_child.correspondance.attributes:
          hessian_data_attribute = unioned_child.correspondance.addAttribute(child_hessian_name, rows=merged_hessian.rows, cols=merged_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            unioned_child.correspondance.addAttribute(new_name, computed_attribute=merged_hessian)
            self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (merged_hessian, hessian_data_attribute)
      else:
        if child_hessian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(child_hessian_name, computed_attribute=merged_hessian)

  def __generateGlobalJacobianForUnion(self, current: attribute, wrt: List[attribute]):
    assert self.__source is not None
    gradient_attribute_name = self.__gradientName(
      current,
      wrt,
      filled_for_source=True
    )
    if gradient_attribute_name in current.correspondance.attributes:
      result = current.correspondance[gradient_attribute_name]
      self.__global_jacobian = result
      return result

    unioned_children_global_jacobians: List[attribute] = []
    children_on_path: List[attribute] = self.__path_dict[current]
    for unioned_child in current.children:
      unioned_child_global_jacobian_name = self.__gradientName(
        unioned_child,
        wrt
      )
      if unioned_child_global_jacobian_name in unioned_child.correspondance.attributes:
        unioned_child_global_jacobian = unioned_child.correspondance[unioned_child_global_jacobian_name]
        unioned_children_global_jacobians.append(unioned_child_global_jacobian)
      else:
        used_children: List[attribute] = []
        for child in children_on_path:
          if child in self.__unioned_child_to_its_children[unioned_child]:
            used_children.append(child)
        if len(used_children) == 0:
          unioned_children_global_jacobians.append(attribute.zeros(unioned_child.size, 1))
          continue

        used_children_global_jacobian_name = self.__combinedGradientName(
          used_children,
          wrt
        )
        if used_children_global_jacobian_name in unioned_child.correspondance.attributes:
          children_global_jacobian = unioned_child.correspondance[used_children_global_jacobian_name]
        else:
          used_children_global_jacobians: List[attribute] = []
          for child in used_children:
            used_children_global_jacobian = self.__generateGradientThroughRecursion(child, wrt)
            used_children_global_jacobians.append(used_children_global_jacobian)

          children_global_jacobian_rows = sum([x.size for x in used_children])
          children_global_jacobian_cols = sum([x.cols for x in used_children_global_jacobians])
          children_global_jacobian_items = [attribute(float_value=0.0) for _ in range(children_global_jacobian_rows * children_global_jacobian_cols)]
          row_offset = 0
          col_offset = 0
          for item in used_children_global_jacobians:
            for i in range(item.rows):
              for j in range(item.cols):
                children_global_jacobian_items[(i + row_offset) * children_global_jacobian_cols + j + col_offset] = item[i, j]
            col_offset += item.cols
            row_offset += item.rows
          children_global_jacobian = attribute.to_array(children_global_jacobian_items, rows=children_global_jacobian_rows, cols=children_global_jacobian_cols)
          if used_children_global_jacobian_name not in unioned_child.correspondance.attributes:
            unioned_child.correspondance.addAttribute(used_children_global_jacobian_name, computed_attribute=children_global_jacobian)

        child_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in used_children])}'
        child_local_jacobian = unioned_child.correspondance[child_jacobian_name]
        unioned_child_global_jacobian = child_local_jacobian.mul_explicit(children_global_jacobian)
        if unioned_child_global_jacobian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(unioned_child_global_jacobian_name, computed_attribute=unioned_child_global_jacobian)
        unioned_children_global_jacobians.append(unioned_child.correspondance[unioned_child_global_jacobian_name])

    if self.__save_intermediate:
      for (index, item) in enumerate(unioned_children_global_jacobians):
        if item.fullName not in self.__intermediate_compute_pairs:
          new_name = item.name + "_evaluated"
          if new_name not in item.correspondance.attributes:
            evaluated_result = item.correspondance.addAttribute(new_name, rows=item.rows, cols=item.cols)
            self.__intermediate_compute_pairs[item.fullName] = (item, evaluated_result)
            unioned_children_global_jacobians[index] = evaluated_result
        else:
          unioned_children_global_jacobians[index] = self.__intermediate_compute_pairs[item.fullName][1]

    max_cols = max([x.cols for x in unioned_children_global_jacobians])
    max_rows = max([x.rows for x in unioned_children_global_jacobians])
    for (index, jacobian) in enumerate(unioned_children_global_jacobians):
      assert jacobian.rows == max_rows, f"differentiator.__generateGlobalJacobianForUnion: jacobian rows {jacobian.rows} is not equal to max rows {max_rows}"
      if jacobian.name == gradient_attribute_name:
        assert jacobian.cols == max_cols, f"differentiator.__generateGlobalJacobianForUnion: jacobian cols {jacobian.cols} is not equal to max cols {max_cols}"
        continue
      expanded_jacobian_list = [0.0 for _ in range(max_rows * max_cols)]
      for i in range(jacobian.rows):
        for j in range(jacobian.cols):
          expanded_jacobian_list[i * max_cols + j] = jacobian[i, j]
      expanded_jacobian = attribute.to_array(expanded_jacobian_list, rows=max_rows, cols=max_cols)
      if gradient_attribute_name not in current.children[index].correspondance.attributes:
        current.children[index].correspondance.addAttribute(gradient_attribute_name, computed_attribute=expanded_jacobian)

    res = current.correspondance.addAttribute(gradient_attribute_name)
    self.__global_jacobian = res
    return res

  def __generateGlobalHessianForUnion(self, current: attribute, wrt: List[attribute]) -> attribute:
    assert self.__source is not None
    global_hessian_name = self.__hessianName(
      current,
      wrt,
      filled_for_source=True
    )
    if global_hessian_name in current.correspondance.attributes:
      return current.correspondance[global_hessian_name]

    unioned_children_global_hessians: List[attribute] = []
    children_on_path: List[attribute] = self.__path_dict[current]
    for unioned_child in current.children:
      unioned_child_global_hessian_name = self.__hessianName(
        unioned_child,
        wrt
      )
      if unioned_child_global_hessian_name in unioned_child.correspondance.attributes:
        unioned_child_global_hessian = unioned_child.correspondance[unioned_child_global_hessian_name]
        unioned_children_global_hessians.append(unioned_child_global_hessian)
      else:
        used_children: List[attribute] = []
        for child in children_on_path:
          if child in self.__unioned_child_to_its_children[unioned_child]:
            used_children.append(child)
        if len(used_children) == 0:
          unioned_children_global_hessians.append(attribute.zeros(current.size, 1))
          continue

        num_hessians = unioned_child.size
        next_children = used_children
        unioned_child_local_jacobian_name = f'd_{unioned_child.fullName}_d_{"__".join([x.fullName for x in next_children])}'
        unioned_child_local_jacobian = unioned_child.correspondance[unioned_child_local_jacobian_name]
        unioned_child_local_hessian_name = self.__hessianName(
          unioned_child,
          next_children
        )
        unioned_child_local_hessian = unioned_child.correspondance[unioned_child_local_hessian_name]

        next_children_global_jacobian_name = self.__combinedGradientName(
          next_children,
          wrt
        )
        next_children_global_jacobian = unioned_child.correspondance[next_children_global_jacobian_name]

        next_children_global_hessians: List[attribute] = []
        for child in next_children:
          child_global_hessian = self.__generateHessianThroughRecursion(child, wrt)
          next_children_global_hessians.append(child_global_hessian)
        next_children_total_size = sum([x.size for x in next_children])
        local_hessian_size = next_children_total_size * next_children_total_size
        assert local_hessian_size * num_hessians == unioned_child_local_hessian.size, f"differentiator.__generateGlobalHessianForUnion: local hessian size {local_hessian_size} * num hessians {num_hessians} is not equal to unioned child local hessian size {unioned_child_local_hessian.size}"
        local_hessian_rows = int(math.sqrt(local_hessian_size))
        assert local_hessian_rows * local_hessian_rows == local_hessian_size, f"differentiator.__generateGlobalHessianForUnion: local hessian rows {local_hessian_rows} is not equal to local hessian size {local_hessian_size}"
        local_hessian_cols = local_hessian_rows
        unioned_child_global_hessian_items = []
        for N in range(num_hessians):
          nth_joined_child_local_hessian = attribute.to_array([unioned_child_local_hessian[i] for i in range(N * local_hessian_size, (N + 1) * local_hessian_size)], rows=local_hessian_rows, cols=local_hessian_cols)
          first_part_hessian = next_children_global_jacobian.transpose().mul_explicit(nth_joined_child_local_hessian.mul_explicit(next_children_global_jacobian))
          local_gradient_size = unioned_child_local_jacobian.size // num_hessians
          assert local_gradient_size * num_hessians == unioned_child_local_jacobian.size, f"differentiator.__generateGlobalHessianForUnion: local gradient size {local_gradient_size} * num hessians {num_hessians} is not equal to unioned child local jacobian size {unioned_child_local_jacobian.size}"

          second_part_hessian_array = [0.0 for _ in range(next_children_global_jacobian.cols * next_children_global_jacobian.cols)]
          block_offset = 0
          for (index, child) in enumerate(next_children):
            next_child_global_hessian = next_children_global_hessians[index]
            child_size = child.size
            hessian_size = next_child_global_hessian.size // child_size
            assert hessian_size * child_size == next_child_global_hessian.size, f"differentiator.__generateGlobalHessianForUnion: hessian size {hessian_size} * child size {child_size} is not equal to child global hessian size {next_child_global_hessian.size}"
            hessian_rows = int(math.sqrt(hessian_size))
            assert hessian_rows * hessian_rows == hessian_size, f"differentiator.__generateGlobalHessianForUnion: hessian rows {hessian_rows} is not equal to hessian size {hessian_size}"
            hessian_cols = hessian_rows
            for i in range(hessian_rows):
              for j in range(hessian_cols):
                for k in range(child_size):
                  second_part_hessian_array[(block_offset + i) * next_children_global_jacobian.cols + (block_offset + j)] += unioned_child_local_jacobian[N * unioned_child_local_jacobian.cols + k + block_offset] * next_child_global_hessian[k * hessian_size + i * hessian_cols + j]
            block_offset += child_size
          second_part_hessian = attribute.to_array(second_part_hessian_array, rows=next_children_global_jacobian.cols, cols=next_children_global_jacobian.cols)
          final_hessian = first_part_hessian.add_explicit(second_part_hessian)
          for i in range(final_hessian.size):
            unioned_child_global_hessian_items.append(final_hessian[i])
        unioned_child_global_hessian = attribute.to_array(unioned_child_global_hessian_items, rows=num_hessians * next_children_global_jacobian.cols, cols=next_children_global_jacobian.cols)
        if unioned_child_global_hessian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(unioned_child_global_hessian_name, computed_attribute=unioned_child_global_hessian)
        unioned_children_global_hessians.append(unioned_child.correspondance[unioned_child_global_hessian_name])

    if self.__save_intermediate:
      for (index, item) in enumerate(unioned_children_global_hessians):
        if item.fullName not in self.__intermediate_compute_pairs:
          new_name = item.name + "_evaluated"
          if new_name not in item.correspondance.attributes:
            evaluated_result = item.correspondance.addAttribute(new_name, rows=item.rows, cols=item.cols)
            self.__intermediate_compute_pairs[item.fullName] = (item, evaluated_result)
            unioned_children_global_hessians[index] = evaluated_result
        else:
          unioned_children_global_hessians[index] = self.__intermediate_compute_pairs[item.fullName][1]

    largest_cols = max([x.cols for x in unioned_children_global_hessians])
    largest_rows = max([x.rows for x in unioned_children_global_hessians])
    assert largest_rows % largest_cols == 0, f"differentiator.__generateGlobalHessianForUnion: largest rows {largest_rows} is not divisible by largest cols {largest_cols}"

    for (index, unioned_child) in enumerate(current.children):
      expanded_hessian_items = [0.0 for _ in range(largest_rows * largest_cols)]
      num_hessians = unioned_child.size
      unexpanded_hessian = unioned_children_global_hessians[index]
      hessian_size = unexpanded_hessian.size // num_hessians
      assert hessian_size * num_hessians == unexpanded_hessian.size, f"differentiator.__generateGlobalHessianForUnion: hessian size {hessian_size} * num hessians {num_hessians} is not equal to unexpanded hessian size {unexpanded_hessian.size}"
      assert largest_rows // largest_cols == num_hessians, f"differentiator.__generateGlobalHessianForUnion: largest rows {largest_rows} // largest cols {largest_cols} is not equal to num hessians {num_hessians}"
      for i in range(num_hessians):
        for j in range(unexpanded_hessian.cols):
          for k in range(unexpanded_hessian.cols):
            expanded_hessian_items[i * largest_cols * largest_cols + j * largest_cols + k] = unexpanded_hessian[i * hessian_size + j * unexpanded_hessian.cols + k]
      expanded_hessian = attribute.to_array(expanded_hessian_items, rows=largest_rows, cols=largest_cols)
      if self.__save_intermediate:
        if global_hessian_name not in unioned_child.correspondance.attributes:
          hessian_data_attribute = unioned_child.correspondance.addAttribute(global_hessian_name, rows=expanded_hessian.rows, cols=expanded_hessian.cols)
          if hessian_data_attribute.fullName not in self.__intermediate_compute_pairs:
            new_name = hessian_data_attribute.name + "_pre_evaluated"
            if new_name not in unioned_child.correspondance.attributes:
              unioned_child.correspondance.addAttribute(new_name, computed_attribute=expanded_hessian)
              self.__intermediate_compute_pairs[hessian_data_attribute.fullName] = (expanded_hessian, hessian_data_attribute)
      else:
        if global_hessian_name not in unioned_child.correspondance.attributes:
          unioned_child.correspondance.addAttribute(global_hessian_name, computed_attribute=expanded_hessian)

    res = current.correspondance.addAttribute(global_hessian_name)
    return res

  def __generateHessianThroughPathDict(self, wrt: List[attribute], autodiff_engine: autodiff) -> None:

    assert self.__source is not None
    source_hessian_name = self.__hessianName(self.__source, wrt)
    if source_hessian_name in self.__source.correspondance.attributes:
      from yasps.attribute import SPD
      self.__hessian = self.__source.correspondance.attributes[
        source_hessian_name
      ]
      assert self.__hessian is not None
      # ok this is a temp fix
      # this is for scenario that the children of the energy is just data, there's no join nor anything
      # in this case it is very likely when we compute the jacobian we already computed the hessian
      # so let's just determine the projection cases here
      if self.__projection_method >= 0 and self.__hessian.operator != SPD:
        self.__hessian = self.__hessian.spd(self.__projection_method)
        self.__project_entire_hessian = False
      return
    self.__hessian = self.__generateHessianThroughRecursion(self.__source, wrt)

  def __generateHessianThroughRecursion(self, current: attribute, wrt: List[attribute]) -> attribute:
    from yasps.attribute import JOIN, UNION, DATA, CONSTANT
    if current.operator == DATA or current.operator == CONSTANT:
      return attribute.zeros(current.size, current.size * current.size)
    hessian_attribute_name = self.__hessianName(current, wrt)
    if hessian_attribute_name in current.correspondance.attributes:
      return current.correspondance[hessian_attribute_name]
    if current.operator != JOIN and current.operator != UNION:
      return self.__generateGlobalHessianForEnergy(current, wrt)
    if current.operator == JOIN:
      return self.__generateGlobalHessianForJoin(current, wrt)
    return self.__generateGlobalHessianForUnion(current, wrt)

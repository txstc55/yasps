from yasps.attribute import attribute
from typing import List, Tuple, Set, Optional
from yasps.hessian import hessian
from yasps.path import path
from yasps.gradientIndicesKernel import gradientIndicesKernel
class differentiator:
  def __init__(self):
    pass

  def diff1(self, source: List[attribute], global_targets: List[attribute], local_targets: List[attribute] = [], dynamic_instances = False):
    # diff1 is used for gradient or first order jacobian
    pass

  def diff2(self, source: List[attribute], target1: List[attribute], target2: List[attribute], local_targets: List[attribute] = [], projection_method = 1, save_intermediate = False, separate_hessian_jacobian = False, dynamic_instances = False):
    # diff2 is used for hessian
    pass

  def __diff2_hessian_all(self, source: List[attribute], global_targets: List[attribute], local_targets: List[attribute] = [], projection_method = 1, save_intermediate = False, separate_hessian_jacobian = False, dynamic_instances = False):
    pass






  #########################################################
  ## Hessian differentiation, each differentiation
  ## will return us a Hessian matrix object
  #########################################################
  def __diff2_hessian_single(self, source: attribute, global_targets: List[attribute], local_targets: List[attribute] = [], projection_method = 1, save_intermediate = False, separate_hessian_jacobian = False, dynamic_instances = False) -> hessian:
    hessian_local = hessian(global_targets, local_targets, dynamic_instances)
    wrt_start_indices = hessian_local.wrt_start_indices # this is computed when hessian is initialized
    paths = path(global_targets, local_targets)
    paths.getRoots(source, [source]) # get the roots and store it in the path object
    paths.getPathDict() # get the path in a dictionary format, this will be used for differentiation

    # initialize the indices kernels
    if dynamic_instances
      hessian_local.indicies_kernels_dynamic = [gradientIndicesKernel(paths.path_dict, paths.unioned_child_to_its_children, paths.wrt, source)] # set the indices kernel, we will need it to compute the index for placement
    else:
      hessian_local.indices_kernels = [gradientIndicesKernel(paths.path_dict, paths.unioned_child_to_its_children, paths.wrt, source)]

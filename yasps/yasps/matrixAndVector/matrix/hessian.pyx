from __future__ import annotations
from typing import Optional, List, Dict, Tuple

from yasps.matrix import matrix
from yasps.gradient import gradient
from yasps.hessianAndGradientKernel import hessianAndGradientKernel
from yasps.coordinateCompressionKernel import coordinateCompressionKernel
from yasps.attribute import attribute
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.codeGenerator import codeGenerator
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from yasps.helper import timed
from yasps.placementReorderKernel import placementReorderKernel


class hessian(matrix):
  """
  Hessian stores how to assemble one or more symbolic second-order terms.
  The actual numeric gradient/vector buffers are only created when compute() runs.
  """

  def __init__(self, wrt: List[attribute], local_targets: List[attribute] = [], dynamic_instances = False):
    total_size = 0
    self.__wrt: List[attribute] = []
    for item in wrt:
      if item.isDynamic:
        raise ValueError("hessian.__init__: wrt can not contain dynamic attributes.")
      self.__wrt.append(item)
      total_size += item.correspondance.numInstances * item.size

    if total_size <= 0:
      raise ValueError(f"hessian.__init__: total size {total_size} must be positive.")

    super().__init__(total_size, total_size, symmetric_storage=True)

    self.__dynamic_instances: bool = dynamic_instances
    self.__gradient: Optional[gradient] = None
    self.__local_targets: List[List[attribute]] = [list(local_targets)]
    self.__wrt_start_indices: List[int] = []
    self.__gradient_segments_start_cpu: List[int] = []
    self.__gradient_segments_start: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.uint32)
    self.__is_setup: bool = False

    # Numeric buffers are allocated lazily because many symbolic Hessians are never computed.
    self.__diagonal: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__diagonal_blocks: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__diagonal_blocks_inverse: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__diagonal_blocks_start: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.uint32)
    self.__diagonal_blocks_start_cpu: List[int] = []
    self.__diagonal_blocks_local_sizes: List[int] = []
    self.__compute_wrt_start_indices()

    # Each list entry corresponds to one symbolic contribution before Hessian addition.
    self.__global_gradients: List[attribute] = []
    self.__global_hessians: List[Optional[attribute]] = []
    self.__gradient_only: List[bool] = []
    self.__global_jacobians: List[Optional[attribute]] = []
    self.__global_inner_hessians: List[Optional[attribute]] = []
    self.__project_entire_hessian: List[bool] = []
    self.__projection_methods: List[int] = []
    self.__separate_hessian_jacobian: List[bool] = []
    self.__intermediate_compute_pairs: List[Dict[str, Tuple[attribute, attribute]]] = []
    self.__merged_hessian_and_gradient_attributes: List[Optional[attribute]] = []
    self.__hessian_and_gradient_kernels: List[Optional[hessianAndGradientKernel]] = []
    self.__sources: List[attribute] = []

    # for separation of hessian and jacobian
    self.__global_jacobian_block_nonzero_attributes: List[List[attribute]] = []
    self.__global_jacobian_block_nonzero_local_positions: List[List[int]] = []
    self.__global_jacobian_children_sizes: List[List[int]] = []
    self.__global_jacobian_children_spans: List[List[int]] = []
    self.__placement_reorder_kernels: List[placementReorderKernel] = []

    self.__global_gradients_dynamic: List[attribute] = []
    self.__global_hessians_dynamic: List[Optional[attribute]] = []
    self.__gradient_only_dynamic: List[bool] = []
    self.__global_jacobians_dynamic: List[Optional[attribute]] = []
    self.__global_inner_hessians_dynamic: List[Optional[attribute]] = []
    self.__project_entire_hessian_dynamic: List[bool] = []
    self.__projection_methods_dynamic: List[int] = []
    self.__separate_hessian_jacobian_dynamic: List[bool] = []
    self.__intermediate_compute_pairs_dynamic: List[Dict[str, Tuple[attribute, attribute]]] = []
    self.__merged_hessian_and_gradient_attributes_dynamic: List[Optional[attribute]] = []
    self.__hessian_and_gradient_kernels_dynamic: List[Optional[hessianAndGradientKernel]] = []
    self.__sources_dynamic: List[attribute] = []

    # for separation of hessian and jacobian
    self.__global_jacobian_block_nonzero_attributes_dynamic: List[List[attribute]] = []
    self.__global_jacobian_block_nonzero_local_positions_dynamic: List[List[int]] = []
    self.__global_jacobian_children_sizes_dynamic: List[List[int]] = []
    self.__global_jacobian_children_spans_dynamic: List[List[int]] = []
    self.__placement_reorder_kernels_dynamic: List[placementReorderKernel] = []

    self.__indices_kernels: List[gradientIndicesKernel] = []
    self.__indices_kernels_dynamic: List[gradientIndicesKernel] = []


    self.__block_indices_gpu: List[gpuarray.GPUArray] = []
    self.__block_indices_gpu_dynamic: List[gpuarray.GPUArray] = []

    self.__compression_kernel: Optional[coordinateCompressionKernel] = None
    self.__compression_kernel_dynamic: Optional[coordinateCompressionKernel] = None

  def __compute_wrt_start_indices(self):
    gradient_sizes = [item.size * item.correspondance.numInstances for item in self.__wrt]
    diagonal_block_sizes = [item.size * item.size * item.correspondance.numInstances for item in self.__wrt]

    gradient_segment_start = [0]
    diagonal_block_start = [0]
    for i in range(1, len(gradient_sizes)):
      gradient_segment_start.append(gradient_segment_start[i - 1] + gradient_sizes[i - 1])
      diagonal_block_start.append(diagonal_block_start[i - 1] + diagonal_block_sizes[i - 1])
    if len(gradient_sizes) > 0:
      gradient_segment_start.append(gradient_segment_start[-1] + gradient_sizes[-1])
      diagonal_block_start.append(diagonal_block_start[-1] + diagonal_block_sizes[-1])

    self.__wrt_start_indices = gradient_segment_start
    self.__gradient_segments_start_cpu = gradient_segment_start
    self.__diagonal_blocks_start_cpu = diagonal_block_start
    self.__diagonal_blocks_local_sizes = diagonal_block_sizes

  @property
  def sources(self) -> List[attribute]:
    return self.__sources

  @property
  def sources_dynamic(self) -> List[attribute]:
    return self.__sources_dynamic

  @property
  def wrt_start_indices(self) -> List[int]:
    return self.__wrt_start_indices

  @property
  def wrt(self) -> List[attribute]:
    return self.__wrt

  @property
  def local_targets(self) -> List[List[attribute]]:
    return self.__local_targets

  @sources.setter
  def sources(self, sources: List[attribute]) -> None:
    self.__sources = sources

  @sources_dynamic.setter
  def sources_dynamic(self, sources: List[attribute]) -> None:
    self.__sources_dynamic = sources

  @local_targets.setter
  def local_targets(self, value: List[List[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.local_targets: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.local_targets: each item must be a list.")
      if any(not isinstance(att, attribute) for att in item):
        raise TypeError("hessian.local_targets: all nested items must be yasps.attribute.attribute.")
    self.__local_targets = value

  @property
  def dynamic_instances(self) -> bool:
    return self.__dynamic_instances

  @property
  def gradient(self) -> Optional[gradient]:
    return self.__gradient

  @gradient.setter
  def gradient(self, value: Optional[gradient]) -> None:
    if value is None:
      self.__gradient = None
      return
    if not isinstance(value, gradient):
      raise TypeError("hessian.gradient: value must be yasps.gradient.gradient.")
    if value.size != self.cols:
      raise ValueError(f"hessian.gradient: gradient size {value.size} does not match hessian size {self.cols}.")
    value.hessian = self
    self.__gradient = value

  @property
  def diagonal(self) -> gpuarray.GPUArray:
    return self.__diagonal

  @property
  def diagonal_blocks(self) -> gpuarray.GPUArray:
    return self.__diagonal_blocks

  @property
  def diagonal_blocks_inverse(self) -> gpuarray.GPUArray:
    return self.__diagonal_blocks_inverse

  @property
  def diagonal_blocks_start(self) -> gpuarray.GPUArray:
    return self.__diagonal_blocks_start

  @property
  def diagonal_blocks_start_cpu(self) -> List[int]:
    return self.__diagonal_blocks_start_cpu

  @property
  def diagonal_blocks_local_sizes(self) -> List[int]:
    return self.__diagonal_blocks_local_sizes

  @property
  def gradient_segments_start(self) -> gpuarray.GPUArray:
    return self.__gradient_segments_start

  @property
  def gradient_segments_start_cpu(self) -> List[int]:
    return self.__gradient_segments_start_cpu

  @property
  def global_gradients(self) -> List[attribute]:
    return self.__global_gradients

  @global_gradients.setter
  def global_gradients(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_gradients: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_gradients: all items must be yasps.attribute.attribute.")
    self.__global_gradients = value

  @property
  def global_hessians(self) -> List[Optional[attribute]]:
    return self.__global_hessians

  @global_hessians.setter
  def global_hessians(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_hessians: value must be a list.")
    if any(item is not None and not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_hessians: all items must be attributes or None.")
    self.__global_hessians = value

  @property
  def gradient_only(self) -> List[bool]:
    return self.__gradient_only

  @gradient_only.setter
  def gradient_only(self, value: List[bool]) -> None:
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.gradient_only: value must be a list of bool.")
    self.__gradient_only = list(value)

  @property
  def global_jacobians(self) -> List[Optional[attribute]]:
    return self.__global_jacobians

  @global_jacobians.setter
  def global_jacobians(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobians: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, attribute):
        raise TypeError("hessian.global_jacobians: items must be attributes or None.")
    self.__global_jacobians = value

  @property
  def global_inner_hessians(self) -> List[Optional[attribute]]:
    return self.__global_inner_hessians

  @global_inner_hessians.setter
  def global_inner_hessians(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_inner_hessians: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, attribute):
        raise TypeError("hessian.global_inner_hessians: items must be attributes or None.")
    self.__global_inner_hessians = value

  @property
  def project_entire_hessian(self) -> List[bool]:
    return self.__project_entire_hessian

  @project_entire_hessian.setter
  def project_entire_hessian(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.project_entire_hessian: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.project_entire_hessian: all items must be bool.")
    self.__project_entire_hessian = value

  @property
  def projection_methods(self) -> List[int]:
    return self.__projection_methods

  @projection_methods.setter
  def projection_methods(self, value: List[int]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.projection_methods: value must be a list.")
    if any(not isinstance(item, int) for item in value):
      raise TypeError("hessian.projection_methods: all items must be int.")
    self.__projection_methods = value

  @property
  def separate_hessian_jacobian(self) -> List[bool]:
    return self.__separate_hessian_jacobian

  @separate_hessian_jacobian.setter
  def separate_hessian_jacobian(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.separate_hessian_jacobian: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.separate_hessian_jacobian: all items must be bool.")
    self.__separate_hessian_jacobian = value

  @property
  def intermediate_compute_pairs(self) -> List[Dict[str, Tuple[attribute, attribute]]]:
    return self.__intermediate_compute_pairs

  @intermediate_compute_pairs.setter
  def intermediate_compute_pairs(self, value: List[Dict[str, Tuple[attribute, attribute]]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.intermediate_compute_pairs: value must be a list.")
    self.__intermediate_compute_pairs = value

  @property
  def merged_hessian_and_gradient_attributes(self) -> List[Optional[attribute]]:
    return self.__merged_hessian_and_gradient_attributes

  @merged_hessian_and_gradient_attributes.setter
  def merged_hessian_and_gradient_attributes(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.merged_hessian_and_gradient_attributes: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, attribute):
        raise TypeError("hessian.merged_hessian_and_gradient_attributes: items must be attributes or None.")
    self.__merged_hessian_and_gradient_attributes = value

  @property
  def hessian_and_gradient_kernels(self) -> List[Optional[hessianAndGradientKernel]]:
    return self.__hessian_and_gradient_kernels

  @hessian_and_gradient_kernels.setter
  def hessian_and_gradient_kernels(self, value: List[Optional[hessianAndGradientKernel]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.hessian_and_gradient_kernels: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, hessianAndGradientKernel):
        raise TypeError("hessian.hessian_and_gradient_kernels: items must be hessianAndGradientKernel or None.")
    self.__hessian_and_gradient_kernels = value

  @property
  def global_jacobian_block_nonzero_attributes(self) -> List[List[attribute]]:
    return self.__global_jacobian_block_nonzero_attributes

  @global_jacobian_block_nonzero_attributes.setter
  def global_jacobian_block_nonzero_attributes(self, value: List[List[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_block_nonzero_attributes: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_block_nonzero_attributes: each item must be a list.")
      if any(not isinstance(att, attribute) for att in item):
        raise TypeError("hessian.global_jacobian_block_nonzero_attributes: nested items must be attributes.")
    self.__global_jacobian_block_nonzero_attributes = value

  @property
  def global_jacobian_block_nonzero_local_positions(self) -> List[List[int]]:
    return self.__global_jacobian_block_nonzero_local_positions

  @global_jacobian_block_nonzero_local_positions.setter
  def global_jacobian_block_nonzero_local_positions(self, value: List[List[int]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_block_nonzero_local_positions: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_block_nonzero_local_positions: each item must be a list.")
      if any(not isinstance(position, int) for position in item):
        raise TypeError("hessian.global_jacobian_block_nonzero_local_positions: nested items must be int.")
    self.__global_jacobian_block_nonzero_local_positions = value

  @property
  def global_jacobian_children_sizes(self) -> List[List[int]]:
    return self.__global_jacobian_children_sizes

  @global_jacobian_children_sizes.setter
  def global_jacobian_children_sizes(self, value: List[List[int]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_children_sizes: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_children_sizes: each item must be a list.")
      if any(not isinstance(size, int) for size in item):
        raise TypeError("hessian.global_jacobian_children_sizes: nested items must be int.")
    self.__global_jacobian_children_sizes = value

  @property
  def global_jacobian_children_spans(self) -> List[List[int]]:
    return self.__global_jacobian_children_spans

  @global_jacobian_children_spans.setter
  def global_jacobian_children_spans(self, value: List[List[int]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_children_spans: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_children_spans: each item must be a list.")
      if any(not isinstance(span, int) for span in item):
        raise TypeError("hessian.global_jacobian_children_spans: nested items must be int.")
    self.__global_jacobian_children_spans = value


  @property
  def global_gradients_dynamic(self) -> List[attribute]:
    return self.__global_gradients_dynamic

  @global_gradients_dynamic.setter
  def global_gradients_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_gradients_dynamic: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_gradients_dynamic: all items must be yasps.attribute.attribute.")
    self.__global_gradients_dynamic = value

  @property
  def global_hessians_dynamic(self) -> List[Optional[attribute]]:
    return self.__global_hessians_dynamic

  @global_hessians_dynamic.setter
  def global_hessians_dynamic(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_hessians_dynamic: value must be a list.")
    if any(item is not None and not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_hessians_dynamic: all items must be attributes or None.")
    self.__global_hessians_dynamic = value

  @property
  def gradient_only_dynamic(self) -> List[bool]:
    return self.__gradient_only_dynamic

  @gradient_only_dynamic.setter
  def gradient_only_dynamic(self, value: List[bool]) -> None:
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.gradient_only_dynamic: value must be a list of bool.")
    self.__gradient_only_dynamic = list(value)

  @property
  def global_jacobians_dynamic(self) -> List[Optional[attribute]]:
    return self.__global_jacobians_dynamic

  @global_jacobians_dynamic.setter
  def global_jacobians_dynamic(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobians_dynamic: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, attribute):
        raise TypeError("hessian.global_jacobians_dynamic: items must be attributes or None.")
    self.__global_jacobians_dynamic = value

  @property
  def global_inner_hessians_dynamic(self) -> List[Optional[attribute]]:
    return self.__global_inner_hessians_dynamic

  @global_inner_hessians_dynamic.setter
  def global_inner_hessians_dynamic(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_inner_hessians_dynamic: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, attribute):
        raise TypeError("hessian.global_inner_hessians_dynamic: items must be attributes or None.")
    self.__global_inner_hessians_dynamic = value

  @property
  def project_entire_hessian_dynamic(self) -> List[bool]:
    return self.__project_entire_hessian_dynamic

  @project_entire_hessian_dynamic.setter
  def project_entire_hessian_dynamic(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.project_entire_hessian_dynamic: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.project_entire_hessian_dynamic: all items must be bool.")
    self.__project_entire_hessian_dynamic = value

  @property
  def projection_methods_dynamic(self) -> List[int]:
    return self.__projection_methods_dynamic

  @projection_methods_dynamic.setter
  def projection_methods_dynamic(self, value: List[int]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.projection_methods_dynamic: value must be a list.")
    if any(not isinstance(item, int) for item in value):
      raise TypeError("hessian.projection_methods_dynamic: all items must be int.")
    self.__projection_methods_dynamic = value

  @property
  def separate_hessian_jacobian_dynamic(self) -> List[bool]:
    return self.__separate_hessian_jacobian_dynamic

  @separate_hessian_jacobian_dynamic.setter
  def separate_hessian_jacobian_dynamic(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.separate_hessian_jacobian_dynamic: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.separate_hessian_jacobian_dynamic: all items must be bool.")
    self.__separate_hessian_jacobian_dynamic = value

  @property
  def intermediate_compute_pairs_dynamic(self) -> List[Dict[str, Tuple[attribute, attribute]]]:
    return self.__intermediate_compute_pairs_dynamic

  @intermediate_compute_pairs_dynamic.setter
  def intermediate_compute_pairs_dynamic(self, value: List[Dict[str, Tuple[attribute, attribute]]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.intermediate_compute_pairs_dynamic: value must be a list.")
    self.__intermediate_compute_pairs_dynamic = value

  @property
  def merged_hessian_and_gradient_attributes_dynamic(self) -> List[Optional[attribute]]:
    return self.__merged_hessian_and_gradient_attributes_dynamic

  @merged_hessian_and_gradient_attributes_dynamic.setter
  def merged_hessian_and_gradient_attributes_dynamic(self, value: List[Optional[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.merged_hessian_and_gradient_attributes_dynamic: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, attribute):
        raise TypeError("hessian.merged_hessian_and_gradient_attributes_dynamic: items must be attributes or None.")
    self.__merged_hessian_and_gradient_attributes_dynamic = value

  @property
  def hessian_and_gradient_kernels_dynamic(self) -> List[Optional[hessianAndGradientKernel]]:
    return self.__hessian_and_gradient_kernels_dynamic

  @hessian_and_gradient_kernels_dynamic.setter
  def hessian_and_gradient_kernels_dynamic(self, value: List[Optional[hessianAndGradientKernel]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.hessian_and_gradient_kernels_dynamic: value must be a list.")
    for item in value:
      if item is not None and not isinstance(item, hessianAndGradientKernel):
        raise TypeError("hessian.hessian_and_gradient_kernels_dynamic: items must be hessianAndGradientKernel or None.")
    self.__hessian_and_gradient_kernels_dynamic = value

  @property
  def global_jacobian_block_nonzero_attributes_dynamic(self) -> List[List[attribute]]:
    return self.__global_jacobian_block_nonzero_attributes_dynamic

  @global_jacobian_block_nonzero_attributes_dynamic.setter
  def global_jacobian_block_nonzero_attributes_dynamic(self, value: List[List[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_block_nonzero_attributes_dynamic: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_block_nonzero_attributes_dynamic: each item must be a list.")
      if any(not isinstance(att, attribute) for att in item):
        raise TypeError("hessian.global_jacobian_block_nonzero_attributes_dynamic: nested items must be attributes.")
    self.__global_jacobian_block_nonzero_attributes_dynamic = value

  @property
  def global_jacobian_block_nonzero_local_positions_dynamic(self) -> List[List[int]]:
    return self.__global_jacobian_block_nonzero_local_positions_dynamic

  @global_jacobian_block_nonzero_local_positions_dynamic.setter
  def global_jacobian_block_nonzero_local_positions_dynamic(self, value: List[List[int]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_block_nonzero_local_positions_dynamic: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_block_nonzero_local_positions_dynamic: each item must be a list.")
      if any(not isinstance(position, int) for position in item):
        raise TypeError("hessian.global_jacobian_block_nonzero_local_positions_dynamic: nested items must be int.")
    self.__global_jacobian_block_nonzero_local_positions_dynamic = value

  @property
  def global_jacobian_children_sizes_dynamic(self) -> List[List[int]]:
    return self.__global_jacobian_children_sizes_dynamic

  @global_jacobian_children_sizes_dynamic.setter
  def global_jacobian_children_sizes_dynamic(self, value: List[List[int]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_children_sizes_dynamic: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_children_sizes_dynamic: each item must be a list.")
      if any(not isinstance(size, int) for size in item):
        raise TypeError("hessian.global_jacobian_children_sizes_dynamic: nested items must be int.")
    self.__global_jacobian_children_sizes_dynamic = value

  @property
  def global_jacobian_children_spans_dynamic(self) -> List[List[int]]:
    return self.__global_jacobian_children_spans_dynamic

  @global_jacobian_children_spans_dynamic.setter
  def global_jacobian_children_spans_dynamic(self, value: List[List[int]]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobian_children_spans_dynamic: value must be a list.")
    for item in value:
      if not isinstance(item, list):
        raise TypeError("hessian.global_jacobian_children_spans_dynamic: each item must be a list.")
      if any(not isinstance(span, int) for span in item):
        raise TypeError("hessian.global_jacobian_children_spans_dynamic: nested items must be int.")
    self.__global_jacobian_children_spans_dynamic = value


  @property
  def indices_kernels(self) -> List[gradientIndicesKernel]:
    return self.__indices_kernels

  @indices_kernels.setter
  def indices_kernels(self, value: List[gradientIndicesKernel]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.indices_kernels: value must be a list.")
    if any(not isinstance(item, gradientIndicesKernel) for item in value):
      raise TypeError("hessian.indices_kernels: all items must be gradientIndicesKernel.")
    self.__indices_kernels = value

  @property
  def indices_kernels_dynamic(self) -> List[gradientIndicesKernel]:
    return self.__indices_kernels_dynamic

  @indices_kernels_dynamic.setter
  def indices_kernels_dynamic(self, value: List[gradientIndicesKernel]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.indices_kernels_dynamic: value must be a list.")
    if any(not isinstance(item, gradientIndicesKernel) for item in value):
      raise TypeError("hessian.indices_kernels_dynamic: all items must be gradientIndicesKernel.")
    self.__indices_kernels_dynamic = value

  @property
  def block_indices_gpu(self) -> List[gpuarray.GPUArray]:
    return self.__block_indices_gpu

  @block_indices_gpu.setter
  def block_indices_gpu(self, value: List[gpuarray.GPUArray]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.block_indices_gpu: value must be a list.")
    if any(not isinstance(item, gpuarray.GPUArray) for item in value):
      raise TypeError("hessian.block_indices_gpu: all items must be GPUArray.")
    self.__block_indices_gpu = value

  @property
  def block_indices_gpu_dynamic(self) -> List[gpuarray.GPUArray]:
    return self.__block_indices_gpu_dynamic

  @block_indices_gpu_dynamic.setter
  def block_indices_gpu_dynamic(self, value: List[gpuarray.GPUArray]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.block_indices_gpu_dynamic: value must be a list.")
    if any(not isinstance(item, gpuarray.GPUArray) for item in value):
      raise TypeError("hessian.block_indices_gpu_dynamic: all items must be GPUArray.")
    self.__block_indices_gpu_dynamic = value

  @property
  def placement_reorder_kernels(self) -> List[placementReorderKernel]:
    return self.__placement_reorder_kernels

  @placement_reorder_kernels.setter
  def placement_reorder_kernels(self, value: List[placementReorderKernel]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.placement_reorder_kernels: value must be a list.")
    if any(not isinstance(item, placementReorderKernel) for item in value):
      raise TypeError("hessian.placement_reorder_kernels: all items must be placementReorderKernel.")
    self.__placement_reorder_kernels = value

  @property
  def placement_reorder_kernels_dynamic(self) -> List[placementReorderKernel]:
    return self.__placement_reorder_kernels_dynamic

  @placement_reorder_kernels_dynamic.setter
  def placement_reorder_kernels_dynamic(self, value: List[placementReorderKernel]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.placement_reorder_kernels_dynamic: value must be a list.")
    if any(not isinstance(item, placementReorderKernel) for item in value):
      raise TypeError("hessian.placement_reorder_kernels_dynamic: all items must be placementReorderKernel.")
    self.__placement_reorder_kernels_dynamic = value

  def __add__(self, other: hessian):
    if not isinstance(other, hessian):
      raise TypeError(f"hessian.__add__: unsupported operand type(s) for +: 'hessian' and '{type(other).__name__}'")
    if len(self.__wrt) != len(other.wrt):
      raise ValueError("hessian.__add__: wrt length mismatch.")
    for left, right in zip(self.__wrt, other.wrt):
      if left.hash != right.hash:
        raise ValueError("hessian.__add__: wrt mismatch.")

    result = hessian(self.__wrt)
    result.local_targets = self.__local_targets + other.local_targets
    result.global_gradients = self.__global_gradients + other.global_gradients
    result.global_hessians = self.__global_hessians + other.global_hessians
    result.gradient_only = self.__gradient_only + other.gradient_only
    result.global_jacobians = self.__global_jacobians + other.global_jacobians
    result.global_inner_hessians = self.__global_inner_hessians + other.global_inner_hessians
    result.project_entire_hessian = self.__project_entire_hessian + other.project_entire_hessian
    result.projection_methods = self.__projection_methods + other.projection_methods
    result.separate_hessian_jacobian = self.__separate_hessian_jacobian + other.separate_hessian_jacobian
    result.intermediate_compute_pairs = self.__intermediate_compute_pairs + other.intermediate_compute_pairs
    result.merged_hessian_and_gradient_attributes = self.__merged_hessian_and_gradient_attributes + other.merged_hessian_and_gradient_attributes
    result.hessian_and_gradient_kernels = self.__hessian_and_gradient_kernels + other.hessian_and_gradient_kernels
    result.sources = self.__sources + other.sources
    result.global_jacobian_block_nonzero_attributes = self.__global_jacobian_block_nonzero_attributes + other.global_jacobian_block_nonzero_attributes
    result.global_jacobian_block_nonzero_local_positions = self.__global_jacobian_block_nonzero_local_positions + other.global_jacobian_block_nonzero_local_positions
    result.global_jacobian_children_sizes = self.__global_jacobian_children_sizes + other.global_jacobian_children_sizes
    result.global_jacobian_children_spans = self.__global_jacobian_children_spans + other.global_jacobian_children_spans
    result.placement_reorder_kernels = self.__placement_reorder_kernels + other.placement_reorder_kernels

    result.global_gradients_dynamic = self.__global_gradients_dynamic + other.global_gradients_dynamic
    result.global_hessians_dynamic = self.__global_hessians_dynamic + other.global_hessians_dynamic
    result.gradient_only_dynamic = self.__gradient_only_dynamic + other.gradient_only_dynamic
    result.global_jacobians_dynamic = self.__global_jacobians_dynamic + other.global_jacobians_dynamic
    result.global_inner_hessians_dynamic = self.__global_inner_hessians_dynamic + other.global_inner_hessians_dynamic
    result.project_entire_hessian_dynamic = self.__project_entire_hessian_dynamic + other.project_entire_hessian_dynamic
    result.projection_methods_dynamic = self.__projection_methods_dynamic + other.projection_methods_dynamic
    result.separate_hessian_jacobian_dynamic = self.__separate_hessian_jacobian_dynamic + other.separate_hessian_jacobian_dynamic
    result.intermediate_compute_pairs_dynamic = self.__intermediate_compute_pairs_dynamic + other.intermediate_compute_pairs_dynamic
    result.merged_hessian_and_gradient_attributes_dynamic = self.__merged_hessian_and_gradient_attributes_dynamic + other.merged_hessian_and_gradient_attributes_dynamic
    result.hessian_and_gradient_kernels_dynamic = self.__hessian_and_gradient_kernels_dynamic + other.hessian_and_gradient_kernels_dynamic
    result.sources_dynamic = self.__sources_dynamic + other.sources_dynamic
    result.global_jacobian_block_nonzero_attributes_dynamic = self.__global_jacobian_block_nonzero_attributes_dynamic + other.global_jacobian_block_nonzero_attributes_dynamic
    result.global_jacobian_block_nonzero_local_positions_dynamic = self.__global_jacobian_block_nonzero_local_positions_dynamic + other.global_jacobian_block_nonzero_local_positions_dynamic
    result.global_jacobian_children_sizes_dynamic = self.__global_jacobian_children_sizes_dynamic + other.global_jacobian_children_sizes_dynamic
    result.global_jacobian_children_spans_dynamic = self.__global_jacobian_children_spans_dynamic + other.global_jacobian_children_spans_dynamic
    result.placement_reorder_kernels_dynamic = self.__placement_reorder_kernels_dynamic + other.placement_reorder_kernels_dynamic

    result.indices_kernels = self.__indices_kernels + other.indices_kernels
    result.indices_kernels_dynamic = self.__indices_kernels_dynamic + other.indices_kernels_dynamic
    result.block_indices_gpu = self.__block_indices_gpu + other.block_indices_gpu
    result.block_indices_gpu_dynamic = self.__block_indices_gpu_dynamic + other.block_indices_gpu_dynamic
    return result

  @timed("hessian.getSparseIndices")
  def getSparseIndices(self):
    if len(self.__indices_kernels) == 0:
      return

    for item in self.__indices_kernels:
      item.computeIndices(self.__wrt_start_indices)

    self.__compression_kernel = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels],
      [x.outputBlockDimensions for x in self.__indices_kernels],
      [x.numTotalCoordinates for x in self.__indices_kernels],
      self.__wrt
    )
    self.__compression_kernel.compressCoordinatesAndDimensions()
    self.__block_indices_gpu = self.__compression_kernel.lookupArrays

    print("placement reorder kernels size:", len(self.__placement_reorder_kernels))
    print("separate hessian jacobian size:", len(self.__separate_hessian_jacobian))
    print("project entire hessian size:", len(self.__project_entire_hessian))
    for i in range(len(self.__placement_reorder_kernels)):
      if self.__separate_hessian_jacobian[i] and not self.__project_entire_hessian[i]:
        # we will need to initialize the placement reorder kernel and do the computation
        self.__placement_reorder_kernels[i].generateKernel(
          self.__global_jacobian_children_spans[i],
          self.__indices_kernels[i].maxNumIndicesNeeded,
          self.__sources[i]
        )
        self.__placement_reorder_kernels[i].reorderPlacementIndices(
          self.__indices_kernels[i],
          self.__block_indices_gpu[i],
        )

    total_block_size = self.__compression_kernel.totalBlockSize
    if self.blocks_flattened.size < total_block_size:
      self.blocks_flattened = gpuarray.zeros(total_block_size, dtype=np.float64)

    num_unique_dimensions = self.__compression_kernel.numUniqueDimensions
    self.blocks_start_indices = self.__compression_kernel.uniqueDimensionsOuterIndices.get().tolist()[:num_unique_dimensions + 1]
    self.block_positions = self.__compression_kernel.uniqueCoordinates
    self.block_counts = self.__compression_kernel.uniqueDimensionsBlockCounts.get().tolist()[:num_unique_dimensions]
    self.block_dimensions = self.__compression_kernel.uniqueDimensions.get().tolist()[:num_unique_dimensions * 2]

  @timed("hessian.getSparseIndicesDynamic")
  def getSparseIndicesDynamic(self):
    if len(self.__indices_kernels_dynamic) == 0:
      return

    for item in self.__indices_kernels_dynamic:
      item.computeIndices(self.__wrt_start_indices)

    self.__compression_kernel_dynamic = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels_dynamic],
      [x.outputBlockDimensions for x in self.__indices_kernels_dynamic],
      [x.numTotalCoordinates for x in self.__indices_kernels_dynamic],
      self.__wrt
    )
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()

    lookup_arrays = self.__compression_kernel_dynamic.lookupArrays
    self.__block_indices_gpu_dynamic = []
    tmp_count = 0
    for item in self.__indices_kernels_dynamic:
      if item.numTotalCoordinates > 0:
        self.__block_indices_gpu_dynamic.append(lookup_arrays[tmp_count])
        tmp_count += 1
      else:
        self.__block_indices_gpu_dynamic.append(gpuarray.empty(0, dtype=np.uint32))

    for i in range(len(self.__placement_reorder_kernels_dynamic)):
      if self.__separate_hessian_jacobian_dynamic[i] and not self.__project_entire_hessian_dynamic[i]:
        # we will need to initialize the placement reorder kernel and do the computation
        self.__placement_reorder_kernels_dynamic[i].generateKernel(
          self.__global_jacobian_children_spans_dynamic[i],
          self.__indices_kernels_dynamic[i].maxNumIndicesNeeded,
          self.__sources_dynamic[i]
        )
        # reorder the indices if needed
        if (self.__indices_kernels_dynamic[i].numTotalCoordinates > 0):
          self.__placement_reorder_kernels_dynamic[i].reorderPlacementIndices(
            self.__indices_kernels_dynamic[i],
            self.__block_indices_gpu_dynamic[i],
          )

    total_block_size = self.__compression_kernel_dynamic.totalBlockSize
    if self.blocks_flattened_dynamic.size < total_block_size:
      self.blocks_flattened_dynamic = gpuarray.zeros(total_block_size, dtype=np.float64)

    num_unique_dimensions = self.__compression_kernel_dynamic.numUniqueDimensions
    self.blocks_start_indices_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsOuterIndices.get().tolist()[:num_unique_dimensions + 1]
    self.block_positions_dynamic = self.__compression_kernel_dynamic.uniqueCoordinates
    self.block_counts_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsBlockCounts.get().tolist()[:num_unique_dimensions]
    self.block_dimensions_dynamic = self.__compression_kernel_dynamic.uniqueDimensions.get().tolist()[:num_unique_dimensions * 2]

  @timed("hessian.getSparseIndicesDynamicAgain")
  def getSparseIndicesDynamicAgain(self):
    if len(self.__indices_kernels_dynamic) == 0:
      return
    if self.__compression_kernel_dynamic is None:
      self.getSparseIndicesDynamic()
      return

    for item in self.__indices_kernels_dynamic:
      item.computeIndices(self.__wrt_start_indices)

    self.__compression_kernel_dynamic.updateCoordinates(
      [x.outputCoordinates for x in self.__indices_kernels_dynamic],
      [x.outputBlockDimensions for x in self.__indices_kernels_dynamic],
      [x.numTotalCoordinates for x in self.__indices_kernels_dynamic]
    )
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()

    lookup_arrays = self.__compression_kernel_dynamic.lookupArrays
    self.__block_indices_gpu_dynamic = []
    tmp_count = 0
    for item in self.__indices_kernels_dynamic:
      if item.numTotalCoordinates > 0:
        self.__block_indices_gpu_dynamic.append(lookup_arrays[tmp_count])
        tmp_count += 1
      else:
        self.__block_indices_gpu_dynamic.append(gpuarray.empty(0, dtype=np.uint32))

    for i in range(len(self.__placement_reorder_kernels_dynamic)):
      # reorder the indices if needed
      if self.__separate_hessian_jacobian_dynamic[i] and not self.__project_entire_hessian_dynamic[i]:
        if (self.__indices_kernels_dynamic[i].numTotalCoordinates > 0):
          self.__placement_reorder_kernels_dynamic[i].reorderPlacementIndices(
            self.__indices_kernels_dynamic[i],
            self.__block_indices_gpu_dynamic[i],
          )

    total_block_size = self.__compression_kernel_dynamic.totalBlockSize
    if self.blocks_flattened_dynamic.size < total_block_size:
      self.blocks_flattened_dynamic = gpuarray.zeros(total_block_size, dtype=np.float64)

    num_unique_dimensions = self.__compression_kernel_dynamic.numUniqueDimensions
    self.blocks_start_indices_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsOuterIndices.get().tolist()[:num_unique_dimensions + 1]
    self.block_positions_dynamic = self.__compression_kernel_dynamic.uniqueCoordinates
    self.block_counts_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsBlockCounts.get().tolist()[:num_unique_dimensions]
    self.block_dimensions_dynamic = self.__compression_kernel_dynamic.uniqueDimensions.get().tolist()[:num_unique_dimensions * 2]

  def __buildMergedHessianAndGradientAttribute(
    self,
    global_gradient: attribute,
    global_hessian: Optional[attribute],
    gradient_only: bool,
    global_jacobian: Optional[attribute],
    global_inner_hessian: Optional[attribute],
    project_entire_hessian: bool,
    separate_hessian_jacobian: bool,
    source: attribute,
    global_jacobian_block_nonzero_attributes: List[attribute],
  ) -> attribute:
    merged_hessian_and_gradient = []
    merged_hessian_rows = 0
    merged_hessian_cols = 0

    if gradient_only:
      for i in range(global_gradient.size):
        merged_hessian_and_gradient.append(global_gradient[i])
      merged_hessian_rows = 1
      merged_hessian_cols = global_gradient.size
    elif separate_hessian_jacobian and not project_entire_hessian:
      assert global_jacobian is not None
      assert global_inner_hessian is not None
      for i in range(global_inner_hessian.rows):
        for j in range(global_inner_hessian.cols):
          merged_hessian_and_gradient.append(global_inner_hessian[i, j])
      for item in global_jacobian_block_nonzero_attributes:
        merged_hessian_and_gradient.append(item)
      for i in range(global_gradient.size):
        merged_hessian_and_gradient.append(global_gradient[i])
      merged_hessian_rows = 1
      merged_hessian_cols = len(merged_hessian_and_gradient)

    else:
      assert global_hessian is not None
      for i in range(global_hessian.rows):
        for j in range(global_hessian.cols):
          merged_hessian_and_gradient.append(global_hessian[i, j])
      for i in range(global_gradient.size):
        merged_hessian_and_gradient.append(global_gradient[i])
      merged_hessian_rows = len(merged_hessian_and_gradient) // global_gradient.size
      merged_hessian_cols = global_gradient.size
    merged_attribute = attribute.to_array(merged_hessian_and_gradient, rows=merged_hessian_rows, cols=merged_hessian_cols)
    # The symbolic Hessian name contains its full generation configuration.
    # Carry that identity into the merged compute attribute as well so two
    # Hessian modes for the same source and targets cannot collide here.
    derivative_name = global_gradient.name if global_hessian is None else global_hessian.name
    if gradient_only:
      merged_attribute_name = f'hessian_and_gradient_gradient_only_{derivative_name}'
    else:
      merged_attribute_name = f'hessian_and_gradient_{derivative_name}'
    if merged_attribute_name in source.correspondance.attributes:
      return source.correspondance[merged_attribute_name]
    return source.correspondance.addAttribute(
      merged_attribute_name,
      computed_attribute=merged_attribute,
      rows=merged_hessian_rows,
      cols=merged_hessian_cols
    )

  def __ensureTermKernel(self, index: int, dynamic_term = False) -> None:
    if not dynamic_term:
      global_gradients = self.__global_gradients
      global_hessians = self.__global_hessians
      gradient_only = self.__gradient_only
      global_jacobians = self.__global_jacobians
      global_inner_hessians = self.__global_inner_hessians
      project_entire_hessian = self.__project_entire_hessian
      projection_methods = self.__projection_methods
      separate_hessian_jacobian = self.__separate_hessian_jacobian
      merged_attributes = self.__merged_hessian_and_gradient_attributes
      kernels = self.__hessian_and_gradient_kernels
      sources = self.__sources
      indices_kernels = self.__indices_kernels
      global_jacobian_block_nonzero_attributes = self.__global_jacobian_block_nonzero_attributes
      global_jacobian_block_nonzero_local_positions = self.__global_jacobian_block_nonzero_local_positions
      global_jacobian_children_sizes = self.__global_jacobian_children_sizes
      global_jacobian_children_spans = self.__global_jacobian_children_spans
    else:
      global_gradients = self.__global_gradients_dynamic
      global_hessians = self.__global_hessians_dynamic
      gradient_only = self.__gradient_only_dynamic
      global_jacobians = self.__global_jacobians_dynamic
      global_inner_hessians = self.__global_inner_hessians_dynamic
      project_entire_hessian = self.__project_entire_hessian_dynamic
      projection_methods = self.__projection_methods_dynamic
      separate_hessian_jacobian = self.__separate_hessian_jacobian_dynamic
      merged_attributes = self.__merged_hessian_and_gradient_attributes_dynamic
      kernels = self.__hessian_and_gradient_kernels_dynamic
      sources = self.__sources_dynamic
      indices_kernels = self.__indices_kernels_dynamic
      global_jacobian_block_nonzero_attributes = self.__global_jacobian_block_nonzero_attributes_dynamic
      global_jacobian_block_nonzero_local_positions = self.__global_jacobian_block_nonzero_local_positions_dynamic
      global_jacobian_children_sizes = self.__global_jacobian_children_sizes_dynamic
      global_jacobian_children_spans = self.__global_jacobian_children_spans_dynamic


    if (
      index >= len(global_gradients)
      or index >= len(global_hessians)
      or index >= len(gradient_only)
    ):
      raise ValueError("hessian.__ensureTermKernel: symbolic term metadata is incomplete.")

    while len(merged_attributes) <= index:
      merged_attributes.append(None)
    while len(kernels) <= index:
      kernels.append(None)

    if merged_attributes[index] is None:
      merged_attributes[index] = self.__buildMergedHessianAndGradientAttribute(
        global_gradients[index],
        global_hessians[index],
        gradient_only[index],
        global_jacobians[index] if index < len(global_jacobians) else None,
        global_inner_hessians[index] if index < len(global_inner_hessians) else None,
        project_entire_hessian[index],
        separate_hessian_jacobian[index],
        sources[index],
        global_jacobian_block_nonzero_attributes[index]
      )

    if kernels[index] is None:
      assert merged_attributes[index] is not None
      codegen: codeGenerator = codeGenerator(merged_attributes[index])
      codegen.generateCode()
      jacobian_rows = 0
      jacobian_cols = 0
      inner_hessian_rows = 0
      if index < len(global_jacobians) and global_jacobians[index] is not None:
        jacobian_rows = global_jacobians[index].rows
        jacobian_cols = global_jacobians[index].cols
      if index < len(global_inner_hessians) and global_inner_hessians[index] is not None:
        inner_hessian_rows = global_inner_hessians[index].rows
      kernels[index] = hessianAndGradientKernel(
        merged_attributes[index],
        project_entire_hessian[index],
        projection_methods[index],
        gradient_only[index],
        (separate_hessian_jacobian[index] and not project_entire_hessian[index]),
        jacobian_rows,
        jacobian_cols,
        inner_hessian_rows,
        dynamic_term = dynamic_term
      )
      kernels[index].generateKernel(
        indices_kernels[index].outputUniqueGradientSizesCPU.tolist(),
        indices_kernels[index].maxChildGradientSize,
        self.__wrt,
        indices_kernels[index].maxNumIndicesNeeded,
        global_jacobian_block_nonzero_attributes[index],
        global_jacobian_block_nonzero_local_positions[index],
        global_jacobian_children_sizes[index],
        global_jacobian_children_spans[index]
      )

  def __setupCompute(self) -> None:
    if self.__is_setup:
      return

    total_diagonal_block_size = self.__diagonal_blocks_start_cpu[-1] if len(self.__diagonal_blocks_start_cpu) > 0 else 1
    self.__gradient_segments_start = gpuarray.to_gpu(np.array(self.__gradient_segments_start_cpu, dtype=np.uint32))
    self.__diagonal = gpuarray.zeros(self.cols, dtype=np.float64)
    self.__diagonal_blocks_start = gpuarray.to_gpu(np.array(self.__diagonal_blocks_start_cpu, dtype=np.uint32))
    self.__diagonal_blocks = gpuarray.zeros(total_diagonal_block_size, dtype=np.float64)
    self.__diagonal_blocks_inverse = gpuarray.zeros(total_diagonal_block_size, dtype=np.float64)

    if len(self.__indices_kernels) > 0:
      self.getSparseIndices()
    if len(self.__indices_kernels_dynamic) > 0:
      self.getSparseIndicesDynamic()

    # Symbolic term kernels only depend on the differentiated expressions, so we build them once.
    for index in range(len(self.__global_gradients)):
      self.__ensureTermKernel(index, False)
    for index in range(len(self.__global_gradients_dynamic)):
      self.__ensureTermKernel(index, True)

    self.__is_setup = True

  def __computeOneTerm(
    self,
    index: int,
    indices_kernel: gradientIndicesKernel,
    lookup: gpuarray.GPUArray,
    hessian_blocks: gpuarray.GPUArray,
    merged_attributes: List[Optional[attribute]],
    kernels: List[Optional[hessianAndGradientKernel]],
    intermediate_compute_pairs: List[Dict[str, Tuple[attribute, attribute]]],
    is_dynamic
  ) -> None:
    assert self.__gradient is not None

    if index < len(intermediate_compute_pairs):
      for _, value in intermediate_compute_pairs[index].items():
        value[0].compute()
        value[1].updateValue(value[0].value)

    merged_attribute = merged_attributes[index]
    kernel = kernels[index]
    assert merged_attribute is not None
    assert kernel is not None
    # TODO: This doesn't need to be done every iteration for the static parts
    if is_dynamic:
      kernel.generateKernel(
        indices_kernel.outputUniqueGradientSizesCPU.tolist(),
        indices_kernel.maxChildGradientSize,
        self.__wrt,
        indices_kernel.maxNumIndicesNeeded
      )

    counts_gpu = [x.children_primitive_counts_gpu for x in merged_attribute.deviceKernel.kernelPrimitiveUnions]
    arguments: List[gpuarray.GPUArray] = [x.value for x in merged_attribute.deviceKernel.kernelDatas]
    arguments += [x.value for x in merged_attribute.deviceKernel.kernelConnectivity]
    arguments += [x.compressedRows for x in merged_attribute.deviceKernel.kernelConnectivity if x.dimension == 0]
    arguments += counts_gpu
    kernel.compute(
      arguments,
      indices_kernel,
      lookup,
      self.__gradient.value,
      hessian_blocks,
      self.__diagonal,
      self.__diagonal_blocks,
      self.__diagonal_blocks_start,
      self.__gradient_segments_start
    )

  @timed("hessian.compute")
  def compute(self, local_gradient: Optional[gradient] = None):
    if local_gradient is not None:
      self.gradient = local_gradient
    if self.__gradient is None:
      self.__gradient = gradient(self.__wrt, self)
    else:
      self.__gradient.hessian = self

    if not self.__is_setup:
      self.__setupCompute()
    elif len(self.__indices_kernels_dynamic) > 0:
      self.getSparseIndicesDynamicAgain()

    self.__gradient.value.fill(0)
    self.__diagonal.fill(0)
    self.__diagonal_blocks.fill(0)
    self.blocks_flattened.fill(0)
    self.blocks_flattened_dynamic.fill(0)

    for index, indices_kernel in enumerate(self.__indices_kernels):
      if index >= len(self.__block_indices_gpu):
        raise ValueError("hessian.compute: static sparse lookup is missing.")
      self.__computeOneTerm(
        index,
        indices_kernel,
        self.__placement_reorder_kernels[index].reordered_lookups if (self.__separate_hessian_jacobian[index] and not self.__project_entire_hessian[index]) else self.__block_indices_gpu[index],
        self.blocks_flattened,
        self.__merged_hessian_and_gradient_attributes,
        self.__hessian_and_gradient_kernels,
        self.__intermediate_compute_pairs,
        False
      )

    for index, indices_kernel in enumerate(self.__indices_kernels_dynamic):
      if indices_kernel.numTotalCoordinates == 0:
        continue
      if index >= len(self.__block_indices_gpu_dynamic):
        raise ValueError("hessian.compute: dynamic sparse lookup is missing.")
      self.__computeOneTerm(
        index,
        indices_kernel,
        self.__placement_reorder_kernels_dynamic[index].reordered_lookups if (self.__separate_hessian_jacobian_dynamic[index] and not self.__project_entire_hessian_dynamic[index]) else self.__block_indices_gpu_dynamic[index],
        self.blocks_flattened_dynamic,
        self.__merged_hessian_and_gradient_attributes_dynamic,
        self.__hessian_and_gradient_kernels_dynamic,
        self.__intermediate_compute_pairs_dynamic,
        True
      )
    return self

  @property
  def hash(self) -> int:
    return hash(tuple([item.hash for item in self.__wrt]))

  def __hash__(self) -> int:
    return self.hash

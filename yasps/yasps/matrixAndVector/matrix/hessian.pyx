from __future__ import annotations
from typing import Optional, List

from yasps.matrix import matrix
from yasps.gradient import gradient
from yasps.hessianAndGradientKernel import hessianAndGradientKernel
from yasps.coordinateCompressionKernel import coordinateCompressionKernel
from yasps.attribute import attribute
from yasps.gradientIndicesKernel import gradientIndicesKernel
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from yasps.helper import timed # for timing


class hessian(matrix):
  """
  The hessian matrix initialize with a list of wrt
  This will determine the size of the matrix
  As well as becoming an identity for matrix.
  The matrices witht the same wrt can be added up
  """
  def __init__(self, wrt: List[attribute], local_targets: List[attribute] = [], dynamic_instances = False):
    # Hessian is symmetric so it is represented by a single size (square matrix).
    # we will first determine the size of the matrix
    total_size = 0
    for item in wrt:
      total_size += item.correspondance.numInstances * item.size

    if total_size <= 0:
      raise ValueError(f"hessian.__init__: total size {total_size} must be positive.")

    super().__init__(total_size, total_size)

    self.__dynamic_instances = dynamic_instances # if this hessian is dynamic
    self.__gradient: Optional[gradient] = None # we will later on initialize a gradient, because when computing hessian, gradient is free
    self.__wrt: List[attribute] = wrt
    self.__local_targets: List[List[attribute]] = [local_targets]
    self.__wrt_start_indices: List[int] = []

    ##########################################################################
    ## This part will be fixed as long as the wrt is set
    ##########################################################################

    # for diagonal blocks, no matter how you change the Hessian, if the wrt is set, then the diagonal blocks are set
    self.__diagonal_blocks: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store the block diagonal
    self.__diagonal_blocks_inverse: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store the inverse of the diagonal blocks
    self.__diagonal_blocks_start: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store for each attribute, where does the accumulated diagonal block start
    self.__diagonal_blocks_start_cpu: List[int] = []
    self.__diagonal_blocks_local_sizes: List[int] = []
    self.__compute_wrt_start_indices() # allocate some static properties on cpu side

    ##########################################################################
    ## For every hessian, there's a kernel for computing
    ## For every hessian, there's the symbolic information that tells us
    ## how this hessian is computed symbolically
    ## Whenever we do addition, we merge the lists
    ##########################################################################
    # the hessian and gradient kernel is a list, for the same reason that hessians can be added together
    self.__hessian_and_gradient_kernels: List[hessianAndGradientKernel] = []
    self.__hessian_and_gradient_kernels_dynamic: List[hessianAndGradientKernel] = []

    # the symbolic hessian related stuffs
    self.__global_hessians: List[attribute] = []
    self.__global_jacobians: List[attribute] = []
    self.__global_inner_hessians: List[attribute] = []
    self.__project_entire_hessian: List[bool] = []
    self.__projection_methods: List[bool] = []

    # we do the same for the dynamic parts
    self.__global_hessians_dynamic: List[attribute] = []
    self.__global_jacobians_dynamic: List[attribute] = []
    self.__global_inner_hessians_dynamic: List[attribute] = []
    self.__project_entire_hessian_dynamic: List[bool] = []
    self.__projection_methods_dynamic: List[bool] = []

    # for computing the indices
    self.__indices_kernels: List[gradientIndicesKernel] = []
    self.__indices_kernels_dynamic: List[gradientIndicesKernel] = []

    # This is the lookup array, where we check for each block for the energy, how to place it back into the global matrix
    self.__block_indices_gpu: List[gpuarray.GPUArray] = [] # the lookup array for the blocks. We initialize this as an array because we can add up hessians together, and for each hessian there's the lookup array
    self.__block_indices_gpu_dynamic: List[gpuarray.GPUArray] = [] # also a lookup array, but this is for the dynamic hessian part

    ##########################################################################
    ## For compressing the indices
    ##########################################################################
    self.__compression_kernel: Optional[coordinateCompressionKernel] = None # for compressing the indices for dynamic energies
    self.__compression_kernel_dynamic: Optional[coordinateCompressionKernel] = None

  def __compute_wrt_start_indices(self):
    for item in self.__wrt:
      if item.isDynamic:
        # for wrt let's disallow dynamic attributes
        raise ValueError("hessian.__compute_wrt_start_indices: wrt is a dynamic attributes.")

    gradient_sizes = [item.size * item.correspondance.numInstances for item in self.__wrt]
    diagonal_block_sizes = [item.size * item.size * item.correspondance.numInstances for item in self.__wrt]
    diagonal_block_start = [0]
    gradient_segment_start = [0]
    for i in range(1, len(diagonal_block_start)):
      diagonal_block_start.append(diagonal_block_sizes[i - 1] + diagonal_block_start[i - 1])
      gradient_segment_start.append(gradient_segment_start[i - 1] + gradient_sizes[i - 1])
    diagonal_block_start.append(diagonal_block_start[-1] + diagonal_block_sizes[-1])
    gradient_segment_start.append(gradient_segment_start[-1] + gradient_sizes[-1])
    self.__wrt_start_indices = gradient_segment_start
    self.__diagonal_blocks_start_cpu = diagonal_block_start
    self.__diagonal_blocks_local_sizes = diagonal_block_sizes

  @property
  def wrt_start_indices(self) -> List[int]:
    return self.__wrt_start_indices

  @property
  def wrt(self) -> List[attribute]:
    """Variables this Hessian is defined over."""
    return self.__wrt

  @property
  def dynamic_instances(self) -> bool:
    """Whether this Hessian tracks dynamic instance terms."""
    return self.__dynamic_instances

  def __add__(self, other: hessian):
    # Addition is only supported between hessian objects with identical wrt signatures.
    if not isinstance(other, hessian):
      raise TypeError(f"hessian.__add__: unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'")

    # Check wrt compatibility by hash, because wrt identity and order determine matrix layout.
    if len(self.__wrt) != len(other.__wrt):
      raise ValueError("hessian.__add__: wrt length mismatch.")
    for left, right in zip(self.__wrt, other.__wrt):
      if left.hash != right.hash:
        raise ValueError("hessian.__add__: wrt mismatch (hash check failed).")

    result = hessian(self.__wrt)

    # Merge all cached symbolic/structural containers from this and the other child Hessians.
    result.__local_targets = self.__local_targets + other.local_targets
    result.__hessian_and_gradient_kernels = self.__hessian_and_gradient_kernels + other.__hessian_and_gradient_kernels
    result.__hessian_and_gradient_kernels_dynamic = self.__hessian_and_gradient_kernels_dynamic + other.__hessian_and_gradient_kernels_dynamic
    result.__global_hessians = self.__global_hessians + other.__global_hessians
    result.__global_jacobians = self.__global_jacobians + other.__global_jacobians
    result.__global_inner_hessians = self.__global_inner_hessians + other.__global_inner_hessians
    result.__project_entire_hessian = self.__project_entire_hessian + other.__project_entire_hessian
    result.__projection_methods = self.__projection_methods + other.__projection_methods
    result.__global_hessians_dynamic = self.__global_hessians_dynamic + other.__global_hessians_dynamic
    result.__global_jacobians_dynamic = self.__global_jacobians_dynamic + other.__global_jacobians_dynamic
    result.__global_inner_hessians_dynamic = self.__global_inner_hessians_dynamic + other.__global_inner_hessians_dynamic
    result.__project_entire_hessian_dynamic = self.__project_entire_hessian_dynamic + other.__project_entire_hessian_dynamic
    result.__projection_methods_dynamic = self.__projection_methods_dynamic + other.__projection_methods_dynamic
    result.__indices_kernels = self.__indices_kernels + other.__indices_kernels
    result.__indices_kernels_dynamic = self.__indices_kernels_dynamic + other.__indices_kernels_dynamic
    result.__block_indices_gpu = self.__block_indices_gpu + other.__block_indices_gpu
    result.__block_indices_gpu_dynamic = self.__block_indices_gpu_dynamic + other.__block_indices_gpu_dynamic
    return result

  @property
  def gradient(self) -> gradient:
    return self.__gradient

  @gradient.setter
  def gradient(self, value: gradient) -> None:
    if value.size != self.cols:
      raise ValueError(f"hessian.gradient: gradient size {value.size} does not match hessian size {self.cols}")
    self.__gradient = value

  @property
  def hessian_and_gradient_kernels(self) -> List[hessianAndGradientKernel]:
    """Cached symbolic+numeric kernels for each accumulated Hessian term."""
    return self.__hessian_and_gradient_kernels

  @hessian_and_gradient_kernels.setter
  def hessian_and_gradient_kernels(self, value: List[hessianAndGradientKernel]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.hessian_and_gradient_kernels: value must be a list.")
    if any(not isinstance(item, hessianAndGradientKernel) for item in value):
      raise TypeError("hessian.hessian_and_gradient_kernels: all items must be hessianAndGradientKernel.")
    self.__hessian_and_gradient_kernels = value

  @property
  def hessian_and_gradient_kernels_dynamic(self) -> List[hessianAndGradientKernel]:
    """Cached kernels for the dynamic subset of hessian terms."""
    return self.__hessian_and_gradient_kernels_dynamic

  @hessian_and_gradient_kernels_dynamic.setter
  def hessian_and_gradient_kernels_dynamic(self, value: List[hessianAndGradientKernel]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.hessian_and_gradient_kernels_dynamic: value must be a list.")
    if any(not isinstance(item, hessianAndGradientKernel) for item in value):
      raise TypeError("hessian.hessian_and_gradient_kernels_dynamic: all items must be hessianAndGradientKernel.")
    self.__hessian_and_gradient_kernels_dynamic = value

  @property
  def global_hessians(self) -> List[attribute]:
    """Global merged Hessian symbolic attributes for each energy contribution."""
    return self.__global_hessians

  @global_hessians.setter
  def global_hessians(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_hessians: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_hessians: all items must be yasps.attribute.attribute.")
    self.__global_hessians = value

  @property
  def global_jacobians(self) -> List[attribute]:
    """Global Jacobian symbolic attributes for each tracked energy."""
    return self.__global_jacobians

  @global_jacobians.setter
  def global_jacobians(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobians: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_jacobians: all items must be yasps.attribute.attribute.")
    self.__global_jacobians = value

  @property
  def global_inner_hessians(self) -> List[attribute]:
    """Global inner-Hessian symbolic attributes (projected Hessian mode)."""
    return self.__global_inner_hessians

  @global_inner_hessians.setter
  def global_inner_hessians(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_inner_hessians: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_inner_hessians: all items must be yasps.attribute.attribute.")
    self.__global_inner_hessians = value

  @property
  def project_entire_hessian(self) -> List[bool]:
    """Per-term flag for whether each Hessian term is fully projected."""
    return self.__project_entire_hessian

  @project_entire_hessian.setter
  def project_entire_hessian(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.project_entire_hessian: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.project_entire_hessian: all items must be bool.")
    self.__project_entire_hessian = value

  @property
  def projection_methods(self) -> List[bool]:
    """Per-term projection mode index/method for assembled Hessian blocks."""
    return self.__projection_methods

  @projection_methods.setter
  def projection_methods(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.projection_methods: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.projection_methods: all items must be bool.")
    self.__projection_methods = value

  @property
  def global_hessians_dynamic(self) -> List[attribute]:
    """Global merged Hessian symbolic attributes for dynamic instances."""
    return self.__global_hessians_dynamic

  @global_hessians_dynamic.setter
  def global_hessians_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_hessians_dynamic: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_hessians_dynamic: all items must be yasps.attribute.attribute.")
    self.__global_hessians_dynamic = value

  @property
  def global_jacobians_dynamic(self) -> List[attribute]:
    """Dynamic global Jacobian symbolic attributes."""
    return self.__global_jacobians_dynamic

  @global_jacobians_dynamic.setter
  def global_jacobians_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_jacobians_dynamic: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_jacobians_dynamic: all items must be yasps.attribute.attribute.")
    self.__global_jacobians_dynamic = value

  @property
  def global_inner_hessians_dynamic(self) -> List[attribute]:
    """Dynamic inner-Hessian symbolic attributes."""
    return self.__global_inner_hessians_dynamic

  @global_inner_hessians_dynamic.setter
  def global_inner_hessians_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.global_inner_hessians_dynamic: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError("hessian.global_inner_hessians_dynamic: all items must be yasps.attribute.attribute.")
    self.__global_inner_hessians_dynamic = value

  @property
  def project_entire_hessian_dynamic(self) -> List[bool]:
    """Per-dynamic-term flag for full Hessian projection."""
    return self.__project_entire_hessian_dynamic

  @project_entire_hessian_dynamic.setter
  def project_entire_hessian_dynamic(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.project_entire_hessian_dynamic: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.project_entire_hessian_dynamic: all items must be bool.")
    self.__project_entire_hessian_dynamic = value

  @property
  def projection_methods_dynamic(self) -> List[bool]:
    """Projection mode used for each dynamic term."""
    return self.__projection_methods_dynamic

  @projection_methods_dynamic.setter
  def projection_methods_dynamic(self, value: List[bool]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.projection_methods_dynamic: value must be a list.")
    if any(not isinstance(item, bool) for item in value):
      raise TypeError("hessian.projection_methods_dynamic: all items must be bool.")
    self.__projection_methods_dynamic = value

  @property
  def indices_kernels(self) -> List[gradientIndicesKernel]:
    """Gradient-index kernels for static Hessian contributions."""
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
    """Gradient-index kernels for dynamic Hessian contributions."""
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
    """Block lookup arrays that map local block-local coordinates into global matrix storage."""
    return self.__block_indices_gpu

  @block_indices_gpu.setter
  def block_indices_gpu(self, value: List[gpuarray.GPUArray]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.block_indices_gpu: value must be a list.")
    if any(not isinstance(item, gpuarray.GPUArray) for item in value):
      raise TypeError("hessian.block_indices_gpu: all items must be pycuda.gpuarray.GPUArray.")
    self.__block_indices_gpu = value

  @property
  def block_indices_gpu_dynamic(self) -> List[gpuarray.GPUArray]:
    """Dynamic block lookup arrays for dynamic Hessian term placement."""
    return self.__block_indices_gpu_dynamic

  @block_indices_gpu_dynamic.setter
  def block_indices_gpu_dynamic(self, value: List[gpuarray.GPUArray]) -> None:
    if not isinstance(value, list):
      raise TypeError("hessian.block_indices_gpu_dynamic: value must be a list.")
    if any(not isinstance(item, gpuarray.GPUArray) for item in value):
      raise TypeError("hessian.block_indices_gpu_dynamic: all items must be pycuda.gpuarray.GPUArray.")
    self.__block_indices_gpu_dynamic = value

  @timed("hessian.getSparseIndices")
  def getSparseIndices(self):
    for item in self.__indices_kernels:
      item.computeIndices(self.__wrt_start_indices)

    # after the index are computed, we can start compressing them
    self.__compression_kernel = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels],
      [x.outputBlockDimensions for x in self.__indices_kernels],
      [x.numTotalCoordinates for x in self.__indices_kernels],
      self.__wrt
    )
    # perform the compression
    assert self.__compression_kernel is not None
    self.__compression_kernel.compressCoordinatesAndDimensions()
    lookup_arrays = self.__compression_kernel.lookupArrays
    self.__block_indices_gpu = lookup_arrays

    total_block_size = self.__compression_kernel.totalBlockSize
    self.__blocks_flattened = gpuarray.empty(total_block_size, dtype = np.float64)

    num_unique_dimensions = self.__compressionKernel.numUniqueDimensions # get how many unique block dimensions there are

    self.__blocks_start_indices = self.__compressionKernel.uniqueDimensionsOuterIndices.get().tolist()[: self.__compressionKernel.numUniqueDimensions + 1]

    self.__block_positions = self.__compressionKernel.uniqueCoordinates

    self.__block_counts = self.__compressionKernel.uniqueDimensionsBlockCounts.get().tolist()

    # here we set the unique dimensions to generate the code
    self.__block_dimensions = self.__compressionKernel.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]



  @timed("hessian.getSparseIndicesDynamic")
  def getSparseIndicesDynamic(self):
    if len(self.__indices_kernels_dynamic) == 0:
      return

    # Each dynamic index kernel computes the local block coordinates for the current dynamic instances.
    for item in self.__indices_kernels_dynamic:
      item.computeIndices(self.__wrt_start_indices)

    # Build and run a compression kernel that deduplicates dynamic coordinates across all terms.
    self.__compression_kernel_dynamic = coordinateCompressionKernel(
      [x.outputCoordinates for x in self.__indices_kernels_dynamic],
      [x.outputBlockDimensions for x in self.__indices_kernels_dynamic],
      [x.numTotalCoordinates for x in self.__indices_kernels_dynamic],
      self.__wrt
    )
    assert self.__compression_kernel_dynamic is not None
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()

    # Map each dynamic term's local blocks back to the global packed dynamic storage.
    lookup_arrays = self.__compression_kernel_dynamic.lookupArrays
    self.__block_indices_gpu_dynamic = []
    tmp_count = 0
    for i in range(len(self.__indices_kernels_dynamic)):
      if self.__indices_kernels_dynamic[i].numTotalCoordinates > 0:
        self.__block_indices_gpu_dynamic.append(lookup_arrays[tmp_count])
        tmp_count += 1

    # Allocate compressed dynamic storage and cache compressed metadata.
    total_block_size = self.__compression_kernel_dynamic.totalBlockSize
    self.__blocks_flattened_dynamic = gpuarray.empty(total_block_size, dtype=np.float64)
    num_unique_dimensions = self.__compression_kernel_dynamic.numUniqueDimensions
    self.__blocks_start_indices_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsOuterIndices.get().tolist()[: self.__compression_kernel_dynamic.numUniqueDimensions + 1]
    self.__block_positions_dynamic = self.__compression_kernel_dynamic.uniqueCoordinates
    self.__block_counts_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsBlockCounts.get().tolist()
    self.__block_dimensions_dynamic = self.__compression_kernel_dynamic.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]

  @timed("hessian.getSparseIndicesDynamicAgain")
  def getSparseIndicesDynamicAgain(self):
    # Recompute dynamic coordinates and reuse the existing compression kernel object.
    if len(self.__indices_kernels_dynamic) == 0:
      return
    assert self.__compression_kernel_dynamic is not None, "hessian.getSparseIndicesDynamicAgain: compression kernel for dynamic part is not initialized."

    for item in self.__indices_kernels_dynamic:
      item.computeIndices(self.__wrt_start_indices)

    self.__compression_kernel_dynamic.updateCoordinates(
      [x.outputCoordinates for x in self.__indices_kernels_dynamic],
      [x.outputBlockDimensions for x in self.__indices_kernels_dynamic],
      [x.numTotalCoordinates for x in self.__indices_kernels_dynamic]
    )
    self.__compression_kernel_dynamic.compressCoordinatesAndDimensions()

    # Refresh per-term lookup arrays for the updated compressed layout.
    lookup_arrays = self.__compression_kernel_dynamic.lookupArrays
    self.__block_indices_gpu_dynamic = []
    tmp_count = 0
    for i in range(len(self.__indices_kernels_dynamic)):
      if self.__indices_kernels_dynamic[i].numTotalCoordinates > 0:
        self.__block_indices_gpu_dynamic.append(lookup_arrays[tmp_count])
        tmp_count += 1

    # Ensure dynamic block storage is large enough for the new compressed structure.
    total_block_size = self.__compression_kernel_dynamic.totalBlockSize
    if self.__blocks_flattened_dynamic.size < total_block_size:
      self.__blocks_flattened_dynamic = gpuarray.zeros(total_block_size, dtype=np.float64)

    num_unique_dimensions = self.__compression_kernel_dynamic.numUniqueDimensions
    self.__blocks_start_indices_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsOuterIndices.get().tolist()[: self.__compression_kernel_dynamic.numUniqueDimensions + 1]
    self.__block_positions_dynamic = self.__compression_kernel_dynamic.uniqueCoordinates
    self.__block_counts_dynamic = self.__compression_kernel_dynamic.uniqueDimensionsBlockCounts.get().tolist()
    self.__block_dimensions_dynamic = self.__compression_kernel_dynamic.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]

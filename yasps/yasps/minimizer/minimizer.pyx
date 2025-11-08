# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import List, Tuple, Set, Optional
from yasps.energy import energy
from yasps.attribute import attribute
from yasps.solverKernel import solverKernel
from yasps.coordinateCompressionKernel import coordinateCompressionKernel
from yasps.diagonalBlockInverseKernel import diagonalBlockInverseKernel
import time
import ctypes
from yasps.helper import timed
# def unique_row_view(data):
#   b = np.ascontiguousarray(data).view(
#     np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
#   )
#   u = np.unique(b).view(data.dtype).reshape(-1, data.shape[1])
#   return u

class minimizer:
  def __init__(self):
    self.__energies: List[energy] = []
    self.__wrt: List[attribute] = []
    self.__gradient: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__diagonal: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # only store the diagonal elements
    self.__diagonal_blocks: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store the block diagonal
    self.__diagonal_blocks_inverse: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store the inverse of the diagonal blocks
    self.__diagonal_blocks_start: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store for each attribute, where does the accumulated diagonal block start
    self.__diagonal_blocks_start_cpu: List[int] = []
    self.__gradient_segments_start: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # store for each attribute, where does the gradient segment start
    self.__gradient_segments_start_cpu: List[int] = []

    self.__blockDimensions: List[int] = [] # record the dimension of blocks
    # self.__blocks: List[gpuarray.GPUArray] = [] # for each different block dimensions, the datas
    self.__blocksFlattened: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # the flattened blocks
    self.__blocksStartIndices: List[int] = [] # for each different block dimensions, where do they start, this is to navigate through the flattened blocks
    # self.__blocksStartIndicesGPU: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each block, where do they start, this is to navigate through the flattened blocks
    self.__blockPositions: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each block, what's its coordinate, we will use it for spmv
    # self.__blockPositionsList: List[gpuarray.GPUArray] = [] # for each different block sizes, for each block, what's its coordinate, we will use it for spmv, this is just segmented from blockPositions
    self.__blockCounts: List[int] = [] # record for each size of block, the number of blocks

    ## here we have all the same items, but for dynamic energy
    self.__energiesDynamic: List[energy] = [] # for energies with dynamic instances
    self.__blockDimensionsDynamic: List[int] = [] # record the dimension of blocks for dynamic energies
    self.__blocksFlattenedDynamic: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # the flattened blocks for dynamic energies
    self.__blocksStartIndicesDynamic: List[int] = [] # for each different block dimensions, where do they start, this is to navigate through the flattened blocks for dynamic energies
    self.__blockPositionsDynamic: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each block, what's its coordinate, we will use it for spmv for dynamic energies
    self.__blockCountsDynamic: List[int] = [] # record for each size of block, the number of blocks for dynamic energies
    self.__compressionKernelDynamic = None # for compressing the indices for dynamic energies

    # those are the attributes associated with the gradient
    # they are fixed
    self.__gradientSizes: List[int] = []
    self.__gradientSegments: List[gpuarray.GPUArray] = []
    self.__wrtStartIndices: List[int] = []
    self.__compressionKernel = None # for compressing the indices

    self.__seen_pre_targets_full_names: Set[str] = set() # for recording partial tagets for any energy added, because maybe for some energy it doesnt want to optimize wrt all the targets supported in the end



    self.__solver: Optional[solverKernel] = None
    ## auxilary variables for solver
    self.__d_p1_b: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_r: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_c: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_q: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_s: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__solution: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__solutionSegments: List[gpuarray.GPUArray] = []

    self.__diagonalBlockInverseKernel: Optional[diagonalBlockInverseKernel] = None

  @property
  def solutionSegments(self) -> List[gpuarray.GPUArray]:
    return self.__solutionSegments

  @property
  def gradient(self) -> gpuarray.GPUArray:
    return self.__gradient

  @property
  def gradientSegments(self) -> List[gpuarray.GPUArray]:
    return self.__gradientSegments

  @property
  def energies(self) -> List[energy]:
    return self.__energies

  @property
  def energiesDynamic(self) -> List[energy]:
    return self.__energiesDynamic

  @property
  def wrt(self) -> List[attribute]:
    return self.__wrt

  @property
  def diagonal(self) -> gpuarray.GPUArray:
    return self.__diagonal

  def addEnergies(self, energies: List[energy]) -> None:
    for item in energies:
      if item.hash in [energy.hash for energy in self.__energies]:
        raise ValueError("minimizer.addEnergies: energies has duplicate energies.")
    self.__energies.extend(energies)

  def addEnergy(self, e: attribute, targets: List[attribute] = [], projection_method = 1, save_intermediate = False, gradient_only = False, dynamic_instances = False) -> None:
    if e.name == "":
      raise ValueError("scene.addEnergy: energy attribute must have a name.")
    from yasps.energy import energy
    for t in targets:
      self.__seen_pre_targets_full_names.add(t.fullName)
    newEnergy = energy(e, targets, projection_method, save_intermediate, gradient_only)
    if newEnergy.hash in [energy.hash for energy in self.__energies]:
      raise ValueError("minimizer.addEnergy: energy already exists.")
    if not dynamic_instances:
      self.__energies.append(newEnergy)
    else:
      self.__energiesDynamic.append(newEnergy)


  def addWrt(self, wrt: List[attribute]) -> None:
    start = time.time()
    seenAttributeHashes: Set[int] = set()
    from yasps.attribute import DATA
    for att in wrt:
      if att.hash in seenAttributeHashes:
        raise ValueError(f"minimizer.addWrt: wrt {att.fullName} is duplicate attribute.")
      if att.operator is not DATA:
        raise ValueError(f"minimizer.addWrt: wrt {att.fullName} is non-data attribute.")
      if att.isDynamic:
        raise ValueError(f"minimizer.addWrt: wrt {att.fullName} is dynamic attribute.")
      seenAttributeHashes.add(attribute.hash)

    # we check if the target matches the pre_targets_full_names set
    target_full_name_set = set([t.fullName for t in wrt])
    # check if the target full name set contains all the pre_targets_full_names
    if not self.__seen_pre_targets_full_names.issubset(target_full_name_set):
      missing = self.__seen_pre_targets_full_names - target_full_name_set
      raise ValueError(f"minimizer.addWrt: target is missing attributes {missing} that are required by the energies added.")

    self.__wrt.extend(wrt)
    self.__getGradientSize() # get the size of the gradient
    start = time.time()
    self.__getSparseIndices() # get the sparse indices
    self.__getSparseIndicesDynamic()
    end = time.time()
    print(f"Sparse indices generation: {1000.0 * (end - start)} ms")
    # exit()


  def __getGradientSize(self) -> None:
    for item in self.wrt:
      if item.isDynamic:
        # for wrt let's disallow dynamic attributes
        raise ValueError("minimizer.__getGradientSize: wrt is a dynamic attributes.")
      self.__gradientSizes.append(item.size * item.correspondance.numInstances)
    # allocate the array
    self.__gradient = gpuarray.zeros(sum(self.__gradientSizes), dtype = np.float64)
    self.__diagonal = gpuarray.zeros(sum(self.__gradientSizes), dtype = np.float64)
    diagonal_block_sizes = [item.size * item.size * item.correspondance.numInstances for item in self.wrt]
    # do a prefix sum on the diagonal block sizes
    diagonal_block_start = [0 for _ in diagonal_block_sizes]
    gradient_segment_start = [0 for _ in diagonal_block_sizes]
    for i in range(1, len(diagonal_block_start)):
      diagonal_block_start[i] = diagonal_block_sizes[i - 1] + diagonal_block_start[i - 1]
      gradient_segment_start[i] = gradient_segment_start[i - 1] + self.__gradientSizes[i - 1]
    diagonal_block_start.append(diagonal_block_start[-1] + diagonal_block_sizes[-1])
    gradient_segment_start.append(gradient_segment_start[-1] + self.__gradientSizes[-1])
    self.__diagonal_blocks = gpuarray.zeros(sum(diagonal_block_sizes), dtype = np.float64)
    self.__diagonal_blocks_inverse = gpuarray.zeros(sum(diagonal_block_sizes), dtype = np.float64)
    self.__diagonal_blocks_start = gpuarray.to_gpu(np.array(diagonal_block_start, dtype = np.uint32))
    self.__diagonal_blocks_start_cpu = diagonal_block_start
    self.__gradient_segments_start = gpuarray.to_gpu(np.array(gradient_segment_start, dtype = np.uint32))
    self.__gradient_segments_start_cpu = gradient_segment_start
    # assign the gradient segments by reference
    start = 0
    self.__wrtStartIndices.append(start) # get where each data element starts
    # here we compute for each data, where does it reside in the fianl solution arrays
    for size in self.__gradientSizes:
      self.__gradientSegments.append(self.__gradient[start:start + size])
      start += size
      self.__wrtStartIndices.append(start)
    print(f"The size of the gradient is: {sum(self.__gradientSizes)}")
    # print(f"The gradient segments sizes are: {self.__gradientSizes}")

  def __getblockSizes(self):
    pass

  def __getSparseIndicesGPU(self):
    pass

  def __getSparseIndices(self):
    if len(self.__energies) == 0:
      return
    for local_energy in self.__energies:
      local_energy.getSparseIndices(self.wrt, self.__wrtStartIndices)
    self.__compressionKernel = coordinateCompressionKernel([x.outputCoordinates for x in self.energies], [x.outputBlockDimensions for x in self.energies], [x.numTotalCoordinates for x in self.energies], self.wrt)
    self.__compressionKernel.compressCoordinatesAndDimensions()
    # set for each energy, for where does the block reside for each coordinate
    lookupArrays = self.__compressionKernel.lookupArrays
    for i in range(len(self.energies)):
      self.__energies[i].block_indices_gpu = lookupArrays[i]
    # we also initialize the space for blocks flattened
    totalBlockSize = self.__compressionKernel.totalBlockSize
    self.__blocksFlattened = gpuarray.empty(totalBlockSize, dtype=np.float64)
    num_unique_dimensions = self.__compressionKernel.numUniqueDimensions # get how many unique block dimensions there are
    self.__blocksStartIndices = self.__compressionKernel.uniqueDimensionsOuterIndices.get().tolist()[: self.__compressionKernel.numUniqueDimensions + 1]
    self.__blockPositions = self.__compressionKernel.uniqueCoordinates
    self.__blockCounts = self.__compressionKernel.uniqueDimensionsBlockCounts.get().tolist()

    # here we set the unique dimensions to generate the code
    self.__blockDimensions = self.__compressionKernel.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]

  @timed("minimizer.__getSparseIndicesDynamic")
  def __getSparseIndicesDynamic(self):
    if len(self.__energiesDynamic) == 0:
      return
    for local_energy in self.__energiesDynamic:
      local_energy.getSparseIndices(self.wrt, self.__wrtStartIndices)
    self.__compressionKernelDynamic = coordinateCompressionKernel([x.outputCoordinates for x in self.__energiesDynamic], [x.outputBlockDimensions for x in self.__energiesDynamic], [x.numTotalCoordinates for x in self.__energiesDynamic], self.wrt)
    self.__compressionKernelDynamic.compressCoordinatesAndDimensions()
    # set for each energy, for where does the block reside for each coordinate
    lookupArrays = self.__compressionKernelDynamic.lookupArrays
    tmp_count = 0
    for i in range(len(self.__energiesDynamic)):
      if self.__energiesDynamic[i].numTotalCoordinates > 0:
        self.__energiesDynamic[i].block_indices_gpu = lookupArrays[tmp_count]
        tmp_count += 1
    # we also initialize the space for blocks flattened
    totalBlockSize = self.__compressionKernelDynamic.totalBlockSize
    self.__blocksFlattenedDynamic = gpuarray.empty(totalBlockSize, dtype=np.float64)
    num_unique_dimensions = self.__compressionKernelDynamic.numUniqueDimensions # get how many unique block dimensions there are
    self.__blocksStartIndicesDynamic = self.__compressionKernelDynamic.uniqueDimensionsOuterIndices.get().tolist()[: self.__compressionKernelDynamic.numUniqueDimensions + 1]
    self.__blockPositionsDynamic = self.__compressionKernelDynamic.uniqueCoordinates
    self.__blockCountsDynamic = self.__compressionKernelDynamic.uniqueDimensionsBlockCounts.get().tolist()
    # here we set the unique dimensions to generate the code
    self.__blockDimensionsDynamic = self.__compressionKernelDynamic.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]

  @timed("minimizer.__getSparseIndicesDynamicAgain")
  def __getSparseIndicesDynamicAgain(self):
    if len(self.__energiesDynamic) == 0:
      return
    for local_energy in self.__energiesDynamic:
      local_energy.getSparseIndicesAgain()
    assert self.__compressionKernelDynamic is not None, "minimizer.__getSparseIndicesDynamicAgain: compression kernel for dynamic energies is not initialized."
    self.__compressionKernelDynamic.updateCoordinates([x.outputCoordinates for x in self.__energiesDynamic], [x.outputBlockDimensions for x in self.__energiesDynamic], [x.numTotalCoordinates for x in self.__energiesDynamic])
    print("Updated num total coordinates are: ", [x.numTotalCoordinates for x in self.__energiesDynamic])
    self.__compressionKernelDynamic.compressCoordinatesAndDimensions()
    # set for each energy, for where does the block reside for each coordinate
    lookupArrays = self.__compressionKernelDynamic.lookupArrays
    tmp_count = 0
    for i in range(len(self.__energiesDynamic)):
      if self.__energiesDynamic[i].numTotalCoordinates > 0:
        self.__energiesDynamic[i].block_indices_gpu = lookupArrays[tmp_count]
        tmp_count += 1
    # we also initialize the space for blocks flattened
    totalBlockSize = self.__compressionKernelDynamic.totalBlockSize
    if self.__blocksFlattenedDynamic.size < totalBlockSize:
      # if the size is not enough, we need to reallocate
      self.__blocksFlattenedDynamic = gpuarray.zeros(totalBlockSize, dtype=np.float64)
    num_unique_dimensions = self.__compressionKernelDynamic.numUniqueDimensions # get how many unique block dimensions there are
    self.__blocksStartIndicesDynamic = self.__compressionKernelDynamic.uniqueDimensionsOuterIndices.get().tolist()[: self.__compressionKernelDynamic.numUniqueDimensions + 1]
    self.__blockPositionsDynamic = self.__compressionKernelDynamic.uniqueCoordinates
    self.__blockCountsDynamic = self.__compressionKernelDynamic.uniqueDimensionsBlockCounts.get().tolist()
    # here we set the unique dimensions to generate the code
    self.__blockDimensionsDynamic = self.__compressionKernelDynamic.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]

  def generateHessianAndGradient(self):
    start = time.time()
    for e in (self.__energies + self.__energiesDynamic):
      e.generateHessianAndGradient()
    end = time.time()
    print(f"Autodiff computation: {1000.0 * (end - start)} ms")

  @timed("minimizer.computeSolution")
  def computeSolution(self, tolerance = 1e-3) -> List[gpuarray.GPUArray]:
    error_code = self.computeHessianAndGradient(tolerance = tolerance)
    if error_code < 0:
      return []
    return self.solutionSegments

  def computeHessianAndGradient(self, tolerance = 1e-3, maxIterations = 20000):
    # set gradient and hessian to 0
    self.__gradient.fill(0)
    if self.__blocksFlattened.shape[0] > 0:
      self.__blocksFlattened.fill(0)
    if self.__blocksFlattenedDynamic.shape[0] > 0:
      self.__blocksFlattenedDynamic.fill(0)
    self.__diagonal.fill(0)
    self.__diagonal_blocks.fill(0)
    for e in self.energies:
      e.computeHessianAndGradient(
        self.__gradient,
        self.__blocksFlattened,
        self.__diagonal,
        self.__diagonal_blocks,
        self.__diagonal_blocks_start,
        self.__gradient_segments_start
      )

    # for dynamic energies we need to get the sparse indices again
    self.__getSparseIndicesDynamicAgain()
    for e in self.energiesDynamic:
      if e.numTotalCoordinates > 0:
        e.computeHessianAndGradient(
          self.__gradient,
          self.__blocksFlattenedDynamic,
          self.__diagonal,
          self.__diagonal_blocks,
          self.__diagonal_blocks_start,
          self.__gradient_segments_start
        )
    # print("--------------------------------------------------------")
    # print("Block counts dynamic is: ", self.__blockCountsDynamic)
    # print("--------------------------------------------------------")


    if self.__diagonalBlockInverseKernel is None:
      # initialize the diagonal block inverse kernel
      self.__diagonal_blocks_start_cpu = self.__diagonal_blocks_start.get().tolist()
      self.__diagonalBlockInverseKernel = diagonalBlockInverseKernel(
        set([item.size for item in self.wrt]),
        self.__diagonal_blocks_start_cpu,
        [item.correspondance.numInstances for item in self.__wrt],
        [item.size for item in self.__wrt],
        len(self.__wrt)
      )
    assert self.__diagonalBlockInverseKernel is not None
    self.__diagonalBlockInverseKernel.computeDiagonalBlockInverse(self.__diagonal_blocks, self.__diagonal_blocks_inverse)
    # # let's check the result
    # diagonal_block_cpu = self.__diagonal_blocks.get()
    # diagonal_block_inverse_cpu = self.__diagonal_blocks_inverse.get()
    # start = 0
    # for i in range(len(self.__wrt)):
    #   # first get the size
    #   block_size = self.__wrt[i].size
    #   # get the number of instances
    #   num_instances = self.__wrt[i].correspondance.numInstances
    #   for j in range(num_instances):
    #     local_block = diagonal_block_cpu[start: start + block_size * block_size].reshape((block_size, block_size))
    #     local_block_inverse = diagonal_block_inverse_cpu[start: start + block_size * block_size].reshape((block_size, block_size))
    #     mul_result = local_block @ local_block_inverse
    #     identity = np.eye(block_size)
    #     if not np.allclose(mul_result, identity, atol=1e-6):
    #       print(f"Warning: Diagonal block inverse verification failed for attribute {self.__wrt[i].fullName}, instance {j}.")
    #       print("Original block:")
    #       print(local_block)
    #       print("Inverse block:")
    #       print(local_block_inverse)
    #       print("Product:")
    #       print(mul_result)
    #       exit()
    #     start += block_size * block_size


    # now we have the hessian and gradient
    # we need to solve the system
    if self.__solver is None:
      self.__solver = solverKernel(self.__blockDimensions + self.__blockDimensionsDynamic)
      self.__d_p1_b = gpuarray.empty(self.__gradient.shape, dtype = np.float64)
      self.__d_r = gpuarray.empty(self.__gradient.shape, dtype = np.float64)
      self.__d_c = gpuarray.empty(self.__gradient.shape, dtype = np.float64)
      self.__d_q = gpuarray.empty(self.__gradient.shape, dtype = np.float64)
      self.__d_s = gpuarray.empty(self.__gradient.shape, dtype = np.float64)
      self.__solution = gpuarray.empty(self.__gradient.shape, dtype = np.float64)
      # print("Gradient shape is: ")
      # print(self.__gradient.shape)
      count = 0
      for i in range(len(self.__gradientSegments)):
        self.__solutionSegments.append(self.__solution[count: count + self.__gradientSegments[i].shape[0]])
        count += self.__gradientSegments[i].shape[0]

    # if needed, we will also need to update the kernel with the latest blcok dimensions
    assert self.__solver is not None
    self.__solver.updateBlockDimensions(self.__blockDimensions + self.__blockDimensionsDynamic)

    # setting zeros
    self.__d_p1_b.fill(0)
    self.__d_r.fill(0)
    self.__d_c.fill(0)
    self.__d_q.fill(0)
    self.__d_s.fill(0)
    self.__solution.fill(0)


    error_code = self.__solver.computeSolution(
      maxIterations,
      tolerance,
      self.__blocksFlattened,
      self.__blockPositions,
      self.__blocksStartIndices,
      self.__blockCounts,
      self.__blockDimensions,
      self.__blocksFlattenedDynamic,
      self.__blockPositionsDynamic,
      self.__blocksStartIndicesDynamic,
      self.__blockCountsDynamic,
      self.__blockDimensionsDynamic,
      self.__diagonal,
      self.__diagonal_blocks_inverse,
      self.__diagonal_blocks_start_cpu,
      [item.correspondance.numInstances for item in self.__wrt],
      [item.size for item in self.__wrt],
      self.__gradient_segments_start_cpu,
      len(self.__wrt),
      self.__gradient,
      self.__d_p1_b,
      self.__d_r,
      self.__d_c,
      self.__d_q,
      self.__d_s,
      self.__solution
    )
    return error_code



    # print("gradient")
    # print(self.__gradient.get())
    # exit(0)


# #######################################################################################
# ## for checking the hessian and diagonals
# #######################################################################################
#     full_mat = np.zeros((self.__gradient.shape[0], self.__gradient.shape[0]))
#     for i in range(len(self.__blockDimensions)):
#       # print(f"Dimension: {self.__blockDimensions[i][0]}, {self.__blockDimensions[i][0]}")

#       block_rows = self.__blockDimensions[i][0]
#       block_cols = self.__blockDimensions[i][1]
#       block_size = block_rows * block_cols
#       for j in range(self.__blockPositionsList[i].get().shape[0] // 2):
#         # print(f"Pos {self.__blockPositionsList[i].get()[j]}")
#         # pos = self.__blockPositionsList[i].get()[j]
#         x_pos = self.__blockPositionsList[i].get()[j * 2]
#         y_pos = self.__blockPositionsList[i].get()[j * 2 + 1]
#         if x_pos == y_pos:
#           full_mat[x_pos: x_pos + block_rows, y_pos: y_pos + block_cols] += self.__blocks[i].get()[j * block_size: (j + 1) * block_size].reshape(block_rows, block_cols)
#         else:
#           full_mat[x_pos: x_pos + block_rows, y_pos: y_pos + block_cols] += self.__blocks[i].get()[j * block_size: (j + 1) * block_size].reshape(block_rows, block_cols)
#           full_mat[y_pos: y_pos + block_cols, x_pos: x_pos + block_rows] += self.__blocks[i].get()[j * block_size: (j + 1) * block_size].reshape(block_rows, block_cols).T
#         # block = self.__blocks[i].get()[j * block_size: (j + 1) * block_size]
#         # print(block.reshape(self.__blockDimensions[i][0], self.__blockDimensions[i][1]))
#     print("Assembled full mat: ")
#     print(full_mat)
#     # ## get the eigen value and eigen vector of the mat
#     # ev, evec = np.linalg.eig(full_mat)
#     # for value in ev:
#     #   if value <= 0:
#     #     print("Negative eigen value")
#     #     print(ev)
#     #     exit(1)


#     gradient = self.__gradient.get()
#     cpu_solution = np.linalg.solve(full_mat, gradient)
#     print("CPU Solution")
#     print(cpu_solution)

#     ## update the gpu solution to the cpu solution
#     self.__solution.set(cpu_solution)

#     # print("GPU Solution")
#     # print(self.__solution.get())
#     # exit(0)

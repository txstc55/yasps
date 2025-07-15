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





    self.__solver: Optional[solverKernel] = None
    ## auxilary variables for solver
    self.__d_p1_b: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_r: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_c: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_q: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__d_s: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__solution: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__solutionSegments: List[gpuarray.GPUArray] = []

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

  def addEnergy(self, e: attribute, projection_method = 1, save_intermediate = False, gradient_only = False, dynamic_instances = False) -> None:
    if e.name == "":
      raise ValueError("scene.addEnergy: energy attribute must have a name.")
    from yasps.energy import energy
    newEnergy = energy(e, projection_method, save_intermediate, gradient_only)
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
        raise ValueError(f"minimizer.addWrt: wrt {att} is duplicate attribute.")
      if att.operator is not DATA:
        raise ValueError(f"minimizer.addWrt: wrt {att} is non-data attribute.")
      if att.isDynamic:
        raise ValueError(f"minimizer.addWrt: wrt {att} is dynamic attribute.")
      seenAttributeHashes.add(attribute.hash)
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
    self.__gradient = gpuarray.empty(sum(self.__gradientSizes), dtype = np.float64)
    self.__diagonal = gpuarray.empty(sum(self.__gradientSizes), dtype = np.float64)
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
    for i in range(len(self.__energiesDynamic)):
      self.__energiesDynamic[i].block_indices_gpu = lookupArrays[i]
    # we also initialize the space for blocks flattened
    totalBlockSize = self.__compressionKernelDynamic.totalBlockSize
    self.__blocksFlattenedDynamic = gpuarray.empty(totalBlockSize, dtype=np.float64)
    num_unique_dimensions = self.__compressionKernelDynamic.numUniqueDimensions # get how many unique block dimensions there are
    self.__blocksStartIndicesDynamic = self.__compressionKernelDynamic.uniqueDimensionsOuterIndices.get().tolist()[: self.__compressionKernelDynamic.numUniqueDimensions + 1]
    self.__blockPositionsDynamic = self.__compressionKernelDynamic.uniqueCoordinates
    self.__blockCountsDynamic = self.__compressionKernelDynamic.uniqueDimensionsBlockCounts.get().tolist()
    # here we set the unique dimensions to generate the code
    self.__blockDimensionsDynamic = self.__compressionKernelDynamic.uniqueDimensions.get().tolist()[: num_unique_dimensions * 2]


  def generateHessianAndGradient(self):
    start = time.time()
    for e in (self.__energies + self.__energiesDynamic):
      e.generateHessianAndGradient(self.wrt)
    end = time.time()
    print(f"Autodiff computation: {1000.0 * (end - start)} ms")

  @timed("minimizer.computeSolution")
  def computeSolution(self, tolerance = 1e-3) -> List[gpuarray.GPUArray]:
    self.computeHessianAndGradient(tolerance = tolerance)
    return self.solutionSegments

  def computeHessianAndGradient(self, tolerance = 1e-3):
    # set gradient and hessian to 0
    self.__gradient.fill(0)
    if self.__blocksFlattened.shape[0] > 0:
      self.__blocksFlattened.fill(0)
    if self.__blocksFlattenedDynamic.shape[0] > 0:
      self.__blocksFlattenedDynamic.fill(0)
    self.__diagonal.fill(0)

    # for dynamic energies we need to get the sparse indices again
    self.__getSparseIndicesDynamic()

    for e in self.energies:
      e.computeHessianAndGradient(self.__gradient, self.__blocksFlattened, self.__diagonal)
    for e in self.energiesDynamic:
      e.computeHessianAndGradient(self.__gradient, self.__blocksFlattenedDynamic, self.__diagonal)

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

    cuda_context = pycuda.autoinit.context
    context_ptr = int(cuda_context.handle)
    context_ptr_c = ctypes.c_void_p(context_ptr)
    # call the kernel
    self.__solver.computeSolution(
      context_ptr_c,
      20000,
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
      self.__gradient,
      self.__d_p1_b,
      self.__d_r,
      self.__d_c,
      self.__d_q,
      self.__d_s,
      self.__solution
    )



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

# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import List, Tuple, Set, Optional, Dict
from yasps.energy import energy
from yasps.attribute import attribute
from yasps.solverKernel import solverKernel
import time
import ctypes

def unique_row_view(data):
  b = np.ascontiguousarray(data).view(
    np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
  )
  u = np.unique(b).view(data.dtype).reshape(-1, data.shape[1])
  return u

class minimizer:
  def __init__(self):
    self.__energies: List[energy] = []
    self.__wrt: List[attribute] = []
    self.__gradient: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)
    self.__diagonal: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # only store the diagonal elements

    self.__blockDimensions: List[Tuple[int, int]] = [] # record the dimension of blocks
    self.__blocks: List[gpuarray.GPUArray] = [] # for each different block dimensions, the datas
    self.__blocksFlattened: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # the flattened blocks
    self.__blocksStartIndices: List[int] = [] # for each different block dimensions, where do they start, this is to navigate through the flattened blocks
    self.__blocksStartIndicesGPU: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each block, where do they start, this is to navigate through the flattened blocks
    self.__blockPositions: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each block, what's its coordinate, we will use it for spmv
    self.__blockPositionsList: List[gpuarray.GPUArray] = [] # for each different block sizes, for each block, what's its coordinate, we will use it for spmv, this is just segmented from blockPositions
    self.__blockCounts: List[int] = [] # record for each size of block, the number of blocks
    self.__gradientSizes: List[int] = []
    self.__gradientSegments: List[gpuarray.GPUArray] = []
    self.__wrtStartIndices: List[int] = []
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
  def wrt(self) -> List[attribute]:
    return self.__wrt

  def addEnergies(self, energies: List[energy]) -> None:
    for item in energies:
      if item.hash in [energy.hash for energy in self.__energies]:
        raise ValueError("minimizer.addEnergies: energies has duplicate energies.")
    self.__energies.extend(energies)

  def addEnergy(self, e: attribute, projection_method = 1, save_intermediate = False, gradient_only = False) -> None:
    if e.name == "":
      raise ValueError("scene.addEnergy: energy attribute must have a name.")
    from yasps.energy import energy
    newEnergy = energy(e, projection_method, save_intermediate, gradient_only)
    if newEnergy.hash in [energy.hash for energy in self.__energies]:
      raise ValueError("minimizer.addEnergy: energy already exists.")
    self.__energies.append(newEnergy)


  def addWrt(self, wrt: List[attribute]) -> None:
    start = time.time()
    seenAttributeHashes: Set[int] = set()
    from yasps.attribute import DATA
    for att in wrt:
      if att.hash in seenAttributeHashes:
        raise ValueError("minimizer.__init__: wrt has duplicate attributes.")
      if att.operator is not DATA:
        raise ValueError("minimizer.__init__: wrt has non-data attributes.")
      seenAttributeHashes.add(attribute.hash)
    self.__wrt.extend(wrt)
    self.__getGradientSize() # get the size of the gradient
    self.__getSparseIndices() # get the sparse indices
    end = time.time()
    print(f"Sparse indices generation: {1000.0 * (end - start)} ms")


  def __getGradientSize(self) -> None:
    for item in self.wrt:
      self.__gradientSizes.append(item.size * item.correspondance.numInstances)
    # allocate the array
    self.__gradient = gpuarray.empty(sum(self.__gradientSizes), dtype = np.float64)
    self.__diagonal = gpuarray.empty(sum(self.__gradientSizes), dtype = np.float64)
    # assign the gradient segments by reference
    start = 0
    self.__wrtStartIndices.append(start) # get where each data element starts
    for size in self.__gradientSizes:
      self.__gradientSegments.append(self.__gradient[start:start + size])
      start += size
      self.__wrtStartIndices.append(start)
    print(f"The size of the gradient is: {sum(self.__gradientSizes)}")
    # print(f"The gradient segments sizes are: {self.__gradientSizes}")

  def __getblockSizes(self):
    pass

  def __getSparseIndices(self):
    for local_energy in self.energies:
      local_energy.getSparseIndices(self.wrt, self.__wrtStartIndices)
    # ok now we have the indices for blocks in their uncompressed order
    # what we need to do is to find the compressed sparse indices
    uncompressedIndicesNumpy: np.ndarray = np.empty((0, 2), dtype = np.uint32)
    index_start: List[int] = [0] # for each energy, where does the index start
    energy_gradient_sizes: List[List[int]] = []
    for local_energy in self.energies:
      gradient_sizes_local = local_energy.gradient_sizes_cpu # we need to know how many blocks there are, and what are the block sizes
      energy_gradient_sizes.append(gradient_sizes_local) # know how large each block is
      if not local_energy.gradient_only: # only do the ones that are not gradient only
        print("energy not gradient only")
        # for now let's say i don't really care if we only store the upper triangular parts
        # here we expand the indices to store the location of upper triangular blocks of the local hessian
        # but the stored global matrix doesn't have to be upper triangular
        indices = local_energy.indices # get the gradient indices, by iterating over them we get the position for each small block in the hessian
        local_energy.clearIndices() # we don't need the indices anymore
        # here we get the coordinate for each block in the hessian
        arr = indices.reshape(-1, len(gradient_sizes_local))
        N, L = arr.shape
        row_indices, col_indices = np.triu_indices(L, k=0)
        first_values = arr[:, row_indices].ravel()
        second_values = arr[:, col_indices].ravel()
        uncompressedIndicesNumpy = np.vstack((uncompressedIndicesNumpy, np.vstack((first_values, second_values)).T))
      # know for each local_energy, where does the first coordinate starts
      index_start.append(uncompressedIndicesNumpy.shape[0])
    print("Uncompressed indices set")
    # we now need to know for each energy, what are the dimension of the blocks
    energy_hessian_block_dimensions: List[List[Tuple[int, int]]] = []
    for sizes in energy_gradient_sizes:
      energy_hessian_block_dimensions.append([])
      for i in range(len(sizes)):
        for j in range(i, len(sizes)):
          energy_hessian_block_dimensions[-1].append((sizes[i], sizes[j]))
    # get the unique block sizes and sort them based on the block sizes
    self.__blockDimensions = list(set([item for sublist in energy_hessian_block_dimensions for item in sublist] + [(item[1], item[0]) for sublist in energy_hessian_block_dimensions for item in sublist])) # add the block dimensions and the transposed block dimension to be safe
    self.__blockDimensions.sort()

    # we now need to know, for each energy, for each block, where does it reside in the compressed coordinates in different block sizes
    uncompressedIndicesByDimensions: List[np.ndarray] = [np.zeros((0, 2), dtype = np.uint32) for _ in range(len(self.__blockDimensions))]
    for i in range(len(index_start) - 1):
      start = index_start[i]
      end = index_start[i + 1]
      # ok now first, we check where to put the blocks
      hessian_block_dimensions = energy_hessian_block_dimensions[i]
      where_to_put = [self.__blockDimensions.index(item) for item in hessian_block_dimensions]
      where_to_put_transposed = [self.__blockDimensions.index((item[1], item[0])) for item in hessian_block_dimensions]
      where_to_put_unique = list(set(where_to_put + where_to_put_transposed)) # get the unique blocks
      where_to_put_grouped = [[] for _ in range(len(where_to_put_unique))]
      for j in range(len(where_to_put)):
        where_to_put_grouped[where_to_put_unique.index(where_to_put[j])].append(j)
        where_to_put_grouped[where_to_put_unique.index(where_to_put_transposed[j])].append(j)
      # ok now we know where to put each coordinate
      # let's do it
      step = len(hessian_block_dimensions)
      for j in range(len(where_to_put_unique)):
        group = where_to_put_grouped[j] # get the blocks that have the same dimension
        # get the subset in group
        subsets = [uncompressedIndicesNumpy[start + k : end : step] for k in group]
        subset = np.vstack(subsets)
        left_mask = subset[:, 0] <= subset[:, 1]
        left_subset = subset[left_mask]
        right_subset = subset[~left_mask]
        if len(right_subset) > 0:
          right_subset = right_subset[:, [1, 0]]
        uncompressedIndicesByDimensions[where_to_put_unique[j]] = np.vstack([uncompressedIndicesByDimensions[where_to_put_unique[j]], left_subset, right_subset])
    print("Compressed indices set")
    # ok now we have the indices put to their corresponding place
    # we can remove the duplicates
    encoded = [((arr_64[:, 0].astype(np.uint64) << np.uint64(32)) | arr_64[:, 1].astype(np.uint64)) for arr_64 in uncompressedIndicesByDimensions]
    compressedIndices = [
      np.unique(item)
      for item in encoded
    ]
    # clear up memory
    del uncompressedIndicesByDimensions
    # decode compressedIndices to uint32
    compressedIndicesDecoded = [
      np.vstack((item >> np.uint64(32), item & np.uint64(0xFFFFFFFF))).T.astype(np.uint32)
      for item in compressedIndices
    ]
    print(f"There are {sum([len(x) for x in compressedIndices])} unique blocks")

    # ###################################################
    # ## remove this code, this is for analysis
    # ###################################################

    # start = time.time()
    # sorted_block_sizes = []
    # sorted_positions = []

    # for i in range(len(compressedIndicesDecoded)):
    #   dimension = self.__blockDimensions[i]
    #   dimension = sorted(dimension)
    #   index = i
    #   if dimension in sorted_block_sizes:
    #     index = sorted_block_sizes.index(dimension)
    #   else:
    #     sorted_block_sizes.append(dimension)
    #     sorted_positions.append([])
    #     index = len(sorted_block_sizes) - 1
    #   sorted_positions[index] += compressedIndicesDecoded[i]

    # totalNNZ = 0
    # for i in range(len(sorted_positions)):
    #   uniquePairs = {tuple(sorted((x, y))) for x, y in sorted_positions[i]}
    #   diagonalPairs = sum(x == y for x, y in uniquePairs)
    #   uniquePairsCount = len(uniquePairs)
    #   diagonalBlockCount = diagonalPairs
    #   dimension = sorted_block_sizes[i]
    #   nnz = dimension[0] * dimension[1] * uniquePairsCount * 2 - dimension[0] * dimension[1] * diagonalBlockCount
    #   totalNNZ += nnz
    # print(f"Total NNZ is: {totalNNZ}")
    # end = time.time()
    # print(f"Analysis: {1000.0 * (end - start)} ms")
    # ###################################################
    # ## End of analysis
    # ###################################################


    self.__blockCounts = [len(item) for item in compressedIndices]
    self.__blockPositions = gpuarray.to_gpu(np.concatenate([x.flatten() for x in compressedIndicesDecoded]))
    # now we segment the list to the positions list
    total_count: int = 0
    for i in range(len(self.__blockCounts)):
      last_count = total_count
      total_count += self.__blockCounts[i]
      if last_count != total_count: # if there are any blocks
        self.__blockPositionsList.append(self.__blockPositions[last_count * 2: total_count * 2])
      else:
        self.__blockPositionsList.append(gpuarray.empty(0, dtype = np.uint32))
    # now we need to allocate the gpu arrays, first we allocate memory for the entire chunk
    blocksStartIndices_cpu = [0]
    for i in range(len(self.__blockDimensions)):
      blocksStartIndices_cpu.append(blocksStartIndices_cpu[-1] + self.__blockCounts[i] * self.__blockDimensions[i][0] * self.__blockDimensions[i][1])
    self.__blocksStartIndices = blocksStartIndices_cpu
    self.__blocksStartIndicesGPU = gpuarray.to_gpu(np.array(blocksStartIndices_cpu, dtype = np.uint32))
    self.__blocksFlattened = gpuarray.empty(blocksStartIndices_cpu[-1], dtype = np.float64)

    # now we need to assign the correct block to each block
    for i in range(len(self.__blockDimensions)):
      if blocksStartIndices_cpu[i] != blocksStartIndices_cpu[i + 1]: # block is not empty
        self.__blocks.append(self.__blocksFlattened[blocksStartIndices_cpu[i]: blocksStartIndices_cpu[i + 1]])
      else:
        self.__blocks.append(gpuarray.empty(0, dtype = np.float64))

    # Preprocess compressed indices into searchable structures
    compressed_values = []
    for indices in compressedIndices:
      if len(indices) == 0:
        # Handle empty case
        compressed_values.append(np.array([], dtype=np.uint32))
        continue
      compressed_values.append(np.arange(len(indices), dtype=np.uint32))

    for i in range(len(index_start) - 1):
      start = index_start[i]
      end = index_start[i + 1]
      uncompressedIndicesLocal = uncompressedIndicesNumpy[start:end, :]
      # Original logic for where_to_check
      hessian_block_dimensions = energy_hessian_block_dimensions[i]
      where_to_check = [self.__blockDimensions.index(item) for item in hessian_block_dimensions]
      where_to_check += [self.__blockDimensions.index((item[1], item[0])) for item in hessian_block_dimensions]
      self.__energies[i].hessian_blocks_where_to_check = gpuarray.to_gpu(np.array(where_to_check, dtype=np.uint32))
      # Vectorized coordinate processing
      x = uncompressedIndicesLocal[:, 0].astype(np.int64)
      y = uncompressedIndicesLocal[:, 1].astype(np.int64)
      mask = x <= y
      x_flipped = np.where(mask, x, y)
      y_flipped = np.where(mask, y, x)
      # Calculate block indices
      m = len(hessian_block_dimensions)
      j_indices = np.arange(len(x)) % m
      where_to_check_np = np.array(where_to_check, dtype=np.uint32)
      where_to_check_indices = j_indices + (m * (~mask)).astype(int)
      block_indices = where_to_check_np[where_to_check_indices]
      # Create combined keys
      current_keys = (x_flipped << 32) | y_flipped
      # Prepare result array
      hessian_block_indices = np.empty(len(x), dtype=np.uint32)
      # Process unique blocks in bulk
      unique_blocks, inverse = np.unique(block_indices, return_inverse=True)
      for block in unique_blocks:
        block_mask = (block_indices == block)
        bk = current_keys[block_mask]
        # Get precomputed search structures
        block_keys = compressedIndices[block]
        block_values = compressed_values[block]
        # Binary search
        idx = np.searchsorted(block_keys, bk)
        # Verify matches
        valid = (idx < len(block_keys)) & (block_keys[idx] == bk)
        hessian_block_indices[block_mask] = np.where(valid, block_values[idx], 0)  # Handle missing as needed
      # GPU transfer remains the same
      self.energies[i].block_indices_gpu = gpuarray.to_gpu(hessian_block_indices.astype(np.uint32))
    print("Sparse indices set")

  def generateHessianAndGradient(self):
    start = time.time()
    for e in self.energies:
      e.generateHessianAndGradient(self.wrt)
    end = time.time()
    print(f"Autodiff computation: {1000.0 * (end - start)} ms")

  def computeSolution(self, tolerance = 1e-3) -> List[gpuarray.GPUArray]:
    self.computeHessianAndGradient(tolerance = tolerance)
    return self.solutionSegments

  def computeHessianAndGradient(self, tolerance = 1e-3):
    # set gradient and hessian to 0
    self.__gradient.fill(0)
    if self.__blocksFlattened.shape[0] > 0:
      self.__blocksFlattened.fill(0)
    self.__diagonal.fill(0)
    for e in self.energies:
      e.computeHessianAndGradient(self.__blocksStartIndicesGPU, self.__gradient, self.__blocksFlattened, self.__diagonal)
    # print("Gradient is before solve: ")
    # print(self.__gradient.get())

    # now we have the hessian and gradient
    # we need to solve the system
    if self.__solver is None:
      self.__solver = solverKernel(self.__blockDimensions)
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

    assert self.__solver is not None
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
    self.__solver.computeSolution(context_ptr_c, 20000, tolerance, self.__blocksFlattened, self.__blockPositions, self.__blocksStartIndices, self.__blockCounts, self.__diagonal, self.__gradient, self.__d_p1_b, self.__d_r, self.__d_c, self.__d_q, self.__d_s, self.__solution)
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

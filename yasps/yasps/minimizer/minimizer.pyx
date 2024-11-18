# cython: language_level=3
from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from typing import List, Dict, Union, Tuple, Set
from yasps.energy import energy
from yasps.attribute import attribute

class minimizer:
  def __init__(self):
    self.__energies: List[energy] = []
    self.__wrt: List[attribute] = []
    self.__gradient: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64)

    self.__diagonalBlockSizes: List[int] = [] # the size of diagonal blocks
    self.__diagonalBlocks: List[gpuarray.GPUArray] = [] # for each different diagonal block sizes, the datas, since therer is no repeated data, we know exactly the starting position of each block
    self.__diagonalBlocksFlattened: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # the flattened diagonal blocks, the unflattened version is just the segment of this memory
    self.__diagonalBlocksStartIndex: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each diagonal block, where do they start, this is to navigate through the flattened diagonal blocks
    self.__diagonalBlockCounts: List[int] = [] # for each size of diagonal block, the number of blocks
    self.__offDiagonalBlockDimensions: List[Tuple[int, int]] = [] # record the size of the off diagonal blocks
    self.__offDiagonalBlocks: List[gpuarray.GPUArray] = [] # for each different off diagonal block sizes, the datas
    self.__offDiagonalBlocksFlattened: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.float64) # the flattened off diagonal blocks
    self.__offDiagonalBlocksStartIndex: gpuarray.GPUArray = gpuarray.empty(0, dtype = np.uint32) # for each off diagonal block, where do they start, this is to navigate through the flattened off diagonal blocks
    self.__offDiagonalBlockPositions: List[gpuarray.GPUArray] = [] # for each different off diagonal block sizes, for each block, what's its coordinate, we will use it for spmv
    self.__offDiagonalBlockCounts: List[int] = [] # record for each size of off diagonal block, the number of blocks
    self.__gradientSizes: List[int] = []
    self.__gradientSegments: List[gpuarray.GPUArray] = []
    self.__wrtStartIndices: List[int] = []
    # first we check if the wrt has duplicates
    # if it has duplicates, we will raise an error

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

  def addEnergy(self, energy: energy) -> None:
    if energy.hash in [energy.hash for energy in self.__energies]:
      raise ValueError("minimizer.addEnergy: energy already exists.")
    self.__energies.append(energy)


  def addWrt(self, wrt: List[attribute]) -> None:
    seenAttributeHashes: Set[int] = set()
    from yasps.attribute import DATA
    for attribute in wrt:
      if attribute.hash in seenAttributeHashes:
        raise ValueError("minimizer.__init__: wrt has duplicate attributes.")
      if attribute.operator is not DATA:
        raise ValueError("minimizer.__init__: wrt has non-data attributes.")
      seenAttributeHashes.add(attribute.hash)
    self.__wrt.extend(wrt)
    # at this time, we can start initializing the dense vectors
    # and everything else
    self.__getGradientSize() # get the size of the gradient
    print(f"The size of the gradient is: {sum(self.__gradientSizes)}")
    self.__getBlockSizes() # get the block sizes
    print(f"The size of the diagonal blocks are: {self.__diagonalBlockSizes}")
    self.__getSparseIndices() # get the sparse indices
    print(f"sparse indices computation is done")


  def __getGradientSize(self) -> None:
    for item in self.wrt:
      self.__gradientSizes.append(item.size * item.correspondance.numInstances)
    # allocate the array
    self.__gradient = gpuarray.empty(sum(self.__gradientSizes), dtype = np.float64)
    # assign the gradient segments by reference
    start = 0
    self.__wrtStartIndices.append(start) # get where each data element starts
    for size in self.__gradientSizes:
      self.__gradientSegments.append(self.__gradient[start:start + size])
      start += size
      self.__wrtStartIndices.append(start)
    print(f"The size of the gradient is: {sum(self.__gradientSizes)}")
    print(f"The gradient segments sizes are: {self.__gradientSizes}")

  def __getBlockSizes(self):
    # for diagonal blocks we don't remove duplicates
    # this is for easier access and knowing where the coordinate start and end
    total_count: int = 0
    for item in self.wrt:
      self.__diagonalBlockSizes.append(item.size)
      self.__diagonalBlockCounts.append(item.correspondance.numInstances)
      total_count += item.size * item.size * item.correspondance.numInstances
    # allocate the gpu array
    self.__diagonalBlocksFlattened = gpuarray.empty(total_count, dtype = np.float64)
    # now we assign the segments to the diagonal blocks unflattened version
    total_count = 0
    diagonal_blocks_start: List[int] = [0]
    for i in range(len(self.__diagonalBlockSizes)):
      size = self.__diagonalBlockSizes[i]
      count = self.__diagonalBlockCounts[i]
      self.__diagonalBlocks.append(self.__diagonalBlocksFlattened[total_count:total_count + size * size * count])
      total_count += size * size * count
      diagonal_blocks_start.append(total_count)
    # finally convert it to gpu array
    self.__diagonalBlocksStartIndex = gpuarray.to_gpu(np.array(diagonal_blocks_start, dtype = np.uint32))
    # offDiagonalSizes = []
    # # now get the off diagonal block sizes
    # for energy in self.energies:
    #   # we will accumulate the off diagonal block sizes
    #   offDiagonalSizes += energy.getHessianOffDiagonalBlockSizes(self.wrt)
    # self.__offDiagonalBlockDimensions = list(set(offDiagonalSizes))

  def __getSparseIndices(self):
    for energy in self.energies:
      energy.getSparseIndices(self.wrt, self.__wrtStartIndices)
    # ok now we have the indices for all off diagonal block in their uncompressed order
    # what we need to do is to find the compressed sparse indices
    uncompressedIndices: List[Tuple[int, int]] = []
    index_start: List[int] = [0] # for each energy, where does the index start
    energy_gradient_sizes: List[List[int]] = []
    for energy in self.energies:
      # for now let's say i don't really care if we only store the upper triangular parts
      # here we expand the indices to store the location of upper triangular blocks of the local hessian
      # but the stored global matrix doesn't have to be upper triangular
      indices = energy.indices # get the gradient indices, by iterating over them we get the position for each small block in the hessian
      gradient_sizes_local = energy.gradient_sizes_cpu # we need to know how many blocks there are, and what are the block sizes
      energy_gradient_sizes.append(gradient_sizes_local) # know how large each block is

      # here we get the coordinate for each block in the hessian
      arr = indices.reshape(-1, len(gradient_sizes_local))
      N, L = arr.shape
      row_indices, col_indices = np.triu_indices(L, k=1)
      first_values = arr[:, row_indices].ravel()
      second_values = arr[:, col_indices].ravel()
      flattened_pairs = list(zip(first_values, second_values))
      uncompressedIndices += flattened_pairs

      # know for each energy, where does the first coordinate starts
      index_start.append(len(uncompressedIndices))

    # we now need to know for each energy, what are the dimension of the blocks
    energy_hessian_block_dimensions: List[List[Tuple[int, int]]] = []
    for sizes in energy_gradient_sizes:
      energy_hessian_block_dimensions.append([])
      for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
          energy_hessian_block_dimensions[-1].append((sizes[i], sizes[j]))

    # get the unique block sizes and sort them based on the block sizes
    self.__offDiagonalBlockDimensions = list(set([item for sublist in energy_hessian_block_dimensions for item in sublist]))
    self.__offDiagonalBlockDimensions.sort()

    # we now need to know, for each energy, for each block, where does it reside in the compressed coordinates in different block sizes
    uncompressedIndicesByDimensions: List[List[Tuple[int, int]]] = [[] for _ in range(len(self.__offDiagonalBlockDimensions))]
    for i in range(len(index_start) - 1):
      start = index_start[i]
      end = index_start[i + 1]
      # ok now first, we check where to put the blocks
      hessian_block_dimensions = energy_hessian_block_dimensions[i]
      where_to_put = [self.__offDiagonalBlockDimensions.index(item) for item in hessian_block_dimensions]
      # ok now we know where to put each coordinate
      # let's do it
      for j in range(len(where_to_put)):
        # for each dimension, we need to put all the uncompressed indices
        uncompressedIndicesByDimensions[where_to_put[j]] += uncompressedIndices[start + j : end : len(hessian_block_dimensions)]
    # ok now we have the indices put to their corresponding place
    # we can remove the duplicates
    compressedIndices: List[List[Tuple[int, int]]] = [list(map(tuple, (np.unique(np.array(item), axis = 0)))) for item in uncompressedIndicesByDimensions]
    self.__offDiagonalBlockCounts = [len(item) for item in compressedIndices]
    self.__offDiagonalBlockPositions = [gpuarray.to_gpu(np.array(item, dtype = np.uint32)) for item in compressedIndices]
    # now we need to allocate the gpu arrays, first we allocate memory for the entire chunk
    offDiagonalBlocksStartIndices_cpu = [0]
    for i in range(len(self.__offDiagonalBlockDimensions)):
      offDiagonalBlocksStartIndices_cpu.append(offDiagonalBlocksStartIndices_cpu[-1] + self.__offDiagonalBlockCounts[i] * self.__offDiagonalBlockDimensions[i][0] * self.__offDiagonalBlockDimensions[i][1])
    self.__offDiagonalBlocksStartIndices = gpuarray.to_gpu(np.array(offDiagonalBlocksStartIndices_cpu, dtype = np.uint32))
    self.__offDiagonalBlocksFlattened = gpuarray.empty(offDiagonalBlocksStartIndices_cpu[-1], dtype = np.float64)
    self.__offDiagonalBlocksStartIndex = gpuarray.to_gpu(np.array(offDiagonalBlocksStartIndices_cpu, dtype = np.uint32))

    # now we need to assign the correct block to each off diagonal block
    for i in range(len(self.__offDiagonalBlockDimensions)):
      self.__offDiagonalBlocks.append(self.__offDiagonalBlocksFlattened[offDiagonalBlocksStartIndices_cpu[i]: offDiagonalBlocksStartIndices_cpu[i + 1]])

    # now we sotre all the unique coordinates for each dimension
    self.__offDiagonalBlockIndices = [gpuarray.to_gpu(np.array(item, dtype = np.uint32)) for item in compressedIndices]

    # now for each of the energy, they need to know where to put the blocks
    for i in range(len(index_start) - 1):
      start = index_start[i]
      end = index_start[i + 1]
      uncompressedIndicesLocal = uncompressedIndices[start:end] # get the coordinates in the uncompressed order
      hessian_block_dimensions = energy_hessian_block_dimensions[i] # get the dimensions of the blocks
      where_to_check = [self.__offDiagonalBlockDimensions.index(item) for item in hessian_block_dimensions] # we need to know where to check (the index of that dimension)
      self.__energies[i].hessian_off_diagonal_block_where_to_check = gpuarray.to_gpu(np.array(where_to_check, dtype = np.uint32)) # we store where to check for each block
      # ok now we know for each coordinate, which block to check
      # we need to get the index of the block
      hessian_block_indices: List[int] = [compressedIndices[where_to_check[j % len(where_to_check)]].index(uncompressedIndicesLocal[j]) for j in range(len(uncompressedIndicesLocal))]
      self.energies[i].block_indices_gpu = gpuarray.to_gpu(np.array(hessian_block_indices, dtype = np.uint32)) # we now store for each smaller block, what is the index in the data array

  def generateHessianAndGradient(self):
    for energy in self.energies:
      energy.generateHessianAndGradient(self.wrt)

  def computeHessianAndGradient(self):
    # set gradient and hessian to 0
    self.__gradient.fill(0)
    self.__diagonalBlocksFlattened.fill(0)
    self.__offDiagonalBlocksFlattened.fill(0)
    for energy in self.energies:
      energy.computeHessianAndGradient(self.__diagonalBlocksStartIndex, self.__offDiagonalBlocksStartIndex, self.__gradient, self.__diagonalBlocksFlattened, self.__offDiagonalBlocksFlattened)

    print(self.__diagonalBlocksFlattened.get())

    for i in range(len(self.__offDiagonalBlockDimensions)):
      print(f"Dimension: {self.__offDiagonalBlockDimensions[i][0]}, {self.__offDiagonalBlockDimensions[i][0]}")
      block_size = self.__offDiagonalBlockDimensions[i][0] * self.__offDiagonalBlockDimensions[i][1]
      for j in range(self.__offDiagonalBlockPositions[i].get().shape[0]):
        print(f"Pos {self.__offDiagonalBlockPositions[i].get()[j]}")
        block = self.__offDiagonalBlocks[i].get()[j * block_size: (j + 1) * block_size]
        print(block.reshape(self.__offDiagonalBlockDimensions[i][0], self.__offDiagonalBlockDimensions[i][1]))

    # we assemble a mat to see
    # mat = np.zeros((self.__gradient.size, self.__gradient.size))
    # for i in range(len(self.__diagonalBlockSizes)):
    #   size = self.__diagonalBlockSizes[i]
    #   start = self.__diagonalBlocksStartIndex.get()[i]
    #   end = self.__diagonalBlocksStartIndex.get()[i + 1]
    #   num_instances = (end - start) // size
    #   for j in range(num_instances):
    #     mat[start + j * size: start + (j + 1) * size, start + j * size: start + (j + 1) * size] = self.__diagonalBlocks[i].get()[j * size: (j + 1) * size]

    # for i in range(len(self.__offDiagonalBlockDimensions)):
    #   row_size = self.__offDiagonalBlockDimensions[i][0]
    #   col_size = self.__offDiagonalBlockDimensions[i][1]

    #   coordinate_row =

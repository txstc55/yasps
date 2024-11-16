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
    self.__diagonalBlockCounts: List[int] = [] # for each size of diagonal block, the number of blocks
    self.__offDiagonalBlockDimensions: List[Tuple[int, int]] = [] # record the size of the off diagonal blocks
    self.__OffDiagonalBlocks: List[gpuarray.GPUArray] = [] # for each different off diagonal block sizes, the datas
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
    self.__getBlockSizes() # get the block sizes
    self.__getSparseIndices() # get the sparse indices


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
    for item in self.wrt:
      if item.size not in self.__diagonalBlockSizes:
        self.__diagonalBlockSizes.append(item.size)
        self.__diagonalBlockCounts.append(item.correspondance.numInstances)
      else:
        index = self.__diagonalBlockSizes.index(item.size)
        self.__diagonalBlockCounts[index] += item.correspondance.numInstances
    # allocate the gpu array
    for size in self.__diagonalBlockSizes:
      self.__diagonalBlocks.append(gpuarray.empty(size * size, dtype = np.float64))
    offDiagonalSizes = []
    # now get the off diagonal block sizes
    for energy in self.energies:
      # we will accumulate the off diagonal block sizes
      offDiagonalSizes += energy.getHessianOffDiagonalBlockSizes(self.wrt)
    self.__offDiagonalBlockDimensions = list(set(offDiagonalSizes))

  def __getSparseIndices(self):
    for energy in self.energies:
      energy.getSparseIndices(self.wrt, self.__wrtStartIndices)
    # ok now we have the indices for all off diagonal block in their uncompressed order
    # what we need to do is to find the compressed sparse indices
    uncompressedIndices: List[Tuple[int, int]] = []
    index_start: List[int] = [] # for each energy, where does the index start
    energy_gradient_sizes: List[List[int]] = []
    for energy in self.energies:
      # for now let's say i don't really care if we only store the upper triangular parts
      # here we expand the indices to store the location of upper triangular blocks of the local hessian
      # but the stored global matrix doesn't have to be upper triangular
      indices = energy.indices # get the list of indices
      gradient_sizes_local = energy.gradientSizes # get how many items are in each list of indices
      energy_gradient_sizes.append(gradient_sizes_local) # know how large each block is
      arr = indices.reshape(-1, len(gradient_sizes_local))
      N, L = arr.shape
      row_indices, col_indices = np.triu_indices(L, k=1)
      first_values = arr[:, row_indices].ravel()
      second_values = arr[:, col_indices].ravel()
      flattened_pairs = list(zip(first_values, second_values))
      uncompressedIndices += flattened_pairs
      index_start.append(len(uncompressedIndices))

    energy_hessian_block_dimensions: List[List[Tuple[int, int]]] = []
    for sizes in energy_gradient_sizes:
      energy_hessian_block_dimensions.append([])
      for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
          energy_hessian_block_dimensions[-1].append((sizes[i], sizes[j]))

    # get the unique block sizes and sort them to be the off diagonal block sizes
    self.__offDiagonalBlockDimensions = list(set([item for sublist in energy_hessian_block_dimensions for item in sublist]))
    self.__offDiagonalBlockDimensions.sort()

    uncompressedIndicesByDimensions = [[] for _ in range(len(self.__offDiagonalBlockDimensions))]
    for i in range(len(index_start) - 1):
      start = index_start[i]
      end = index_start[i + 1]
      # ok now first, we check where to put the blocks
      hessian_block_dimensions = energy_hessian_block_dimensions[i]
      where_to_put = [self.__offDiagonalBlockDimensions.index(item) for item in hessian_block_dimensions]
      # ok now we know where to put each coordinate
      # let's do it
      for j in range(len(where_to_put)):
        uncompressedIndicesByDimensions[where_to_put[j]] += uncompressedIndices[start + j : end : len(hessian_block_dimensions)]
    # ok now we have the indices put to their corresponding place
    # we can remove the duplicates
    compressedIndices = [list(map(tuple, (np.unique(np.array(item), axis = 0)))) for item in uncompressedIndicesByDimensions]
    self.__offDiagonalBlockCounts = [len(item) for item in compressedIndices]
    self.__offDiagonalBlocks = [gpuarray.empty(count * count, dtype = np.float64) for count in self.__offDiagonalBlockCounts]
    self.__offDiagonalBlockIndices = [gpuarray.to_gpu(np.array(item, dtype = np.int32)) for item in compressedIndices]

    # now for each of the energy, they need to know where to put the blocks







    # # Convert the list of tuples to a NumPy array, then remove the
    # np_array = np.array(uncompressedIndices)
    # unique_sorted_array = np.unique(np_array, axis=0)
    # compressedIndices = list(map(tuple, unique_sorted_array))

    # now we need to determine the block sizes





  def generateHessianAndGradient(self):
    for energy in self.energies:
      energy.generateHessianAndGradient(self.wrt)

  def computeHessianAndGradient(self):
    # set gradient to zero
    self.__gradient.fill(0)
    for energy in self.energies:
      energy.computeHessianAndGradient(self.__gradient)

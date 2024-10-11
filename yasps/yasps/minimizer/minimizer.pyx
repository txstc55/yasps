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
    self.__diagonalBlocks: List[gpuarray.GPUArray] = []
    self.__diagonalBlockSizes: List[int] = []
    self.__diagonalBlockCounts: List[int] = []
    self.__OffDiagonalBlocks: List[gpuarray.GPUArray] = []
    self.__offDiagonalBlockSizes: List[Tuple[int, int]] = []
    self.__gradientSizes: List[int] = []
    self.__gradientSegments: List[gpuarray.GPUArray] = []
    # first we check if the wrt has duplicates
    # if it has duplicates, we will raise an error

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


  def __getGradientSize(self) -> None:
    for item in self.__wrt:
      self.__gradientSizes.append(item.size * item.correspondance.numInstances)
    # allocate the array
    self.__gradient = gpuarray.empty(sum(self.__gradientSizes), dtype = np.float64)
    # assign the gradient segments by reference
    start = 0
    for size in self.__gradientSizes:
      self.__gradientSegments.append(self.__gradient[start:start + size])
      start += size
    print(f"The size of the gradient is: {sum(self.__gradientSizes)}")
    print(f"The gradient segments sizes are: {self.__gradientSizes}")

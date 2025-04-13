from __future__ import annotations
from yasps.attribute import attribute
from typing import List, Tuple, Dict
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
import pycuda.driver as cuda
from yasps.attribute import JOIN, DATA, UNION
from yasps.helper import timed
from math import exp


# The gradient indices kernel needs to produce couple of things:
# The most obvious one is globally, how to map each block to the global matrix
# The not so obvious one is locally, within this local hessian, how can I compress iter
# The compression is used so that when variables are dependent on each other
# we can reduce the size of the hessian
# The third thing is the dimension of each block
# The third thing is just the size of the compressed hessian
# so that we can pre allocate a kernel for that specific hessian size (for projection)
# The first three things should have same dimension, as in the output array have the same size
class gradientIndicesKernel:
  def __init__(self, path_dict: Dict[attribute, List[attribute]], wrt: List[attribute], wrt_start_indices: List[int], energy: attribute):
    self.__path_dict: Dict[attribute, List[attribute]] = path_dict
    self.__wrt_start_indices: List[int] = wrt_start_indices
    self.__energy: attribute = energy
    self.__gradient_size: int = 0
    self.__used_join_attributes: List[attribute] = [] # all the join attributes, we will use its connectivities for indexing
    self.__gradientSizeForEachPart: Dict[attribute, int] = {} # determine for each attribute, the size of the gradient being used
    self.__indexSizeForEachPart: Dict[attribute, int] = {} # determine for each attribute, the number of indices needed
    self.__getUsedJoinAttributes() # get the attributes that are join operations, so we can just grab the connectivities later on
    self.__gradientSize, self.__indexSize = self.__getGradientSize(self.__path_dict, self.__energy)
    self.__positionInWrtStartIndices: Dict[attribute, int] = {} # we record the position in the wrt start indices
    for i in range(len(wrt)):
      self.__positionInWrtStartIndices[wrt[i]] = wrt_start_indices[i]
    self.__kernelString = ""
    # the output data
    self.__outputIndices = gpuarray.empty(self.__indexSizeForEachPart[energy] * energy.correspondance.numInstances, dtype=np.int32)
    self.__outputSizes = gpuarray.empty(self.__indexSizeForEachPart[energy] * energy.correspondance.numInstances, dtype=np.int32)
    self.__indices_kernel = None # the kernel for computing the indices
    self.__generateKernel()


  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(int(x.gpudata))

  def __getGradientSize(self, path_dict: Dict[attribute, List[attribute]], current_attribute: attribute) -> Tuple[int, int]:
    # we first get the list of attributes to look for
    children_attributes: List[attribute] = path_dict[current_attribute]
    total_size = 0
    index_size = 0
    for child in children_attributes:
      if child.operator == DATA:
        total_size += child.size
        index_size += 1
      elif child.operator == JOIN:
        child_gradient_size, child_index_size = self.__getGradientSize(path_dict, child)
        total_size += child.through.dimension * child_gradient_size
        index_size += child.through.dimension * child_index_size
      elif child.operator == UNION:
        # do nothing for now
        total_size += 0
        index_size += 0
    self.__gradientSizeForEachPart[current_attribute] = total_size # also record that for each join operation, the size we need to reserve
    self.__indexSizeForEachPart[current_attribute] = index_size
    return total_size, index_size

  def __getUsedJoinAttributes(self):
    # we literally just go over the attributes
    for att in self.__path_dict.keys():
      if att.operator == JOIN:
        self.__used_join_attributes.append(att)

  @timed("Generate Kernel for Index")
  def __generateKernel(self):
    # ok now we need to generate the kernel
    # what we basically aim to do
    # is to have an isolated kernel that calls the children kernel to fetch the data
    self.__kernelString += '''
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
'''
    # now, for each parent-child relationship, we will have a kernel
    for parent in self.__path_dict.keys():
      for child in self.__path_dict[parent]:
        # now put a kernel function header
        self.__kernelString += f'''
__device__ inline void {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes])}, const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index);
'''
      # we also add a header function that will be used to fetch the index for the entire parent and children
      self.__kernelString += f'''
__device__ inline void {parent.fullName}_get_indices({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes])}, const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index);
'''
    # ok we have produced the header functions, we will start the actual implementation
    for parent in self.__path_dict.keys():
      for child in self.__path_dict[parent]:
        # we will now construct the actual function
        self.__kernelString += f'''
__device__ inline void {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes])}, const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index){{'''
        if child.operator == DATA:
          # we have reached the bottom
          # we now record that index, added by offset, and the size of this attribute
          # we determine its position in wrt
          pos: int = self.__positionInWrtStartIndices[child]
          self.__kernelString += f'''
  outputIndices[0] = wrtStartIndices[{pos}] + index * {child.size if child.correspondance.type == "primitive" else 0}; // map the index to the index in final gradient array
  outputSizes[0] = {child.size}; // provide the size of this attribute for a single instance
'''
        elif child.operator == JOIN:
          # we are at a join operator again
          # we will need to get the children attributes
          self.__kernelString += f'''
  // we still need to keep going, go ahead and call the function that accumulates indices
  {child.fullName}_get_indices({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, wrtStartIndices, outputIndices, outputSizes, index);
'''
        self.__kernelString += f'''
}} // end of kernel for grabbing indices from {child.fullName} to {parent.fullName}
'''
      # now we have done the kernel for each children
      # we will need to do a kernel for total accumulation
      self.__kernelString += f'''
  __device__ inline void {parent.fullName}_get_indices({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes])}, const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index){{
    const int num_children = {len(self.__path_dict[parent])};
    const int childrenIndexSizes[num_children] = {{{", ".join([str(self.__indexSizeForEachPart[x]) for x in self.__path_dict[parent]])}}}; // here we first know how much space to save for each child
    // we expand the for loop directly
'''
      index_accumulation: int = 0
      for ind in range(len(self.__path_dict[parent])):
        child = self.__path_dict[parent][ind]
        # call the function
        if child.operator == JOIN:
          # we make sure the index size is divisible
          assert self.__indexSizeForEachPart[child] % child.through.dimension == 0
          self.__kernelString += f'''
  for (unsigned int i = 0; i < {child.through.dimension}, i++){{
    {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, wrtStartIndices, outputIndices + {index_accumulation} + {self.__indexSizeForEachPart[child] // child.through.dimension} * i, outputSizes + {index_accumulation} + {self.__indexSizeForEachPart[child] // child.through.dimension} * i, {(child.fullName + f"_indices[index * {child.through.dimension} + i]")});
  }}
'''
          # add the index accumulation
          index_accumulation += self.__indexSizeForEachPart[child]
        elif child.operator == DATA:
          self.__kernelString += f'''
  {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, wrtStartIndices, outputIndices + {index_accumulation}, outputSizes + {index_accumulation}, index);
'''
          # add the index accumulation
          index_accumulation += self.__indexSizeForEachPart[child]
      self.__kernelString += f'''
}} // end of kernel for grabbing indices from of {parent.fullName}
'''

    # now we can do the global kernel
    self.__kernelString += f'''
__global__ void {self.__energy.fullName}_get_indices_global_function({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int NUM_INSTANCES) {{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < NUM_INSTANCES){{
    {self.__energy.fullName}_get_indices({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, wrtStartIndices, outputIndices + index * {self.__indexSizeForEachPart[self.__energy]}, outputSizes + index * {self.__indexSizeForEachPart[self.__energy]}, index);
  }}
}}
'''
    # now we do the function on c side
    self.__kernelString += f'''
extern "C" void get_indices({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int NUM_INSTANCES) {{
  {self.__energy.fullName}_get_indices_global_function<<<NUM_INSTANCES / 256, 256>>>({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes])}, wrtStartIndices, outputIndices, outputSizes, NUM_INSTANCES);
}}
'''

    # ok now we compile the kernel by saving it to a file and then calling nvcc
    file_name = f".yasps_tmp/{self.__energy.fullName}_get_indices"
    f = open(f"{file_name}.cu", 'w')
    f.write(self.__kernelString)
    f.close()
    # we will now compile this kernel
    import os
    os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda")
    self.__indices_kernel = ctypes.CDLL(f"{file_name}.so").get_indices # get the compiled kernel
    self.__indices_kernel.restype = None # set the return type to None
    self.__indices_kernel.argtypes = [ctypes.POINTER(ctypes.c_void_p)] * len(self.__used_join_attributes) + [ctypes.POINTER(ctypes.c_void_p)] * 3 + [ctypes.c_uint32]

  def computeIndices(self, wrt_start_indices: List[int]):
    # first let's convert wrt_start_indices to a pycuda array
    wrt_start_indices_gpu = gpuarray.to_gpu(wrt_start_indices)
    # then we get all the gpu arrays for the connectivity
    connectivity_list_gpu = [self.__to_void_p(x.through.connectivity.value) for x in self.__used_join_attributes]
    # we clear the output arrays
    self.__outputIndices.fill(0)
    self.__outputSizes.fill(0)
    # now we invoke the kernel
    if self.__indices_kernel is not None:
      self.__indices_kernel(*connectivity_list_gpu, wrt_start_indices_gpu.gpudata, self.__to_void_p(self.__outputIndices), self.__to_void_p(self.__outputSizes), self.__energy.correspondance.numInstances)

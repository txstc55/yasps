from __future__ import annotations
from yasps.attribute import attribute
from typing import List, Tuple, Dict
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
from yasps.attribute import JOIN, DATA, UNION
from yasps.helper import timed
import os

compression_kernel_string = '''
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <vector>
__global__ void computePermutation(const unsigned int* indices, const unsigned int* index_sizes, int* permutation, unsigned int* total_sizes, unsigned int N, unsigned int K) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    int unique_count = 0;
    unsigned int total_size = 0;
    for (unsigned int i = 0; i < K; ++i) {
      // first we check if the index already exists in this local array
      const unsigned int idx_i = indices[tid * K + i];
      bool found = false;
      for (unsigned int j = 0; j < i; j++){
        const unsigned int idx_j = indices[tid * K + j];
        if (idx_j == idx_i) {
          // we found a duplicate
          permutation[tid * K + i] = -permutation[tid * K + j];
          found = true;
          break;
        }
      }
      if (!found) {
        unique_count++; // make the result we get to always exclude 0
        permutation[tid * K + i] = unique_count;
        total_size += index_sizes[tid * K + i];
      }
    }
    total_sizes[tid] = total_size;
  }
}
extern "C" {
void compress_indices(
  const unsigned int* d_indices,         // indices for each local gradient
  const unsigned int* d_index_sizes,           // sizes of each variable for each index
  int* d_permutations,                   // permutations for compression
  unsigned int* d_total_sizes,           // total sizes per gradient
  unsigned int* d_unique_sizes,          // unique gradient sizes
  unsigned int* d_grouped_indices,       // grouped indices by gradient size
  unsigned int* d_offsets,               // offsets for compressed indices
  unsigned int* d_num_unique,            // number of unique sizes
  unsigned int num_instances,
  unsigned int num_indices_for_each_instance
) {
  // Compute permutation and total sizes
  computePermutation<<<(num_instances + 256 - 1) / 256, 256>>>(
    d_indices, d_index_sizes, d_permutations, d_total_sizes, num_instances, num_indices_for_each_instance
  );
  // Wrap existing memory (no extra allocation)
  auto total_sizes_begin = thrust::device_pointer_cast(d_total_sizes);
  auto total_sizes_end   = total_sizes_begin + num_instances;
  auto grouped_indices_begin = thrust::device_pointer_cast(d_grouped_indices);

  // Initialize d_grouped_indices as [0, 1, 2, ..., num_instances-1]
  thrust::sequence(grouped_indices_begin, grouped_indices_begin + num_instances);

  // Sort total_sizes and reorder grouped_indices accordingly
  thrust::sort_by_key(
    total_sizes_begin, total_sizes_end, grouped_indices_begin
  );

  // Compute flags marking unique starts (no extra alloc for sorted_values)
  thrust::device_vector<unsigned int> flags(num_instances);
  flags[0] = 1;
  thrust::transform(
    total_sizes_begin + 1, total_sizes_end,
    total_sizes_begin,
    flags.begin() + 1,
    thrust::not_equal_to<unsigned int>()
  );
  // Get the unique count directly
  unsigned int unique_count = thrust::count(flags.begin(), flags.end(), 1u);
  cudaMemcpy(d_num_unique, &unique_count, sizeof(unsigned int), cudaMemcpyHostToDevice);

  // Compute offsets directly into provided memory (no extra alloc)
  thrust::copy_if(
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator((int)num_instances),
    flags.begin(),
    thrust::device_pointer_cast(d_offsets),
    thrust::identity<unsigned int>()
  );
  // Set the last offset to num_instances explicitly
  cudaMemcpy(d_offsets + unique_count, &num_instances, sizeof(unsigned int), cudaMemcpyHostToDevice);
  // Copy unique sizes directly into provided memory
  thrust::copy_if(
    total_sizes_begin, total_sizes_end,
    flags.begin(),
    thrust::device_pointer_cast(d_unique_sizes),
    thrust::identity<unsigned int>()
  );
  cudaDeviceSynchronize();
}
} // extern "C"
'''

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
  @timed("gradientIndicesKernel.__init__")
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
      self.__positionInWrtStartIndices[wrt[i]] = i
    self.__kernelString = ""
    self.__numInstances: int = 0 # for checking the number of instances
    self.__maxInstances: int = 0 # for allocating the largest gpu array
    # the output data
    ####################################################
    # Here are the uncompressed output indices
    ####################################################
    self.__outputIndices = gpuarray.empty(0, dtype=np.uint32) # this will record the raw indices accumulated
    self.__outputSizes = gpuarray.empty(0, dtype=np.uint32) # this will record the sizes of the attributes
    ####################################################
    # Here are information needed for compressed indices
    ####################################################
    self.__outputPermutations = gpuarray.empty(0, dtype=np.int32) # this will record how to compress the matrix locally
    self.__outputTotalSizes = gpuarray.empty(0, dtype=np.uint32) # this will record the total size of the attributes after compression
    self.__outputUniqueSizes = gpuarray.empty(0, dtype=np.uint32) # this will record the unique sizes of the attributes after compression
    self.__outputGroupedIndices = gpuarray.empty(0, dtype=np.uint32) # this will record the grouped indices by the compressed gradient size
    self.__outputOffsets = gpuarray.empty(0, dtype=np.uint32) # this will record the offsets used to find the starting and ending points of the grouped indices
    self.__outputNumUniqueSizes = gpuarray.empty(0, dtype=np.uint32) # this will record the number of unique sizes of the attributes after compression
    ####################################################
    # Here are the kernels
    ####################################################
    self.__indices_kernel = None # the kernel for computing the indices
    self.__compression_kernel = None # the kernel for compressing the indices
    self.__generateKernel() # generate and compile the kernel
    self.__getCompressionKernel() # generate or just get the compression kernel

    # we can pre allocate the spaces for unique sizes
    self.__outputUniqueSizes = gpuarray.empty(self.__gradientSizeForEachPart[energy], np.uint32) # this is the largest possible size
    self.__outputOffsets = gpuarray.empty(self.__gradientSizeForEachPart[energy] + 1, np.uint32) # this will record the offsets used to find the starting and ending points of the grouped indices
    self.__outputNumUniqueSizes = gpuarray.empty(1, np.uint32) # this will record the number of unique sizes of the attributes after compression


  @property
  def indexSizes(self):
    return self.__indexSizeForEachPart[self.__energy] # return how many indices are needed for the energy, this is used for allocation

  @property
  def outputIndices(self):
    return self.__outputIndices

  @property
  def outputSizes(self):
    return self.__outputSizes

  def __getCompressionKernel(self):
    if self.__compression_kernel is None:
      file_name = ".yasps_tmp/compression_kernel"
      # check if the file exists
      if not os.path.exists(f'{file_name}.so'):
        # generate the kernel
        f = open(f"{file_name}.cu", 'w')
        f.write(compression_kernel_string)
        f.close()
        os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda")
        self.__compression_kernel = ctypes.CDLL(f"{file_name}.so").compress_indices # get the compiled kernel
        self.__compression_kernel.restype = None # set the return type to None
        self.__compression_kernel.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 2
      else:
        self.__compression_kernel = ctypes.CDLL(f"{file_name}.so").compress_indices # get the compiled kernel
        self.__compression_kernel.restype = None # set the return type to None
        self.__compression_kernel.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 2



  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))


  def __getGradientSize(self, path_dict: Dict[attribute, List[attribute]], current_attribute: attribute) -> Tuple[int, int]:
    # we first get the list of attributes to look for
    children_attributes: List[attribute] = path_dict[current_attribute]
    total_size = 0
    index_size = 0
    for child in children_attributes:
      if child.operator == DATA:
        # we stop the recursion when child is a DATA attribute
        if child not in self.__gradientSizeForEachPart:
          self.__gradientSizeForEachPart[child] = child.size # the gradient size for the child is its own size
        if child not in self.__indexSizeForEachPart:
          self.__indexSizeForEachPart[child] = 1 # the index size for the child is 1
        total_size += child.size
        index_size += 1
      elif child.operator == JOIN:
        child_gradient_size, child_index_size = self.__getGradientSize(path_dict, child)
        total_size += child_gradient_size
        index_size += child_index_size
      elif child.operator == UNION:
        # do nothing for now
        total_size += 0
        index_size += 0
    if current_attribute.operator == JOIN:
      total_size = total_size * current_attribute.through.dimension
      index_size = index_size * current_attribute.through.dimension
    self.__gradientSizeForEachPart[current_attribute] = total_size # also record that for each join operation, the size we need to reserve
    self.__indexSizeForEachPart[current_attribute] = index_size
    return total_size, index_size

  def __getUsedJoinAttributes(self):
    # we literally just go over the attributes
    for att in self.__path_dict.keys():
      if att.operator == JOIN:
        self.__used_join_attributes.append(att)

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
__device__ inline void {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index);
'''
      # we also add a header function that will be used to fetch the index for the entire parent and children
      self.__kernelString += f'''
__device__ inline void {parent.fullName}_get_indices({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index);
'''
    # ok we have produced the header functions, we will start the actual implementation
    for parent in self.__path_dict.keys():
      for child in self.__path_dict[parent]:
        # we will now construct the actual function
        self.__kernelString += f'''
__device__ inline void {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index){{'''
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
  {child.fullName}_get_indices({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}wrtStartIndices, outputIndices, outputSizes, index);
'''
        self.__kernelString += f'''
}} // end of kernel for grabbing indices from {child.fullName} to {parent.fullName}
'''
      # now we have done the kernel for each children
      # we will need to do a kernel for total accumulation
      # print("path dict keys")
      # print([x.fullName for x in self.__path_dict.keys()])
      # print("parent: ")
      # print(parent.fullName)
      # print("Index sizes for each part")
      # print(self.__indexSizeForEachPart)
      # print("Children")
      # print([x.fullName for x in self.__path_dict[parent]])
      self.__kernelString += f'''
__device__ inline void {parent.fullName}_get_indices({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int index){{
  // const int num_children = {len(self.__path_dict[parent])};
  // const int childrenIndexSizes[num_children] = {{{", ".join([str(self.__indexSizeForEachPart[x]) for x in self.__path_dict[parent]])}}}; // here we first know how much space to save for each child
  // we expand the for loop directly
'''
      index_accumulation: int = 0
      for ind in range(len(self.__path_dict[parent])):
        child = self.__path_dict[parent][ind]
        # call the function
        if child.operator == JOIN:
          # we make sure the index size is divisible
          assert self.__indexSizeForEachPart[child] % child.through.dimension == 0, f"Index size {self.__indexSizeForEachPart[child]} is not divisible by dimension {child.through.dimension}"
          self.__kernelString += f'''
  for (unsigned int i = 0; i < {child.through.dimension}; i++){{
    {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}wrtStartIndices, outputIndices + {index_accumulation} + {self.__indexSizeForEachPart[child] // child.through.dimension} * i, outputSizes + {index_accumulation} + {self.__indexSizeForEachPart[child] // child.through.dimension} * i, {(child.fullName + f"_indices[index * {child.through.dimension} + i]")});
  }}
'''
          # add the index accumulation
          index_accumulation += self.__indexSizeForEachPart[child]
        elif child.operator == DATA:
          self.__kernelString += f'''
  {parent.fullName}_get_indices_from_{child.fullName}({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}wrtStartIndices, outputIndices + {index_accumulation}, outputSizes + {index_accumulation}, index);
'''
          # add the index accumulation
          index_accumulation += self.__indexSizeForEachPart[child]
      self.__kernelString += f'''
}} // end of kernel for grabbing indices from of {parent.fullName}
'''

    # now we can do the global kernel
    self.__kernelString += f'''
__global__ void {self.__energy.fullName}_get_indices_global_function({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int NUM_INSTANCES) {{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < NUM_INSTANCES) {{
    {self.__energy.fullName}_get_indices({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}wrtStartIndices, outputIndices + index * {self.indexSizes}, outputSizes + index * {self.indexSizes}, index);
  }}
}}
'''
    # now we do the function on c side
    self.__kernelString += f'''
// for checking cuda error
#define CUDA_CHECK_ERROR(ans)                                                  \
  {{ cudaAssert((ans), __FILE__, __LINE__); }}
inline void cudaAssert(cudaError_t code, const char *file, int line,
                        bool abort = true) {{
                        if (code != cudaSuccess) {{
    fprintf(stderr, "CUDA Error: %s at %s:%d\\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }}
}}
extern "C" void get_indices({", ".join([f"const unsigned int* {x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}const unsigned int* wrtStartIndices, unsigned int* outputIndices, unsigned int* outputSizes, unsigned int NUM_INSTANCES) {{
  {self.__energy.fullName}_get_indices_global_function<<<(NUM_INSTANCES + 32 - 1) / 32, 32>>>({", ".join([f"{x.fullName}_indices" for x in self.__used_join_attributes]) + ", " if self.__used_join_attributes else ""}wrtStartIndices, outputIndices, outputSizes, NUM_INSTANCES);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}}
'''

    # ok now we compile the kernel by saving it to a file and then calling nvcc
    file_name = f".yasps_tmp/{self.__energy.fullName}_get_indices"
    f = open(f"{file_name}.cu", 'w')
    f.write(self.__kernelString)
    f.close()
    # we will now compile this kernel
    os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda")
    self.__indices_kernel = ctypes.CDLL(f"{file_name}.so").get_indices # get the compiled kernel
    self.__indices_kernel.restype = None # set the return type to None
    self.__indices_kernel.argtypes = [ctypes.c_void_p] * len(self.__used_join_attributes) + [ctypes.c_void_p] * 3 + [ctypes.c_uint32]


  @timed("gradientIndicesKernel.__reallocate")
  def __reallocate(self):
    newNumInstances: int = self.__energy.correspondance.numInstances
    if newNumInstances > self.__maxInstances:
      # resize the gpu arrays
      self.__outputIndices = gpuarray.empty(self.indexSizes * newNumInstances, dtype=np.uint32)
      self.__outputSizes = gpuarray.empty(self.indexSizes * newNumInstances, dtype=np.uint32)
      self.__outputPermutations = gpuarray.empty(self.indexSizes * newNumInstances, dtype=np.int32)
      self.__outputTotalSizes = gpuarray.empty(newNumInstances, dtype=np.uint32)
      self.__outputGroupedIndices = gpuarray.empty(self.indexSizes * newNumInstances, dtype=np.uint32)
      self.__maxInstances = newNumInstances # update the maximum size

    self.__numInstances = newNumInstances # update the number of instances
    # we clear the output arrays
    self.__outputIndices.fill(0)
    self.__outputSizes.fill(0)
    self.__outputPermutations.fill(0)
    self.__outputTotalSizes.fill(0)
    self.__outputGroupedIndices.fill(0)
    self.__outputUniqueSizes.fill(0)
    self.__outputOffsets.fill(0)

  @timed("gradientIndicesKernel.__computeIndices")
  def __computeIndices(self, wrt_start_indices: List[int]):
    # first let's convert wrt_start_indices to a pycuda array
    wrt_start_indices_gpu = gpuarray.to_gpu(np.array(wrt_start_indices, dtype=np.uint32))
    print("wrt start indices:", wrt_start_indices)
    # then we get all the gpu arrays for the connectivity
    connectivity_list_gpu = [self.__to_void_p(x.through.value) for x in self.__used_join_attributes]
    # now we invoke the kernel
    if self.__indices_kernel is not None:
      self.__indices_kernel(*connectivity_list_gpu, self.__to_void_p(wrt_start_indices_gpu), self.__to_void_p(self.__outputIndices), self.__to_void_p(self.__outputSizes), self.__numInstances)

  @timed("gradientIndicesKernel.__compressIndicesLocal")
  def __compressIndicesLocal(self):
    if self.__compression_kernel is not None:
      self.__compression_kernel(self.__to_void_p(self.__outputIndices), self.__to_void_p(self.__outputSizes), self.__to_void_p(self.__outputPermutations), self.__to_void_p(self.__outputTotalSizes), self.__to_void_p(self.__outputUniqueSizes), self.__to_void_p(self.__outputGroupedIndices), self.__to_void_p(self.__outputOffsets), self.__to_void_p(self.__outputNumUniqueSizes), self.__numInstances, self.indexSizes)
      print("Indices", self.__outputIndices.get())
      print("Index sizes:", self.__outputSizes.get())


  @timed("gradientIndicesKernel.computeIndices")
  def computeIndices(self, wrt_start_indices: List[int]):
    self.__reallocate()
    self.__computeIndices(wrt_start_indices)
    self.__compressIndicesLocal()
    print("There are", self.__outputNumUniqueSizes.get()[0], "unique sizes")
    print("The unique sizes are:", self.__outputUniqueSizes.get())
    print("Offsets:", self.__outputOffsets.get())
    print("Grouped Indices:", self.__outputGroupedIndices.get())
    print("")

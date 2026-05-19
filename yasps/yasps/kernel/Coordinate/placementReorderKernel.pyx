# The purpose of the placement reorder kernel
# is for the separate jacobian and hessian case
# when we do that, because we know each block of the final Hessian h_ij is done by
# J_i * H_ij * J_j^T
# now when we finish this block, we will need to figure out how to place it back
# and that's the problem, because currently placement indices are in the order of sub block by sub block
# and each h_ij contains multiple sub blocks, and they are not technically sequential anymore
# so we need to have a way to gather the placement indices
# and redo them

from typing import List
from yasps.attribute import attribute
import os
import ctypes
from yasps.helper import prune_duplicate_functions, timed
import pycuda.gpuarray as gpuarray
from yasps.gradientIndicesKernel import gradientIndicesKernel
import numpy as np
class placementReorderKernel:
  def __init__(self):
    self.__energy: Optional[attribute] = None
    self.__kernel = None
    self.__reordered_lookups = gpuarray.empty(0, dtype=np.uint32) # the reordered lookups
    self.__kernelString: str = ""

  def __generate_kernel_string(self,
    global_jacobian_children_spans: List[int],
    max_num_indices: int, # the maximum number of indices for each instance
    energy: attribute # the actual attribute
  ):
    self.__energy = energy
    span_count = len(global_jacobian_children_spans)
    large_block_count = (span_count * (span_count + 1)) // 2 # this is the number of large blocks in the hessian, which is also the number of small blocks in the gradient
    global_jacobian_children_spans_outer = [0]
    for span in global_jacobian_children_spans:
      global_jacobian_children_spans_outer.append(global_jacobian_children_spans_outer[-1] + span)
    self.__kernelString = f"""
#include <cuda_runtime.h>
#include <vector>
__device__ __constant__ unsigned short int jacobian_block_spans_outer[{span_count + 1}] = {{{', '.join(str(span) for span in global_jacobian_children_spans_outer)}}}; // this is the size of each jacobian block, this will also tell us how to segment the hessian blocks

__device__ inline int upper_tri_index_compute(int i, int j) {{
  return i * {span_count} - (i * (i - 1)) / 2 + (j - i);
}}

__global__ void reorderPlacementIndicesGlobal(
  const unsigned short int* segment_sizes, // the size of each segment in the gradient (also the hessian, it's just cartesian product)
  const short int* local_permutations, // how do i locally compress the hessian and gradient
  const unsigned int* lookups, // the lookup table
  const unsigned int* coordinatesOuter, // this will tell us for each instance, the starting and ending index in the lookup table for putting the hessian blocks into the global hessian data array
  unsigned int* lookupsPermuted, // the permuted lookup table
  const unsigned int numInstances // the total number of instances
){{
  unsigned int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= numInstances){{
    return;
  }}
  const unsigned int start = coordinatesOuter[instance];
  const unsigned int end = coordinatesOuter[instace + 1];
  unsigned short int num_small_blocks_per_large_block[{large_block_count}] = {{0}};
  // we first need to determine, in each large blocks, how many small blocks there are
  unsigned short int current_row = 0;
  unsigned short int current_col = 0;
  unsigned short int large_block_row_index = 0; // which large block are we in (on the row, we just record the index)
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    unsigned short int permutation_i = local_permutations[instance * {max_num_indices} + i];
    unsigned short int block_row_size = segment_sizes[instance * {max_num_indices} + i];
    current_col = current_row; // reset the column to the start of the block
    current_row += block_row_size;
    if (permutation_i == 0){{
      continue;
    }}
    for (large_block_row_index; large_block_row_index < {span_count}; large_block_row_index++){{
      if (jacobian_block_spans_outer[large_block_row_index] < current_row && jacobian_block_spans_outer[large_block_row_index + 1] >= current_row){{
        break;
      }}
    }}
    unsigned short int large_block_col_index = 0; // which large block we are in (on the col, we just record the index)
    for (unsigned int j = i; j < {max_num_indices}; j++){{
      unsigned short int permutation_j = local_permutations[instance * {max_num_indices} + j];
      unsigned short int block_col_size = segment_sizes[instance * {max_num_indices} + j];
      current_col += block_col_size;
      if (permutation_j == 0){{
        continue;
      }}
      for (large_block_col_index; large_block_col_index < {span_count}; large_block_col_index++){{
        if (jacobian_block_spans_outer[large_block_col_index] < current_col && jacobian_block_spans_outer[large_block_col_index + 1] >= current_col){{
          break;
        }}
      }}
      // at this point we have determined which large block we are in, we just need to put it in the correct one
      // because we only care about the upper triangular part, we have to do a bit of computation
      const unsigned short int upper_tri_index = upper_tri_index_compute(large_block_row_index, large_block_col_index);
      num_small_blocks_per_large_block[upper_tri_index] += 1;
    }}
  }}
  // now compute the outer indices
  unsigned short int num_small_blocks_per_large_block_outer[{large_block_count + 1}] = {{0}};
  for (unsigned int i = 0; i < {large_block_count}; i++){{
    num_small_blocks_per_large_block_outer[i + 1] = num_small_blocks_per_large_block_outer[i] + num_small_blocks_per_large_block[i];
  }}

  // ok now we place things back
  unsigned short int large_block_added_count[{large_block_count}] = {{0}}; // this will keep track of how many small blocks have been added to each large block, so that we know where to place the next one

  // we basically need to repeat the logic, but we save some memory by doing this
  current_row = 0;
  current_col = 0;
  large_block_row_index = 0;
  unsigned short int total_blocks = 0;
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    unsigned short int permutation_i = local_permutations[instance * {max_num_indices} + i];
    unsigned short int block_row_size = segment_sizes[instance * {max_num_indices} + i];
    current_col = current_row; // reset the column to the start of the block
    current_row += block_row_size;
    if (permutation_i == 0){{
      continue;
    }}
    for (large_block_row_index; large_block_row_index < {span_count}; large_block_row_index++){{
      if (jacobian_block_spans_outer[large_block_row_index] < current_row && jacobian_block_spans_outer[large_block_row_index + 1] >= current_row){{
        break;
      }}
    }}
    unsigned short int large_block_col_index = 0; // which large block we are in (on the col, we just record the index)
    for (unsigned int j = i; j < {max_num_indices}; j++){{
      unsigned short int permutation_j = local_permutations[instance * {max_num_indices} + j];
      unsigned short int block_col_size = segment_sizes[instance * {max_num_indices} + j];
      current_col += block_col_size;
      if (permutation_j == 0){{
        continue;
      }}
      for (large_block_col_index; large_block_col_index < {span_count}; large_block_col_index++){{
        if (jacobian_block_spans_outer[large_block_col_index] < current_col && jacobian_block_spans_outer[large_block_col_index + 1] >= current_col){{
          break;
        }}
      }}
      // at this point we have determined which large block we are in, we just need to put it in the correct one
      // because we only care about the upper triangular part, we have to do a bit of computation
      const unsigned short int upper_tri_index = upper_tri_index_compute(large_block_row_index, large_block_col_index);
      const unsigned int true_block_placement = num_small_blocks_per_large_block_outer[upper_tri_index] + large_block_added_count[upper_tri_index];
      large_block_added_count[upper_tri_index] += 1;
      lookupsPermuted[start + true_block_placement] = lookups[start + total_blocks];
      total_blocks += 1;
    }}
  }}
  if (total_blocks != (end - start)){{
    printf("Total blocks is %d, but the actual total blocks should be: %d\\n", total_blocks, end - start);
  }}
}}


extern "C"{{
int reorderPlacementIndices(
  const unsigned short int* segment_sizes, // the size of each segment in the gradient (also the hessian, it's just cartesian product)
  const short int* local_permutations, // how do i locally compress the hessian and gradient
  const unsigned int* lookups, // the lookup table
  const unsigned int* coordinatesOuter, // this will tell us for each instance, the starting and ending index in the lookup table for putting the hessian blocks into the global hessian data array
  unsigned int* lookupsPermuted, // the permuted lookup table
  const unsigned int numInstances // the total number of instances
){{
  const unsigned int threadsPerBlock = 256;
  const unsigned int blocks = (numInstances + threadsPerBlock - 1) / threadsPerBlock;
  reorderPlacementIndicesGlobal<<<blocks, threadsPerBlock>>>(
    segment_sizes,
    local_permutations,
    lookups,
    coordinatesOuter,
    lookupsPermuted,
    numInstances
  );
  cudaDeviceSynchronize();
  return 0;
}}


}} // end of extern c

"""
    # self.__generateKernel()

  def __generateKernel(
    self,
    global_jacobian_children_spans: List[int],
    max_num_indices: int, # the maximum number of indices for each instance
    energy: attribute # the actual attribute
  ):
    # ok now we compile the kernel by saving it to a file and then calling nvcc
    file_name = f".yasps_tmp/{self.__energy.fullName}_reorder_placement"
    if os.path.exists(f'{file_name}.so'):
      # we just use that file?
      self.__kernel = ctypes.CDLL(f"{file_name}.so").reorderPlacementIndices # get the compiled kernel
      self.__kernel.restype = ctypes.c_int # set the return type to None
      self.__kernel.argtypes = [
        ctypes.c_void_p, # segment_sizes
        ctypes.c_void_p, # local_permutations
        ctypes.c_void_p, # lookups
        ctypes.c_void_p, # coordinatesOuter
        ctypes.c_void_p, # lookupsPermuted
        ctypes.c_uint32 # numInstances
      ]
      return
    else:
      self.__generate_kernel_string(global_jacobian_children_spans, max_num_indices, energy)
      with open(f"{file_name}.cu", "w") as f:
        f.write(self.__kernelString)
      # now we compile the kernel using nvcc
      os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.cu -o {file_name}.so -O3 -arch=sm_89 -lcudart -lcuda")
      self.__kernel = ctypes.CDLL(f"{file_name}.so").reorderPlacementIndices # get the compiled kernel
      self.__kernel.restype = ctypes.c_int # set the return type to None
      self.__kernel.argtypes = [
        ctypes.c_void_p, # segment_sizes
        ctypes.c_void_p, # local_permutations
        ctypes.c_void_p, # lookups
        ctypes.c_void_p, # coordinatesOuter
        ctypes.c_void_p, # lookupsPermuted
        ctypes.c_uint32 # numInstances
      ]
  def generateKernel(
    self,
    global_jacobian_children_spans: List[int],
    max_num_indices: int, # the maximum number of indices for each instance
    energy: attribute # the actual attribute
  ):
    if self.__kernel is not None:
      return
    self.__generateKernel(global_jacobian_children_spans, max_num_indices, energy)

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @timed("placementReorderKernel.reorderPlacementIndices")
  def reorderPlacementIndices(
    self,
    giKernel: gradientIndicesKernel,
    lookups: gpuarray.GPUArray,
  ):
    if self.__energy.numInstances <= 0:
      return
    assert self.__kernel is not None, "placementReorderKernel.reorderPlacementindices: Kernel has not been compiled yet"
    if self.__energy.numInstances > self.__reordered_lookups.size:
      self.__reordered_lookups = gpuarray.zeros(self.__energy.numInstances, dtype=np.uint32)
    self.__kernel(
      self.__to_void_p(giKernel.outputSizes),
      self.__to_void_p(giKernel.outputPermutations),
      self.__to_void_p(lookups),
      self.__to_void_p(giKernel.outputCompressedCoordinateCountsOuter),
      self.__to_void_p(self.__reordered_lookups),
      ctypes.c_uint32(self.__energy.numInstances)
    )

  @property
  def reordered_lookups(self):
    return self.__reordered_lookups

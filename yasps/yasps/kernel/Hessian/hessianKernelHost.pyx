from typing import List
from yasps.attribute import attribute
from yasps.deviceKernel import deviceKernel
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
class hessianKernelHost:
  def __init__(self, att: attribute, unique_gradient_sizes: List[int], max_child_gradient_size: int, project_entire_hessian: bool):
    self.__att = att
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    self.__kernelString = f'''
#include "allHeaders.cuh"
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

extern "C"
int compute_hessian_and_gradient_with_compression(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  {"".join([f"const unsigned int* {x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
  const unsigned int* segment_indices,            // where to place the gradient for each segment of the local gradient / hessian we generated
  const unsigned short int* segment_sizes,        // how large is each segment of the gradient before compression
  const short int* local_permutations,            // how do i locally compress the hessian and gradient
  const unsigned int* lookups,                    // how to place the current block inside the hessian
  const unsigned int* coordinatesOuter,           // this will tell us for each instance, the starting and ending index in the lookup table for putting the hessian blocks into the global hessian data array
  const unsigned int* groupedIndicesInner, // we need to know which instance will correspond to the current size
  const unsigned int* groupedIndicesOuter, // the outer indices that will indicate for each gradient size, what's the start and end in the inner array
  const unsigned int nth_gradient_size,    // this indicates which position we are in the outer array, let's keep it in the host function just so that we have a 1 to 1 match
  const unsigned int projection_method,
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal,    // the diagonal, we will use it for preconditioning
  double* diagonal_blocks, // store the diagonal blocks, use it for block preconditioning
  const unsigned int* diagonal_blocks_start, // for each attribute, where does the diagonal block start
  const unsigned int* gradient_segments_start, // for each attribute, where does the gradient start
  const short unsigned int* unique_gradient_sizes, // the unique gradient sizes, on cpu
  const unsigned int num_unique_gradient_sizes // the number of unique gradient sizes
  ){{
  if (num_unique_gradient_sizes == 0){{
    return 0; // nothing to do
  }}
  // size_t before, after;
  // cudaDeviceGetLimit(&before, cudaLimitStackSize);
  std::vector<unsigned int> unique_gradient_sizes_instance_count;
  unique_gradient_sizes_instance_count.resize(num_unique_gradient_sizes);
  // copy the outer indices
  std::vector<unsigned int> outer_indices;
  outer_indices.resize(num_unique_gradient_sizes + 1);
  cudaMemcpy(&outer_indices[0], groupedIndicesOuter, sizeof(unsigned int) * (num_unique_gradient_sizes + 1), cudaMemcpyDeviceToHost);

  // get the count
  for (unsigned int i = 0; i < num_unique_gradient_sizes; i++) {{
    unique_gradient_sizes_instance_count[i] = outer_indices[i + 1] - outer_indices[i];
  }}

  std::vector<cudaStream_t> streams;
  streams.resize(num_unique_gradient_sizes);
  for (unsigned int i = 0; i < num_unique_gradient_sizes; i++) {{
    cudaStreamCreate(&streams[i]);
  }}

  // cudaDeviceSynchronize();
  // cudaDeviceSetLimit(cudaLimitStackSize, 128);
  for (unsigned int i = 0; i < num_unique_gradient_sizes; i++) {{
'''
    if project_entire_hessian:
      self.__kernelString += f'''
    switch(unique_gradient_sizes[i]){{
'''
      # now we add the for loop to instantiate the known gradient sizes template functions
      for size in sorted(unique_gradient_sizes):
        if size != 0:
          self.__kernelString += f'''
      case {size}:
        compute_hessian_and_gradient_global_function_final_gradient_size_{size}<<<(unique_gradient_sizes_instance_count[i] + 31) / 32, 32, 0, streams[i]>>>(
          {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
          {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
          {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
          {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
          segment_indices,            // where to place the gradient for each segment of the local gradient / hessian we generated
          segment_sizes,        // how large is each segment of the gradient before compression
          local_permutations,            // how do i locally compress the hessian and gradient
          lookups,                    // how to place the current block inside the hessian
          coordinatesOuter,           // this will tell us for each instance, the starting and ending index in the lookup table for putting the hessian blocks into the global hessian data array
          groupedIndicesInner, // we need to know which instance will correspond to the current size
          groupedIndicesOuter, // the outer indices that will indicate for each gradient size, what's the start and end in the inner array
          i,    // this indicates which position we are in the outer array, let's keep it in the host function just so that we have a 1 to 1 match
          projection_method,
          gradient,   // the gradient output
          hessian_blocks, // the blocks that will constitute the hessian
          diagonal,    // the diagonal, we will use it for preconditioning
          diagonal_blocks, // store the diagonal blocks, use it for block preconditioning
          diagonal_blocks_start, // for each attribute, where does the diagonal block start
          gradient_segments_start // for each attribute, where does the gradient start
        );
        break;
'''
      self.__kernelString += '''
      default:
        printf("Invalid gradient size, %u\\n", unique_gradient_sizes[i]);
        break;
    }
'''
    else:
      # this is the case where we are not projecting the entire Hessian
      # this means the compressed gradient size doesn't matter anymore, what we recorded is the largest block size
        self.__kernelString += f'''
    compute_hessian_and_gradient_global_function_final_gradient_size_{max_child_gradient_size}<<<(unique_gradient_sizes_instance_count[i] + 31) / 32, 32, 0, streams[i]>>>(
      {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
      {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
      {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
      {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
      segment_indices,            // where to place the gradient for each segment of the local gradient / hessian we generated
      segment_sizes,        // how large is each segment of the gradient before compression
      local_permutations,            // how do i locally compress the hessian and gradient
      lookups,                    // how to place the current block inside the hessian
      coordinatesOuter,           // this will tell us for each instance, the starting and ending index in the lookup table for putting the hessian blocks into the global hessian data array
      groupedIndicesInner, // we need to know which instance will correspond to the current size
      groupedIndicesOuter, // the outer indices that will indicate for each gradient size, what's the start and end in the inner array
      i,    // this indicates which position we are in the outer array, let's keep it in the host function just so that we have a 1 to 1 match
      projection_method,
      gradient,   // the gradient output
      hessian_blocks, // the blocks that will constitute the hessian
      diagonal,    // the diagonal, we will use it for preconditioning
      diagonal_blocks, // store the diagonal blocks, use it for block preconditioning
      diagonal_blocks_start, // for each attribute, where does the diagonal block start
      gradient_segments_start // for each attribute, where does the gradient start
    );
'''
    self.__kernelString +='''
  }
  // cudaDeviceSynchronize();
  // cudaDeviceSetLimit(cudaLimitStackSize, 128);
  // close the streams
  for (unsigned int i = 0; i < num_unique_gradient_sizes; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error during kernel execution: %s\\n", cudaGetErrorString(err));
    return -1;  // Return error to Python
  }
  // cudaDeviceGetLimit(&after, cudaLimitStackSize);
  // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\\n");
  // printf("stack size: %zu -> %zu\\n", before, after);
  // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\\n");
  return 0; // success
}
'''

  @property
  def kernelString(self):
    return self.__kernelString

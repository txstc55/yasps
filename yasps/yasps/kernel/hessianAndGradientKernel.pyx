# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
import pycuda.gpuarray as gpuarray
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.primitiveUnion import primitiveUnion
from yasps.helper import prune_duplicate_functions, timed
import os
import ctypes
from typing import List, Set
import hashlib
import subprocess

class hessianAndGradientKernel:
  att_name_to_gradient_sizes: dict[str, Set[int]] = {}  # maps attribute names to their unique gradient sizes, this way we only need to generate unique gradient sizes once
  att_name_to_kernel: dict[str, hessianAndGradientKernel] = {}  # maps attribute names to their hessian and gradient kernel instances, this way we can just return the previous existing kernel


  def __init__(self, att: attribute, project_entire_hessian: bool, projection_method: int = 1, gradeient_only: bool = False):
    self.__kernelString: str = ""
    self.__headerFileString: str = ""
    self.__kernel = None # the kernel for computhing the gradient and hessians
    self.__unique_gradient_sizes: Set[int] = set([]) # this will tell us the unique gradient sizes, we will use this to generate and regenerate kernel when there are new gradient sizes
    self.__project_entire_hessian = project_entire_hessian
    self.__projection_method = projection_method
    self.__gradient_only = gradeient_only
    self.__att = att
    # self.__generateKernel(att)

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))


  def generateKernel(self, unique_gradient_sizes: List[int], wrt: List[attribute]) -> None:
    # check if our unique gradient sizes contains the input gradient sizes
    if set(unique_gradient_sizes).issubset(self.__unique_gradient_sizes):
      return
    self.__unique_gradient_sizes.update(unique_gradient_sizes)
    # print("Unique gradient sizes updated to", unique_gradient_sizes)
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    wrt_names = "_".join([att.fullName for att in wrt])
    size_names = "_".join([str(size) for size in unique_gradient_sizes])
    full_file_name = f"compute_hessian_and_gradient_for_{self.__att.fullName}_wrt_{wrt_names}_with_sizes_{size_names}"
    full_file_name_hashed = int(hashlib.sha256(full_file_name.encode('utf-8')).hexdigest(), 16)
    file_name = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}"
    # print(f"full file name: {full_file_name}\nhashed: {file_name}.cu")
    print(f"hashed: {file_name}.cu")
    if not os.path.exists(f'{file_name}.so'):
      # add the includes and the evd function
      self.__headerFileString += '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

// For small matrix < 4
template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output, int choice) {
  if (choice == 0){
    return;
  }
  const int M = 4;
  // Initialize an M x M matrix with zeros
  Eigen::Matrix<double, M, M> symMtr = Eigen::Matrix<double, M, M>::Identity();

  // Copy the input N x N matrix into the top-left corner of the M x M matrix
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      symMtr(row, col) = A[row * N + col];
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, M, M>> eigenSolver(symMtr);
  const Eigen::Matrix<double, M, M>& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < M; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1.0;
    }
  }

  Eigen::Matrix<double, M, M> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();

  // Copy the top-left N x N submatrix back to A
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      output[row * N + col] = A_reconstructed(row, col);
    }
  }
  return;
}

template <unsigned int N>
__device__ void spd_projection(const double *A, double* output, int choice) {
  if (choice == 0){
    return;
  }
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1.0;
    }
  }

  Eigen::Matrix<double, N, N> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();

  Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> outputMap(output);
  outputMap = A_reconstructed;
  return;
}

template <unsigned int N>
__device__ void spd_projection_inplace(double *A, int choice) {
  if (choice == 0){
    return;
  }
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1.0;
    }
  }

  // Reconstruct the matrix directly without using an intermediate matrix
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += B(i, k) * eigenValues[k] * B(j, k);
      }
      A[i * N + j] = sum;
    }
  }
  return;
}
'''

      # we first generate the header file
      for item in (sortedDependency+ [self.__att.deviceKernel]):
        self.__headerFileString += f'''
extern "C" {{
{item.kernelHeader};
}}'''
        if "scene0_handMesh_bone_level_4_d_scene0_handMesh_bones_matrix_global_current_d_scene0_handMesh_bone_level_1_theta__scene0_handMesh_bone_level_2_theta__scene0_handMesh_bone_level_3_theta__scene0_handMesh_bone_level_4_theta_filled_for_scene0_handMesh_skin_with_1_weights_surface_repulse_energy_0_device_function" in item.kernelHeader:
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
          print("Pay attention to this file")
          print(f".yasps_tmp/{item.attributeName}.cu")
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      for unique_gradient_size in unique_gradient_sizes:
        if unique_gradient_size == 0:
          continue
        self.__headerFileString += f'''
extern "C" {{
__global__ void compute_hessian_and_gradient_global_function_final_gradient_size_{unique_gradient_size}(
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
  const unsigned int nth_gradient_size,    // this indicates which position we are in the outer array
  const unsigned int max_num_indices, // the maximum number of indices for each instance
  const unsigned int projection_method,
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal    // the diagonal, we will use it for preconditioning
);
}}
'''

      with open(".yasps_tmp/allHeaders.cuh", 'w') as f:
        f.write(self.__headerFileString)
        f.close()

      compile_jobs = []
      obj_files = []
      seen_obj_files = set([])
      for item in (sortedDependency + [self.__att.deviceKernel]):
        # we check if the .o file exists
        cu_file = f".yasps_tmp/{item.attributeName}.cu"
        obj_file = f".yasps_tmp/{item.attributeName}.o"
        if not obj_file in seen_obj_files:
          obj_files.append(obj_file)
        if (not os.path.exists(obj_file)) and (not obj_file in seen_obj_files):
          with open(cu_file, 'w') as f:
            f.write(f'''
#include "allHeaders.cuh"
extern "C"{{
{item.kernelString}
}}
''')
            compile_cmd = [
              "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++11", "-O3", "-arch=sm_86",
              "-c", cu_file, "-o", obj_file,
              "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
              "--relocatable-device-code=true"
            ]
            print("Command is")
            print(" ".join(compile_cmd))
            job = subprocess.Popen(compile_cmd)
            compile_jobs.append(job)
        seen_obj_files.add(obj_file)

      # now actually generate the global kernel
      attributeName: str = ""
      if self.__att.name == "":
        attributeName = f'attr_{self.__att.hash}'.replace("-", "_neg_")
      else:
        attributeName = self.__att.fullName
      for unique_gradient_size in unique_gradient_sizes:
        if unique_gradient_size == 0:
          continue
        cu_file = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}_fgs_{unique_gradient_size}.cu"
        obj_file = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}_fgs_{unique_gradient_size}.o"
        obj_files.append(obj_file)
        if not os.path.exists(obj_file):
          with open(cu_file, 'w') as f:
            f.write(f'''
#include "allHeaders.cuh"
extern "C"{{
__global__ void compute_hessian_and_gradient_global_function_final_gradient_size_{unique_gradient_size}(
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
  const unsigned int nth_gradient_size,    // this indicates which position we are in the outer array
  const unsigned int max_num_indices, // the maximum number of indices for each instance
  const unsigned int projection_method,
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal    // the diagonal, we will use it for preconditioning
){{
  const unsigned int N = {unique_gradient_size}; // the size of the gradient and hessian, this is the unique gradient size
  // get the start and end position of the current gradient size
  const unsigned int start = groupedIndicesOuter[nth_gradient_size];
  const unsigned int end = groupedIndicesOuter[nth_gradient_size + 1];
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= end - start){{
    return;
  }}
  index = start + index; // add to begin
  const unsigned int instance = groupedIndicesInner[index]; // this will tell us which instance of the hessian we are computing
// determine if we are computing both the hessian and gradient
#if {int(not self.__gradient_only)} // are we computing both the hessian and gradient
  Eigen::Matrix<double, {self.__att.cols + 1}, {self.__att.cols}{", Eigen::RowMajor" if self.__att.cols > 1 else ""}> hg_mat = Eigen::Matrix<double, {self.__att.cols + 1}, {self.__att.cols}, Eigen::RowMajor>::Zero(); // get the merged gradient and hessian
#else // we are only computing the gradient
  Eigen::Matrix<double, 1, {self.__att.cols}> hg_mat = Eigen::Matrix<double, 1, {self.__att.cols}>::Zero(); // get the gradient
#endif
  // now we call the device function
  {attributeName}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
    instance,
    hg_mat.data()
  );
  // ok we now first put the gradient into the correct place
  unsigned int gradient_offset = 0;
  for (unsigned int i = 0; i < max_num_indices; i++){{
    // we will first get the segment size
    unsigned short int segment_size = segment_sizes[instance * max_num_indices + i];
    // and the position for this segment
    unsigned int segment_placement = segment_indices[instance * max_num_indices + i];
    if (segment_placement == 0){{
      gradient_offset += 1;
      continue; // we encountered space reserved for union, skip
    }}else{{
      segment_placement -= 1; // make it 0 indexed
    }}
    // now we access the gradient and put it into the correct place
    for (unsigned int j = 0; j < segment_size; j++){{
#if {int(not self.__gradient_only)} // did we compute the hessian
      atomicAdd(&gradient[segment_placement + j], hg_mat({self.__att.cols}, gradient_offset + j));
#else
      atomicAdd(&gradient[segment_placement + j], hg_mat(0, gradient_offset + j));
#endif
    }}
    gradient_offset += segment_size;
  }}

  // ok now we have the gradient, we need to compress the hessian locally
  // we have the local_permutations, and we know the size of the matrix
#if {int(not self.__gradient_only)}
  // we will only start this part if we are not just doing gradient
  // first of all, allocate a matrix
  Eigen::Matrix<double, N, N> compressed_hessian = Eigen::Matrix<double, N, N>::Zero();
  unsigned int row_offset = 0;
  for (unsigned int i = 0; i < max_num_indices; i++){{
    unsigned int col_offset = 0;
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = local_permutations[instance * max_num_indices + i];
    if (permutation_i == 0){{
      row_offset += 1; // done with the row since it's reserved for union empty space
      continue; // we encountered space reserved for union, skip
    }}
    if (permutation_i < 0){{
      // this block position exists, we need to get the negative of it
      permutation_i = -permutation_i;
    }}
    permutation_i -= 1; // back to 0 indexed
    unsigned short int segment_size_i = segment_sizes[instance * max_num_indices + i];
    for (unsigned int j = 0; j < max_num_indices; j++){{
      short int permutation_j = local_permutations[instance * max_num_indices + j];
      if (permutation_j == 0){{
        col_offset += 1; // done with the column since it's reserved for union empty space
        continue; // we encountered space reserved for union, skip
      }}
      if (permutation_j < 0){{
        // this block position exists, we need to get the negative of it
        permutation_j = -permutation_j;
      }}
      permutation_j -= 1; // back to 0 indexed
      // ok at this point we know the correct position to put in the compressed hessian
      unsigned short int segment_size_j = segment_sizes[instance * max_num_indices + j];
      for (unsigned int k = 0; k < segment_size_i; k++){{
        for (unsigned int l = 0; l < segment_size_j; l++){{
          // we put the block into the compressed hessian
          compressed_hessian(permutation_i + k, permutation_j + l) += hg_mat(row_offset + k, col_offset + l);
        }}
      }}
      col_offset += segment_size_j;
    }}
    row_offset += segment_size_i;
  }}
  // now we have the compressed hessian
  // we will project it if needed

#if {int(self.__project_entire_hessian)} // do we need to project the hessian here
  // project the hessian
  if (N < 4){{
    spd_projection_small<N>(compressed_hessian.data(), compressed_hessian.data(), projection_method);
  }}else{{
    spd_projection_inplace<N>(compressed_hessian.data(), projection_method);
  }}
#endif // end of projection

  // we will now put the compressed hessian into the global hessian blocks
  // as well as the diagonal blocks
  const unsigned int coordinate_start = coordinatesOuter[instance];
  const unsigned int coordinate_end = coordinatesOuter[instance];
  row_offset = 0;
  unsigned int valid_block_counts = 0;
  for (unsigned int i = 0; i < max_num_indices; i++){{
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = local_permutations[instance * max_num_indices + i]; // get the permuted placement
    if (permutation_i > 0){{
      // make it 0 indexed first
      permutation_i -= 1;
      unsigned short int segment_size_i = segment_sizes[instance * max_num_indices + i];
      unsigned int segment_index_i = segment_indices[instance * max_num_indices + i] - 1;
      // we know exactly the row block, we now check for column block
      for (unsigned int j = i; j < max_num_indices; j++){{
        short int permutation_j = local_permutations[instance * max_num_indices + j]; // get the permuted placement
        if (permutation_j > 0){{
          // ok we have found a valid block
          // first again we make it 0 indexed
          permutation_j -= 1;
          unsigned short int segment_size_j = segment_sizes[instance * max_num_indices + j];
          unsigned int segment_index_j = segment_indices[instance * max_num_indices + j] - 1;
          // we now need to get the index
          unsigned int placement_index = lookups[coordinate_start + valid_block_counts];
          // now we put the block in
          if (segment_index_i < segment_index_j){{
            for (unsigned int k = 0; k < segment_size_i; k++){{
              for (unsigned int l = 0; l < segment_size_j; l++){{
                // this is a block in the upper triangle
                atomicAdd(&hessian_blocks[placement_index + k * segment_size_j + l], compressed_hessian(permutation_i + k, permutation_j + l));
              }}
            }}
          }}else{{
            for (unsigned int k = 0; k < segment_size_j; k++){{
              for (unsigned int l = 0; l < segment_size_i; l++){{
                // put the transpose block in
                atomicAdd(&hessian_blocks[placement_index + k * segment_size_i + l], compressed_hessian(permutation_i + l, permutation_j + k));
              }}
            }}
          }}
          // additionally, if it is a diagonal block, we also need to put the diagonal elements
          if (i == j){{
            // get the placement
            unsigned int segment_index = segment_indices[instance * max_num_indices + i] - 1;
            for (unsigned int k = 0; k < segment_size_i; k++){{
              atomicAdd(&diagonal[segment_index + k], compressed_hessian(permutation_i + k, permutation_j + k));
            }}
          }}
          valid_block_counts++;
        }}
      }}
    }}
  }}
#endif // end of gradient only
}}
}}
''')
            compile_cmd = [
              "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++11", "-O3", "-arch=sm_86",
              "-c", cu_file, "-o", obj_file,
              "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
              "--relocatable-device-code=true"
            ]
            print("Command is")
            print(" ".join(compile_cmd))
            job = subprocess.Popen(compile_cmd)
            compile_jobs.append(job)

      # now we add the c functions that will go over all the unique gradient sizes
      self.__kernelString += f'''
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
void compute_hessian_and_gradient_with_compression(
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
  const unsigned int max_num_indices, // the maximum number of indices for each instance
  const unsigned int projection_method,
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal,    // the diagonal, we will use it for preconditioning
  const short unsigned int* unique_gradient_sizes, // the unique gradient sizes, on cpu
  const unsigned int num_unique_gradient_sizes // the number of unique gradient sizes
){{
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

  for (unsigned int i = 0; i < num_unique_gradient_sizes; i++) {{
    switch(unique_gradient_sizes[i]){{
  '''
      # now we add the for loop to instantiate the known gradient sizes template functions
      for size in self.__unique_gradient_sizes:
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
          max_num_indices, // the maximum number of indices for each instance
          projection_method,
          gradient,   // the gradient output
          hessian_blocks, // the blocks that will constitute the hessian
          diagonal    // the diagonal, we will use it for preconditioning
        );
        {{
        // Step 1: Check for launch failure
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {{
          printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
        }}

        // Step 2: Check for runtime errors (like illegal memory access)
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {{
          printf("CUDA runtime error after kernel: %s\\n", cudaGetErrorString(err));
        }}
        }}

        break;
'''
      self.__kernelString += '''
      default:
        printf("Invalid gradient size, %u\\n", unique_gradient_sizes[i]);
        break;
    }
  }
  // close the streams
  for (unsigned int i = 0; i < num_unique_gradient_sizes; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }
}
'''

      # prune duplicate functions
      self.__kernelString = prune_duplicate_functions(self.__kernelString)
      # generate the code to check
      f = open(f"{file_name}.cu", "w")
      f.write(self.__kernelString)
      f.close()
      # Generate global kernel .o file
      kernel_cu_file = f"{file_name}.cu"
      kernel_obj_file = f"{file_name}.o"
      kernel_compile_cmd = [
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++11", "-O3", "-arch=sm_86",
        "-c", kernel_cu_file, "-o", kernel_obj_file,
        "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
        "--relocatable-device-code=true"
      ]
      print("Kernel compile command: ")
      print(" ".join(kernel_compile_cmd))
      subprocess.run(kernel_compile_cmd, check=True)
      job = subprocess.Popen(kernel_compile_cmd)
      compile_jobs.append(job)
      # Wait for all compilation jobs
      for job in compile_jobs:
        job.wait()


      # obj_files = list(set(obj_files))
      # Device link step: critical for CUDA separable compilation
      device_link_obj = f"{file_name}_device_link.o"
      dlink_cmd = [
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_86",
        *(obj_files + [kernel_obj_file]), "-o", device_link_obj,
        "--relocatable-device-code=true"
      ]
      subprocess.run(dlink_cmd, check=True)
      print("Device link command: ")
      print(" ".join(dlink_cmd))

      # Final shared object linking
      final_link_cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_86",
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", f"{file_name}.so",
        "-lcudart", "-lcuda"
      ]
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)


      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute_hessian_and_gradient_with_compression # get the compiled kernel
      self.__kernel.restype = None # set the return type to None
      self.__kernel.argtypes = [
        # Data arrays
        *(ctypes.c_void_p for _ in sortedDatas),             # const double* for each data array
        *(ctypes.c_void_p for _ in sortedConnectivities),     # const unsigned int* for each connectivity index array
        *(ctypes.c_void_p for _ in (x for x in sortedConnectivities if x.dimension == 0)),  # const unsigned int* for CSR arrays
        *(ctypes.c_void_p for x in sortedPrimitiveUnions),    # const unsigned int* for each primitive union counts array
        # Other inputs
        ctypes.c_void_p,    # segment_indices
        ctypes.c_void_p,    # segment_sizes
        ctypes.c_void_p,    # local_permutations
        ctypes.c_void_p,    # lookups
        ctypes.c_void_p,    # coordinatesOuter
        ctypes.c_void_p,    # groupedIndicesInner
        ctypes.c_void_p,    # groupedIndicesOuter
        # Scalars
        ctypes.c_uint32,      # nth_gradient_size
        ctypes.c_uint32,      # max_num_indices
        ctypes.c_uint32,      # projection_method
        # Outputs
        ctypes.c_void_p,    # gradient
        ctypes.c_void_p,    # hessian_blocks
        ctypes.c_void_p,    # diagonal
        # Other CPU arrays
        ctypes.c_void_p,    # unique_gradient_sizes
        ctypes.c_uint,      # num_unique_gradient_sizes
      ]
    else:
      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute_hessian_and_gradient_with_compression # get the compiled kernel
      self.__kernel.restype = None # set the return type to None
      self.__kernel.argtypes = [
        # Data arrays
        *(ctypes.c_void_p for _ in sortedDatas),             # const double* for each data array
        *(ctypes.c_void_p for _ in sortedConnectivities),     # const unsigned int* for each connectivity index array
        *(ctypes.c_void_p for _ in (x for x in sortedConnectivities if x.dimension == 0)),  # const unsigned int* for CSR arrays
        *(ctypes.c_void_p for x in sortedPrimitiveUnions),    # const unsigned int* for each primitive union counts array
        # Other inputs
        ctypes.c_void_p,    # segment_indices
        ctypes.c_void_p,    # segment_sizes
        ctypes.c_void_p,    # local_permutations
        ctypes.c_void_p,    # lookups
        ctypes.c_void_p,    # coordinatesOuter
        ctypes.c_void_p,    # groupedIndicesInner
        ctypes.c_void_p,    # groupedIndicesOuter
        # Scalars
        ctypes.c_uint32,      # nth_gradient_size
        ctypes.c_uint32,      # max_num_indices
        ctypes.c_uint32,      # projection_method
        # Outputs
        ctypes.c_void_p,    # gradient
        ctypes.c_void_p,    # hessian_blocks
        ctypes.c_void_p,    # diagonal
        # Other CPU arrays
        ctypes.c_void_p,    # unique_gradient_sizes
        ctypes.c_uint,      # num_unique_gradient_sizes
      ]

  @timed("hessianAndGradientKernel.compute")
  def compute(
    self,
    attributeArgs: List[gpuarray.GPUArray],
    giKernel: gradientIndicesKernel,
    lookups: gpuarray.GPUArray,
    gradient: gpuarray.GPUArray,
    hessian_blocks: gpuarray.GPUArray,
    diagonal: gpuarray.GPUArray,
  ):
    # print("Unique gradient sizes cpu before hessian kernel:", giKernel.outputUniqueGradientSizesCPU)
    # print("Num unique gradient sizes cpu before hessian kernel:", giKernel.numUniqueGradientSizesCPU)
    assert self.__kernel is not None
    # ## let's save all the inputs
    # attributes = [x.get() for x in attributeArgs]
    # gikernel_stuffs = [x.get() for x in [
    #   giKernel.outputIndices,
    #   giKernel.outputSizes,
    #   giKernel.outputPermutations,
    #   lookups,
    #   giKernel.outputCompressedCoordinateCountsOuter,
    #   giKernel.outputGroupedIndicesInner,
    #   giKernel.outputGroupedIndicesOuter
    # ]]
    # extra_cint_args = [0, giKernel.maxNumIndicesNeeded, self.__projection_method]
    # outputs = [x.get() for x in [gradient, hessian_blocks, diagonal]]
    # unique_gradient_sizes_cpu = giKernel.outputUniqueGradientSizesCPU
    # num_unique_gradient_sizes_cpu = giKernel.numUniqueGradientSizesCPU

    # # now we save everything in a numpy npz file
    # import numpy as np
    # np.savez("dumped_kernel_data.npz",
    #          attributes=np.array(attributes, dtype=object),
    #          gikernel_stuffs=np.array(gikernel_stuffs, dtype=object),
    #          extra_cint_args=np.array(extra_cint_args),
    #          outputs=np.array(outputs, dtype=object),
    #          unique_gradient_sizes_cpu=np.array(unique_gradient_sizes_cpu),
    #          num_unique_gradient_sizes_cpu=np.array(num_unique_gradient_sizes_cpu))

    self.__kernel(
      *[self.__to_void_p(x) for x in attributeArgs],
      self.__to_void_p(giKernel.outputIndices),
      self.__to_void_p(giKernel.outputSizes),
      self.__to_void_p(giKernel.outputPermutations),
      self.__to_void_p(lookups),
      self.__to_void_p(giKernel.outputCompressedCoordinateCountsOuter),
      self.__to_void_p(giKernel.outputGroupedIndicesInner),
      self.__to_void_p(giKernel.outputGroupedIndicesOuter),
      ctypes.c_uint32(0),
      ctypes.c_uint32(giKernel.maxNumIndicesNeeded),
      ctypes.c_uint32(self.__projection_method),
      self.__to_void_p(gradient),
      self.__to_void_p(hessian_blocks),
      self.__to_void_p(diagonal),
      giKernel.outputUniqueGradientSizesCPU.ctypes.data_as(ctypes.c_void_p),
      ctypes.c_uint32(giKernel.numUniqueGradientSizesCPU)
    )

  @property
  def kernelString(self) -> str:
    return self.__kernelString

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
from yasps.context import context
import numpy as np

class hessianAndGradientKernel:
  att_name_to_gradient_sizes: dict[str, Set[int]] = {}  # maps attribute names to their unique gradient sizes, this way we only need to generate unique gradient sizes once
  att_name_to_kernel: dict[str, hessianAndGradientKernel] = {}  # maps attribute names to their hessian and gradient kernel instances, this way we can just return the previous existing kernel


  def __init__(self, att: attribute, project_entire_hessian: bool, projection_method: int = 1, gradeient_only: bool = False, clear_separation: bool = True, jacobian_rows = 0, jacobian_cols = 0, hessian_row_size = 0, dynamic_term = False):
    self.__kernelString: str = ""
    self.__headerFileString: str = ""
    self.__kernel = None # the kernel for computhing the gradient and hessians
    self.__unique_gradient_sizes: Set[int] = set([]) # this will tell us the unique gradient sizes, we will use this to generate and regenerate kernel when there are new gradient sizes
    self.__project_entire_hessian = project_entire_hessian
    self.__projection_method = projection_method
    self.__gradient_only = gradeient_only
    self.__att = att
    self.__clear_separation = clear_separation
    self.__jacobian_rows = jacobian_rows
    self.__jacobian_cols = jacobian_cols
    self.__hessian_row_size = hessian_row_size
    self.__additional_compile_flags = []  # --ptxas-options=-v,-warn-spills,-warn-lmem-usage  use this for memory checking
    self.__dynamic_terms = dynamic_term
    self.__context = context()
    # self.__generateKernel(att)

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @timed("hessianAndGradientKernel.generateKernel")
  def generateKernel(self, unique_gradient_sizes: List[int], max_child_gradient_size: int, wrt: List[attribute], max_num_indices: int) -> None:
    # check if our unique gradient sizes contains the input gradient sizes
    # print("unique gradient sizes are", unique_gradient_sizes)
    if (set(unique_gradient_sizes).issubset(self.__unique_gradient_sizes) and self.__project_entire_hessian) or (max_child_gradient_size in self.__unique_gradient_sizes and not self.__project_entire_hessian):
      return
    if self.__project_entire_hessian:
      print("Unique gradient sizes before is", unique_gradient_sizes)
      self.__unique_gradient_sizes.update(unique_gradient_sizes)
      print("Unique gradient sizes after is", self.__unique_gradient_sizes)
    else:
      print("Max child gradient size before is", max_child_gradient_size)
      self.__unique_gradient_sizes.add(max_child_gradient_size)
      print("Unique gradient sizes after is", self.__unique_gradient_sizes)
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    wrt_names = "_".join([att.fullName for att in wrt])
    size_names = "_".join([str(size) for size in unique_gradient_sizes])
    full_file_name = f"compute_hessian_and_gradient_for_{self.__att.fullName}_wrt_{wrt_names}_with_sizes_{size_names}"
    full_file_name_hashed = int(hashlib.sha256(full_file_name.encode('utf-8')).hexdigest(), 16)
    file_name = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}" + ("" if self.__project_entire_hessian else "_no_proj")
    # print(f"full file name: {full_file_name}\nhashed: {file_name}.cu")
    print(f"hashed: {file_name}.cu")
    if not os.path.exists(f'{file_name}.so'):
      # add the includes and the evd function
      self.__headerFileString = ""
      self.__headerFileString += '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>
#ifndef EIGEN_PROJECTION
#define EIGEN_PROJECTION
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
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
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
    for (unsigned int i = 0; i < N * N; i++){
      output[i] = A[i];
    }
    return;
  }
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
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
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
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
#endif // EIGEN_PROJECTION
'''

      # we first generate the header file
      for item in (sortedDependency+ [self.__att.deviceKernel]):
        self.__headerFileString += f'''
extern "C" {{
{item.kernelHeader};
}}'''
      for unique_gradient_size in self.__unique_gradient_sizes:
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
  const unsigned int projection_method,
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal,    // the diagonal, we will use it for preconditioning
  double* diagonal_blocks, // store the diagonal blocks, use it for block preconditioning
  const unsigned int* diagonal_blocks_start, // for each attribute, where does the diagonal block start
  const unsigned int* gradient_segments_start // for each attribute, where does the gradient start
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
            f.close()
          compile_cmd = [
            "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
            "-O3",
            "-c", cu_file, "-o", obj_file,
            "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
            "--relocatable-device-code=true",
          ] + self.__additional_compile_flags
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
      for unique_gradient_size in self.__unique_gradient_sizes:
        if unique_gradient_size == 0:
          continue
        cu_file = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}_fgs_{unique_gradient_size}" + (".cu" if self.__project_entire_hessian else "_no_proj.cu")
        obj_file = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}_fgs_{unique_gradient_size}" + (".o" if self.__project_entire_hessian else "_no_proj.o")
        obj_files.append(obj_file)
        # if not os.path.exists(obj_file):
        if True: # always regenerate the kernel because header has been replaced
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
  const unsigned int projection_method,
  double* gradient,   // the gradient output
  double* hessian_blocks, // the blocks that will constitute the hessian
  double* diagonal,    // the diagonal, we will use it for preconditioning
  double* diagonal_blocks, // store the diagonal blocks, use it for block preconditioning
  const unsigned int* diagonal_blocks_start, // for each attribute, where does the diagonal block start
  const unsigned int* gradient_segments_start // for each attribute, where does the gradient start
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
  Eigen::Matrix<double, {self.__att.rows}, {self.__att.cols}{", Eigen::RowMajor" if self.__att.cols > 1 else ""}> hg_mat = Eigen::Matrix<double, {self.__att.rows}, {self.__att.cols}, Eigen::RowMajor>::Zero(); // get the merged gradient and hessian
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
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    // we will first get the segment size
    unsigned short int segment_size = segment_sizes[instance * {max_num_indices} + i];
    // and the position for this segment
    unsigned int segment_placement = segment_indices[instance * {max_num_indices} + i];
    if (segment_placement == 0){{
      gradient_offset += segment_size;
      continue; // we encountered space reserved for union, skip
    }}else if (segment_placement == 1){{
      // this is a special case where we want the variable to be in the matrix, but not in the final hessian
      // we keep it because it's necessary for the hessian projection
      gradient_offset += segment_size; // skip this segment
      continue; // skip
    }}
    segment_placement -= 2; // make it 0 indexed
    // now we access the gradient and put it into the correct place
    for (unsigned int j = 0; j < segment_size; j++){{
#if {int(not self.__gradient_only)} // did we compute the hessian
      atomicAdd(&gradient[segment_placement + j], hg_mat({self.__att.rows - 1}, gradient_offset + j));
#else
      atomicAdd(&gradient[segment_placement + j], hg_mat(0, gradient_offset + j));
#endif
    }}
    gradient_offset += segment_size;
  }}
  // Now check if we are also computing the Hessian
#if {int(not self.__gradient_only)}
#if {int(self.__project_entire_hessian)} // ok now check if we need to project the entire Hessian matrix
  Eigen::Matrix<double, N, N> compressed_hessian = Eigen::Matrix<double, N, N>::Zero(); // first we allocate the matrix
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    unsigned int col_offset = 0;
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = local_permutations[instance * {max_num_indices} + i];
    if (permutation_i == 0){{
      row_offset += segment_sizes[instance * {max_num_indices} + i]; // done with the row since it's reserved for union empty space
      continue; // we encountered space reserved for union, skip
    }}
    if (permutation_i < 0){{
      // this block position exists, we need to get the negative of it
      permutation_i = -permutation_i;
    }}
    permutation_i -= 1; // back to 0 indexed
    unsigned short int segment_size_i = segment_sizes[instance * {max_num_indices} + i];
    for (unsigned int j = 0; j < {max_num_indices}; j++){{
      short int permutation_j = local_permutations[instance * {max_num_indices} + j];
      if (permutation_j == 0){{
        col_offset += segment_sizes[instance * {max_num_indices} + j]; // done with the column since it's reserved for union empty space
        continue; // we encountered space reserved for union, skip
      }}
      if (permutation_j < 0){{
        // this block position exists, we need to get the negative of it
        permutation_j = -permutation_j;
      }}
      permutation_j -= 1; // back to 0 indexed
      // ok at this point we know the correct position to put in the compressed hessian
      unsigned short int segment_size_j = segment_sizes[instance * {max_num_indices} + j];
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
  // project the hessian
  if (N < 4){{
    spd_projection_small<N>(compressed_hessian.data(), compressed_hessian.data(), projection_method);
  }}else{{
    spd_projection_inplace<N>(compressed_hessian.data(), projection_method);
  }}


  // we will now put the compressed hessian into the global hessian blocks
  // as well as the diagonal blocks
  const unsigned int coordinate_start = coordinatesOuter[instance];
  const unsigned int coordinate_end = coordinatesOuter[instance];
  row_offset = 0;
  unsigned int valid_block_counts = 0;
  for (unsigned int i = 0; i < {max_num_indices}; i++){{
    // we first determine what's the correct position to put in the compressed hessian
    short int permutation_i = local_permutations[instance * {max_num_indices} + i]; // get the permuted placement
    unsigned int segment_index_i = segment_indices[instance * {max_num_indices} + i];
    if (permutation_i > 0 && segment_index_i >= 2){{
      // make it 0 indexed first
      permutation_i -= 1;
      unsigned short int segment_size_i = segment_sizes[instance * {max_num_indices} + i];
      segment_index_i -= 2;
      // we know exactly the row block, we now check for column block
      for (unsigned int j = i; j < {max_num_indices}; j++){{
        short int permutation_j = local_permutations[instance * {max_num_indices} + j]; // get the permuted placement
        unsigned int segment_index_j = segment_indices[instance * {max_num_indices} + j];
        if (permutation_j > 0 && segment_index_j >= 2){{
          // ok we have found a valid block
          // first again we make it 0 indexed
          permutation_j -= 1;
          unsigned short int segment_size_j = segment_sizes[instance * {max_num_indices} + j];
          segment_index_j -= 2;
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
            unsigned int segment_index = segment_indices[instance * {max_num_indices} + i] - 2;
            for (unsigned int k = 0; k < segment_size_i; k++){{
              atomicAdd(&diagonal[segment_index + k], compressed_hessian(permutation_i + k, permutation_j + k));
            }}
            // now we do the block diagonal placement
            // we first need to determine where to put it in the global diagonal blocks array
            int which_attribute = 0;
            for (int k = 0; k < 123456; k++){{ // for now lets just make 123456 our default, need to change it later
              if (segment_index < gradient_segments_start[k + 1]){{
                break;
              }}
              which_attribute += 1;
            }}
            // now determine which instance in that attribute
            const unsigned int diagonal_block_start = diagonal_blocks_start[which_attribute];
            const unsigned int diff = segment_index - gradient_segments_start[which_attribute];
            const unsigned int which_instance = diff / (segment_size_i);
            const unsigned int diagonal_block_placement = diagonal_block_start + which_instance * segment_size_i * segment_size_i;
            // now we put the diagonal block
            for (unsigned int k = 0; k < segment_size_i; k++){{
              for (unsigned int l = 0; l < segment_size_i; l++){{
              atomicAdd(&diagonal_blocks[diagonal_block_placement + k * segment_size_i + l], compressed_hessian(permutation_i + k, permutation_j + l));
              }}
            }}
          }}
          valid_block_counts++;
        }}
      }}
    }}
  }}


#else // we are not projecting the entire Hessian matrix, there's room for optimization now

  // first we allocate a new array, which computes that for each index
  short int unique_segment_placements[{max_num_indices}] = {{0}}; // this will count how many unique positions we can put the segment, and this is 0 based index
  unsigned short int unique_segment_placements_counts[{max_num_indices}] = {{0}}; // this will count how many segments are placed in each unique position, this is used for the compression
  short int inverse_map[{max_num_indices}] = {{0}}; // this will map the original segment index to the unique position index, this is also 0 based index
  short int unique_segment_sizes[{max_num_indices}] = {{0}}; // this will store the size of the segment for each unique position, this is used for the compression
  unsigned short int unique_segment_placement_first_i[{max_num_indices}] = {{0}}; // this will store, the first i for each unique segment placement


  unsigned short int current_unique_position_index = 0;
  short int last_placement = -1;
  for (unsigned short int i = 0; i < {max_num_indices}; i++){{
    short int permutation_i = local_permutations[instance * {max_num_indices} + i]; // get the permuted placement
    if (permutation_i == 0){{
      continue; // we encountered space reserved for union, skip
    }}
    if (permutation_i < 0) {{
      permutation_i = -permutation_i;
    }}
    permutation_i -= 1; // make it 0 indexed
    if (permutation_i > last_placement){{
      // this means its a new placement, record it
      unique_segment_placements[current_unique_position_index] = permutation_i;
      unique_segment_placements_counts[current_unique_position_index] += 1;
      unique_segment_sizes[current_unique_position_index] = segment_sizes[instance * {max_num_indices} + i];
      unique_segment_placement_first_i[current_unique_position_index] = i;
      inverse_map[i] = current_unique_position_index;
      current_unique_position_index += 1;
      last_placement = permutation_i;

    }}else{{
      // increment the count
      for (unsigned short int j = 0; j < current_unique_position_index; j++){{
        if (unique_segment_placements[j] == permutation_i){{
          unique_segment_placements_counts[j] += 1;
          inverse_map[i] = j;
          break;
        }}
      }}
    }}
  }}

  // now we construct the outer index array, which marks the range
  short int unique_segment_placements_outer[{max_num_indices} + 1] = {{0}};
  for (unsigned short int i = 0; i < current_unique_position_index; i++){{
    unique_segment_placements_outer[i + 1] = unique_segment_placements_outer[i] + unique_segment_placements_counts[i];
  }}
  // now we actually construct the inner array, which marks for each placement, where in the matrix column can we find the start of the block
  short int unique_segment_placements_inner[{max_num_indices}] = {{0}};
  for (unsigned short int i = 0; i < {max_num_indices}; i++){{
    unique_segment_placements_counts[i] = 0;  // reuse this array for local index counting
  }}



  unsigned int column_offset = 0;
  for (unsigned short int i = 0; i < {max_num_indices}; i++){{
    // first check if it is a unique permutation
    short int permutation_i = local_permutations[instance * {max_num_indices} + i];
    if (permutation_i == 0){{
      column_offset += segment_sizes[instance * {max_num_indices} + i];
      continue; // we encountered space reserved for union, skip
    }}
    // now we get the outer index
    const unsigned short int gid = inverse_map[i]; // find out which unique group it belongs to
    const unsigned short int start = unique_segment_placements_outer[gid]; // get the start of the group
    const unsigned short int local_count = unique_segment_placements_counts[gid]; // get how many segments have been placed in this group so far

    unique_segment_placements_inner[start + local_count] = column_offset; // record the column offset for this segment
    unique_segment_placements_counts[gid] = local_count + 1; // increment the count for this group
    column_offset += segment_sizes[instance * {max_num_indices} + i]; // increment the column offset
  }}


  // now we know for each placement index, the range of columns (and also rows) that corresponds to the same placement
  // we can start the accumulation and placement
  unsigned short int valid_block_counts = 0;
  const unsigned int coordinate_start = coordinatesOuter[instance];
  for (unsigned short int i = 0; i < current_unique_position_index; i++){{
    // this is one group
    // let's get the group's segment length
    const unsigned short int segment_size_i = unique_segment_sizes[i]; // what is the block row size
    unsigned int segment_index_i = segment_indices[instance * {max_num_indices} + unique_segment_placement_first_i[i]]; // what is its position in the global atrix
    if (segment_index_i < 2){{
      continue; // this is reserved for attributes that participate in the differentiation, but not in the final hessian, we skip it
    }}
    unsigned short int group_count_1 = unique_segment_placements_counts[i]; // how many rows in this block
    // let's also get the placement index
    for (unsigned short int j = i; j < current_unique_position_index; j++){{
      const unsigned short int segment_size_j = unique_segment_sizes[j]; // what is the block column size
      unsigned int segment_index_j = segment_indices[instance * {max_num_indices} + unique_segment_placement_first_i[j]]; // what is its position in the global matrix
      if (segment_index_j < 2){{
        continue; // this is reserved for attributes that participate in the differentiation, but not in the final hessian, we skip it
      }}
      unsigned short int group_count_2 = unique_segment_placements_counts[j]; // how many columns in this block
      // now we will start to accumulate the block
      unsigned int placement_index = lookups[coordinate_start + valid_block_counts];
      unsigned int diagonal_block_placement = 0; // initialize the diagonal block placement
      if (i == j){{
        const unsigned int segment_index = segment_index_i - 2;
        // now we do the block diagonal placement
        // we first need to determine where to put it in the global diagonal blocks array
        int which_attribute = 0;
        for (int k = 0; k < 123456; k++){{ // for now lets just make 123456 our default, need to change it later
          if (segment_index < gradient_segments_start[k + 1]){{
            break;
          }}
          which_attribute += 1;
        }}
        // now determine which instance in that attribute
        const unsigned int diagonal_block_start = diagonal_blocks_start[which_attribute];
        const unsigned int diff = segment_index - gradient_segments_start[which_attribute];
        const unsigned int which_instance = diff / (segment_size_i);
        diagonal_block_placement = diagonal_block_start + which_instance * segment_size_i * segment_size_i;
      }}

      if (segment_index_i < segment_index_j) {{
        for (unsigned int k = 0; k < segment_size_i; k++) {{
          for (unsigned int l = 0; l < segment_size_j; l++) {{
            double acc = 0.0;

            for (unsigned short int g1_index = 0; g1_index < group_count_1; g1_index++) {{
              const unsigned int column_offset_i =
                unique_segment_placements_inner[unique_segment_placements_outer[i] + g1_index];

              for (unsigned short int g2_index = 0; g2_index < group_count_2; g2_index++) {{
                const unsigned int column_offset_j =
                  unique_segment_placements_inner[unique_segment_placements_outer[j] + g2_index];

                acc += hg_mat(column_offset_i + k, column_offset_j + l);
              }}
            }}

            atomicAdd(
              &hessian_blocks[placement_index + k * segment_size_j + l],
              acc
            );
          }}
        }}
      }}else {{
        for (unsigned int k = 0; k < segment_size_j; k++) {{
          for (unsigned int l = 0; l < segment_size_i; l++) {{
            double acc = 0.0;

            for (unsigned short int g1_index = 0; g1_index < group_count_1; g1_index++) {{
              const unsigned int column_offset_i =
                unique_segment_placements_inner[unique_segment_placements_outer[i] + g1_index];

              for (unsigned short int g2_index = 0; g2_index < group_count_2; g2_index++) {{
                const unsigned int column_offset_j =
                  unique_segment_placements_inner[unique_segment_placements_outer[j] + g2_index];

                // This computes block[l, k], matching your transpose placement.
                acc += hg_mat(column_offset_i + l, column_offset_j + k);
              }}
            }}
            atomicAdd(
              &hessian_blocks[placement_index + k * segment_size_i + l],
              acc
            );
            if (i == j) {{
              const unsigned int segment_index = segment_index_i - 2;
              atomicAdd(
                &diagonal_blocks[diagonal_block_placement + k * segment_size_i + l],
                acc
              );
            }}
          }}
        }}
      }}
      valid_block_counts++; // finally increment
    }}
  }}

#endif // end of checking if we are projecting the entire Hessian
#endif // end of checking if we are doing gradient only
}}
}}
''')
            f.close()
          compile_cmd = [
            "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
            "-O3",
            "-c", cu_file, "-o", obj_file,
            "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
            "--relocatable-device-code=true",
          ] + self.__additional_compile_flags
          print("Command is")
          print(" ".join(compile_cmd))
          job = subprocess.Popen(compile_cmd)
          compile_jobs.append(job)

      # now we add the c functions that will go over all the unique gradient sizes
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
      if self.__project_entire_hessian:
        self.__kernelString += f'''
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
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
        "-O3",
        "-c", kernel_cu_file, "-o", kernel_obj_file,
        "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
        "--relocatable-device-code=true",
      ] + self.__additional_compile_flags
      print("Kernel compile command: ")
      print(" ".join(kernel_compile_cmd))
      job = subprocess.Popen(kernel_compile_cmd)
      compile_jobs.append(job)
      # Wait for all compilation jobs
      for job in compile_jobs:
        job.wait()


      # obj_files = list(set(obj_files))
      # Device link step: critical for CUDA separable compilation
      device_link_obj = f"{file_name}_device_link.o"
      dlink_cmd = [
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3",
        *(obj_files + [kernel_obj_file]), "-o", device_link_obj,
        "--relocatable-device-code=true",
      ] + self.__additional_compile_flags
      print("Device link command: ")
      print(" ".join(dlink_cmd))
      subprocess.run(dlink_cmd, check=True)
      # Final shared object linking
      final_link_cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3",
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", f"{file_name}.so",
        "-lcudart", "-lcuda",
      ] + self.__additional_compile_flags
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)


      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute_hessian_and_gradient_with_compression # get the compiled kernel
      self.__kernel.restype = ctypes.c_int # set the return type to None
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
        ctypes.c_uint32,      # projection_method
        # Outputs
        ctypes.c_void_p,    # gradient
        ctypes.c_void_p,    # hessian_blocks
        ctypes.c_void_p,    # diagonal
        ctypes.c_void_p,    # diagonal blocks
        ctypes.c_void_p,    # diagonal blocks start
        ctypes.c_void_p,    # gradient segments start
        # Other CPU arrays
        ctypes.c_void_p,    # unique_gradient_sizes
        ctypes.c_uint,      # num_unique_gradient_sizes
      ]
    else:
      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute_hessian_and_gradient_with_compression # get the compiled kernel
      self.__kernel.restype = ctypes.c_int # set the return type to None
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
        ctypes.c_uint32,      # projection_method
        # Outputs
        ctypes.c_void_p,    # gradient
        ctypes.c_void_p,    # hessian_blocks
        ctypes.c_void_p,    # diagonal
        ctypes.c_void_p,    # diagonal blocks
        ctypes.c_void_p,    # diagonal blocks start
        ctypes.c_void_p,    # gradient segments start
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
    diagonal_blocks: gpuarray.GPUArray,
    diagonal_blocks_start: gpuarray.GPUArray,
    gradient_segments_start: gpuarray.GPUArray
  ):
    # print("Unique gradient sizes cpu before hessian kernel:", giKernel.outputUniqueGradientSizesCPU)
    # print("Num unique gradient sizes cpu before hessian kernel:", giKernel.numUniqueGradientSizesCPU)
    if giKernel.numUniqueGradientSizesCPU == 0:
      # there is nothing to compute
      return
    assert self.__kernel is not None
    # self.__context.useNamedContext("dynamic_hessian" if self.__dynamic_terms else "static_hessian")
    self.__context.useDefaultContext()
    # self.__context.useNamedContext("hessian")
    error_code = self.__kernel(
      *[self.__to_void_p(x) for x in attributeArgs],
      self.__to_void_p(giKernel.outputIndices),
      self.__to_void_p(giKernel.outputSizes),
      self.__to_void_p(giKernel.outputPermutations),
      self.__to_void_p(lookups),
      self.__to_void_p(giKernel.outputCompressedCoordinateCountsOuter),
      self.__to_void_p(giKernel.outputGroupedIndicesInner),
      self.__to_void_p(giKernel.outputGroupedIndicesOuter),
      ctypes.c_uint32(0),
      ctypes.c_uint32(self.__projection_method),
      self.__to_void_p(gradient),
      self.__to_void_p(hessian_blocks),
      self.__to_void_p(diagonal),
      self.__to_void_p(diagonal_blocks),
      self.__to_void_p(diagonal_blocks_start),
      self.__to_void_p(gradient_segments_start),
      giKernel.outputUniqueGradientSizesCPU.ctypes.data_as(ctypes.c_void_p),
      ctypes.c_uint32(giKernel.numUniqueGradientSizesCPU)
    )
    if error_code != 0:
      raise RuntimeError(f"HessianAndGradientKernel.compute: Kernel execution failed with error code {error_code}")

  @property
  def kernelString(self) -> str:
    return self.__kernelString

# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from typing import List
from yasps.helper import prune_duplicate_functions
import os
import ctypes
from yasps.helper import timed
import pycuda.gpuarray as gpuarray
from yasps.primitiveUnion import primitiveUnion
import subprocess
from yasps.context import context

class globalKernel:
  @timed("globalKernel.__init__")
  def __init__(self, att: attribute):
    self.__kernelString: str = ""
    self.__headerFileString: str = ""
    self.__att = att
    self.__kernel = None
    self.__additional_compile_flags = []  # --ptxas-options=-v,-warn-spills,-warn-lmem-usage  use this for memory checking
    self.__generateKernel()
    self.__context = context()

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @timed("globalKernel.__generateKernel")
  def __generateKernel(self) -> None:
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    file_name = f".yasps_tmp/compute_{self.__att.fullNameWithHash}"
    if not os.path.exists(f'{file_name}.so'):
      print(f"File {file_name}.so does not exist, compiling")
      self.__headerFileString += '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// Fast FP64 LDLT test for a local Hessian that is already positive
// semidefinite. It avoids the eigendecomposition without modifying A.
template <unsigned int N>
__device__ __forceinline__ bool is_positive_semidefinite(const double *A) {
  double lower[N * N] = {};
  double diagonal[N] = {};
  double scale = 1.0;
  for (unsigned int i = 0; i < N; ++i) {
    const double magnitude = A[i * N + i] < 0.0 ? -A[i * N + i] : A[i * N + i];
    scale = scale > magnitude ? scale : magnitude;
    lower[i * N + i] = 1.0;
  }
  const double tolerance = scale * 1.0e-10;
  for (unsigned int column = 0; column < N; ++column) {
    double pivot = A[column * N + column];
    for (unsigned int k = 0; k < column; ++k) {
      const double value = lower[column * N + k];
      pivot -= value * value * diagonal[k];
    }
    if (pivot < -tolerance) return false;
    const double pivot_magnitude = pivot < 0.0 ? -pivot : pivot;
    if (pivot_magnitude <= tolerance) {
      diagonal[column] = 0.0;
      for (unsigned int row = column + 1; row < N; ++row) {
        double residual = A[row * N + column];
        for (unsigned int k = 0; k < column; ++k) {
          residual -= lower[row * N + k] * lower[column * N + k] * diagonal[k];
        }
        const double residual_magnitude = residual < 0.0 ? -residual : residual;
        if (residual_magnitude > tolerance) return false;
      }
      continue;
    }
    diagonal[column] = pivot;
    for (unsigned int row = column + 1; row < N; ++row) {
      double value = A[row * N + column];
      for (unsigned int k = 0; k < column; ++k) {
        value -= lower[row * N + k] * lower[column * N + k] * diagonal[k];
      }
      lower[row * N + column] = value / pivot;
    }
  }
  return true;
}
// For small matrix < 4
template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output, int choice) {
  if (choice == 0){
    for (int i = 0; i < N * N; i++) {
      output[i] = A[i];
    }
    return;
  }
  if (N == 1){
    output[0] = choice == 1 ? abs(A[0]) : (A[0] < 1e-6 ? 1e-6: A[0]);
    return;
  }
  if (is_positive_semidefinite<N>(A)) {
    for (unsigned int i = 0; i < N * N; ++i) output[i] = A[i];
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
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1e-6;
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
    for (int i = 0; i < N * N; i++) {
      output[i] = A[i];
    }
    return;
  }
  if (is_positive_semidefinite<N>(A)) {
    for (unsigned int i = 0; i < N * N; ++i) output[i] = A[i];
    return;
  }

  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1e-6;
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
  if (is_positive_semidefinite<N>(A)) return;
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
'''
      # we first generate the header file
      for item in (sortedDependency+ [self.__att.deviceKernel]):
        self.__headerFileString += f'''
extern "C" {{
{item.kernelHeader};
}}'''
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
            "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-arch=sm_89",
            "-c", cu_file, "-o", obj_file,
            "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
          ] + self.__additional_compile_flags
          print("Command is")
          print(" ".join(compile_cmd))
          job = subprocess.Popen(compile_cmd)
          compile_jobs.append(job)
        seen_obj_files.add(obj_file)

      # now actually generate the global kernel
      attributeName: str = ""
      if self.__att.name == "":
        attributeName = self.__att.fullName.replace("-", "_neg_")
      else:
        attributeName = self.__att.fullName.replace("-", "_neg_")

      kernelRawName = f'''
__global__ void {attributeName}_global_function({
  "".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
  double* result,
  unsigned int MAX_INDEX
)'''
      self.__kernelString += '''
#include "allHeaders.cuh"
'''
      self.__kernelString += f'''
extern "C" {{
{kernelRawName}{{
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  // now we call the device function
  {attributeName}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f'{x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
    index,
    result + index * {self.__att.size}
  );
}}
}}
'''
      self.__kernelString += f'''
extern "C"
int compute(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
  double* result,
  unsigned int MAX_INDEX
){{
  // cudaDeviceSynchronize();
  // cudaDeviceSetLimit(cudaLimitStackSize, 128);
  {attributeName}_global_function<<<(MAX_INDEX + 31) / 32, 32>>>(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
    result,
    MAX_INDEX
  );
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {{
    fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
    return -1;
  }}
  return 0;
}}
'''
      self.__kernelString = prune_duplicate_functions(self.__kernelString)
      f = open(f"{file_name}.cu", 'w')
      f.write(self.__kernelString)
      f.close()

      # Generate global kernel .o file
      kernel_cu_file = f"{file_name}.cu"
      kernel_obj_file = f"{file_name}.o"
      kernel_compile_cmd = [
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-arch=sm_89",
        "-c", kernel_cu_file, "-o", kernel_obj_file,
        "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
      ] + self.__additional_compile_flags
      print("Kernel compile command: ")
      print(" ".join(kernel_compile_cmd))
      job = subprocess.Popen(kernel_compile_cmd)
      compile_jobs.append(job)
      # Wait for all compilation jobs
      for job in compile_jobs:
        job.wait()


      obj_files = list(set(obj_files))
      # Device link step: critical for CUDA separable compilation
      device_link_obj = f"{file_name}_device_link.o"
      dlink_cmd = [
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89",
        *(obj_files + [kernel_obj_file]), "-o", device_link_obj,
      ] + self.__additional_compile_flags
      subprocess.run(dlink_cmd, check=True)
      print("Device link command: ")
      print(" ".join(dlink_cmd))

      # Final shared object linking
      final_link_cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89",
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", f"{file_name}.so",
        "-lcudart", "-lcuda",
      ] + self.__additional_compile_flags
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)

      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute
      self.__kernel.argtypes = [
        *[ctypes.c_void_p for _ in sortedDatas],
        *[ctypes.c_void_p for _ in sortedConnectivities],
        *[ctypes.c_void_p for x in sortedConnectivities if x.dimension == 0],
        *[ctypes.c_void_p for x in sortedPrimitiveUnions],
        ctypes.c_void_p,  # result
        ctypes.c_uint  # MAX_INDEX
      ]
      self.__kernel.restype = ctypes.c_int
    else:
      print(f"File {file_name}.so does exists, linking")
      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute
      self.__kernel.argtypes = [
        *[ctypes.c_void_p for _ in sortedDatas],
        *[ctypes.c_void_p for _ in sortedConnectivities],
        *[ctypes.c_void_p for x in sortedConnectivities if x.dimension == 0],
        *[ctypes.c_void_p for x in sortedPrimitiveUnions],
        ctypes.c_void_p,  # result
        ctypes.c_uint  # MAX_INDEX
      ]
      self.__kernel.restype = ctypes.c_int

  @timed("globalKernel.compute")
  def compute(self, output):
    assert self.__kernel is not None
    if self.__att.correspondance.numInstances == 0:
      return # there is nothing to compute
    counts_gpu = [x.children_primitive_counts_gpu for x in self.__att.deviceKernel.kernelPrimitiveUnions]
    args = [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelDatas]
    args += [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelConnectivity]
    args += [self.__to_void_p(x.compressedRows) for x in self.__att.deviceKernel.kernelConnectivity if x.dimension == 0]
    args += [self.__to_void_p(x) for x in counts_gpu]
    args += [self.__to_void_p(output)]
    args += [ctypes.c_uint32(self.__att.correspondance.numInstances)]
    self.__context.useDefaultContext()
    error_code = self.__kernel(*args)
    if error_code != 0:
      raise RuntimeError(f"globalKernel.compute: Kernel execution failed with error code {error_code}")





  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self):
    return self.__kernel

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

testing_kernel = ""

class globalKernel:
  @timed("globalKernel.__init__")
  def __init__(self, att: attribute):
    self.__kernelString: str = ""
    self.__att = att
    self.__kernel = None
    self.__generateKernel()

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  def __generateKernel(self) -> None:
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    file_name = f".yasps_tmp/compute_{self.__att.fullName}"
    if not os.path.exists(f'{file_name}.so'):
      print(f"File {file_name}.so does not exist, compiling")
      self.__kernelString += '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// here we add code for spd projection
template <unsigned int N>
__device__ void spd_projection(const double *A, double* output, int choice){{
  // Define M as the maximum of N and 4, because 3 by 3 evd is wrong somehow
  const int M = (N < 4) ? 4 : N;

  // Initialize an M x M matrix with zeros
  Eigen::Matrix<double, M, M> symMtr = Eigen::Matrix<double, M, M>::Zero();

  // Copy the input N x N matrix into the top-left corner of the M x M matrix
  for (int row = 0; row < N; ++row) {{
    for (int col = 0; col < N; ++col) {{
      symMtr(row, col) = A[row * N + col];
    }}
  }}
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, M, M>> eigenSolver(symMtr);
  const Eigen::Matrix<double, M, M> B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();
  if (choice == 1) {{
    for (int i = 0; i < M; i++) {{
      if (eigenValues.data()[i] < 0) {{
        eigenValues.data()[i] = abs(eigenValues.data()[i]);
      }}
    }}
  }}else{{
    for (int i = 0; i < M; i++) {{
      if (eigenValues.data()[i] < 0) {{
        eigenValues.data()[i] = 0.0;
      }}
    }}
  }}
  // Reconstruct the matrix without using a diagonal matrix
  // Scale columns of B by corresponding eigenvalues
  Eigen::Matrix<double, M, M> C;
  for (int i = 0; i < N; ++i) {{
    C.col(i) = B.col(i) * eigenValues[i];
  }}
  // Compute the reconstructed matrix: A_reconstructed = C * B.transpose()
  Eigen::Matrix<double, M, M> A_reconstructed = C * B.transpose();
  // Copy the top-left N x N submatrix back to A
  for (int row = 0; row < N; ++row) {{
    for (int col = 0; col < N; ++col) {{
      output[row * N + col] = A_reconstructed(row, col);
    }}
  }}
  return;
}}
'''

      for item in sortedDependency:
        self.__kernelString += f"{item.kernelHeader};"
      self.__kernelString += f"{self.__att.deviceKernel.kernelHeader};"

      for item in sortedDependency:
        self.__kernelString += item.kernelString
      self.__kernelString += self.__att.deviceKernel.kernelString

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
      self.__kernelString += f'''
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
'''
      self.__kernelString += f'''
extern "C"
void compute(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sortedPrimitiveUnions])}
  double* result,
  unsigned int MAX_INDEX
){{
  {attributeName}_global_function<<<(MAX_INDEX + 255) / 256, 256>>>(
    {"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sortedPrimitiveUnions])}
    result,
    MAX_INDEX
  );
  cudaDeviceSynchronize();
}}
'''
      self.__kernelString = prune_duplicate_functions(self.__kernelString)
      f = open(f"{file_name}.cu", 'w')
      f.write(self.__kernelString)
      f.close()
      os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda '-I/usr/include/eigen3' --expt-relaxed-constexpr --disable-warnings -std=c++11")
      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute
      self.__kernel.argtypes = [
        *[ctypes.c_void_p for _ in sortedDatas],
        *[ctypes.c_void_p for _ in sortedConnectivities],
        *[ctypes.c_void_p for x in sortedConnectivities if x.dimension == 0],
        *[ctypes.c_void_p for x in sortedPrimitiveUnions],
        ctypes.c_void_p,  # result
        ctypes.c_uint  # MAX_INDEX
      ]
      self.__kernel.restype = None
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
      self.__kernel.restype = None

  @timed("globalKernel.compute")
  def compute(self, output):
    assert self.__kernel is not None
    counts_gpu = [x.children_primitive_counts_gpu for x in self.__att.deviceKernel.kernelPrimitiveUnions]
    args = [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelDatas]
    args += [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelConnectivity]
    args += [self.__to_void_p(x.compressedRows) for x in self.__att.deviceKernel.kernelConnectivity if x.dimension == 0]
    args += [self.__to_void_p(x) for x in counts_gpu]
    args += [self.__to_void_p(output)]
    args += [ctypes.c_uint32(self.__att.correspondance.numInstances)]
    # print("Counts check")
    # print([x.get() for x in counts_gpu])
    # print([x.fullName for x in self.__att.deviceKernel.kernelPrimitiveUnions])
    # print("Num instance check")
    # print(self.__att.correspondance.numInstances)
    self.__kernel(*args)





  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self):
    return self.__kernel

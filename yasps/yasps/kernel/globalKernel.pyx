# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from pycuda.compiler import SourceModule
import pycuda.driver as pd
from typing import Optional, List
from yasps.helper import get_mangled_name
from yasps.helper import prune_duplicate_functions

testing_kernel = ""

class globalKernel:
  def __init__(self, att: attribute):
    self.__kernelString: str = ""
    self.__kernel: Optional[pd.Function] = None
    self.__generateKernel(att)


  def __generateKernel(self, attr: attribute) -> None:
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = attr.deviceKernel.dependents
    sortedDatas: List[attribute] = attr.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = attr.deviceKernel.kernelConnectivity
    # add the includes
    self.__kernelString += f'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#include <Eigen/Core>
#include <Eigen/Dense>

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
  Eigen::Matrix<double, M, M> B = eigenSolver.eigenvectors();
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
    self.__kernelString += f"{attr.deviceKernel.kernelHeader};"

    for item in sortedDependency:
      self.__kernelString += item.kernelString
    self.__kernelString += attr.deviceKernel.kernelString

    # now actually generate the global kernel
    attributeName: str = ""
    if attr.name == "":
      attributeName = attr.fullName.replace("-", "_neg_")
    else:
      attributeName = attr.fullName.replace("-", "_neg_")

    kernelRawName = f'''
__global__ void {attributeName}_global_function({"".join([f"const double* {x.code_generation_data_name}, " for x in sortedDatas])}{"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sortedConnectivities])}{"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}double* result, unsigned int MAX_INDEX)'''
    self.__kernelString += f'''
{kernelRawName}{{
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  // now we call the device function
  {attributeName}_device_function({"".join([f"{x.code_generation_data_name}, " for x in sortedDatas])}{"".join([f"{x.code_generation_index_name}, " for x in sortedConnectivities])}{"".join([f"{x.code_generation_csr_name}, " for x in sortedConnectivities if x.dimension == 0])}index, result + index * {attr.size});
}}
'''
    # prune duplicate functions
    self.__kernelString = prune_duplicate_functions(self.__kernelString)
    # generate the code to check
    f = open("testing_kernel.cu", "w")
    f.write(self.__kernelString)
    f.close()


    mod = SourceModule(
      self.__kernelString,
      options = ["-std=c++11", '-O3', '-I/usr/include/eigen3', "--expt-relaxed-constexpr", "--disable-warnings"],
      no_extern_c = True
    )
    kernel_name: str = get_mangled_name(kernelRawName, f'{attributeName}_global_function')
    self.__kernel = mod.get_function(kernel_name)



  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self) -> pd.Function:
    return self.__kernel

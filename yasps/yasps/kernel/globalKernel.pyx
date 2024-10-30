# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from pycuda.compiler import SourceModule
import pycuda.driver as pd
from typing import Optional, List
from yasps.helper import get_mangled_name

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
    # generate the code to check
    f = open("testing_kernel.cu", "w")
    f.write(self.__kernelString)
    f.close()

    # if "stable" in attributeName:
    #   f = open("/home/xuan/Desktop/research/yasps/tests/energy/testing_kernel_stable.cu", 'r')
    #   self.__kernelString = f.read()
    #   f.close()

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

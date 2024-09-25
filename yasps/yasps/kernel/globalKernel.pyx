# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from pycuda.compiler import SourceModule
import pycuda.driver as pd
from typing import Optional, List
from yasps.helper import mangle_function_name

testing_kernel = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#include <Eigen/Core>
#include <Eigen/Dense>

__device__ __inline__ void scene0_mesh1_box_vertices_position_device_function(const double* scene0_mesh1_box_vertices_position_global_data, unsigned int scene0_mesh1_box_vertices_index, double* result);
__device__ __inline__ void attr_2135261969964008520466497175698682107904885953521130276238228463269580701963254917487818494896317884355041057720712784118276025015450406366006716703617834_device_function(const double* scene0_mesh1_box_vertices_position_global_data, unsigned int scene0_mesh1_box_vertices_index, double* result);
__device__ __inline__ void scene0_mesh1_box_vertices_position_device_function(const double* scene0_mesh1_box_vertices_position_global_data, unsigned int scene0_mesh1_box_vertices_index, double* result){

  #pragma unroll
  for (unsigned int i = 0; i < 3; i++) {
    result[i] = scene0_mesh1_box_vertices_position_global_data[scene0_mesh1_box_vertices_index * 3 + i];
  }
  printf("Checking box vertex 0: %lf\\n", scene0_mesh1_box_vertices_position_global_data[0]);
}
__device__ __inline__ void attr_2135261969964008520466497175698682107904885953521130276238228463269580701963254917487818494896317884355041057720712784118276025015450406366006716703617834_device_function(const double* scene0_mesh1_box_vertices_position_global_data, unsigned int scene0_mesh1_box_vertices_index, double* result){

  double scene0_mesh1_box_vertices_position_local_data_temp[3];

  scene0_mesh1_box_vertices_position_device_function(scene0_mesh1_box_vertices_position_global_data, scene0_mesh1_box_vertices_index, scene0_mesh1_box_vertices_position_local_data_temp);

  printf("Id: %u, data: %lf, %lf, %lf\\n", scene0_mesh1_box_vertices_index, scene0_mesh1_box_vertices_position_local_data_temp[0], scene0_mesh1_box_vertices_position_local_data_temp[1], scene0_mesh1_box_vertices_position_local_data_temp[2]);


  Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> scene0_mesh1_box_vertices_position_local_data(scene0_mesh1_box_vertices_position_local_data_temp);


  Eigen::Matrix<double, 1, 3, Eigen::RowMajor> INTERMEDIATE_0 = scene0_mesh1_box_vertices_position_local_data * 1.5;

  // put the result back
  #pragma unroll
  for (unsigned int i = 0; i < 3; i++){
    result[i] = INTERMEDIATE_0.data()[i];
  }

}
__global__ void attr_2135261969964008520466497175698682107904885953521130276238228463269580701963254917487818494896317884355041057720712784118276025015450406366006716703617834_global_function(const double* scene0_mesh1_box_vertices_position_global_data, double* result, unsigned int MAX_INDEX){
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){
    return;
  }
  // now we call the device function
  attr_2135261969964008520466497175698682107904885953521130276238228463269580701963254917487818494896317884355041057720712784118276025015450406366006716703617834_device_function(scene0_mesh1_box_vertices_position_global_data, index, result + index * 3);
}
'''


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
      attributeName = f'attr_{attr.hash}'
    else:
      attributeName = attr.fullName

    self.__kernelString += f'''
__global__ void {attributeName}_global_function({"".join([f"const double* {x.fullName}_global_data, " for x in sortedDatas])}{"".join([f"const unsigned int* {x.fullName}_global_indices, " for x in sortedConnectivities])}double* result, unsigned int MAX_INDEX){{
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  // now we call the device function
  {attributeName}_device_function({"".join([f"{x.fullName}_global_data, " for x in sortedDatas])}{"".join([f"{x.fullName}_global_indices, " for x in sortedConnectivities])}index, result + index * {attr.size});
}}
'''

    # # for debugging
    # self.__kernelString = testing_kernel
    # compile the code with eigen library and get the function
    mod = SourceModule(
      self.__kernelString,
      options = ["-std=c++11", '-O3', '-I/usr/include/eigen3', "--expt-relaxed-constexpr"],
      no_extern_c = True
    )
    # print(self.__kernelString)

    # get the mangled name
    input_types = []
    for _ in sortedDatas:
      input_types.append("const double*")
    for _ in sortedConnectivities:
      input_types.append("const unsigned int*")
    input_types.append("double*")
    input_types.append("unsigned int")
    kernel_name: str = mangle_function_name(f"{attributeName}_global_function", input_types)
    self.__kernel = mod.get_function(kernel_name)



  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self) -> pd.Function:
    return self.__kernel

from __future__ import annotations

import ctypes
import os
import time
from typing import List

import numpy as np
import pycuda.gpuarray as gpuarray

from yasps.context import context
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.helper import timed


rectangular_coordinate_kernel_string = r'''
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

__global__ void count_rectangular_coordinates(
  const unsigned int* row_indices,
  const short* row_permutations,
  unsigned int row_stride,
  const unsigned int* column_indices,
  const short* column_permutations,
  unsigned int column_stride,
  unsigned int* coordinate_outer,
  unsigned int num_instances
) {
  unsigned int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance < num_instances) {
    unsigned int row_count = 0;
    unsigned int column_count = 0;
    for (unsigned int i = 0; i < row_stride; ++i) {
      if (row_permutations[instance * row_stride + i] > 0 &&
          row_indices[instance * row_stride + i] >= 2) {
        ++row_count;
      }
    }
    for (unsigned int j = 0; j < column_stride; ++j) {
      if (column_permutations[instance * column_stride + j] > 0 &&
          column_indices[instance * column_stride + j] >= 2) {
        ++column_count;
      }
    }
    coordinate_outer[instance + 1] = row_count * column_count;
  }
}

__global__ void generate_rectangular_coordinates(
  const unsigned int* row_indices,
  const unsigned short* row_sizes,
  const short* row_permutations,
  unsigned int row_stride,
  const unsigned int* column_indices,
  const unsigned short* column_sizes,
  const short* column_permutations,
  unsigned int column_stride,
  const unsigned int* coordinate_outer,
  unsigned int* coordinates,
  unsigned short* dimensions,
  unsigned int num_instances
) {
  unsigned int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance < num_instances) {
    unsigned int output = coordinate_outer[instance];
    for (unsigned int i = 0; i < row_stride; ++i) {
      unsigned int row_slot = instance * row_stride + i;
      if (row_permutations[row_slot] <= 0 || row_indices[row_slot] < 2) {
        continue;
      }
      for (unsigned int j = 0; j < column_stride; ++j) {
        unsigned int column_slot = instance * column_stride + j;
        if (column_permutations[column_slot] <= 0 ||
            column_indices[column_slot] < 2) {
          continue;
        }
        coordinates[2 * output] = row_indices[row_slot] - 2;
        coordinates[2 * output + 1] = column_indices[column_slot] - 2;
        dimensions[2 * output] = row_sizes[row_slot];
        dimensions[2 * output + 1] = column_sizes[column_slot];
        ++output;
      }
    }
  }
}

extern "C"
int count_rectangular_coordinate_entries(
  const unsigned int* row_indices,
  const short* row_permutations,
  unsigned int row_stride,
  const unsigned int* column_indices,
  const short* column_permutations,
  unsigned int column_stride,
  unsigned int* coordinate_outer,
  unsigned int num_instances,
  unsigned int& total_coordinates
) {
  cudaMemset(coordinate_outer, 0, sizeof(unsigned int) * (num_instances + 1));
  count_rectangular_coordinates<<<(num_instances + 255) / 256, 256>>>(
    row_indices, row_permutations, row_stride,
    column_indices, column_permutations, column_stride,
    coordinate_outer, num_instances
  );
  thrust::device_ptr<unsigned int> outer(coordinate_outer);
  thrust::inclusive_scan(outer + 1, outer + num_instances + 1, outer + 1);
  cudaMemcpy(
    &total_coordinates,
    coordinate_outer + num_instances,
    sizeof(unsigned int),
    cudaMemcpyDeviceToHost
  );
  cudaError_t err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : -1;
}

extern "C"
int generate_rectangular_coordinate_entries(
  const unsigned int* row_indices,
  const unsigned short* row_sizes,
  const short* row_permutations,
  unsigned int row_stride,
  const unsigned int* column_indices,
  const unsigned short* column_sizes,
  const short* column_permutations,
  unsigned int column_stride,
  const unsigned int* coordinate_outer,
  unsigned int* coordinates,
  unsigned short* dimensions,
  unsigned int num_instances
) {
  generate_rectangular_coordinates<<<(num_instances + 255) / 256, 256>>>(
    row_indices, row_sizes, row_permutations, row_stride,
    column_indices, column_sizes, column_permutations, column_stride,
    coordinate_outer, coordinates, dimensions, num_instances
  );
  cudaError_t err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : -1;
}
'''


class secondOrderJacobianIndicesKernel:
  def __init__(
    self,
    row_indices_kernel: gradientIndicesKernel,
    column_indices_kernel: gradientIndicesKernel,
    row_start_indices: List[int],
    column_start_indices: List[int]
  ):
    self.__row_indices_kernel = row_indices_kernel
    self.__column_indices_kernel = column_indices_kernel
    self.__row_start_indices = list(row_start_indices)
    self.__column_start_indices = list(column_start_indices)
    self.__coordinates = gpuarray.empty(0, dtype=np.uint32)
    self.__dimensions = gpuarray.empty(0, dtype=np.uint16)
    self.__coordinate_outer = gpuarray.empty(0, dtype=np.uint32)
    self.__num_total_coordinates = 0
    self.__count_kernel = None
    self.__generate_kernel = None
    self.__context = context()

  @property
  def rowIndicesKernel(self):
    return self.__row_indices_kernel

  @property
  def columnIndicesKernel(self):
    return self.__column_indices_kernel

  @property
  def outputCoordinates(self):
    return self.__coordinates

  @property
  def outputBlockDimensions(self):
    return self.__dimensions

  @property
  def coordinateCountsOuter(self):
    return self.__coordinate_outer

  @property
  def numTotalCoordinates(self):
    return self.__num_total_coordinates

  @property
  def numInstances(self):
    return self.__row_indices_kernel.numInstances

  def __to_void_p(self, value):
    if value is None or value.size == 0:
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(int(value.gpudata))

  def __loadKernels(self):
    if self.__count_kernel is not None:
      return
    file_name = ".yasps_constant/rectangular_coordinate_kernel"
    if not os.path.exists(f"{file_name}.so"):
      time_start = time.time()
      with open(f"{file_name}.cu", "w") as output:
        output.write(rectangular_coordinate_kernel_string)
      result = os.system(
        f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so "
        f"{file_name}.cu -O3 -arch=sm_89 -lcudart -lcuda"
      )
      if result != 0:
        raise RuntimeError(
          "secondOrderJacobianIndicesKernel: failed to compile rectangular coordinate kernel."
        )
      print(
        "Time taken to compile rectangular coordinate kernel: "
        f"{(time.time() - time_start) * 1000.0} ms"
      )
    library = ctypes.CDLL(f"{file_name}.so")
    self.__count_kernel = library.count_rectangular_coordinate_entries
    self.__count_kernel.restype = ctypes.c_int
    self.__count_kernel.argtypes = (
      [ctypes.c_void_p] * 2
      + [ctypes.c_uint32]
      + [ctypes.c_void_p] * 2
      + [ctypes.c_uint32]
      + [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    )
    self.__generate_kernel = library.generate_rectangular_coordinate_entries
    self.__generate_kernel.restype = ctypes.c_int
    self.__generate_kernel.argtypes = (
      [ctypes.c_void_p] * 3
      + [ctypes.c_uint32]
      + [ctypes.c_void_p] * 3
      + [ctypes.c_uint32]
      + [ctypes.c_void_p] * 3
      + [ctypes.c_uint32]
    )

  @timed("secondOrderJacobianIndicesKernel.computeIndices")
  def computeIndices(self):
    self.__context.useDefaultContext()
    self.__row_indices_kernel.computeIndices(self.__row_start_indices)
    self.__column_indices_kernel.computeIndices(self.__column_start_indices)
    if self.__row_indices_kernel.numInstances != self.__column_indices_kernel.numInstances:
      raise ValueError(
        "secondOrderJacobianIndicesKernel: row and column index streams "
        "must have the same number of source instances."
      )

    num_instances = self.numInstances
    if num_instances == 0:
      self.__num_total_coordinates = 0
      return
    if self.__coordinate_outer.size < num_instances + 1:
      self.__coordinate_outer = gpuarray.zeros(num_instances + 1, dtype=np.uint32)
    self.__loadKernels()

    total_coordinates = ctypes.c_uint32(0)
    error_code = self.__count_kernel(
      self.__to_void_p(self.__row_indices_kernel.outputIndices),
      self.__to_void_p(self.__row_indices_kernel.outputPermutations),
      ctypes.c_uint32(self.__row_indices_kernel.maxNumIndicesNeeded),
      self.__to_void_p(self.__column_indices_kernel.outputIndices),
      self.__to_void_p(self.__column_indices_kernel.outputPermutations),
      ctypes.c_uint32(self.__column_indices_kernel.maxNumIndicesNeeded),
      self.__to_void_p(self.__coordinate_outer),
      ctypes.c_uint32(num_instances),
      ctypes.byref(total_coordinates)
    )
    if error_code != 0:
      raise RuntimeError(
        "secondOrderJacobianIndicesKernel: failed to count rectangular coordinates."
      )
    self.__num_total_coordinates = total_coordinates.value
    required_entries = self.__num_total_coordinates * 2
    if self.__coordinates.size < required_entries:
      self.__coordinates = gpuarray.empty(required_entries, dtype=np.uint32)
      self.__dimensions = gpuarray.empty(required_entries, dtype=np.uint16)
    if self.__num_total_coordinates == 0:
      return

    error_code = self.__generate_kernel(
      self.__to_void_p(self.__row_indices_kernel.outputIndices),
      self.__to_void_p(self.__row_indices_kernel.outputSizes),
      self.__to_void_p(self.__row_indices_kernel.outputPermutations),
      ctypes.c_uint32(self.__row_indices_kernel.maxNumIndicesNeeded),
      self.__to_void_p(self.__column_indices_kernel.outputIndices),
      self.__to_void_p(self.__column_indices_kernel.outputSizes),
      self.__to_void_p(self.__column_indices_kernel.outputPermutations),
      ctypes.c_uint32(self.__column_indices_kernel.maxNumIndicesNeeded),
      self.__to_void_p(self.__coordinate_outer),
      self.__to_void_p(self.__coordinates),
      self.__to_void_p(self.__dimensions),
      ctypes.c_uint32(num_instances)
    )
    if error_code != 0:
      raise RuntimeError(
        "secondOrderJacobianIndicesKernel: failed to generate rectangular coordinates."
      )

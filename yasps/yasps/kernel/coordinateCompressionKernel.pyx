from __future__ import annotations
from ast import Str
from yasps.attribute import attribute
from typing import List, Tuple, Dict
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
from yasps.attribute import JOIN, DATA, UNION
from yasps.helper import timed
import os
import pycuda.driver as cuda



get_unique_coords_kernel_string: str = '''
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <cuda_runtime.h>
#include <vector>

struct CoordDim {
  // store the dimension as well as the coordinates
  unsigned int row, col;
  unsigned short h, w;
  unsigned int placeholder; // for making it 16 bytes in total


  __host__ __device__
  bool operator<(const CoordDim& other) const {
    if (h != other.h) return h < other.h;
    if (w != other.w) return w < other.w;
    if (row != other.row) return row < other.row;
    return col < other.col;
  }

  __host__ __device__
  bool operator==(const CoordDim& other) const {
    return row == other.row && col == other.col && h == other.h && w == other.w;
  }
};
// make two arrays into 1
__global__ void pack_coord_dim_kernel(CoordDim* output, const unsigned int* coords, const unsigned short int* dims, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    output[i] = CoordDim{
      coords[2 * i], coords[2 * i + 1],
      dims[2 * i], dims[2 * i + 1],
      0
    };
  }
}

extern "C"
void get_unique_coords(
  const unsigned int* coords,           // array of device pointers, size K * 2
  const unsigned short int* dims,       // array of device pointers, size K * 2
  const unsigned int NUM_COORDINATES,                // the total number of coordinates, K
  CoordDim* uncompressedCoordinatesAndDimensionsTmp, // we will use it to determine the number of unique coordinates
  unsigned int& num_unique_coords
) {
  // printf("Starting get_unique_coords...\\n");
  // printf("Total number of coordinates: %u\\n", NUM_COORDINATES);
  // Step 1: flatten and accumulate the arrays into one giant array
  pack_coord_dim_kernel<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    coords,
    dims,
    NUM_COORDINATES
  );
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\\n", cudaGetErrorString(err));
  }
  // printf("Finished packing coordinates...\\n");

  // Step2: deduplicate the coordinates
  thrust::sort(thrust::device, uncompressedCoordinatesAndDimensionsTmp, uncompressedCoordinatesAndDimensionsTmp + NUM_COORDINATES);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\\n", cudaGetErrorString(err));
  }
  // printf("Finished sorting coordinates...\\n");

  // get the unique coordinates as well as number of unique coordinates
  auto unique_end = thrust::unique(thrust::device, uncompressedCoordinatesAndDimensionsTmp, uncompressedCoordinatesAndDimensionsTmp + NUM_COORDINATES);
  num_unique_coords = unique_end - uncompressedCoordinatesAndDimensionsTmp;
  // printf("Finished getting unique coordinates...\\n");
  // printf("Unique coordinates: %u\\n", num_unique_coords);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\\n", cudaGetErrorString(err));
  }
}
'''


compress_unique_coords_kernel_string = '''
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <cuda_runtime.h>
#include <vector>

struct CoordDim {
  // store the dimension as well as the coordinates
  unsigned int row, col;
  unsigned short h, w;
  unsigned int placeholder; // for making it 16 bytes in total


  __host__ __device__
  bool operator<(const CoordDim& other) const {
    if (h != other.h) return h < other.h;
    if (w != other.w) return w < other.w;
    if (row != other.row) return row < other.row;
    return col < other.col;
  }

  __host__ __device__
  bool operator==(const CoordDim& other) const {
    return row == other.row && col == other.col && h == other.h && w == other.w;
  }
};

struct CoordCompare {
  __host__ __device__
  bool operator()(const thrust::tuple<unsigned int, unsigned int>& a,
                  const thrust::tuple<unsigned int, unsigned int>& b) const {
    if (thrust::get<0>(a) != thrust::get<0>(b)) return thrust::get<0>(a) < thrust::get<0>(b);
    return thrust::get<1>(a) < thrust::get<1>(b);
  }
};

// make two arrays into 1
__global__ void pack_coord_dim_kernel(CoordDim* output, const unsigned int* coords, const unsigned short int* dims, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    output[i] = CoordDim{
      coords[2 * i], coords[2 * i + 1],
      dims[2 * i], dims[2 * i + 1],
      0
    };
  }
}

// unpack the coordinates
__global__ void extract_coords_kernel(const CoordDim* input, unsigned int* coords_out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    coords_out[2 * i]     = input[i].row;
    coords_out[2 * i + 1] = input[i].col;
  }
}

// unpack the dimensions
__global__ void extract_dims_kernel(const CoordDim* input, unsigned short int* dims_out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    dims_out[2 * i]     = input[i].h;
    dims_out[2 * i + 1] = input[i].w;
  }
}

__global__ void map_index_to_start_position_in_data(unsigned int* indexInUniqueCoords, const unsigned short int* unique_dims_out, const unsigned int* unique_dims_outer_indices, unsigned int NUM_COORDINATES, unsigned num_unique_dims){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NUM_COORDINATES) {
    unsigned int index = indexInUniqueCoords[i];
    for (unsigned int j = 0; j < num_unique_dims; j++){
      if (index >= unique_dims_outer_indices[j] && index < unique_dims_outer_indices[j + 1]){
        // first we determine its the Nth index in this dimension's block
        index -= unique_dims_outer_indices[j];
        unsigned int indCopy = index;
        index = 0;
        // now we add up the actual data size
        for (unsigned int k = 0; k < j; k++){
          index += unique_dims_out[k * 2] * unique_dims_out[k * 2 + 1] * (unique_dims_outer_indices[k + 1] - unique_dims_outer_indices[k]);
        }
        index += indCopy * unique_dims_out[j * 2] * unique_dims_out[j * 2 + 1];
        indexInUniqueCoords[i] = index;
        break;
      }
    }
  }
}

__global__ void map_outer_indices_to_position_in_data(unsigned int* outerIndices, unsigned int* uniqueDimsBlockCounts, const unsigned int* outerIndicesCpy, const unsigned short int* unique_dims_out, const unsigned num_unique_dims){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= num_unique_dims){
    unsigned int index = 0;
    for (unsigned int j = 0; j < i; j++){
      index += unique_dims_out[j * 2] * unique_dims_out[j * 2 + 1] * (outerIndicesCpy[j + 1] - outerIndicesCpy[j]);
    }
    outerIndices[i] = index;
    if (i < num_unique_dims){
      uniqueDimsBlockCounts[i] = outerIndicesCpy[i + 1] - outerIndicesCpy[i];
    }
  }
}

struct CoordToTuple {
  __host__ __device__
  thrust::tuple<unsigned int, unsigned int> operator()(const CoordDim& c) const {
    return thrust::make_tuple(c.row, c.col);
  }
};


extern "C"
void compress_unique_coords(
  const unsigned int* coords,           // array of device pointers, size K
  const unsigned short int* dims,       // array of device pointers, size K
  const unsigned int NUM_COORDINATES,                // the total number of coordinates, K
  CoordDim* uncompressedCoordinatesAndDimensionsTmp, // this array contains the sorted and compressed coordinates without duplicates
  const unsigned int num_unique_coords, // the number of unique coordinates
  unsigned int* unique_coords_out, // actually copy the unique coordinates to putput
  unsigned short int* unique_dims_out, // actually copy the unique dimensions to output
  unsigned int* unique_dims_outer_indices, // compute for each unique dimension, where does it start in the data array
  unsigned int* unique_dims_block_counts, // the number of blocks for each unique dimension
  unsigned int* lookup, // compute for each coordinate, where does it start in the data array
  unsigned int& num_unique_dims // get the number of unique dimensions
) {
  CoordDim* unique_coord_dims; // this wil copy the unique coordinates and dimensions
  cudaMalloc(&unique_coord_dims, num_unique_coords * sizeof(CoordDim));
  cudaMemcpy(unique_coord_dims, uncompressedCoordinatesAndDimensionsTmp, num_unique_coords * sizeof(CoordDim), cudaMemcpyDeviceToDevice);

  // we first copy the unique coordinates to output
  extract_coords_kernel<<<(num_unique_coords + 255) / 256, 256>>>(uncompressedCoordinatesAndDimensionsTmp, unique_coords_out, num_unique_coords);
  cudaDeviceSynchronize();

  // now we want to extract the unique dimensions
  // Deduplicate by (h, w) only
  cudaDeviceSynchronize();
  auto unique_dim_end = thrust::unique(
    thrust::device,
    uncompressedCoordinatesAndDimensionsTmp,
    uncompressedCoordinatesAndDimensionsTmp + num_unique_coords,
    [] __device__ (const CoordDim& a, const CoordDim& b) {
      return a.h == b.h && a.w == b.w;
    }
  );
  cudaDeviceSynchronize();
  num_unique_dims = unique_dim_end - uncompressedCoordinatesAndDimensionsTmp;
  // printf("num_unique_dims: %u\\n", num_unique_dims);
  // now we copy the unique dimensions to output
  extract_dims_kernel<<<(num_unique_dims + 255) / 256, 256>>>(uncompressedCoordinatesAndDimensionsTmp, unique_dims_out, num_unique_dims);


  // we will now copy the unique dimensions but in CoordDim format
  CoordDim* unique_coord_dims_dimension_only;
  cudaMalloc(&unique_coord_dims_dimension_only, num_unique_dims * sizeof(CoordDim));
  // copy the unique coordinates to output
  cudaMemcpy(unique_coord_dims_dimension_only, uncompressedCoordinatesAndDimensionsTmp, num_unique_dims * sizeof(CoordDim), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();


  // ok this time, we want to compute how many times each dimension appears in the compressed array
  // and store it in the unique_dims_out array
  auto coorddim_cmp = [] __device__ (const CoordDim& a, const CoordDim& b) {
    if (a.h != b.h) return a.h < b.h;
    return a.w < b.w;
  };

  thrust::lower_bound(
    thrust::device,
    unique_coord_dims,
    unique_coord_dims + num_unique_coords,
    unique_coord_dims_dimension_only,
    unique_coord_dims_dimension_only + num_unique_dims,
    unique_dims_outer_indices,
    coorddim_cmp
  );
  // now, this outer_indicies will tell us the start and end position of each dimension in the final compressed coordinates
  thrust::device_ptr<unsigned int> outer_ptr(unique_dims_outer_indices);
  outer_ptr[num_unique_dims] = num_unique_coords;

  // we will now once again copy the coordinates and the dimensions back to the tmp array
  pack_coord_dim_kernel<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    coords,
    dims,
    NUM_COORDINATES
  );
  cudaDeviceSynchronize();


  thrust::device_ptr<unsigned int> lookup_ptr(lookup);
  thrust::lower_bound(
    thrust::device,
    unique_coord_dims,
    unique_coord_dims + num_unique_coords,
    uncompressedCoordinatesAndDimensionsTmp,
    uncompressedCoordinatesAndDimensionsTmp + NUM_COORDINATES,
    lookup_ptr
  );
  cudaDeviceSynchronize();

  // now we map the index to the actual start position in the data array
  map_index_to_start_position_in_data<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    lookup,
    unique_dims_out,
    unique_dims_outer_indices,
    NUM_COORDINATES,
    num_unique_dims
  );
  cudaDeviceSynchronize();

  // finally, we do this for the outer indices too
  // we first make a copy of the outer indices
  unsigned int* unique_dims_outer_indices_copy;
  cudaMalloc(&unique_dims_outer_indices_copy, (num_unique_dims + 1) * sizeof(unsigned int));
  cudaMemcpy(unique_dims_outer_indices_copy, unique_dims_outer_indices, (num_unique_dims + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  map_outer_indices_to_position_in_data<<<(num_unique_dims + 255) / 256, 256>>>(
    unique_dims_outer_indices,
    unique_dims_block_counts,
    unique_dims_outer_indices_copy,
    unique_dims_out,
    num_unique_dims
  );
  cudaDeviceSynchronize();

  // free the copy of the outer indices
  cudaFree(unique_coord_dims);
  cudaFree(unique_dims_outer_indices_copy);
  cudaFree(unique_coord_dims_dimension_only);
}
'''


# copy the gpu data
def gpu_copy_slice(dst: gpuarray.GPUArray, dst_offset: int, src: gpuarray.GPUArray, count: int):
  itemsize = dst.dtype.itemsize
  assert dst.gpudata is not None
  assert src.gpudata is not None
  cuda.memcpy_dtod(
    int(dst.gpudata) + dst_offset * itemsize,
    int(src.gpudata),
    count * itemsize
  )

# we need for the compression kernel:
# all of the coordinates
# all of the dimensions
# how many coordinates there are for each energys
# for output, we will have:
# a really long array that has all the coordinates, sorted first by dimension, then by coordinate value, with no repeated values
# an array that indicates the unique dimension
# an outer array that indicates where the data starts for each dimension
# and a really long array that indicates for each energy, the position we need to look for in the data array to put the data back
class coordinateCompressionKernel:
  def __init__(self, coordinates: List[gpuarray.GPUArray], dimensions: List[gpuarray.GPUArray], num_coordinates: List[int], wrt: List[attribute]):
    self.__coordinates: List[gpuarray.GPUArray] = coordinates
    self.__dimensions: List[gpuarray.GPUArray] = dimensions
    self.__num_coordinates : List[int]= num_coordinates
    self.__uniqueCoordinates: gpuarray.GPUArray
    self.__uniqueDimensions: gpuarray.GPUArray
    self.__uniqueDimensionsOuterIndices: gpuarray.GPUArray # for each dimension, whats the start and end position inside the data array
    self.__uniqueDimensionsBlockCounts: gpuarray.GPUArray # for each dimension, how many blocks of data are there
    self.__lookupArray: gpuarray.GPUArray # should have the same size as the total number of coordinates
    self.__num_unique_coords: int = 0
    self.__num_unique_dimensions: int = 0

    # we compute the maximum possible number of unique dimensions

    wrt_sizes = [x.size for x in wrt]
    unique_wrt_sizes = set(wrt_sizes)
    largest_num_unique_dimensions = len(unique_wrt_sizes) ** 2 # the maximum size is just the square of len
    print(f"largest_num_unique_dimensions: {largest_num_unique_dimensions}")
    self.__uniqueDimensions = gpuarray.empty(largest_num_unique_dimensions * 2, np.uint16) # allocate the array
    self.__uniqueDimensionsOuterIndices = gpuarray.empty(largest_num_unique_dimensions + 1, np.uint32) # allocate the array
    self.__uniqueDimensionsBlockCounts = gpuarray.empty(largest_num_unique_dimensions, np.uint32) # allocate the array

    # kernels
    self.__get_unique_coords_kernel = None # the kernel that gets the unique coordinates as well as the unique number of coordinates
    self.__compress_unique_coords_kernel = None # the kernel that compresses the unique coordinates and check the position in the actual data
    # invoke functions

  @property
  def uniqueCoordinates(self):
    return self.__uniqueCoordinates

  @property
  def uniqueDimensions(self):
    return self.__uniqueDimensions

  @property
  def uniqueDimensionsOuterIndices(self):
    return self.__uniqueDimensionsOuterIndices

  @property
  def uniqueDimensionsBlockCounts(self):
    return self.__uniqueDimensionsBlockCounts

  @property
  def lookupArray(self):
    return self.__lookupArray

  @property
  def lookupArrays(self):
    # this is a bit different, we slice it to match each input length
    arrays = []
    count = 0
    for total_coordinates in self.__num_coordinates:
      arrays.append(self.__lookupArray[count:count+total_coordinates])
      count += total_coordinates
    return arrays

  @property
  def numUniqueCoordinates(self):
    return self.__num_unique_coords

  @property
  def numUniqueDimensions(self):
    return self.__num_unique_dimensions

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @property
  def totalBlockSize(self):
    return self.__uniqueDimensionsOuterIndices.get()[self.__num_unique_dimensions]

  @timed("coordinateCompressionKernel.__getUniqueCoordinatesAndDimensions")
  def __getUniqueCoordinatesAndDimensions(self, uncompressedCoordinates, uncompressedDimensions, total_coordinates, uncompressedCoordinatesAndDimensionsTmp):
    # in this function we will get the unqique coordinates in the uncompressedCoordinatesAndDimensionsTmp
    # we will later on use this array in another function to allocate the space for actual unique coordinates
    # as well as determining the number of blocks for each dimension
    # now we check if kernel exists
    if self.__get_unique_coords_kernel is None:
      file_name = ".yasps_tmp/get_unique_coords_kernel"
      # check if the file exists
      if not os.path.exists(f'{file_name}.so'):
        # generate the kernel
        f = open(f"{file_name}.cu", 'w')
        f.write(get_unique_coords_kernel_string)
        f.close()
        os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda")
        self.__get_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").get_unique_coords # get the compiled kernel
        self.__get_unique_coords_kernel.restype = None # set the return type to None
        self.__get_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint32)] * 1
      else:
        self.__get_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").get_unique_coords # get the compiled kernel
        self.__get_unique_coords_kernel.restype = None # set the return type to None
        self.__get_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint32)] * 1

    # we have confirmed the kernel is not none
    assert self.__get_unique_coords_kernel is not None
    # call the kernel
    num_unique_coords = ctypes.c_uint32(0)
    self.__get_unique_coords_kernel(
      self.__to_void_p(uncompressedCoordinates),
      self.__to_void_p(uncompressedDimensions),
      ctypes.c_uint32(total_coordinates),
      self.__to_void_p(uncompressedCoordinatesAndDimensionsTmp),
      ctypes.byref(num_unique_coords)
    )
    # here we will get the unique number of coordinates
    self.__num_unique_coords = num_unique_coords.value
    print(f"Number of unique coordinates: {self.__num_unique_coords}")

  @timed("coordinateCompressionKernel.__getUniqueCoordinatesStartAndEnd")
  def __getUniqueCoordinatesStartAndEnd(self, uncompressedCoordinates, uncompressedDimensions, total_coordinates, uncompressedCoordinatesAndDimensionsTmp):
    # now we allocate two arrays for unique dimensions and unique coordinates
    self.__uniqueCoordinates = gpuarray.GPUArray(self.__num_unique_coords * 2, dtype=np.uint32)
    if self.__compress_unique_coords_kernel is None:
      file_name = ".yasps_tmp/compress_unique_coords_kernel"
      if not os.path.exists(f'{file_name}.so'):
        # generate the kernel
        f = open(f"{file_name}.cu", 'w')
        f.write(compress_unique_coords_kernel_string)
        f.close()
        os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_86 -lcudart -lcuda --extended-lambda")
        self.__compress_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").compress_unique_coords # get the compiled kernel
        self.__compress_unique_coords_kernel.restype = None # set the return type to None
        self.__compress_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] + [ctypes.c_uint32] +[ctypes.c_void_p] * 5 + [ctypes.POINTER(ctypes.c_uint32)] * 1
      else:
        self.__compress_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").compress_unique_coords # get the compiled kernel
        self.__compress_unique_coords_kernel.restype = None # set the return type to None
        self.__compress_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] + [ctypes.c_uint32] +[ctypes.c_void_p] * 5 + [ctypes.POINTER(ctypes.c_uint32)] * 1
    assert self.__compress_unique_coords_kernel is not None


    num_unique_dims = ctypes.c_uint32(0)
    self.__compress_unique_coords_kernel(
      self.__to_void_p(uncompressedCoordinates),
      self.__to_void_p(uncompressedDimensions),
      ctypes.c_uint32(total_coordinates),
      self.__to_void_p(uncompressedCoordinatesAndDimensionsTmp),
      ctypes.c_uint32(self.__num_unique_coords),
      self.__to_void_p(self.__uniqueCoordinates),
      self.__to_void_p(self.__uniqueDimensions),
      self.__to_void_p(self.__uniqueDimensionsOuterIndices),
      self.__to_void_p(self.__uniqueDimensionsBlockCounts),
      self.__to_void_p(self.__lookupArray),
      ctypes.byref(num_unique_dims)
    )
    self.__num_unique_dimensions = num_unique_dims.value
    print(f"Number of unique dimensions: {self.__num_unique_dimensions}")
    print(f"Total block size: {self.__uniqueDimensionsOuterIndices.get()[self.__num_unique_dimensions]}")



  @timed("coordinateCompressionKernel.__compressCoordinatesAndDimensions")
  def __compressCoordinatesAndDimensions(self):
    total_coordinates = sum(self.__num_coordinates)
    print(f"Total coordinates: {total_coordinates}")
    # create a new data type for coordinates and dimensions
    coord_dim_dtype = np.dtype([
      ('row', np.uint32),
      ('col', np.uint32),
      ('h', np.uint16),
      ('w', np.uint16),
      ('placeholder', np.uint32)
    ])
    uncompressedCoordinatesAndDimensionsTmp: gpuarray.GPUArray = gpuarray.empty(total_coordinates, coord_dim_dtype)
    uncompressedCoordinates = gpuarray.empty(total_coordinates * 2, np.uint32)
    uncompressedDimensions = gpuarray.empty(total_coordinates * 2, np.uint16)
    self.__lookupArray = gpuarray.empty(total_coordinates, np.uint32)
    # we first compy all the coordinates and dimensions into the uncompressed array
    count = 0
    for i in range(len(self.__num_coordinates)):
      num_coordinate = self.__num_coordinates[i]
      # copy coordinates and dimensions into the uncompressed array
      gpu_copy_slice(uncompressedCoordinates, count, self.__coordinates[i], num_coordinate * 2)
      gpu_copy_slice(uncompressedDimensions, count, self.__dimensions[i], num_coordinate * 2)
      count += num_coordinate * 2

    # this will put the unique coordinates in the uncompressedCoordinatesAndDimensionsTmp
    # we will then allocate space to copy the actual unique coordinates, and also compute the unique dimensions
    self.__getUniqueCoordinatesAndDimensions(uncompressedCoordinates, uncompressedDimensions, total_coordinates, uncompressedCoordinatesAndDimensionsTmp)
    # now we get the compressed lookup table and the unique coordinates
    self.__getUniqueCoordinatesStartAndEnd(uncompressedCoordinates, uncompressedDimensions, total_coordinates, uncompressedCoordinatesAndDimensionsTmp)


    ########################################################################################
    ########################################################################################
    ## UNCOMMENT THE CODE FOR DEBUGGING THE RESULTS
    ########################################################################################
    ########################################################################################
    # # we now do a cpu check
    # # first we check if the unique dimensions are the same
    # unique_dimensions_cpu = self.__uniqueDimensions.get().flatten()
    # print(f"unique_dimensions_cpu: {unique_dimensions_cpu}")
    # unique_dimensions_set = set([])
    # for i in range(self.__num_unique_dimensions):
    #   unique_dimensions_set.add((unique_dimensions_cpu[i * 2], unique_dimensions_cpu[i * 2 + 1]))
    # unique_dimensions_raw_dict = {}
    # for i in range(len(self.__num_coordinates)):
    #   dimensions = self.__dimensions[i].get()
    #   for j in range(len(dimensions) // 2):
    #     dimension = (dimensions[j * 2], dimensions[j * 2 + 1])
    #     if dimension not in unique_dimensions_raw_dict:
    #       unique_dimensions_raw_dict[dimension] = 0
    #     unique_dimensions_raw_dict[dimension] += 1

    # if unique_dimensions_set != set(unique_dimensions_raw_dict.keys()):
    #   raise ValueError(f"Unique dimensions do not match, {unique_dimensions_set} != {set(unique_dimensions_raw_dict.keys())}")

    # # now we check if the count is correct
    # unique_dimensions_block_counts_cpu = self.__uniqueDimensionsBlockCounts.get().flatten()
    # print(f"Unique dimensions block counts: {unique_dimensions_block_counts_cpu}")
    # print(f"unique_dimensions_raw_dict: {unique_dimensions_raw_dict}")
    # # for i in range(self.__num_unique_dimensions):
    # #   dimension = (unique_dimensions_cpu[i * 2], unique_dimensions_cpu[i * 2 + 1])
    # #   count = unique_dimensions_block_counts_cpu[i]
    # #   if count != unique_dimensions_raw_dict[dimension]:
    # #     raise ValueError(f"Unique dimensions block counts do not match for dimension {dimension}, count {count} does not match {unique_dimensions_raw_dict[dimension]}")

    # # now we check the outer indices
    # unique_dimensions_outer_indices_cpu = self.__uniqueDimensionsOuterIndices.get().flatten()
    # print(f"Unique dimensions outer indices: {unique_dimensions_outer_indices_cpu}")
    # # total_count = 0
    # # for i in range(self.__num_unique_dimensions):
    # #   dimension = (unique_dimensions_cpu[i * 2], unique_dimensions_cpu[i * 2 + 1])
    # #   if unique_dimensions_outer_indices_cpu[i] != total_count:
    # #     raise ValueError(f"Unique dimensions outer indices do not match for dimension {dimension}, index {unique_dimensions_outer_indices_cpu[i]} does not match {total_count}")
    # #   total_count += unique_dimensions_block_counts_cpu[i] * dimension[0] * dimension[1]
    # # if not(int(unique_dimensions_outer_indices_cpu[self.__num_unique_dimensions]) == int(total_count)):
    # #   raise ValueError(f"Unique dimensions for final outer index does not match, count {unique_dimensions_outer_indices_cpu[self.__num_unique_dimensions]} does not match {total_count}")

    # # finally we check if the index is correct
    # count = 0
    # unique_coordinates_cpu = self.__uniqueCoordinates.get().flatten()
    # lookup_cpu = self.__lookupArray.get().flatten()
    # for i in range(len(self.__coordinates)):
    #   coordinates = self.__coordinates[i].get().flatten()
    #   for j in range(len(coordinates) // 2):
    #     coordinate = (int(coordinates[j * 2]), int(coordinates[j * 2 + 1]))
    #     lookup = lookup_cpu[count]
    #     for k in range(len(unique_dimensions_outer_indices_cpu) - 1):
    #       start = unique_dimensions_outer_indices_cpu[k]
    #       end = unique_dimensions_outer_indices_cpu[k + 1]
    #       if start <= lookup < end:
    #         # minus the start
    #         lookup -= start
    #         lookup = (lookup // (unique_dimensions_cpu[k * 2] * unique_dimensions_cpu[k * 2 + 1])) + sum(unique_dimensions_block_counts_cpu[:k])
    #         break
    #     foundCoordinate = (int(unique_coordinates_cpu[lookup * 2]), int(unique_coordinates_cpu[lookup * 2 + 1]))
    #     if not(coordinate == foundCoordinate):
    #       print(coordinates[: 20].reshape(-1, 2))
    #       print(unique_coordinates_cpu[: 20].reshape(-1, 2))
    #       print(lookup_cpu[:20])
    #       raise ValueError(f"Coordinate {coordinate} does not match found coordinate {foundCoordinate} at index {count}, raw lookup: {lookup_cpu[count]}, modified lookup: {lookup}")
    #     count += 1




  def compressCoordinatesAndDimensions(self):
    self.__compressCoordinatesAndDimensions()

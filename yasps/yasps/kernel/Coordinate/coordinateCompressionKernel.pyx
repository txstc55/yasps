from __future__ import annotations
# from ast import Str
from yasps.attribute import attribute
from typing import List, Optional
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
from yasps.helper import timed
import os
import pycuda.driver as cuda
from yasps.context import context
import time

coord_dim_dtype = np.dtype([
  ('row', np.uint32),
  ('col', np.uint32),
  ('h', np.uint16),
  ('w', np.uint16),
  ('placeholder', np.uint32)
])
assert coord_dim_dtype.itemsize == 16

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
static_assert(sizeof(CoordDim) == 16, "CoordDim layout must remain 16 bytes");
// make two arrays into 1
__global__ void pack_coord_dim_kernel(CoordDim* output, const unsigned int* coords, const unsigned short int* dims, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    output[i] = CoordDim{
      coords[2 * i], coords[2 * i + 1],
      dims[2 * i], dims[2 * i + 1],
      static_cast<unsigned int>(i)
    };
  }
}

__global__ void mark_unique_coord_runs(
  const CoordDim* sorted_coord_dims,
  unsigned int* run_ordinals,
  unsigned int N
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    run_ordinals[i] = (
      i == 0 || !(sorted_coord_dims[i] == sorted_coord_dims[i - 1])
    ) ? 1u : 0u;
  }
}

__global__ void scatter_unique_coord_ordinals(
  const CoordDim* sorted_coord_dims,
  const unsigned int* run_ordinals,
  unsigned int* lookup,
  unsigned int N
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    lookup[sorted_coord_dims[i].placeholder] = run_ordinals[i] - 1u;
  }
}

extern "C"
int get_unique_coords(
  unsigned int* coords,
  unsigned short int* dims,
  const unsigned int NUM_COORDINATES,
  CoordDim* uncompressedCoordinatesAndDimensionsTmp,
  unsigned int* lookup,
  unsigned int& num_unique_coords
) {
  // After packing, the concatenated coordinate storage is available as an
  // N-element scratch array without increasing peak memory. The packed
  // placeholder carries each occurrence's original index through the sort.
  unsigned int* run_ordinals = coords;

  pack_coord_dim_kernel<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    coords,
    dims,
    NUM_COORDINATES
  );
  thrust::sort(
    thrust::device,
    uncompressedCoordinatesAndDimensionsTmp,
    uncompressedCoordinatesAndDimensionsTmp + NUM_COORDINATES
  );

  mark_unique_coord_runs<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    run_ordinals,
    NUM_COORDINATES
  );
  thrust::inclusive_scan(
    thrust::device,
    run_ordinals,
    run_ordinals + NUM_COORDINATES,
    run_ordinals
  );
  scatter_unique_coord_ordinals<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    run_ordinals,
    lookup,
    NUM_COORDINATES
  );

  auto unique_end = thrust::unique(thrust::device, uncompressedCoordinatesAndDimensionsTmp, uncompressedCoordinatesAndDimensionsTmp + NUM_COORDINATES);
  num_unique_coords = unique_end - uncompressedCoordinatesAndDimensionsTmp;
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
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
static_assert(sizeof(CoordDim) == 16, "CoordDim layout must remain 16 bytes");

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

__global__ void prepare_unique_metadata(
  const CoordDim* unique_coord_dims,
  unsigned int* unique_coords_out,
  unsigned int* block_offsets,
  unsigned int* dimension_ordinals,
  unsigned int num_unique_coords
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_unique_coords) {
    const CoordDim current = unique_coord_dims[i];
    unique_coords_out[2 * i] = current.row;
    unique_coords_out[2 * i + 1] = current.col;
    block_offsets[i] = static_cast<unsigned int>(current.h) * current.w;
    dimension_ordinals[i] = (
      i == 0
      || current.h != unique_coord_dims[i - 1].h
      || current.w != unique_coord_dims[i - 1].w
    ) ? 1u : 0u;
  }
}

__global__ void write_dimension_metadata(
  const CoordDim* unique_coord_dims,
  const unsigned int* block_offsets,
  const unsigned int* dimension_ordinals,
  unsigned short int* unique_dims_out,
  unsigned int* unique_dims_outer_indices,
  unsigned int* unique_dims_block_counts,
  unsigned int num_unique_coords
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_unique_coords) {
    const CoordDim current = unique_coord_dims[i];
    const bool starts_dimension = (
      i == 0
      || current.h != unique_coord_dims[i - 1].h
      || current.w != unique_coord_dims[i - 1].w
    );
    if (starts_dimension) {
      unsigned int dimension = dimension_ordinals[i] - 1u;
      unique_dims_out[2 * dimension] = current.h;
      unique_dims_out[2 * dimension + 1] = current.w;
      unique_dims_outer_indices[dimension] = block_offsets[i];
      // Temporarily store the first unique-coordinate ordinal. The finalizer
      // converts adjacent starts into block counts.
      unique_dims_block_counts[dimension] = i;
    }
  }
}

__global__ void map_lookup_to_data_offsets(
  unsigned int* lookup,
  const unsigned int* block_offsets,
  unsigned int num_coordinates
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_coordinates) {
    lookup[i] = block_offsets[lookup[i]];
  }
}

__global__ void finalize_dimension_metadata(
  const CoordDim* unique_coord_dims,
  const unsigned int* block_offsets,
  unsigned int* unique_dims_outer_indices,
  unsigned int* unique_dims_block_counts,
  unsigned int num_unique_coords,
  unsigned int num_unique_dims
) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (unsigned int dimension = 0; dimension < num_unique_dims; dimension++) {
      const unsigned int start = unique_dims_block_counts[dimension];
      const unsigned int end = dimension + 1 < num_unique_dims
        ? unique_dims_block_counts[dimension + 1]
        : num_unique_coords;
      unique_dims_block_counts[dimension] = end - start;
    }
    const CoordDim last = unique_coord_dims[num_unique_coords - 1];
    unique_dims_outer_indices[num_unique_dims] = (
      block_offsets[num_unique_coords - 1]
      + static_cast<unsigned int>(last.h) * last.w
    );
  }
}


extern "C"
int compress_unique_coords(
  unsigned int* coords,
  const unsigned short int* dims,
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
  (void)dims;
  unsigned int* dimension_ordinals = coords;
  unsigned int* block_offsets = coords + NUM_COORDINATES;

  prepare_unique_metadata<<<(num_unique_coords + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    unique_coords_out,
    block_offsets,
    dimension_ordinals,
    num_unique_coords
  );
  thrust::exclusive_scan(
    thrust::device,
    block_offsets,
    block_offsets + num_unique_coords,
    block_offsets
  );
  thrust::inclusive_scan(
    thrust::device,
    dimension_ordinals,
    dimension_ordinals + num_unique_coords,
    dimension_ordinals
  );
  num_unique_dims = thrust::device_pointer_cast(dimension_ordinals)[
    num_unique_coords - 1
  ];

  write_dimension_metadata<<<(num_unique_coords + 255) / 256, 256>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    block_offsets,
    dimension_ordinals,
    unique_dims_out,
    unique_dims_outer_indices,
    unique_dims_block_counts,
    num_unique_coords
  );
  map_lookup_to_data_offsets<<<(NUM_COORDINATES + 255) / 256, 256>>>(
    lookup,
    block_offsets,
    NUM_COORDINATES
  );
  finalize_dimension_metadata<<<1, 1>>>(
    uncompressedCoordinatesAndDimensionsTmp,
    block_offsets,
    unique_dims_outer_indices,
    unique_dims_block_counts,
    num_unique_coords,
    num_unique_dims
  );
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
    return -1;
  }
  return 0;
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
  def __init__(
    self,
    coordinates: List[gpuarray.GPUArray],
    dimensions: List[gpuarray.GPUArray],
    num_coordinates: List[int],
    wrt: List[attribute],
    column_wrt: Optional[List[attribute]] = None
  ):
    self.__num_coordinates : List[int] = []
    self.__coordinates: List[gpuarray.GPUArray] = []
    self.__dimensions: List[gpuarray.GPUArray] = []
    # we only care about the ones with coordinates
    for i in range(len(num_coordinates)):
      num_coordinate = num_coordinates[i]
      if num_coordinate > 0:
        self.__num_coordinates.append(num_coordinate)
        self.__coordinates.append(coordinates[i])
        self.__dimensions.append(dimensions[i])

    self.__uniqueCoordinates: gpuarray.GPUArray = gpuarray.empty(2, np.uint32)
    # temporary arrays for uncompressed coordinates and dimensions
    # used for compressing coordinates
    self.__uncompressedCoordinatesAndDimensionsTmp: gpuarray.GPUArray = gpuarray.empty(1, coord_dim_dtype)
    self.__uncompressedCoordinates = gpuarray.empty(2, np.uint32)
    self.__uncompressedDimensions = gpuarray.empty(2, np.uint16)

    self.__uniqueDimensions: gpuarray.GPUArray
    self.__uniqueDimensionsOuterIndices: gpuarray.GPUArray # for each dimension, whats the start and end position inside the data array
    self.__uniqueDimensionsBlockCounts: gpuarray.GPUArray # for each dimension, how many blocks of data are there
    self.__lookupArray: gpuarray.GPUArray = gpuarray.empty(1, np.uint32) # should have the same size as the total number of coordinates
    self.__num_unique_coords: int = 0
    self.__num_unique_dimensions: int = 0
    self.__total_coordinates: int = 0
    self.__uniqueDimensionsCPU = None
    self.__uniqueDimensionsOuterIndicesCPU = None
    self.__uniqueDimensionsBlockCountsCPU = None

    # We compute the maximum possible number of unique block dimensions.
    # Hessians pass only ``wrt``, so their row and column size sets are the
    # same. Rectangular matrices pass a separate column target list.
    row_wrt_sizes = {x.size for x in wrt}
    if column_wrt is None:
      column_wrt_sizes = row_wrt_sizes
    else:
      column_wrt_sizes = {x.size for x in column_wrt}
    largest_num_unique_dimensions = (
      len(row_wrt_sizes) * len(column_wrt_sizes)
    )
    # print(f"largest_num_unique_dimensions: {largest_num_unique_dimensions}")
    self.__uniqueDimensions = gpuarray.zeros(largest_num_unique_dimensions * 2, np.uint16)
    self.__uniqueDimensionsOuterIndices = gpuarray.zeros(largest_num_unique_dimensions + 1, np.uint32)
    self.__uniqueDimensionsBlockCounts = gpuarray.zeros(largest_num_unique_dimensions, np.uint32)

    # kernels
    self.__get_unique_coords_kernel = None # the kernel that gets the unique coordinates as well as the unique number of coordinates
    self.__compress_unique_coords_kernel = None # the kernel that compresses the unique coordinates and check the position in the actual data
    # invoke functions
    self.__context = context()

  def updateCoordinates(self, coordinates: List[gpuarray.GPUArray], dimensions: List[gpuarray.GPUArray], num_coordinates: List[int]):
    self.__num_coordinates = []
    self.__coordinates = []
    self.__dimensions = []
    # we only care about the ones with coordinates
    for i in range(len(num_coordinates)):
      num_coordinate = num_coordinates[i]
      if num_coordinate > 0:
        self.__num_coordinates.append(num_coordinate)
        self.__coordinates.append(coordinates[i])
        self.__dimensions.append(dimensions[i])


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
  def uniqueDimensionsCPU(self):
    if self.__num_unique_dimensions == 0:
      return np.empty(0, dtype=np.uint16)
    if self.__uniqueDimensionsCPU is None:
      active_size = self.__num_unique_dimensions * 2
      self.__uniqueDimensionsCPU = self.__uniqueDimensions[:active_size].get()
      self.__uniqueDimensionsCPU.setflags(write=False)
    return self.__uniqueDimensionsCPU

  @property
  def uniqueDimensionsOuterIndicesCPU(self):
    if self.__num_unique_dimensions == 0:
      return np.zeros(1, dtype=np.uint32)
    if self.__uniqueDimensionsOuterIndicesCPU is None:
      active_size = self.__num_unique_dimensions + 1
      self.__uniqueDimensionsOuterIndicesCPU = self.__uniqueDimensionsOuterIndices[:active_size].get()
      self.__uniqueDimensionsOuterIndicesCPU.setflags(write=False)
    return self.__uniqueDimensionsOuterIndicesCPU

  @property
  def uniqueDimensionsBlockCountsCPU(self):
    if self.__num_unique_dimensions == 0:
      return np.empty(0, dtype=np.uint32)
    if self.__uniqueDimensionsBlockCountsCPU is None:
      active_size = self.__num_unique_dimensions
      self.__uniqueDimensionsBlockCountsCPU = self.__uniqueDimensionsBlockCounts[:active_size].get()
      self.__uniqueDimensionsBlockCountsCPU.setflags(write=False)
    return self.__uniqueDimensionsBlockCountsCPU

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
    if self.__total_coordinates == 0:
      return 0
    return self.__num_unique_coords

  @property
  def numUniqueDimensions(self):
    if self.__total_coordinates == 0:
      return 0
    return self.__num_unique_dimensions

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @property
  def totalBlockSize(self):
    if self.__total_coordinates == 0:
      return 0
    return int(self.uniqueDimensionsOuterIndicesCPU[-1])

  @timed("coordinateCompressionKernel.__getUniqueCoordinatesAndDimensions")
  def __getUniqueCoordinatesAndDimensions(self, uncompressedCoordinates, uncompressedDimensions, total_coordinates, uncompressedCoordinatesAndDimensionsTmp):
    # in this function we will get the unqique coordinates in the uncompressedCoordinatesAndDimensionsTmp
    # we will later on use this array in another function to allocate the space for actual unique coordinates
    # as well as determining the number of blocks for each dimension
    # now we check if kernel exists
    if self.__get_unique_coords_kernel is None:
      file_name = ".yasps_constant/get_unique_coords_kernel_scan_v1"
      # check if the file exists
      if not os.path.exists(f'{file_name}.so'):
        time_start = time.time()
        # generate the kernel
        f = open(f"{file_name}.cu", 'w')
        f.write(get_unique_coords_kernel_string)
        f.close()
        os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_89 -lcudart -lcuda")
        time_end = time.time()
        print(f"Time taken to compile get_unique_coords kernel: {(time_end - time_start) * 1000.0} ms")
        self.__get_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").get_unique_coords # get the compiled kernel
        self.__get_unique_coords_kernel.restype = ctypes.c_int # set the return type to None
        self.__get_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] * 2 + [ctypes.POINTER(ctypes.c_uint32)]
      else:
        self.__get_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").get_unique_coords # get the compiled kernel
        self.__get_unique_coords_kernel.restype = ctypes.c_int # set the return type to None
        self.__get_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] * 2 + [ctypes.POINTER(ctypes.c_uint32)]

    # we have confirmed the kernel is not none
    assert self.__get_unique_coords_kernel is not None
    # call the kernel
    num_unique_coords = ctypes.c_uint32(0)
    error_code = self.__get_unique_coords_kernel(
      self.__to_void_p(uncompressedCoordinates),
      self.__to_void_p(uncompressedDimensions),
      ctypes.c_uint32(total_coordinates),
      self.__to_void_p(uncompressedCoordinatesAndDimensionsTmp),
      self.__to_void_p(self.__lookupArray),
      ctypes.byref(num_unique_coords)
    )
    if error_code != 0:
      raise RuntimeError(f"coordinateCompressionKernel._getUniqueCoordinatesAndDimensions: Error in get_unique_coords kernel: {error_code}")
    # here we will get the unique number of coordinates
    self.__num_unique_coords = num_unique_coords.value
    # print(f"Number of unique coordinates: {self.__num_unique_coords}")

  @timed("coordinateCompressionKernel.__getUniqueCoordinatesStartAndEnd")
  def __getUniqueCoordinatesStartAndEnd(self, uncompressedCoordinates, uncompressedDimensions, total_coordinates, uncompressedCoordinatesAndDimensionsTmp):
    # now we allocate two arrays for unique dimensions and unique coordinates
    if self.__num_unique_coords * 2 > self.__uniqueCoordinates.size:
      self.__uniqueCoordinates = gpuarray.empty(int(self.__num_unique_coords * 3), dtype=np.uint32)
    if self.__compress_unique_coords_kernel is None:
      file_name = ".yasps_constant/compress_unique_coords_kernel_scan_v1"
      if not os.path.exists(f'{file_name}.so'):
        # generate the kernel
        time_start = time.time()
        f = open(f"{file_name}.cu", 'w')
        f.write(compress_unique_coords_kernel_string)
        f.close()
        os.system(f"nvcc -Xcompiler -fPIC -shared -o {file_name}.so {file_name}.cu -O3 -arch=sm_89 -lcudart -lcuda --extended-lambda")
        time_end = time.time()
        print(f"Time taken to compile compress_unique_coords kernel: {(time_end - time_start) * 1000.0} ms")
        self.__compress_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").compress_unique_coords # get the compiled kernel
        self.__compress_unique_coords_kernel.restype = ctypes.c_int # set the return type to None
        self.__compress_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] + [ctypes.c_uint32] +[ctypes.c_void_p] * 5 + [ctypes.POINTER(ctypes.c_uint32)] * 1
      else:
        self.__compress_unique_coords_kernel = ctypes.CDLL(f"{file_name}.so").compress_unique_coords # get the compiled kernel
        self.__compress_unique_coords_kernel.restype = ctypes.c_int # set the return type to None
        self.__compress_unique_coords_kernel.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] + [ctypes.c_uint32] +[ctypes.c_void_p] * 5 + [ctypes.POINTER(ctypes.c_uint32)] * 1
    assert self.__compress_unique_coords_kernel is not None


    num_unique_dims = ctypes.c_uint32(0)
    error_code = self.__compress_unique_coords_kernel(
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
    if error_code != 0:
      raise RuntimeError(f"coordinateCompressionKernel.__getUniqueCoordinatesStartAndEnd: Error in compress unique coords kernel: {error_code}")
    self.__num_unique_dimensions = num_unique_dims.value



  @timed("coordinateCompressionKernel.__compressCoordinatesAndDimensions")
  def __compressCoordinatesAndDimensions(self):
    # we first compy all the coordinates and dimensions into the uncompressed array
    count = 0
    for i in range(len(self.__num_coordinates)):
      num_coordinate = self.__num_coordinates[i]
      # copy coordinates and dimensions into the uncompressed array
      gpu_copy_slice(self.__uncompressedCoordinates, count, self.__coordinates[i], num_coordinate * 2)
      gpu_copy_slice(self.__uncompressedDimensions, count, self.__dimensions[i], num_coordinate * 2)
      count += num_coordinate * 2

    # this will put the unique coordinates in the uncompressedCoordinatesAndDimensionsTmp
    # we will then allocate space to copy the actual unique coordinates, and also compute the unique dimensions
    self.__getUniqueCoordinatesAndDimensions(self.__uncompressedCoordinates, self.__uncompressedDimensions, self.__total_coordinates, self.__uncompressedCoordinatesAndDimensionsTmp)
    # now we get the compressed lookup table and the unique coordinates
    self.__getUniqueCoordinatesStartAndEnd(self.__uncompressedCoordinates, self.__uncompressedDimensions, self.__total_coordinates, self.__uncompressedCoordinatesAndDimensionsTmp)

  @timed("coordinateCompressionKernel.compressCoordinatesAndDimensions")
  def compressCoordinatesAndDimensions(self):
    # first we check if we need to reallocate space
    self.__total_coordinates = sum(self.__num_coordinates)
    self.__num_unique_coords = 0
    self.__num_unique_dimensions = 0
    self.__uniqueDimensionsCPU = None
    self.__uniqueDimensionsOuterIndicesCPU = None
    self.__uniqueDimensionsBlockCountsCPU = None
    if self.__total_coordinates == 0:
      return # nothing we need to do
    # allocate space if needed
    if self.__total_coordinates > self.__lookupArray.size:
      self.__uncompressedCoordinatesAndDimensionsTmp: gpuarray.GPUArray = gpuarray.empty(self.__total_coordinates, coord_dim_dtype)
      self.__uncompressedCoordinates = gpuarray.empty(self.__total_coordinates * 2, np.uint32)
      self.__uncompressedDimensions = gpuarray.empty(self.__total_coordinates * 2, np.uint16)
      self.__lookupArray = gpuarray.empty(self.__total_coordinates, np.uint32)
    self.__context.useDefaultContext()
    self.__compressCoordinatesAndDimensions()

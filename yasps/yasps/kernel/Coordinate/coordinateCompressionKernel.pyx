from __future__ import annotations
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import pycuda.gpuarray as gpuarray
from yasps.context import context
from yasps.helper import timed

grouped_coordinates_source = r'''
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

using Key = unsigned long long;
static thread_local std::string last_error;

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Reuse CUB scratch; free the old allocation before growing it.
struct Buffer {
  void* data = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (data) cudaFree(data); }
  void reserve(size_t bytes) {
    if (bytes <= capacity) return;
    if (data) check(cudaFree(data));
    data = nullptr;
    capacity = 0;
    check(cudaMalloc(&data, bytes));
    capacity = bytes;
  }
};

// A logical concatenation: only these small pointer/offset tables are copied.
// Multiple energy inputs therefore need ONE selection scan per dimension.
struct Sources {
  const unsigned int* direct_coordinates;
  const unsigned short* direct_dimensions;
  const unsigned int* const* coordinates;
  const unsigned short* const* dimensions;
  const unsigned int* offsets;
  unsigned int count;
  __host__ __device__ unsigned int locate(unsigned int& i) const {
    if (count == 1) return 0;
    unsigned int lo = 0, hi = count;
    while (lo + 1 < hi) {
      unsigned int mid = lo + (hi - lo) / 2;
      if (offsets[mid] <= i) lo = mid; else hi = mid;
    }
    i -= offsets[lo];
    return lo;
  }
};

struct CoordinateKey {
  Sources sources;
  unsigned int index_bits;
  __host__ __device__ Key operator()(unsigned int i) const {
    unsigned int source = sources.locate(i);
    auto c = sources.count == 1 ? sources.direct_coordinates : sources.coordinates[source];
    return (Key(c[2ull * i]) << index_bits) | c[2ull * i + 1];
  }
};

struct CoordinateExtent {
  Sources sources;
  __host__ __device__ unsigned int operator()(unsigned int i) const {
    unsigned int source = sources.locate(i);
    auto c = sources.count == 1 ? sources.direct_coordinates : sources.coordinates[source];
    return max(c[2ull * i], c[2ull * i + 1]);
  }
};

struct MatchesDimension {
  Sources sources;
  unsigned int dimension;
  __host__ __device__ bool operator()(unsigned int i) const {
    unsigned int source = sources.locate(i);
    auto d = sources.count == 1 ? sources.direct_dimensions : sources.dimensions[source];
    return ((unsigned(d[2ull * i]) << 16) | d[2ull * i + 1]) == dimension;
  }
};

// Each occurrence searches only its own dimension group. Return a scalar
// VALUE offset, not a coordinate ordinal. No original-index sort payload.
__global__ void make_lookup(const unsigned int* coordinates, const unsigned short* dimensions, unsigned int n,
  const Key* unique_keys, const unsigned short* unique_dimensions, const unsigned int* coordinate_outer,
  const unsigned int* value_outer, unsigned int num_dimensions, unsigned int index_bits, unsigned int* lookup) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int h = dimensions[2ull * i], w = dimensions[2ull * i + 1];
  unsigned int dimension = (h << 16) | w;
  unsigned int lo = 0, hi = num_dimensions;
  while (lo < hi) {
    unsigned int mid = lo + (hi - lo) / 2;
    unsigned int key = (unsigned(unique_dimensions[2 * mid]) << 16) | unique_dimensions[2 * mid + 1];
    if (key < dimension) lo = mid + 1; else hi = mid;
  }
  unsigned int group = lo, start = coordinate_outer[group];
  Key key = (Key(coordinates[2ull * i]) << index_bits) | coordinates[2ull * i + 1];
  lo = start;
  hi = coordinate_outer[group + 1];
  while (lo < hi) {
    unsigned int mid = lo + (hi - lo) / 2;
    if (unique_keys[mid] < key) lo = mid + 1; else hi = mid;
  }
  lookup[i] = value_outer[group] + (lo - start) * h * w;
}

__global__ void unpack_coordinates(const Key* keys, unsigned int* coordinates, unsigned int n, unsigned int index_bits) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    coordinates[2ull * i] = unsigned(keys[i] >> index_bits);
    coordinates[2ull * i + 1] = unsigned(keys[i] & ((Key(1) << index_bits) - 1));
  }
}

extern "C" const char* coordinate_compression_error() { return last_error.c_str(); }

// Pointer/count arrays and candidate dimensions are small HOST arrays.
// Raw coordinates/dimensions remain in the original GPU producer buffers.
// Selection uses a parallel scan; radix sorting only sees 64-bit (row,col) keys.
extern "C" int get_unique_grouped_coordinates(const unsigned int* const* coordinates, const unsigned short* const* dimensions,
  const unsigned int* counts, unsigned int num_sources, const unsigned int* candidates, unsigned int num_candidates,
  Key* keys, unsigned short* unique_dimensions, unsigned int* value_outer, unsigned int* block_counts,
  unsigned int* coordinate_outer, unsigned int& num_unique, unsigned int& num_dimensions, unsigned int& index_bits) {
  try {
    Buffer scratch, count_buffer, alternate, source_table;
    count_buffer.reserve(sizeof(unsigned int));
    auto selected_count = static_cast<unsigned int*>(count_buffer.data);
    auto indices = thrust::make_counting_iterator<unsigned int>(0);
    std::vector<unsigned int> active, raw_counts;
    size_t raw_offset = 0, expected = 0, largest = 0;
    for (unsigned int source = 0; source < num_sources; ++source) expected += counts[source];
    Sources sources{coordinates[0], dimensions[0], nullptr, nullptr, nullptr, num_sources};
    if (num_sources > 1) {
      size_t pointer_bytes = num_sources * sizeof(void*);
      source_table.reserve(2 * pointer_bytes + (num_sources + 1) * sizeof(unsigned int));
      auto table = static_cast<char*>(source_table.data);
      sources.coordinates = reinterpret_cast<const unsigned int* const*>(table);
      sources.dimensions = reinterpret_cast<const unsigned short* const*>(table + pointer_bytes);
      sources.offsets = reinterpret_cast<const unsigned int*>(table + 2 * pointer_bytes);
      std::vector<unsigned int> offsets{0};
      for (unsigned int source = 0; source < num_sources; ++source) offsets.push_back(offsets.back() + counts[source]);
      check(cudaMemcpy(table, coordinates, pointer_bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpy(table + pointer_bytes, dimensions, pointer_bytes, cudaMemcpyHostToDevice));
      check(cudaMemcpy(table + 2 * pointer_bytes, offsets.data(), offsets.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    // Compact row/column bits without changing their lexicographic order.
    // Smaller global indices then need fewer radix passes, while uint32
    // coordinates (including the full 32-bit range) retain the same API.
    auto extent = thrust::make_transform_iterator(indices, CoordinateExtent{sources});
    size_t extent_bytes = 0;
    check(cub::DeviceReduce::Max(nullptr, extent_bytes, extent, selected_count, expected));
    scratch.reserve(extent_bytes);
    check(cub::DeviceReduce::Max(scratch.data, extent_bytes, extent, selected_count, expected));
    unsigned int maximum = 0;
    check(cudaMemcpy(&maximum, selected_count, sizeof(maximum), cudaMemcpyDeviceToHost));
    index_bits = 32 - __builtin_clz(maximum | 1u);
    auto input = thrust::make_transform_iterator(indices, CoordinateKey{sources, index_bits});
    for (unsigned int d = 0; d < num_candidates; ++d) {
      auto flags = thrust::make_transform_iterator(indices, MatchesDimension{sources, candidates[d]});
      size_t bytes = 0;
      check(cub::DeviceSelect::Flagged(nullptr, bytes, input, flags, keys + raw_offset, selected_count, expected));
      scratch.reserve(bytes);
      check(cub::DeviceSelect::Flagged(scratch.data, bytes, input, flags, keys + raw_offset, selected_count, expected));
      unsigned int group_count = 0;
      check(cudaMemcpy(&group_count, selected_count, sizeof(group_count), cudaMemcpyDeviceToHost));
      raw_offset += group_count;
      if (group_count) {
        active.push_back(candidates[d]);
        raw_counts.push_back(group_count);
        largest = std::max(largest, size_t(group_count));
      }
    }
    if (raw_offset != expected) throw std::runtime_error("Block dimension is absent from differentiation target sizes");
    alternate.reserve(largest * sizeof(Key));
    auto other = static_cast<Key*>(alternate.data);
    std::vector<unsigned short> dims;
    std::vector<unsigned int> outer{0}, data_outer{0}, unique_counts;
    raw_offset = 0;
    size_t unique_offset = 0, data_offset = 0;
    for (unsigned int d = 0; d < active.size(); ++d) {
      unsigned int count = raw_counts[d];
      cub::DoubleBuffer<Key> buffers(keys + raw_offset, other);
      size_t bytes = 0;
      check(cub::DeviceRadixSort::SortKeys(nullptr, bytes, buffers, count, 0, 2 * index_bits));
      scratch.reserve(bytes);
      check(cub::DeviceRadixSort::SortKeys(scratch.data, bytes, buffers, count, 0, 2 * index_bits));
      // Unique's input/output are distinct. Compacting the current group's
      // prefix cannot overwrite any later group's raw coordinates.
      Key* sorted = buffers.Current();
      Key* compacted = sorted == other ? keys + unique_offset : other;
      bytes = 0;
      check(cub::DeviceSelect::Unique(nullptr, bytes, sorted, compacted, selected_count, count));
      scratch.reserve(bytes);
      check(cub::DeviceSelect::Unique(scratch.data, bytes, sorted, compacted, selected_count, count));
      unsigned int unique = 0;
      check(cudaMemcpy(&unique, selected_count, sizeof(unique), cudaMemcpyDeviceToHost));
      if (compacted != keys + unique_offset) check(cudaMemcpy(keys + unique_offset, compacted, size_t(unique) * sizeof(Key), cudaMemcpyDeviceToDevice));
      unsigned int h = active[d] >> 16, w = active[d] & 65535;
      data_offset += size_t(unique) * h * w;
      if (data_offset > UINT_MAX) throw std::runtime_error("Compressed scalar offsets exceed uint32 storage");
      unique_offset += unique;
      dims.push_back(h);
      dims.push_back(w);
      unique_counts.push_back(unique);
      outer.push_back(unique_offset);
      data_outer.push_back(data_offset);
      raw_offset += count;
    }
    num_unique = unique_offset;
    num_dimensions = active.size();
    check(cudaMemcpy(unique_dimensions, dims.data(), dims.size() * sizeof(unsigned short), cudaMemcpyHostToDevice));
    check(cudaMemcpy(block_counts, unique_counts.data(), unique_counts.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(coordinate_outer, outer.data(), outer.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    check(cudaMemcpy(value_outer, data_outer.data(), data_outer.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    check(cudaDeviceSynchronize());
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

extern "C" int finalize_grouped_coordinates(const unsigned int* const* coordinates, const unsigned short* const* dimensions,
  const unsigned int* counts, unsigned int num_sources, const Key* keys, const unsigned short* unique_dimensions,
  const unsigned int* value_outer, const unsigned int* coordinate_outer, unsigned int num_dimensions,
  unsigned int num_unique, unsigned int index_bits, unsigned int* unique_coordinates, unsigned int* lookup) {
  try {
    size_t offset = 0;
    for (unsigned int source = 0; source < num_sources; ++source) {
      make_lookup<<<(counts[source] + 255) / 256, 256>>>(coordinates[source], dimensions[source], counts[source], keys,
        unique_dimensions, coordinate_outer, value_outer, num_dimensions, index_bits, lookup + offset);
      offset += counts[source];
    }
    unpack_coordinates<<<(num_unique + 255) / 256, 256>>>(keys, unique_coordinates, num_unique, index_bits);
    check(cudaDeviceSynchronize());
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}
'''

_libraries = {}


class coordinateCompressionKernel:
  """Dimension-grouped coordinate compression with scalar-offset lookups.

  Select directly from producer arrays; radix-sort 64-bit coordinate keys with
  scratch sized to the largest dimension group. Retain only unique coordinates,
  dimension metadata and original-occurrence-to-value-offset lookups.
  """
  def __init__(self, coordinates, dimensions, num_coordinates, wrt, column_wrt=None):
    row_sizes = sorted({x.size for x in wrt})
    column_sizes = row_sizes if column_wrt is None else sorted({x.size for x in column_wrt})
    if any(size <= 0 or size > 65535 for size in row_sizes + column_sizes):
      raise ValueError("Block dimensions must fit positive uint16 values")
    self.__candidates = np.array([(h << 16) | w for h in row_sizes for w in column_sizes], dtype=np.uint32)
    capacity = len(self.__candidates)
    # Keep a valid pointer for downstream GPUArray[:0] views on empty scenes.
    self.__uniqueCoordinates = gpuarray.empty(2, np.uint32)
    self.__uniqueDimensions = gpuarray.zeros(2 * capacity, np.uint16)
    self.__uniqueDimensionsOuterIndices = gpuarray.zeros(capacity + 1, np.uint32)
    self.__uniqueDimensionsBlockCounts = gpuarray.zeros(capacity, np.uint32)
    self.__coordinateOuterIndices = gpuarray.zeros(capacity + 1, np.uint32)
    self.__lookupArray = gpuarray.empty(1, np.uint32)
    self.__num_unique_coords = self.__num_unique_dimensions = self.__total_coordinates = 0
    self.__uniqueDimensionsCPU = self.__uniqueDimensionsOuterIndicesCPU = self.__uniqueDimensionsBlockCountsCPU = None
    self.__context = context()
    self.updateCoordinates(coordinates, dimensions, num_coordinates)

  def updateCoordinates(self, coordinates, dimensions, num_coordinates):
    if len(coordinates) != len(dimensions) or len(coordinates) != len(num_coordinates):
      raise ValueError("Coordinate, dimension, and count lists must have matching lengths")
    self.__coordinates, self.__dimensions, self.__num_coordinates = [], [], []
    for coords, dims, count in zip(coordinates, dimensions, num_coordinates):
      if count < 0 or coords.size < 2 * count or dims.size < 2 * count:
        raise ValueError("Invalid coordinate count or undersized input buffer")
      if coords.dtype != np.uint32 or dims.dtype != np.uint16:
        raise ValueError("Coordinates must be uint32 and dimensions uint16")
      if count:
        self.__coordinates.append(coords)
        self.__dimensions.append(dims)
        self.__num_coordinates.append(int(count))

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

  @property
  def totalBlockSize(self):
    if self.__total_coordinates == 0:
      return 0
    return int(self.uniqueDimensionsOuterIndicesCPU[-1])

  def __loadLibrary(self):
    digest = hashlib.sha256(grouped_coordinates_source.encode()).hexdigest()[:16]
    path = Path('.yasps_constant').resolve() / f'grouped_coordinates_{digest}'
    key = str(path)
    if key not in _libraries:
      path.parent.mkdir(parents=True, exist_ok=True)
      if not path.with_suffix('.so').exists():
        path.with_suffix('.cu').write_text(grouped_coordinates_source)
        subprocess.run(['nvcc', '-Xcompiler', '-fPIC', '-shared', '-O3', '-arch=sm_89', str(path.with_suffix('.cu')), '-o', str(path.with_suffix('.so'))], check=True)
      library = ctypes.CDLL(str(path.with_suffix('.so')))
      pointer, uint = ctypes.c_void_p, ctypes.c_uint32
      library.get_unique_grouped_coordinates.argtypes = [pointer] * 3 + [uint, pointer, uint] + [pointer] * 5 + [ctypes.POINTER(uint)] * 3
      library.get_unique_grouped_coordinates.restype = ctypes.c_int
      library.finalize_grouped_coordinates.argtypes = [pointer] * 3 + [uint] + [pointer] * 4 + [uint] * 3 + [pointer] * 2
      library.finalize_grouped_coordinates.restype = ctypes.c_int
      library.coordinate_compression_error.restype = ctypes.c_char_p
      _libraries[key] = library
    return _libraries[key]

  @timed('coordinateCompressionKernel.compressCoordinatesAndDimensions')
  def compressCoordinatesAndDimensions(self):
    self.__total_coordinates = sum(self.__num_coordinates)
    self.__num_unique_coords = self.__num_unique_dimensions = 0
    self.__uniqueDimensionsCPU = self.__uniqueDimensionsOuterIndicesCPU = self.__uniqueDimensionsBlockCountsCPU = None
    self.__context.useDefaultContext()
    self.__uniqueDimensions.fill(0)
    self.__uniqueDimensionsOuterIndices.fill(0)
    self.__uniqueDimensionsBlockCounts.fill(0)
    if not self.__total_coordinates:
      self.__uniqueCoordinates = gpuarray.empty(2, np.uint32)
      self.__lookupArray = gpuarray.empty(1, np.uint32)
      return
    if not self.__coordinates:
      raise ValueError('Raw coordinates were released after compression; call updateCoordinates before recompressing.')
    if self.__total_coordinates > np.iinfo(np.uint32).max:
      raise ValueError('Coordinate count exceeds uint32 storage')
    library = self.__loadLibrary()
    coordinates = np.array([int(x.gpudata) for x in self.__coordinates], dtype=np.uintp)
    dimensions = np.array([int(x.gpudata) for x in self.__dimensions], dtype=np.uintp)
    counts = np.array(self.__num_coordinates, dtype=np.uint32)
    # Do not retain the raw-sized buffer when millions of duplicates collapse.
    keys = gpuarray.empty(self.__total_coordinates, np.uint64)
    unique, dimension_count, index_bits = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    status = library.get_unique_grouped_coordinates(coordinates.ctypes.data, dimensions.ctypes.data, counts.ctypes.data, len(counts), self.__candidates.ctypes.data, len(self.__candidates), int(keys.gpudata), int(self.__uniqueDimensions.gpudata), int(self.__uniqueDimensionsOuterIndices.gpudata), int(self.__uniqueDimensionsBlockCounts.gpudata), int(self.__coordinateOuterIndices.gpudata), ctypes.byref(unique), ctypes.byref(dimension_count), ctypes.byref(index_bits))
    if status:
      raise RuntimeError('Coordinate compression: ' + library.coordinate_compression_error().decode())
    self.__num_unique_coords, self.__num_unique_dimensions = unique.value, dimension_count.value
    if self.__uniqueCoordinates.size < 2 * unique.value:
      self.__uniqueCoordinates = gpuarray.empty(2 * unique.value, np.uint32)
    if self.__lookupArray.size < self.__total_coordinates:
      self.__lookupArray = gpuarray.empty(self.__total_coordinates, np.uint32)
    status = library.finalize_grouped_coordinates(coordinates.ctypes.data, dimensions.ctypes.data, counts.ctypes.data, len(counts), int(keys.gpudata), int(self.__uniqueDimensions.gpudata), int(self.__uniqueDimensionsOuterIndices.gpudata), int(self.__coordinateOuterIndices.gpudata), dimension_count.value, unique.value, index_bits.value, int(self.__uniqueCoordinates.gpudata), int(self.__lookupArray.gpudata))
    if status:
      raise RuntimeError('Coordinate lookup generation: ' + library.coordinate_compression_error().decode())
    self.__coordinates = []
    self.__dimensions = []

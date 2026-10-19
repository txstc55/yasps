"""Second pass for coordinate-grouped, fully materialized local Hessian blocks.

The first pass preserves every instance's projection and J^T H J arithmetic.
Only its destination changes: all members add to their group's block buffer.
This pass adds each independent block once to the usual sparse representation.
Different groups may overlap, so global writes must still be atomic.
"""
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


class reducedHessian:
  def __init__(self, max_indices, num_attributes, scalar_diagonal, separated):
    self.values = gpuarray.empty(0, np.float64)
    source = r'''
    __global__ void scatter_groups(
        const unsigned int* indices, const unsigned short* sizes,
        const short* permutations, const unsigned int* outer,
        const unsigned int* local_lookups, const unsigned int* global_lookups,
        const double* values, double* blocks, double* diagonal,
        double* diagonal_blocks, const unsigned int* diagonal_starts,
        const unsigned int* gradient_starts, unsigned int group_count) {
      const unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
      if (group >= group_count) return;
      const size_t base = (size_t)group * MAX_INDICES;
      unsigned int pair = outer[group];
      for (unsigned int i = 0; i < MAX_INDICES; ++i) {
        const unsigned int index_i = indices[base + i];
        if (permutations[base + i] <= 0 || index_i < 2) continue;
        const unsigned int size_i = sizes[base + i];
        for (unsigned int j = i; j < MAX_INDICES; ++j) {
          const unsigned int index_j = indices[base + j];
          if (permutations[base + j] <= 0 || index_j < 2) continue;
          const unsigned int size_j = sizes[base + j];
          const unsigned int rows = index_i < index_j ? size_i : size_j;
          const unsigned int cols = index_i < index_j ? size_j : size_i;
          const unsigned int local = local_lookups[pair];
          const unsigned int global = global_lookups[pair++];
          const bool same_target = index_i == index_j;
          const bool write_diagonal = same_target && (SEPARATED || i == j);
          unsigned int diagonal_start = 0;
          if (write_diagonal) {
            const unsigned int scalar_start = index_i - 2;
            unsigned int attribute = 0;
            while (attribute + 1 < NUM_ATTRIBUTES &&
                   scalar_start >= gradient_starts[attribute + 1]) ++attribute;
            const unsigned int instance =
                (scalar_start - gradient_starts[attribute]) / size_i;
            diagonal_start = diagonal_starts[attribute] + instance * size_i * size_i;
          }
          for (unsigned int row = 0; row < rows; ++row) {
            for (unsigned int col = 0; col < cols; ++col) {
              const unsigned int offset = row * cols + col;
              const double value = values[local + offset];
              atomicAdd(blocks + global + offset, value);
              if (write_diagonal) {
                atomicAdd(diagonal_blocks + diagonal_start + offset, value);
                if (SCALAR_DIAGONAL && row == col)
                  atomicAdd(diagonal + index_i - 2 + row, value);
              }
            }
          }
        }
      }
    }
    '''
    module = SourceModule(source, options=[
        f"-DMAX_INDICES={int(max_indices)}", f"-DNUM_ATTRIBUTES={int(num_attributes)}",
        f"-DSCALAR_DIAGONAL={int(scalar_diagonal)}", f"-DSEPARATED={int(separated)}"], no_extern_c=False)
    self.scatter = module.get_function("scatter_groups")

  @property
  def workspaceBytes(self):
    return self.values.nbytes

  def clear(self, indices):
    count = indices.coordinateGroupStorageSize
    if self.values.size < count:
      self.values = gpuarray.empty(count, np.float64)
    if count:
      self.values[:count].fill(0)
    # The producer and scatter both use the default stream, as does fill.

  def compute(self, indices, lookups, blocks, diagonal, diagonal_blocks,
              diagonal_starts, gradient_starts):
    count = indices.numCoordinateGroups
    if not count or not indices.numTotalCoordinates:
      return
    self.scatter(indices.outputIndices, indices.outputSizes, indices.outputPermutations,
        indices.outputCompressedCoordinateCountsOuter,
        indices.coordinateGroupLocalLookups, lookups, self.values,
        blocks, diagonal, diagonal_blocks, diagonal_starts, gradient_starts,
        np.uint32(count), block=(32, 1, 1), grid=((count + 31) // 32, 1, 1))
    cuda.Context.synchronize()

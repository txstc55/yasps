#include <cstdint>

static __device__ double yasps_mas_atomic_add(double* address, double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits, assumed;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

template <unsigned int ROWS, unsigned int COLS>
__device__ __forceinline__ void yasps_mas_assemble_static_block(
    unsigned int source_id, unsigned long long value_start,
    unsigned long long position_start, const double* values,
    const unsigned int* positions,
    const unsigned int* scalar_boundary_to_node,
    unsigned int fine_scalar_dofs, const unsigned int* fine_node_domains,
    const unsigned int* fine_node_local_offsets,
    unsigned int level_count, unsigned int fine_node_count,
    const unsigned long long* matrix_offsets,
    const unsigned int* padded_sizes, double* matrices, int* status) {
  const unsigned long long position_id = position_start + source_id;
  const unsigned int scalar_row = positions[2 * position_id];
  const unsigned int scalar_col = positions[2 * position_id + 1];
  if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
    atomicExch(status, 1);
    return;
  }
  const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
  const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
  if (fine_row == 0xffffffffu || fine_col == 0xffffffffu) {
    atomicExch(status, 1);
    return;
  }
  const double* source = values + value_start
      + static_cast<unsigned long long>(source_id) * ROWS * COLS;
  bool present_in_previous_level = false;
  for (unsigned int level = 0; level < level_count; ++level) {
    const unsigned int map_row = level * fine_node_count + fine_row;
    const unsigned int map_col = level * fine_node_count + fine_col;
    const unsigned int domain = fine_node_domains[map_row];
    const bool internal = domain == fine_node_domains[map_col];
    if (!internal || present_in_previous_level) {
      present_in_previous_level = internal;
      continue;
    }
    const unsigned int local_row = fine_node_local_offsets[map_row];
    const unsigned int local_col = fine_node_local_offsets[map_col];
    const unsigned int padded = padded_sizes[domain];
    double* matrix = matrices + matrix_offsets[domain];
#pragma unroll
    for (unsigned int row = 0; row < ROWS; ++row) {
#pragma unroll
      for (unsigned int col = 0; col < COLS; ++col) {
        const double value = __ldg(source + row * COLS + col);
        unsigned int target_row = local_row + row;
        unsigned int target_col = local_col + col;
#if YASPS_MAS_SYMMETRIC_STORAGE
        if (fine_row == fine_col && target_row > target_col) continue;
        if (target_row > target_col) {
          const unsigned int temporary = target_row;
          target_row = target_col;
          target_col = temporary;
        }
#endif
        yasps_mas_atomic_add(
            matrix + target_row * padded + target_col, value);
      }
    }
    present_in_previous_level = true;
  }
}

extern "C" __global__ void yasps_mas_specialized_static_assembly(
    const double* values, const unsigned int* positions,
    const unsigned int* scalar_boundary_to_node,
    unsigned int fine_scalar_dofs, const unsigned int* fine_node_domains,
    const unsigned int* fine_node_local_offsets,
    unsigned int level_count, unsigned int fine_node_count,
    const unsigned long long* matrix_offsets,
    const unsigned int* padded_sizes, double* matrices, int* status) {
  const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  // YASPS_MAS_GENERATED_STATIC_CATEGORY_DISPATCH
}

// Immutable static coordinates and hierarchy maps are resolved once during
// setup. Each block then carries the first Schwarz-bank destination at which
// its endpoints meet. Later levels are produced by the adjacent propagation
// kernels, so the numerical hot path needs no coordinate or level-map search.
template <unsigned int ROWS, unsigned int COLS>
__device__ __forceinline__ void yasps_mas_assemble_precomputed_static_block(
    unsigned int global_id, unsigned int source_id,
    unsigned long long value_start, const double* values,
    const unsigned long long* destination_offsets,
    const unsigned long long* transpose_offsets,
    const unsigned char* destination_strides, double* matrices) {
  const unsigned long long destination = destination_offsets[global_id];
  if (destination == 0xffffffffffffffffull) return;
  const unsigned long long transpose = transpose_offsets[global_id];
  const unsigned int stride = destination_strides[global_id];
  const double* source = values + value_start
      + static_cast<unsigned long long>(source_id) * ROWS * COLS;
#pragma unroll
  for (unsigned int row = 0; row < ROWS; ++row) {
#pragma unroll
    for (unsigned int col = 0; col < COLS; ++col) {
      const double value = __ldg(source + row * COLS + col);
#if YASPS_MAS_SYMMETRIC_STORAGE
      if (transpose == 0xffffffffffffffffull) {
        if (row <= col)
          yasps_mas_atomic_add(
              matrices + destination
                  + static_cast<unsigned long long>(row) * stride + col,
              value);
      } else if (destination < transpose) {
        yasps_mas_atomic_add(
            matrices + destination
                + static_cast<unsigned long long>(row) * stride + col,
            value);
      } else {
        yasps_mas_atomic_add(
            matrices + transpose
                + static_cast<unsigned long long>(col) * stride + row,
            value);
      }
#else
      yasps_mas_atomic_add(
          matrices + destination
              + static_cast<unsigned long long>(row) * stride + col,
          value);
#endif
    }
  }
}

extern "C" __global__ void yasps_mas_precomputed_static_assembly(
    const double* values,
    const unsigned long long* destination_offsets,
    const unsigned long long* transpose_offsets,
    const unsigned char* destination_strides, double* matrices) {
  const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  // YASPS_MAS_GENERATED_PRECOMPUTED_STATIC_CATEGORY_DISPATCH
}

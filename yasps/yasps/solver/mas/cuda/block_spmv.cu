#include <cstdint>
#include <cub/warp/warp_reduce.cuh>

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

template <unsigned int warps_per_block>
static __device__ __forceinline__ void yasps_mas_accumulate_quadratic_block(
    double quadratic, double* output, double* warp_sums) {
  if (!output) return;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
#pragma unroll
  for (unsigned int offset = 16; offset; offset >>= 1)
    quadratic += __shfl_down_sync(0xffffffffu, quadratic, offset);
  if (lane == 0u) warp_sums[warp] = quadratic;
  __syncthreads();
  if (warp == 0u) {
    quadratic = lane < warps_per_block ? warp_sums[lane] : 0.0;
#pragma unroll
    for (unsigned int offset = 16; offset; offset >>= 1)
      quadratic += __shfl_down_sync(0xffffffffu, quadratic, offset);
    if (lane == 0u) yasps_mas_atomic_add(output, quadratic);
  }
}

// Launch once per shape category. Positions contain scalar row/column starts.
extern "C" __global__ void yasps_mas_block_spmv(
    const double* values, const std::uint32_t* positions,
    std::uint64_t value_start, std::uint64_t position_start,
    std::uint32_t count, std::uint32_t rows, std::uint32_t cols,
    const double* x, double* y, bool symmetric_storage) {
  const std::uint32_t block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= count) return;
  const std::uint32_t row = positions[2 * (position_start + block_id)];
  const std::uint32_t col = positions[2 * (position_start + block_id) + 1];
  const double* block = values + value_start + block_id * rows * cols;
  for (std::uint32_t local_row = 0; local_row < rows; ++local_row) {
    double sum = 0.0;
    for (std::uint32_t local_col = 0; local_col < cols; ++local_col)
      sum += block[local_row * cols + local_col] * x[col + local_col];
    yasps_mas_atomic_add(y + row + local_row, sum);
  }
  if (symmetric_storage && row != col) {
    for (std::uint32_t local_col = 0; local_col < cols; ++local_col) {
      double sum = 0.0;
      for (std::uint32_t local_row = 0; local_row < rows; ++local_row)
        sum += block[local_row * cols + local_col] * x[row + local_row];
      yasps_mas_atomic_add(y + col + local_col, sum);
    }
  }
}

// The persistent JIT compiles this section with one concrete row/column shape.
// Keeping the dimensions and symmetry mode as compile-time constants lets
// NVCC unroll the tiny dense products. One warp processes 32 Hessian blocks,
// matching YASPS's generated SpMV instead of wasting a CUDA block on each
// small matrix. Adjacent contributions with the same output row are combined
// in shared memory before reaching global atomics.
#if defined(YASPS_MAS_BLOCK_ROWS) && defined(YASPS_MAS_BLOCK_COLS) && \
    defined(YASPS_MAS_SYMMETRIC_STORAGE)
#ifndef YASPS_MAS_WARPS_PER_BLOCK
#define YASPS_MAS_WARPS_PER_BLOCK 1
#endif
#ifndef YASPS_MAS_SEGMENTED_REDUCTION
#define YASPS_MAS_SEGMENTED_REDUCTION 1
#endif
#ifndef YASPS_MAS_FUSE_AUXILIARY
#define YASPS_MAS_FUSE_AUXILIARY 0
#endif
#if defined(YASPS_MAS_MIXED_SPMV_ARITHMETIC)
using yasps_mas_product_t = float;
#else
using yasps_mas_product_t = double;
#endif

static __device__ __forceinline__ yasps_mas_product_t
yasps_mas_load_product_x(const double* address) {
  return static_cast<yasps_mas_product_t>(__ldg(address));
}

template <unsigned int rows, unsigned int cols>
static __device__ __forceinline__ double yasps_mas_process_one_auxiliary(
    const unsigned long long* descriptor, unsigned int block_id,
    const double* x, double* y) {
  const double* values = reinterpret_cast<const double*>(descriptor[0]);
  const unsigned int* positions =
      reinterpret_cast<const unsigned int*>(descriptor[1]);
  const unsigned long long value_start = descriptor[2];
  const unsigned long long position_start = descriptor[3];
  const unsigned long long position_id = position_start + block_id;
  const unsigned int row_start = positions[2 * position_id];
  const unsigned int col_start = positions[2 * position_id + 1];
  const double* block = values + value_start +
      static_cast<unsigned long long>(block_id) * rows * cols;
  double quadratic = 0.0;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) {
    yasps_mas_product_t value = 0.0f;
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col)
      value += __ldg(block + row * cols + col) *
          yasps_mas_load_product_x(x + col_start + col);
    yasps_mas_atomic_add(
        y + row_start + row, static_cast<double>(value));
    quadratic += __ldg(x + row_start + row) * static_cast<double>(value);
  }
#if YASPS_MAS_SYMMETRIC_STORAGE
  if (row_start != col_start) {
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) {
      yasps_mas_product_t value = 0.0f;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        value += __ldg(block + row * cols + col) *
            yasps_mas_load_product_x(x + row_start + row);
      yasps_mas_atomic_add(
          y + col_start + col, static_cast<double>(value));
    }
    quadratic *= 2.0;
  }
#endif
  return quadratic;
}

static __device__ __forceinline__ double yasps_mas_process_auxiliary(
    const unsigned long long* descriptors, unsigned int descriptor_count,
    unsigned int auxiliary_id, const double* x, double* y) {
  constexpr unsigned int descriptor_stride = 6;
  for (unsigned int category = 0; category < descriptor_count; ++category) {
    const unsigned long long* descriptor =
        descriptors + category * descriptor_stride;
    const unsigned int count = static_cast<unsigned int>(descriptor[4]);
    if (auxiliary_id < count) {
      switch (static_cast<unsigned int>(descriptor[5])) {
        // YASPS_MAS_AUXILIARY_SHAPE_CASES
        default: return 0.0;
      }
    }
    auxiliary_id -= count;
  }
  return 0.0;
}

extern "C" __global__ void yasps_mas_specialized_mapped_block_spmv(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_node_scalar_starts, unsigned int level_index,
    unsigned int fine_node_count, const double* x, double* y,
    double* quadratic_output) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  constexpr unsigned int warps_per_block = YASPS_MAS_WARPS_PER_BLOCK;
  const unsigned int lane = threadIdx.x & (warp_size - 1);
  const unsigned int local_warp = threadIdx.x / warp_size;
  const unsigned int source_id =
      (blockIdx.x * warps_per_block + local_warp) * warp_size + lane;
  __shared__ double quadratic_warps[warps_per_block];

  __shared__ double row_results[warps_per_block * warp_size * rows];
  __shared__ unsigned int row_starts[warps_per_block * warp_size];
  const unsigned int warp_base = local_warp * warp_size;

  unsigned int row_start = 0xffffffffu;
  unsigned int col_start = 0xffffffffu;
  double local_row_results[rows];
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    local_row_results[row] = 0.0;

  if (source_id < count) {
    const unsigned long long position_id = position_start + source_id;
    const unsigned int scalar_row = positions[2 * position_id];
    const unsigned int scalar_col = positions[2 * position_id + 1];
    if (level_index == 0) {
      row_start = scalar_row;
      col_start = scalar_col;
    } else {
      const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
      const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
      const unsigned int map_offset = level_index * fine_node_count;
      row_start = fine_node_scalar_starts[map_offset + fine_row];
      col_start = fine_node_scalar_starts[map_offset + fine_col];
    }
    const double* matrix = values + value_start +
        static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) {
      double sum = 0.0;
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        sum += __ldg(matrix + row * cols + col)
            * __ldg(x + col_start + col);
      local_row_results[row] = sum;
    }
  }

  row_starts[warp_base + lane] = row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    row_results[(warp_base + lane) * rows + row] = local_row_results[row];
  // Every reduction is confined to this warp's slice of shared memory.  Do
  // not make the other independent small-block warps wait at a block-wide
  // barrier.
  __syncwarp();

  if (source_id < count &&
      (lane == 0 || row_starts[warp_base + lane - 1] != row_start)) {
    double sums[rows];
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      sums[row] = 0.0;
    for (unsigned int other = lane;
         other < warp_size &&
         row_starts[warp_base + other] == row_start; ++other) {
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sums[row] += row_results[(warp_base + other) * rows + row];
    }
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      yasps_mas_atomic_add(y + row_start + row, sums[row]);
  }

#if YASPS_MAS_SYMMETRIC_STORAGE
  if (source_id < count) {
    const double* matrix = values + value_start +
        static_cast<unsigned long long>(source_id) * rows * cols;
    const unsigned long long position_id = position_start + source_id;
    const unsigned int scalar_row = positions[2 * position_id];
    const unsigned int scalar_col = positions[2 * position_id + 1];
    if (scalar_row != scalar_col) {
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col) {
        double sum = 0.0;
#pragma unroll
        for (unsigned int row = 0; row < rows; ++row)
          sum += __ldg(matrix + row * cols + col)
              * __ldg(x + row_start + row);
        yasps_mas_atomic_add(y + col_start + col, sum);
      }
    }
  }
#endif
  double quadratic = 0.0;
  if (source_id < count) {
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      quadratic += __ldg(x + row_start + row) * local_row_results[row];
#if YASPS_MAS_SYMMETRIC_STORAGE
    const unsigned long long position_id = position_start + source_id;
    if (positions[2 * position_id] != positions[2 * position_id + 1])
      quadratic *= 2.0;
#endif
  }
  yasps_mas_accumulate_quadratic_block<warps_per_block>(
      quadratic, quadratic_output, quadratic_warps);
}

// Fine PCG never needs a hierarchy lookup: YASPS block coordinates are
// already scalar starts in the level-zero vector.  Keep a distinct generated
// entry point so NVCC can remove the runtime level branch and every mapping
// operand from the hot SpMV path, while the mapped kernel above remains
// available for optional materialized/coarse solves.
extern "C" __global__ void yasps_mas_specialized_fine_block_spmv(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, const double* x, double* y,
    const unsigned long long* auxiliary_descriptors,
    unsigned int auxiliary_descriptor_count, unsigned int auxiliary_count,
    double* quadratic_output) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  constexpr unsigned int warps_per_block = YASPS_MAS_WARPS_PER_BLOCK;
  const unsigned int lane = threadIdx.x & (warp_size - 1);
  const unsigned int local_warp = threadIdx.x / warp_size;
  const unsigned int source_id =
      (blockIdx.x * warps_per_block + local_warp) * warp_size + lane;
  __shared__ double quadratic_warps[warps_per_block];
#if !YASPS_MAS_SEGMENTED_REDUCTION
  const unsigned int warp_base = local_warp * warp_size;
  __shared__ yasps_mas_product_t row_results[
      warps_per_block * warp_size * rows];
  __shared__ unsigned int row_starts[warps_per_block * warp_size];
#else
  using WarpReduce = cub::WarpReduce<yasps_mas_product_t, warp_size>;
  __shared__ typename WarpReduce::TempStorage reduction[warps_per_block];
#endif

  unsigned int row_start = 0xffffffffu;
  unsigned int col_start = 0xffffffffu;
  yasps_mas_product_t local_row_results[rows];
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    local_row_results[row] = 0.0;
  if (source_id < count) {
    const unsigned long long position_id = position_start + source_id;
    row_start = positions[2 * position_id];
    col_start = positions[2 * position_id + 1];
    const double* matrix = values + value_start
        + static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) {
      yasps_mas_product_t sum = 0.0f;
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        sum += __ldg(matrix + row * cols + col)
            * yasps_mas_load_product_x(x + col_start + col);
      local_row_results[row] = sum;
    }
  }
#if YASPS_MAS_SEGMENTED_REDUCTION
  const unsigned int previous = __shfl_up_sync(0xffffffffu, row_start, 1);
  const bool segment_head = lane == 0 || previous != row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) {
    const yasps_mas_product_t reduced = WarpReduce(reduction[local_warp])
        .HeadSegmentedReduce(local_row_results[row], segment_head, cub::Sum());
    if (source_id < count && segment_head)
      yasps_mas_atomic_add(
          y + row_start + row, static_cast<double>(reduced));
    __syncwarp();
  }
#else
  row_starts[warp_base + lane] = row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    row_results[(warp_base + lane) * rows + row] = local_row_results[row];
  __syncwarp();
  if (source_id < count &&
      (lane == 0 || row_starts[warp_base + lane - 1] != row_start)) {
    yasps_mas_product_t sums[rows];
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) sums[row] = 0.0;
    for (unsigned int other = lane;
         other < warp_size && row_starts[warp_base + other] == row_start;
         ++other) {
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sums[row] += row_results[(warp_base + other) * rows + row];
    }
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      yasps_mas_atomic_add(y + row_start + row, sums[row]);
  }
#endif
#if YASPS_MAS_SYMMETRIC_STORAGE
  if (source_id < count && row_start != col_start) {
    const double* matrix = values + value_start
        + static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) {
      yasps_mas_product_t sum = 0.0f;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sum += __ldg(matrix + row * cols + col)
            * yasps_mas_load_product_x(x + row_start + row);
      yasps_mas_atomic_add(
          y + col_start + col, static_cast<double>(sum));
    }
  }
#endif
  double quadratic = 0.0;
  if (source_id < count) {
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      quadratic += __ldg(x + row_start + row)
          * static_cast<double>(local_row_results[row]);
#if YASPS_MAS_SYMMETRIC_STORAGE
    if (row_start != col_start) quadratic *= 2.0;
#endif
  }
#if YASPS_MAS_FUSE_AUXILIARY
  {
    if (source_id < auxiliary_count)
      quadratic += yasps_mas_process_auxiliary(
          auxiliary_descriptors, auxiliary_descriptor_count,
          source_id, x, y);
  }
#endif
  yasps_mas_accumulate_quadratic_block<warps_per_block>(
      quadratic, quadratic_output, quadratic_warps);
}

// Same generated arithmetic for two disjoint buffers of one shape. Collision
// Hessians commonly match the dominant static category; treating the pair as one
// logical block stream removes a kernel launch per PCG iteration without
// copying or concatenating either live YASPS buffer.
extern "C" __global__ void yasps_mas_specialized_mapped_block_spmv_pair(
    const double* values_a, unsigned long long value_start_a,
    const unsigned int* positions_a, unsigned long long position_start_a,
    unsigned int count_a,
    const double* values_b, unsigned long long value_start_b,
    const unsigned int* positions_b, unsigned long long position_start_b,
    unsigned int count_b, const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_node_scalar_starts, unsigned int level_index,
    unsigned int fine_node_count, const double* x, double* y,
    double* quadratic_output) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  constexpr unsigned int warps_per_block = YASPS_MAS_WARPS_PER_BLOCK;
  const unsigned int lane = threadIdx.x & (warp_size - 1);
  const unsigned int local_warp = threadIdx.x / warp_size;
  const unsigned int source_id =
      (blockIdx.x * warps_per_block + local_warp) * warp_size + lane;
  __shared__ double quadratic_warps[warps_per_block];
  const unsigned int count = count_a + count_b;

  __shared__ double row_results[warps_per_block * warp_size * rows];
  __shared__ unsigned int row_starts[warps_per_block * warp_size];
  const unsigned int warp_base = local_warp * warp_size;
  unsigned int row_start = 0xffffffffu;
  unsigned int col_start = 0xffffffffu;
  unsigned int scalar_row = 0xffffffffu;
  unsigned int scalar_col = 0xffffffffu;
  const double* matrix = nullptr;
  double local_row_results[rows];
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    local_row_results[row] = 0.0;

  if (source_id < count) {
    const bool second = source_id >= count_a;
    const unsigned int local_id = second ? source_id - count_a : source_id;
    const unsigned int* positions = second ? positions_b : positions_a;
    const double* values = second ? values_b : values_a;
    const unsigned long long position_start = second
        ? position_start_b : position_start_a;
    const unsigned long long value_start = second ? value_start_b : value_start_a;
    const unsigned long long position_id = position_start + local_id;
    scalar_row = positions[2 * position_id];
    scalar_col = positions[2 * position_id + 1];
    if (level_index == 0) {
      row_start = scalar_row;
      col_start = scalar_col;
    } else {
      const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
      const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
      const unsigned int map_offset = level_index * fine_node_count;
      row_start = fine_node_scalar_starts[map_offset + fine_row];
      col_start = fine_node_scalar_starts[map_offset + fine_col];
    }
    matrix = values + value_start +
        static_cast<unsigned long long>(local_id) * rows * cols;
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) {
      double sum = 0.0;
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        sum += __ldg(matrix + row * cols + col)
            * __ldg(x + col_start + col);
      local_row_results[row] = sum;
    }
  }

  row_starts[warp_base + lane] = row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    row_results[(warp_base + lane) * rows + row] = local_row_results[row];
  // The four small-shape warps own disjoint shared-memory slices.
  __syncwarp();
  if (source_id < count &&
      (lane == 0 || row_starts[warp_base + lane - 1] != row_start)) {
    double sums[rows];
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) sums[row] = 0.0;
    for (unsigned int other = lane;
         other < warp_size &&
         row_starts[warp_base + other] == row_start; ++other) {
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sums[row] += row_results[(warp_base + other) * rows + row];
    }
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      yasps_mas_atomic_add(y + row_start + row, sums[row]);
  }
#if YASPS_MAS_SYMMETRIC_STORAGE
  if (source_id < count && scalar_row != scalar_col) {
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) {
      double sum = 0.0;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sum += __ldg(matrix + row * cols + col)
            * __ldg(x + row_start + row);
      yasps_mas_atomic_add(y + col_start + col, sum);
    }
  }
#endif
  double quadratic = 0.0;
  if (source_id < count) {
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      quadratic += __ldg(x + row_start + row) * local_row_results[row];
#if YASPS_MAS_SYMMETRIC_STORAGE
    if (scalar_row != scalar_col) quadratic *= 2.0;
#endif
  }
  yasps_mas_accumulate_quadratic_block<warps_per_block>(
      quadratic, quadratic_output, quadratic_warps);
}

extern "C" __global__ void yasps_mas_specialized_fine_block_spmv_pair(
    const double* values_a, unsigned long long value_start_a,
    const unsigned int* positions_a, unsigned long long position_start_a,
    unsigned int count_a,
    const double* values_b, unsigned long long value_start_b,
    const unsigned int* positions_b, unsigned long long position_start_b,
    unsigned int count_b, const unsigned int* live_counts_b,
    const unsigned long long* live_starts_b,
    const unsigned long long* live_position_offsets_b,
    unsigned int live_category_b, unsigned int capacity_b,
    const double* x, double* y,
    const unsigned long long* auxiliary_descriptors,
    unsigned int auxiliary_descriptor_count, unsigned int auxiliary_count,
    double* quadratic_output) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  constexpr unsigned int warps_per_block = YASPS_MAS_WARPS_PER_BLOCK;
  const unsigned int lane = threadIdx.x & (warp_size - 1);
  const unsigned int local_warp = threadIdx.x / warp_size;
  const unsigned int source_id =
      (blockIdx.x * warps_per_block + local_warp) * warp_size + lane;
  __shared__ double quadratic_warps[warps_per_block];
#if !YASPS_MAS_SEGMENTED_REDUCTION
  const unsigned int warp_base = local_warp * warp_size;
  __shared__ yasps_mas_product_t row_results[
      warps_per_block * warp_size * rows];
  __shared__ unsigned int row_starts[warps_per_block * warp_size];
#else
  using WarpReduce = cub::WarpReduce<yasps_mas_product_t, warp_size>;
  __shared__ typename WarpReduce::TempStorage reduction[warps_per_block];
#endif
  const bool device_count_b = live_counts_b != nullptr;
  const unsigned int active_b = device_count_b
      ? live_counts_b[live_category_b] : count_b;
  const unsigned int launched_b = device_count_b ? capacity_b : count_b;
  const unsigned int count = count_a + launched_b;
  bool valid = false;
  unsigned int row_start = 0xffffffffu;
  unsigned int col_start = 0xffffffffu;
  const double* matrix = nullptr;
  yasps_mas_product_t local_row_results[rows];
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    local_row_results[row] = 0.0;
  if (source_id < count) {
    const bool second = source_id >= count_a;
    const unsigned int local_id = second ? source_id - count_a : source_id;
    valid = !second || local_id < active_b;
    const unsigned int* positions = second ? positions_b : positions_a;
    const double* values = second ? values_b : values_a;
    const unsigned long long position_start = second
        ? (device_count_b
            ? live_position_offsets_b[live_category_b]
            : position_start_b)
        : position_start_a;
    const unsigned long long value_start = second
        ? (device_count_b ? live_starts_b[live_category_b] : value_start_b)
        : value_start_a;
    if (valid) {
      const unsigned long long position_id = position_start + local_id;
      row_start = positions[2 * position_id];
      col_start = positions[2 * position_id + 1];
      matrix = values + value_start
          + static_cast<unsigned long long>(local_id) * rows * cols;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row) {
        yasps_mas_product_t sum = 0.0f;
#pragma unroll
        for (unsigned int col = 0; col < cols; ++col)
          sum += __ldg(matrix + row * cols + col)
              * yasps_mas_load_product_x(x + col_start + col);
        local_row_results[row] = sum;
      }
    }
  }
#if YASPS_MAS_SEGMENTED_REDUCTION
  const unsigned int previous = __shfl_up_sync(0xffffffffu, row_start, 1);
  const bool segment_head = lane == 0 || previous != row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) {
    const yasps_mas_product_t reduced = WarpReduce(reduction[local_warp])
        .HeadSegmentedReduce(local_row_results[row], segment_head, cub::Sum());
    if (valid && segment_head)
      yasps_mas_atomic_add(
          y + row_start + row, static_cast<double>(reduced));
    __syncwarp();
  }
#else
  row_starts[warp_base + lane] = row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    row_results[(warp_base + lane) * rows + row] = local_row_results[row];
  __syncwarp();
  if (valid &&
      (lane == 0 || row_starts[warp_base + lane - 1] != row_start)) {
    yasps_mas_product_t sums[rows];
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) sums[row] = 0.0;
    for (unsigned int other = lane;
         other < warp_size && row_starts[warp_base + other] == row_start;
         ++other) {
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sums[row] += row_results[(warp_base + other) * rows + row];
    }
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      yasps_mas_atomic_add(y + row_start + row, sums[row]);
  }
#endif
#if YASPS_MAS_SYMMETRIC_STORAGE
  if (valid && row_start != col_start) {
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) {
      yasps_mas_product_t sum = 0.0f;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sum += __ldg(matrix + row * cols + col)
            * yasps_mas_load_product_x(x + row_start + row);
      yasps_mas_atomic_add(
          y + col_start + col, static_cast<double>(sum));
    }
  }
#endif
  double quadratic = 0.0;
  if (valid) {
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      quadratic += __ldg(x + row_start + row)
          * static_cast<double>(local_row_results[row]);
#if YASPS_MAS_SYMMETRIC_STORAGE
    if (row_start != col_start) quadratic *= 2.0;
#endif
  }
#if YASPS_MAS_FUSE_AUXILIARY
  {
    if (source_id < auxiliary_count)
      quadratic += yasps_mas_process_auxiliary(
          auxiliary_descriptors, auxiliary_descriptor_count,
          source_id, x, y);
  }
#endif
  yasps_mas_accumulate_quadratic_block<warps_per_block>(
      quadratic, quadratic_output, quadratic_warps);
}

// Fine-only static split. The row pass consumes the native YASPS row order;
// the transpose pass consumes a one-time column-sorted block-id permutation.
// Both orientations therefore use segmented warp reductions rather than one
// FP64 global atomic per block row/column.
extern "C" __global__ void yasps_mas_specialized_fine_static_row_spmv(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, const double* x, double* y,
    const unsigned long long* auxiliary_descriptors,
    unsigned int auxiliary_descriptor_count, unsigned int auxiliary_count,
    double* quadratic_output) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warps_per_block = YASPS_MAS_WARPS_PER_BLOCK;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int source_id = blockIdx.x * blockDim.x + threadIdx.x;
  using WarpReduce = cub::WarpReduce<double, 32>;
  __shared__ typename WarpReduce::TempStorage reduction[warps_per_block];
  __shared__ double quadratic_warps[warps_per_block];
  unsigned int row_start = 0xffffffffu;
  unsigned int col_start = 0xffffffffu;
  double local[rows];
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) local[row] = 0.0;
  if (source_id < count) {
    const unsigned long long position = position_start + source_id;
    row_start = positions[2 * position];
    col_start = positions[2 * position + 1];
    const double* matrix = values + value_start +
        static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) {
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        local[row] += __ldg(matrix + row * cols + col) *
            __ldg(x + col_start + col);
    }
  }
  const unsigned int previous = __shfl_up_sync(0xffffffffu, row_start, 1);
  const bool head = lane == 0u || previous != row_start;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) {
    const double sum = WarpReduce(reduction[warp]).HeadSegmentedReduce(
        local[row], head, cub::Sum());
    if (source_id < count && head)
      yasps_mas_atomic_add(y + row_start + row, sum);
    __syncwarp();
  }
  double quadratic = 0.0;
  if (source_id < count) {
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      quadratic += __ldg(x + row_start + row) * local[row];
#if YASPS_MAS_SYMMETRIC_STORAGE
    if (row_start != col_start) quadratic *= 2.0;
#endif
  }
#if YASPS_MAS_FUSE_AUXILIARY
  {
    if (source_id < auxiliary_count)
      quadratic += yasps_mas_process_auxiliary(
          auxiliary_descriptors, auxiliary_descriptor_count,
          source_id, x, y);
  }
#endif
  yasps_mas_accumulate_quadratic_block<warps_per_block>(
      quadratic, quadratic_output, quadratic_warps);
}

extern "C" __global__ void yasps_mas_specialized_fine_static_transpose_spmv(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    const unsigned int* sorted_block_ids, unsigned int count,
    const double* x, double* y) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warps_per_block = YASPS_MAS_WARPS_PER_BLOCK;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5u;
  const unsigned int sorted_id = blockIdx.x * blockDim.x + threadIdx.x;
  using WarpReduce = cub::WarpReduce<double, 32>;
  __shared__ typename WarpReduce::TempStorage reduction[warps_per_block];
  unsigned int output_start = 0xffffffffu;
  double local[cols];
#pragma unroll
  for (unsigned int col = 0; col < cols; ++col) local[col] = 0.0;
  if (sorted_id < count) {
    const unsigned int source_id = sorted_block_ids[sorted_id];
    const unsigned long long position = position_start + source_id;
    const unsigned int row_start = positions[2 * position];
    output_start = positions[2 * position + 1];
    if (row_start != output_start) {
      const double* matrix = values + value_start +
          static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col) {
#pragma unroll
        for (unsigned int row = 0; row < rows; ++row)
          local[col] += __ldg(matrix + row * cols + col) *
              __ldg(x + row_start + row);
      }
    }
  }
  const unsigned int previous =
      __shfl_up_sync(0xffffffffu, output_start, 1);
  const bool head = lane == 0u || previous != output_start;
#pragma unroll
  for (unsigned int col = 0; col < cols; ++col) {
    const double sum = WarpReduce(reduction[warp]).HeadSegmentedReduce(
        local[col], head, cub::Sum());
    if (sorted_id < count && head && output_start != 0xffffffffu)
      yasps_mas_atomic_add(y + output_start + col, sum);
    __syncwarp();
  }
}

// Static topology can afford one setup-time permutation. Split the symmetric
// product into its two orientations: the stored-row pass retains contiguous
// value traversal, while the transpose pass follows block ids sorted by their
// output column and combines equal outputs before issuing global atomics.
extern "C" __global__ void yasps_mas_specialized_block_spmv_row_only(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_node_scalar_starts, unsigned int level_index,
    unsigned int fine_node_count, const double* x, double* y) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  const unsigned int lane = threadIdx.x;
  const unsigned int source_id = blockIdx.x * warp_size + lane;
  __shared__ double results[warp_size * rows];
  __shared__ unsigned int output_starts[warp_size];
  unsigned int output = 0xffffffffu;
  double local[rows];
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row) local[row] = 0.0;
  if (source_id < count) {
    const unsigned long long position_id = position_start + source_id;
    const unsigned int scalar_row = positions[2 * position_id];
    const unsigned int scalar_col = positions[2 * position_id + 1];
    unsigned int input;
    if (level_index == 0) {
      output = scalar_row;
      input = scalar_col;
    } else {
      const unsigned int map_offset = level_index * fine_node_count;
      output = fine_node_scalar_starts[
          map_offset + scalar_boundary_to_node[scalar_row]];
      input = fine_node_scalar_starts[
          map_offset + scalar_boundary_to_node[scalar_col]];
    }
    const double* matrix = values + value_start
        + static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) {
      double value = 0.0;
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        value += matrix[row * cols + col] * x[input + col];
      local[row] = value;
    }
  }
  output_starts[lane] = output;
#pragma unroll
  for (unsigned int row = 0; row < rows; ++row)
    results[lane * rows + row] = local[row];
  __syncthreads();
  if (source_id < count &&
      (lane == 0 || output_starts[lane - 1] != output)) {
    double sums[rows];
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row) sums[row] = 0.0;
    for (unsigned int other = lane;
         other < warp_size && output_starts[other] == output; ++other) {
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        sums[row] += results[other * rows + row];
    }
#pragma unroll
    for (unsigned int row = 0; row < rows; ++row)
      yasps_mas_atomic_add(y + output + row, sums[row]);
  }
}

extern "C" __global__ void yasps_mas_specialized_block_spmv_transpose_sorted(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    const unsigned int* sorted_block_ids, unsigned int count,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_node_scalar_starts, unsigned int level_index,
    unsigned int fine_node_count, const double* x, double* y) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  const unsigned int lane = threadIdx.x;
  const unsigned int sorted_id = blockIdx.x * warp_size + lane;
  __shared__ double results[warp_size * cols];
  __shared__ unsigned int output_starts[warp_size];
  unsigned int output = 0xffffffffu;
  double local[cols];
#pragma unroll
  for (unsigned int col = 0; col < cols; ++col) local[col] = 0.0;
  if (sorted_id < count) {
    const unsigned int source_id = sorted_block_ids[sorted_id];
    const unsigned long long position_id = position_start + source_id;
    const unsigned int scalar_row = positions[2 * position_id];
    const unsigned int scalar_col = positions[2 * position_id + 1];
    unsigned int input;
    if (level_index == 0) {
      output = scalar_col;
      input = scalar_row;
    } else {
      const unsigned int map_offset = level_index * fine_node_count;
      output = fine_node_scalar_starts[
          map_offset + scalar_boundary_to_node[scalar_col]];
      input = fine_node_scalar_starts[
          map_offset + scalar_boundary_to_node[scalar_row]];
    }
    const double* matrix = values + value_start
        + static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) {
      double value = 0.0;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        value += matrix[row * cols + col] * x[input + row];
      local[col] = value;
    }
  }
  output_starts[lane] = output;
#pragma unroll
  for (unsigned int col = 0; col < cols; ++col)
    results[lane * cols + col] = local[col];
  __syncthreads();
  if (sorted_id < count &&
      (lane == 0 || output_starts[lane - 1] != output)) {
    double sums[cols];
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) sums[col] = 0.0;
    for (unsigned int other = lane;
         other < warp_size && output_starts[other] == output; ++other) {
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        sums[col] += results[other * cols + col];
    }
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col)
      yasps_mas_atomic_add(y + output + col, sums[col]);
  }
}

// Reorder changing numerical values into the one-time static column order.
// The topology/order is immutable; only this coalesced value copy is refreshed
// for each Hessian. Both source and destination block dimensions are fixed at
// JIT compilation, so the block-area loop disappears.
extern "C" __global__ void yasps_mas_gather_transpose_ordered_values(
    const double* values, unsigned long long value_start,
    const unsigned int* sorted_block_ids, unsigned int count,
    double* ordered_values) {
  constexpr unsigned int area =
      YASPS_MAS_BLOCK_ROWS * YASPS_MAS_BLOCK_COLS;
  const unsigned int entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= count * area) return;
  const unsigned int ordered_block = entry / area;
  const unsigned int local = entry - ordered_block * area;
  const unsigned int source_block = sorted_block_ids[ordered_block];
  ordered_values[entry] = values[value_start
      + static_cast<unsigned long long>(source_block) * area + local];
}

// Transpose half of symmetric SpMV over a physically column-ordered block
// stream. Unlike the id-indirection variant above, matrix reads are contiguous
// and adjacent equal output blocks are combined before one global atomic.
extern "C" __global__ void yasps_mas_specialized_block_spmv_transpose_ordered(
    const double* ordered_values, const unsigned int* ordered_positions,
    unsigned int count, const double* x, double* y) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int warp_size = 32;
  const unsigned int lane = threadIdx.x;
  const unsigned int source_id = blockIdx.x * warp_size + lane;
  __shared__ double results[warp_size * cols];
  __shared__ unsigned int output_starts[warp_size];
  unsigned int output = 0xffffffffu;
  double local[cols];
#pragma unroll
  for (unsigned int col = 0; col < cols; ++col) local[col] = 0.0;
  if (source_id < count) {
    const unsigned int input = ordered_positions[2 * source_id];
    output = ordered_positions[2 * source_id + 1];
    const double* matrix = ordered_values
        + static_cast<unsigned long long>(source_id) * rows * cols;
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) {
      double value = 0.0;
#pragma unroll
      for (unsigned int row = 0; row < rows; ++row)
        value += matrix[row * cols + col] * x[input + row];
      local[col] = value;
    }
  }
  output_starts[lane] = output;
#pragma unroll
  for (unsigned int col = 0; col < cols; ++col)
    results[lane * cols + col] = local[col];
  __syncthreads();
  if (source_id < count &&
      (lane == 0 || output_starts[lane - 1] != output)) {
    double sums[cols];
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col) sums[col] = 0.0;
    for (unsigned int other = lane;
         other < warp_size && output_starts[other] == output; ++other) {
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        sums[col] += results[other * cols + col];
    }
#pragma unroll
    for (unsigned int col = 0; col < cols; ++col)
      yasps_mas_atomic_add(y + output + col, sums[col]);
  }
}

// Large runtime blocks expose enough row parallelism to use a 16-thread
// sub-warp per Hessian block. This complements (rather than replaces) the
// lane-per-block kernel above: small matrices benefit from row aggregation,
// while 8x8/12x12 matrices avoid serializing 64--144 FMAs on one lane.
extern "C" __global__ void yasps_mas_cooperative_mapped_block_spmv(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_node_scalar_starts, unsigned int level_index,
    unsigned int fine_node_count, const double* x, double* y,
    double* quadratic_output) {
  constexpr unsigned int rows = YASPS_MAS_BLOCK_ROWS;
  constexpr unsigned int cols = YASPS_MAS_BLOCK_COLS;
  constexpr unsigned int group_size = 16;
  constexpr unsigned int groups_per_block = 8;
  const unsigned int group = threadIdx.x / group_size;
  const unsigned int lane = threadIdx.x % group_size;
  const unsigned int source_id = blockIdx.x * groups_per_block + group;
  const bool active = source_id < count;
  unsigned int scalar_row = 0u;
  unsigned int scalar_col = 0u;
  unsigned int row_start = 0u;
  unsigned int col_start = 0u;
  const double* matrix = nullptr;
  if (active) {
    const unsigned long long position_id = position_start + source_id;
    scalar_row = positions[2 * position_id];
    scalar_col = positions[2 * position_id + 1];
    if (level_index == 0) {
      row_start = scalar_row;
      col_start = scalar_col;
    } else {
      const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
      const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
      const unsigned int map_offset = level_index * fine_node_count;
      row_start = fine_node_scalar_starts[map_offset + fine_row];
      col_start = fine_node_scalar_starts[map_offset + fine_col];
    }
    matrix = values + value_start +
        static_cast<unsigned long long>(source_id) * rows * cols;
  }
  double quadratic = 0.0;
  if (active) {
    for (unsigned int row = lane; row < rows; row += group_size) {
      double sum = 0.0;
#pragma unroll
      for (unsigned int col = 0; col < cols; ++col)
        sum += matrix[row * cols + col] * x[col_start + col];
      yasps_mas_atomic_add(y + row_start + row, sum);
      quadratic += x[row_start + row] * sum;
    }
#if YASPS_MAS_SYMMETRIC_STORAGE
    if (scalar_row != scalar_col) {
      for (unsigned int col = lane; col < cols; col += group_size) {
        double sum = 0.0;
#pragma unroll
        for (unsigned int row = 0; row < rows; ++row)
          sum += matrix[row * cols + col] * x[row_start + row];
        yasps_mas_atomic_add(y + col_start + col, sum);
      }
    }
#endif
  }
  for (unsigned int offset = group_size / 2; offset; offset >>= 1)
    quadratic += __shfl_down_sync(0xffffffffu, quadratic, offset, group_size);
#if YASPS_MAS_SYMMETRIC_STORAGE
  if (active && scalar_row != scalar_col) quadratic *= 2.0;
#endif
  __shared__ double quadratic_groups[groups_per_block];
  if (lane == 0u) quadratic_groups[group] = quadratic;
  __syncthreads();
  if (quadratic_output && threadIdx.x < 32u) {
    quadratic = threadIdx.x < groups_per_block
        ? quadratic_groups[threadIdx.x] : 0.0;
#pragma unroll
    for (unsigned int offset = 16; offset; offset >>= 1)
      quadratic += __shfl_down_sync(0xffffffffu, quadratic, offset);
    if (threadIdx.x == 0u)
      yasps_mas_atomic_add(quadratic_output, quadratic);
  }
}
#endif

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

// One thread computes one local output scalar. Domains are nonoverlapping at a
// level; callers gather/scatter with their static domain scalar index arrays.
extern "C" __global__ void yasps_mas_dense_inverse_apply(
    const double* inverses, const double* gathered_residuals, double* corrections,
    const std::uint64_t* matrix_offsets, const std::uint64_t* vector_offsets,
    const std::uint32_t* sizes, const std::uint32_t* padded_sizes,
    std::uint32_t domain_count) {
  const std::uint32_t domain = blockIdx.x;
  if (domain >= domain_count) return;
  const std::uint32_t n = sizes[domain];
  const std::uint32_t padded = padded_sizes[domain];
  const double* inverse = inverses + matrix_offsets[domain];
  const double* residual = gathered_residuals + vector_offsets[domain];
  for (std::uint32_t row = threadIdx.x; row < n; row += blockDim.x) {
    double value = 0.0;
    for (std::uint32_t col = 0; col < n; ++col)
      value += inverse[row * padded + col] * residual[col];
    corrections[vector_offsets[domain] + row] = value;
  }
}

extern "C" __global__ void yasps_mas_gather_domains(
    const double* level_vector, double* packed_vectors,
    const std::uint64_t* level_scalar_to_packed,
    std::uint32_t level_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar < level_dofs)
    packed_vectors[level_scalar_to_packed[scalar]] = level_vector[scalar];
}

extern "C" __global__ void yasps_mas_scatter_domains(
    const double* packed_vectors, double* level_vector,
    const std::uint64_t* level_scalar_to_packed,
    std::uint32_t level_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar < level_dofs)
    level_vector[scalar] = packed_vectors[level_scalar_to_packed[scalar]];
}

// Fuse fine-to-level restriction with the domain gather.  This removes two
// short-lived device arrays and one kernel launch per hierarchy level.
extern "C" __global__ void yasps_mas_restrict_gather_domains(
    const double* fine, double* packed_vectors,
    const std::uint32_t* fine_scalar_to_level_scalar,
    const std::uint64_t* level_scalar_to_packed,
    std::uint32_t fine_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar < fine_dofs) {
    const std::uint32_t level_scalar = fine_scalar_to_level_scalar[scalar];
    yasps_mas_atomic_add(
        packed_vectors + level_scalar_to_packed[level_scalar], fine[scalar]);
  }
}

// Fuse the domain scatter with level-to-fine prolongation.  Every fine scalar
// has exactly one parent scalar, so no atomics are required here.
extern "C" __global__ void yasps_mas_scatter_prolongate_add(
    const double* packed_vectors, double* fine,
    const std::uint32_t* fine_scalar_to_level_scalar,
    const std::uint64_t* level_scalar_to_packed,
    std::uint32_t fine_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar < fine_dofs) {
    const std::uint32_t level_scalar = fine_scalar_to_level_scalar[scalar];
    fine[scalar] += packed_vectors[level_scalar_to_packed[level_scalar]];
  }
}

// All hierarchy levels share one packed residual arena. One fine scalar walks
// its precomputed destination at every level, matching GIPC's BuildMultiLevelR
// lifecycle in a single kernel launch.
extern "C" __global__ void yasps_mas_restrict_all_levels(
    const double* fine, double* packed_vectors,
    const std::uint64_t* fine_scalar_to_packed,
    std::uint32_t level_count, std::uint32_t fine_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar >= fine_dofs) return;
  const double value = fine[scalar];
  for (std::uint32_t level = 0; level < level_count; ++level)
    yasps_mas_atomic_add(
        packed_vectors + fine_scalar_to_packed[level * fine_dofs + scalar], value);
}

// Sum the local corrections from every hierarchy level back to one fine
// scalar. This is the heterogeneous-block equivalent of GIPC CollectFinalZ.
extern "C" __global__ void yasps_mas_collect_all_levels(
    const double* packed_vectors, double* fine,
    const std::uint64_t* fine_scalar_to_packed,
    std::uint32_t level_count, std::uint32_t fine_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar >= fine_dofs) return;
  double value = 0.0;
  for (std::uint32_t level = 0; level < level_count; ++level)
    value += packed_vectors[fine_scalar_to_packed[level * fine_dofs + scalar]];
  fine[scalar] = value;
}

// GIPC-style mixed precision: local inverses and multilevel work vectors are
// FP32, while the Hessian, PCG vectors, reductions, and final correction stay
// FP64. Inversion itself is still performed in FP64 before this one-time cast.
extern "C" __global__ void yasps_mas_cast_inverse_to_float(
    const double* source, float* destination, std::uint64_t count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) destination[index] = static_cast<float>(source[index]);
}

// Adjacent-level reduction prevents the coarsest thread from rereading every
// fine node. Each parent only sums its immediate child blocks, like GIPC's
// repeated BANKSIZE aggregation, while still supporting irregular domains.
extern "C" __global__ void yasps_mas_restrict_adjacent_nodes_mixed(
    float* packed, const std::uint64_t* parent_packed_starts,
    const std::uint64_t* child_packed_starts,
    const std::uint64_t* parent_offsets,
    const std::uint32_t* parent_children,
    const std::uint32_t* parent_dimensions,
    std::uint32_t parent_count) {
  const std::uint32_t parent = blockIdx.x * blockDim.x + threadIdx.x;
  if (parent >= parent_count) return;
  const std::uint64_t output = parent_packed_starts[parent];
  const std::uint64_t begin = parent_offsets[parent];
  const std::uint64_t end = parent_offsets[parent + 1];
  const std::uint32_t dimension = parent_dimensions[parent];
  for (std::uint32_t component = 0; component < dimension; ++component) {
    float sum = 0.0f;
    for (std::uint64_t index = begin; index < end; ++index) {
      const std::uint32_t child = parent_children[index];
      sum += packed[child_packed_starts[child] + component];
    }
    packed[output + component] = sum;
  }
}

// One thread owns one fine variable block. Fine nodes are statically sorted by
// their complete ancestry so every parent is contiguous. Warp-segmented sums
// therefore reduce each parent before one FP32 atomic per warp segment, which
// is the irregular/heterogeneous equivalent of GIPC BuildMultiLevelR.
extern "C" __global__ void yasps_mas_restrict_warp_nodes_mixed(
    const double* fine, float* packed,
    const std::uint32_t* restriction_order,
    const std::uint64_t* fine_node_to_packed_starts,
    const std::uint32_t* fine_node_scalar_offsets,
    const std::uint32_t* fine_node_dimensions,
    std::uint32_t maximum_dimension,
    std::uint32_t level_count, std::uint32_t fine_node_count) {
  const std::uint32_t ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= fine_node_count) return;
  const std::uint32_t fine_node = restriction_order[ordered_node];
  const std::uint32_t input = fine_node_scalar_offsets[fine_node];
  const std::uint32_t dimension = fine_node_dimensions[fine_node];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int active = __activemask();

  for (std::uint32_t component = 0; component < maximum_dimension; ++component) {
    const bool valid = component < dimension;
    const float original = valid ? static_cast<float>(fine[input + component]) : 0.0f;
    if (valid) {
      const std::uint64_t level_zero =
          fine_node_to_packed_starts[fine_node] + component;
      // Level zero is never collapsed: its bank connectivity constructs the
      // level-one supernodes.  Therefore every fine scalar owns this slot and
      // can avoid an atomic on the hottest restriction level.
      packed[level_zero] = original;
    }
    for (std::uint32_t level = 1; level < level_count; ++level) {
      std::uint64_t key = valid
          ? fine_node_to_packed_starts[level * fine_node_count + fine_node] + component
          : 0xffffffffffffffffull - ordered_node;
      float value = original;
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const std::uint64_t other_key = __shfl_down_sync(active, key, offset);
        const float other_value = __shfl_down_sync(active, value, offset);
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      const std::uint64_t previous_key = __shfl_up_sync(active, key, 1);
      const bool segment_head = lane == 0 || previous_key != key;
      if (valid && segment_head) atomicAdd(packed + key, value);
    }
  }
}

extern "C" __global__ void yasps_mas_dense_inverse_apply_mixed(
    const float* inverses, const float* gathered_residuals, float* corrections,
    const std::uint64_t* matrix_offsets, const std::uint64_t* vector_offsets,
    const std::uint32_t* sizes, const std::uint32_t* padded_sizes,
    std::uint32_t domain_count) {
  const std::uint32_t domain = blockIdx.x;
  if (domain >= domain_count) return;
  const std::uint32_t n = sizes[domain];
  const std::uint32_t padded = padded_sizes[domain];
  const float* inverse = inverses + matrix_offsets[domain];
  const float* residual = gathered_residuals + vector_offsets[domain];
  for (std::uint32_t row = threadIdx.x; row < n; row += blockDim.x) {
    float value = 0.0f;
    for (std::uint32_t col = 0; col < n; ++col)
      value = fmaf(inverse[row * padded + col], residual[col], value);
    corrections[vector_offsets[domain] + row] = value;
  }
}

// GIPC-style block-pair inverse application generalized to heterogeneous YASPS
// variables. One CUDA thread owns one (row node, column node) dense subblock,
// reads its coefficients contiguously, and accumulates a short row-node vector
// in shared memory. This avoids the stride-P warp loads of scalar-row SpMV and
// uses all node-pair parallelism available inside a runtime-sized Schwarz bank.
extern "C" __global__ void yasps_mas_dense_inverse_apply_block_pairs(
    const float* inverses, const float* gathered_residuals, float* corrections,
    const std::uint64_t* matrix_offsets, const std::uint64_t* vector_offsets,
    const std::uint32_t* sizes, const std::uint32_t* padded_sizes,
    const std::uint64_t* domain_node_offsets,
    const std::uint32_t* domain_nodes,
    const std::uint32_t* node_local_offsets,
    const std::uint32_t* node_dimensions,
    std::uint32_t maximum_node_dimension, std::uint32_t domain_count) {
  const std::uint32_t domain = blockIdx.x;
  if (domain >= domain_count) return;
  const std::uint32_t n = sizes[domain];
  const std::uint32_t padded = padded_sizes[domain];
  const std::uint64_t vector_offset = vector_offsets[domain];
  const std::uint64_t begin = domain_node_offsets[domain];
  const std::uint32_t node_count = static_cast<std::uint32_t>(
      domain_node_offsets[domain + 1] - begin);
  const float* inverse = inverses + matrix_offsets[domain];
  extern __shared__ float workspace[];
  float* residual = workspace;
  float* result = residual + padded;
  for (std::uint32_t scalar = threadIdx.x; scalar < padded;
       scalar += blockDim.x) {
    residual[scalar] = scalar < n
        ? gathered_residuals[vector_offset + scalar] : 0.0f;
    result[scalar] = 0.0f;
  }
  __syncthreads();

  const std::uint32_t pair_count = node_count * node_count;
  const unsigned int lane = threadIdx.x & 31u;
  for (std::uint32_t pair_base = 0; pair_base < pair_count;
       pair_base += blockDim.x) {
    const std::uint32_t pair = pair_base + threadIdx.x;
    const bool active = pair < pair_count;
    const std::uint32_t row_ordinal = active ? pair / node_count : 0u;
    const std::uint32_t col_ordinal = active
        ? pair - row_ordinal * node_count : 0u;
    const std::uint32_t row_node = active
        ? domain_nodes[begin + row_ordinal] : 0u;
    const std::uint32_t col_node = active
        ? domain_nodes[begin + col_ordinal] : 0u;
    const std::uint32_t row_start = active
        ? node_local_offsets[row_node] : 0u;
    const std::uint32_t col_start = active
        ? node_local_offsets[col_node] : 0u;
    const std::uint32_t row_dimension = active
        ? node_dimensions[row_node] : 0u;
    const std::uint32_t col_dimension = active
        ? node_dimensions[col_node] : 0u;
    for (std::uint32_t row = 0; row < maximum_node_dimension; ++row) {
      const bool component_active = active && row < row_dimension;
      float value = 0.0f;
      if (component_active) {
        const float* matrix_row = inverse
            + static_cast<std::uint64_t>(row_start + row) * padded + col_start;
        for (std::uint32_t col = 0; col < col_dimension; ++col)
          value = fmaf(matrix_row[col], residual[col_start + col], value);
      }
      const unsigned int key = component_active
          ? row_start + row : 0xffffffffu - threadIdx.x;
      const unsigned int mask = __activemask();
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int other_key = __shfl_down_sync(mask, key, offset);
        const float other_value = __shfl_down_sync(mask, value, offset);
        if (lane + offset < 32 && other_key == key) value += other_value;
      }
      const unsigned int previous_key = __shfl_up_sync(mask, key, 1);
      const bool segment_head = lane == 0 || previous_key != key;
      if (component_active && segment_head)
        atomicAdd(result + key, value);
    }
  }
  __syncthreads();
  for (std::uint32_t row = threadIdx.x; row < n; row += blockDim.x)
    corrections[vector_offset + row] = result[row];
}

// Large row-major domains need a different mapping: the row-per-thread kernel
// above makes neighboring lanes read with ``padded`` stride. Here one warp
// owns one output row, so inverse coefficients are fetched contiguously. The
// residual is staged once per domain and shared by every row warp.
extern "C" __global__ void yasps_mas_dense_inverse_apply_mixed_cooperative(
    const float* inverses, const float* gathered_residuals, float* corrections,
    const std::uint64_t* matrix_offsets, const std::uint64_t* vector_offsets,
    const std::uint32_t* sizes, const std::uint32_t* padded_sizes,
    std::uint32_t domain_count) {
  const std::uint32_t domain = blockIdx.x;
  if (domain >= domain_count) return;
  constexpr std::uint32_t warp_size = 32;
  const std::uint32_t lane = threadIdx.x & (warp_size - 1);
  const std::uint32_t warp = threadIdx.x / warp_size;
  const std::uint32_t warp_count = blockDim.x / warp_size;
  const std::uint32_t n = sizes[domain];
  const std::uint32_t padded = padded_sizes[domain];
  const std::uint64_t vector_offset = vector_offsets[domain];
  const float* inverse = inverses + matrix_offsets[domain];
  extern __shared__ float residual[];
  for (std::uint32_t col = threadIdx.x; col < n; col += blockDim.x)
    residual[col] = gathered_residuals[vector_offset + col];
  __syncthreads();
  for (std::uint32_t row = warp; row < n; row += warp_count) {
    float value = 0.0f;
    for (std::uint32_t col = lane; col < n; col += warp_size)
      value = fmaf(inverse[row * padded + col], residual[col], value);
    for (std::uint32_t offset = warp_size / 2; offset; offset >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0) corrections[vector_offset + row] = value;
  }
}

// One thread owns one fine variable block and gathers its corresponding block
// correction from every level. No atomics are required on prolongation.
extern "C" __global__ void yasps_mas_collect_nodes_mixed(
    const float* packed, double* fine,
    const std::uint64_t* fine_node_to_packed_starts,
    const std::uint32_t* fine_node_scalar_offsets,
    const std::uint32_t* fine_node_dimensions,
    std::uint32_t level_count, std::uint32_t fine_node_count) {
  const std::uint32_t fine_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (fine_node >= fine_node_count) return;
  const std::uint32_t output = fine_node_scalar_offsets[fine_node];
  const std::uint32_t dimension = fine_node_dimensions[fine_node];
  for (std::uint32_t component = 0; component < dimension; ++component) {
    float sum = 0.0f;
    for (std::uint32_t level = 0; level < level_count; ++level) {
      const std::uint64_t input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      sum += packed[input + component];
    }
    fine[output + component] = static_cast<double>(sum);
  }
}

// Add overlapping collision-edge corrections after the static multilevel
// collection. Each off-diagonal dynamic block owns one small dense inverse;
// output atomics are required because contact edges overlap at their nodes.
extern "C" __global__ void yasps_mas_apply_dynamic_edge_domains(
    const float* inverses, unsigned int edge_padded_size,
    const unsigned int* positions, const unsigned int* counts,
    const unsigned long long* position_offsets, const unsigned int* shapes,
    unsigned int category_count, unsigned int block_count,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* edge_node_counts,
    const double* residual, double* correction) {
  const unsigned int edge = blockIdx.x;
  if (edge >= block_count) return;
  unsigned int category = 0;
  while (category + 1 < category_count &&
         edge >= position_offsets[category + 1])
    ++category;
  if (category >= category_count ||
      edge >= position_offsets[category] + counts[category]) return;
  const unsigned int rows = shapes[2 * category];
  const unsigned int cols = shapes[2 * category + 1];
  const unsigned int scalar_row = positions[2 * edge];
  const unsigned int scalar_col = positions[2 * edge + 1];
  if (scalar_row == scalar_col) return;
  const unsigned int row_node = scalar_boundary_to_node[scalar_row];
  const unsigned int col_node = scalar_boundary_to_node[scalar_col];
  const unsigned int row_degree = edge_node_counts[row_node];
  const unsigned int col_degree = edge_node_counts[col_node];
  const float row_weight = rsqrtf(static_cast<float>(
      row_degree ? row_degree : 1u));
  const float col_weight = rsqrtf(static_cast<float>(
      col_degree ? col_degree : 1u));
  const unsigned int n = rows + cols;
  const float* inverse = inverses
      + static_cast<unsigned long long>(edge) * edge_padded_size * edge_padded_size;
  for (unsigned int row = threadIdx.x; row < n; row += blockDim.x) {
    float value = 0.0f;
    for (unsigned int col = 0; col < n; ++col) {
      const double source = col < rows
          ? residual[scalar_row + col] * row_weight
          : residual[scalar_col + col - rows] * col_weight;
      value = fmaf(inverse[row * edge_padded_size + col],
                   static_cast<float>(source), value);
    }
    double* output = row < rows
        ? correction + scalar_row + row
        : correction + scalar_col + row - rows;
    const float output_weight = row < rows ? row_weight : col_weight;
    yasps_mas_atomic_add(
        output, static_cast<double>(value * output_weight));
  }
}

extern "C" __global__ void yasps_mas_apply_dynamic_group_domains(
    const float* inverses, unsigned int group_stride,
    const unsigned int* group_active_sizes,
    const unsigned int* group_scalar_indices,
    const unsigned int* group_scalar_nodes,
    const unsigned int* group_node_counts, unsigned int group_count,
    const double* residual, double* correction) {
  const unsigned int group = blockIdx.x;
  if (group >= group_count) return;
  const unsigned int n = group_active_sizes[group];
  const float* inverse = inverses
      + static_cast<unsigned long long>(group) * group_stride * group_stride;
  for (unsigned int row = threadIdx.x; row < n; row += blockDim.x) {
    float value = 0.0f;
    for (unsigned int col = 0; col < n; ++col) {
      const unsigned int scalar = group_scalar_indices[group * group_stride + col];
      const unsigned int node = group_scalar_nodes[group * group_stride + col];
      const float weight = 0.5f * rsqrtf(static_cast<float>(
          max(1u, group_node_counts[node])));
      value = fmaf(inverse[row * group_stride + col],
                   static_cast<float>(residual[scalar]) * weight, value);
    }
    const unsigned int scalar = group_scalar_indices[group * group_stride + row];
    const unsigned int node = group_scalar_nodes[group * group_stride + row];
    const float weight = 0.5f * rsqrtf(static_cast<float>(
        max(1u, group_node_counts[node])));
    yasps_mas_atomic_add(correction + scalar,
                         static_cast<double>(value * weight));
  }
}

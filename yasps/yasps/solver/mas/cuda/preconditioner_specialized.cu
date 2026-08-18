#include <cstdint>

#if !defined(YASPS_MAS_MAX_DIMENSION) || !defined(YASPS_MAS_LEVEL_COUNT) || \
    !defined(YASPS_MAS_MAX_PADDED_SIZE)
#error "preconditioner dimensions must be supplied at compile time"
#endif

#if defined(YASPS_MAS_COMPACT_PACKED_OFFSETS)
using yasps_mas_packed_offset_t = unsigned int;
static __device__ __forceinline__ unsigned int yasps_mas_segment_key(
    unsigned int value) { return value; }
static __device__ __forceinline__ unsigned int yasps_mas_packed_offset(
    unsigned int value) { return value; }
#else
using yasps_mas_packed_offset_t = unsigned long long;
static __device__ __forceinline__ unsigned int yasps_mas_segment_key(
    unsigned long long value) {
  return static_cast<unsigned int>(value >> 32);
}
static __device__ __forceinline__ unsigned int yasps_mas_packed_offset(
    unsigned long long value) {
  return static_cast<unsigned int>(value & 0xffffffffull);
}
#endif

template <unsigned int DIMENSION>
__device__ __forceinline__ void yasps_mas_restrict_one_node(
    const double* __restrict__ fine, float* __restrict__ packed,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    unsigned int fine_node, unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int input = fine_node_scalar_offsets[fine_node];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int active = __activemask();
  float original[DIMENSION];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    original[component] = static_cast<float>(fine[input + component]);
    const unsigned int level_zero = yasps_mas_packed_offset(
        fine_node_level_keys[fine_node]) + component;
    packed[level_zero] = original[component];
  }
#pragma unroll
  for (unsigned int level = 1; level < level_count; ++level) {
    const unsigned int map_index = level * fine_node_count + fine_node;
    const yasps_mas_packed_offset_t packed_key =
        fine_node_level_keys[map_index];
    const unsigned int key = yasps_mas_segment_key(packed_key);
    const unsigned int output = yasps_mas_packed_offset(packed_key);
    const unsigned int previous_key = __shfl_up_sync(active, key, 1);
    const bool segment_head = lane == 0 || previous_key != key;
#pragma unroll
    for (unsigned int component = 0; component < DIMENSION; ++component) {
      float value = original[component];
#pragma unroll
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int other_key =
            __shfl_down_sync(active, key, offset);
        const float other_value =
            __shfl_down_sync(active, value, offset);
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      if (segment_head) atomicAdd(packed + output + component, value);
    }
  }
}

// Level zero is a disjoint partition, so its gather, inverse multiply, and
// scatter can be fused without atomics.  This companion only materializes the
// genuinely reduced levels in the packed workspace.
template <unsigned int DIMENSION>
__device__ __forceinline__ void yasps_mas_restrict_one_node_coarse(
    const double* __restrict__ fine, float* __restrict__ packed,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    unsigned int fine_node, unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int input = fine_node_scalar_offsets[fine_node];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int active = __activemask();
  float original[DIMENSION];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component)
    original[component] = static_cast<float>(fine[input + component]);
#pragma unroll
  for (unsigned int level = 1; level < level_count; ++level) {
    const unsigned int map_index = level * fine_node_count + fine_node;
    const yasps_mas_packed_offset_t packed_key =
        fine_node_level_keys[map_index];
    const unsigned int key = yasps_mas_segment_key(packed_key);
    const unsigned int output = yasps_mas_packed_offset(packed_key);
    const unsigned int previous_key = __shfl_up_sync(active, key, 1);
    const bool segment_head = lane == 0 || previous_key != key;
#pragma unroll
    for (unsigned int component = 0; component < DIMENSION; ++component) {
      float value = original[component];
#pragma unroll
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int other_key = __shfl_down_sync(active, key, offset);
        const float other_value = __shfl_down_sync(active, value, offset);
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      if (segment_head) atomicAdd(packed + output + component, value);
    }
  }
}

extern "C" __global__ void yasps_mas_restrict_warp_nodes_mixed_specialized(
    const double* __restrict__ fine, float* __restrict__ packed,
    const unsigned int* __restrict__ restriction_order,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    unsigned int fine_node_count) {
  constexpr unsigned int max_dimension = YASPS_MAS_MAX_DIMENSION;
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= fine_node_count) return;
  const unsigned int fine_node = restriction_order[ordered_node];
  const unsigned int input = fine_node_scalar_offsets[fine_node];
  const unsigned int dimension = fine_node_dimensions[fine_node];
  switch (dimension) {
    // YASPS_MAS_GENERATED_RESTRICT_DIMENSION_CASES
    default: break;
  }
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int active = __activemask();

  float original[max_dimension];

#pragma unroll
  for (unsigned int component = 0; component < max_dimension; ++component) {
    const bool valid = component < dimension;
    original[component] =
        valid ? static_cast<float>(fine[input + component]) : 0.0f;
    if (valid) {
      const unsigned int level_zero = yasps_mas_packed_offset(
          fine_node_level_keys[fine_node]) + component;
      packed[level_zero] = original[component];
    }
  }
#pragma unroll
  for (unsigned int level = 1; level < level_count; ++level) {
    const unsigned int map_index = level * fine_node_count + fine_node;
    const yasps_mas_packed_offset_t packed_key =
        fine_node_level_keys[map_index];
    const unsigned int key = yasps_mas_segment_key(packed_key);
    const unsigned int output = yasps_mas_packed_offset(packed_key);
    const unsigned int previous_key = __shfl_up_sync(active, key, 1);
    const bool segment_head = lane == 0 || previous_key != key;
    const unsigned int key_1 = __shfl_down_sync(active, key, 1);
    const unsigned int key_2 = __shfl_down_sync(active, key, 2);
    const unsigned int key_4 = __shfl_down_sync(active, key, 4);
    const unsigned int key_8 = __shfl_down_sync(active, key, 8);
    const unsigned int key_16 = __shfl_down_sync(active, key, 16);
#pragma unroll
    for (unsigned int component = 0; component < max_dimension; ++component) {
      float value = original[component];
#pragma unroll
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const float other_value = __shfl_down_sync(active, value, offset);
        const unsigned int other_key =
            offset == 1 ? key_1 : offset == 2 ? key_2 :
            offset == 4 ? key_4 : offset == 8 ? key_8 : key_16;
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      if (component < dimension && segment_head)
        atomicAdd(packed + output + component, value);
    }
  }
}

extern "C" __global__ void
yasps_mas_restrict_coarse_warp_nodes_mixed_specialized(
    const double* __restrict__ fine, float* __restrict__ packed,
    const unsigned int* __restrict__ restriction_order,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    unsigned int fine_node_count) {
  const unsigned int ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= fine_node_count) return;
  const unsigned int fine_node = restriction_order[ordered_node];
  switch (fine_node_dimensions[fine_node]) {
    // YASPS_MAS_GENERATED_RESTRICT_COARSE_DIMENSION_CASES
    default: break;
  }
}

// The PCG recurrence previously streamed the full solution, direction,
// residual, and product in one scalar kernel and then streamed the residual
// again for multilevel restriction.  One thread already owns a complete fine
// variable here, so update the recurrence vectors while their components are
// hot and immediately restrict the new residual.
template <unsigned int DIMENSION>
__device__ __forceinline__ void yasps_mas_update_restrict_one_node(
    double* __restrict__ solution, const double* __restrict__ direction,
    double* __restrict__ residual, const double* __restrict__ product,
    const double* __restrict__ state, float* __restrict__ packed,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    unsigned int fine_node, unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int input = fine_node_scalar_offsets[fine_node];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int active = __activemask();
  const double alpha = state[5];
  float updated[DIMENSION];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    const unsigned int scalar = input + component;
    solution[scalar] += alpha * direction[scalar];
    const double value = residual[scalar] - alpha * product[scalar];
    residual[scalar] = value;
    updated[component] = static_cast<float>(value);
    const unsigned int level_zero = yasps_mas_packed_offset(
        fine_node_level_keys[fine_node]) + component;
    packed[level_zero] = updated[component];
  }
#pragma unroll
  for (unsigned int level = 1; level < level_count; ++level) {
    const unsigned int map_index = level * fine_node_count + fine_node;
    const yasps_mas_packed_offset_t packed_key =
        fine_node_level_keys[map_index];
    const unsigned int key = yasps_mas_segment_key(packed_key);
    const unsigned int output = yasps_mas_packed_offset(packed_key);
    const unsigned int previous_key = __shfl_up_sync(active, key, 1);
    const bool segment_head = lane == 0 || previous_key != key;
    const unsigned int key_1 = __shfl_down_sync(active, key, 1);
    const unsigned int key_2 = __shfl_down_sync(active, key, 2);
    const unsigned int key_4 = __shfl_down_sync(active, key, 4);
    const unsigned int key_8 = __shfl_down_sync(active, key, 8);
    const unsigned int key_16 = __shfl_down_sync(active, key, 16);
#pragma unroll
    for (unsigned int component = 0; component < DIMENSION; ++component) {
      float value = updated[component];
#pragma unroll
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const float other_value = __shfl_down_sync(active, value, offset);
        const unsigned int other_key =
            offset == 1 ? key_1 : offset == 2 ? key_2 :
            offset == 4 ? key_4 : offset == 8 ? key_8 : key_16;
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      if (segment_head) atomicAdd(packed + output + component, value);
    }
  }
}

extern "C" __global__ void
yasps_mas_update_restrict_warp_nodes_mixed_specialized(
    double* __restrict__ solution, const double* __restrict__ direction,
    double* __restrict__ residual, const double* __restrict__ product,
    const double* __restrict__ state, float* __restrict__ packed,
    const unsigned int* __restrict__ restriction_order,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    unsigned int fine_node_count) {
  const unsigned int ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= fine_node_count || state[7] == -1.0) return;
  const unsigned int fine_node = restriction_order[ordered_node];
  const unsigned int dimension = fine_node_dimensions[fine_node];
  switch (dimension) {
    // YASPS_MAS_GENERATED_UPDATE_RESTRICT_DIMENSION_CASES
    default: break;
  }
}

template <unsigned int DIMENSION>
__device__ __forceinline__ void yasps_mas_update_restrict_one_node_coarse(
    double* __restrict__ solution, const double* __restrict__ direction,
    double* __restrict__ residual, const double* __restrict__ product,
    const double* __restrict__ state, float* __restrict__ packed,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    unsigned int fine_node, unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int input = fine_node_scalar_offsets[fine_node];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int active = __activemask();
  const double alpha = state[5];
  float updated[DIMENSION];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    const unsigned int scalar = input + component;
    solution[scalar] += alpha * direction[scalar];
    const double value = residual[scalar] - alpha * product[scalar];
    residual[scalar] = value;
    updated[component] = static_cast<float>(value);
  }
#pragma unroll
  for (unsigned int level = 1; level < level_count; ++level) {
    const unsigned int map_index = level * fine_node_count + fine_node;
    const yasps_mas_packed_offset_t packed_key =
        fine_node_level_keys[map_index];
    const unsigned int key = yasps_mas_segment_key(packed_key);
    const unsigned int output = yasps_mas_packed_offset(packed_key);
    const unsigned int previous_key = __shfl_up_sync(active, key, 1);
    const bool segment_head = lane == 0 || previous_key != key;
#pragma unroll
    for (unsigned int component = 0; component < DIMENSION; ++component) {
      float value = updated[component];
#pragma unroll
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const unsigned int other_key = __shfl_down_sync(active, key, offset);
        const float other_value = __shfl_down_sync(active, value, offset);
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      if (segment_head) atomicAdd(packed + output + component, value);
    }
  }
}

extern "C" __global__ void
yasps_mas_update_restrict_coarse_warp_nodes_mixed_specialized(
    double* __restrict__ solution, const double* __restrict__ direction,
    double* __restrict__ residual, const double* __restrict__ product,
    const double* __restrict__ state, float* __restrict__ packed,
    const unsigned int* __restrict__ restriction_order,
    const yasps_mas_packed_offset_t* __restrict__ fine_node_level_keys,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    unsigned int fine_node_count) {
  const unsigned int ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= fine_node_count || state[7] == -1.0) return;
  const unsigned int fine_node = restriction_order[ordered_node];
  switch (fine_node_dimensions[fine_node]) {
    // YASPS_MAS_GENERATED_UPDATE_RESTRICT_COARSE_DIMENSION_CASES
    default: break;
  }
}

template <unsigned int DIMENSION>
__device__ __forceinline__ void yasps_mas_restrict_fixed_dimension_body(
    const double* fine, float* packed,
    const unsigned int* restriction_order,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    unsigned int ordered_count, unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= ordered_count) return;
  const unsigned int fine_node = restriction_order[ordered_node];
  const unsigned int input = fine_node_scalar_offsets[fine_node];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int active = __activemask();
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    const float original = static_cast<float>(fine[input + component]);
    packed[fine_node_to_packed_starts[fine_node] + component] = original;
#pragma unroll
    for (unsigned int level = 1; level < level_count; ++level) {
      const unsigned long long key =
          fine_node_to_packed_starts[level * fine_node_count + fine_node]
          + component;
      float value = original;
#pragma unroll
      for (unsigned int offset = 1; offset < 32; offset <<= 1) {
        const unsigned long long other_key =
            __shfl_down_sync(active, key, offset);
        const float other_value = __shfl_down_sync(active, value, offset);
        const bool source_active = lane + offset < 32
            && (active & (1u << (lane + offset)));
        if (source_active && other_key == key) value += other_value;
      }
      const unsigned long long previous_key =
          __shfl_up_sync(active, key, 1);
      if ((lane == 0 || previous_key != key)) atomicAdd(packed + key, value);
    }
  }
}

template <unsigned int DIMENSION>
__device__ __forceinline__ void yasps_mas_collect_fixed_dimension_body(
    const float* packed, double* fine,
    const unsigned int* restriction_order,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    unsigned int ordered_count, unsigned int fine_node_count,
    float coarsest_level_weight) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  const unsigned int ordered_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordered_node >= ordered_count) return;
  const unsigned int fine_node = restriction_order[ordered_node];
  const unsigned int output = fine_node_scalar_offsets[fine_node];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    float sum = 0.0f;
#pragma unroll
    for (unsigned int level = 0; level < level_count; ++level) {
      const unsigned long long input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      const float weight = level + 1 == level_count
          ? coarsest_level_weight : 1.0f;
      sum += weight * packed[input + component];
    }
    fine[output + component] = static_cast<double>(sum);
  }
}

// YASPS_MAS_GENERATED_DIMENSION_KERNELS

template <unsigned int DIMENSION, unsigned int LEVEL_COUNT>
__device__ __forceinline__ void yasps_mas_collect_one_node(
    const float* packed, double* fine,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node, unsigned int fine_node_count,
    const float (&level_weights)[LEVEL_COUNT],
    float duplicate_level_weight, float fine_level_weight) {
  const unsigned int output = fine_node_scalar_offsets[fine_node];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    float sum = 0.0f;
#pragma unroll
    for (unsigned int level = 0; level < LEVEL_COUNT; ++level) {
      const unsigned long long input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      const float topology_weight = fine_node_level_active[
          level * fine_node_count + fine_node]
          ? 1.0f : duplicate_level_weight;
      const float level_weight = level == 0
          ? fine_level_weight : level_weights[level];
      sum += topology_weight * level_weight
          * packed[input + component];
    }
    fine[output + component] = static_cast<double>(sum);
  }
}

template <unsigned int DIMENSION, unsigned int LEVEL_COUNT>
__device__ __forceinline__ void yasps_mas_collect_one_node_dots(
    const float* packed, double* fine, const double* residual,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node, unsigned int fine_node_count,
    const float (&level_weights)[LEVEL_COUNT],
    float duplicate_level_weight, float fine_level_weight,
    double& local_rz, double& local_residual2) {
  const unsigned int output = fine_node_scalar_offsets[fine_node];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    float sum = 0.0f;
#pragma unroll
    for (unsigned int level = 0; level < LEVEL_COUNT; ++level) {
      const unsigned long long input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      const float topology_weight = fine_node_level_active[
          level * fine_node_count + fine_node]
          ? 1.0f : duplicate_level_weight;
      const float level_weight = level == 0
          ? fine_level_weight : level_weights[level];
      sum += topology_weight * level_weight * packed[input + component];
    }
    const double correction = static_cast<double>(sum);
    const double r = residual[output + component];
    fine[output + component] = correction;
    local_rz += r * correction;
    local_residual2 += r * r;
  }
}

extern "C" __global__ void yasps_mas_collect_nodes_mixed_specialized(
    const float* packed, double* fine,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node_count, float fine_level_weight) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  constexpr unsigned int max_dimension = YASPS_MAS_MAX_DIMENSION;
  // These positive weights are part of the compiled preconditioner shape.
  // Keeping them literal lets NVCC unroll the six-level collection without
  // a device-memory lookup or a runtime branch per level.
  constexpr float level_weights[level_count] = {
      // YASPS_MAS_GENERATED_LEVEL_WEIGHTS
  };
  constexpr float duplicate_level_weight =
      YASPS_MAS_GENERATED_DUPLICATE_LEVEL_WEIGHT;
  const unsigned int fine_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (fine_node >= fine_node_count) return;
  const unsigned int output = fine_node_scalar_offsets[fine_node];
  const unsigned int dimension = fine_node_dimensions[fine_node];
  switch (dimension) {
    // YASPS_MAS_GENERATED_COLLECT_DIMENSION_CASES
    default: break;
  }
#pragma unroll
  for (unsigned int component = 0; component < max_dimension; ++component) {
    if (component >= dimension) continue;
    float sum = 0.0f;
#pragma unroll
    for (unsigned int level = 0; level < level_count; ++level) {
      const unsigned long long input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      const float topology_weight = fine_node_level_active[
          level * fine_node_count + fine_node]
          ? 1.0f : duplicate_level_weight;
      const float level_weight = level == 0
          ? fine_level_weight : level_weights[level];
      sum += topology_weight * level_weight
          * packed[input + component];
    }
    fine[output + component] = static_cast<double>(sum);
  }
}

static __device__ __forceinline__ double yasps_mas_warp_sum(
    double value, unsigned int active) {
  for (unsigned int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(active, value, offset);
  return value;
}

extern "C" __global__ void
yasps_mas_collect_nodes_mixed_specialized_dots(
    const float* packed, double* fine, const double* residual, double* state,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node_count, float fine_level_weight) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  constexpr float level_weights[level_count] = {
      // YASPS_MAS_GENERATED_LEVEL_WEIGHTS
  };
  constexpr float duplicate_level_weight =
      YASPS_MAS_GENERATED_DUPLICATE_LEVEL_WEIGHT;
  const unsigned int fine_node = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rz = 0.0;
  double local_residual2 = 0.0;
  if (fine_node < fine_node_count) {
    const unsigned int dimension = fine_node_dimensions[fine_node];
    switch (dimension) {
      // YASPS_MAS_GENERATED_COLLECT_DOT_DIMENSION_CASES
      default: break;
    }
  }
  const unsigned int active = __activemask();
  local_rz = yasps_mas_warp_sum(local_rz, active);
  local_residual2 = yasps_mas_warp_sum(local_residual2, active);
  __shared__ double warp_rz[16];
  __shared__ double warp_residual2[16];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_rz[warp] = local_rz;
    warp_residual2[warp] = local_residual2;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned int warp_count = (blockDim.x + 31u) / 32u;
    double block_rz = lane < warp_count ? warp_rz[lane] : 0.0;
    double block_residual2 = lane < warp_count ? warp_residual2[lane] : 0.0;
    block_rz = yasps_mas_warp_sum(block_rz, __activemask());
    block_residual2 = yasps_mas_warp_sum(
        block_residual2, __activemask());
    if (lane == 0) {
      atomicAdd(state + 3, block_rz);
      atomicAdd(state + 4, block_residual2);
    }
  }
}

template <unsigned int DIMENSION, unsigned int LEVEL_COUNT>
__device__ __forceinline__ void yasps_mas_collect_coarse_one_node(
    const float* packed, double* fine,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node, unsigned int fine_node_count,
    const float (&level_weights)[LEVEL_COUNT],
    float duplicate_level_weight) {
  const unsigned int output = fine_node_scalar_offsets[fine_node];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    float sum = 0.0f;
#pragma unroll
    for (unsigned int level = 1; level < LEVEL_COUNT; ++level) {
      const unsigned long long input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      const float topology_weight = fine_node_level_active[
          level * fine_node_count + fine_node]
          ? 1.0f : duplicate_level_weight;
      sum += topology_weight * level_weights[level]
          * packed[input + component];
    }
    fine[output + component] += static_cast<double>(sum);
  }
}

template <unsigned int DIMENSION, unsigned int LEVEL_COUNT>
__device__ __forceinline__ void yasps_mas_collect_coarse_one_node_dots(
    const float* packed, double* fine, const double* residual,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node, unsigned int fine_node_count,
    const float (&level_weights)[LEVEL_COUNT],
    float duplicate_level_weight, double& local_rz,
    double& local_residual2) {
  const unsigned int output = fine_node_scalar_offsets[fine_node];
#pragma unroll
  for (unsigned int component = 0; component < DIMENSION; ++component) {
    float coarse_sum = 0.0f;
#pragma unroll
    for (unsigned int level = 1; level < LEVEL_COUNT; ++level) {
      const unsigned long long input =
          fine_node_to_packed_starts[level * fine_node_count + fine_node];
      const float topology_weight = fine_node_level_active[
          level * fine_node_count + fine_node]
          ? 1.0f : duplicate_level_weight;
      coarse_sum += topology_weight * level_weights[level]
          * packed[input + component];
    }
    const unsigned int scalar = output + component;
    const double correction = fine[scalar] + static_cast<double>(coarse_sum);
    const double r = residual[scalar];
    fine[scalar] = correction;
    local_rz += r * correction;
    local_residual2 += r * r;
  }
}

extern "C" __global__ void
yasps_mas_collect_coarse_nodes_mixed_specialized(
    const float* packed, double* fine,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  constexpr float level_weights[level_count] = {
      // YASPS_MAS_COARSE_LEVEL_WEIGHTS
  };
  constexpr float duplicate_level_weight =
      YASPS_MAS_GENERATED_DUPLICATE_LEVEL_WEIGHT;
  const unsigned int fine_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (fine_node >= fine_node_count) return;
  switch (fine_node_dimensions[fine_node]) {
    // YASPS_MAS_GENERATED_COLLECT_COARSE_DIMENSION_CASES
    default: break;
  }
}

extern "C" __global__ void
yasps_mas_collect_coarse_nodes_mixed_specialized_dots(
    const float* packed, double* fine, const double* residual, double* state,
    const yasps_mas_packed_offset_t* fine_node_to_packed_starts,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_dimensions,
    const unsigned char* fine_node_level_active,
    unsigned int fine_node_count) {
  constexpr unsigned int level_count = YASPS_MAS_LEVEL_COUNT;
  constexpr float level_weights[level_count] = {
      // YASPS_MAS_COARSE_DOT_LEVEL_WEIGHTS
  };
  constexpr float duplicate_level_weight =
      YASPS_MAS_GENERATED_DUPLICATE_LEVEL_WEIGHT;
  const unsigned int fine_node = blockIdx.x * blockDim.x + threadIdx.x;
  double local_rz = 0.0;
  double local_residual2 = 0.0;
  if (fine_node < fine_node_count) {
    switch (fine_node_dimensions[fine_node]) {
      // YASPS_MAS_GENERATED_COLLECT_COARSE_DOT_DIMENSION_CASES
      default: break;
    }
  }
  const unsigned int active = __activemask();
  local_rz = yasps_mas_warp_sum(local_rz, active);
  local_residual2 = yasps_mas_warp_sum(local_residual2, active);
  __shared__ double warp_rz[16];
  __shared__ double warp_residual2[16];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_rz[warp] = local_rz;
    warp_residual2[warp] = local_residual2;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned int warp_count = (blockDim.x + 31u) / 32u;
    double block_rz = lane < warp_count ? warp_rz[lane] : 0.0;
    double block_residual2 = lane < warp_count ? warp_residual2[lane] : 0.0;
    block_rz = yasps_mas_warp_sum(block_rz, __activemask());
    block_residual2 = yasps_mas_warp_sum(
        block_residual2, __activemask());
    if (lane == 0) {
      atomicAdd(state + 3, block_rz);
      atomicAdd(state + 4, block_residual2);
    }
  }
}

template <int N, int P>
__device__ __forceinline__ void yasps_mas_apply_exact_domain(
    const float* inverses, const float* residuals, float* corrections,
    unsigned long long matrix_offset, unsigned long long vector_offset) {
  const float* inverse = inverses + matrix_offset;
  const float* residual = residuals + vector_offset;
  for (unsigned int row = threadIdx.x; row < N; row += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (unsigned int col = 0; col < N; ++col)
      value = fmaf(inverse[row * P + col], residual[col], value);
    corrections[vector_offset + row] = value;
  }
}

extern "C" __global__ void yasps_mas_dense_inverse_apply_mixed_specialized(
    const float* inverses, const float* residuals, float* corrections,
    const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    unsigned int domain_count) {
  const unsigned int domain = blockIdx.x;
  if (domain >= domain_count) return;
  const unsigned int key = (sizes[domain] << 8) | padded_sizes[domain];
  switch (key) {
// YASPS_MAS_GENERATED_INVERSE_CASES
    default: break;
  }
}

template <int N, int P>
__device__ __forceinline__ void yasps_mas_apply_exact_domain_warp(
    const float* inverses, const float* residuals, float* corrections,
    unsigned long long matrix_offset, unsigned long long vector_offset,
    unsigned int lane) {
  const float* inverse = inverses + matrix_offset;
  const float* residual = residuals + vector_offset;
  for (unsigned int row = lane; row < N; row += 32u) {
    float value = 0.0f;
#pragma unroll
    for (unsigned int col = 0; col < N; ++col)
      value = fmaf(inverse[row * P + col], residual[col], value);
    corrections[vector_offset + row] = value;
  }
}

template <int N, int P>
__device__ __forceinline__ void yasps_mas_apply_fine_domain_warp(
    const float* inverses, const double* fine_residual,
    double* fine_correction, const unsigned int* packed_to_fine,
    const float* staged_residual,
    unsigned long long matrix_offset, unsigned long long vector_offset,
    unsigned int lane, float fine_level_weight) {
  const float* inverse = inverses + matrix_offset;
  for (unsigned int row = lane; row < N; row += 32u) {
    float value = 0.0f;
#pragma unroll
    for (unsigned int col = 0; col < N; ++col) {
      value = fmaf(
          inverse[row * P + col], staged_residual[col], value);
    }
    fine_correction[packed_to_fine[vector_offset + row]] =
        static_cast<double>(fine_level_weight * value);
  }
}

extern "C" __global__ void
yasps_mas_apply_fine_inverse_warp_domains_specialized(
    const float* inverses, const double* fine_residual,
    double* fine_correction, const unsigned int* packed_to_fine,
    const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    unsigned int fine_domain_count, float fine_level_weight) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps_per_block = blockDim.x >> 5u;
  const unsigned int domain = blockIdx.x * warps_per_block + warp;
  if (domain >= fine_domain_count) return;
  __shared__ float staged[4u * YASPS_MAS_MAX_PADDED_SIZE];
  float* warp_residual = staged + warp * YASPS_MAS_MAX_PADDED_SIZE;
  const unsigned int size = sizes[domain];
  const unsigned long long vector_offset = vector_offsets[domain];
  for (unsigned int col = lane; col < size; col += 32u)
    warp_residual[col] = static_cast<float>(
        fine_residual[packed_to_fine[vector_offset + col]]);
  __syncwarp();
  const unsigned int key = (sizes[domain] << 8) | padded_sizes[domain];
  switch (key) {
// YASPS_MAS_GENERATED_FINE_INVERSE_CASES
    default: break;
  }
}

// Four independent Schwarz banks share one 128-thread block.  The previous
// one-domain/96-thread mapping left two of its three warps completely idle for
// banks that fit within one warp. Size and stride remain compile-time constants.
extern "C" __global__ void
yasps_mas_dense_inverse_apply_warp_domains_specialized(
    const float* inverses, const float* residuals, float* corrections,
    const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    unsigned int domain_count) {
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warps_per_block = blockDim.x >> 5u;
  const unsigned int domain = blockIdx.x * warps_per_block + warp;
  if (domain >= domain_count) return;
  const unsigned int key = (sizes[domain] << 8) | padded_sizes[domain];
  switch (key) {
// YASPS_MAS_GENERATED_WARP_INVERSE_CASES
    default: break;
  }
}

// Four warps cooperate on one generated dense bank. Each eight-lane subgroup
// owns one row and splits its columns, replacing the serial N-FMA lane loop
// with ceil(N/8) FMAs plus a three-step subgroup reduction.
template <int N, int P>
__device__ __forceinline__ void yasps_mas_apply_exact_domain_subwarp(
    const float* inverses, const float* residuals, float* corrections,
    unsigned long long matrix_offset, unsigned long long vector_offset) {
  const float* inverse = inverses + matrix_offset;
  const float* residual = residuals + vector_offset;
  const unsigned int warp = threadIdx.x >> 5;
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int subgroup = lane >> 3;
  const unsigned int subgroup_lane = lane & 7u;
  for (unsigned int row_base = warp * 4u; row_base < N;
       row_base += 16u) {
    const unsigned int row = row_base + subgroup;
    float value = 0.0f;
#pragma unroll
    for (unsigned int col = subgroup_lane; col < N; col += 8u)
      if (row < N)
        value = fmaf(inverse[row * P + col], residual[col], value);
    value += __shfl_down_sync(0xffffffffu, value, 4, 8);
    value += __shfl_down_sync(0xffffffffu, value, 2, 8);
    value += __shfl_down_sync(0xffffffffu, value, 1, 8);
    if (row < N && subgroup_lane == 0)
      corrections[vector_offset + row] = value;
  }
}

extern "C" __global__ void
yasps_mas_dense_inverse_apply_subwarp_domains_specialized(
    const float* inverses, const float* residuals, float* corrections,
    const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    unsigned int domain_count) {
  const unsigned int domain = blockIdx.x;
  if (domain >= domain_count) return;
  const unsigned int key = (sizes[domain] << 8) | padded_sizes[domain];
  switch (key) {
// YASPS_MAS_GENERATED_SUBWARP_INVERSE_CASES
    default: break;
  }
}

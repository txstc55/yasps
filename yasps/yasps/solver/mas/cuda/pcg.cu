#include <cmath>
#include <cstdint>
#include <cuda_device_runtime_api.h>

enum PCGScalar : std::uint32_t {
  PCG_RHS_NORM2 = 0,
  PCG_RZ = 1,
  PCG_CURVATURE = 2,
  PCG_NEXT_RZ = 3,
  PCG_RESIDUAL_NORM2 = 4,
  PCG_ALPHA = 5,
  PCG_BETA = 6,
  PCG_STATUS = 7,
  PCG_STATUS_VALUE = 8,
  PCG_RELATIVE_TOLERANCE = 9,
  PCG_ITERATION = 10,
  PCG_MAX_ITERATIONS = 11,
  PCG_REFERENCE_RZ = 12,
  PCG_SCALAR_COUNT = 13
};

constexpr double PCG_NON_SPD = -1.0;
constexpr double PCG_CONTINUE = 0.0;
constexpr double PCG_CONVERGED = 1.0;
constexpr double PCG_RESTART = 2.0;

static __device__ __forceinline__ double atomic_add_double(
    double* address, double value) {
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

static __device__ __forceinline__ double warp_sum(double value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

extern "C" __global__ void yasps_mas_dot_single(
    const double* left, const double* right, double* output,
    std::uint32_t output_slot, std::uint32_t count) {
  double sum = 0.0;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x)
    sum += left[index] * right[index];
  sum = warp_sum(sum);
  __shared__ double warp_sums[8];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0;
    sum = warp_sum(sum);
    if (lane == 0) atomic_add_double(output + output_slot, sum);
  }
}

extern "C" __global__ void yasps_mas_dot_two(
    const double* left_a, const double* right_a,
    const double* left_b, const double* right_b, double* output,
    std::uint32_t output_slot_a, std::uint32_t output_slot_b,
    std::uint32_t count) {
  double sum_a = 0.0;
  double sum_b = 0.0;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    sum_a += left_a[index] * right_a[index];
    sum_b += left_b[index] * right_b[index];
  }
  sum_a = warp_sum(sum_a);
  sum_b = warp_sum(sum_b);
  __shared__ double warp_sums_a[8];
  __shared__ double warp_sums_b[8];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums_a[warp] = sum_a;
    warp_sums_b[warp] = sum_b;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned int warp_count = (blockDim.x + 31) / 32;
    sum_a = lane < warp_count ? warp_sums_a[lane] : 0.0;
    sum_b = lane < warp_count ? warp_sums_b[lane] : 0.0;
    sum_a = warp_sum(sum_a);
    sum_b = warp_sum(sum_b);
    if (lane == 0) {
      atomic_add_double(output + output_slot_a, sum_a);
      atomic_add_double(output + output_slot_b, sum_b);
    }
  }
}

extern "C" __global__ void yasps_mas_dot_single_partials(
    const double* left, const double* right, double* partials,
    std::uint32_t count) {
  double sum = 0.0;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x)
    sum += left[index] * right[index];
  sum = warp_sum(sum);
  __shared__ double warp_sums[8];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0;
    sum = warp_sum(sum);
    if (lane == 0) partials[blockIdx.x] = sum;
  }
}

extern "C" __global__ void yasps_mas_dot_two_partials(
    const double* left_a, const double* right_a,
    const double* left_b, const double* right_b,
    double* partials_a, double* partials_b, std::uint32_t count) {
  double sum_a = 0.0, sum_b = 0.0;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    sum_a += left_a[index] * right_a[index];
    sum_b += left_b[index] * right_b[index];
  }
  sum_a = warp_sum(sum_a);
  sum_b = warp_sum(sum_b);
  __shared__ double warp_a[8];
  __shared__ double warp_b[8];
  const unsigned int lane = threadIdx.x & 31u;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_a[warp] = sum_a;
    warp_b[warp] = sum_b;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned int warp_count = (blockDim.x + 31) / 32;
    sum_a = lane < warp_count ? warp_a[lane] : 0.0;
    sum_b = lane < warp_count ? warp_b[lane] : 0.0;
    sum_a = warp_sum(sum_a);
    sum_b = warp_sum(sum_b);
    if (lane == 0) {
      partials_a[blockIdx.x] = sum_a;
      partials_b[blockIdx.x] = sum_b;
    }
  }
}

extern "C" __global__ void yasps_mas_residual_from_product(
    const double* rhs, const double* product, double* residual,
    std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) residual[index] = rhs[index] - product[index];
}

extern "C" __global__ void yasps_mas_add_solution_in_place(
    double* destination, const double* correction, std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) destination[index] += correction[index];
}

extern "C" __global__ void yasps_mas_prepare_iteration(double* state) {
  if (blockIdx.x || threadIdx.x) return;
  const double curvature = state[PCG_CURVATURE];
  const double rz = state[PCG_RZ];
  // These two reductions execute later in this iteration. Clearing their
  // accumulators here avoids two scalar-only CUDA launches in every graph.
  state[PCG_NEXT_RZ] = 0.0;
  state[PCG_RESIDUAL_NORM2] = 0.0;
  if (!isfinite(curvature) || !isfinite(rz) || rz <= 0.0) {
    state[PCG_ALPHA] = 0.0;
    state[PCG_STATUS] = PCG_NON_SPD;
    state[PCG_STATUS_VALUE] = curvature;
  } else if (curvature <= 0.0) {
    // Roundoff can destroy conjugacy on very ill-conditioned SPD systems even
    // when both A and the explicitly symmetrized preconditioner remain SPD.
    // A zero-length step followed by beta=0 restarts from M^-1 r entirely on
    // device instead of reporting a false non-SPD breakdown.
    state[PCG_ALPHA] = 0.0;
    state[PCG_STATUS] = PCG_RESTART;
  } else {
    state[PCG_ALPHA] = rz / curvature;
    state[PCG_STATUS] = PCG_CONTINUE;
  }
}

extern "C" __global__ void yasps_mas_prepare_iteration_partials(
    double* state, const double* partials, std::uint32_t partial_count) {
  if (blockIdx.x || threadIdx.x) return;
  double curvature = 0.0;
  for (std::uint32_t index = 0; index < partial_count; ++index)
    curvature += partials[index];
  state[PCG_CURVATURE] = curvature;
  const double rz = state[PCG_RZ];
  state[PCG_NEXT_RZ] = 0.0;
  state[PCG_RESIDUAL_NORM2] = 0.0;
  if (!isfinite(curvature) || !isfinite(rz) || rz <= 0.0) {
    state[PCG_ALPHA] = 0.0;
    state[PCG_STATUS] = PCG_NON_SPD;
    state[PCG_STATUS_VALUE] = curvature;
  } else if (curvature <= 0.0) {
    state[PCG_ALPHA] = 0.0;
    state[PCG_STATUS] = PCG_RESTART;
  } else {
    state[PCG_ALPHA] = rz / curvature;
    state[PCG_STATUS] = PCG_CONTINUE;
  }
}

extern "C" __global__ void yasps_mas_update_solution_residual(
    double* solution, const double* direction, double* residual,
    const double* product, const double* state, std::uint32_t count) {
  if (state[PCG_STATUS] == PCG_NON_SPD) return;
  const double alpha = state[PCG_ALPHA];
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    solution[index] += alpha * direction[index];
    residual[index] -= alpha * product[index];
  }
}

extern "C" __global__ void yasps_mas_finish_iteration(double* state) {
  if (blockIdx.x || threadIdx.x || state[PCG_STATUS] == PCG_NON_SPD) return;
  const bool restart = state[PCG_STATUS] == PCG_RESTART;
  const double rz = state[PCG_RZ];
  const double next_rz = state[PCG_NEXT_RZ];
  const double residual2 = state[PCG_RESIDUAL_NORM2];
  if (!isfinite(next_rz) || next_rz < 0.0 ||
      !isfinite(residual2) || residual2 < 0.0 ||
      !isfinite(rz) || rz <= 0.0) {
    state[PCG_BETA] = 0.0;
    state[PCG_STATUS] = PCG_NON_SPD;
    state[PCG_STATUS_VALUE] = next_rz;
    return;
  }
  state[PCG_BETA] = restart ? 0.0 : next_rz / rz;
  state[PCG_RZ] = next_rz;
  state[PCG_STATUS_VALUE] = residual2;
  const double denominator = fmax(state[PCG_REFERENCE_RZ], 1.0e-300);
  state[PCG_STATUS] = next_rz
          <= state[PCG_RELATIVE_TOLERANCE] * denominator
      ? PCG_CONVERGED : PCG_CONTINUE;
}

extern "C" __global__ void yasps_mas_finish_iteration_partials(
    double* state, const double* partials_rz,
    const double* partials_residual2, std::uint32_t partial_count,
    unsigned int checked) {
  if (blockIdx.x || threadIdx.x || state[PCG_STATUS] == PCG_NON_SPD) return;
  const bool restart = state[PCG_STATUS] == PCG_RESTART;
  double next_rz = 0.0, residual2 = 0.0;
  for (std::uint32_t index = 0; index < partial_count; ++index) {
    next_rz += partials_rz[index];
    residual2 += partials_residual2[index];
  }
  const double rz = state[PCG_RZ];
  state[PCG_NEXT_RZ] = next_rz;
  state[PCG_RESIDUAL_NORM2] = residual2;
  if (checked && (!isfinite(next_rz) || next_rz < 0.0 ||
                  !isfinite(residual2) || residual2 < 0.0 ||
                  !isfinite(rz) || rz <= 0.0)) {
    state[PCG_BETA] = 0.0;
    state[PCG_STATUS] = PCG_NON_SPD;
    state[PCG_STATUS_VALUE] = next_rz;
    return;
  }
  state[PCG_BETA] = restart ? 0.0 : next_rz / rz;
  state[PCG_RZ] = next_rz;
  state[PCG_STATUS_VALUE] = residual2;
  if (checked) {
    const double denominator = fmax(state[PCG_REFERENCE_RZ], 1.0e-300);
    state[PCG_STATUS] = next_rz
            <= state[PCG_RELATIVE_TOLERANCE] * denominator
        ? PCG_CONVERGED : PCG_CONTINUE;
  }
}

extern "C" __global__ void yasps_mas_update_direction(
    const double* preconditioned, double* direction, double* state,
    double* next_product, float* next_packed_residual,
    std::uint32_t count, std::uint32_t packed_count) {
  // No thread reads curvature during this kernel, so thread zero can prepare
  // the accumulator for the next iteration without a separate fill launch.
  if (blockIdx.x == 0 && threadIdx.x == 0)
    state[PCG_CURVATURE] = 0.0;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) next_product[index] = 0.0;
  const std::uint32_t coarse_count = packed_count - count;
  if (index < coarse_count)
    next_packed_residual[count + index] = 0.0f;
  if (state[PCG_STATUS] == PCG_NON_SPD) return;
  const double beta = state[PCG_BETA];
  if (index < count)
    direction[index] = preconditioned[index] + beta * direction[index];
}

extern "C" __global__ void yasps_mas_initialize_recurrence(
    double* state, const double* rhs, const double* residual,
    const double* preconditioned, double* direction, std::uint32_t count,
    std::uint32_t residual_is_rhs) {
  double rhs2 = 0.0;
  double residual2 = 0.0;
  double rz = 0.0;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const double r = residual[index];
    rhs2 += rhs[index] * rhs[index];
    residual2 += r * r;
    rz += r * preconditioned[index];
    direction[index] = preconditioned[index];
  }
  rhs2 = warp_sum(rhs2);
  residual2 = warp_sum(residual2);
  rz = warp_sum(rz);
  __shared__ double warp_rhs2[8];
  __shared__ double warp_residual2[8];
  __shared__ double warp_rz[8];
  const unsigned int lane = threadIdx.x & 31;
  const unsigned int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_rhs2[warp] = rhs2;
    warp_residual2[warp] = residual2;
    warp_rz[warp] = rz;
  }
  __syncthreads();
  if (warp == 0) {
    const unsigned int warp_count = (blockDim.x + 31) / 32;
    rhs2 = lane < warp_count ? warp_rhs2[lane] : 0.0;
    residual2 = lane < warp_count ? warp_residual2[lane] : 0.0;
    rz = lane < warp_count ? warp_rz[lane] : 0.0;
    rhs2 = warp_sum(rhs2);
    residual2 = warp_sum(residual2);
    rz = warp_sum(rz);
    if (lane == 0) {
      atomic_add_double(state + PCG_RHS_NORM2, rhs2);
      atomic_add_double(state + PCG_RESIDUAL_NORM2, residual2);
      atomic_add_double(state + PCG_RZ, rz);
      if (residual_is_rhs)
        atomic_add_double(state + PCG_REFERENCE_RZ, rz);
    }
  }
}

// Reproduce GIPC/YASPS's fixed-budget fast path: all recurrence scalars stay
// on device and the host launches the next iteration without a status copy.
// The checked path still copies one two-double packet when tolerance stopping
// or breakdown reporting is requested.
extern "C" __global__ void yasps_mas_finish_iteration_unchecked(
    double* state) {
  if (blockIdx.x || threadIdx.x) return;
  const double rz = state[PCG_RZ];
  const double next_rz = state[PCG_NEXT_RZ];
  state[PCG_BETA] = state[PCG_STATUS] == PCG_RESTART
      ? 0.0 : next_rz / rz;
  state[PCG_RZ] = next_rz;
  state[PCG_STATUS_VALUE] = state[PCG_RESIDUAL_NORM2];
}

// The body of a CUDA conditional-graph while loop calls this after one full
// recurrence. Convergence, breakdown, and the iteration budget are evaluated
// on device, so the host reads one completion packet per solve instead of one
// packet (and stream synchronization) per PCG iteration.
extern "C" __global__ void yasps_mas_update_pcg_loop(
    unsigned long long conditional_handle, double* state) {
  if (blockIdx.x || threadIdx.x) return;
  state[PCG_ITERATION] += 1.0;
  const bool keep_running =
      state[PCG_STATUS] == PCG_CONTINUE &&
      state[PCG_ITERATION] < state[PCG_MAX_ITERATIONS];
  cudaGraphSetConditional(
      static_cast<cudaGraphConditionalHandle>(conditional_handle),
      keep_running ? 1u : 0u);
}

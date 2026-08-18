#include <cmath>

#if !defined(YASPS_MAS_ACTIVE_SIZE) || !defined(YASPS_MAS_STORAGE_STRIDE) || \
    !defined(YASPS_MAS_INVERSE_GROUPS)
#error "exact inverse dimensions must be supplied at compile time"
#endif

// Generated Gauss-Jordan inverse for an active N x N principal matrix stored
// in a P-stride conservative bank.  Unoccupied coordinates are written as an
// identity complement, so an underfilled domain performs only its active
// pivot sweeps. Multiple small domains share one CUDA block.
extern "C" __global__ void yasps_mas_inverse_gj_exact_strided(
    const double* input, float* output, const int* sizes, int* status,
    int matrix_count, double pivot_tolerance, int* any_failure) {
  constexpr int N = YASPS_MAS_ACTIVE_SIZE;
  constexpr int P = YASPS_MAS_STORAGE_STRIDE;
  constexpr int GROUPS = YASPS_MAS_INVERSE_GROUPS;
  const int group = threadIdx.x / N;
  const int lane = threadIdx.x - group * N;
  const int matrix_id = blockIdx.x * GROUPS + group;
  const bool active = group < GROUPS && matrix_id < matrix_count;
  extern __shared__ double shared[];
  double* matrix = shared + group * (N * N + N);
  double* column = matrix + N * N;
  double smallest_pivot = 1.7976931348623157e+308;
  double largest_pivot = 0.0;
  if (active) {
    if (lane == 0) status[matrix_id] = 0;
    const double* source = input + static_cast<unsigned long long>(matrix_id) * P * P;
    for (int row = 0; row < N; ++row)
      matrix[row * N + lane] = source[row * P + lane];
  }
  __syncthreads();

#pragma unroll 1
  for (int pivot = 0; pivot < N; ++pivot) {
    const double diagonal = active ? matrix[pivot * N + pivot] : 1.0;
    if (active && lane == 0) {
      const double magnitude = fabs(diagonal);
      smallest_pivot = fmin(smallest_pivot, magnitude);
      largest_pivot = fmax(largest_pivot, magnitude);
    }
    if (active) {
      column[lane] = matrix[lane * N + pivot];
      matrix[lane * N + pivot] = lane == pivot ? 1.0 : 0.0;
    }
    __syncthreads();
    if (active) matrix[pivot * N + lane] /= diagonal;
    __syncthreads();
    if (active) {
      const double pivot_value = matrix[pivot * N + lane];
#pragma unroll 1
      for (int row = 0; row < N; ++row)
        if (row != pivot)
          matrix[row * N + lane] -= column[row] * pivot_value;
    }
    __syncthreads();
  }

  if (active) {
    float* destination = output
        + static_cast<unsigned long long>(matrix_id) * P * P;
    // The bank may be conservatively overallocated. Initialize its inactive
    // complement without making it participate in the O(N^3) inverse.
    for (int entry = lane; entry < P * P; entry += N) {
      const int row = entry / P;
      const int col = entry - row * P;
      destination[entry] = (row == col && row >= N) ? 1.0f : 0.0f;
    }
    bool finite = true;
    for (int row = 0; row < N; ++row) {
      const double value = 0.5 * (
          matrix[row * N + lane] + matrix[lane * N + row]);
      finite = finite && isfinite(value);
      destination[row * P + lane] = static_cast<float>(value);
    }
    if (!finite) {
      atomicExch(status + matrix_id, 2);
      atomicExch(any_failure, 1);
    }
    if (lane == 0 && (
        !isfinite(smallest_pivot) || !isfinite(largest_pivot)
        || smallest_pivot <= pivot_tolerance
        || smallest_pivot <= largest_pivot * 1.0e-7)) {
      atomicExch(status + matrix_id, 2);
      atomicExch(any_failure, 1);
    }
  }
  (void)sizes;
}

// One block handles one exceptional underfilled bank.  The active dimensions
// are compile-time constants while P remains the conservative storage stride,
// so an underfilled bank factors only its populated principal matrix.
extern "C" __global__ void yasps_mas_inverse_spd_exact_fallback(
    const double* input, float* output, const int* sizes, int* status,
    int matrix_count, double pivot_tolerance, int* any_failure) {
  constexpr int N = YASPS_MAS_ACTIVE_SIZE;
  constexpr int P = YASPS_MAS_STORAGE_STRIDE;
  const int matrix_id = blockIdx.x;
  if (matrix_id >= matrix_count || status[matrix_id] == 0) return;
  const int tid = threadIdx.x;
  extern __shared__ double lower[];
  const double* source = input
      + static_cast<unsigned long long>(matrix_id) * P * P;
  float* destination = output
      + static_cast<unsigned long long>(matrix_id) * P * P;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < N * N; entry += blockDim.x) {
    const int row = entry / N;
    const int col = entry - row * N;
    lower[entry] = source[row * P + col];
  }
  __syncthreads();

#pragma unroll 1
  for (int k = 0; k < N; ++k) {
    if (tid == 0) {
      double diagonal = lower[k * N + k];
#pragma unroll 1
      for (int s = 0; s < k; ++s)
        diagonal -= lower[k * N + s] * lower[k * N + s];
      if (!isfinite(diagonal) || diagonal <= pivot_tolerance) {
        status[matrix_id] = k + 1;
        atomicExch(any_failure, 1);
      } else {
        lower[k * N + k] = sqrt(diagonal);
      }
    }
    __syncthreads();
    if (status[matrix_id]) return;
    for (int row = k + 1 + tid; row < N; row += blockDim.x) {
      double value = lower[row * N + k];
#pragma unroll 1
      for (int s = 0; s < k; ++s)
        value -= lower[row * N + s] * lower[k * N + s];
      lower[row * N + k] = value / lower[k * N + k];
    }
    __syncthreads();
  }

  if (tid < N) {
    double column[N];
#pragma unroll 1
    for (int row = 0; row < N; ++row) {
      double value = row == tid ? 1.0 : 0.0;
#pragma unroll 1
      for (int s = 0; s < row; ++s)
        value -= lower[row * N + s] * column[s];
      column[row] = value / lower[row * N + row];
    }
#pragma unroll 1
    for (int row = N - 1; row >= 0; --row) {
      double value = column[row];
#pragma unroll 1
      for (int s = row + 1; s < N; ++s)
        value -= lower[s * N + row] * column[s];
      column[row] = value / lower[row * N + row];
    }
    for (int row = 0; row < N; ++row)
      destination[row * P + tid] = static_cast<float>(column[row]);
  }
  __syncthreads();
  for (int entry = tid; entry < P * P; entry += blockDim.x) {
    const int row = entry / P;
    const int col = entry - row * P;
    if (row >= N || col >= N)
      destination[entry] = row == col ? 1.0f : 0.0f;
  }
  for (int row = tid; row < N; row += blockDim.x) {
    for (int col = row + 1; col < N; ++col) {
      const float value = 0.5f * (
          destination[row * P + col] + destination[col * P + row]);
      destination[row * P + col] = value;
      destination[col * P + row] = value;
    }
  }
  (void)sizes;
}

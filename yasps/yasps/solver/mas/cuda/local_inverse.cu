#include <cmath>
#include <cstdint>

// One CUDA block cooperates on one padded FP64 matrix. Padding must be an
// identity block, never a zero block. Status is nonzero on a bad pivot.
extern "C" __global__ void yasps_mas_inverse_gauss_jordan(
    const double* input, double* output, const int* sizes, int* status,
    int n, double pivot_tolerance, int* any_failure) {
  extern __shared__ double shared[];
  double* a = shared;
  double* inverse = a + n * n;
  double* factors = inverse + n * n;
  const int matrix_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int total = n * n;
  const double* source = input + matrix_id * total;
  double* destination = output + matrix_id * total;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < total; entry += blockDim.x) {
    a[entry] = source[entry];
    inverse[entry] = (entry / n == entry % n) ? 1.0 : 0.0;
  }
  __syncthreads();
  for (int pivot = 0; pivot < n; ++pivot) {
    const double diagonal = a[pivot * n + pivot];
    if (tid == 0 && (!isfinite(diagonal) || fabs(diagonal) <= pivot_tolerance)) {
      status[matrix_id] = pivot + 1;
      atomicExch(any_failure, 1);
    }
    __syncthreads();
    if (status[matrix_id]) return;
    for (int col = tid; col < n; col += blockDim.x) {
      a[pivot * n + col] /= diagonal;
      inverse[pivot * n + col] /= diagonal;
    }
    for (int row = tid; row < n; row += blockDim.x)
      factors[row] = (row == pivot) ? 0.0 : a[row * n + pivot];
    __syncthreads();
    for (int entry = tid; entry < total; entry += blockDim.x) {
      const int row = entry / n;
      const int col = entry % n;
      if (row != pivot) {
        a[entry] -= factors[row] * a[pivot * n + col];
        inverse[entry] -= factors[row] * inverse[pivot * n + col];
      }
    }
    __syncthreads();
  }
  for (int entry = tid; entry < total; entry += blockDim.x)
    destination[entry] = inverse[entry];
}

// Mixed-storage variant: all elimination remains FP64, but the reusable
// preconditioner bank is written directly as FP32.  This removes the full-size
// temporary inverse arena and the separate post-inversion cast kernel.
extern "C" __global__ void yasps_mas_inverse_gauss_jordan_mixed(
    const double* input, float* output, const int* sizes, int* status,
    int n, double pivot_tolerance, int* any_failure) {
  extern __shared__ double shared[];
  double* a = shared;
  double* inverse = a + n * n;
  double* factors = inverse + n * n;
  const int matrix_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int total = n * n;
  const double* source = input + matrix_id * total;
  float* destination = output + matrix_id * total;
  (void)sizes;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < total; entry += blockDim.x) {
    a[entry] = source[entry];
    inverse[entry] = entry / n == entry % n ? 1.0 : 0.0;
  }
  __syncthreads();
  for (int pivot = 0; pivot < n; ++pivot) {
    const double diagonal = a[pivot * n + pivot];
    if (tid == 0 && (!isfinite(diagonal) || fabs(diagonal) <= pivot_tolerance)) {
      status[matrix_id] = pivot + 1;
      atomicExch(any_failure, 1);
    }
    __syncthreads();
    if (status[matrix_id]) return;
    for (int col = tid; col < n; col += blockDim.x) {
      a[pivot * n + col] /= diagonal;
      inverse[pivot * n + col] /= diagonal;
    }
    for (int row = tid; row < n; row += blockDim.x)
      factors[row] = row == pivot ? 0.0 : a[row * n + pivot];
    __syncthreads();
    for (int entry = tid; entry < total; entry += blockDim.x) {
      const int row = entry / n;
      const int col = entry % n;
      if (row != pivot) {
        a[entry] -= factors[row] * a[pivot * n + col];
        inverse[entry] -= factors[row] * inverse[pivot * n + col];
      }
    }
    __syncthreads();
  }
  for (int entry = tid; entry < total; entry += blockDim.x)
    destination[entry] = static_cast<float>(inverse[entry]);
}

// Transform A into A^-1 in place so a generated instantiation can choose as
// many independent banks per CUDA block as its runtime dimension permits.
template <int N, int GROUPS>
__device__ __forceinline__ void yasps_mas_inverse_gj_packed_body(
    const double* input, float* output, int* status, int matrix_count,
    double pivot_tolerance, int* any_failure, double* shared) {
  const int group = threadIdx.x / N;
  const int lane = threadIdx.x - group * N;
  const int matrix_id = blockIdx.x * GROUPS + group;
  const bool active = group < GROUPS && matrix_id < matrix_count;
  double* matrix = shared + group * (N * N + N);
  double* column = matrix + N * N;
  const double* source = active ? input + matrix_id * N * N : input;
  double smallest_pivot = 1.7976931348623157e+308;
  double largest_pivot = 0.0;
  if (active) {
    if (lane == 0) status[matrix_id] = 0;
    for (int row = 0; row < N; ++row)
      matrix[row * N + lane] = source[row * N + lane];
  }
  __syncthreads();

  for (int pivot = 0; pivot < N; ++pivot) {
    // The caller's CG contract guarantees SPD banks. Avoid a status broadcast
    // at every pivot.  Lane zero still tracks the Schur-pivot spread: once it
    // exceeds what FP32 inverse storage can safely represent, a second kernel
    // replaces only that bank with the more stable Cholesky result.
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
      for (int row = 0; row < N; ++row)
        if (row != pivot)
          matrix[row * N + lane] -= column[row] * pivot_value;
    }
    __syncthreads();
  }
  if (active) {
    float* destination = output + matrix_id * N * N;
    bool finite = true;
    for (int row = 0; row < N; ++row) {
      const double value = 0.5 * (
          matrix[row * N + lane] + matrix[lane * N + row]);
      finite = finite && isfinite(value);
      destination[row * N + lane] = static_cast<float>(value);
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
}

#if defined(YASPS_MAS_INVERSE_SIZE) && defined(YASPS_MAS_INVERSE_GROUPS)

extern "C" __global__ void yasps_mas_inverse_gj_packed_specialized(
    const double* input, float* output, const int* sizes, int* status,
    int matrix_count, double pivot_tolerance, int* any_failure) {
  extern __shared__ double shared[];
  (void)sizes;
  yasps_mas_inverse_gj_packed_body<
      YASPS_MAS_INVERSE_SIZE, YASPS_MAS_INVERSE_GROUPS>(
      input, output, status, matrix_count, pivot_tolerance, any_failure,
      shared);
}

#endif

extern "C" __global__ void yasps_mas_inverse_spd(
    const double* input, double* output, const int* sizes, int* status,
    int n, double pivot_tolerance, int* any_failure) {
  extern __shared__ double shared[];
  double* lower = shared;
  double* inverse = lower + n * n;
  const int matrix_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int total = n * n;
  const double* source = input + matrix_id * total;
  double* destination = output + matrix_id * total;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < total; entry += blockDim.x) {
    lower[entry] = source[entry];
    inverse[entry] = 0.0;
  }
  __syncthreads();
  for (int k = 0; k < n; ++k) {
    if (tid == 0) {
      double diagonal = lower[k * n + k];
      for (int s = 0; s < k; ++s) diagonal -= lower[k * n + s] * lower[k * n + s];
      if (!isfinite(diagonal) || diagonal <= pivot_tolerance) {
        status[matrix_id] = k + 1;
        atomicExch(any_failure, 1);
      }
      else lower[k * n + k] = sqrt(diagonal);
    }
    __syncthreads();
    if (status[matrix_id]) return;
    for (int i = k + 1 + tid; i < n; i += blockDim.x) {
      double value = lower[i * n + k];
      for (int s = 0; s < k; ++s) value -= lower[i * n + s] * lower[k * n + s];
      lower[i * n + k] = value / lower[k * n + k];
    }
    __syncthreads();
  }
  // Each participating thread solves one or more identity right-hand sides.
  for (int rhs = tid; rhs < n; rhs += blockDim.x) {
    for (int row = 0; row < n; ++row) {
      double value = (row == rhs) ? 1.0 : 0.0;
      for (int s = 0; s < row; ++s) value -= lower[row * n + s] * inverse[s * n + rhs];
      inverse[row * n + rhs] = value / lower[row * n + row];
    }
    for (int row = n - 1; row >= 0; --row) {
      double value = inverse[row * n + rhs];
      for (int s = row + 1; s < n; ++s) value -= lower[s * n + row] * inverse[s * n + rhs];
      inverse[row * n + rhs] = value / lower[row * n + row];
    }
  }
  __syncthreads();
  for (int entry = tid; entry < total; entry += blockDim.x)
    destination[entry] = inverse[entry];
}

// The solver applies local inverses in mixed precision, following GIPC's
// storage pattern. Keep factorization and triangular solves in FP64, then
// write the final bank directly as FP32 so numeric updates do not materialize
// and reread a throwaway FP64 inverse array.
extern "C" __global__ void yasps_mas_inverse_spd_mixed(
    const double* input, float* output, const int* sizes, int* status,
    int n, double pivot_tolerance, int* any_failure) {
  extern __shared__ double shared[];
  double* lower = shared;
  double* inverse = lower + n * n;
  const int matrix_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int total = n * n;
  const double* source = input + matrix_id * total;
  float* destination = output + matrix_id * total;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < total; entry += blockDim.x) {
    lower[entry] = source[entry];
    inverse[entry] = 0.0;
  }
  __syncthreads();
  for (int k = 0; k < n; ++k) {
    if (tid == 0) {
      double diagonal = lower[k * n + k];
      for (int s = 0; s < k; ++s)
        diagonal -= lower[k * n + s] * lower[k * n + s];
      if (!isfinite(diagonal) || diagonal <= pivot_tolerance) {
        status[matrix_id] = k + 1;
        atomicExch(any_failure, 1);
      }
      else lower[k * n + k] = sqrt(diagonal);
    }
    __syncthreads();
    if (status[matrix_id]) return;
    for (int i = k + 1 + tid; i < n; i += blockDim.x) {
      double value = lower[i * n + k];
      for (int s = 0; s < k; ++s)
        value -= lower[i * n + s] * lower[k * n + s];
      lower[i * n + k] = value / lower[k * n + k];
    }
    __syncthreads();
  }
  for (int rhs = tid; rhs < n; rhs += blockDim.x) {
    for (int row = 0; row < n; ++row) {
      double value = (row == rhs) ? 1.0 : 0.0;
      for (int s = 0; s < row; ++s)
        value -= lower[row * n + s] * inverse[s * n + rhs];
      inverse[row * n + rhs] = value / lower[row * n + row];
    }
    for (int row = n - 1; row >= 0; --row) {
      double value = inverse[row * n + rhs];
      for (int s = row + 1; s < n; ++s)
        value -= lower[s * n + row] * inverse[s * n + rhs];
      inverse[row * n + rhs] = value / lower[row * n + row];
    }
  }
  __syncthreads();
  for (int entry = tid; entry < total; entry += blockDim.x)
    destination[entry] = static_cast<float>(inverse[entry]);
}

// Fixed-size specialization used by the immutable inverse buckets.  Keeping
// only the Cholesky factor in shared memory halves the per-domain shared-memory
// footprint.  One lane owns one identity RHS in thread-local storage, so every
// launched warp performs useful triangular-solve work instead of reserving 96
// threads for runtime-sized domains that fit within one warp.
template <int N>
__device__ __forceinline__ void yasps_mas_inverse_spd_mixed_fixed_body(
    const double* input, float* output, int* status,
    double pivot_tolerance, int* any_failure, double* lower) {
  const int matrix_id = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int total = N * N;
  const double* source = input + matrix_id * total;
  float* destination = output + matrix_id * total;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < total; entry += blockDim.x)
    lower[entry] = source[entry];
  __syncthreads();

  for (int k = 0; k < N; ++k) {
    if (tid == 0) {
      double diagonal = lower[k * N + k];
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
      for (int s = 0; s < k; ++s)
        value -= lower[row * N + s] * lower[k * N + s];
      lower[row * N + k] = value / lower[k * N + k];
    }
    __syncthreads();
  }

  if (tid < N) {
    double column[N];
    for (int row = 0; row < N; ++row) {
      double value = row == tid ? 1.0 : 0.0;
      for (int s = 0; s < row; ++s)
        value -= lower[row * N + s] * column[s];
      column[row] = value / lower[row * N + row];
    }
    for (int row = N - 1; row >= 0; --row) {
      double value = column[row];
      for (int s = row + 1; s < N; ++s)
        value -= lower[s * N + row] * column[s];
      column[row] = value / lower[row * N + row];
    }
    for (int row = 0; row < N; ++row)
      destination[row * N + tid] = static_cast<float>(column[row]);
  }
}

#if defined(YASPS_MAS_INVERSE_SIZE)

extern "C" __global__ void yasps_mas_inverse_spd_mixed_fixed_specialized(
    const double* input, float* output, const int* sizes, int* status,
    int runtime_n, double pivot_tolerance, int* any_failure) {
  extern __shared__ double lower[];
  (void)sizes;
  (void)runtime_n;
  yasps_mas_inverse_spd_mixed_fixed_body<YASPS_MAS_INVERSE_SIZE>(
      input, output, status, pivot_tolerance, any_failure, lower);
}

#endif

// Hybrid inverse fallback. Packed Gauss-Jordan is substantially faster for
// normal banks, but an ill-conditioned SPD bank can lose definiteness when its
// explicit inverse is rounded to FP32. Status==2 marks just those banks. Pack
// the same number of banks per block as the fast path, then let a completely
// unmarked block exit after one vote. This avoids launching one empty block for
// every Schwarz domain on the overwhelmingly common path.
template <int N, int GROUPS>
__device__ __forceinline__ void yasps_mas_inverse_spd_packed_fallback_body(
    const double* input, float* output, int* status, int matrix_count,
    double pivot_tolerance, int* any_failure, double* shared) {
  const int group = threadIdx.x / N;
  const int lane = threadIdx.x - group * N;
  const int matrix_id = blockIdx.x * GROUPS + group;
  const bool exists = group < GROUPS && matrix_id < matrix_count;
  const bool marked = exists && status[matrix_id] != 0;
  if (__syncthreads_count(marked) == 0) return;
  double* lower = shared + group * N * N;
  if (marked) {
    if (lane == 0) status[matrix_id] = 0;
    const double* source = input + matrix_id * N * N;
    for (int row = 0; row < N; ++row)
      lower[row * N + lane] = source[row * N + lane];
  }
  __syncthreads();

#pragma unroll 1
  for (int k = 0; k < N; ++k) {
    if (marked && lane == 0 && status[matrix_id] == 0) {
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
    if (marked && status[matrix_id] == 0 && lane > k) {
      double value = lower[lane * N + k];
#pragma unroll 1
      for (int s = 0; s < k; ++s)
        value -= lower[lane * N + s] * lower[k * N + s];
      lower[lane * N + k] = value / lower[k * N + k];
    }
    __syncthreads();
  }

  if (marked && status[matrix_id] == 0) {
    double column[N];
#pragma unroll 1
    for (int row = 0; row < N; ++row) {
      double value = row == lane ? 1.0 : 0.0;
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
    float* destination = output + matrix_id * N * N;
    for (int row = 0; row < N; ++row)
      destination[row * N + lane] = static_cast<float>(column[row]);
  }
  __syncthreads();
  if (marked && status[matrix_id] == 0) {
    float* destination = output + matrix_id * N * N;
    for (int col = lane + 1; col < N; ++col) {
      const float value = 0.5f * (
          destination[lane * N + col] + destination[col * N + lane]);
      destination[lane * N + col] = value;
      destination[col * N + lane] = value;
    }
  }
}

#if defined(YASPS_MAS_INVERSE_SIZE) && defined(YASPS_MAS_INVERSE_GROUPS)

extern "C" __global__ void yasps_mas_inverse_spd_mixed_fallback_specialized(
    const double* input, float* output, const int* sizes, int* status,
    int matrix_count, double pivot_tolerance, int* any_failure) {
  extern __shared__ double shared[];
  (void)sizes;
  yasps_mas_inverse_spd_packed_fallback_body<
      YASPS_MAS_INVERSE_SIZE, YASPS_MAS_INVERSE_GROUPS>(
      input, output, status, matrix_count, pivot_tolerance, any_failure,
      shared);
}

#endif

// Ragged version of the mixed SPD inverse. One launch covers all hierarchy
// domains, including heterogeneous padded sizes. Matrix offsets are static,
// while factorization and inverse construction remain FP64 and only the final
// bank is stored as FP32.
extern "C" __global__ void yasps_mas_inverse_spd_mixed_ragged(
    const double* input, float* output,
    const unsigned long long* matrix_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    int* status, unsigned int domain_count,
    double pivot_tolerance, int* any_failure) {
  const unsigned int matrix_id = blockIdx.x;
  if (matrix_id >= domain_count) return;
  extern __shared__ double shared[];
  const int n = static_cast<int>(padded_sizes[matrix_id]);
  double* lower = shared;
  double* inverse = lower + n * n;
  const int tid = threadIdx.x;
  const int total = n * n;
  const unsigned long long matrix_offset = matrix_offsets[matrix_id];
  const double* source = input + matrix_offset;
  float* destination = output + matrix_offset;
  if (tid == 0) status[matrix_id] = 0;
  for (int entry = tid; entry < total; entry += blockDim.x) {
    lower[entry] = source[entry];
    inverse[entry] = 0.0;
  }
  __syncthreads();
  for (int k = 0; k < n; ++k) {
    if (tid == 0) {
      double diagonal = lower[k * n + k];
      for (int s = 0; s < k; ++s)
        diagonal -= lower[k * n + s] * lower[k * n + s];
      if (!isfinite(diagonal) || diagonal <= pivot_tolerance) {
        status[matrix_id] = k + 1;
        atomicExch(any_failure, 1);
      } else {
        lower[k * n + k] = sqrt(diagonal);
      }
    }
    __syncthreads();
    if (status[matrix_id]) return;
    for (int i = k + 1 + tid; i < n; i += blockDim.x) {
      double value = lower[i * n + k];
      for (int s = 0; s < k; ++s)
        value -= lower[i * n + s] * lower[k * n + s];
      lower[i * n + k] = value / lower[k * n + k];
    }
    __syncthreads();
  }
  for (int rhs = tid; rhs < n; rhs += blockDim.x) {
    for (int row = 0; row < n; ++row) {
      double value = row == rhs ? 1.0 : 0.0;
      for (int s = 0; s < row; ++s)
        value -= lower[row * n + s] * inverse[s * n + rhs];
      inverse[row * n + rhs] = value / lower[row * n + row];
    }
    for (int row = n - 1; row >= 0; --row) {
      double value = inverse[row * n + rhs];
      for (int s = row + 1; s < n; ++s)
        value -= lower[s * n + row] * inverse[s * n + rhs];
      inverse[row * n + rhs] = value / lower[row * n + row];
    }
  }
  __syncthreads();
  for (int entry = tid; entry < total; entry += blockDim.x)
    destination[entry] = static_cast<float>(inverse[entry]);
}

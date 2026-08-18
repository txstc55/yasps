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

// Scalar transfer indices are precomputed once from the static node map.
extern "C" __global__ void yasps_mas_restrict(
    const double* fine, double* coarse,
    const std::uint32_t* fine_scalar_to_parent_scalar,
    std::uint32_t fine_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar < fine_dofs)
    yasps_mas_atomic_add(coarse + fine_scalar_to_parent_scalar[scalar], fine[scalar]);
}

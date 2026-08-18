#include <cstdint>

extern "C" __global__ void yasps_mas_prolongate_add(
    const double* coarse, double* fine,
    const std::uint32_t* fine_scalar_to_parent_scalar,
    std::uint32_t fine_dofs) {
  const std::uint32_t scalar = blockIdx.x * blockDim.x + threadIdx.x;
  if (scalar < fine_dofs)
    fine[scalar] += coarse[fine_scalar_to_parent_scalar[scalar]];
}

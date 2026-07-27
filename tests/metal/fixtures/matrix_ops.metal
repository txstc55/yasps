#include <metal_stdlib>
#include "metalMatrix.metal"
using namespace metal;

kernel void yasps_test_matrix_ops(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0) {
    return;
  }
  YaspsMatrix<2, 2> left_matrix = {};
  YaspsMatrix<2, 2> right_matrix = {};
  for (uint element = 0; element < 4; ++element) {
    left_matrix.values[element] = left[element];
    right_matrix.values[element] = right[element];
  }
  YaspsMatrix<2, 2> result =
      left_matrix * right_matrix + left_matrix.inverse();
  for (uint element = 0; element < 4; ++element) {
    output[element] = result.values[element];
  }
}

kernel void yasps_test_matrix_guards(
    device const float *input [[buffer(0)]],
    device float *projection [[buffer(1)]],
    device float *inverse [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0) {
    return;
  }
  float local[9];
  float local_inverse_input[9];
  float local_inverse[9];
  for (uint element = 0; element < 9; ++element) {
    local[element] = input[element];
    local_inverse_input[element] = input[element];
  }
  yasps_spd_projection_inplace<3>(local, 2);
  yasps_symmetric_pseudoinverse<3>(
      local_inverse_input,
      local_inverse);
  for (uint element = 0; element < 9; ++element) {
    projection[element] = local[element];
    inverse[element] = local_inverse[element];
  }
}

kernel void yasps_test_projection_12(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0) {
    return;
  }
  float local[144];
  for (uint element = 0; element < 144; ++element) {
    local[element] = input[element];
  }
  yasps_spd_projection_inplace<12>(local, 2);
  for (uint element = 0; element < 144; ++element) {
    output[element] = local[element];
  }
}

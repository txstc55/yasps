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

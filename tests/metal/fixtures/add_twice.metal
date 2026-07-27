#include <metal_stdlib>
using namespace metal;

extern float yasps_test_twice(float value);

kernel void yasps_test_add_twice(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    output[index] = yasps_test_twice(input[index]) + 1.0f;
  }
}

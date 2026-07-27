#include <metal_stdlib>
using namespace metal;

#define YASPS_FILL(TYPE, NAME)                                           \
kernel void yasps_fill_##NAME(                                          \
  device TYPE* output [[buffer(0)]],                                    \
  constant TYPE& value [[buffer(1)]],                                   \
  constant uint& count [[buffer(2)]],                                   \
  uint index [[thread_position_in_grid]]                                \
) {                                                                     \
  if (index < count) {                                                   \
    output[index] = value;                                               \
  }                                                                     \
}

YASPS_FILL(float, float)
YASPS_FILL(int, int)
YASPS_FILL(uint, uint)
YASPS_FILL(long, long)
YASPS_FILL(ulong, ulong)
YASPS_FILL(short, short)
YASPS_FILL(ushort, ushort)
YASPS_FILL(char, char)
YASPS_FILL(uchar, uchar)

#define YASPS_BINARY(TYPE, NAME, OPERATION, OPERATOR)                    \
kernel void yasps_##OPERATION##_##NAME##_array(                         \
  device const TYPE* left [[buffer(0)]],                                \
  device const TYPE* right [[buffer(1)]],                               \
  device TYPE* output [[buffer(2)]],                                    \
  constant uint& count [[buffer(3)]],                                   \
  uint index [[thread_position_in_grid]]                                \
) {                                                                     \
  if (index < count) {                                                   \
    output[index] = left[index] OPERATOR right[index];                   \
  }                                                                     \
}                                                                       \
kernel void yasps_##OPERATION##_##NAME##_scalar(                        \
  device const TYPE* left [[buffer(0)]],                                \
  constant TYPE& right [[buffer(1)]],                                   \
  device TYPE* output [[buffer(2)]],                                    \
  constant uint& count [[buffer(3)]],                                   \
  uint index [[thread_position_in_grid]]                                \
) {                                                                     \
  if (index < count) {                                                   \
    output[index] = left[index] OPERATOR right;                          \
  }                                                                     \
}                                                                       \
kernel void yasps_##OPERATION##_##NAME##_reverse_scalar(                \
  constant TYPE& left [[buffer(0)]],                                    \
  device const TYPE* right [[buffer(1)]],                               \
  device TYPE* output [[buffer(2)]],                                    \
  constant uint& count [[buffer(3)]],                                   \
  uint index [[thread_position_in_grid]]                                \
) {                                                                     \
  if (index < count) {                                                   \
    output[index] = left OPERATOR right[index];                          \
  }                                                                     \
}

#define YASPS_BINARY_SET(TYPE, NAME)                                     \
YASPS_BINARY(TYPE, NAME, add, +)                                        \
YASPS_BINARY(TYPE, NAME, subtract, -)                                   \
YASPS_BINARY(TYPE, NAME, multiply, *)                                   \
YASPS_BINARY(TYPE, NAME, divide, /)

YASPS_BINARY_SET(float, float)
YASPS_BINARY_SET(int, int)
YASPS_BINARY_SET(uint, uint)

#define YASPS_NEGATE(TYPE, NAME)                                        \
kernel void yasps_negate_##NAME(                                       \
  device const TYPE* input [[buffer(0)]],                               \
  device TYPE* output [[buffer(1)]],                                    \
  constant uint& count [[buffer(2)]],                                   \
  uint index [[thread_position_in_grid]]                                \
) {                                                                     \
  if (index < count) {                                                   \
    output[index] = -input[index];                                       \
  }                                                                     \
}

YASPS_NEGATE(float, float)
YASPS_NEGATE(int, int)

#define YASPS_CONVERT(FROM, FROM_NAME, TO, TO_NAME)                      \
kernel void yasps_convert_##FROM_NAME##_to_##TO_NAME(                   \
  device const FROM* input [[buffer(0)]],                               \
  device TO* output [[buffer(1)]],                                      \
  constant uint& count [[buffer(2)]],                                   \
  uint index [[thread_position_in_grid]]                                \
) {                                                                     \
  if (index < count) {                                                   \
    output[index] = TO(input[index]);                                    \
  }                                                                     \
}

YASPS_CONVERT(float, float, int, int)
YASPS_CONVERT(float, float, uint, uint)
YASPS_CONVERT(int, int, float, float)
YASPS_CONVERT(int, int, uint, uint)
YASPS_CONVERT(uint, uint, float, float)
YASPS_CONVERT(uint, uint, int, int)

kernel void yasps_abs_float(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {
  if (index < count) {
    output[index] = fabs(input[index]);
  }
}

kernel void yasps_abs_int(
  device const int* input [[buffer(0)]],
  device int* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {
  if (index < count) {
    output[index] = abs(input[index]);
  }
}

kernel void yasps_reduce_sum_float(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  threadgroup float values[256];
  values[local_index] = index < count ? input[index] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (local_index < stride) {
      values[local_index] += values[local_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    output[group_index] = values[0];
  }
}

kernel void yasps_reduce_max_float(
  device const float* input [[buffer(0)]],
  device float* output [[buffer(1)]],
  constant uint& count [[buffer(2)]],
  uint index [[thread_position_in_grid]],
  uint local_index [[thread_index_in_threadgroup]],
  uint group_index [[threadgroup_position_in_grid]]
) {
  threadgroup float values[256];
  values[local_index] =
    index < count ? input[index] : -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (local_index < stride) {
      values[local_index] = max(
        values[local_index],
        values[local_index + stride]
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (local_index == 0) {
    output[group_index] = values[0];
  }
}

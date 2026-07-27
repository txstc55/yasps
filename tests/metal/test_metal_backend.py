from pathlib import Path

import numpy as np

from yasps.backend import gpuarray
from yasps.backend.metal_codegen import translate_device_kernel


def test_shared_buffer_views_and_modular_kernel_link(tmp_path):
  fixture = Path(__file__).with_name("fixtures")
  library = gpuarray.compile_metal(
    [fixture / "twice.metal", fixture / "add_twice.metal"],
    tmp_path / "test.metallib",
  )
  kernel = gpuarray.MetalKernel(library, "yasps_test_add_twice")

  input_array = gpuarray.to_gpu(np.arange(8, dtype=np.float32))
  output_array = gpuarray.zeros_like(input_array)
  kernel.dispatch(
    [input_array, output_array, np.uint32(input_array.size)],
    input_array.size,
  )

  np.testing.assert_array_equal(
    output_array.get(), np.arange(8, dtype=np.float32) * 2.0 + 1.0
  )
  output_array[2:6].fill(9)
  np.testing.assert_array_equal(
    output_array.get(),
    np.array([1, 3, 9, 9, 9, 9, 13, 15], dtype=np.float32),
  )


def test_fixed_size_shader_matrix_operations(tmp_path):
  fixture = Path(__file__).with_name("fixtures")
  matrix_include = (
    Path(__file__).parents[2]
    / "yasps"
    / "yasps"
    / "kernel"
    / "Compute"
  )
  library = gpuarray.compile_metal(
    [fixture / "matrix_ops.metal"],
    tmp_path / "matrix_ops.metallib",
    include_dirs=[matrix_include],
  )
  kernel = gpuarray.MetalKernel(library, "yasps_test_matrix_ops")

  left_host = np.array([[4, 1], [2, 3]], dtype=np.float32)
  right_host = np.array([[1, 2], [3, 4]], dtype=np.float32)
  left = gpuarray.to_gpu(left_host.ravel())
  right = gpuarray.to_gpu(right_host.ravel())
  output = gpuarray.zeros(4, np.float32)
  kernel.dispatch([left, right, output], 1)

  np.testing.assert_allclose(
    output.get().reshape(2, 2),
    left_host @ right_host + np.linalg.inv(left_host),
    rtol=2e-6,
    atol=2e-6,
  )


def test_generated_eigen_source_translates_and_links(tmp_path):
  cuda_header = """
__device__ void generated_device_function(
 const double* input,
 const unsigned int* indices,
 const unsigned int instance,
 double* result
)""".strip()
  cuda_source = f"""
{cuda_header}{{
 using RowMat = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
 Eigen::Map<RowMat> out(result);
 Eigen::Matrix<double, 2, 2, Eigen::RowMajor> local;
 local << input[indices[instance] * 4],
     input[indices[instance] * 4 + 1],
     input[indices[instance] * 4 + 2],
     input[indices[instance] * 4 + 3];
 out.noalias() = local.inverse().transpose();
}}
"""
  metal_source, metal_header = translate_device_kernel(
    cuda_source, cuda_header, 2, 2
  )
  fixture = Path(__file__).with_name("fixtures")
  matrix_include = (
    Path(__file__).parents[2]
    / "yasps"
    / "yasps"
    / "kernel"
    / "Compute"
  )
  helper = tmp_path / "generated_helper.metal"
  helper.write_text(
    '#include "metalMatrix.metal"\n' + metal_source,
    encoding="utf-8",
  )
  wrapper = tmp_path / "generated_wrapper.metal"
  wrapper.write_text(
    f"""
#include "metalMatrix.metal"
extern {metal_header};
kernel void generated_global(
  device const float* input [[buffer(0)]],
  device const uint* indices [[buffer(1)]],
  device float* output [[buffer(2)]],
  constant uint& count [[buffer(3)]],
  uint instance [[thread_position_in_grid]]) {{
 if (instance >= count) return;
 float local[4];
 generated_device_function(input, indices, instance, local);
 for (uint index = 0; index < 4; ++index) {{
  output[instance * 4 + index] = local[index];
 }}
}}
""",
    encoding="utf-8",
  )
  library = gpuarray.compile_metal(
    [helper, wrapper],
    tmp_path / "generated.metallib",
    include_dirs=[matrix_include],
  )
  kernel = gpuarray.MetalKernel(library, "generated_global")

  host = np.array(
    [[[4, 1], [2, 3]], [[2, 0], [1, 5]]], dtype=np.float32
  )
  input_array = gpuarray.to_gpu(host.ravel())
  indices = gpuarray.to_gpu(np.array([1, 0], dtype=np.uint32))
  output = gpuarray.zeros(8, np.float32)
  kernel.dispatch(
    [input_array, indices, output, np.uint32(2)],
    2,
  )

  expected = np.stack(
    [np.linalg.inv(host[1]).T, np.linalg.inv(host[0]).T]
  )
  np.testing.assert_allclose(
    output.get().reshape(2, 2, 2),
    expected,
    rtol=2e-6,
    atol=2e-6,
  )

from pathlib import Path

import numpy as np

from yasps.backend import gpuarray
from yasps.backend.metal_codegen import translate_device_kernel
from yasps.backend.metal_hessian_codegen import (
  translate_hessian_kernel,
)


def test_array_arithmetic_fill_copy_and_reductions_stay_on_metal(
  monkeypatch,
):
  left_host = np.linspace(-3.0, 5.0, 1025, dtype=np.float32)
  right_host = np.linspace(2.0, -1.0, 1025, dtype=np.float32)
  left = gpuarray.to_gpu(left_host)
  right = gpuarray.to_gpu(right_host)
  copied = gpuarray.empty_like(left)

  original_numpy_view = gpuarray.GPUArray._numpy_view

  def reject_host_array_access(_self):
    raise AssertionError("array operation touched host storage")

  monkeypatch.setattr(
    gpuarray.GPUArray,
    "_numpy_view",
    reject_host_array_access,
  )
  combined = left + right * 2.0
  absolute = abs(combined)
  total = gpuarray.sum(combined)
  maximum = gpuarray.max(absolute)
  copied.set(combined)
  right.fill(7.0)
  monkeypatch.setattr(
    gpuarray.GPUArray,
    "_numpy_view",
    original_numpy_view,
  )

  expected = left_host + right_host * 2.0
  np.testing.assert_allclose(combined.get(), expected, rtol=1.0e-6)
  np.testing.assert_allclose(copied.get(), expected, rtol=1.0e-6)
  np.testing.assert_allclose(
    total.get()[0],
    expected.sum(dtype=np.float32),
    rtol=2.0e-6,
  )
  np.testing.assert_allclose(
    maximum.get()[0],
    np.abs(expected).max(),
    rtol=1.0e-6,
  )
  np.testing.assert_array_equal(
    right.get(),
    np.full(right.size, 7.0, dtype=np.float32),
  )


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


def test_batched_dispatch_preserves_cross_kernel_dependencies(tmp_path):
  fixture = Path(__file__).with_name("fixtures")
  library = gpuarray.compile_metal(
    [fixture / "twice.metal", fixture / "add_twice.metal"],
    tmp_path / "batched.metallib",
  )
  kernel = gpuarray.MetalKernel(library, "yasps_test_add_twice")
  input_array = gpuarray.to_gpu(np.arange(1025, dtype=np.float32))
  intermediate = gpuarray.empty_like(input_array)
  output = gpuarray.empty_like(input_array)

  arguments = np.uint32(input_array.size)
  gpuarray.dispatch_batch(
    [
      (
        kernel,
        [input_array, intermediate, arguments],
        input_array.size,
        32,
      ),
      (
        kernel,
        [intermediate, output, arguments],
        input_array.size,
        32,
      ),
    ],
    "test_dependency",
  )

  np.testing.assert_array_equal(
    output.get(),
    np.arange(1025, dtype=np.float32) * 4.0 + 3.0,
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


def test_small_matrix_guards_contain_nonfinite_curvature(tmp_path):
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
    tmp_path / "matrix_guards.metallib",
    include_dirs=[matrix_include],
  )
  kernel = gpuarray.MetalKernel(library, "yasps_test_matrix_guards")

  invalid = np.eye(3, dtype=np.float32)
  invalid[0, 1] = np.nan
  input_array = gpuarray.to_gpu(invalid.ravel())
  projection = gpuarray.empty(9, np.float32)
  inverse = gpuarray.empty(9, np.float32)
  kernel.dispatch([input_array, projection, inverse], 1)

  np.testing.assert_array_equal(
    projection.get(),
    np.zeros(9, dtype=np.float32),
  )
  np.testing.assert_array_equal(
    inverse.get().reshape(3, 3),
    np.eye(3, dtype=np.float32),
  )


def test_small_matrix_guards_preserve_finite_path(tmp_path):
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
    tmp_path / "matrix_finite.metallib",
    include_dirs=[matrix_include],
  )
  kernel = gpuarray.MetalKernel(library, "yasps_test_matrix_guards")

  finite = np.array(
    [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]],
    dtype=np.float32,
  )
  input_array = gpuarray.to_gpu(finite.ravel())
  projection = gpuarray.empty(9, np.float32)
  inverse = gpuarray.empty(9, np.float32)
  kernel.dispatch([input_array, projection, inverse], 1)

  np.testing.assert_allclose(
    projection.get().reshape(3, 3),
    finite,
    rtol=2.0e-5,
    atol=2.0e-5,
  )
  np.testing.assert_allclose(
    inverse.get().reshape(3, 3),
    np.linalg.inv(finite),
    rtol=2.0e-5,
    atol=2.0e-5,
  )


def test_stable_neo_hookean_size_projection_matches_eigh(tmp_path):
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
    tmp_path / "matrix_projection_12.metallib",
    include_dirs=[matrix_include],
  )
  kernel = gpuarray.MetalKernel(
    library,
    "yasps_test_projection_12",
  )

  rng = np.random.default_rng(1234)
  source = rng.standard_normal((12, 12)).astype(np.float32)
  source = (source + source.T) * np.float32(0.5)
  eigenvalues, eigenvectors = np.linalg.eigh(source)
  expected = (
    eigenvectors
    * np.maximum(eigenvalues, np.float32(0.0))
  ) @ eigenvectors.T
  input_array = gpuarray.to_gpu(source.ravel())
  output = gpuarray.empty(144, np.float32)
  kernel.dispatch([input_array, output], 1)

  np.testing.assert_allclose(
    output.get().reshape(12, 12),
    expected,
    rtol=3.0e-4,
    atol=3.0e-4,
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


def test_generated_resize_pointer_constructor_translates():
  cuda_header = """
__device__ void generated_resize(
 const double* input,
 const unsigned int instance,
 double* result
)""".strip()
  cuda_source = f"""
{cuda_header}{{
 using RowMat = Eigen::Matrix<double, 3, 1, Eigen::RowMajor>;
 Eigen::Map<RowMat> out(result);
 Eigen::Matrix<double, 1, 3, Eigen::RowMajor> temporary;
 temporary << input[instance * 3],
     input[instance * 3 + 1],
     input[instance * 3 + 2];
 Eigen::Matrix<double, 3, 1, Eigen::RowMajor>
     resized((temporary).data());
 out.noalias() = resized;
}}
"""

  metal_source, _ = translate_device_kernel(
    cuda_source,
    cuda_header,
    3,
    1,
  )

  assert "resized((temporary).data())" not in metal_source
  assert (
    "resized = yasps_matrix_from_pointer<3, 1>"
    "((temporary).data());"
  ) in metal_source


def test_complete_generated_output_writes_directly_to_result():
  cuda_header = """
__device__ void generated_complete_output(
 const double* input,
 double* result
)""".strip()
  cuda_source = f"""
{cuda_header}{{
 using RowMat = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
 Eigen::Map<RowMat> out(result);
 out << input[0], input[1], input[2], input[3];
}}
"""

  metal_source, _ = translate_device_kernel(
    cuda_source,
    cuda_header,
    2,
    2,
  )

  assert "RowMat out" not in metal_source
  assert "result[0] = input[0];" in metal_source
  assert "result[3] = input[3];" in metal_source
  assert "yasps_output_index" not in metal_source


def test_nested_generated_output_keeps_local_matrix():
  cuda_header = """
__device__ void generated_nested_output(
 double* result
)""".strip()
  cuda_source = f"""
{cuda_header}{{
 using RowMat = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;
 Eigen::Map<RowMat> out(result);
 nested_device_function(out.data());
 out << 1.0, 2.0, 3.0, 4.0;
}}
"""

  metal_source, _ = translate_device_kernel(
    cuda_source,
    cuda_header,
    2,
    2,
  )

  assert "RowMat out = {};" in metal_source
  assert "nested_device_function(out.data());" in metal_source
  assert "yasps_output_index" in metal_source


def test_generated_const_matrix_map_uses_pointer_backed_view():
  cuda_header = """
__device__ void generated_map(
 double* input,
 double* result
)""".strip()
  cuda_source = f"""
{cuda_header}{{
 using RowMat = Eigen::Matrix<double, 1, 1, Eigen::RowMajor>;
 Eigen::Map<RowMat> out(result);
 Eigen::Map<const Eigen::Matrix<double, 12, 12, Eigen::RowMajor>>
     mapped(input);
 out(0, 0) = mapped(3, 4);
}}
"""

  metal_source, _ = translate_device_kernel(
    cuda_source,
    cuda_header,
    1,
    1,
  )

  assert (
    "YaspsMatrixView<12, 12> mapped = "
    "yasps_matrix_view<12, 12>(input);"
  ) in metal_source
  assert "yasps_matrix_from_pointer<12, 12>(input)" not in metal_source


def test_implicit_hessian_translation_collapses_size_groups():
  cuda_source = """
extern "C" {
__global__ void compute_hessian(
  const unsigned int* groupedIndicesInner,
  const unsigned int* groupedIndicesOuter,
  const unsigned int nth_gradient_size,
  const unsigned int projection_method
) {
  const unsigned int start =
    groupedIndicesOuter[nth_gradient_size];
  const unsigned int end =
    groupedIndicesOuter[nth_gradient_size + 1];
  unsigned int index =
    blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= end - start) {
    return;
  }
  index = start + index;
  const unsigned int instance = groupedIndicesInner[index];
}
}
"""

  metal_source = translate_hessian_kernel(
    cuda_source,
    "compute_hessian",
    header_include=None,
    collapse_groups=True,
    max_threads_per_threadgroup=32,
  )

  assert "[[max_total_threads_per_threadgroup(32)]]" in metal_source
  assert "uint total_instance_count [[id(3)]];" in metal_source
  assert "const uint start = 0;" in metal_source
  assert "const uint end = total_instance_count;" in metal_source
  assert "groupedIndicesOuter[nth_gradient_size + 1]" not in metal_source


def test_implicit_hessian_translation_skips_redundant_output_zeroing():
  cuda_source = """
extern "C" {
__global__ void compute_hessian(
  double* output
) {
  Eigen::Matrix<double, 49, 48, Eigen::RowMajor> hg_mat =
    Eigen::Matrix<double, 49, 48, Eigen::RowMajor>::Zero();
  generated_device_function(hg_mat.data());
  output[0] = hg_mat(0, 0);
}
}
"""

  metal_source = translate_hessian_kernel(
    cuda_source,
    "compute_hessian",
    header_include=None,
  )

  assert "YaspsMatrix<49, 48> hg_mat;" in metal_source
  assert "YaspsMatrix<49, 48> hg_mat = {};" not in metal_source


def test_generated_scalar_matrix_product_unwraps():
  cuda_header = """
__device__ void generated_scalar_product(
 const double* input,
 const unsigned int instance,
 double* result
)""".strip()
  cuda_source = f"""
{cuda_header}{{
 Eigen::Matrix<double, 1, 3, Eigen::RowMajor> left;
 Eigen::Matrix<double, 3, 1, Eigen::RowMajor> right;
 left << input[0], input[1], input[2];
 right << input[3], input[4], input[5];
 result[0] = left * right;
}}
"""

  metal_source, _ = translate_device_kernel(
    cuda_source,
    cuda_header,
    1,
    1,
  )

  assert (
    "result[0] = yasps_scalar_value(left * right);"
    in metal_source
  )

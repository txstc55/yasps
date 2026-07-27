from pathlib import Path

import numpy as np

from yasps.backend import gpuarray


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

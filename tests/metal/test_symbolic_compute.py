import numpy as np

from yasps import scene


def test_fused_scalar_expression_runs_on_metal(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  simulation = scene("metal_scalar_expression")
  mesh = simulation.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=4)
  left = points.addAttribute("left")
  right = points.addAttribute("right")
  left.updateValue(np.array([1, 2, 3, 4], dtype=np.float64))
  right.updateValue(np.array([10, 20, 30, 40], dtype=np.float64))
  result = points.addAttribute(
    "result",
    computed_attribute=left + right * 2.0,
  )

  output = result.compute().value

  assert output.dtype == np.dtype(np.float32)
  np.testing.assert_allclose(output.get(), [21, 42, 63, 84])


def test_matrix_source_is_emitted_for_cuda_and_metal(
  tmp_path,
  monkeypatch,
):
  monkeypatch.chdir(tmp_path)
  simulation = scene("metal_matrix_expression")
  mesh = simulation.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=2)
  matrix = points.addAttribute("matrix", rows=2, cols=2)
  vector = points.addAttribute("vector", rows=2, cols=1)
  matrix_host = np.array(
    [[[2, 1], [0, 3]], [[4, 2], [1, 5]]],
    dtype=np.float64,
  )
  vector_host = np.array(
    [[[3], [4]], [[2], [6]]],
    dtype=np.float64,
  )
  matrix.updateValue(matrix_host)
  vector.updateValue(vector_host)
  result = points.addAttribute(
    "result",
    computed_attribute=matrix * vector + vector,
  )

  output = result.compute().value.get().reshape(2, 2, 1)

  np.testing.assert_allclose(
    output,
    matrix_host.astype(np.float32) @ vector_host.astype(np.float32)
    + vector_host,
  )
  cuda_source = result.deviceKernel.kernelString
  metal_source = result.deviceKernel.metalKernelString
  assert "Eigen::Matrix<double" in cuda_source
  assert "const double*" in cuda_source
  assert "YaspsMatrix<" in metal_source
  assert "device const float*" in metal_source

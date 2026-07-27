import numpy as np

from yasps import scene


def test_fused_quadratic_hessian_and_cg_solution():
  simulation = scene("metal_quadratic_hessian_solver_test")
  mesh = simulation.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=4)
  values = points.addAttribute("value", rows=1, cols=1)
  expected = np.array([1.0, -2.0, 3.5, -4.25], dtype=np.float32)
  values.updateValue(expected)
  quadratic = mesh.addAttribute(
    "quadratic",
    computed_attribute=0.5 * values * values
  )

  simulation.addEnergy(quadratic, projection_method=-1)
  simulation.addMinimizeTarget([values])
  solution = simulation.minimizeEnergy(maxIterations=20)[0]

  assert solution.dtype == np.float32
  np.testing.assert_allclose(
    simulation.gradient.get(),
    expected,
    rtol=2.0e-5,
    atol=2.0e-5
  )
  np.testing.assert_allclose(
    solution.get(),
    expected,
    rtol=2.0e-5,
    atol=2.0e-5
  )

import numpy as np

from yasps import differentiator, scene, vector


def test_materialized_hessian_solves_arbitrary_right_hand_side():
  model = scene("direct_hessian_solve_regression")
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=2)
  x = points.addAttribute("x", rows=1, cols=1)
  stiffness = points.addConstant("stiffness", rows=1, cols=1)
  x.updateValue(np.array([3.0, -2.0]))
  stiffness.updateValue(np.array([2.0, 4.0]))
  energy = points.addAttribute(
    "energy",
    computed_attribute=0.5 * stiffness * x * x
  )

  system = differentiator().diff2(
    [energy], [x], [x], projection_method=-1
  )
  system.compute()

  rhs = vector(2)
  rhs.updateValue(np.array([4.0, 8.0]))
  solution = system.solve(
    rhs, tolerance=1.0e-12, maxIterations=100
  )
  np.testing.assert_allclose(solution.value.get(), np.array([2.0, 2.0]))
  assert system.last_solve_error_code >= 0

  # With no explicit rhs, solve the assembled Newton system H dx = gradient.
  newton_step = system.solve(tolerance=1.0e-12, maxIterations=100)
  np.testing.assert_allclose(newton_step.value.get(), np.array([3.0, -2.0]))


def test_two_hessian_objects_reuse_identical_merged_expression():
  model = scene("direct_hessian_shared_expression_regression")
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=2)
  x = points.addAttribute("x", rows=1, cols=1)
  x.updateValue(np.array([1.0, -2.0]))
  energy = points.addAttribute(
    "energy", computed_attribute=0.5 * x * x
  )

  first = differentiator().diff2(
    [energy], [x], [x], projection_method=-1
  )
  second = differentiator().diff2(
    [energy], [x], [x], projection_method=-1
  )

  # Forward and adjoint systems commonly materialize the same term in
  # separate matrix objects. The second compute must reuse, rather than
  # collide with, the first merged code-generation attribute.
  first.compute()
  second.compute()
  rhs = np.array([3.0, -4.0])
  np.testing.assert_allclose(
    first.solve(rhs).value.get(), rhs, rtol=1.0e-10, atol=1.0e-10
  )
  np.testing.assert_allclose(
    second.solve(rhs).value.get(), rhs, rtol=1.0e-10, atol=1.0e-10
  )


if __name__ == "__main__":
  test_materialized_hessian_solves_arbitrary_right_hand_side()
  test_two_hessian_objects_reuse_identical_merged_expression()
  print("direct Hessian solve regression tests passed")

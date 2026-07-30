import numpy as np

from yasps import differentiator, scene, vector


def _finite_difference_mixed(energy, x, theta, x_values, theta_values):
  epsilon = 2.0e-4
  rows = x_values.size
  cols = theta_values.size
  result = np.zeros((rows, cols), dtype=np.float64)

  def total_energy(current_x, current_theta):
    x.updateValue(current_x)
    theta.updateValue(current_theta)
    return float(np.sum(energy.compute().value.get()))

  flat_x = x_values.reshape(-1).copy()
  flat_theta = theta_values.reshape(-1).copy()
  for row in range(rows):
    for column in range(cols):
      values = []
      for row_sign, column_sign in [
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0)
      ]:
        displaced_x = flat_x.copy()
        displaced_theta = flat_theta.copy()
        displaced_x[row] += row_sign * epsilon
        displaced_theta[column] += column_sign * epsilon
        values.append(total_energy(displaced_x, displaced_theta))
      result[row, column] = (
        values[0] - values[1] - values[2] + values[3]
      ) / (4.0 * epsilon * epsilon)

  x.updateValue(flat_x)
  theta.updateValue(flat_theta)
  return result


def _make_point_energy(name="mixed_regression"):
  model = scene(name)
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=3)
  x = points.addAttribute("x", rows=2, cols=1)
  theta = points.addConstant("theta", rows=1, cols=1)
  unrelated_constant = points.addConstant(
    "unrelated_constant", rows=1, cols=1
  )
  unrelated_data = points.addAttribute("unrelated_data", rows=1, cols=1)

  x_values = np.array([
    [1.0, 2.0],
    [-3.0, 0.5],
    [4.0, -2.0]
  ])
  theta_values = np.array([2.0, -1.0, 0.25])
  x.updateValue(x_values)
  theta.updateValue(theta_values)
  unrelated_constant.updateValue(np.array([7.0, 8.0, 9.0]))
  unrelated_data.updateValue(np.array([-2.0, 3.0, 5.0]))

  expression = (
    theta * x.dot(x)
    + unrelated_constant * x[0] * x[0]
    + unrelated_data * x[1]
  )
  energy = points.addAttribute("energy", computed_attribute=expression)
  return energy, x, theta, x_values, theta_values


def test_mixed_second_order_jacobian_constant_target_and_spmv():
  energy, x, theta, x_values, theta_values = _make_point_energy()
  mixed = differentiator().diff2([energy], [x], [theta])
  mixed.compute()

  expected = np.zeros((6, 3), dtype=np.float64)
  for instance in range(3):
    expected[2 * instance:2 * instance + 2, instance] = (
      2.0 * x_values[instance]
    )

  np.testing.assert_allclose(mixed.toDense(), expected)
  finite_difference = _finite_difference_mixed(
    energy, x, theta, x_values, theta_values
  )
  np.testing.assert_allclose(
    mixed.toDense(), finite_difference, rtol=2.0e-6, atol=2.0e-6
  )

  assert mixed.rows == 6
  assert mixed.cols == 3
  assert mixed.block_dimensions == [2, 1]
  assert mixed.block_counts == [3]
  assert len(mixed.row_outer_jacobians) == 1
  assert len(mixed.column_outer_jacobians) == 1
  assert len(mixed.inner_hessians) == 1

  right = vector(3)
  right.updateValue(np.array([1.5, -2.0, 0.25]))
  np.testing.assert_allclose(
    mixed.matVecProduct(right).value.get(),
    expected @ right.value.get()
  )
  left = vector(6)
  left.updateValue(np.arange(1.0, 7.0))
  np.testing.assert_allclose(
    mixed.transposeMatVecProduct(left).value.get(),
    expected.T @ left.value.get()
  )


def test_mixed_second_order_jacobian_compressed_coordinates_and_orientation():
  energy, x, theta, x_values, _ = _make_point_energy(
    "mixed_compressed_regression"
  )
  # Reversing target order must materialize the transpose-shaped derivative,
  # rather than relying on Hessian symmetry in storage or SpMV.
  mixed = differentiator().diff2(
    [energy, energy],
    [theta],
    [x],
    compress_coordinates=True
  )
  mixed.compute()

  expected = np.zeros((3, 6), dtype=np.float64)
  for instance in range(3):
    expected[instance, 2 * instance:2 * instance + 2] = (
      4.0 * x_values[instance]
    )
  np.testing.assert_allclose(mixed.toDense(), expected)
  assert mixed.block_dimensions == [1, 2]
  assert mixed.block_counts == [3]


def test_dynamic_mixed_second_order_coordinates_regenerate():
  model = scene("mixed_dynamic_regression")
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=4)
  x = points.addAttribute("x", rows=2, cols=1)
  theta = points.addConstant("theta", rows=1, cols=1)
  x_values = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [-2.0, 1.0],
    [0.5, -3.0]
  ])
  x.updateValue(x_values)
  theta.updateValue(np.array([1.0, 2.0, 3.0, 4.0]))

  pairs = mesh.addPrimitive("pairs", numInstances=2, isDynamic=True)
  pair_to_point = pairs.addConnectivity(
    "pair_to_point",
    points,
    np.array([[0], [1]], dtype=np.uint32),
    1
  )
  pair_x = pairs.addAttribute(
    "x", through=pair_to_point, source=x
  )
  pair_theta = pairs.addAttribute(
    "theta", through=pair_to_point, source=theta
  )
  energy = pairs.addAttribute(
    "energy",
    computed_attribute=pair_theta * pair_x.dot(pair_x)
  )

  mixed = differentiator().diff2(
    [energy], [x], [theta], dynamic_instances=True
  )
  mixed.compute()
  expected_first = np.zeros((8, 4), dtype=np.float64)
  for instance in [0, 1]:
    expected_first[2 * instance:2 * instance + 2, instance] = (
      2.0 * x_values[instance]
    )
  np.testing.assert_allclose(mixed.toDense(), expected_first)

  pair_to_point.updateConnectivity(
    np.array([[3], [0], [2]], dtype=np.uint32)
  )
  pairs.updateNumInstances(3)
  mixed.compute()
  expected_second = np.zeros((8, 4), dtype=np.float64)
  for instance in [3, 0, 2]:
    expected_second[2 * instance:2 * instance + 2, instance] += (
      2.0 * x_values[instance]
    )
  np.testing.assert_allclose(mixed.toDense(), expected_second)

  pairs.updateNumInstances(0)
  mixed.compute()
  np.testing.assert_allclose(mixed.toDense(), np.zeros((8, 4)))


def test_mixed_derivative_is_not_reused_from_projected_hessian_cache():
  model = scene("mixed_projection_cache_regression")
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=1)
  x = points.addAttribute("x", rows=1, cols=1)
  y = points.addAttribute("y", rows=1, cols=1)
  x.updateValue(np.array([2.0]))
  y.updateValue(np.array([-3.0]))
  energy = points.addAttribute(
    "energy", computed_attribute=x * y
  )

  # The raw Hessian of x*y is indefinite. Building its PSD-projected Hessian
  # first must not contaminate the unprojected mixed derivative cache.
  differentiator().diff2(
    [energy], [x, y], [x, y], projection_method=2
  )
  mixed = differentiator().diff2([energy], [x], [y])
  mixed.compute()

  np.testing.assert_allclose(mixed.toDense(), np.array([[1.0]]))
  mixed_repeated = differentiator().diff2([energy], [x], [y])
  mixed_repeated.compute()
  np.testing.assert_allclose(mixed_repeated.toDense(), np.array([[1.0]]))


if __name__ == "__main__":
  test_mixed_second_order_jacobian_constant_target_and_spmv()
  test_mixed_second_order_jacobian_compressed_coordinates_and_orientation()
  test_dynamic_mixed_second_order_coordinates_regenerate()
  test_mixed_derivative_is_not_reused_from_projected_hessian_cache()
  print("second-order Jacobian regression tests passed")

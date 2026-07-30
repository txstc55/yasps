import numpy as np

from yasps import attribute, differentiator, gradient, scene, vector


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


def test_matrix_product_accepts_gradient_and_materializes_lazily():
  energy, x, theta, x_values, _ = _make_point_energy(
    "mixed_lazy_matvec_regression"
  )
  mixed = differentiator().diff2([energy], [x], [theta])
  right = gradient([theta])
  right.updateValue(np.array([1.5, -2.0, 0.25]))

  expected = np.zeros((6, 3), dtype=np.float64)
  for instance in range(3):
    expected[2 * instance:2 * instance + 2, instance] = (
      2.0 * x_values[instance]
    )

  # No explicit mixed.compute() is required, and a gradient is a valid
  # vector-valued right-hand side for the adjoint products.
  result = mixed @ right
  np.testing.assert_allclose(
    result.value.get(), expected @ right.value.get()
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


def test_mixed_chain_rule_retains_two_outer_jacobians_and_recursive_term():
  model = scene("mixed_recursive_chain_regression")
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=1)
  x = points.addAttribute("x", rows=1, cols=1)
  theta = points.addConstant("theta", rows=1, cols=1)
  x_value = 0.7
  theta_value = -1.2
  x.updateValue(np.array([x_value]))
  theta.updateValue(np.array([theta_value]))

  inner = points.addAttribute(
    "inner", computed_attribute=x * theta
  )
  terms = mesh.addPrimitive("terms", numInstances=1)
  term_to_point = terms.addConnectivity(
    "term_to_point",
    points,
    np.array([[0]], dtype=np.uint32),
    1
  )
  joined_inner = terms.addAttribute(
    "inner", through=term_to_point, source=inner
  )
  energy = terms.addAttribute(
    "energy", computed_attribute=joined_inner.sin()
  )
  mixed = differentiator().diff2([energy], [x], [theta])
  mixed.compute()

  product = x_value * theta_value
  expected_outer_term = -product * np.sin(product)
  expected_recursive_term = np.cos(product)
  expected = expected_outer_term + expected_recursive_term
  np.testing.assert_allclose(mixed.toDense(), np.array([[expected]]))

  row_outer = mixed.row_outer_jacobians[0]
  column_outer = mixed.column_outer_jacobians[0]
  inner_hessian = mixed.inner_hessians[0]
  row_outer.compute()
  column_outer.compute()
  inner_hessian.compute()

  # Both outer Jacobians use the same inner-variable space and mask the
  # differentiation target on the opposite side.
  row_outer_value = row_outer.value.get().reshape(
    row_outer.rows, row_outer.cols
  )
  column_outer_value = column_outer.value.get().reshape(
    column_outer.rows, column_outer.cols
  )
  assert row_outer_value.shape == (1, 2)
  assert column_outer_value.shape == (1, 2)
  row_nonzero = np.flatnonzero(row_outer_value[0])
  column_nonzero = np.flatnonzero(column_outer_value[0])
  assert row_nonzero.size == 1
  assert column_nonzero.size == 1
  row_local = int(row_nonzero[0])
  column_local = int(column_nonzero[0])
  np.testing.assert_allclose(
    row_outer_value[0, row_local], theta_value
  )
  np.testing.assert_allclose(
    column_outer_value[0, column_local], x_value
  )
  np.testing.assert_allclose(
    inner_hessian.value.get().reshape(
      inner_hessian.rows, inner_hessian.cols
    ),
    np.array([[-np.sin(product)]])
  )
  inner_value = inner_hessian.value.get().reshape(
    inner_hessian.rows, inner_hessian.cols
  )
  numeric_outer_term = (
    row_outer_value.T @ inner_value @ column_outer_value
  )
  np.testing.assert_allclose(
    mixed.toDense()[0, 0]
    - numeric_outer_term[row_local, column_local],
    expected_recursive_term
  )


def test_uncompressed_occurrences_are_grouped_by_rectangular_block_size():
  model = scene("mixed_block_layout_regression")
  mesh = model.addMesh("mesh")
  points = mesh.addPrimitive("points", numInstances=2)
  x = points.addAttribute("x", rows=2, cols=1)
  a = points.addAttribute("a", rows=1, cols=1)
  theta = points.addConstant("theta", rows=1, cols=1)
  beta = points.addConstant("beta", rows=3, cols=1)
  x.updateValue(np.array([[1.0, 2.0], [3.0, 4.0]]))
  a.updateValue(np.array([5.0, 6.0]))
  theta.updateValue(np.array([7.0, 8.0]))
  beta.updateValue(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

  ones2 = attribute.to_array([1.0, 1.0], rows=2, cols=1)
  ones3 = attribute.to_array([1.0, 1.0, 1.0], rows=3, cols=1)
  row_linear = x.dot(ones2) + a
  column_linear = theta + beta.dot(ones3)
  energy = points.addAttribute(
    "energy", computed_attribute=row_linear * column_linear
  )

  # Duplicating the source produces two occurrences at every coordinate.
  # With the default uncompressed layout they remain separate stored blocks.
  mixed = differentiator().diff2(
    [energy, energy], [x, a], [theta, beta]
  )
  mixed.compute()

  assert mixed.compress_coordinates is False
  assert mixed.block_dimensions == [1, 1, 1, 3, 2, 1, 2, 3]
  assert mixed.block_counts == [4, 4, 4, 4]

  expected = np.zeros((6, 8), dtype=np.float64)
  for instance in range(2):
    rows = [2 * instance, 2 * instance + 1, 4 + instance]
    columns = [
      instance,
      2 + 3 * instance,
      3 + 3 * instance,
      4 + 3 * instance
    ]
    expected[np.ix_(rows, columns)] = 2.0
  np.testing.assert_allclose(mixed.toDense(), expected)


if __name__ == "__main__":
  test_mixed_second_order_jacobian_constant_target_and_spmv()
  test_matrix_product_accepts_gradient_and_materializes_lazily()
  test_mixed_second_order_jacobian_compressed_coordinates_and_orientation()
  test_dynamic_mixed_second_order_coordinates_regenerate()
  test_mixed_derivative_is_not_reused_from_projected_hessian_cache()
  test_mixed_chain_rule_retains_two_outer_jacobians_and_recursive_term()
  test_uncompressed_occurrences_are_grouped_by_rectangular_block_size()
  print("second-order Jacobian regression tests passed")

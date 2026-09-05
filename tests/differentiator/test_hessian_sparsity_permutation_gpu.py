"""GPU assembly regressions with independent small dense CPU references.

Run from a fresh working directory so generated CUDA caches cannot hide a change:
  PYTHONPATH=/path/to/yasps/yasps python -m pytest /path/to/this/file -q

The matrices below are deliberately small. They test generated derivatives,
coordinates, symmetric scatter, diagonal blocks, and SpMV without allocating
the dense matrices used by the large simulation benchmarks.
"""

import numpy as np
import pycuda.gpuarray as gpuarray
import pytest

from yasps import attribute, differentiator, scene, solver, vector


def _linear(variable, coefficients, offset=None):
  entries = []
  for row in coefficients:
    value = 0.0 if offset is None else float(offset[len(entries)])
    for index, coefficient in enumerate(row):
      if coefficient != 0.0:
        value = value + float(coefficient) * variable[index]
    entries.append(value)
  return attribute.to_array(entries, rows=len(entries), cols=1)


def _quadratic(variable, hessian, linear, model):
  # Runtime constants avoid the separate pre-existing code-generator limitation
  # when a projected Hessian consists entirely of Python literals.
  coefficients = model.addConstant("quadratic_coefficients", rows=len(linear), cols=len(linear))
  coefficients.updateValue(hessian)
  result = 0.0
  for row in range(len(linear)):
    result = result + float(linear[row]) * variable[row]
    for column in range(len(linear)):
      if hessian[row, column] != 0.0:
        result = result + 0.5 * coefficients[row, column] * variable[row] * variable[column]
  return result


def _inner_matrix(size):
  rng = np.random.default_rng(209 + size)
  basis, _ = np.linalg.qr(rng.normal(size=(size, size)))
  eigenvalues = np.linspace(-1.7, 3.1, size)
  return (basis * eigenvalues) @ basis.T


def _project(matrix, method):
  if method < 1:
    return matrix
  values, vectors = np.linalg.eigh(matrix)
  values = np.abs(values) if method == 1 else np.maximum(values, 0.0)
  return (vectors * values) @ vectors.T


def _dense_from_blocks(hessian):
  dense = np.zeros((hessian.rows, hessian.cols), dtype=np.float64)
  for suffix in ("", "_dynamic"):
    counts = getattr(hessian, "block_counts" + suffix)
    dimensions = getattr(hessian, "block_dimensions" + suffix)
    starts = getattr(hessian, "blocks_start_indices" + suffix)
    positions = getattr(hessian, "block_positions" + suffix).get()[:2 * sum(counts)].reshape(-1, 2)
    values = getattr(hessian, "blocks_flattened" + suffix).get()
    coordinate_offset = 0
    for group, count in enumerate(counts):
      rows, columns = dimensions[2 * group:2 * group + 2]
      for block_index in range(count):
        row, column = map(int, positions[coordinate_offset + block_index])
        offset = starts[group] + block_index * rows * columns
        block = values[offset:offset + rows * columns].reshape(rows, columns)
        dense[row:row + rows, column:column + columns] += block
        if row != column:
          dense[column:column + columns, row:row + rows] += block.T
      coordinate_offset += count
  return dense


def _check_values(hessian, expected_hessian, expected_gradient, target_shapes):
  hessian.compute()
  dense = _dense_from_blocks(hessian)
  np.testing.assert_allclose(dense, expected_hessian, rtol=3e-11, atol=3e-11)
  np.testing.assert_allclose(hessian.gradient.value.get(), expected_gradient, rtol=3e-11, atol=3e-11)

  # Block-Jacobi data has to include the transpose contributions when several
  # local occurrences, or several sparsity components, land on one vertex.
  expected_diagonal_blocks = []
  offset = 0
  for count, size in target_shapes:
    for index in range(count):
      start = offset + index * size
      expected_diagonal_blocks.extend(expected_hessian[start:start + size, start:start + size].ravel())
    offset += count * size
  np.testing.assert_allclose(hessian.diagonal_blocks.get(), expected_diagonal_blocks, rtol=3e-11, atol=3e-11)
  direction_values = np.linspace(-0.8, 0.9, hessian.cols)
  direction = vector(hessian.cols)
  direction.updateValue(direction_values)
  for transpose in (False, True):
    np.testing.assert_allclose(hessian.spmv(direction, transpose=transpose).value.get(), expected_hessian @ direction_values, rtol=3e-11, atol=3e-11)
  return dense


def _scatter_reference(hessian, gradient, indices, local_hessian, local_gradient):
  np.add.at(hessian, (indices[:, None], indices[None, :]), local_hessian)
  np.add.at(gradient, indices, local_gradient)


@pytest.mark.parametrize("pattern,projection", [("interlaced", -1), ("interlaced", 1), ("interlaced", 2), ("connected", 1), ("zero_axes", 1)])
def test_rectangular_sparsity_permutation_and_repeated_vertices(pattern, projection):
  model = scene(f"sparsity_gpu_{pattern}_{'raw' if projection < 0 else projection}")
  mesh = model.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=67)
  position = vertices.addAttribute("position", rows=2, cols=1)
  values = np.sin(np.arange(134) * 0.17).reshape(-1, 2)
  position.updateValue(values)

  if pattern != "connected":
    # Components are {rows 0,3; cols 0,3,6}, {rows 1,4; cols 1,4,7},
    # and {row 2; cols 2,5}. Both axes require a nontrivial permutation.
    jacobian = np.array([[0.3, 0, 0, -0.7, 0, 0, 1.1, 0], [0, 0.6, 0, 0, 0.8, 0, 0, -0.4], [0, 0, 1.2, 0, 0, -0.9, 0, 0], [-0.8, 0, 0, 0.4, 0, 0, 0.5, 0], [0, 1.1, 0, 0, -0.2, 0, 0, 0.7]])
    if pattern == "zero_axes":
      jacobian[4, :] = 0.0
      jacobian[:, 7] = 0.0
  else:
    jacobian = np.arange(32, dtype=np.float64).reshape(4, 8) / 41.0 + 0.13
  boundary_offset = np.zeros(len(jacobian))
  if pattern == "zero_axes":
    # A literal-zero boundary output is removed before differentiation by
    # JOIN compression. A nonzero constant retains the actual zero J row.
    boundary_offset[4] = 0.15
  inner_hessian = _inner_matrix(len(jacobian))
  linear = np.linspace(-0.3, 0.4, len(jacobian))
  connectivity_values = np.array([[3, 1, 5, 0], [1, 1, 2, 0], [0, 2, 0, 2], [5, 5, 5, 5]] + [[i, (i + 1) % 67, (i + 9) % 67, (i + 3) % 67] for i in range(67)], dtype=np.uint32)
  maps = mesh.addPrimitive("maps", numInstances=len(connectivity_values))
  to_vertices = maps.addConnectivity("to_vertices", vertices, connectivity_values, 4)
  local_position = maps.addAttribute("local_position", through=to_vertices, source=position)
  boundary = maps.addAttribute("boundary", computed_attribute=_linear(local_position, jacobian, boundary_offset))
  terms = mesh.addPrimitive("terms", numInstances=len(connectivity_values))
  to_maps = terms.addConnectivity("to_maps", maps, np.arange(len(connectivity_values), dtype=np.uint32), 1)
  joined_boundary = terms.addAttribute("boundary", through=to_maps, source=boundary)
  energy = terms.addAttribute("energy", computed_attribute=_quadratic(joined_boundary, inner_hessian, linear, model))
  result = differentiator().diff2([energy], [position], [position], projection_method=projection, separate_hessian_jacobian=True)
  assert result.project_entire_hessian == [False]
  assert result.global_jacobians[0].rows == jacobian.shape[0]
  assert result.global_jacobians[0].cols == jacobian.shape[1]

  expected_hessian = np.zeros((values.size, values.size))
  expected_gradient = np.zeros(values.size)
  local_hessian = jacobian.T @ _project(inner_hessian, projection) @ jacobian
  for connectivity in connectivity_values:
    indices = (2 * connectivity[:, None] + np.arange(2)).ravel()
    boundary_values = jacobian @ values[connectivity].ravel() + boundary_offset
    local_gradient = jacobian.T @ (inner_hessian @ boundary_values + linear)
    _scatter_reference(expected_hessian, expected_gradient, indices, local_hessian, local_gradient)
  _check_values(result, expected_hessian, expected_gradient, [(len(values), 2)])

  # This comparison also checks the legacy non-separated execution path using
  # exactly the same projection boundary, avoiding a change in mathematics.
  reference = differentiator().diff2([energy], [position], [position], projection_method=projection, separate_hessian_jacobian=False)
  _check_values(reference, expected_hessian, expected_gradient, [(len(values), 2)])

  if projection == 1:
    inertia = vertices.addAttribute("inertia", computed_attribute=0.25 * position.dot(position))
    regularizer = differentiator().diff2([inertia], [position], [position], projection_method=-1, separate_hessian_jacobian=True)
    combined = result + regularizer
    expected_hessian += 0.5 * np.eye(values.size)
    expected_gradient += 0.5 * values.ravel()
    _check_values(combined, expected_hessian, expected_gradient, [(len(values), 2)])
    rhs_values = np.cos(np.arange(values.size) * 0.23)
    rhs = vector(values.size)
    rhs.updateValue(rhs_values)
    # Both implementations compare r.T M^-1 r with tolerance times its
    # initial value, so the supplied tolerance is a squared residual ratio.
    for backend in ("jacobian", "mas"):
      linear_solver = solver(backend)
      status = linear_solver.computeSolution(combined, rhs, gpuarray.zeros(values.size, dtype=np.float64), tolerance=1e-18, maxIterations=1000, zero_initial_guess=True)
      assert status == 0
      actual_solution = linear_solver.solution.get()
      np.testing.assert_allclose(actual_solution, np.linalg.solve(expected_hessian, rhs_values), rtol=2e-8, atol=2e-8)
      assert np.linalg.norm(expected_hessian @ actual_solution - rhs_values) / np.linalg.norm(rhs_values) < 2e-8


@pytest.mark.parametrize("restrict_targets", [False, True])
def test_mixed_target_dimensions_and_global_offsets(restrict_targets):
  model = scene(f"sparsity_gpu_mixed_targets_{restrict_targets}")
  mesh = model.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=7)
  a = vertices.addAttribute("a", rows=2, cols=1)
  b = vertices.addAttribute("b", rows=3, cols=1)
  a_values = np.linspace(-0.7, 0.6, 14).reshape(-1, 2)
  b_values = np.linspace(0.5, -0.4, 21).reshape(-1, 3)
  a.updateValue(a_values)
  b.updateValue(b_values)
  joined_values = attribute.to_array([a[0], b[0], b[1], a[1], b[2]], rows=5, cols=1)
  jacobian = np.array([[0.2, 0, -0.4, 0, 0], [0, 0.7, 0, -0.3, 0.6], [0.9, 0, 0.8, 0, 0], [0, -0.2, 0, 0.5, 0.4]])
  boundary = vertices.addAttribute("boundary", computed_attribute=_linear(joined_values, jacobian))
  terms = mesh.addPrimitive("terms", numInstances=7)
  connectivity = terms.addConnectivity("to_vertices", vertices, np.arange(6, -1, -1, dtype=np.uint32), 1)
  joined_boundary = terms.addAttribute("boundary", through=connectivity, source=boundary)
  inner_hessian = _inner_matrix(4)
  linear = np.linspace(0.1, -0.2, 4)
  energy = terms.addAttribute("energy", computed_attribute=_quadratic(joined_boundary, inner_hessian, linear, model))
  targets = [b, a]
  local_targets = [a] if restrict_targets else []
  result = differentiator().diff2([energy], targets, targets, local_targets=local_targets, projection_method=1, separate_hessian_jacobian=True)

  expected_hessian = np.zeros((35, 35))
  expected_gradient = np.zeros(35)
  for index in range(7):
    indices = np.array([21 + 2 * index, 3 * index, 3 * index + 1, 21 + 2 * index + 1, 3 * index + 2])
    local_values = np.array([a_values[index, 0], b_values[index, 0], b_values[index, 1], a_values[index, 1], b_values[index, 2]])
    derivative = jacobian.copy()
    if restrict_targets:
      derivative[:, [1, 2, 4]] = 0.0
    local_hessian = derivative.T @ _project(inner_hessian, 1) @ derivative
    local_gradient = derivative.T @ (inner_hessian @ (jacobian @ local_values) + linear)
    _scatter_reference(expected_hessian, expected_gradient, indices, local_hessian, local_gradient)
  _check_values(result, expected_hessian, expected_gradient, [(7, 3), (7, 2)])


def test_union_padding_nested_joins_and_dynamic_static_addition():
  model = scene("sparsity_gpu_union_dynamic")
  mesh = model.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=11)
  position = vertices.addAttribute("position", rows=2, cols=1)
  values = np.sin(np.arange(22) * 0.31).reshape(-1, 2)
  position.updateValue(values)
  branch_connectivities = [np.array([[3, 1], [1, 1], [7, 0], [4, 8]], dtype=np.uint32), np.array([[5], [2], [10]], dtype=np.uint32)]
  branch_jacobians = [np.array([[0.3, 0, 0.7, 0], [0, -0.4, 0, 0.9], [0.2, 0, -0.6, 0]]), np.array([[0.5, 0], [0, 0.8], [-0.3, 0]])]
  branches = []
  boundary_maps = []
  for branch_index, (connectivity_values, jacobian) in enumerate(zip(branch_connectivities, branch_jacobians)):
    branch = mesh.addPrimitive(f"branch_{branch_index}", numInstances=len(connectivity_values))
    connectivity = branch.addConnectivity("to_vertices", vertices, connectivity_values, connectivity_values.shape[1])
    joined_position = branch.addAttribute("position", through=connectivity, source=position)
    branch.addAttribute("boundary", computed_attribute=_linear(joined_position, jacobian))
    branches.append(branch)
    for vertex_indices in connectivity_values:
      global_map = np.zeros((3, values.size))
      indices = (2 * vertex_indices[:, None] + np.arange(2)).ravel()
      for column, global_column in enumerate(indices):
        global_map[:, global_column] += jacobian[:, column]
      boundary_maps.append(global_map)
  union = mesh.addPrimitiveUnion("union", branches)
  union_boundary = union.addAttribute("boundary")
  outer_values = np.array([[0, 4], [5, 2], [1, 1], [6, 3]], dtype=np.uint32)
  terms = mesh.addPrimitive("terms", numInstances=len(outer_values), isDynamic=True)
  connectivity = terms.addConnectivity("to_union", union, outer_values, 2)
  joined_boundary = terms.addAttribute("boundary", through=connectivity, source=union_boundary)
  inner_hessian = _inner_matrix(6)
  linear = np.linspace(-0.2, 0.3, 6)
  energy = terms.addAttribute("energy", computed_attribute=_quadratic(joined_boundary, inner_hessian, linear, model))
  dynamic = differentiator().diff2([energy], [position], [position], projection_method=1, separate_hessian_jacobian=True, dynamic_instances=True)
  inertia = vertices.addAttribute("inertia", computed_attribute=0.25 * position.dot(position))
  static = differentiator().diff2([inertia], [position], [position], projection_method=-1, separate_hessian_jacobian=True)
  combined = static + dynamic

  # Changing active branches changes the padded local coordinate stream. The
  # graph, derivative kernel, and precomputed sparsity layout stay the same.
  for active in (outer_values, np.array([[6, 0]], dtype=np.uint32), np.empty((0, 2), dtype=np.uint32), np.array([[4, 5], [3, 0], [2, 2], [6, 1]], dtype=np.uint32)):
    terms.updateNumInstances(len(active))
    if len(active) > 0:
      connectivity.updateConnectivity(active)
    expected_hessian = 0.5 * np.eye(values.size)
    expected_gradient = 0.5 * values.ravel()
    for first, second in active:
      jacobian = np.vstack([boundary_maps[int(first)], boundary_maps[int(second)]])
      expected_hessian += jacobian.T @ _project(inner_hessian, 1) @ jacobian
      expected_gradient += jacobian.T @ (inner_hessian @ jacobian @ values.ravel() + linear)
    _check_values(combined, expected_hessian, expected_gradient, [(len(values), 2)])


def test_nonlinear_join_keeps_full_second_order_chain_rule():
  model = scene("sparsity_gpu_nonlinear_fallback")
  mesh = model.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=5)
  position = vertices.addAttribute("position", rows=2, cols=1)
  values = np.array([[-0.8, -0.4], [0.2, -0.3], [-0.1, 0.7], [0.6, 0.4], [0.3, -0.8]])
  position.updateValue(values)
  boundary = vertices.addAttribute("boundary", computed_attribute=attribute.to_array([position[0] * position[0] + position[1], position[1] * position[1] - position[0]], rows=2, cols=1))
  terms = mesh.addPrimitive("terms", numInstances=5)
  connectivity = terms.addConnectivity("to_vertices", vertices, np.arange(5, dtype=np.uint32), 1)
  joined_boundary = terms.addAttribute("boundary", through=connectivity, source=boundary)
  energy = terms.addAttribute("energy", computed_attribute=0.5 * joined_boundary.dot(joined_boundary))
  hessian = differentiator().diff2([energy], [position], [position], projection_method=1, separate_hessian_jacobian=True)
  assert hessian.project_entire_hessian == [True]
  expected_hessian = np.zeros((10, 10))
  expected_gradient = np.zeros(10)
  for index, (x, y) in enumerate(values):
    boundary_value = np.array([x * x + y, y * y - x])
    jacobian = np.array([[2.0 * x, 1.0], [-1.0, 2.0 * y]])
    # The second term is essential: reducing the derivative to J.T @ H @ J
    # would silently turn this nonlinear case into a Gauss-Newton Hessian.
    local_hessian = jacobian.T @ jacobian + np.diag(2.0 * boundary_value)
    expected_hessian[2 * index:2 * index + 2, 2 * index:2 * index + 2] = _project(local_hessian, 1)
    expected_gradient[2 * index:2 * index + 2] = jacobian.T @ boundary_value
  _check_values(hessian, expected_hessian, expected_gradient, [(len(values), 2)])


def test_cancelled_linear_boundary_has_empty_jacobian():
  model = scene("sparsity_gpu_empty_jacobian")
  mesh = model.addMesh("mesh")
  vertices = mesh.addPrimitive("vertices", numInstances=3)
  position = vertices.addAttribute("position", rows=2, cols=1)
  position.updateValue(np.array([[0.1, 0.2], [-0.3, 0.5], [0.7, -0.2]]))
  # These expressions retain position in the dependency graph, but both
  # derivative rows simplify to zero. They exercise an empty sparse J with
  # an otherwise valid coordinate stream and nonzero projected inner H.
  first = (position[0] + position[1]) - (position[1] + position[0])
  second = 2.0 * position[0] - (position[0] + position[0])
  boundary = vertices.addAttribute("boundary", computed_attribute=attribute.to_array([first, second], rows=2, cols=1))
  terms = mesh.addPrimitive("terms", numInstances=3)
  connectivity = terms.addConnectivity("to_vertices", vertices, np.arange(3, dtype=np.uint32), 1)
  joined_boundary = terms.addAttribute("boundary", through=connectivity, source=boundary)
  energy = terms.addAttribute("energy", computed_attribute=_quadratic(joined_boundary, _inner_matrix(2), np.array([0.3, -0.2]), model))
  hessian = differentiator().diff2([energy], [position], [position], projection_method=1, separate_hessian_jacobian=True)
  assert all(hessian.global_jacobians[0][index].isZero > 0 for index in range(hessian.global_jacobians[0].size))
  _check_values(hessian, np.zeros((6, 6)), np.zeros(6), [(3, 2)])

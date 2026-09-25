"""Regression tests for numerical/topological separation in MAS.

The CPU tests import the standalone MAS implementation from this checkout,
without requiring compiled YASPS extensions or a CUDA context. GPU tests use
real ``yasps.matrix`` storage and skip only when CUDA/YASPS are unavailable.

Run just the CPU contract tests with::

  python -m unittest discover -s yasps/tests -p test_mas_explicit_hierarchy.py -k CPU

Run this file directly to include the CUDA adapter/facade tests. All dense
references are deliberately tiny (at most eight scalar degrees of freedom).
"""

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_MAS_DIRECTORY = Path(__file__).resolve().parents[1] / "yasps" / "solver" / "mas"
_PACKAGE_NAME = "_yasps_explicit_hierarchy_test_mas"
_SPEC = importlib.util.spec_from_file_location(
  _PACKAGE_NAME,
  _MAS_DIRECTORY / "__init__.py",
  submodule_search_locations=[str(_MAS_DIRECTORY)],
)
_MAS = importlib.util.module_from_spec(_SPEC)
sys.modules[_PACKAGE_NAME] = _MAS
_SPEC.loader.exec_module(_MAS)
BlockSparseMatrixView = _MAS.BlockSparseMatrixView
MASSolver = _MAS.MASSolver


def _topology(dimensions, edges, dtype=np.int64):
  """Pack per-block shapes, not per-category shapes or node coordinates."""
  dimensions = np.asarray(dimensions, dtype=np.int64)
  offsets = np.cumsum(np.r_[0, dimensions[:-1]], dtype=np.int64)
  pairs = [(node, node) for node in range(len(dimensions))] + list(edges)
  coordinates = np.asarray(
    [(offsets[row], offsets[col]) for row, col in pairs], dtype=dtype,
  ).ravel()
  shapes = np.asarray(
    [(dimensions[row], dimensions[col]) for row, col in pairs], dtype=dtype,
  ).ravel()
  return coordinates, shapes, len(pairs)


def _system(variant=0, dynamic=True, dimensions=(2, 1, 2, 1), static_variant=None):
  """Build block input and an independent dense SPD reference."""
  if static_variant is None:
    static_variant = variant
  dimensions = np.asarray(dimensions, dtype=np.int64)
  offsets = np.cumsum(np.r_[0, dimensions[:-1]], dtype=np.int64)
  count = int(dimensions.sum())
  static = []
  for node, size in enumerate(dimensions):
    diagonal = np.diag(4.0 + variant + np.arange(int(size)) / 5 + node / 4)
    diagonal += 0.05 * np.ones((size, size))
    static.append(((node, node), diagonal))
  static_edges = ((0, 2), (1, 3)) if static_variant == 0 else ((0, 3), (1, 2))
  for row, col in static_edges:
    shape = (int(dimensions[row]), int(dimensions[col]))
    values = -0.025 * (1 + variant) * np.arange(1, 1 + np.prod(shape))
    static.append(((row, col), values.reshape(shape)))

  moving = []
  if dynamic:
    # Dynamic diagonal contributions must add to, not replace, static data.
    for node, size in enumerate(dimensions):
      moving.append(((node, node), np.eye(size) * 0.3))
    dynamic_edges = ((0, 1), (2, 3)) if variant == 0 else ((0, 2), (1, 3))
    for row, col in dynamic_edges:
      moving.append(((row, col), np.full((dimensions[row], dimensions[col]), -0.07)))

  reference = np.zeros((count, count), dtype=np.float64)
  for (row, col), values in static + moving:
    row_slice = slice(int(offsets[row]), int(offsets[row] + dimensions[row]))
    col_slice = slice(int(offsets[col]), int(offsets[col] + dimensions[col]))
    reference[row_slice, col_slice] += values
    if row != col:
      reference[col_slice, row_slice] += values.T
  view = BlockSparseMatrixView.from_blocks(dimensions, static, moving)
  return view, reference


def _host_snapshot(view):
  return {
    f"{part}_{field}": np.array(getattr(view, f"{part}_{field}"), copy=True)
    for part in ("static", "dynamic")
    for field in ("values", "positions", "category_starts", "category_counts", "block_dimensions")
  }


class ExplicitHierarchyCPUTests(unittest.TestCase):
  def make_solver(self):
    return MASSolver(
      inverse_backend="cpu_reference", max_domain_dofs=4,
      max_levels=3, target_nodes_per_partition=2,
    )

  def assert_cpu_solution(self, implementation, view, reference):
    before = _host_snapshot(view)
    rhs = np.linspace(-0.4, 0.7, view.rows)
    implementation.solve(view, rhs, tolerance=1e-22, max_iterations=100)
    self.assertTrue(implementation.statistics.converged, implementation.statistics)
    np.testing.assert_allclose(
      implementation.solution, np.linalg.solve(reference, rhs), rtol=1e-9, atol=1e-11,
    )
    probe = np.linspace(0.7, -0.3, view.rows)
    numeric = implementation._numeric
    actual = numeric.levels[0].matvec(probe, implementation.hierarchy.levels[0])
    np.testing.assert_allclose(actual, reference @ probe, rtol=1e-13, atol=1e-13)
    for field, previous in before.items():
      np.testing.assert_array_equal(getattr(view, field), previous)

  def test_global_offsets_heterogeneous_dimensions_and_deduplicated_graph(self):
    implementation = self.make_solver()
    dimensions = (2, 1, 3, 2)
    edges = ((0, 1), (1, 0), (0, 1), (2, 3), (0, 3))
    coordinates, shapes, count = _topology(dimensions, edges)
    # The API cannot assume blocks arrive grouped by dimension or sorted.
    permutation = np.arange(count)[::-1]
    coordinates = coordinates.reshape(-1, 2)[permutation].ravel()
    shapes = shapes.reshape(-1, 2)[permutation].ravel()
    coordinates_before, shapes_before = coordinates.copy(), shapes.copy()
    implementation.rebuild_hierarchy_from_blocks(coordinates, shapes, count)
    fine = implementation.hierarchy.levels[0]
    np.testing.assert_array_equal(fine.node_dimensions, dimensions)
    np.testing.assert_array_equal(fine.node_scalar_offsets, (0, 2, 3, 6))
    self.assertEqual(fine.graph.edges, ((0, 1), (0, 3), (2, 3)))
    self.assertEqual(implementation.hierarchy_build_count, 1)
    self.assertIsNone(implementation._numeric)
    self.assertIsNone(implementation._cuda_runtime)
    np.testing.assert_array_equal(coordinates, coordinates_before)
    np.testing.assert_array_equal(shapes, shapes_before)

  def test_invalid_topology_metadata_is_rejected(self):
    valid_coordinates = np.array([0, 0, 2, 2], dtype=np.int64)
    valid_shapes = np.array([2, 2, 1, 1], dtype=np.int64)
    invalid = [
      ("short coordinates", valid_coordinates[:-1], valid_shapes, 2),
      ("long coordinates", np.r_[valid_coordinates, 0, 0], valid_shapes, 2),
      ("short dimensions", valid_coordinates, valid_shapes[:-1], 2),
      ("long dimensions", valid_coordinates, np.r_[valid_shapes, 1, 1], 2),
      ("negative count", valid_coordinates, valid_shapes, -1),
      ("fractional count", valid_coordinates, valid_shapes, 2.5),
      ("boolean count", valid_coordinates, valid_shapes, True),
      ("count too large", valid_coordinates, valid_shapes, 3),
      ("count too small", valid_coordinates, valid_shapes, 1),
      ("float coordinates", valid_coordinates.astype(float), valid_shapes, 2),
      ("float dimensions", valid_coordinates, valid_shapes.astype(float), 2),
      ("rank two coordinates", valid_coordinates.reshape(2, 2), valid_shapes, 2),
      ("rank two dimensions", valid_coordinates, valid_shapes.reshape(2, 2), 2),
      ("negative offset", np.array([-1, -1, 2, 2]), valid_shapes, 2),
      ("zero dimension", valid_coordinates, np.array([2, 2, 0, 0]), 2),
      ("negative dimension", valid_coordinates, np.array([2, 2, -1, -1]), 2),
      ("missing initial variable", np.array([2, 2]), np.array([1, 1]), 1),
      ("gap in variable coverage", np.array([0, 0, 3, 3]), valid_shapes, 2),
      ("start inside variable", np.array([0, 0, 1, 1]), valid_shapes, 2),
      ("conflicting dimensions", np.array([0, 0, 0, 2]), np.array([2, 2, 1, 1]), 2),
      (
        "total dimension overflow",
        np.array([0, 0, np.iinfo(np.int64).max, np.iinfo(np.int64).max]),
        np.array([np.iinfo(np.int64).max, np.iinfo(np.int64).max, 1, 1]), 2,
      ),
    ]
    for label, coordinates, shapes, count in invalid:
      with self.subTest(case=label):
        with self.assertRaises((TypeError, ValueError)):
          self.make_solver().rebuild_hierarchy_from_blocks(coordinates, shapes, count)

  def test_explicit_graph_survives_actual_static_and_dynamic_connectivity_changes(self):
    implementation = self.make_solver()
    edges = ((0, 1), (1, 2), (2, 3))
    implementation.rebuild_hierarchy_from_blocks(*_topology((2, 1, 2, 1), edges))
    hierarchy = implementation.hierarchy
    for variant, dynamic in ((0, True), (1, True), (1, False)):
      with self.subTest(variant=variant, dynamic=dynamic):
        view, reference = _system(variant, dynamic)
        self.assert_cpu_solution(implementation, view, reference)
        self.assertIs(implementation.hierarchy, hierarchy)
        self.assertEqual(hierarchy.levels[0].graph.edges, edges)
        self.assertEqual(implementation.hierarchy_build_count, 1)
        # (1, 2) exists only in the chosen topology for the first system.
        # Topological edges must not inject artificial numeric blocks.
        if variant == 0:
          self.assertNotIn((1, 2), implementation._numeric.levels[0].blocks)
          self.assertIn((0, 2), implementation._numeric.levels[0].blocks)

  def test_rebuild_discards_old_numeric_state_and_uses_new_graph(self):
    implementation = self.make_solver()
    view, reference = _system()
    first_edges = ((0, 1), (1, 2), (2, 3))
    second_edges = ((0, 2), (1, 3))
    implementation.rebuild_hierarchy_from_blocks(*_topology((2, 1, 2, 1), first_edges))
    self.assert_cpu_solution(implementation, view, reference)
    old_hierarchy = implementation.hierarchy
    old_numeric, old_preconditioner = implementation._numeric, implementation._preconditioner
    before = _host_snapshot(view)

    implementation.rebuild_hierarchy_from_blocks(*_topology((2, 1, 2, 1), second_edges))
    self.assertIsNot(implementation.hierarchy, old_hierarchy)
    self.assertEqual(implementation.hierarchy.levels[0].graph.edges, second_edges)
    self.assertIsNone(implementation._numeric)
    self.assertIsNone(implementation._preconditioner)
    self.assertIsNone(implementation._cuda_runtime)
    self.assertEqual(implementation.hierarchy_build_count, 2)
    for field, previous in before.items():
      np.testing.assert_array_equal(getattr(view, field), previous)
    second_hierarchy = implementation.hierarchy
    self.assert_cpu_solution(implementation, view, reference)
    self.assertIs(implementation.hierarchy, second_hierarchy)
    self.assertIsNot(implementation._numeric, old_numeric)
    self.assertIsNot(implementation._preconditioner, old_preconditioner)
    self.assertEqual(implementation.hierarchy_build_count, 2)

  def test_incompatible_actual_variable_layout_requires_explicit_rebuild(self):
    implementation = self.make_solver()
    implementation.rebuild_hierarchy_from_blocks(*_topology((2, 1, 2, 1), ((0, 1),)))
    hierarchy = implementation.hierarchy
    for dimensions in ((1, 2, 1, 2), (2, 1, 2, 2)):
      with self.subTest(dimensions=dimensions):
        view, _ = _system(dimensions=dimensions)
        with self.assertRaises(ValueError):
          implementation.solve(view, np.ones(view.rows))
        self.assertIs(implementation.hierarchy, hierarchy)
        self.assertEqual(implementation.hierarchy_build_count, 1)

  def test_actual_variable_type_ids_do_not_invalidate_explicit_layout(self):
    implementation = self.make_solver()
    implementation.rebuild_hierarchy_from_blocks(*_topology((2, 1, 2, 1), ((0, 2),)))
    hierarchy = implementation.hierarchy
    view, reference = _system()
    view.variable_type_ids = np.array([11, 12, 13, 14], dtype=np.int64)
    view.invalidate_static_structure()
    self.assert_cpu_solution(implementation, view, reference)
    self.assertIs(implementation.hierarchy, hierarchy)
    self.assertEqual(implementation.hierarchy_build_count, 1)

  def test_default_static_graph_builds_once_and_numerics_stay_current(self):
    implementation = self.make_solver()
    first, reference = _system(dynamic=False)
    self.assert_cpu_solution(implementation, first, reference)
    hierarchy = implementation.hierarchy
    self.assertEqual(hierarchy.levels[0].graph.edges, ((0, 2), (1, 3)))
    self.assertEqual(implementation.hierarchy_build_count, 1)

    # The default remains static-only; neither newly present dynamic edges
    # nor different actual static positions implicitly repartition it.
    for variant in (0, 1):
      changed, reference = _system(variant=variant, dynamic=True)
      self.assert_cpu_solution(implementation, changed, reference)
      self.assertIs(implementation.hierarchy, hierarchy)
      self.assertEqual(implementation.hierarchy_build_count, 1)

  def test_actual_layout_arrays_may_be_column_shaped(self):
    implementation = self.make_solver()
    implementation.rebuild_hierarchy_from_blocks(*_topology((2, 1, 2, 1), ((0, 2),)))
    hierarchy = implementation.hierarchy
    view, reference = _system()
    view.variable_dimensions = view.variable_dimensions.reshape(-1, 1)
    view.variable_scalar_offsets = view.variable_scalar_offsets.reshape(-1, 1)
    view.invalidate_static_structure()
    for _ in range(2):
      self.assert_cpu_solution(implementation, view, reference)
      self.assertIs(implementation.hierarchy, hierarchy)
      self.assertEqual(implementation.hierarchy_build_count, 1)


class ExplicitHierarchyGPUTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    try:
      import pycuda.driver as cuda
      cuda.init()
      if cuda.Device.count() == 0:
        raise unittest.SkipTest("No CUDA device is available")
      import pycuda.autoinit  # noqa: F401
      import pycuda.gpuarray as gpuarray
    except ImportError as error:
      raise unittest.SkipTest(f"PyCUDA is unavailable: {error}") from error
    except (cuda.Error, RuntimeError) as error:
      raise unittest.SkipTest(f"CUDA is unavailable: {error}") from error
    try:
      from yasps import matrix, vector
      jacobi = importlib.import_module("yasps.solver.jacobianPCGSolver")
    except ImportError as error:
      raise unittest.SkipTest(f"Compiled YASPS is unavailable: {error}") from error
    # A non-editable YASPS installation supplies the compiled matrix/vector
    # dependencies, but the Python adapter/facade under test must come from
    # this checkout. No solver methods or numerical kernels are patched.
    package_name = "_yasps_explicit_hierarchy_test_public"
    spec = importlib.util.spec_from_file_location(
      package_name,
      _MAS_DIRECTORY.parent / "__init__.py",
      submodule_search_locations=[str(_MAS_DIRECTORY.parent)],
    )
    public = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = public
    sys.modules[package_name + ".jacobianPCGSolver"] = jacobi
    spec.loader.exec_module(public)
    cls.gpuarray = gpuarray
    cls.matrix_type, cls.vector_type = matrix, vector
    cls.adapter_type, cls.facade_type = public.masSolver, public.solver

  def setUp(self):
    self.solvers = []

  def tearDown(self):
    for selected in self.solvers:
      selected.reset()

  def make_solver(self, facade=True):
    options = dict(max_domain_dofs=4, max_levels=3, target_nodes_per_partition=2)
    selected = self.facade_type("mas", **options) if facade else self.adapter_type(**options)
    self.solvers.append(selected)
    return selected

  def core(self, selected):
    adapter = selected.implementation if isinstance(selected, self.facade_type) else selected
    return adapter.implementation

  def gpu_topology(self, edges):
    coordinates, shapes, count = _topology((2, 1, 2, 1), edges, dtype=np.uint32)
    return self.gpuarray.to_gpu(coordinates), self.gpuarray.to_gpu(shapes), count

  def set_matrix_parts(self, matrix, view):
    for part in ("static", "dynamic"):
      suffix = "" if part == "static" else "_dynamic"
      for matrix_field, view_field in (
        ("blocks_flattened", "values"), ("block_positions", "positions"),
      ):
        values = getattr(view, f"{part}_{view_field}").ravel()
        if not values.size:
          # YASPS reserves a dummy buffer even for zero active blocks.
          values = np.zeros(2 if view_field == "positions" else 1, values.dtype)
        setattr(matrix, matrix_field + suffix, values)
      for matrix_field, view_field in (
        ("blocks_start_indices", "category_starts"),
        ("block_counts", "category_counts"), ("block_dimensions", "block_dimensions"),
      ):
        setattr(matrix, matrix_field + suffix, getattr(view, f"{part}_{view_field}").ravel().tolist())

  def make_matrix(self, view):
    matrix = self.matrix_type(view.rows, view.cols, symmetric_storage=True)
    self.set_matrix_parts(matrix, view)
    return matrix

  def matrix_snapshot(self, matrix):
    return {
      field + suffix: getattr(matrix, field + suffix).get()
      for suffix in ("", "_dynamic")
      for field in ("blocks_flattened", "block_positions")
    }

  def assert_gpu_solution(self, selected, matrix, reference):
    before = self.matrix_snapshot(matrix)
    rhs_host = np.linspace(-0.4, 0.7, matrix.rows)
    rhs = self.gpuarray.to_gpu(rhs_host)
    result = selected.computeSolution(
      matrix, rhs, None, tolerance=1e-16, maxIterations=100, zero_initial_guess=True,
    )
    self.assertEqual(result, 0, selected.statistics)
    self.assertEqual(selected.statistics["execution_backend"], "cuda")
    solution = selected.solution.get()
    np.testing.assert_allclose(solution, np.linalg.solve(reference, rhs_host), rtol=1e-7, atol=1e-9)
    self.assertLess(np.linalg.norm(reference @ solution - rhs_host) / np.linalg.norm(rhs_host), 1e-7)

    probe_host = np.linspace(0.7, -0.3, matrix.rows)
    probe = self.vector_type(matrix.rows)
    probe.updateValue(probe_host)
    np.testing.assert_allclose(matrix.spmv(probe).value.get(), reference @ probe_host, rtol=1e-12, atol=1e-12)
    runtime = self.core(selected)._cuda_runtime
    np.testing.assert_allclose(runtime.matvec(probe.value).get(), reference @ probe_host, rtol=1e-12, atol=1e-12)
    for field, previous in before.items():
      np.testing.assert_array_equal(getattr(matrix, field).get(), previous)

  def test_facade_keeps_explicit_graph_across_actual_numeric_connectivity_changes(self):
    selected = self.make_solver()
    coordinates, shapes, count = self.gpu_topology(((0, 1), (1, 2), (2, 3)))
    coordinates_before, shapes_before = coordinates.get(), shapes.get()
    selected.rebuildHierarchy(coordinates, shapes, count)
    core = self.core(selected)
    hierarchy = core.hierarchy

    first_view, first_reference = _system()
    first_matrix = self.make_matrix(first_view)
    self.assert_gpu_solution(selected, first_matrix, first_reference)
    self.assertIs(core.hierarchy, hierarchy)
    self.assertEqual(core.hierarchy_build_count, 1)
    first_runtime = core._cuda_runtime

    # A matrix's static topology is immutable between explicit rebuilds.
    # New values and arbitrary dynamic metadata must reuse its runtime.
    changed, reference = _system(variant=1, static_variant=0)
    self.set_matrix_parts(first_matrix, changed)
    self.assert_gpu_solution(selected, first_matrix, reference)
    self.assertIs(core.hierarchy, hierarchy)
    self.assertIs(core._cuda_runtime, first_runtime)
    self.assertEqual(core.hierarchy_build_count, 1)

    # A different matrix can have different static coordinates and shapes.
    # It must recreate numerical scatter maps, not the selected hierarchy.
    different, reference = _system(variant=1)
    self.assert_gpu_solution(selected, self.make_matrix(different), reference)
    self.assertIs(core.hierarchy, hierarchy)
    self.assertIsNot(core._cuda_runtime, first_runtime)
    self.assertEqual(core.hierarchy.levels[0].graph.edges, ((0, 1), (1, 2), (2, 3)))
    self.assertEqual(core.hierarchy_build_count, 1)
    np.testing.assert_array_equal(coordinates.get(), coordinates_before)
    np.testing.assert_array_equal(shapes.get(), shapes_before)

  def test_direct_adapter_rebuild_replaces_gpu_runtime_and_connectivity(self):
    selected = self.make_solver(facade=False)
    selected.rebuildHierarchy(*self.gpu_topology(((0, 1), (1, 2), (2, 3))))
    view, reference = _system()
    matrix = self.make_matrix(view)
    self.assert_gpu_solution(selected, matrix, reference)
    core = self.core(selected)
    old_hierarchy, old_runtime = core.hierarchy, core._cuda_runtime
    before = self.matrix_snapshot(matrix)

    selected.rebuildHierarchy(*self.gpu_topology(((0, 2), (1, 3))))
    self.assertIsNot(core.hierarchy, old_hierarchy)
    self.assertIsNone(core._cuda_runtime)
    self.assertEqual(core.hierarchy.levels[0].graph.edges, ((0, 2), (1, 3)))
    self.assertEqual(core.hierarchy_build_count, 2)
    for field, previous in before.items():
      np.testing.assert_array_equal(getattr(matrix, field).get(), previous)

    second_hierarchy = core.hierarchy
    changed_view, changed_reference = _system(variant=1)
    self.set_matrix_parts(matrix, changed_view)
    self.assert_gpu_solution(selected, matrix, changed_reference)
    self.assertIs(core.hierarchy, second_hierarchy)
    self.assertIsNot(core._cuda_runtime, old_runtime)
    self.assertEqual(core.hierarchy_build_count, 2)

  def test_gpu_api_requires_flat_integer_device_arrays_and_matching_counts(self):
    for facade in (False, True):
      selected = self.make_solver(facade=facade)
      coordinates, shapes, count = self.gpu_topology(((0, 1),))
      cases = [
        (coordinates.get(), shapes, count),
        (coordinates, shapes.get(), count),
        (coordinates.astype(np.float64), shapes, count),
        (coordinates, shapes.astype(np.float64), count),
        (coordinates[:-1], shapes, count),
        (coordinates, shapes[:-1], count),
        (coordinates.reshape((count, 2)), shapes, count),
        (coordinates, shapes.reshape((count, 2)), count),
        (coordinates, shapes, count + 1),
        (coordinates, shapes, -1),
        (coordinates, shapes, 1.5),
      ]
      for case, arguments in enumerate(cases):
        with self.subTest(facade=facade, case=case):
          with self.assertRaises((TypeError, ValueError)):
            selected.rebuildHierarchy(*arguments)

  def test_gpu_actual_layout_mismatch_is_rejected_without_rebuilding(self):
    selected = self.make_solver()
    selected.rebuildHierarchy(*self.gpu_topology(((0, 1),)))
    core = self.core(selected)
    hierarchy = core.hierarchy
    view, _ = _system(dimensions=(1, 2, 1, 2))
    rhs = self.gpuarray.to_gpu(np.ones(view.rows))
    with self.assertRaises(ValueError):
      selected.computeSolution(self.make_matrix(view), rhs, None, zero_initial_guess=True)
    self.assertIs(core.hierarchy, hierarchy)
    self.assertEqual(core.hierarchy_build_count, 1)

  def test_gpu_default_static_only_graph_still_solves_current_numerics(self):
    selected = self.make_solver()
    first_view, reference = _system(dynamic=False)
    matrix = self.make_matrix(first_view)
    self.assert_gpu_solution(selected, matrix, reference)
    core = self.core(selected)
    hierarchy = core.hierarchy
    self.assertEqual(hierarchy.levels[0].graph.edges, ((0, 2), (1, 3)))
    changed, reference = _system(variant=1, static_variant=0)
    self.set_matrix_parts(matrix, changed)
    self.assert_gpu_solution(selected, matrix, reference)
    self.assertIs(core.hierarchy, hierarchy)
    self.assertEqual(core.hierarchy_build_count, 1)

    different, reference = _system(variant=1)
    self.assert_gpu_solution(selected, self.make_matrix(different), reference)
    self.assertIs(core.hierarchy, hierarchy)
    self.assertEqual(core.hierarchy_build_count, 1)


if __name__ == "__main__":
  unittest.main(verbosity=2)

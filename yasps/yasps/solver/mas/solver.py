"""Public standalone MAS solver API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .hierarchy import Hierarchy, build_hierarchy
from .block_graph import BlockGraph
from .cuda_runtime import DeviceMASRuntime, is_pycuda_array
from .local_inverse import make_inverse_backend
from .matrix_view import BlockSparseMatrixView, to_host
from .numeric_assembly import NumericHierarchy, assemble_numeric_hierarchy
from .pcg import PCGResult, pcg
from .preconditioner import MASPreconditioner


@dataclass
class SolverStatistics:
  solve_mode: str = ""
  execution_backend: str = "cpu"
  converged: bool = False
  iterations: int = 0
  coarse_iterations: int = 0
  final_residual: float = float("inf")
  relative_residual: float = float("inf")
  breakdown: str | None = None
  hierarchy_build_count: int = 0
  hierarchy_build_seconds: float = 0.0
  metis_seconds_per_level: list[float] = field(default_factory=list)
  ordering_backends: list[str] = field(default_factory=list)
  nodes_per_level: list[int] = field(default_factory=list)
  domains_per_level: list[int] = field(default_factory=list)
  coarsening_ratios: list[float] = field(default_factory=list)
  domain_scalar_sizes: list[list[int]] = field(default_factory=list)
  numeric_coarse_assembly_seconds: float = 0.0
  numeric_update_wall_seconds: float = 0.0
  local_matrix_assembly_seconds: float = 0.0
  local_inverse_seconds: float = 0.0
  inverse_backends: list[str] = field(default_factory=list)
  inverse_fallback_reasons: list[str] = field(default_factory=list)
  mas_application_seconds: float = 0.0
  mas_applications: int = 0
  fine_spmv_seconds: float = 0.0
  pcg_seconds: float = 0.0
  total_solve_seconds: float = 0.0
  gpu_memory_bytes: int = 0
  cuda_runtime_build_count: int = 0
  numeric_rebuild_count: int = 0
  numeric_state_reused: bool = False
  numeric_rebuild_reason: str = ""
  dynamic_block_count: int = 0
  dynamic_upload_reallocations: int = 0
  upload_capacities: dict[str, int] = field(default_factory=dict)
  spmv_kernel_compile_count: int = 0
  spmv_kernel_cache_hits: int = 0
  spmv_kernel_compile_seconds: float = 0.0
  static_assembly_kernel_compile_seconds: float = 0.0
  dynamic_assembly_kernel_compile_seconds: float = 0.0
  jit_library_compile_count: int = 0
  jit_library_cache_hits: int = 0
  jit_library_compile_seconds: float = 0.0
  jit_library_paths: list[str] = field(default_factory=list)
  spmv_specializations: list[tuple[int, int]] = field(default_factory=list)
  pcg_graph_build_count: int = 0
  pcg_graph_cache_hits: int = 0
  pcg_graph_build_seconds: float = 0.0
  configured_max_domain_dofs: int = 0
  configured_max_levels: int = 0
  configured_target_nodes_per_partition: int = 0
  configured_minimum_domains_for_next_level: int = 0
  configured_fixed_inverse_bucket_size: int = 0
  configured_duplicate_level_weight: float = 0.0
  configured_adaptive_fine_level_weight: bool = False
  configured_dense_collision_fine_level_weight: float = 0.0
  configured_dense_collision_block_ratio: float = 0.0
  configured_dynamic_edge_block_ratio: float = 0.0
  configured_dynamic_numeric_rebuild_interval: int = 1
  active_fine_level_weight: float = 0.0
  dynamic_edge_domains_active: bool = False

  def as_dict(self) -> dict:
    return asdict(self)


class MASSolver:
  def __init__(
    self,
    max_domain_dofs: int | None = None,
    max_levels: int | None = None,
    target_nodes_per_partition: int | None = None,
    fine_correction_iterations: int = 8,
    solve_mode: str = "mas_pcg",
    inverse_backend: str = "cooperative_gauss_jordan",
    *,
    minimum_domains_for_next_level: int = 0,
    cuda_threads_per_block: int = 96,
    cuda_fixed_inverse_bucket_size: int | None = None,
    dynamic_edge_domains: bool = False,
    dynamic_edge_block_ratio: float = 0.0,
    collision_aware_reorder: bool = False,
    collision_merge_across_types: bool = False,
    duplicate_level_weight: float = 0.5,
    adaptive_fine_level_weight: bool | None = None,
    dense_collision_fine_level_weight: float = 3.0,
    dense_collision_block_ratio: float = 0.1,
    coarsest_level_weight: float = 2.5,
    level_weights: tuple[float, ...] | None = None,
    cuda_enable_inverse_fallback: bool = True,
    mixed_spmv: bool = False,
    mixed_spmv_tolerance_factor: float = 1.0,
    global_coarse_domain: bool = True,
    domain_dof_schedule: tuple[int, ...] | None = None,
    target_node_schedule: tuple[int, ...] | None = None,
    numeric_rebuild_interval: int = 1,
    dynamic_numeric_rebuild_interval: int = 1,
    allow_cpu_fallback: bool = True,
  ):
    if (any(value is not None and value <= 0 for value in (
        max_domain_dofs, max_levels, target_nodes_per_partition)) or
        fine_correction_iterations < 0):
      raise ValueError("domain capacity/levels must be positive and correction iterations non-negative")
    if int(minimum_domains_for_next_level) < 0:
      raise ValueError(
        "minimum_domains_for_next_level must be non-negative"
      )
    if cuda_fixed_inverse_bucket_size is not None and cuda_fixed_inverse_bucket_size <= 0:
      raise ValueError("cuda_fixed_inverse_bucket_size must be positive")
    if solve_mode not in ("mas_pcg", "reduced_then_fine"):
      raise ValueError("solve_mode must be 'mas_pcg' or 'reduced_then_fine'")
    self.max_domain_dofs = None if max_domain_dofs is None else int(max_domain_dofs)
    self.max_levels = None if max_levels is None else int(max_levels)
    self.target_nodes_per_partition = (
      None if target_nodes_per_partition is None
      else int(target_nodes_per_partition)
    )
    self._configured_max_domain_dofs = 0
    self._configured_max_levels = 0
    self._configured_target_nodes_per_partition = 0
    self.minimum_domains_for_next_level = int(
      minimum_domains_for_next_level
    )
    self.fine_correction_iterations = int(fine_correction_iterations)
    self.solve_mode = solve_mode
    self.inverse_backend_name = inverse_backend
    self.cuda_threads_per_block = int(cuda_threads_per_block)
    self.cuda_fixed_inverse_bucket_size = (
      None if cuda_fixed_inverse_bucket_size is None
      else int(cuda_fixed_inverse_bucket_size)
    )
    self.dynamic_edge_domains = bool(dynamic_edge_domains)
    if (not np.isfinite(dynamic_edge_block_ratio)
        or dynamic_edge_block_ratio < 0.0):
      raise ValueError(
        "dynamic_edge_block_ratio must be finite and non-negative"
      )
    self.dynamic_edge_block_ratio = float(dynamic_edge_block_ratio)
    self.collision_aware_reorder = bool(collision_aware_reorder)
    self.collision_merge_across_types = bool(
      collision_merge_across_types
    )
    if (not np.isfinite(duplicate_level_weight)
        or not 0.0 <= duplicate_level_weight <= 1.0):
      raise ValueError("duplicate_level_weight must be between zero and one")
    self.duplicate_level_weight = float(duplicate_level_weight)
    self.adaptive_fine_level_weight = (
      level_weights is None if adaptive_fine_level_weight is None
      else bool(adaptive_fine_level_weight)
    )
    if (not np.isfinite(dense_collision_fine_level_weight)
        or dense_collision_fine_level_weight <= 0):
      raise ValueError("dense collision fine-level weight must be positive")
    if (not np.isfinite(dense_collision_block_ratio)
        or dense_collision_block_ratio <= 0):
      raise ValueError("dense collision block ratio must be positive")
    self.dense_collision_fine_level_weight = float(
      dense_collision_fine_level_weight
    )
    self.dense_collision_block_ratio = float(
      dense_collision_block_ratio
    )
    if not np.isfinite(coarsest_level_weight) or coarsest_level_weight <= 0:
      raise ValueError("coarsest_level_weight must be finite and positive")
    self.coarsest_level_weight = float(coarsest_level_weight)
    if level_weights is not None and (
      not level_weights
      or any(not np.isfinite(weight) or weight <= 0
         for weight in level_weights)
    ):
      raise ValueError("level_weights must contain finite positive values")
    self.level_weights = (
      None if level_weights is None
      else tuple(float(weight) for weight in level_weights)
    )
    self.cuda_enable_inverse_fallback = bool(
      cuda_enable_inverse_fallback
    )
    self.mixed_spmv = bool(mixed_spmv)
    if (not np.isfinite(mixed_spmv_tolerance_factor)
        or not 0.0 < mixed_spmv_tolerance_factor <= 1.0):
      raise ValueError(
        "mixed_spmv_tolerance_factor must be in (0, 1]"
      )
    self.mixed_spmv_tolerance_factor = float(
      mixed_spmv_tolerance_factor
    )
    self.global_coarse_domain = bool(global_coarse_domain)
    self.domain_dof_schedule = (
      None if domain_dof_schedule is None else
      tuple(int(value) for value in domain_dof_schedule)
    )
    self.target_node_schedule = (
      None if target_node_schedule is None else
      tuple(int(value) for value in target_node_schedule)
    )
    if int(numeric_rebuild_interval) <= 0:
      raise ValueError("numeric rebuild interval must be positive")
    if int(dynamic_numeric_rebuild_interval) <= 0:
      raise ValueError(
        "dynamic numeric rebuild interval must be positive"
      )
    self.numeric_rebuild_interval = int(numeric_rebuild_interval)
    self.dynamic_numeric_rebuild_interval = int(
      dynamic_numeric_rebuild_interval
    )
    self._numeric_rebuild_age = 0
    self._preconditioner_dynamic_block_count = 0
    self._preconditioner_dynamic_edge_active = False
    self.allow_cpu_fallback = bool(allow_cpu_fallback)
    self._hierarchy: Hierarchy | None = None
    self._numeric: NumericHierarchy | None = None
    self._preconditioner: MASPreconditioner | None = None
    self._cuda_runtime: DeviceMASRuntime | None = None
    self._solution: np.ndarray | None = None
    self._statistics = SolverStatistics(solve_mode=solve_mode)
    self._hierarchy_build_count = 0
    self._cuda_runtime_build_count = 0

  @property
  def solution(self):
    return None if self._solution is None else self._solution.copy()

  @property
  def device_solution(self):
    """Borrow the current solution buffer without copying it.

    This is intended for device-resident adapters that immediately consume
    the solution in the same CUDA context. Callers that need ownership
    should continue to use :attr:`solution`.
    """
    return self._solution

  @property
  def statistics(self) -> SolverStatistics:
    return self._statistics

  @property
  def hierarchy(self) -> Hierarchy | None:
    return self._hierarchy

  @property
  def hierarchy_build_count(self) -> int:
    return self._hierarchy_build_count

  def build_hierarchy(self, matrix_view: BlockSparseMatrixView) -> Hierarchy:
    if not isinstance(matrix_view, BlockSparseMatrixView):
      raise TypeError("matrix_view must be a BlockSparseMatrixView")
    dimensions = to_host(matrix_view.variable_dimensions, np.int64).reshape(-1)
    graph = BlockGraph.from_static_view(matrix_view)
    maximum_dimension = int(dimensions.max(initial=1))
    # The default describes the requested MAS policy, not a matrix layout.
    # A singleton variable can exceed that policy, so only raise capacity
    # enough to hold the largest dimension observed at runtime.
    recommended = (max(48, maximum_dimension), 6, 8)
    max_domain_dofs = self.max_domain_dofs or recommended[0]
    max_levels = self.max_levels or recommended[1]
    target_nodes = self.target_nodes_per_partition or recommended[2]
    self._configured_max_domain_dofs = max_domain_dofs
    self._configured_max_levels = max_levels
    self._configured_target_nodes_per_partition = target_nodes
    hierarchy = build_hierarchy(
      matrix_view, max_domain_dofs=max_domain_dofs,
      max_levels=max_levels,
      target_nodes_per_partition=target_nodes,
      minimum_domains_for_next_level=(
        self.minimum_domains_for_next_level
      ),
      global_coarse_domain=self.global_coarse_domain,
      domain_dof_schedule=self.domain_dof_schedule,
      target_node_schedule=self.target_node_schedule,
    )
    self._hierarchy = hierarchy
    self._hierarchy_build_count += 1
    return hierarchy

  def _level_weights_for_hierarchy(
    self, hierarchy: Hierarchy,
  ) -> tuple[float, ...] | None:
    """Retain the configured coarsest weight after an early cutoff."""
    if self.level_weights is None:
      return None
    level_count = hierarchy.number_of_levels
    if len(self.level_weights) == level_count:
      return self.level_weights
    if len(self.level_weights) > level_count:
      if level_count == 1:
        return self.level_weights[:1]
      return (
        self.level_weights[:level_count - 1] +
        self.level_weights[-1:]
      )
    raise ValueError(
      "level_weights must contain at least one weight per constructed "
      f"hierarchy level ({level_count})"
    )

  def reset(self) -> None:
    self._hierarchy = None
    self._numeric = None
    self._preconditioner = None
    self._cuda_runtime = None
    self._solution = None
    self._hierarchy_build_count = 0
    self._cuda_runtime_build_count = 0
    self._numeric_rebuild_age = 0
    self._preconditioner_dynamic_block_count = 0
    self._preconditioner_dynamic_edge_active = False
    self._statistics = SolverStatistics(solve_mode=self.solve_mode)

  def _ensure_hierarchy(self, view: BlockSparseMatrixView) -> Hierarchy:
    signature = view.structure_signature()
    if self._hierarchy is None or self._hierarchy.static_signature != signature:
      return self.build_hierarchy(view)
    return self._hierarchy

  def _run_reduced_then_fine(
    self,
    numeric: NumericHierarchy,
    hierarchy: Hierarchy,
    rhs: np.ndarray,
    initial_guess: np.ndarray | None,
    tolerance: float,
    max_iterations: int,
    preconditioner: MASPreconditioner,
  ) -> tuple[PCGResult, int]:
    coarse_index = len(hierarchy.levels) - 1
    coarse_rhs = hierarchy.restrict_fine_to_level(rhs, coarse_index)
    coarse_level = hierarchy.levels[coarse_index]
    coarse_numeric = numeric.levels[coarse_index]
    coarse_result = pcg(
      lambda vector: coarse_numeric.matvec(vector, coarse_level),
      coarse_rhs,
      tolerance=min(tolerance, 1e-10),
      max_iterations=max_iterations,
    )
    coarse_guess = hierarchy.prolong_level_to_fine(coarse_result.solution, coarse_index)
    if initial_guess is not None:
      coarse_guess += initial_guess
    fine_result = pcg(
      lambda vector: numeric.levels[0].matvec(vector, hierarchy.levels[0]),
      rhs,
      preconditioner.apply,
      coarse_guess,
      tolerance,
      min(max_iterations, self.fine_correction_iterations),
    )
    return fine_result, coarse_result.iterations

  def solve(
    self,
    matrix_view: BlockSparseMatrixView,
    right_hand_side,
    initial_guess=None,
    tolerance: float = 1e-3,
    max_iterations: int = 20_000,
  ) -> int:
    if not isinstance(matrix_view, BlockSparseMatrixView):
      raise TypeError("matrix_view must be a BlockSparseMatrixView")
    total_started = perf_counter()
    hierarchy = self._ensure_hierarchy(matrix_view)
    rhs_source = right_hand_side.value if hasattr(right_hand_side, "value") else right_hand_side
    guess_source = initial_guess.value if hasattr(initial_guess, "value") else initial_guess
    if is_pycuda_array(rhs_source) and self.inverse_backend_name != "cpu_reference":
      try:
        return self._solve_cuda(
          matrix_view, hierarchy, rhs_source, guess_source,
          tolerance, max_iterations, total_started,
        )
      except RuntimeError:
        if not self.allow_cpu_fallback:
          raise
    rhs = to_host(rhs_source, np.float64).reshape(-1)
    if rhs.size != matrix_view.rows:
      raise ValueError("right_hand_side size must match matrix rows")
    guess = None
    if initial_guess is not None:
      guess = to_host(guess_source, np.float64).reshape(-1)
      if guess.size != rhs.size:
        raise ValueError("initial_guess size must match matrix rows")

    numeric = assemble_numeric_hierarchy(matrix_view, hierarchy)
    backend = make_inverse_backend(
      self.inverse_backend_name,
      threads_per_block=self.cuda_threads_per_block,
      allow_cpu_fallback=self.allow_cpu_fallback,
    )
    preconditioner = MASPreconditioner(
      hierarchy, numeric, backend, self.coarsest_level_weight,
      self._level_weights_for_hierarchy(hierarchy),
    )
    self._numeric, self._preconditioner = numeric, preconditioner
    spmv_seconds = 0.0

    def fine_matvec(vector):
      nonlocal spmv_seconds
      started = perf_counter()
      result = numeric.levels[0].matvec(vector, hierarchy.levels[0])
      spmv_seconds += perf_counter() - started
      return result

    coarse_iterations = 0
    if self.solve_mode == "mas_pcg":
      result = pcg(fine_matvec, rhs, preconditioner.apply, guess, tolerance, max_iterations)
    else:
      result, coarse_iterations = self._run_reduced_then_fine(
        numeric, hierarchy, rhs, guess, tolerance, max_iterations, preconditioner
      )
    self._solution = result.solution.copy()
    ratios = [
      hierarchy.levels[index + 1].number_of_nodes / hierarchy.levels[index].number_of_nodes
      for index in range(len(hierarchy.levels) - 1)
    ]
    inverse_results = preconditioner.build.inverse_results
    self._statistics = SolverStatistics(
      solve_mode=self.solve_mode,
      execution_backend="cpu",
      converged=result.converged,
      iterations=result.iterations,
      coarse_iterations=coarse_iterations,
      final_residual=result.final_residual,
      relative_residual=result.relative_residual,
      breakdown=result.breakdown,
      hierarchy_build_count=self._hierarchy_build_count,
      hierarchy_build_seconds=hierarchy.build_seconds,
      metis_seconds_per_level=[level.metis_seconds for level in hierarchy.levels],
      ordering_backends=[level.ordering_backend for level in hierarchy.levels],
      nodes_per_level=hierarchy.nodes_per_level,
      domains_per_level=hierarchy.domains_per_level,
      coarsening_ratios=ratios,
      domain_scalar_sizes=[level.domain_scalar_sizes.tolist() for level in hierarchy.levels],
      numeric_coarse_assembly_seconds=numeric.coarse_assembly_seconds,
      local_matrix_assembly_seconds=numeric.local_assembly_seconds,
      local_inverse_seconds=sum(item.seconds for item in inverse_results),
      inverse_backends=[item.backend for item in inverse_results],
      inverse_fallback_reasons=[item.fallback_reason for item in inverse_results if item.fallback_reason],
      mas_application_seconds=preconditioner.application_seconds,
      mas_applications=preconditioner.application_count,
      fine_spmv_seconds=spmv_seconds,
      pcg_seconds=result.seconds,
      total_solve_seconds=perf_counter() - total_started,
      gpu_memory_bytes=preconditioner.build.device_bytes,
      configured_max_domain_dofs=self._configured_max_domain_dofs,
      configured_max_levels=self._configured_max_levels,
      configured_target_nodes_per_partition=(
        self._configured_target_nodes_per_partition
      ),
      configured_minimum_domains_for_next_level=(
        self.minimum_domains_for_next_level
      ),
    )
    return result.iterations

  def _solve_cuda(
    self,
    matrix_view: BlockSparseMatrixView,
    hierarchy: Hierarchy,
    rhs,
    initial_guess,
    tolerance: float,
    max_iterations: int,
    total_started: float,
  ) -> int:
    algorithm = (
      "gauss_jordan"
      if self.inverse_backend_name == "cooperative_gauss_jordan"
      else "spd"
    )
    level_weights = self._level_weights_for_hierarchy(hierarchy)
    reused_runtime = (
      self._cuda_runtime is not None
      and self._cuda_runtime.hierarchy is hierarchy
      and self._cuda_runtime.inverse_algorithm == algorithm
      and self._cuda_runtime.threads_per_block == self.cuda_threads_per_block
      and self._cuda_runtime.fixed_inverse_bucket_size
      == self.cuda_fixed_inverse_bucket_size
      and self._cuda_runtime.configured_level_weights
      == level_weights
      and self._cuda_runtime.duplicate_level_weight
      == self.duplicate_level_weight
      and self._cuda_runtime.adaptive_fine_level_weight
      == self.adaptive_fine_level_weight
      and self._cuda_runtime.dense_collision_fine_level_weight
      == self.dense_collision_fine_level_weight
      and self._cuda_runtime.dense_collision_block_ratio
      == self.dense_collision_block_ratio
      and self._cuda_runtime.enable_inverse_fallback
      == self.cuda_enable_inverse_fallback
      and self._cuda_runtime.base_mixed_spmv == self.mixed_spmv
      and self._cuda_runtime.dynamic_edge_block_ratio
      == self.dynamic_edge_block_ratio
    )
    current_dynamic_block_count = int(to_host(
      matrix_view.dynamic_category_counts, np.uint64
    ).sum(dtype=np.uint64))
    numeric_rebuild_reason = "initial-runtime"
    if reused_runtime:
      runtime = self._cuda_runtime
      # A preconditioner assembled from an earlier SPD system remains a
      # valid fixed SPD preconditioner for the current PCG solve.  Force
      # an update when contact turns on/off or an optional collision
      # domain bank changes activation, then permit a separately bounded
      # lag while contact remains active.  The true current Hessian is
      # still used by every SpMV and the stopping test.
      dynamic_connectivity_transition = (
        (current_dynamic_block_count > 0) !=
        (self._preconditioner_dynamic_block_count > 0)
      )
      desired_dynamic_edge_active = bool(
        self.dynamic_edge_domains and
        current_dynamic_block_count > 0 and
        current_dynamic_block_count /
        max(1, hierarchy.levels[0].number_of_nodes) >=
        self.dynamic_edge_block_ratio
      )
      dynamic_edge_activation_transition = (
        desired_dynamic_edge_active !=
        self._preconditioner_dynamic_edge_active
      )
      active_interval = (
        self.dynamic_numeric_rebuild_interval
        if current_dynamic_block_count > 0 else
        self.numeric_rebuild_interval
      )
      interval_expired = (
        self._numeric_rebuild_age + 1 >=
        active_interval
      )
      rebuild_preconditioner = (
        dynamic_connectivity_transition or
        dynamic_edge_activation_transition or interval_expired
      )
      numeric_rebuild_reason = (
        "dynamic-connectivity-transition"
        if dynamic_connectivity_transition else
        "dynamic-edge-activation-transition"
        if dynamic_edge_activation_transition else
        "interval" if interval_expired else "lagged-numeric-reuse"
      )
      runtime.update_numeric(
        matrix_view,
        rebuild_preconditioner=rebuild_preconditioner,
      )
      self._numeric_rebuild_age = (
        0 if rebuild_preconditioner else
        self._numeric_rebuild_age + 1
      )
      if rebuild_preconditioner:
        self._preconditioner_dynamic_block_count = (
          current_dynamic_block_count
        )
        self._preconditioner_dynamic_edge_active = (
          runtime.dynamic_edge_domains_active
        )
    else:
      runtime = DeviceMASRuntime(
        matrix_view,
        hierarchy,
        inverse_algorithm=algorithm,
        threads_per_block=self.cuda_threads_per_block,
        fixed_inverse_bucket_size=self.cuda_fixed_inverse_bucket_size,
        dynamic_edge_domains=self.dynamic_edge_domains,
        dynamic_edge_block_ratio=self.dynamic_edge_block_ratio,
        collision_aware_reorder=self.collision_aware_reorder,
        collision_merge_across_types=(
          self.collision_merge_across_types
        ),
        duplicate_level_weight=self.duplicate_level_weight,
        adaptive_fine_level_weight=self.adaptive_fine_level_weight,
        dense_collision_fine_level_weight=(
          self.dense_collision_fine_level_weight
        ),
        dense_collision_block_ratio=self.dense_collision_block_ratio,
        coarsest_level_weight=self.coarsest_level_weight,
        level_weights=level_weights,
        enable_inverse_fallback=self.cuda_enable_inverse_fallback,
        mixed_spmv=self.mixed_spmv,
      )
      self._cuda_runtime = runtime
      self._cuda_runtime_build_count += 1
      self._numeric_rebuild_age = 0
      self._preconditioner_dynamic_block_count = (
        current_dynamic_block_count
      )
      self._preconditioner_dynamic_edge_active = (
        runtime.dynamic_edge_domains_active
      )
    self._numeric = None
    self._preconditioner = None
    coarse_iterations = 0
    solve_tolerance = (
      tolerance * self.mixed_spmv_tolerance_factor
      if self.mixed_spmv else tolerance
    )
    if self.solve_mode == "mas_pcg":
      result = runtime.pcg(
        rhs,
        initial_guess=initial_guess,
        tolerance=solve_tolerance,
        max_iterations=max_iterations,
      )
    else:
      coarse_index = len(hierarchy.levels) - 1
      coarse_rhs = runtime.restrict_fine(rhs, coarse_index)
      coarse = runtime.pcg(
        coarse_rhs,
        level_index=coarse_index,
        use_mas=False,
        tolerance=min(tolerance, 1e-10),
        max_iterations=max_iterations,
      )
      coarse_iterations = coarse.iterations
      fine_guess = runtime.prolong_to_fine(coarse.solution, coarse_index)
      if initial_guess is not None:
        fine_guess = fine_guess + runtime._as_device(initial_guess, np.float64)
      result = runtime.pcg(
        rhs,
        initial_guess=fine_guess,
        tolerance=solve_tolerance,
        max_iterations=min(max_iterations, self.fine_correction_iterations),
      )
    if self.mixed_spmv and result.breakdown is None:
      audit_started = perf_counter()
      true_residual, true_relative = (
        runtime.full_precision_relative_residual(
          result.solution, rhs
        )
      )
      total_iterations = result.iterations
      total_seconds = result.seconds
      # Reliable mixed precision: most systems pass after one
      # bandwidth-saving approximate-operator solve. Only a system
      # whose original FP64 Hessian misses tolerance pays for a tighter
      # restarted solve.
      if (result.converged and true_relative > tolerance
          and total_iterations < max_iterations):
        base_solution = result.solution.copy()
        correction_rhs = runtime._pcg_residual.copy()
        retry = runtime.pcg(
          correction_rhs, initial_guess=None,
          tolerance=tolerance * 0.25,
          max_iterations=max_iterations - total_iterations,
        )
        total_iterations += retry.iterations
        result = retry
        runtime.add_solution_in_place_kernel(
          result.solution, base_solution,
          np.uint32(runtime.fine_dofs), block=(256, 1, 1),
          grid=((runtime.fine_dofs + 255) // 256, 1, 1),
          stream=runtime._pcg_stream,
        )
        true_residual, true_relative = (
          runtime.full_precision_relative_residual(
            result.solution, rhs
          )
        )
      result = PCGResult(
        result.solution, total_iterations, true_residual,
        true_relative,
        bool(result.converged and true_relative <= tolerance),
        total_seconds + perf_counter() - audit_started,
        (
          result.breakdown
          if true_relative <= tolerance else
          "mixed SpMV did not reach the FP64 residual tolerance"
        ),
      )
    # DeviceMASRuntime owns a persistent solution buffer. Keep that buffer
    # in-place between solves; the public ``solution`` accessor already
    # returns a defensive copy to callers. Copying here allocated and freed
    # a full device vector every solve and implicitly synchronized PCG.
    self._solution = result.solution
    ratios = [
      hierarchy.levels[index + 1].number_of_nodes / hierarchy.levels[index].number_of_nodes
      for index in range(len(hierarchy.levels) - 1)
    ]
    self._statistics = SolverStatistics(
      solve_mode=self.solve_mode,
      execution_backend="cuda",
      converged=result.converged,
      iterations=result.iterations,
      coarse_iterations=coarse_iterations,
      final_residual=result.final_residual,
      relative_residual=result.relative_residual,
      breakdown=result.breakdown,
      hierarchy_build_count=self._hierarchy_build_count,
      hierarchy_build_seconds=hierarchy.build_seconds,
      metis_seconds_per_level=[level.metis_seconds for level in hierarchy.levels],
      ordering_backends=[level.ordering_backend for level in hierarchy.levels],
      nodes_per_level=hierarchy.nodes_per_level,
      domains_per_level=hierarchy.domains_per_level,
      coarsening_ratios=ratios,
      domain_scalar_sizes=[level.domain_scalar_sizes.tolist() for level in hierarchy.levels],
      numeric_coarse_assembly_seconds=runtime.numeric_assembly_seconds,
      numeric_update_wall_seconds=runtime.numeric_update_wall_seconds,
      local_matrix_assembly_seconds=runtime.local_assembly_seconds,
      local_inverse_seconds=runtime.inverse_seconds,
      inverse_backends=[f"cuda_{algorithm}"] * len(hierarchy.levels),
      mas_application_seconds=runtime.mas_seconds,
      mas_applications=runtime.mas_applications,
      fine_spmv_seconds=runtime.spmv_seconds,
      pcg_seconds=result.seconds,
      total_solve_seconds=perf_counter() - total_started,
      gpu_memory_bytes=runtime.device_bytes,
      cuda_runtime_build_count=self._cuda_runtime_build_count,
      numeric_rebuild_count=runtime.numeric_rebuild_count,
      numeric_state_reused=reused_runtime,
      numeric_rebuild_reason=numeric_rebuild_reason,
      dynamic_block_count=runtime.dynamic_block_count,
      dynamic_upload_reallocations=runtime.upload_reallocations,
      upload_capacities=runtime.upload_capacities,
      spmv_kernel_compile_count=runtime.spmv_kernel_compile_count,
      spmv_kernel_cache_hits=runtime.spmv_kernel_cache_hits,
      spmv_kernel_compile_seconds=runtime.spmv_kernel_compile_seconds,
      static_assembly_kernel_compile_seconds=(
        runtime.static_assembly_kernel_compile_seconds
      ),
      dynamic_assembly_kernel_compile_seconds=(
        runtime.dynamic_assembly_kernel_compile_seconds
      ),
      jit_library_compile_count=runtime.jit_kernel_compile_count,
      jit_library_cache_hits=runtime.jit_kernel_cache_hits,
      jit_library_compile_seconds=runtime.jit_kernel_compile_seconds,
      jit_library_paths=sorted(runtime.jit_library_paths),
      spmv_specializations=sorted(runtime.spmv_kernels),
      pcg_graph_build_count=runtime.pcg_graph_build_count,
      pcg_graph_cache_hits=runtime.pcg_graph_cache_hits,
      pcg_graph_build_seconds=runtime.pcg_graph_build_seconds,
      configured_max_domain_dofs=self._configured_max_domain_dofs,
      configured_max_levels=self._configured_max_levels,
      configured_target_nodes_per_partition=(
        self._configured_target_nodes_per_partition
      ),
      configured_minimum_domains_for_next_level=(
        self.minimum_domains_for_next_level
      ),
      configured_fixed_inverse_bucket_size=(
        self.cuda_fixed_inverse_bucket_size or 0
      ),
      configured_duplicate_level_weight=self.duplicate_level_weight,
      configured_adaptive_fine_level_weight=(
        self.adaptive_fine_level_weight
      ),
      configured_dense_collision_fine_level_weight=(
        self.dense_collision_fine_level_weight
      ),
      configured_dense_collision_block_ratio=(
        self.dense_collision_block_ratio
      ),
      configured_dynamic_edge_block_ratio=(
        self.dynamic_edge_block_ratio
      ),
      configured_dynamic_numeric_rebuild_interval=(
        self.dynamic_numeric_rebuild_interval
      ),
      active_fine_level_weight=runtime.active_fine_level_weight,
      dynamic_edge_domains_active=runtime.dynamic_edge_domains_active,
    )
    return result.iterations

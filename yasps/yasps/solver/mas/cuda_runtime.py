"""Reusable CUDA numerical assembly, MAS application, SpMV, and PCG.

The hierarchy and every scalar/domain map are static.  Per solve, current
static values and arbitrary dynamic collision blocks are walked through those
maps once, accumulated directly into a fixed local-matrix arena, inverted once,
and reused for all PCG iterations.  This mirrors StiffGIPC's useful numerical
lifecycle while retaining YASPS's heterogeneous block dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .hierarchy import Hierarchy, HierarchyLevel
from .cuda_graph import CapturedGraph
from .jit import compile_cuda_library, spmv_warps_for_shape
from .local_inverse import LocalInverseError, bucket_size
from .matrix_view import BlockSparseMatrixView, to_host
from .pcg import PCGResult


_CUDA_KERNEL_CACHE: dict[int, dict[str, object]] = {}
_CUDA_MEMORY_POOL_CACHE: dict[int, object] = {}
_CUDA_SPMV_KERNEL_CACHE: dict[tuple[int, int, int, bool], dict[str, object]] = {}
_CUDA_STATIC_ASSEMBLY_KERNEL_CACHE: dict[tuple[object, ...], dict[str, object]] = {}
_CUDA_DYNAMIC_ASSEMBLY_KERNEL_CACHE: dict[tuple[object, ...], dict[str, object]] = {}
_CUDA_EXACT_INVERSE_CACHE: dict[tuple[int, int, int], dict[str, object]] = {}
_CUDA_PRECONDITIONER_SPECIALIZATION_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def release_held_device_memory() -> None:
  """Return unused cached allocations to CUDA between independent workloads."""
  for pool in _CUDA_MEMORY_POOL_CACHE.values():
    pool.free_held()


def is_pycuda_array(value) -> bool:
  return hasattr(value, "gpudata") and hasattr(value, "dtype") and hasattr(value, "size")


@dataclass
class DeviceInputPart:
  counts: np.ndarray
  starts: np.ndarray
  shapes: np.ndarray
  position_offsets: np.ndarray
  device_counts: object
  device_starts: object
  device_shapes: object
  device_position_offsets: object
  positions: object
  values: object

  @property
  def block_count(self) -> int:
    return int(self.position_offsets[-1]) if self.position_offsets.size else 0


@dataclass
class UploadSlot:
  array: object
  capacity: int
  host_value: np.ndarray | None = None


@dataclass
class InverseBucket:
  padded_size: int
  active_size: int
  matrix_start: int
  domains: np.ndarray
  batch_sizes: object
  status: object


class DeviceMASRuntime:
  """Static CUDA maps plus numerical state reused across many solves."""

  def __init__(
    self,
    view: BlockSparseMatrixView,
    hierarchy: Hierarchy,
    *,
    inverse_algorithm: str = "spd",
    threads_per_block: int = 96,
    pivot_tolerance: float = 1e-12,
    upload_growth_factor: float = 1.5,
    fixed_inverse_bucket_size: int | None = None,
    dynamic_edge_domains: bool = False,
    dynamic_edge_block_ratio: float = 0.0,
    collision_aware_reorder: bool = False,
    collision_merge_across_types: bool = False,
    duplicate_level_weight: float = 0.5,
    adaptive_fine_level_weight: bool = True,
    dense_collision_fine_level_weight: float = 3.0,
    dense_collision_block_ratio: float = 0.1,
    coarsest_level_weight: float = 2.5,
    level_weights: tuple[float, ...] | None = None,
    enable_inverse_fallback: bool = True,
    mixed_spmv: bool = False,
  ):
    if inverse_algorithm not in ("spd", "gauss_jordan"):
      raise ValueError("inverse_algorithm must be 'spd' or 'gauss_jordan'")
    if threads_per_block not in (32, 64, 96, 128):
      raise ValueError("threads_per_block must be 32, 64, 96, or 128")
    if upload_growth_factor <= 1.0:
      raise ValueError("upload_growth_factor must be greater than one")
    if fixed_inverse_bucket_size is not None and fixed_inverse_bucket_size <= 0:
      raise ValueError("fixed_inverse_bucket_size must be positive")
    try:
      import pycuda.autoinit  # noqa: F401
      import pycuda.driver as cuda
      import pycuda.gpuarray as gpuarray
      from pycuda.compiler import SourceModule
      from pycuda.tools import DeviceMemoryPool
    except Exception as error:  # pragma: no cover - device dependent
      raise RuntimeError(f"CUDA initialization failed: {error}") from error

    self.cuda, self.gpuarray = cuda, gpuarray
    self.view, self.hierarchy = view, hierarchy
    self.inverse_algorithm = inverse_algorithm
    self.threads_per_block = threads_per_block
    self.pivot_tolerance = float(pivot_tolerance)
    self.upload_growth_factor = float(upload_growth_factor)
    self.fixed_inverse_bucket_size = (
      None if fixed_inverse_bucket_size is None
      else int(fixed_inverse_bucket_size)
    )
    self.dynamic_edge_domains = bool(dynamic_edge_domains)
    if (not np.isfinite(dynamic_edge_block_ratio)
        or dynamic_edge_block_ratio < 0.0):
      raise ValueError(
        "dynamic_edge_block_ratio must be finite and non-negative"
      )
    self.dynamic_edge_block_ratio = float(dynamic_edge_block_ratio)
    self.dynamic_edge_domains_active = False
    self.collision_aware_reorder = bool(collision_aware_reorder)
    self.collision_merge_across_types = bool(
      collision_merge_across_types
    )
    if (not np.isfinite(duplicate_level_weight)
        or not 0.0 <= duplicate_level_weight <= 1.0):
      raise ValueError("duplicate_level_weight must be between zero and one")
    self.duplicate_level_weight = float(duplicate_level_weight)
    self.adaptive_fine_level_weight = bool(adaptive_fine_level_weight)
    if (not np.isfinite(dense_collision_fine_level_weight)
        or dense_collision_fine_level_weight <= 0):
      raise ValueError("dense collision fine-level weight must be positive")
    if (not np.isfinite(dense_collision_block_ratio)
        or dense_collision_block_ratio <= 0):
      raise ValueError("dense collision block ratio must be positive")
    self.dense_collision_fine_level_weight = float(
      dense_collision_fine_level_weight
    )
    self.dense_collision_block_ratio = float(dense_collision_block_ratio)
    if not np.isfinite(coarsest_level_weight) or coarsest_level_weight <= 0:
      raise ValueError("coarsest_level_weight must be finite and positive")
    self.coarsest_level_weight = float(coarsest_level_weight)
    if level_weights is not None and (
      not level_weights
      or any(not np.isfinite(weight) or weight <= 0
         for weight in level_weights)
    ):
      raise ValueError("level_weights must contain finite positive values")
    self.configured_level_weights = (
      None if level_weights is None
      else tuple(float(weight) for weight in level_weights)
    )
    self.enable_inverse_fallback = bool(enable_inverse_fallback)
    # Keep this distinct from DeviceJacobiMASRuntime's older experimental
    # ``mixed_spmv`` state; that subclass owns separate value buffers and
    # generated modules.
    self.base_mixed_spmv = bool(mixed_spmv)
    self.spmv_value_buffers: list[object] = []
    self._uploads: dict[str, UploadSlot] = {}
    self.upload_reallocations = 0
    self.numeric_rebuild_count = 0
    self.dynamic_block_count = 0
    self.dynamic_group_count = 0
    self.metadata: list[DeviceInputPart] = []
    self.spmv_kernels: dict[tuple[int, int], object] = {}
    self.spmv_pair_kernels: dict[tuple[int, int], object] = {}
    self.spmv_fine_kernels: dict[tuple[int, int], object] = {}
    self.spmv_fine_pair_kernels: dict[tuple[int, int], object] = {}
    self.spmv_full_kernels: dict[tuple[int, int], object] = {}
    self.spmv_full_pair_kernels: dict[tuple[int, int], object] = {}
    self.spmv_full_fine_kernels: dict[tuple[int, int], object] = {}
    self.spmv_full_fine_pair_kernels: dict[tuple[int, int], object] = {}
    self.spmv_auxiliary_shapes: tuple[tuple[int, int], ...] = ()
    self.spmv_auxiliary_capacity = 0
    self.spmv_auxiliary_descriptor_count = 0
    self.spmv_auxiliary_total_count = 0
    self.spmv_auxiliary_launch_capacity = 0
    self.spmv_fused_auxiliary_shapes: tuple[tuple[int, int], ...] = ()
    self.spmv_primary_shape: tuple[int, int] | None = None
    self._spmv_auxiliary_records_signature: tuple[
      tuple[int, ...], ...
    ] | None = None
    self.spmv_kernel_compile_count = 0
    self.spmv_kernel_cache_hits = 0
    self.spmv_kernel_compile_seconds = 0.0
    self.static_assembly_kernel_compile_seconds = 0.0
    self.dynamic_assembly_kernel_compile_seconds = 0.0
    self.jit_kernel_compile_count = 0
    self.jit_kernel_cache_hits = 0
    self.jit_kernel_compile_seconds = 0.0
    self.jit_library_paths: set[str] = set()
    self.static_assembly_kernel = None
    self.dynamic_assembly_kernel = None
    self.pcg_graph_build_count = 0
    self.pcg_graph_cache_hits = 0
    self.pcg_graph_build_seconds = 0.0
    self._pcg_graphs: dict[tuple[object, ...], CapturedGraph] = {}
    self._numeric_rebuild_graphs: dict[tuple[object, ...], CapturedGraph] = {}
    self._mixed_spmv_conversion_graphs: dict[
      tuple[object, ...], CapturedGraph
    ] = {}
    self.dynamic_block_launch_capacity = 0

    context_key = int(cuda.Context.get_current().handle)
    self.context_key = context_key
    self.SourceModule = SourceModule
    memory_pool = _CUDA_MEMORY_POOL_CACHE.get(context_key)
    if memory_pool is None:
      memory_pool = DeviceMemoryPool()
      _CUDA_MEMORY_POOL_CACHE[context_key] = memory_pool
    self.allocator = memory_pool.allocate
    self._load_kernels(context_key, SourceModule)
    self._pcg_stream = cuda.Stream()
    self.assembly_threads = 512
    self.dynamic_assembly_threads = 64

    self.numeric_assembly_seconds = 0.0
    self.local_assembly_seconds = 0.0
    self.inverse_seconds = 0.0
    self.spmv_seconds = 0.0
    self.mas_seconds = 0.0
    self.mas_applications = 0
    self.device_bytes = 0
    self.static_setup_seconds = 0.0
    self._build_static_state(view)
    # The domain-size buckets are immutable. Capture their specialized
    # inverse launches once so each numerical rebuild issues one graph
    # launch, with no per-solve Python traversal over bucket sizes.  Each
    # bucket writes a disjoint matrix/status span, so fork independent
    # fixed-size kernels across streams and join them inside the graph.
    self._build_numeric_inverse_graph()
    self.update_numeric(view)

  def _load_kernels(self, context_key: int, SourceModule) -> None:
    cached = _CUDA_KERNEL_CACHE.get(context_key)
    if cached is None:
      root = Path(__file__).resolve().parent / "cuda"
      try:
        assembly = SourceModule(
          (root / "local_matrix_assembly.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
        inverse = SourceModule(
          (root / "local_inverse.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
        preconditioner = SourceModule(
          (root / "preconditioner_apply.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
        restriction = SourceModule(
          (root / "restriction.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
        prolongation = SourceModule(
          (root / "prolongation.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
        pcg = SourceModule(
          (root / "pcg.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
        reorder = SourceModule(
          (root / "realtime_reorder.cu").read_text(encoding="utf8"),
          options=["-std=c++17"], no_extern_c=True,
        )
      except Exception as error:  # pragma: no cover - device dependent
        raise RuntimeError(f"CUDA kernel compilation failed: {error}") from error
      cached = {
        "modules": (
          assembly, inverse, preconditioner, restriction, prolongation,
          pcg, reorder
        ),
        "reset_runtime_representatives": reorder.get_function(
          "yasps_mas_reset_runtime_representatives"
        ),
        "build_collision_masks": reorder.get_function(
          "yasps_mas_build_collision_masks"
        ),
        "close_collision_components": reorder.get_function(
          "yasps_mas_close_collision_components"
        ),
        "propagate_runtime_components": reorder.get_function(
          "yasps_mas_propagate_runtime_components"
        ),
        "build_runtime_scalar_maps": reorder.get_function(
          "yasps_mas_build_runtime_scalar_maps"
        ),
        "build_runtime_transfer_maps": reorder.get_function(
          "yasps_mas_build_runtime_transfer_maps"
        ),
        "initialize_domains": assembly.get_function("yasps_mas_initialize_padded_domains"),
        "symmetrize_domains": assembly.get_function(
          "yasps_mas_symmetrize_padded_domains"
        ),
        "fused_assemble": assembly.get_function("yasps_mas_fused_assemble_domains"),
        "fused_assemble_all": assembly.get_function(
          "yasps_mas_fused_assemble_all_categories"
        ),
        "propagate_fine_domains": assembly.get_function(
          "yasps_mas_propagate_fine_domains"
        ),
        "propagate_adjacent_domains": assembly.get_function(
          "yasps_mas_propagate_adjacent_domains"
        ),
        "assemble_dynamic_edges": assembly.get_function(
          "yasps_mas_assemble_dynamic_edge_domains"
        ),
        "assemble_dynamic_groups": assembly.get_function(
          "yasps_mas_assemble_dynamic_group_domains"
        ),
        "complete_dynamic_groups": assembly.get_function(
          "yasps_mas_complete_dynamic_group_domains"
        ),
        "regularize_dynamic_groups": assembly.get_function(
          "yasps_mas_regularize_dynamic_group_domains"
        ),
        "mapped_spmv": assembly.get_function("yasps_mas_mapped_block_spmv"),
        "inverse_spd": inverse.get_function("yasps_mas_inverse_spd"),
        "inverse_spd_mixed": inverse.get_function(
          "yasps_mas_inverse_spd_mixed"
        ),
        "inverse_spd_mixed_ragged": inverse.get_function(
          "yasps_mas_inverse_spd_mixed_ragged"
        ),
        "inverse_gauss_jordan": inverse.get_function("yasps_mas_inverse_gauss_jordan"),
        "inverse_gauss_jordan_mixed": inverse.get_function(
          "yasps_mas_inverse_gauss_jordan_mixed"
        ),
        "inverse_apply": preconditioner.get_function("yasps_mas_dense_inverse_apply"),
        "restrict_all": preconditioner.get_function("yasps_mas_restrict_all_levels"),
        "collect_all": preconditioner.get_function("yasps_mas_collect_all_levels"),
        "cast_inverse_mixed": preconditioner.get_function(
          "yasps_mas_cast_inverse_to_float"
        ),
        "restrict_adjacent_nodes_mixed": preconditioner.get_function(
          "yasps_mas_restrict_adjacent_nodes_mixed"
        ),
        "restrict_warp_nodes_mixed": preconditioner.get_function(
          "yasps_mas_restrict_warp_nodes_mixed"
        ),
        "inverse_apply_mixed": preconditioner.get_function(
          "yasps_mas_dense_inverse_apply_mixed"
        ),
        "inverse_apply_mixed_cooperative": preconditioner.get_function(
          "yasps_mas_dense_inverse_apply_mixed_cooperative"
        ),
        "collect_nodes_mixed": preconditioner.get_function(
          "yasps_mas_collect_nodes_mixed"
        ),
        "apply_dynamic_edges": preconditioner.get_function(
          "yasps_mas_apply_dynamic_edge_domains"
        ),
        "apply_dynamic_groups": preconditioner.get_function(
          "yasps_mas_apply_dynamic_group_domains"
        ),
        "restriction": restriction.get_function("yasps_mas_restrict"),
        "prolongation": prolongation.get_function("yasps_mas_prolongate_add"),
        "dot_single": pcg.get_function("yasps_mas_dot_single"),
        "dot_two": pcg.get_function("yasps_mas_dot_two"),
        "dot_single_partials": pcg.get_function(
          "yasps_mas_dot_single_partials"
        ),
        "dot_two_partials": pcg.get_function(
          "yasps_mas_dot_two_partials"
        ),
        "residual_from_product": pcg.get_function(
          "yasps_mas_residual_from_product"
        ),
        "add_solution_in_place": pcg.get_function(
          "yasps_mas_add_solution_in_place"
        ),
        "prepare_iteration": pcg.get_function("yasps_mas_prepare_iteration"),
        "prepare_iteration_partials": pcg.get_function(
          "yasps_mas_prepare_iteration_partials"
        ),
        "update_solution_residual": pcg.get_function(
          "yasps_mas_update_solution_residual"
        ),
        "finish_iteration": pcg.get_function("yasps_mas_finish_iteration"),
        "finish_iteration_partials": pcg.get_function(
          "yasps_mas_finish_iteration_partials"
        ),
        "update_direction": pcg.get_function("yasps_mas_update_direction"),
        "initialize_recurrence": pcg.get_function(
          "yasps_mas_initialize_recurrence"
        ),
        "finish_iteration_unchecked": pcg.get_function(
          "yasps_mas_finish_iteration_unchecked"
        ),
        "update_pcg_loop": pcg.get_function(
          "yasps_mas_update_pcg_loop"
        ),
      }
      _CUDA_KERNEL_CACHE[context_key] = cached
    self.initialize_domains = cached["initialize_domains"]
    self.symmetrize_domains = cached["symmetrize_domains"]
    self.reset_runtime_representatives_kernel = cached[
      "reset_runtime_representatives"
    ]
    self.build_collision_masks_kernel = cached["build_collision_masks"]
    self.close_collision_components_kernel = cached[
      "close_collision_components"
    ]
    self.propagate_runtime_components_kernel = cached[
      "propagate_runtime_components"
    ]
    self.build_runtime_scalar_maps_kernel = cached[
      "build_runtime_scalar_maps"
    ]
    self.build_runtime_transfer_maps_kernel = cached[
      "build_runtime_transfer_maps"
    ]
    self.fused_assemble_kernel = cached["fused_assemble"]
    self.fused_assemble_all_kernel = cached["fused_assemble_all"]
    self.propagate_fine_domains_kernel = cached["propagate_fine_domains"]
    self.propagate_adjacent_domains_kernel = cached[
      "propagate_adjacent_domains"
    ]
    self.assemble_dynamic_edges_kernel = cached["assemble_dynamic_edges"]
    self.assemble_dynamic_groups_kernel = cached[
      "assemble_dynamic_groups"
    ]
    self.complete_dynamic_groups_kernel = cached[
      "complete_dynamic_groups"
    ]
    self.regularize_dynamic_groups_kernel = cached[
      "regularize_dynamic_groups"
    ]
    self.mapped_spmv_kernel = cached["mapped_spmv"]
    self.inverse_kernel = cached[
      "inverse_spd_mixed"
      if self.inverse_algorithm == "spd"
      else "inverse_gauss_jordan_mixed"
    ]
    self.inverse_spd_mixed_ragged_kernel = cached[
      "inverse_spd_mixed_ragged"
    ]
    self.inverse_spd_mixed_fixed_kernels: dict[int, object] = {}
    self.inverse_spd_mixed_fallback_kernels: dict[int, object] = {}
    self.inverse_gj_packed_kernels: dict[int, object] = {}
    self.inverse_gauss_jordan_mixed_kernel = cached[
      "inverse_gauss_jordan_mixed"
    ]
    self.inverse_apply_kernel = cached["inverse_apply"]
    self.restrict_all_kernel = cached["restrict_all"]
    self.collect_all_kernel = cached["collect_all"]
    self.cast_inverse_mixed_kernel = cached["cast_inverse_mixed"]
    self.restrict_adjacent_nodes_mixed_kernel = cached[
      "restrict_adjacent_nodes_mixed"
    ]
    self.restrict_warp_nodes_mixed_kernel = cached["restrict_warp_nodes_mixed"]
    self.inverse_apply_mixed_kernel = cached["inverse_apply_mixed"]
    self.inverse_apply_mixed_cooperative_kernel = cached[
      "inverse_apply_mixed_cooperative"
    ]
    self.collect_nodes_mixed_kernel = cached["collect_nodes_mixed"]
    self.apply_dynamic_edges_kernel = cached["apply_dynamic_edges"]
    self.apply_dynamic_groups_kernel = cached["apply_dynamic_groups"]
    self.restriction_kernel = cached["restriction"]
    self.prolongation_kernel = cached["prolongation"]
    self.dot_single_kernel = cached["dot_single"]
    self.dot_two_kernel = cached["dot_two"]
    self.dot_single_partials_kernel = cached["dot_single_partials"]
    self.dot_two_partials_kernel = cached["dot_two_partials"]
    self.residual_from_product_kernel = cached["residual_from_product"]
    self.add_solution_in_place_kernel = cached[
      "add_solution_in_place"
    ]
    self.prepare_iteration_kernel = cached["prepare_iteration"]
    self.prepare_iteration_partials_kernel = cached[
      "prepare_iteration_partials"
    ]
    self.update_solution_residual_kernel = cached["update_solution_residual"]
    self.finish_iteration_kernel = cached["finish_iteration"]
    self.finish_iteration_partials_kernel = cached[
      "finish_iteration_partials"
    ]
    self.update_direction_kernel = cached["update_direction"]
    self.initialize_recurrence_kernel = cached["initialize_recurrence"]
    self.finish_iteration_unchecked_kernel = cached["finish_iteration_unchecked"]
    self.update_pcg_loop_kernel = cached["update_pcg_loop"]

  def _record_jit_library(self, library, *, reused: bool = False) -> None:
    path = str(library.path)
    if path in self.jit_library_paths:
      return
    self.jit_library_paths.add(path)
    self.jit_kernel_compile_seconds += (
      0.0 if reused else float(library.compile_seconds)
    )
    if reused or library.cache_hit:
      self.jit_kernel_cache_hits += 1
    else:
      self.jit_kernel_compile_count += 1

  def _get_inverse_specialization(self, padded_size: int) -> dict[str, object]:
    padded_size = int(padded_size)
    if padded_size <= 0:
      raise ValueError("inverse specialization size must be positive")
    groups = max(1, self.threads_per_block // padded_size)
    source = (
      Path(__file__).resolve().parent / "cuda" / "local_inverse.cu"
    ).read_text(encoding="utf8")
    names = (
      "yasps_mas_inverse_gj_packed_specialized",
      "yasps_mas_inverse_spd_mixed_fixed_specialized",
      "yasps_mas_inverse_spd_mixed_fallback_specialized",
    )
    library = compile_cuda_library(
      self.cuda,
      source,
      kernel_names=names,
      label=f"inverse_n{padded_size}_g{groups}",
      options=(
        f"-DYASPS_MAS_INVERSE_SIZE={padded_size}",
        f"-DYASPS_MAS_INVERSE_GROUPS={groups}",
      ),
    )
    self._record_jit_library(library)
    return {
      "groups": groups,
      "packed": library.kernel(names[0]),
      "fixed": library.kernel(names[1]),
      "fallback": library.kernel(names[2]),
    }

  def _ensure_inverse_specializations(self, padded_sizes) -> None:
    for padded_size in sorted({int(value) for value in padded_sizes}):
      specialized = self._get_inverse_specialization(padded_size)
      self.inverse_gj_packed_kernels[padded_size] = specialized["packed"]
      self.inverse_spd_mixed_fixed_kernels[padded_size] = specialized[
        "fixed"
      ]
      self.inverse_spd_mixed_fallback_kernels[padded_size] = specialized[
        "fallback"
      ]

  def _get_specialized_spmv_kernel(
    self, rows: int, cols: int, *, full_precision: bool = False,
  ):
    shape = (int(rows), int(cols))
    use_mixed = self.base_mixed_spmv and not full_precision
    kernel_store = (
      self.spmv_full_kernels if full_precision else self.spmv_kernels
    )
    existing = kernel_store.get(shape)
    if existing is not None:
      return existing
    key = (
      *shape, self.context_key, bool(self.view.symmetric_storage),
      self.spmv_auxiliary_shapes, use_mixed,
    )
    cached = _CUDA_SPMV_KERNEL_CACHE.get(key)
    reused_module = cached is not None
    if cached is None:
      started = perf_counter()
      source = (
        Path(__file__).resolve().parent / "cuda" / "block_spmv.cu"
      ).read_text(encoding="utf8")
      if use_mixed:
        # GIPC-style mixed operator storage: keep Krylov vectors,
        # reductions, and accumulation in FP64 while halving the
        # bandwidth-dominant Hessian value traffic.
        source = source.replace(
          "const double* values", "const float* values"
        ).replace(
          "const double* matrix", "const float* matrix"
        ).replace(
          "const double* block", "const float* block"
        ).replace(
          "double* ordered_values", "float* ordered_values"
        ).replace(
          "reinterpret_cast<const double*>",
          "reinterpret_cast<const float*>",
        )
      auxiliary_cases = "\n".join(
        "        case (({0}u << 16) | {1}u): return "
        "yasps_mas_process_one_auxiliary<{0}, {1}>("
        "descriptor, auxiliary_id, x, y);".format(*other_shape)
        for other_shape in self.spmv_auxiliary_shapes
        if other_shape != shape
      )
      source = source.replace(
        "        // YASPS_MAS_AUXILIARY_SHAPE_CASES",
        auxiliary_cases,
      )
      names = (
        "yasps_mas_specialized_mapped_block_spmv",
        "yasps_mas_cooperative_mapped_block_spmv",
        "yasps_mas_specialized_mapped_block_spmv_pair",
        "yasps_mas_specialized_fine_block_spmv",
        "yasps_mas_specialized_fine_block_spmv_pair",
      )
      try:
        module = compile_cuda_library(
          self.cuda,
          source,
          kernel_names=names,
          label=(
            f"spmv_{shape[0]}x{shape[1]}_"
            f"s{int(bool(self.view.symmetric_storage))}_"
            f"m{int(use_mixed)}"
          ),
          options=(
            f"-DYASPS_MAS_BLOCK_ROWS={shape[0]}",
            f"-DYASPS_MAS_BLOCK_COLS={shape[1]}",
            "-DYASPS_MAS_WARPS_PER_BLOCK="
            f"{spmv_warps_for_shape(*shape)}",
            "-DYASPS_MAS_SEGMENTED_REDUCTION=1",
            "-DYASPS_MAS_FUSE_AUXILIARY="
            f"{int(shape == self.spmv_primary_shape)}",
            "-DYASPS_MAS_SYMMETRIC_STORAGE="
            f"{int(bool(self.view.symmetric_storage))}",
            *(
              ("-DYASPS_MAS_MIXED_SPMV_ARITHMETIC=1",)
              if use_mixed else ()
            ),
          ),
        )
      except Exception as error:  # pragma: no cover - device dependent
        raise RuntimeError(
          f"CUDA SpMV specialization {shape[0]}x{shape[1]} failed: {error}"
        ) from error
      cached = {
        "module": module,
        "kernel": module.kernel(names[0]),
        "cooperative_kernel": module.kernel(names[1]),
        "pair_kernel": module.kernel(names[2]),
        "fine_kernel": module.kernel(names[3]),
        "fine_pair_kernel": module.kernel(names[4]),
      }
      _CUDA_SPMV_KERNEL_CACHE[key] = cached
      if module.cache_hit:
        self.spmv_kernel_cache_hits += 1
      else:
        self.spmv_kernel_compile_count += 1
      self.spmv_kernel_compile_seconds += float(
        module.compile_seconds
      )
    else:
      self.spmv_kernel_cache_hits += 1
    self._record_jit_library(
      cached["module"], reused=reused_module
    )
    kernel = cached[
      "cooperative_kernel" if shape[0] * shape[1] >= 64 else "kernel"
    ]
    kernel_store[shape] = kernel
    pair_store = (
      self.spmv_full_pair_kernels
      if full_precision else self.spmv_pair_kernels
    )
    fine_store = (
      self.spmv_full_fine_kernels
      if full_precision else self.spmv_fine_kernels
    )
    fine_pair_store = (
      self.spmv_full_fine_pair_kernels
      if full_precision else self.spmv_fine_pair_kernels
    )
    pair_store[shape] = cached["pair_kernel"]
    fine_store[shape] = cached["fine_kernel"]
    fine_pair_store[shape] = cached["fine_pair_kernel"]
    return kernel

  def _get_exact_inverse_kernel(self, active_size: int, padded_size: int):
    active_size, padded_size = int(active_size), int(padded_size)
    key = (self.context_key, active_size, padded_size)
    cached = _CUDA_EXACT_INVERSE_CACHE.get(key)
    reused_module = cached is not None
    if cached is None:
      groups = max(1, self.threads_per_block // active_size)
      source = (
        Path(__file__).resolve().parent
        / "cuda" / "local_inverse_exact.cu"
      ).read_text(encoding="utf8")
      names = (
        "yasps_mas_inverse_gj_exact_strided",
        "yasps_mas_inverse_spd_exact_fallback",
      )
      module = compile_cuda_library(
        self.cuda,
        source,
        kernel_names=names,
        label=f"inverse_exact_n{active_size}_p{padded_size}",
        options=(
          f"-DYASPS_MAS_ACTIVE_SIZE={active_size}",
          f"-DYASPS_MAS_STORAGE_STRIDE={padded_size}",
          f"-DYASPS_MAS_INVERSE_GROUPS={groups}",
        ),
      )
      cached = {
        "module": module,
        "kernel": module.kernel(names[0]),
        "fallback": module.kernel(names[1]),
        "groups": groups,
      }
      _CUDA_EXACT_INVERSE_CACHE[key] = cached
    self._record_jit_library(
      cached["module"], reused=reused_module
    )
    return cached

  def _ensure_spmv_specializations(self) -> None:
    for metadata in self.metadata:
      for count, shape in zip(metadata.counts, metadata.shapes):
        if int(count):
          self._get_specialized_spmv_kernel(*map(int, shape))
          if self.base_mixed_spmv:
            self._get_specialized_spmv_kernel(
              *map(int, shape), full_precision=True
            )

  def _get_specialized_static_assembly_kernel(self) -> None:
    static = self.metadata[0]
    descriptor = tuple(
      (int(count), int(start), int(rows), int(cols))
      for count, start, (rows, cols) in zip(
        static.counts, static.starts, static.shapes
      ) if int(count)
    )
    key = (
      self.context_key, bool(self.view.symmetric_storage), descriptor,
    )
    cached = _CUDA_STATIC_ASSEMBLY_KERNEL_CACHE.get(key)
    reused_module = cached is not None
    if cached is None:
      started = perf_counter()
      source = (
        Path(__file__).resolve().parent
        / "cuda" / "static_assembly_specialized.cu"
      ).read_text(encoding="utf8")
      dispatch = []
      precomputed_dispatch = []
      position_start = 0
      for count, value_start, rows, cols in (
        (int(count), int(start), int(shape[0]), int(shape[1]))
        for count, start, shape in zip(
          static.counts, static.starts, static.shapes
        )
        if int(count)
      ):
        end = position_start + count
        dispatch.append(
          f"  if (global_id < {end}u) {{\n"
          f"    yasps_mas_assemble_static_block<{rows}u, {cols}u>(\n"
          f"        global_id - {position_start}u, {value_start}ull,\n"
          f"        {position_start}ull, values, positions,\n"
          "        scalar_boundary_to_node, fine_scalar_dofs,\n"
          "        fine_node_domains, fine_node_local_offsets,\n"
          "        level_count, fine_node_count, matrix_offsets,\n"
          "        padded_sizes, matrices, status);\n"
          "    return;\n"
          "  }"
        )
        precomputed_dispatch.append(
          f"  if (global_id < {end}u) {{\n"
          f"    yasps_mas_assemble_precomputed_static_block<"
          f"{rows}u, {cols}u>(\n"
          f"        global_id, global_id - {position_start}u, "
          f"{value_start}ull, values, destination_offsets,\n"
          "        transpose_offsets, destination_strides, "
          "matrices);\n"
          "    return;\n"
          "  }"
        )
        position_start = end
      source = source.replace(
        "// YASPS_MAS_GENERATED_STATIC_CATEGORY_DISPATCH",
        "\n".join(dispatch),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_PRECOMPUTED_STATIC_CATEGORY_DISPATCH",
        "\n".join(precomputed_dispatch),
      )
      names = (
        "yasps_mas_specialized_static_assembly",
        "yasps_mas_precomputed_static_assembly",
      )
      module = compile_cuda_library(
        self.cuda,
        source,
        kernel_names=names,
        label=(
          "static_assembly_"
          f"s{int(bool(self.view.symmetric_storage))}"
        ),
        options=(
          "-DYASPS_MAS_SYMMETRIC_STORAGE="
          f"{int(bool(self.view.symmetric_storage))}",
        ),
      )
      cached = {
        "module": module,
        "kernel": module.kernel(names[0]),
        "precomputed_kernel": module.kernel(names[1]),
      }
      _CUDA_STATIC_ASSEMBLY_KERNEL_CACHE[key] = cached
      self.static_assembly_kernel_compile_seconds += float(
        module.compile_seconds
      )
    self._record_jit_library(
      cached["module"], reused=reused_module
    )
    self.static_assembly_kernel = cached["kernel"]
    self.precomputed_static_assembly_kernel = cached[
      "precomputed_kernel"
    ]

  def _get_specialized_dynamic_assembly_kernel(self) -> None:
    dynamic = self.metadata[1]
    shapes = tuple(
      (int(rows), int(cols)) for rows, cols in dynamic.shapes
    )
    if not shapes:
      self.dynamic_assembly_kernel = None
      return
    key = (self.context_key, bool(self.view.symmetric_storage), shapes)
    cached = _CUDA_DYNAMIC_ASSEMBLY_KERNEL_CACHE.get(key)
    reused_module = cached is not None
    if cached is None:
      started = perf_counter()
      source = (
        Path(__file__).resolve().parent
        / "cuda" / "dynamic_assembly_specialized.cu"
      ).read_text(encoding="utf8")
      dispatch = "\n".join(
        f"    case {category}u:\n"
        f"      yasps_mas_assemble_dynamic_block<{rows}u, {cols}u>(\n"
        "          source_id, value_starts[category],\n"
        "          position_offsets[category], values, positions,\n"
        "          scalar_boundary_to_node, fine_scalar_dofs,\n"
        "          fine_node_domains, fine_node_local_offsets,\n"
        "          level_count, fine_node_count, matrix_offsets,\n"
        "          padded_sizes, matrices, status);\n"
        "      break;"
        for category, (rows, cols) in enumerate(shapes)
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_DYNAMIC_CATEGORY_DISPATCH",
        dispatch,
      )
      names = ("yasps_mas_specialized_dynamic_assembly",)
      module = compile_cuda_library(
        self.cuda,
        source,
        kernel_names=names,
        label=(
          "dynamic_assembly_"
          f"s{int(bool(self.view.symmetric_storage))}"
        ),
        options=(
          "-DYASPS_MAS_SYMMETRIC_STORAGE="
          f"{int(bool(self.view.symmetric_storage))}",
        ),
      )
      cached = {
        "module": module,
        "kernel": module.kernel(names[0]),
      }
      _CUDA_DYNAMIC_ASSEMBLY_KERNEL_CACHE[key] = cached
      self.dynamic_assembly_kernel_compile_seconds += float(
        module.compile_seconds
      )
    self._record_jit_library(
      cached["module"], reused=reused_module
    )
    self.dynamic_assembly_kernel = cached["kernel"]

  def _get_specialized_preconditioner_kernels(
    self, maximum_dimension: int, level_count: int,
    size_pairs: tuple[tuple[int, int], ...],
    level_weights: tuple[float, ...],
    fine_dimensions: tuple[int, ...],
    duplicate_level_weight: float,
    compact_packed_offsets: bool,
  ):
    key = (
      self.context_key, int(maximum_dimension), int(level_count),
      size_pairs, level_weights, fine_dimensions,
      float(duplicate_level_weight), bool(compact_packed_offsets),
    )
    cached = _CUDA_PRECONDITIONER_SPECIALIZATION_CACHE.get(key)
    reused_module = cached is not None
    if cached is None:
      source = (
        Path(__file__).resolve().parent
        / "cuda" / "preconditioner_specialized.cu"
      ).read_text(encoding="utf8")
      cases = "\n".join(
        "    case ({n}u << 8) | {p}u:\n"
        "      yasps_mas_apply_exact_domain<{n}, {p}>(\n"
        "          inverses, residuals, corrections,\n"
        "          matrix_offsets[domain], vector_offsets[domain]);\n"
        "      break;".format(n=n, p=p)
        for n, p in size_pairs
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_INVERSE_CASES", cases
      )
      warp_cases = "\n".join(
        "    case ({n}u << 8) | {p}u:\n"
        "      yasps_mas_apply_exact_domain_warp<{n}, {p}>(\n"
        "          inverses, residuals, corrections,\n"
        "          matrix_offsets[domain], vector_offsets[domain], lane);\n"
        "      break;".format(n=n, p=p)
        for n, p in size_pairs
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_WARP_INVERSE_CASES", warp_cases
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_SUBWARP_INVERSE_CASES",
        "\n".join(
          "    case ({n}u << 8) | {p}u:\n"
          "      yasps_mas_apply_exact_domain_subwarp<{n}, {p}>(\n"
          "          inverses, residuals, corrections,\n"
          "          matrix_offsets[domain], vector_offsets[domain]);\n"
          "      break;".format(n=n, p=p)
          for n, p in size_pairs
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_LEVEL_WEIGHTS",
        ", ".join(f"{float(weight)!r}f" for weight in level_weights),
      )
      source = source.replace(
        "YASPS_MAS_GENERATED_DUPLICATE_LEVEL_WEIGHT",
        f"{float(duplicate_level_weight)!r}f",
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_RESTRICT_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_restrict_one_node<{dimension}>(\n"
          "          fine, packed, fine_node_level_keys,\n"
          "          fine_node_scalar_offsets, fine_node,\n"
          "          fine_node_count);\n"
          "      return;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_RESTRICT_COARSE_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_restrict_one_node_coarse<"
          f"{dimension}>(\n"
          "          fine, packed, fine_node_level_keys,\n"
          "          fine_node_scalar_offsets, fine_node,\n"
          "          fine_node_count);\n"
          "      return;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_UPDATE_RESTRICT_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_update_restrict_one_node<{dimension}>(\n"
          "          solution, direction, residual, product, state,\n"
          "          packed, fine_node_level_keys,\n"
          "          fine_node_scalar_offsets, fine_node,\n"
          "          fine_node_count);\n"
          "      return;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_UPDATE_RESTRICT_COARSE_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_update_restrict_one_node_coarse<"
          f"{dimension}>(\n"
          "          solution, direction, residual, product, state,\n"
          "          packed, fine_node_level_keys,\n"
          "          fine_node_scalar_offsets, fine_node,\n"
          "          fine_node_count);\n"
          "      return;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_COLLECT_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_collect_one_node<{dimension}>(\n"
          "          packed, fine, fine_node_to_packed_starts,\n"
          "          fine_node_scalar_offsets, fine_node_level_active,\n"
          "          fine_node,\n"
          "          fine_node_count, level_weights,\n"
          "          duplicate_level_weight, fine_level_weight);\n"
          "      return;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_COLLECT_DOT_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_collect_one_node_dots<{dimension}>(\n"
          "          packed, fine, residual,\n"
          "          fine_node_to_packed_starts,\n"
          "          fine_node_scalar_offsets,\n"
          "          fine_node_level_active, fine_node,\n"
          "          fine_node_count, level_weights,\n"
          "          duplicate_level_weight, fine_level_weight,\n"
          "          local_rz, local_residual2);\n"
          "      break;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_COLLECT_COARSE_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_collect_coarse_one_node<"
          f"{dimension}, level_count>(\n"
          "          packed, fine, fine_node_to_packed_starts,\n"
          "          fine_node_scalar_offsets, fine_node_level_active,\n"
          "          fine_node, fine_node_count, level_weights,\n"
          "          duplicate_level_weight);\n"
          "      return;"
          for dimension in fine_dimensions
        ),
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_COLLECT_COARSE_DOT_DIMENSION_CASES",
        "\n".join(
          f"    case {dimension}u:\n"
          f"      yasps_mas_collect_coarse_one_node_dots<"
          f"{dimension}, level_count>(\n"
          "          packed, fine, residual,\n"
          "          fine_node_to_packed_starts,\n"
          "          fine_node_scalar_offsets,\n"
          "          fine_node_level_active, fine_node,\n"
          "          fine_node_count, level_weights,\n"
          "          duplicate_level_weight, local_rz,\n"
          "          local_residual2);\n"
          "      break;"
          for dimension in fine_dimensions
        ),
      )
      compiled_weights = ", ".join(
        f"{float(weight)!r}f" for weight in level_weights
      )
      source = source.replace(
        "// YASPS_MAS_COARSE_LEVEL_WEIGHTS",
        compiled_weights,
      )
      source = source.replace(
        "// YASPS_MAS_COARSE_DOT_LEVEL_WEIGHTS",
        compiled_weights,
      )
      source = source.replace(
        "// YASPS_MAS_GENERATED_FINE_INVERSE_CASES",
        "\n".join(
          "    case ({n}u << 8) | {p}u:\n"
          "      yasps_mas_apply_fine_domain_warp<{n}, {p}>(\n"
          "          inverses, fine_residual, fine_correction,\n"
          "          packed_to_fine, warp_residual,\n"
          "          matrix_offsets[domain],\n"
          "          vector_offsets[domain], lane,\n"
          "          fine_level_weight);\n"
          "      break;".format(n=n, p=p)
          for n, p in size_pairs
        ),
      )
      names = (
        "yasps_mas_restrict_warp_nodes_mixed_specialized",
        "yasps_mas_update_restrict_warp_nodes_mixed_specialized",
        "yasps_mas_restrict_coarse_warp_nodes_mixed_specialized",
        "yasps_mas_update_restrict_coarse_warp_nodes_mixed_specialized",
        "yasps_mas_collect_nodes_mixed_specialized",
        "yasps_mas_collect_nodes_mixed_specialized_dots",
        "yasps_mas_collect_coarse_nodes_mixed_specialized",
        "yasps_mas_collect_coarse_nodes_mixed_specialized_dots",
        "yasps_mas_dense_inverse_apply_mixed_specialized",
        "yasps_mas_dense_inverse_apply_warp_domains_specialized",
        "yasps_mas_dense_inverse_apply_subwarp_domains_specialized",
        "yasps_mas_apply_fine_inverse_warp_domains_specialized",
      )
      module = compile_cuda_library(
        self.cuda,
        source,
        kernel_names=names,
        label="preconditioner",
        options=(
          f"-DYASPS_MAS_MAX_DIMENSION={int(maximum_dimension)}",
          f"-DYASPS_MAS_LEVEL_COUNT={int(level_count)}",
          f"-DYASPS_MAS_MAX_PADDED_SIZE="
          f"{max(padded for _, padded in size_pairs)}",
          *(
            ["-DYASPS_MAS_COMPACT_PACKED_OFFSETS=1"]
            if compact_packed_offsets else []
          ),
        ),
      )
      cached = {
        "module": module,
        "restrict": module.kernel(names[0]),
        "update_restrict": module.kernel(names[1]),
        "restrict_coarse": module.kernel(names[2]),
        "update_restrict_coarse": module.kernel(names[3]),
        "collect": module.kernel(names[4]),
        "collect_dots": module.kernel(names[5]),
        "collect_coarse": module.kernel(names[6]),
        "collect_coarse_dots": module.kernel(names[7]),
        "inverse_apply": module.kernel(names[8]),
        "inverse_apply_warp_domains": module.kernel(names[9]),
        "inverse_apply_subwarp_domains": module.kernel(names[10]),
        "fine_inverse_apply": module.kernel(names[11]),
      }
      _CUDA_PRECONDITIONER_SPECIALIZATION_CACHE[key] = cached
    self._record_jit_library(
      cached["module"], reused=reused_module
    )
    return cached

  def _empty(self, size, dtype):
    return self.gpuarray.empty(size, dtype=dtype, allocator=self.allocator)

  def _zeros(self, size, dtype):
    return self.gpuarray.zeros(size, dtype=dtype, allocator=self.allocator)

  def _to_gpu(self, value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return self.gpuarray.to_gpu(array, allocator=self.allocator)

  def _upload(
    self, key: str, value, dtype, *, immutable: bool = False,
    skip_unchanged: bool = False,
  ):
    dtype = np.dtype(dtype)
    if is_pycuda_array(value):
      result = value if np.dtype(value.dtype) == dtype else value.astype(dtype)
      return result if int(result.size) == 0 else result.reshape(-1)
    host = np.ascontiguousarray(np.asarray(value, dtype=dtype).reshape(-1))
    required = int(host.size)
    slot = self._uploads.get(key)
    if slot is not None and immutable:
      return slot.array[:required]
    if slot is None or slot.capacity < required:
      previous = 0 if slot is None else slot.capacity
      grown = max(
        int(np.ceil(required * self.upload_growth_factor)),
        int(np.ceil(previous * self.upload_growth_factor)),
        1,
      )
      slot = UploadSlot(self._empty(grown, dtype), grown)
      self._uploads[key] = slot
      self.upload_reallocations += 1
    unchanged = bool(
      skip_unchanged and slot.host_value is not None and
      np.array_equal(slot.host_value, host)
    )
    if required and not unchanged:
      self.cuda.memcpy_htod(slot.array.gpudata, host)
    if skip_unchanged and not unchanged:
      slot.host_value = host.copy()
    return slot.array[:required]

  def _domain_layout(self, level: HierarchyLevel):
    sizes = level.domain_scalar_sizes.astype(np.uint32, copy=True)
    if self.fixed_inverse_bucket_size is None:
      padded = np.asarray(
        [bucket_size(int(size)) for size in sizes], dtype=np.uint32
      )
    else:
      if sizes.size and int(sizes.max()) > self.fixed_inverse_bucket_size:
        raise ValueError(
          "fixed inverse bucket is smaller than a hierarchy domain: "
          f"{self.fixed_inverse_bucket_size} < {int(sizes.max())}"
        )
      padded = np.full(
        sizes.shape, self.fixed_inverse_bucket_size, dtype=np.uint32
      )
    node_domains = np.empty(level.number_of_nodes, dtype=np.uint32)
    node_local_offsets = np.empty(level.number_of_nodes, dtype=np.uint32)
    domain_nodes = np.concatenate(level.domains).astype(np.int64, copy=False)
    domain_ids = np.repeat(
      np.arange(len(level.domains), dtype=np.uint32),
      [len(nodes) for nodes in level.domains],
    )
    local_offsets = np.concatenate(level.domain_scalar_offsets).astype(np.uint32, copy=False)
    node_domains[domain_nodes] = domain_ids
    node_local_offsets[domain_nodes] = local_offsets

    vector_offsets = np.cumsum(np.r_[0, sizes[:-1]], dtype=np.uint64)
    dimensions = level.node_dimensions.astype(np.int64, copy=False)
    scalar_nodes = np.repeat(np.arange(level.number_of_nodes, dtype=np.int64), dimensions)
    within_node = np.arange(level.number_of_scalar_dofs, dtype=np.uint64) - np.repeat(
      level.node_scalar_offsets.astype(np.uint64, copy=False), dimensions
    )
    packed_node_starts = (
      vector_offsets[node_domains.astype(np.int64)] + node_local_offsets
    ).astype(np.uint64, copy=False)
    scalar_to_packed = np.repeat(packed_node_starts, dimensions) + within_node
    return sizes, padded, node_domains, node_local_offsets, vector_offsets, scalar_to_packed

  @staticmethod
  def _fine_scalar_to_level(level: HierarchyLevel, fine: HierarchyLevel, node_map):
    dimensions = fine.node_dimensions.astype(np.int64, copy=False)
    scalar_nodes = np.repeat(np.arange(fine.number_of_nodes, dtype=np.int64), dimensions)
    within_node = np.arange(fine.number_of_scalar_dofs, dtype=np.int64) - np.repeat(
      fine.node_scalar_offsets.astype(np.int64, copy=False), dimensions
    )
    starts = level.node_scalar_offsets[np.asarray(node_map, dtype=np.int64)[scalar_nodes]]
    return (starts.astype(np.int64, copy=False) + within_node).astype(np.uint32)

  def _build_static_state(self, view: BlockSparseMatrixView) -> None:
    started = perf_counter()
    fine = self.hierarchy.levels[0]
    self.fine_node_count = fine.number_of_nodes
    self.fine_dofs = fine.number_of_scalar_dofs
    self.level_count = len(self.hierarchy.levels)
    if self.configured_level_weights is None:
      self.level_weights = (
        (self.coarsest_level_weight,) if self.level_count == 1 else
        (1.5,) + (1.0,) * max(0, self.level_count - 2)
        + (self.coarsest_level_weight,)
      )
    else:
      if len(self.configured_level_weights) != self.level_count:
        raise ValueError(
          "level_weights length must equal the constructed hierarchy "
          f"level count ({self.level_count})"
        )
      self.level_weights = self.configured_level_weights

    dimensions = to_host(view.variable_dimensions, np.uint32).reshape(-1)
    offsets = to_host(view.variable_scalar_offsets, np.uint32).reshape(-1)
    unique_dimensions = tuple(sorted(map(int, np.unique(dimensions))))
    possible_shapes = tuple(
      (rows, cols)
      for rows in unique_dimensions for cols in unique_dimensions
    )
    if len(possible_shapes) > 64:
      possible_shapes = tuple(sorted({
        tuple(map(int, shape))
        for raw in (
          view.static_block_dimensions,
          view.dynamic_block_dimensions,
        )
        for shape in to_host(raw, np.uint32).reshape((-1, 2))
      }))
    self.spmv_auxiliary_shapes = possible_shapes
    # One descriptor is six uint64 values. Sixty-four entries cost only
    # 3 KiB and cover repeated static/dynamic categories without any hot-
    # solve allocation. An unusual larger layout simply uses the regular
    # generated per-shape launches.
    self.spmv_auxiliary_capacity = max(64, 2 * len(possible_shapes))
    self.spmv_auxiliary_host = self.cuda.pagelocked_empty(
      self.spmv_auxiliary_capacity * 6, np.uint64
    )
    self.spmv_auxiliary_descriptors = self._empty(
      self.spmv_auxiliary_capacity * 6, np.uint64
    )
    boundary = np.full(view.rows, np.iinfo(np.uint32).max, dtype=np.uint32)
    boundary[offsets] = np.arange(offsets.size, dtype=np.uint32)
    self.boundary_to_node = self._to_gpu(boundary)
    self.fine_dimensions = self._to_gpu(dimensions)

    # Do not add an unchanged local Schwarz space repeatedly merely because
    # other components of the graph continue coarsening. A target domain is
    # an exact duplicate when every parent has one child and those children
    # are precisely one preceding domain. This commonly occurs for rigid or
    # affine blocks beside a much larger deformable subsystem. GIPC handles
    # those with one separate block inverse rather than summing that inverse
    # at every FEM MAS level.
    fine_level_active = np.ones(
      (self.level_count, self.fine_node_count), dtype=np.uint8
    )
    # Runtime aliases change coordinates inside a static bank, but do not
    # make an already duplicated static Schwarz space independent. Keep
    # this topology weight active in collision-aware mode as well; dropping
    # it there over-counted unchanged affine/translation components once
    # per level and substantially damaged PCG convergence.
    for level_index, transfer in enumerate(
      self.hierarchy.adjacent_maps, start=1
    ):
      previous = self.hierarchy.levels[level_index - 1]
      current = self.hierarchy.levels[level_index]
      mapping = np.asarray(
        transfer.fine_node_to_parent, dtype=np.int64
      )
      children: list[list[int]] = [
        [] for _ in range(current.number_of_nodes)
      ]
      for child, parent in enumerate(mapping):
        children[int(parent)].append(child)
      previous_domain = np.empty(
        previous.number_of_nodes, dtype=np.int64
      )
      for domain, nodes in enumerate(previous.domains):
        previous_domain[np.asarray(nodes, dtype=np.int64)] = domain
      duplicate_parent = np.zeros(
        current.number_of_nodes, dtype=bool
      )
      for nodes in current.domains:
        child_nodes = []
        for parent in nodes:
          if len(children[parent]) != 1:
            break
          child_nodes.append(children[parent][0])
        else:
          source_domain = int(previous_domain[child_nodes[0]])
          if (all(previous_domain[child] == source_domain
              for child in child_nodes)
              and set(child_nodes)
              == set(previous.domains[source_domain])):
            duplicate_parent[np.asarray(nodes, dtype=np.int64)] = True
      fine_parents = np.asarray(
        self.hierarchy.composed_node_maps[level_index],
        dtype=np.int64,
      )
      fine_level_active[level_index] = (
        ~duplicate_parent[fine_parents]
      ).astype(np.uint8, copy=False)
    self.fine_node_level_active = self._to_gpu(
      fine_level_active.reshape(-1), np.uint8
    )
    host_layouts = [self._domain_layout(level) for level in self.hierarchy.levels]
    level_node_bases = np.cumsum(
      np.r_[0, [level.number_of_nodes
           for level in self.hierarchy.levels][:-1]],
      dtype=np.uint64,
    )
    self.level_node_bases_host = [
      int(value) for value in level_node_bases
    ]
    self.level_node_counts = [
      level.number_of_nodes for level in self.hierarchy.levels
    ]
    level_domain_bases = np.cumsum(
      np.r_[0, [len(level.domains) for level in self.hierarchy.levels][:-1]],
      dtype=np.uint32,
    )
    sizes = np.concatenate([layout[0] for layout in host_layouts])
    padded = np.concatenate([layout[1] for layout in host_layouts])
    self.domain_count = int(sizes.size)
    self.fine_domain_count = len(self.hierarchy.levels[0].domains)

    grouped: dict[tuple[int, int], list[int]] = {}
    for domain, (active_size, padded_size) in enumerate(zip(sizes, padded)):
      grouped.setdefault(
        (int(padded_size), int(active_size)), []
      ).append(domain)
    matrix_offsets = np.empty(self.domain_count, dtype=np.uint64)
    inverse_bucket_specs = []
    matrix_cursor = 0
    for (padded_size, active_size), domains in sorted(grouped.items()):
      matrix_start = matrix_cursor
      for batch, domain in enumerate(domains):
        matrix_offsets[domain] = matrix_start + batch * padded_size * padded_size
      matrix_cursor += len(domains) * padded_size * padded_size
      inverse_bucket_specs.append((
        padded_size, active_size, matrix_start,
        np.asarray(domains, np.int64),
      ))
    self.matrix_storage_size = int(matrix_cursor)
    maximum_inverse_shared = max(
      (2 * int(size) * int(size) + int(size)) * 8
      for size in padded
    )
    default_shared = int(self.cuda.Context.get_device().get_attribute(
      self.cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
    ))
    if maximum_inverse_shared > default_shared:
      opt_in_shared = int(self.cuda.Context.get_device().get_attribute(
        self.cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
      ))
      if maximum_inverse_shared > opt_in_shared:
        raise ValueError(
          "maximum MAS domain needs "
          f"{maximum_inverse_shared} shared bytes, but this GPU supports "
          f"only {opt_in_shared} opt-in bytes"
        )
      self.inverse_kernel.set_attribute(
        self.cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
        maximum_inverse_shared,
      )
      self.inverse_spd_mixed_ragged_kernel.set_attribute(
        self.cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
        maximum_inverse_shared,
      )

    level_vector_bases = np.cumsum(
      np.r_[0, [level.number_of_scalar_dofs for level in self.hierarchy.levels][:-1]],
      dtype=np.uint64,
    )
    self.packed_vector_size = int(sum(level.number_of_scalar_dofs for level in self.hierarchy.levels))
    self.compact_packed_offsets = bool(
      not self.collision_aware_reorder
      and self.packed_vector_size <= np.iinfo(np.uint32).max
    )
    vector_offsets_parts = []
    fine_to_packed_parts = []
    fine_to_level_parts = []
    fine_node_domains_parts = []
    fine_node_local_parts = []
    fine_node_scalar_starts_parts = []
    packed_node_starts_parts = []
    packed_node_dimensions_parts = []
    fine_node_to_packed_starts_parts = []
    node_domains_parts = []
    node_local_parts = []
    node_domain_ordinals_parts = []
    node_type_parts = []
    domain_nodes_parts = []
    domain_node_offsets = [0]
    for level_index, (level, layout, domain_base, vector_base, node_map) in enumerate(zip(
      self.hierarchy.levels,
      host_layouts,
      level_domain_bases,
      level_vector_bases,
      self.hierarchy.composed_node_maps,
    )):
      _, _, node_domains, node_local, vector_offsets, scalar_to_packed = layout
      node_map = np.asarray(node_map, dtype=np.int64)
      vector_offsets_parts.append(vector_offsets + vector_base)
      fine_to_level = self._fine_scalar_to_level(level, fine, node_map)
      fine_to_level_parts.append(fine_to_level)
      fine_to_packed_parts.append(scalar_to_packed[fine_to_level] + vector_base)
      fine_node_domains_parts.append(node_domains[node_map] + domain_base)
      fine_node_local_parts.append(node_local[node_map])
      fine_node_scalar_starts_parts.append(level.node_scalar_offsets[node_map])
      packed_node_starts = (
        vector_offsets[node_domains.astype(np.int64)] + node_local + vector_base
      ).astype(np.uint64, copy=False)
      packed_node_starts_parts.append(packed_node_starts)
      packed_node_dimensions_parts.append(
        level.node_dimensions.astype(np.uint32, copy=False)
      )
      fine_node_to_packed_starts_parts.append(packed_node_starts[node_map])
      node_domains_parts.append(node_domains + domain_base)
      node_local_parts.append(node_local)
      ordinals = np.empty(level.number_of_nodes, dtype=np.uint32)
      for domain_nodes in level.domains:
        domain_array = np.asarray(domain_nodes, dtype=np.int64)
        ordinals[domain_array] = np.arange(
          domain_array.size, dtype=np.uint32
        )
        domain_nodes_parts.append(
          domain_array.astype(np.uint32, copy=False)
          + np.uint32(level_node_bases[level_index])
        )
        domain_node_offsets.append(
          domain_node_offsets[-1] + int(domain_array.size)
        )
      node_domain_ordinals_parts.append(ordinals)
      node_type_parts.append(
        np.zeros(level.number_of_nodes, dtype=np.int64)
        if level.node_type_ids is None else
        np.asarray(level.node_type_ids, dtype=np.int64)
      )

    vector_offsets = np.concatenate(vector_offsets_parts).astype(np.uint64, copy=False)
    fine_to_packed = np.concatenate(fine_to_packed_parts).astype(np.uint64, copy=False)
    fine_to_level = np.concatenate(fine_to_level_parts).astype(np.uint32, copy=False)
    fine_node_domains = np.concatenate(fine_node_domains_parts).astype(np.uint32, copy=False)
    fine_node_local = np.concatenate(fine_node_local_parts).astype(np.uint32, copy=False)
    fine_node_scalar_starts = np.concatenate(fine_node_scalar_starts_parts).astype(
      np.uint32, copy=False
    )
    packed_node_starts = np.concatenate(packed_node_starts_parts).astype(
      np.uint64, copy=False
    )
    packed_node_dimensions = np.concatenate(packed_node_dimensions_parts).astype(
      np.uint32, copy=False
    )
    fine_node_to_packed_starts = np.concatenate(
      fine_node_to_packed_starts_parts
    ).astype(np.uint64, copy=False)
    self.packed_node_count = int(packed_node_starts.size)
    fine_to_level_node = np.concatenate([
      np.asarray(mapping, dtype=np.uint32)
      for mapping in self.hierarchy.composed_node_maps
    ])
    fine_node_level_keys = (
      ((fine_to_level_node.astype(np.uint64, copy=False)
       + np.repeat(
         level_node_bases, self.fine_node_count
       ).astype(np.uint64, copy=False)) << np.uint64(32))
      | fine_node_to_packed_starts
    )
    compact_fine_node_to_packed_starts = (
      fine_node_to_packed_starts.astype(np.uint32, copy=True)
      if self.compact_packed_offsets else None
    )
    node_domains = np.concatenate(node_domains_parts).astype(
      np.uint32, copy=False
    )
    node_local_offsets = np.concatenate(node_local_parts).astype(
      np.uint32, copy=False
    )
    node_domain_ordinals = np.concatenate(
      node_domain_ordinals_parts
    ).astype(np.uint32, copy=False)
    node_type_ids = np.concatenate(node_type_parts).astype(
      np.int64, copy=False
    )
    domain_nodes = np.concatenate(domain_nodes_parts).astype(
      np.uint32, copy=False
    )
    domain_node_offsets = np.asarray(domain_node_offsets, dtype=np.uint64)
    node_to_next = np.full(
      self.packed_node_count, np.iinfo(np.uint32).max, dtype=np.uint32
    )
    packed_to_next = np.full(
      self.packed_vector_size, np.iinfo(np.uint64).max, dtype=np.uint64
    )
    for level_index, transfer in enumerate(self.hierarchy.adjacent_maps):
      level = self.hierarchy.levels[level_index]
      dimensions_at_level = level.node_dimensions.astype(
        np.int64, copy=False
      )
      scalar_nodes = np.repeat(
        np.arange(level.number_of_nodes, dtype=np.int64),
        dimensions_at_level,
      )
      within_node = (
        np.arange(level.number_of_scalar_dofs, dtype=np.int64)
        - np.repeat(
          level.node_scalar_offsets.astype(np.int64, copy=False),
          dimensions_at_level,
        )
      )
      target_level = self.hierarchy.levels[level_index + 1]
      target_scalars = (
        target_level.node_scalar_offsets[
          transfer.fine_node_to_parent[scalar_nodes]
        ].astype(np.int64, copy=False)
        + within_node
      )
      source_packed = (
        host_layouts[level_index][5]
        + level_vector_bases[level_index]
      ).astype(np.uint64, copy=False)
      target_packed = (
        host_layouts[level_index + 1][5][target_scalars]
        + level_vector_bases[level_index + 1]
      ).astype(np.uint64, copy=False)
      packed_to_next[source_packed] = target_packed
      source_begin = int(level_node_bases[level_index])
      target_begin = int(level_node_bases[level_index + 1])
      source_count = self.hierarchy.levels[level_index].number_of_nodes
      node_to_next[source_begin : source_begin + source_count] = (
        transfer.fine_node_to_parent.astype(np.uint32, copy=False)
        + np.uint32(target_begin)
      )
    ancestry_keys = [np.arange(self.fine_node_count, dtype=np.int64)]
    ancestry_keys.extend(
      np.asarray(mapping, dtype=np.int64)
      for mapping in self.hierarchy.composed_node_maps[1:]
    )
    restriction_order = np.lexsort(tuple(ancestry_keys)).astype(
      np.uint32, copy=False
    )
    self.maximum_fine_dimension = int(dimensions.max(initial=0))
    self.maximum_padded_size = int(padded.max(initial=0))
    self.dynamic_edge_padded_size = max(
      self.maximum_padded_size,
      bucket_size(self.maximum_fine_dimension),
    )
    preconditioner_specialized = self._get_specialized_preconditioner_kernels(
      self.maximum_fine_dimension, self.level_count,
      tuple(sorted({
        (int(size), int(pad))
        for size, pad in zip(sizes, padded)
      })), self.level_weights,
      tuple(sorted(map(int, np.unique(dimensions)))),
      self.duplicate_level_weight,
      self.compact_packed_offsets,
    )
    self.restrict_warp_nodes_mixed_specialized_kernel = (
      preconditioner_specialized["restrict"]
    )
    self.update_restrict_warp_nodes_mixed_specialized_kernel = (
      preconditioner_specialized["update_restrict"]
    )
    self.restrict_coarse_warp_nodes_mixed_specialized_kernel = (
      preconditioner_specialized["restrict_coarse"]
    )
    self.update_restrict_coarse_warp_nodes_mixed_specialized_kernel = (
      preconditioner_specialized["update_restrict_coarse"]
    )
    self.collect_nodes_mixed_specialized_kernel = (
      preconditioner_specialized["collect"]
    )
    self.collect_nodes_mixed_specialized_dots_kernel = (
      preconditioner_specialized["collect_dots"]
    )
    self.collect_coarse_nodes_mixed_specialized_kernel = (
      preconditioner_specialized["collect_coarse"]
    )
    self.collect_coarse_nodes_mixed_specialized_dots_kernel = (
      preconditioner_specialized["collect_coarse_dots"]
    )
    # Collision-edge corrections are appended after collection and must be
    # included in r^T z; retain the standalone reduction in that optional
    # overlapping mode.
    self.fuses_pcg_dots = not self.dynamic_edge_domains_active
    # Updating solution/residual in scalar order is faster than folding it
    # into restriction_order: the latter saves one read but turns four
    # recurrence-vector streams into scattered accesses on real Hessians.
    self.fuses_solution_update_restriction = False
    self.inverse_apply_mixed_specialized_kernel = (
      preconditioner_specialized["inverse_apply"]
    )
    self.inverse_apply_warp_domains_kernel = (
      preconditioner_specialized["inverse_apply_warp_domains"]
    )
    self.inverse_apply_subwarp_domains_kernel = (
      preconditioner_specialized["inverse_apply_subwarp_domains"]
    )
    self.fine_inverse_apply_warp_domains_kernel = (
      preconditioner_specialized["fine_inverse_apply"]
    )
    # Level-zero Schwarz domains are a disjoint partition.  Their
    # restriction, inverse product, and prolongation can therefore bypass
    # both packed vectors and the generic three-kernel path.
    self.fused_fine_domain_apply = False
    # Retain the dimension-generated subwarp kernel for explicit
    # experiments, but route the general path through row warps until an
    # architecture-specific autotuner selects otherwise.
    self.use_subwarp_inverse_apply = False
    packed_scalar_domains = np.repeat(
      np.arange(self.domain_count, dtype=np.uint32),
      sizes.astype(np.int64, copy=False),
    )
    packed_scalar_local_offsets = np.concatenate([
      np.arange(int(size), dtype=np.uint32) for size in sizes
    ])
    level0_packed_to_fine = np.empty(self.fine_dofs, dtype=np.uint32)
    level0_packed_to_fine[
      fine_to_packed[:self.fine_dofs].astype(np.int64, copy=False)
    ] = np.arange(self.fine_dofs, dtype=np.uint32)

    # Resolve immutable static coordinates to the first hierarchy bank in
    # which both endpoints meet. Numerical updates can then scatter by one
    # precomputed address instead of repeating six boundary/map/domain
    # lookups for every Hessian block. Adjacent propagation materializes
    # that contribution at every later level exactly as before.
    static_counts = to_host(
      view.static_category_counts, np.uint64
    ).reshape(-1)
    static_block_count = int(static_counts.sum(dtype=np.uint64))
    static_positions = to_host(
      view.static_positions, np.uint32
    ).reshape(-1)
    if static_positions.size < 2 * static_block_count:
      raise ValueError("static positions buffer is too small")
    static_positions = static_positions[:2 * static_block_count].reshape(
      -1, 2
    )
    invalid_offset = np.iinfo(np.uint64).max
    static_destinations = np.full(
      static_block_count, invalid_offset, dtype=np.uint64
    )
    static_transposes = np.full(
      static_block_count, invalid_offset, dtype=np.uint64
    )
    if padded.size and int(padded.max()) > np.iinfo(np.uint8).max:
      raise ValueError(
        "CUDA static scatter stride exceeds 8-bit device metadata"
      )
    static_strides = np.zeros(static_block_count, dtype=np.uint8)
    if static_block_count:
      scalar_rows = static_positions[:, 0].astype(
        np.int64, copy=False
      )
      scalar_cols = static_positions[:, 1].astype(
        np.int64, copy=False
      )
      if (int(scalar_rows.max(initial=0)) >= self.fine_dofs
          or int(scalar_cols.max(initial=0)) >= self.fine_dofs):
        raise ValueError("static block coordinate exceeds matrix size")
      fine_rows = boundary[scalar_rows]
      fine_cols = boundary[scalar_cols]
      invalid_node = np.iinfo(np.uint32).max
      if (np.any(fine_rows == invalid_node)
          or np.any(fine_cols == invalid_node)):
        raise ValueError("static block coordinate starts inside a node")
      unresolved = np.ones(static_block_count, dtype=bool)
      for level in range(self.level_count):
        map_base = level * self.fine_node_count
        row_indices = map_base + fine_rows.astype(
          np.int64, copy=False
        )
        col_indices = map_base + fine_cols.astype(
          np.int64, copy=False
        )
        row_domains = fine_node_domains[row_indices]
        selected = unresolved & (
          row_domains == fine_node_domains[col_indices]
        )
        if not np.any(selected):
          continue
        domains = row_domains[selected].astype(np.int64, copy=False)
        strides = padded[domains].astype(np.uint64, copy=False)
        row_local = fine_node_local[row_indices[selected]].astype(
          np.uint64, copy=False
        )
        col_local = fine_node_local[col_indices[selected]].astype(
          np.uint64, copy=False
        )
        bases = matrix_offsets[domains]
        static_destinations[selected] = (
          bases + row_local * strides + col_local
        )
        if view.symmetric_storage:
          mirrored = selected & (fine_rows != fine_cols)
          mirrored_domains = row_domains[mirrored].astype(
            np.int64, copy=False
          )
          mirrored_strides = padded[mirrored_domains].astype(
            np.uint64, copy=False
          )
          static_transposes[mirrored] = (
            matrix_offsets[mirrored_domains]
            + fine_node_local[
              (map_base + fine_cols.astype(np.int64))[mirrored]
            ].astype(np.uint64, copy=False) * mirrored_strides
            + fine_node_local[
              (map_base + fine_rows.astype(np.int64))[mirrored]
            ].astype(np.uint64, copy=False)
          )
        static_strides[selected] = strides.astype(
          np.uint8, copy=False
        )
        unresolved[selected] = False

    self.static_destination_offsets = self._to_gpu(
      static_destinations
    )
    self.static_transpose_offsets = self._to_gpu(static_transposes)
    self.static_destination_strides = self._to_gpu(static_strides)

    self.matrix_offsets = self._to_gpu(matrix_offsets)
    self.vector_offsets = self._to_gpu(vector_offsets)
    self.sizes = self._to_gpu(sizes)
    self.padded_sizes = self._to_gpu(padded)
    self.fine_to_packed = self._to_gpu(fine_to_packed)
    self.fine_to_level = self._to_gpu(fine_to_level)
    self.fine_node_domains = self._to_gpu(fine_node_domains)
    self.fine_node_local_offsets = self._to_gpu(fine_node_local)
    self.fine_node_scalar_starts = self._to_gpu(fine_node_scalar_starts)
    self.fine_node_scalar_offsets = self._to_gpu(offsets)
    self.packed_node_starts = self._to_gpu(packed_node_starts)
    self.packed_node_dimensions = self._to_gpu(packed_node_dimensions)
    self.fine_node_to_packed_starts = self._to_gpu(fine_node_to_packed_starts)
    self.fine_node_level_keys = self._to_gpu(fine_node_level_keys)
    self.compact_fine_node_to_packed_starts = (
      None if compact_fine_node_to_packed_starts is None else
      self._to_gpu(compact_fine_node_to_packed_starts)
    )
    self.specialized_restriction_offsets = (
      self.compact_fine_node_to_packed_starts
      if self.compact_packed_offsets else self.fine_node_level_keys
    )
    self.specialized_collection_offsets = (
      self.compact_fine_node_to_packed_starts
      if self.compact_packed_offsets else self.fine_node_to_packed_starts
    )
    self.restriction_order = self._to_gpu(restriction_order)
    self.packed_scalar_domains = self._to_gpu(packed_scalar_domains)
    self.packed_scalar_local_offsets = self._to_gpu(
      packed_scalar_local_offsets
    )
    self.level0_packed_to_fine = self._to_gpu(level0_packed_to_fine)
    self.packed_to_next_packed = self._to_gpu(packed_to_next)
    self.level_node_bases = self._to_gpu(level_node_bases)
    self.fine_to_level_node = self._to_gpu(fine_to_level_node)
    self.node_domains = self._to_gpu(node_domains)
    self.node_local_offsets = self._to_gpu(node_local_offsets)
    self.node_domain_ordinals = self._to_gpu(node_domain_ordinals)
    self.node_type_ids = self._to_gpu(node_type_ids)
    self.domain_nodes = self._to_gpu(domain_nodes)
    self.domain_node_offsets = self._to_gpu(domain_node_offsets)
    self.node_to_next = self._to_gpu(node_to_next)
    self.connection_masks = self._zeros(
      self.packed_node_count, np.uint64
    )
    self.representatives = self._to_gpu(
      np.arange(self.packed_node_count, dtype=np.uint32)
    )
    self.packed_active = self._to_gpu(
      np.ones(self.packed_vector_size, dtype=np.uint8)
    )
    self.level_domain_bases = [int(value) for value in level_domain_bases]
    self.level_domain_counts = [
      len(level.domains) for level in self.hierarchy.levels
    ]



    self.matrices = self._empty(self.matrix_storage_size, np.float64)
    # Inversion arithmetic is FP64 in shared memory for either algorithm;
    # only the reusable GIPC-style bank is materialized, in FP32.
    self.inverses = self._empty(1, np.float64)
    self.mixed_inverses = self._empty(self.matrix_storage_size, np.float32)
    self.packed_residual = self._empty(self.packed_vector_size, np.float32)
    self.packed_correction = self._empty(self.packed_vector_size, np.float32)
    self._preconditioned_output = self._empty(self.fine_dofs, np.float64)
    self._matvec_output = self._empty(self.fine_dofs, np.float64)
    self._pcg_solution = self._empty(self.fine_dofs, np.float64)
    self._pcg_residual = self._empty(self.fine_dofs, np.float64)
    self._pcg_direction = self._empty(self.fine_dofs, np.float64)
    self._pcg_state = self._zeros(13, np.float64)
    self._pcg_curvature = self._pcg_state[2:3]
    self._pcg_next_values = self._pcg_state[3:5]
    self._pcg_relative_tolerance = self._pcg_state[9:10]
    self._pcg_host_status = self.cuda.pagelocked_empty(2, np.float64)
    self._pcg_host_completion = self.cuda.pagelocked_empty(4, np.float64)
    self._pcg_host_initial = self.cuda.pagelocked_empty(13, np.float64)
    self._pcg_host_control = self.cuda.pagelocked_empty(13, np.float64)
    self._pcg_initial_start_event = self.cuda.Event()
    self._pcg_initial_end_event = self.cuda.Event()
    # status[0] reports block-assembly validation; status[1] folds every
    # bucket's inverse failure so the successful hot path needs one D2H.
    self.status = self._zeros(2, np.int32)
    self.inverse_failure = self.status[1:2]
    self.domain_status = self._zeros(self.domain_count, np.int32)
    self.dynamic_edge_capacity = 1
    edge_storage = (
      self.dynamic_edge_capacity
      * self.dynamic_edge_padded_size * self.dynamic_edge_padded_size
    )
    self.dynamic_edge_matrices = self._empty(edge_storage, np.float64)
    self.dynamic_edge_inverses = self._empty(edge_storage, np.float32)
    self.dynamic_edge_status = self._zeros(
      self.dynamic_edge_capacity, np.int32
    )
    self.dynamic_group_active_sizes = self._zeros(
      self.dynamic_edge_capacity, np.uint32
    )
    self.dynamic_group_scalar_indices = self._empty(
      self.dynamic_edge_capacity * self.dynamic_edge_padded_size,
      np.uint32,
    )
    self.dynamic_group_scalar_nodes = self._empty(
      self.dynamic_edge_capacity * self.dynamic_edge_padded_size,
      np.uint32,
    )
    self.dynamic_edge_node_counts = self._zeros(
      self.fine_node_count, np.uint32
    )
    self._assembly_start_event = self.cuda.Event()
    self._assembly_end_event = self.cuda.Event()
    self._inverse_end_event = self.cuda.Event()
    self._numeric_host_status = self.cuda.pagelocked_empty(2, np.int32)
    self._numeric_update_pending = False

    self.inverse_buckets: list[InverseBucket] = []
    for padded_size, active_size, matrix_start, domains in inverse_bucket_specs:
      batch_sizes = self._to_gpu(sizes[domains].astype(np.int32))
      status = self._zeros(len(domains), np.int32)
      self.inverse_buckets.append(
        InverseBucket(
          padded_size, active_size, matrix_start, domains,
          batch_sizes, status,
        )
      )
    self._ensure_inverse_specializations(
      [bucket.padded_size for bucket in self.inverse_buckets]
      + [self.dynamic_edge_padded_size]
    )
    self.exact_inverse_kernels = {
      (bucket.active_size, bucket.padded_size):
        self._get_exact_inverse_kernel(
          bucket.active_size, bucket.padded_size
        )
      for bucket in self.inverse_buckets
      if (self.inverse_algorithm == "gauss_jordan"
        and bucket.active_size < bucket.padded_size)
    }

    buffers = (
      self.boundary_to_node, self.fine_dimensions, self.matrix_offsets,
      self.static_destination_offsets, self.static_transpose_offsets,
      self.static_destination_strides,
      self.fine_node_level_active,
      self.vector_offsets, self.sizes, self.padded_sizes, self.fine_to_packed,
      self.fine_to_level, self.fine_node_domains, self.fine_node_local_offsets,
      self.fine_node_scalar_starts, self.fine_node_scalar_offsets,
      self.packed_node_starts, self.packed_node_dimensions,
      self.fine_node_to_packed_starts, self.fine_node_level_keys,
      self.restriction_order,
      self.packed_scalar_domains, self.packed_scalar_local_offsets,
      self.level0_packed_to_fine,
      self.packed_to_next_packed,
      self.level_node_bases, self.fine_to_level_node,
      self.node_domains, self.node_local_offsets,
      self.node_domain_ordinals, self.node_type_ids,
      self.domain_nodes, self.domain_node_offsets, self.node_to_next,
      self.connection_masks, self.representatives, self.packed_active,
      self.matrices, self.inverses,
      self.mixed_inverses,
      self.packed_residual, self.packed_correction, self._preconditioned_output,
      self._matvec_output, self._pcg_solution, self._pcg_residual,
      self._pcg_direction, self._pcg_state, self.status,
      self.domain_status, self.dynamic_edge_matrices,
      self.dynamic_edge_inverses, self.dynamic_edge_status,
      self.dynamic_edge_node_counts, self.dynamic_group_active_sizes,
      self.dynamic_group_scalar_indices, self.dynamic_group_scalar_nodes,
      self.spmv_auxiliary_descriptors,
    )
    self.device_bytes = sum(int(buffer.nbytes) for buffer in buffers)
    if self.compact_fine_node_to_packed_starts is not None:
      self.device_bytes += int(
        self.compact_fine_node_to_packed_starts.nbytes
      )
    self.device_bytes += sum(
      int(bucket.batch_sizes.nbytes + bucket.status.nbytes)
      for bucket in self.inverse_buckets
    )
    self.static_setup_seconds = perf_counter() - started

  def _ensure_dynamic_edge_capacity(self, required: int) -> None:
    if required <= self.dynamic_edge_capacity:
      return
    previous_bytes = (
      self.dynamic_edge_matrices.nbytes
      + self.dynamic_edge_inverses.nbytes
      + self.dynamic_edge_status.nbytes
      + self.dynamic_group_active_sizes.nbytes
      + self.dynamic_group_scalar_indices.nbytes
      + self.dynamic_group_scalar_nodes.nbytes
    )
    capacity = max(
      int(np.ceil(required * self.upload_growth_factor)),
      int(np.ceil(self.dynamic_edge_capacity * self.upload_growth_factor)),
    )
    storage = (
      capacity * self.dynamic_edge_padded_size
      * self.dynamic_edge_padded_size
    )
    self.dynamic_edge_matrices = self._empty(storage, np.float64)
    self.dynamic_edge_inverses = self._empty(storage, np.float32)
    self.dynamic_edge_status = self._zeros(capacity, np.int32)
    self.dynamic_group_active_sizes = self._zeros(capacity, np.uint32)
    self.dynamic_group_scalar_indices = self._empty(
      capacity * self.dynamic_edge_padded_size, np.uint32
    )
    self.dynamic_group_scalar_nodes = self._empty(
      capacity * self.dynamic_edge_padded_size, np.uint32
    )
    self.dynamic_edge_capacity = capacity
    current_bytes = (
      self.dynamic_edge_matrices.nbytes
      + self.dynamic_edge_inverses.nbytes
      + self.dynamic_edge_status.nbytes
      + self.dynamic_group_active_sizes.nbytes
      + self.dynamic_group_scalar_indices.nbytes
      + self.dynamic_group_scalar_nodes.nbytes
    )
    self.device_bytes += int(current_bytes - previous_bytes)

  def _part_metadata(self, part: str) -> DeviceInputPart:
    counts = to_host(getattr(self.view, f"{part}_category_counts"), np.uint32).reshape(-1)
    starts = to_host(getattr(self.view, f"{part}_category_starts"), np.uint64).reshape(-1)
    shapes = to_host(getattr(self.view, f"{part}_block_dimensions"), np.uint32)
    shapes = shapes.reshape((-1, 2)) if shapes.size else np.empty((0, 2), np.uint32)
    if not (len(counts) == len(starts) == len(shapes)):
      raise ValueError(f"{part} category metadata lengths differ")
    raw_positions = getattr(self.view, f"{part}_positions")
    raw_values = getattr(self.view, f"{part}_values")
    position_size = int(raw_positions.size) if hasattr(raw_positions, "size") else np.asarray(raw_positions).size
    value_size = int(raw_values.size) if hasattr(raw_values, "size") else np.asarray(raw_values).size
    if int(counts.sum(dtype=np.uint64)) * 2 > position_size:
      raise ValueError(f"{part} positions buffer is too small")
    if shapes.size and np.any(shapes == 0):
      raise ValueError(f"{part} has a category with a non-positive shape")
    areas = np.prod(shapes.astype(np.uint64, copy=False), axis=1)
    ends = starts + counts.astype(np.uint64, copy=False) * areas
    if ends.size and int(ends.max(initial=0)) > value_size:
      raise ValueError(f"{part} category exceeds its values buffer")
    position_offsets = np.cumsum(
      np.r_[np.uint64(0), counts.astype(np.uint64, copy=False)],
      dtype=np.uint64,
    )
    positions = self._upload(
      f"{part}_positions", raw_positions, np.uint32, immutable=(part == "static")
    )
    values = self._upload(f"{part}_values", raw_values, np.float64)
    immutable = part == "static"
    return DeviceInputPart(
      counts, starts, shapes, position_offsets,
      self._upload(
        f"{part}_counts", counts, np.uint32, immutable=immutable,
        skip_unchanged=True,
      ),
      self._upload(
        f"{part}_starts", starts, np.uint64, immutable=immutable,
        skip_unchanged=True,
      ),
      self._upload(
        f"{part}_shapes", shapes, np.uint32, immutable=immutable,
        skip_unchanged=True,
      ),
      self._upload(
        f"{part}_position_offsets", position_offsets, np.uint64,
        immutable=immutable, skip_unchanged=True,
      ),
      positions, values,
    )

  def _refresh_static_metadata(self) -> DeviceInputPart:
    """Reuse immutable static descriptors and refresh only live buffers."""
    previous = self.metadata[0]
    raw_positions = self.view.static_positions
    raw_values = self.view.static_values
    positions = self._upload(
      "static_positions", raw_positions, np.uint32, immutable=True
    )
    values = self._upload("static_values", raw_values, np.float64)
    required_positions = previous.block_count * 2
    areas = np.prod(
      previous.shapes.astype(np.uint64, copy=False), axis=1
    )
    required_values = int((
      previous.starts
      + previous.counts.astype(np.uint64, copy=False) * areas
    ).max(initial=0))
    if int(positions.size) < required_positions:
      raise ValueError("static positions buffer is too small")
    if int(values.size) < required_values:
      raise ValueError("static values buffer is too small")
    return DeviceInputPart(
      previous.counts, previous.starts, previous.shapes,
      previous.position_offsets, previous.device_counts,
      previous.device_starts, previous.device_shapes,
      previous.device_position_offsets, positions, values,
    )

  @staticmethod
  def _metadata_value_count(metadata: DeviceInputPart) -> int:
    if not metadata.counts.size:
      return 0
    areas = np.prod(
      metadata.shapes.astype(np.uint64, copy=False), axis=1
    )
    return int((
      metadata.starts
      + metadata.counts.astype(np.uint64, copy=False) * areas
    ).max(initial=0))

  def _ensure_mixed_spmv_buffers(self) -> None:
    if not self.base_mixed_spmv:
      self.spmv_value_buffers = [
        metadata.values for metadata in self.metadata
      ]
      return
    previous_bytes = sum(
      int(array.nbytes) for array in self.spmv_value_buffers
    )
    current = list(self.spmv_value_buffers)
    while len(current) < len(self.metadata):
      current.append(None)
    for index, metadata in enumerate(self.metadata):
      required = self._metadata_value_count(metadata)
      existing = current[index]
      if existing is None or int(existing.size) < required:
        old_capacity = 0 if existing is None else int(existing.size)
        capacity = max(
          required,
          int(np.ceil(old_capacity * self.upload_growth_factor)),
          1,
        )
        current[index] = self._empty(capacity, np.float32)
    self.spmv_value_buffers = current[:len(self.metadata)]
    current_bytes = sum(
      int(array.nbytes) for array in self.spmv_value_buffers
    )
    if hasattr(self, "device_bytes"):
      self.device_bytes += current_bytes - previous_bytes

  def _submit_mixed_spmv_conversion(self, stream) -> None:
    if not self.base_mixed_spmv:
      return
    for metadata, destination in zip(
      self.metadata, self.spmv_value_buffers
    ):
      count = self._metadata_value_count(metadata)
      if count:
        self.cast_inverse_mixed_kernel(
          metadata.values, destination, np.uint64(count),
          block=(256, 1, 1), grid=((count + 255) // 256, 1, 1),
          stream=stream,
        )

  def _mixed_spmv_conversion_graph(self) -> CapturedGraph | None:
    if not self.base_mixed_spmv:
      return None
    key = tuple(
      (
        int(metadata.values.gpudata),
        int(destination.gpudata),
        self._metadata_value_count(metadata),
      )
      for metadata, destination in zip(
        self.metadata, self.spmv_value_buffers
      )
    )
    graph = self._mixed_spmv_conversion_graphs.get(key)
    if graph is None:
      graph = CapturedGraph.capture(
        self._pcg_stream,
        lambda: self._submit_mixed_spmv_conversion(
          self._pcg_stream
        ),
      )
      self._mixed_spmv_conversion_graphs[key] = graph
    return graph

  def _select_primary_spmv_shape(self) -> tuple[int, int] | None:
    totals: dict[tuple[int, int], int] = {}
    for metadata in self.metadata:
      for count, shape in zip(metadata.counts, metadata.shapes):
        key = tuple(map(int, shape))
        totals[key] = totals.get(key, 0) + int(count)
    if not totals:
      return None
    return max(totals, key=lambda shape: (totals[shape], -shape[0], -shape[1]))

  @staticmethod
  def _spmv_threads(
    rows: int, cols: int, *, cooperative: bool = False,
  ) -> int:
    if cooperative:
      return 128
    return 32 * spmv_warps_for_shape(rows, cols)

  def _refresh_dense_spmv_auxiliary_descriptors(self) -> None:
    """Pack sparse shape tails into the dominant fine-level SpMV launch."""
    records: list[tuple[int, int, int, int, int, int]] = []
    main_count = 0
    primary = self.spmv_primary_shape
    supported = set(self.spmv_auxiliary_shapes)
    value_buffers = (
      self.spmv_value_buffers if self.base_mixed_spmv else
      [metadata.values for metadata in self.metadata]
    )
    for metadata, values in zip(self.metadata, value_buffers):
      position_start = 0
      for count_raw, value_start, shape_raw in zip(
        metadata.counts, metadata.starts, metadata.shapes
      ):
        count = int(count_raw)
        rows, cols = map(int, shape_raw)
        shape = (rows, cols)
        if count and shape == primary:
          main_count += count
        elif count and shape in supported:
          records.append((
            int(values.gpudata),
            int(metadata.positions.gpudata),
            int(value_start), position_start, count,
            (rows << 16) | cols,
          ))
        position_start += count

    total_count = sum(record[4] for record in records)
    # This path is deliberately for sparse shape tails. A substantial
    # category keeps its own block-specialized/row-reducing launch.
    tail_limit = max(64, main_count // 64)
    if (
      not main_count or not total_count or total_count > main_count
      or total_count > tail_limit
      or len(records) > self.spmv_auxiliary_capacity
    ):
      records = []
      total_count = 0

    host = self.spmv_auxiliary_host
    signature = tuple(records)
    changed = signature != self._spmv_auxiliary_records_signature
    if changed:
      for descriptor, record in enumerate(records):
        base = 6 * descriptor
        host[base:base + 6] = record
    self.spmv_auxiliary_descriptor_count = len(records)
    self.spmv_auxiliary_total_count = total_count
    if total_count > self.spmv_auxiliary_launch_capacity:
      self.spmv_auxiliary_launch_capacity = max(
        total_count,
        int(np.ceil(max(
          1, self.spmv_auxiliary_launch_capacity
        ) * self.upload_growth_factor)),
      )
    self.spmv_fused_auxiliary_shapes = tuple(sorted({
      (record[5] >> 16, record[5] & 0xffff) for record in records
    }))
    if records and changed:
      self.cuda.memcpy_htod_async(
        self.spmv_auxiliary_descriptors.gpudata,
        host[:len(records) * 6], self._pcg_stream,
      )
    self._spmv_auxiliary_records_signature = signature

  def _submit_numeric_assembly(self, stream) -> None:
    # The conversion is captured with numerical assembly, so there is no
    # Python category loop on the per-solve path and its cost is charged
    # to the dynamic numerical update.
    self._submit_mixed_spmv_conversion(stream)
    self.status.fill(0, stream=stream)
    static, dynamic = self.metadata
    if self.collision_aware_reorder:
      # Rebuild from the immutable static hierarchy every solve.  A
      # contact which disappeared must not leave a stale alias behind.
      self.reset_runtime_representatives_kernel(
        self.representatives, np.uint32(self.packed_node_count),
        block=(256, 1, 1),
        grid=((self.packed_node_count + 255) // 256, 1, 1),
        stream=stream,
      )
      self.connection_masks.fill(0, stream=stream)
      for level_index in range(self.level_count):
        if level_index and self.dynamic_block_launch_capacity:
          self.build_collision_masks_kernel(
            dynamic.positions, dynamic.device_counts,
            dynamic.device_position_offsets,
            np.uint32(dynamic.counts.size), self.boundary_to_node,
            np.uint32(self.fine_dofs), self.fine_to_level_node,
            self.level_node_bases, self.representatives,
            self.node_to_next, self.node_domains,
            self.node_domain_ordinals, self.packed_node_dimensions,
            self.node_type_ids,
            np.uint8(self.collision_merge_across_types),
            np.uint32(level_index - 1),
            np.uint32(self.fine_node_count), self.connection_masks,
            self.status, block=(128, 1, 1),
            grid=((self.dynamic_block_launch_capacity + 127) // 128,
               1, 1), stream=stream,
          )
        domain_count = self.level_domain_counts[level_index]
        self.close_collision_components_kernel(
          self.domain_node_offsets, self.domain_nodes,
          np.uint32(self.level_domain_bases[level_index]),
          np.uint32(domain_count), self.connection_masks,
          self.representatives, self.status, block=(64, 1, 1),
          grid=(domain_count, 1, 1), stream=stream,
        )
        if level_index + 1 < self.level_count:
          source_count = self.level_node_counts[level_index]
          self.propagate_runtime_components_kernel(
            self.representatives, self.node_to_next,
            self.node_domains, self.node_domain_ordinals,
            self.packed_node_dimensions, self.node_type_ids,
            np.uint8(self.collision_merge_across_types),
            np.uint32(self.level_node_bases_host[level_index]),
            np.uint32(source_count), self.connection_masks,
            self.status, block=(256, 1, 1),
            grid=((source_count + 255) // 256, 1, 1),
            stream=stream,
          )
      map_count = self.level_count * self.fine_node_count
      self.build_runtime_scalar_maps_kernel(
        self.fine_to_level_node, self.level_node_bases,
        self.representatives, self.node_local_offsets,
        self.packed_node_starts, np.uint32(self.level_count),
        np.uint32(self.fine_node_count), self.fine_node_local_offsets,
        self.fine_node_to_packed_starts, self.fine_node_level_keys,
        block=(256, 1, 1),
        grid=((map_count + 255) // 256, 1, 1), stream=stream,
      )
      self.build_runtime_transfer_maps_kernel(
        self.representatives, self.node_to_next,
        self.packed_node_starts, self.packed_node_dimensions,
        np.uint32(self.packed_node_count), np.uint64(
          np.iinfo(np.uint64).max
        ), self.packed_to_next_packed, self.packed_active,
        block=(256, 1, 1),
        grid=((self.packed_node_count + 255) // 256, 1, 1),
        stream=stream,
      )
    self.initialize_domains(
      self.matrices, self.matrix_offsets, self.vector_offsets,
      self.packed_active, self.sizes, self.padded_sizes,
      np.uint32(self.domain_count),
      block=(128, 1, 1), grid=(self.domain_count, 1, 1),
      stream=stream,
    )
    if static.block_count:
      if self.collision_aware_reorder:
        self.static_assembly_kernel(
          static.values, static.positions, self.boundary_to_node,
          np.uint32(self.fine_dofs), self.fine_node_domains,
          self.fine_node_local_offsets, np.uint32(self.level_count),
          np.uint32(self.fine_node_count), self.matrix_offsets,
          self.padded_sizes, self.matrices, self.status,
          block=(self.assembly_threads, 1, 1),
          grid=((static.block_count + self.assembly_threads - 1) //
             self.assembly_threads, 1, 1), stream=stream,
        )
      else:
        self.precomputed_static_assembly_kernel(
          static.values, self.static_destination_offsets,
          self.static_transpose_offsets,
          self.static_destination_strides, self.matrices,
          block=(self.assembly_threads, 1, 1),
          grid=((static.block_count + self.assembly_threads - 1) //
             self.assembly_threads, 1, 1), stream=stream,
        )
    if (self.dynamic_block_launch_capacity
        and self.dynamic_assembly_kernel is not None):
      self.dynamic_assembly_kernel(
        dynamic.values, dynamic.positions, dynamic.device_counts,
        dynamic.device_starts, dynamic.device_position_offsets,
        np.uint32(dynamic.counts.size),
        np.uint32(self.dynamic_block_launch_capacity),
        self.boundary_to_node, np.uint32(self.fine_dofs),
        self.fine_node_domains,
        self.fine_node_local_offsets, np.uint32(self.level_count),
        np.uint32(self.fine_node_count), self.matrix_offsets,
        self.padded_sizes, self.matrices, self.status,
        block=(self.dynamic_assembly_threads, 1, 1),
        grid=((self.dynamic_block_launch_capacity
           + self.dynamic_assembly_threads - 1) //
           self.dynamic_assembly_threads, 1, 1), stream=stream,
      )
    for level_index in range(self.level_count - 1):
      count = self.level_domain_counts[level_index]
      self.propagate_adjacent_domains_kernel(
        self.matrix_offsets, self.vector_offsets, self.sizes,
        self.padded_sizes, self.packed_to_next_packed,
        self.packed_active,
        self.packed_scalar_domains, self.packed_scalar_local_offsets,
        np.uint32(self.level_domain_bases[level_index]),
        np.uint32(count), self.matrices,
        block=(128, 1, 1), grid=(count, 1, 1), stream=stream,
      )
    self.symmetrize_domains(
      self.matrices, self.matrix_offsets, self.sizes,
      self.padded_sizes, np.uint32(self.domain_count),
      block=(128, 1, 1), grid=(self.domain_count, 1, 1),
      stream=stream,
    )
    if (self.dynamic_edge_domains_active and dynamic.block_count
        and self.view.symmetric_storage):
      self.dynamic_edge_node_counts.fill(0, stream=stream)
      self.assemble_dynamic_groups_kernel(
        dynamic.values, dynamic.positions, dynamic.device_counts,
        dynamic.device_starts, dynamic.device_position_offsets,
        dynamic.device_shapes, self.dynamic_group_offsets,
        self.dynamic_group_chunk_sizes,
        np.uint32(dynamic.counts.size),
        np.uint32(self.dynamic_edge_capacity), self.boundary_to_node,
        self.fine_dimensions, np.uint32(self.fine_dofs),
        self.fine_node_scalar_offsets,
        self.fine_node_domains, self.fine_node_local_offsets,
        self.matrix_offsets, self.padded_sizes, self.matrices,
        np.uint32(self.dynamic_edge_padded_size),
        self.dynamic_edge_matrices, self.dynamic_group_active_sizes,
        self.dynamic_group_scalar_indices,
        self.dynamic_group_scalar_nodes,
        self.dynamic_edge_node_counts, self.status,
        block=(128, 1, 1), grid=(self.dynamic_edge_capacity, 1, 1),
        stream=stream,
      )
      self.complete_dynamic_groups_kernel(
        dynamic.values, dynamic.positions, dynamic.device_counts,
        dynamic.device_starts, dynamic.device_position_offsets,
        dynamic.device_shapes, np.uint32(dynamic.counts.size),
        np.uint32(self.dynamic_group_count), self.boundary_to_node,
        self.fine_node_domains, self.dynamic_group_active_sizes,
        self.dynamic_group_scalar_nodes,
        np.uint32(self.dynamic_edge_padded_size),
        self.dynamic_edge_matrices, block=(128, 1, 1),
        grid=(self.dynamic_group_count, 1, 1), stream=stream,
      )

  def _submit_numeric_inverse_bucketed(
    self, stream, buckets: list[InverseBucket] | None = None,
  ) -> None:
    for bucket in self.inverse_buckets if buckets is None else buckets:
      count = len(bucket.domains)
      span = count * bucket.padded_size * bucket.padded_size
      destination = self.mixed_inverses
      exact = self.exact_inverse_kernels.get(
        (bucket.active_size, bucket.padded_size)
      )
      spd_fixed = (
        self.inverse_spd_mixed_fixed_kernels.get(bucket.padded_size)
        if self.inverse_algorithm == "spd" else None
      )
      gj_packed = (
        self.inverse_gj_packed_kernels.get(bucket.padded_size)
        if self.inverse_algorithm == "gauss_jordan" else None
      )
      kernel = (
        exact["kernel"] if exact is not None else
        spd_fixed or gj_packed or self.inverse_kernel
      )
      if exact is not None:
        groups = int(exact["groups"])
        threads = groups * bucket.active_size
        shared_bytes = groups * (
          bucket.active_size * bucket.active_size
          + bucket.active_size
        ) * 8
        grid = ((count + groups - 1) // groups, 1, 1)
        runtime_value = count
      elif gj_packed is not None:
        groups = max(1, 96 // bucket.padded_size)
        threads = groups * bucket.padded_size
        shared_bytes = groups * (
          bucket.padded_size * bucket.padded_size
          + bucket.padded_size
        ) * 8
        grid = ((count + groups - 1) // groups, 1, 1)
        runtime_value = count
      elif spd_fixed is not None:
        threads = max(32, bucket.padded_size)
        shared_bytes = bucket.padded_size * bucket.padded_size * 8
        grid = (count, 1, 1)
        runtime_value = bucket.padded_size
      else:
        threads = self.threads_per_block
        shared_bytes = (
          2 * bucket.padded_size * bucket.padded_size
          + bucket.padded_size
        ) * 8
        grid = (count, 1, 1)
        runtime_value = bucket.padded_size
      kernel(
        self.matrices[bucket.matrix_start : bucket.matrix_start + span],
        destination[bucket.matrix_start : bucket.matrix_start + span],
        bucket.batch_sizes, bucket.status, np.int32(runtime_value),
        np.float64(self.pivot_tolerance), self.inverse_failure,
        block=(threads, 1, 1), grid=grid,
        shared=shared_bytes,
        stream=stream,
      )
      if (self.inverse_algorithm == "gauss_jordan"
          and self.enable_inverse_fallback):
        fallback = (
          exact["fallback"] if exact is not None else
          self.inverse_spd_mixed_fallback_kernels.get(
            bucket.padded_size
          )
        )
        if fallback is not None:
          fallback_groups = (
            1 if exact is not None else
            max(1, 96 // bucket.padded_size)
          )
          fallback(
            self.matrices[
              bucket.matrix_start : bucket.matrix_start + span
            ],
            destination[
              bucket.matrix_start : bucket.matrix_start + span
            ],
            bucket.batch_sizes, bucket.status, np.int32(count),
            np.float64(self.pivot_tolerance), self.inverse_failure,
            block=(
              max(32, bucket.active_size)
              if exact is not None else
              fallback_groups * bucket.padded_size,
              1, 1,
            ),
            grid=(
              count if exact is not None else
              (count + fallback_groups - 1) // fallback_groups,
              1, 1,
            ),
            shared=(
              bucket.active_size * bucket.active_size
              if exact is not None else
              fallback_groups * bucket.padded_size
              * bucket.padded_size
            ) * 8,
            stream=stream,
          )

  def _build_numeric_inverse_graph(self) -> None:
    """Capture a reusable fork/join graph for independent size buckets."""
    # Eight streams reach the measured overlap plateau for heterogeneous
    # inverse buckets; using one stream per tiny tail bucket adds event
    # nodes without a meaningful reduction in elapsed time.
    worker_count = min(8, len(self.inverse_buckets))
    if worker_count <= 1:
      self.inverse_stream_count = 1
      self._inverse_streams = []
      self._inverse_stream_groups = [list(self.inverse_buckets)]

      def submit_serial() -> None:
        self._submit_numeric_inverse_bucketed(self._pcg_stream)

      self._numeric_inverse_graph = CapturedGraph.capture(
        self._pcg_stream, submit_serial,
      )
      return

    # Greedily balance the cubic elimination work. This plan depends only
    # on the static hierarchy and is never recomputed on the solve path.
    groups: list[list[InverseBucket]] = [
      [] for _ in range(worker_count)
    ]
    loads = [0 for _ in range(worker_count)]
    weighted = sorted(
      self.inverse_buckets,
      key=lambda bucket: (
        len(bucket.domains) * bucket.active_size ** 3,
        bucket.active_size,
      ),
      reverse=True,
    )
    for bucket in weighted:
      worker = min(range(worker_count), key=loads.__getitem__)
      groups[worker].append(bucket)
      loads[worker] += len(bucket.domains) * bucket.active_size ** 3

    self.inverse_stream_count = worker_count
    self._inverse_stream_groups = groups
    self._inverse_streams = [
      self.cuda.Stream() for _ in range(worker_count)
    ]
    event_flags = self.cuda.event_flags.DISABLE_TIMING
    self._inverse_fork_event = self.cuda.Event(flags=event_flags)
    self._inverse_done_events = [
      self.cuda.Event(flags=event_flags) for _ in range(worker_count)
    ]

    def submit_parallel() -> None:
      self._inverse_fork_event.record(self._pcg_stream)
      for stream, buckets, done in zip(
        self._inverse_streams,
        self._inverse_stream_groups,
        self._inverse_done_events,
      ):
        stream.wait_for_event(self._inverse_fork_event)
        self._submit_numeric_inverse_bucketed(stream, buckets)
        done.record(stream)
      for done in self._inverse_done_events:
        self._pcg_stream.wait_for_event(done)

    self._numeric_inverse_graph = CapturedGraph.capture(
      self._pcg_stream, submit_parallel,
    )

  def _submit_numeric_inverse(self, stream) -> None:
    self._numeric_inverse_graph.launch()
    count = self.dynamic_group_count
    if (not self.dynamic_edge_domains_active or not count
        or not self.view.symmetric_storage):
      return
    padded = self.dynamic_edge_padded_size
    packed_kernel = self.inverse_gj_packed_kernels.get(padded)
    if packed_kernel is not None:
      groups = max(1, 96 // padded)
      packed_kernel(
        self.dynamic_edge_matrices, self.dynamic_edge_inverses,
        self.dynamic_edge_status, self.dynamic_edge_status,
        np.int32(count), np.float64(self.pivot_tolerance),
        self.inverse_failure, block=(groups * padded, 1, 1),
        grid=((count + groups - 1) // groups, 1, 1),
        shared=groups * (padded * padded + padded) * 8,
        stream=stream,
      )
      self.inverse_spd_mixed_fallback_kernels[padded](
        self.dynamic_edge_matrices, self.dynamic_edge_inverses,
        self.dynamic_edge_status, self.dynamic_edge_status,
        np.int32(count), np.float64(self.pivot_tolerance),
        self.inverse_failure, block=(groups * padded, 1, 1),
        grid=((count + groups - 1) // groups, 1, 1),
        shared=groups * padded * padded * 8,
        stream=stream,
      )
    else:
      self.inverse_gauss_jordan_mixed_kernel(
        self.dynamic_edge_matrices, self.dynamic_edge_inverses,
        self.dynamic_edge_status, self.dynamic_edge_status,
        np.int32(padded), np.float64(self.pivot_tolerance),
        self.inverse_failure, block=(self.threads_per_block, 1, 1),
        grid=(count, 1, 1), shared=(2 * padded * padded + padded) * 8,
        stream=stream,
      )

  def _numeric_rebuild_graph(self) -> CapturedGraph:
    descriptor_key = tuple(
      (
        0 if metadata.values.gpudata is None
        else int(metadata.values.gpudata),
        0 if metadata.positions.gpudata is None
        else int(metadata.positions.gpudata),
        0 if metadata.device_counts.gpudata is None
        else int(metadata.device_counts.gpudata),
        0 if metadata.device_starts.gpudata is None
        else int(metadata.device_starts.gpudata),
        0 if metadata.device_position_offsets.gpudata is None
        else int(metadata.device_position_offsets.gpudata),
        0 if metadata.device_shapes.gpudata is None
        else int(metadata.device_shapes.gpudata),
        int(metadata.counts.size),
        0 if not self.base_mixed_spmv else int(
          self.spmv_value_buffers[index].gpudata
        ),
      )
      for index, metadata in enumerate(self.metadata)
    )
    key = (
      descriptor_key, int(self.dynamic_block_launch_capacity),
      tuple(tuple(map(int, shape)) for shape in self.metadata[1].shapes),
      bool(self.dynamic_edge_domains_active),
    )
    graph = self._numeric_rebuild_graphs.get(key)
    if graph is None:
      graph = CapturedGraph.capture(
        self._pcg_stream,
        lambda: self._submit_numeric_assembly(self._pcg_stream),
      )
      self._numeric_rebuild_graphs[key] = graph
    return graph

  def update_numeric(
    self, view: BlockSparseMatrixView, *,
    rebuild_preconditioner: bool = True,
  ) -> None:
    """Recompute current collision collapse/local inverses without static rebuild."""
    if self._numeric_update_pending:
      self._finalize_numeric_update()
    update_started = perf_counter()
    if view.layout_signature != self.view.layout_signature:
      raise ValueError("CUDA runtime cannot change the variable layout")

    def current_operator_signature():
      def part(name, index):
        counts = tuple(map(int, to_host(
          getattr(view, f"{name}_category_counts"), np.uint64
        ).reshape(-1)))
        starts = tuple(map(int, to_host(
          getattr(view, f"{name}_category_starts"), np.uint64
        ).reshape(-1)))
        shapes = tuple(tuple(map(int, shape)) for shape in to_host(
          getattr(view, f"{name}_block_dimensions"), np.uint32
        ).reshape((-1, 2)))
        positions = getattr(view, f"{name}_positions")
        values = getattr(view, f"{name}_values")
        return (
          counts, starts, shapes,
          0 if getattr(positions, "gpudata", None) is None
          else int(positions.gpudata),
          0 if getattr(values, "gpudata", None) is None
          else int(values.gpudata),
          0 if not self.base_mixed_spmv else int(
            self.spmv_value_buffers[index].gpudata
          ),
        )
      static_part = part("static", 0)
      dynamic_part = part("dynamic", 1)
      dynamic_count = sum(dynamic_part[0])
      dynamic_active = bool(
        self.dynamic_edge_domains and dynamic_count and
        dynamic_count / max(1, self.fine_node_count) >=
        self.dynamic_edge_block_ratio
      )
      return (static_part, dynamic_part) + (
        int(self.dynamic_edge_inverses.gpudata),
        int(self.dynamic_edge_capacity),
        int(dynamic_active),
      )

    # YASPS updates numerical values in persistent GPU buffers. If their
    # pointers, active counts, starts, and shapes are unchanged, every
    # captured SpMV node already addresses the current operator. A lagged
    # preconditioner therefore needs no metadata upload or recompilation.
    if (not rebuild_preconditioner and
        current_operator_signature() == self._spmv_launch_signature):
      self.view = view
      conversion_graph = self._mixed_spmv_conversion_graph()
      if conversion_graph is not None:
        conversion_graph.launch()
      self.numeric_assembly_seconds = perf_counter() - update_started
      self.local_assembly_seconds = 0.0
      self.inverse_seconds = 0.0
      self.numeric_update_wall_seconds = self.numeric_assembly_seconds
      self.spmv_seconds = 0.0
      self.mas_seconds = 0.0
      self.mas_applications = 0
      return
    self.view = view
    numeric_started = perf_counter()
    static_metadata = (
      self._refresh_static_metadata()
      if self.metadata else self._part_metadata("static")
    )
    self.metadata = [static_metadata, self._part_metadata("dynamic")]
    self._ensure_mixed_spmv_buffers()
    self.spmv_primary_shape = self._select_primary_spmv_shape()
    self._refresh_dense_spmv_auxiliary_descriptors()
    self.dynamic_block_count = int(self.metadata[1].counts.sum(dtype=np.uint64))
    fine_weight = float(self.level_weights[0])
    dynamic_ratio = self.dynamic_block_count / max(1, self.fine_node_count)
    self.dynamic_edge_domains_active = bool(
      self.dynamic_edge_domains and self.dynamic_block_count and
      dynamic_ratio >= self.dynamic_edge_block_ratio
    )
    # Collision patches are appended after the fused multilevel collect,
    # so only an active patch solve needs the standalone r^Tz reduction.
    # The active bit below is also part of the PCG graph signature.
    self.fuses_pcg_dots = not self.dynamic_edge_domains_active
    if (self.adaptive_fine_level_weight
        and dynamic_ratio >= self.dense_collision_block_ratio):
      fine_weight = max(
        fine_weight, self.dense_collision_fine_level_weight
      )
    self.active_fine_level_weight = fine_weight
    dynamic = self.metadata[1]
    if self.dynamic_edge_domains:
      chunk_sizes = np.asarray([
        max(1, self.dynamic_edge_padded_size // int(rows + cols))
        for rows, cols in dynamic.shapes
      ], dtype=np.uint32)
      group_counts = np.asarray([
        (int(count) + int(chunk) - 1) // int(chunk)
        for count, chunk in zip(dynamic.counts, chunk_sizes)
      ], dtype=np.uint64)
      group_offsets = np.cumsum(
        np.r_[np.uint64(0), group_counts], dtype=np.uint64
      )
      self.dynamic_group_count = int(group_offsets[-1])
      self.dynamic_group_offsets = self._upload(
        "dynamic_group_offsets", group_offsets, np.uint64,
        skip_unchanged=True,
      )
      self.dynamic_group_chunk_sizes = self._upload(
        "dynamic_group_chunk_sizes", chunk_sizes, np.uint32,
        skip_unchanged=True,
      )
    else:
      # Static MAS does not consume collision-patch grouping metadata.
      # Avoid constructing and comparing those host arrays on every
      # solve; the dynamic Hessian blocks are still assembled exactly.
      self.dynamic_group_count = 0
    if self.dynamic_block_count > self.dynamic_block_launch_capacity:
      self.dynamic_block_launch_capacity = max(
        int(np.ceil(
          self.dynamic_block_count * self.upload_growth_factor
        )),
        int(np.ceil(
          max(1, self.dynamic_block_launch_capacity)
          * self.upload_growth_factor
        )),
      )
    if self.dynamic_edge_domains_active:
      self._ensure_dynamic_edge_capacity(self.dynamic_group_count)
    # CUDA graph kernel parameters include every scalar launch argument and
    # device pointer. Numerical values may change in place without a new
    # graph; a resized/replaced category buffer or changed category layout
    # intentionally selects a different cached graph.
    self._spmv_launch_signature = tuple(
      (
        tuple(map(int, metadata.counts)),
        tuple(map(int, metadata.starts)),
        tuple(tuple(map(int, shape)) for shape in metadata.shapes),
        0 if metadata.positions.gpudata is None else int(metadata.positions.gpudata),
        0 if metadata.values.gpudata is None else int(metadata.values.gpudata),
        0 if not self.base_mixed_spmv else int(
          self.spmv_value_buffers[index].gpudata
        ),
      )
      for index, metadata in enumerate(self.metadata)
    ) + (
      int(self.dynamic_edge_inverses.gpudata),
      int(self.dynamic_edge_capacity),
      int(self.dynamic_edge_domains_active),
    )
    static, dynamic = self.metadata
    dynamic_shapes = tuple(
      tuple(map(int, shape)) for shape in dynamic.shapes
    )
    active_dynamic_shapes = {
      shape for count, shape in zip(dynamic.counts, dynamic_shapes)
      if int(count)
    }
    static_shapes = {
      tuple(map(int, shape)) for count, shape in zip(
        static.counts, static.shapes
      ) if int(count)
    }
    device_count_graph = bool(
      self.spmv_primary_shape in static_shapes
      and all(
        shape == self.spmv_primary_shape
        or shape in self.spmv_fused_auxiliary_shapes
        for shape in active_dynamic_shapes
      )
    )
    if device_count_graph:
      self._pcg_spmv_launch_signature = (
        "device-count-pair",
        tuple(map(int, static.counts)),
        tuple(map(int, static.starts)),
        tuple(tuple(map(int, shape)) for shape in static.shapes),
        int(static.positions.gpudata), int(static.values.gpudata),
        dynamic_shapes,
        0 if dynamic.positions.gpudata is None else
        int(dynamic.positions.gpudata),
        0 if dynamic.values.gpudata is None else
        int(dynamic.values.gpudata),
        0 if dynamic.device_counts.gpudata is None else
        int(dynamic.device_counts.gpudata),
        0 if dynamic.device_starts.gpudata is None else
        int(dynamic.device_starts.gpudata),
        0 if dynamic.device_position_offsets.gpudata is None else
        int(dynamic.device_position_offsets.gpudata),
        int(self.dynamic_block_launch_capacity),
        0 if not self.base_mixed_spmv else int(
          self.spmv_value_buffers[0].gpudata
        ),
        0 if not self.base_mixed_spmv else int(
          self.spmv_value_buffers[1].gpudata
        ),
        int(self.spmv_auxiliary_descriptor_count),
        int(self.spmv_auxiliary_launch_capacity),
        tuple(self.spmv_fused_auxiliary_shapes),
        float(self.active_fine_level_weight),
        bool(self.dynamic_edge_domains_active),
      )
    else:
      self._pcg_spmv_launch_signature = (
        self._spmv_launch_signature,
        float(self.active_fine_level_weight),
      )
    self.numeric_assembly_seconds = perf_counter() - numeric_started
    self._ensure_spmv_specializations()
    self._get_specialized_static_assembly_kernel()
    self._get_specialized_dynamic_assembly_kernel()
    self.spmv_seconds = 0.0
    self.mas_seconds = 0.0
    self.mas_applications = 0

    if not rebuild_preconditioner:
      # The current operator metadata/pointers are now live for SpMV,
      # while the preceding SPD local inverses remain a valid PCG
      # preconditioner.  Charge the descriptor refresh to this solve,
      # but correctly report zero assembly/inversion work.
      self.local_assembly_seconds = 0.0
      self.inverse_seconds = 0.0
      conversion_graph = self._mixed_spmv_conversion_graph()
      if conversion_graph is not None:
        conversion_graph.launch()
      self.numeric_update_wall_seconds = self.numeric_assembly_seconds
      return

    self._assembly_start_event.record(self._pcg_stream)
    if self.dynamic_edge_domains_active:
      self._submit_numeric_assembly(self._pcg_stream)
      self._assembly_end_event.record(self._pcg_stream)
      self._submit_numeric_inverse(self._pcg_stream)
    else:
      # Python traverses hierarchy levels and inverse buckets only while
      # capturing a new immutable pointer/capacity signature. The hot
      # path below is one assembly graph plus one inverse graph launch,
      # with a normal CUDA event between them for separate statistics.
      self._numeric_rebuild_graph().launch()
      self._assembly_end_event.record(self._pcg_stream)
      self._numeric_inverse_graph.launch()
    self._inverse_end_event.record(self._pcg_stream)
    self.cuda.memcpy_dtoh_async(
      self._numeric_host_status, self.status.gpudata, self._pcg_stream
    )
    self._numeric_update_submit_seconds = perf_counter() - update_started
    self._numeric_update_pending = True

  def _finalize_numeric_update(self, *, synchronize: bool = True) -> None:
    """Validate an enqueued rebuild after an existing stream barrier."""
    if not self._numeric_update_pending:
      return
    if synchronize:
      self._pcg_stream.synchronize()
    status_values = self._numeric_host_status
    status_code = int(status_values[0])
    if status_code:
      messages = {
        1: "a dynamic block references an unknown scalar boundary",
        3: "a dynamic block shape does not match its endpoint dimensions",
      }
      raise ValueError(messages.get(status_code, f"device assembly failed with status {status_code}"))
    failures = []
    if int(status_values[1]):
      for bucket in self.inverse_buckets:
        failed = np.flatnonzero(bucket.status.get())
        failures.extend(int(bucket.domains[index]) for index in failed)
      edge_failures = np.flatnonzero(
        self.dynamic_edge_status[:self.dynamic_group_count].get()
      )
      if edge_failures.size:
        raise LocalInverseError(
          "CUDA dynamic collision-edge inversion failed for blocks "
          f"{edge_failures.tolist()}"
        )
    if failures:
      raise LocalInverseError(
        f"CUDA {self.inverse_algorithm} inversion failed for global domains {failures}"
      )
    self.local_assembly_seconds = (
      self._assembly_start_event.time_till(
        self._assembly_end_event
      ) * 1e-3
    )
    self.inverse_seconds = (
      self._assembly_end_event.time_till(
        self._inverse_end_event
      ) * 1e-3
    )
    self.numeric_rebuild_count += 1
    # Host uploads/descriptors precede the GPU event range. Summing that
    # measured setup with the assembly/inverse event interval reports the
    # actual construction work without charging subsequent PCG setup that
    # may share the same end-of-initialization synchronization.
    self.numeric_update_wall_seconds = (
      self.numeric_assembly_seconds
      + self.local_assembly_seconds + self.inverse_seconds
    )
    self._numeric_update_pending = False

  def _launch_mapped_spmv(
    self, vector, output, level_index: int, stream=None,
    curvature_output=None, *, full_precision: bool = False,
  ) -> None:
    curvature = (
      np.uintp(0) if curvature_output is None else curvature_output
    )
    categories: dict[
      tuple[int, int], list[
        tuple[DeviceInputPart, object, int, int, int, int, int]
      ]
    ] = {}
    value_buffers = (
      self.spmv_value_buffers
      if self.base_mixed_spmv and not full_precision else
      [metadata.values for metadata in self.metadata]
    )
    for part_index, (metadata, values) in enumerate(zip(
      self.metadata, value_buffers
    )):
      position_start = 0
      for category_index, (count_raw, value_start, shape_raw) in enumerate(zip(
        metadata.counts, metadata.starts, metadata.shapes
      )):
        count = int(count_raw)
        rows, cols = map(int, shape_raw)
        if count or (
          not full_precision and level_index == 0
          and part_index == 1
          and (rows, cols) == self.spmv_primary_shape
          and self.dynamic_block_launch_capacity
        ):
          categories.setdefault((rows, cols), []).append(
            (
              metadata, values, int(value_start), position_start,
              count, part_index, category_index,
            )
          )
        position_start += count

    fused_auxiliary = bool(
      not full_precision
      and level_index == 0
      and self.spmv_auxiliary_descriptor_count
      and self.spmv_auxiliary_total_count
      and self.spmv_primary_shape in categories
    )
    fused_shapes = set(
      self.spmv_fused_auxiliary_shapes if fused_auxiliary else ()
    )
    for (rows, cols), entries in categories.items():
      if (rows, cols) in fused_shapes:
        continue
      kernel = self._get_specialized_spmv_kernel(
        rows, cols, full_precision=full_precision
      )
      cooperative = rows * cols >= 64
      auxiliary = (
        (
          self.spmv_auxiliary_descriptors,
          np.uint32(self.spmv_auxiliary_descriptor_count),
          np.uint32(self.spmv_auxiliary_launch_capacity),
        )
        if fused_auxiliary and (rows, cols) == self.spmv_primary_shape
        else (np.uintp(0), np.uint32(0), np.uint32(0))
      )
      if len(entries) == 2 and not cooperative:
        first, second = entries
        device_count_pair = bool(
          level_index == 0 and not full_precision
          and first[5] == 0 and second[5] == 1
          and (rows, cols) == self.spmv_primary_shape
        )
        count = first[4] + (
          self.dynamic_block_launch_capacity
          if device_count_pair else second[4]
        )
        spmv_threads = self._spmv_threads(rows, cols)
        pair_kernel = (
          (
            self.spmv_full_fine_pair_kernels[(rows, cols)]
            if full_precision else
            self.spmv_fine_pair_kernels[(rows, cols)]
          ) if level_index == 0 else (
            self.spmv_full_pair_kernels[(rows, cols)]
            if full_precision else
            self.spmv_pair_kernels[(rows, cols)]
          )
        )
        common = (
          first[1], np.uint64(first[2]), first[0].positions,
          np.uint64(first[3]), np.uint32(first[4]),
          second[1], np.uint64(second[2]), second[0].positions,
          np.uint64(second[3]), np.uint32(second[4]),
        )
        mapped = (
          self.boundary_to_node, self.fine_node_scalar_starts,
          np.uint32(level_index), np.uint32(self.fine_node_count),
        )
        live_dynamic = (
          second[0].device_counts, second[0].device_starts,
          second[0].device_position_offsets,
          np.uint32(second[6]),
          np.uint32(self.dynamic_block_launch_capacity),
        ) if device_count_pair else (
          np.uintp(0), np.uintp(0), np.uintp(0),
          np.uint32(0), np.uint32(0),
        )
        pair_kernel(
          *common,
          *(live_dynamic if level_index == 0 else mapped),
          vector, output,
          *(
            auxiliary
            if level_index == 0 else ()
          ),
          curvature,
          block=(spmv_threads, 1, 1),
          grid=((count + spmv_threads - 1) // spmv_threads, 1, 1),
          stream=stream,
        )
        continue
      for (
        metadata, values, value_start, position_start, count, _, _,
      ) in entries:
        spmv_threads = self._spmv_threads(
          rows, cols, cooperative=cooperative
        )
        selected_kernel = (
          (
            self.spmv_full_fine_kernels[(rows, cols)]
            if full_precision else
            self.spmv_fine_kernels[(rows, cols)]
          )
          if level_index == 0 and not cooperative else kernel
        )
        prefix = (
          values, np.uint64(value_start), metadata.positions,
          np.uint64(position_start), np.uint32(count),
        )
        mapped = (
          self.boundary_to_node, self.fine_node_scalar_starts,
          np.uint32(level_index), np.uint32(self.fine_node_count),
        )
        selected_kernel(
          *prefix, *( () if level_index == 0 and not cooperative
                else mapped), vector, output,
          *(
            auxiliary
            if level_index == 0 and not cooperative else ()
          ),
          curvature,
          block=(spmv_threads, 1, 1),
          grid=(((count + 7) // 8 if cooperative else
             (count + spmv_threads - 1) // spmv_threads), 1, 1),
          stream=stream,
        )

  def matvec(
    self, vector, level_index: int = 0, *, reuse_workspace: bool = False,
    stream=None, clear_output: bool = True, curvature_output=None,
    full_precision: bool = False,
  ):
    started = perf_counter()
    level = self.hierarchy.levels[level_index]
    x = self._as_device(vector, np.float64)
    if int(x.size) != level.number_of_scalar_dofs:
      raise ValueError("device matvec input has the wrong scalar size")
    if reuse_workspace and level_index == 0:
      output = self._matvec_output
      if clear_output:
        output.fill(0.0, stream=stream)
    else:
      output = self._zeros(level.number_of_scalar_dofs, np.float64)
    launch_options = (
      {"full_precision": True} if full_precision else {}
    )
    self._launch_mapped_spmv(
      x, output, level_index, stream=stream,
      curvature_output=curvature_output, **launch_options,
    )
    self.spmv_seconds += perf_counter() - started
    return output

  def _as_device(self, value, dtype):
    dtype = np.dtype(dtype)
    if is_pycuda_array(value):
      result = value if np.dtype(value.dtype) == dtype else value.astype(dtype)
      return result if int(result.size) == 0 else result.reshape(-1)
    return self._to_gpu(np.asarray(value, dtype=dtype).reshape(-1))

  def precondition(
    self, fine_residual, *, reuse_workspace: bool = False, stream=None,
    clear_workspace: bool = True, dot_output=None,
    fused_solution_update=None,
  ):
    started = perf_counter()
    fine = self._as_device(fine_residual, np.float64)
    if int(fine.size) != self.fine_dofs:
      raise ValueError("device residual has the wrong scalar size")
    if reuse_workspace:
      packed_residual = self.packed_residual
      packed_correction = self.packed_correction
      final = self._preconditioned_output
    else:
      packed_residual = self._empty(self.packed_vector_size, np.float32)
      packed_correction = self._empty(self.packed_vector_size, np.float32)
      final = self._empty(self.fine_dofs, np.float64)
    large_problem = self.fine_dofs >= 200_000
    restriction_threads = 256
    if clear_workspace:
      if self.fused_fine_domain_apply:
        if self.packed_vector_size > self.fine_dofs:
          packed_residual[self.fine_dofs:].fill(
            0.0, stream=stream
          )
      else:
        packed_residual.fill(0.0, stream=stream)
    restriction_grid = (
      (self.fine_node_count + restriction_threads - 1) //
      restriction_threads, 1, 1
    )
    if fused_solution_update is None:
      restriction_kernel = (
        self.restrict_coarse_warp_nodes_mixed_specialized_kernel
        if self.fused_fine_domain_apply else
        self.restrict_warp_nodes_mixed_specialized_kernel
      )
      restriction_kernel(
        fine, packed_residual, self.restriction_order,
        self.specialized_restriction_offsets,
        self.fine_node_scalar_offsets,
        self.fine_dimensions,
        np.uint32(self.fine_node_count),
        block=(restriction_threads, 1, 1), grid=restriction_grid,
        stream=stream,
      )
    else:
      solution, direction, product, state = fused_solution_update
      update_restriction_kernel = (
        self.update_restrict_coarse_warp_nodes_mixed_specialized_kernel
        if self.fused_fine_domain_apply else
        self.update_restrict_warp_nodes_mixed_specialized_kernel
      )
      update_restriction_kernel(
        solution, direction, fine, product, state, packed_residual,
        self.restriction_order, self.specialized_restriction_offsets,
        self.fine_node_scalar_offsets, self.fine_dimensions,
        np.uint32(self.fine_node_count),
        block=(restriction_threads, 1, 1), grid=restriction_grid,
        stream=stream,
      )
    coarse_domain_count = self.domain_count - self.fine_domain_count
    if self.fused_fine_domain_apply and coarse_domain_count:
      first = self.fine_domain_count
      self.inverse_apply_warp_domains_kernel(
        self.mixed_inverses, packed_residual, packed_correction,
        self.matrix_offsets[first:], self.vector_offsets[first:],
        self.sizes[first:], self.padded_sizes[first:],
        np.uint32(coarse_domain_count), block=(128, 1, 1),
        grid=((coarse_domain_count + 3) // 4, 1, 1), stream=stream,
      )
    elif (not self.fused_fine_domain_apply and
       self.use_subwarp_inverse_apply):
      self.inverse_apply_subwarp_domains_kernel(
        self.mixed_inverses, packed_residual, packed_correction,
        self.matrix_offsets, self.vector_offsets, self.sizes,
        self.padded_sizes, np.uint32(self.domain_count),
        block=(128, 1, 1), grid=(self.domain_count, 1, 1),
        stream=stream,
      )
    elif (not self.fused_fine_domain_apply and large_problem and
       self.maximum_fine_dimension >= 8):
      self.inverse_apply_mixed_cooperative_kernel(
        self.mixed_inverses, packed_residual, packed_correction,
        self.matrix_offsets, self.vector_offsets, self.sizes,
        self.padded_sizes, np.uint32(self.domain_count),
        block=(128, 1, 1), grid=(self.domain_count, 1, 1),
        shared=self.maximum_padded_size * np.dtype(np.float32).itemsize,
        stream=stream,
      )
    elif not self.fused_fine_domain_apply:
      self.inverse_apply_warp_domains_kernel(
        self.mixed_inverses, packed_residual, packed_correction,
        self.matrix_offsets, self.vector_offsets, self.sizes,
        self.padded_sizes, np.uint32(self.domain_count),
        block=(128, 1, 1),
        grid=((self.domain_count + 3) // 4, 1, 1), stream=stream,
      )
    if self.fused_fine_domain_apply:
      self.fine_inverse_apply_warp_domains_kernel(
        self.mixed_inverses, fine, final,
        self.level0_packed_to_fine, self.matrix_offsets,
        self.vector_offsets, self.sizes, self.padded_sizes,
        np.uint32(self.fine_domain_count),
        np.float32(self.active_fine_level_weight),
        block=(128, 1, 1),
        grid=((self.fine_domain_count + 3) // 4, 1, 1), stream=stream,
      )
    if dot_output is None:
      if self.fused_fine_domain_apply:
        self.collect_coarse_nodes_mixed_specialized_kernel(
          packed_correction, final,
          self.specialized_collection_offsets,
          self.fine_node_scalar_offsets, self.fine_dimensions,
          self.fine_node_level_active,
          np.uint32(self.fine_node_count), block=(512, 1, 1),
          grid=((self.fine_node_count + 511) // 512, 1, 1),
          stream=stream,
        )
      else:
        self.collect_nodes_mixed_specialized_kernel(
          packed_correction, final,
          self.specialized_collection_offsets,
          self.fine_node_scalar_offsets, self.fine_dimensions,
          self.fine_node_level_active,
          np.uint32(self.fine_node_count),
          np.float32(self.active_fine_level_weight),
          block=(512, 1, 1),
          grid=((self.fine_node_count + 511) // 512, 1, 1),
          stream=stream,
        )
    else:
      if self.fused_fine_domain_apply:
        self.collect_coarse_nodes_mixed_specialized_dots_kernel(
          packed_correction, final, fine, self._pcg_state,
          self.specialized_collection_offsets,
          self.fine_node_scalar_offsets, self.fine_dimensions,
          self.fine_node_level_active,
          np.uint32(self.fine_node_count), block=(512, 1, 1),
          grid=((self.fine_node_count + 511) // 512, 1, 1),
          stream=stream,
        )
      else:
        self.collect_nodes_mixed_specialized_dots_kernel(
          packed_correction, final, fine, self._pcg_state,
          self.specialized_collection_offsets,
          self.fine_node_scalar_offsets, self.fine_dimensions,
          self.fine_node_level_active,
          np.uint32(self.fine_node_count),
          np.float32(self.active_fine_level_weight),
          block=(512, 1, 1),
          grid=((self.fine_node_count + 511) // 512, 1, 1),
          stream=stream,
        )
    dynamic = self.metadata[1]
    if (self.dynamic_edge_domains_active and dynamic.block_count
        and self.view.symmetric_storage):
      self.apply_dynamic_groups_kernel(
        self.dynamic_edge_inverses,
        np.uint32(self.dynamic_edge_padded_size),
        self.dynamic_group_active_sizes,
        self.dynamic_group_scalar_indices,
        self.dynamic_group_scalar_nodes,
        self.dynamic_edge_node_counts,
        np.uint32(self.dynamic_group_count), fine, final,
        block=(32, 1, 1), grid=(self.dynamic_group_count, 1, 1),
        stream=stream,
      )
    self.mas_seconds += perf_counter() - started
    self.mas_applications += 1
    return final

  @staticmethod
  def _dot(left, right) -> float:
    import pycuda.gpuarray as gpuarray

    return float(gpuarray.dot(left, right).get())

  def full_precision_relative_residual(
    self, solution, right_hand_side,
  ) -> tuple[float, float]:
    """Audit a mixed solve against the original FP64 Hessian values."""
    rhs = self._as_device(right_hand_side, np.float64)
    product = self.matvec(
      solution, 0, reuse_workspace=True, stream=self._pcg_stream,
      full_precision=True,
    )
    self.residual_from_product_kernel(
      rhs, product, self._pcg_residual, np.uint32(self.fine_dofs),
      block=(256, 1, 1),
      grid=((self.fine_dofs + 255) // 256, 1, 1),
      stream=self._pcg_stream,
    )
    self._pcg_state[4:5].fill(0.0, stream=self._pcg_stream)
    reduction_grid = (min(64, (self.fine_dofs + 255) // 256), 1, 1)
    self.dot_single_kernel(
      self._pcg_residual, self._pcg_residual, self._pcg_state,
      np.uint32(4), np.uint32(self.fine_dofs), block=(256, 1, 1),
      grid=reduction_grid, stream=self._pcg_stream,
    )
    self.cuda.memcpy_dtoh_async(
      self._pcg_host_control, self._pcg_state.gpudata,
      self._pcg_stream,
    )
    self._pcg_stream.synchronize()
    residual_norm = float(np.sqrt(max(self._pcg_host_control[4], 0.0)))
    rhs_norm = float(np.sqrt(max(self._pcg_host_control[0], 0.0)))
    return residual_norm, residual_norm / max(
      rhs_norm, np.finfo(np.float64).tiny
    )

  def restrict_fine(self, fine_vector, level_index: int):
    fine = self._as_device(fine_vector, np.float64)
    level_dofs = self.hierarchy.levels[level_index].number_of_scalar_dofs
    result = self._zeros(level_dofs, np.float64)
    mapping = self.fine_to_level[
      level_index * self.fine_dofs : (level_index + 1) * self.fine_dofs
    ]
    self.restriction_kernel(
      fine, result, mapping, np.uint32(self.fine_dofs), block=(128, 1, 1),
      grid=((self.fine_dofs + 127) // 128, 1, 1),
    )
    return result

  def prolong_to_fine(self, level_vector, level_index: int):
    result = self._zeros(self.fine_dofs, np.float64)
    mapping = self.fine_to_level[
      level_index * self.fine_dofs : (level_index + 1) * self.fine_dofs
    ]
    self.prolongation_kernel(
      self._as_device(level_vector, np.float64), result, mapping,
      np.uint32(self.fine_dofs), block=(128, 1, 1),
      grid=((self.fine_dofs + 127) // 128, 1, 1),
    )
    return result

  def _submit_pcg_iteration(
    self, *, use_mas: bool, fixed_budget: bool, tolerance: float,
    reduction_grid, vector_grid, vector_bytes: int, stream,
  ) -> None:
    """Submit one recurrence without allocating or reading host state."""
    product = self.matvec(
      self._pcg_direction, 0, reuse_workspace=True, stream=stream,
      clear_output=False,
      curvature_output=np.uintp(
        int(self._pcg_state.gpudata) +
        2 * np.dtype(np.float64).itemsize
      ),
    )
    self.prepare_iteration_kernel(
      self._pcg_state, block=(1, 1, 1), grid=(1, 1, 1),
      stream=stream,
    )
    fused_preconditioner_dots = bool(
      use_mas and getattr(self, "fuses_pcg_dots", False)
    )
    fused_solution_update = bool(
      use_mas and getattr(
        self, "fuses_solution_update_restriction", False
      )
    )
    if fused_solution_update:
      self.precondition(
        self._pcg_residual, reuse_workspace=True, stream=stream,
        clear_workspace=False,
        dot_output=(
          np.uintp(
            int(self._pcg_state.gpudata) +
            3 * np.dtype(np.float64).itemsize
          )
          if fused_preconditioner_dots else None
        ),
        fused_solution_update=(
          self._pcg_solution, self._pcg_direction,
          product, self._pcg_state,
        ),
      )
    else:
      self._submit_solution_residual_update(
        product, vector_grid=vector_grid, stream=stream,
      )
      if use_mas:
        self.precondition(
          self._pcg_residual, reuse_workspace=True, stream=stream,
          clear_workspace=False,
          dot_output=(
            np.uintp(
              int(self._pcg_state.gpudata) +
              3 * np.dtype(np.float64).itemsize
            )
            if fused_preconditioner_dots else None
          ),
        )
      else:
        self.cuda.memcpy_dtod_async(
          self._preconditioned_output.gpudata,
          self._pcg_residual.gpudata,
          vector_bytes,
          stream,
        )
    if not fused_preconditioner_dots:
      self.dot_two_kernel(
        self._pcg_residual, self._preconditioned_output,
        self._pcg_residual, self._pcg_residual,
        self._pcg_state, np.uint32(3), np.uint32(4),
        np.uint32(self.fine_dofs),
        block=(256, 1, 1), grid=reduction_grid, stream=stream,
      )
    if fixed_budget:
      self.finish_iteration_unchecked_kernel(
        self._pcg_state, block=(1, 1, 1), grid=(1, 1, 1),
        stream=stream,
      )
    else:
      self.finish_iteration_kernel(
        self._pcg_state, block=(1, 1, 1), grid=(1, 1, 1),
        stream=stream,
      )
    self.update_direction_kernel(
      self._preconditioned_output, self._pcg_direction, self._pcg_state,
      self._matvec_output, self.packed_residual,
      np.uint32(self.fine_dofs), np.uint32(self.packed_vector_size),
      block=(256, 1, 1), grid=vector_grid, stream=stream,
    )

  def _submit_solution_residual_update(
    self, product, *, vector_grid, stream,
  ) -> None:
    self.update_solution_residual_kernel(
      self._pcg_solution, self._pcg_direction, self._pcg_residual,
      product, self._pcg_state, np.uint32(self.fine_dofs),
      block=(256, 1, 1), grid=vector_grid, stream=stream,
    )

  def _pcg_iteration_graph(
    self, *, use_mas: bool, fixed_budget: bool, tolerance: float,
    reduction_grid, vector_grid, vector_bytes: int,
  ) -> CapturedGraph:
    key = (
      "iteration", bool(use_mas), bool(fixed_budget),
      self._pcg_spmv_launch_signature,
    )
    graph = self._pcg_graphs.get(key)
    if graph is not None:
      self.pcg_graph_cache_hits += 1
      return graph
    started = perf_counter()
    graph = CapturedGraph.capture(
      self._pcg_stream,
      lambda: self._submit_pcg_iteration(
        use_mas=use_mas, fixed_budget=fixed_budget, tolerance=tolerance,
        reduction_grid=reduction_grid, vector_grid=vector_grid,
        vector_bytes=vector_bytes, stream=self._pcg_stream,
      ),
    )
    self._pcg_graphs[key] = graph
    self.pcg_graph_build_count += 1
    self.pcg_graph_build_seconds += perf_counter() - started
    return graph

  def _pcg_budget_graph(
    self, iteration_graph: CapturedGraph, iterations: int, *, use_mas: bool,
    tolerance: float, reduction_grid, vector_grid, vector_bytes: int,
  ):
    key = (
      "budget", int(iterations), id(iteration_graph),
      self._pcg_spmv_launch_signature,
    )
    graph = self._pcg_graphs.get(key)
    if graph is not None:
      self.pcg_graph_cache_hits += 1
      return graph
    started = perf_counter()
    # A graph-of-child-graphs is cheap to construct but adds measurable
    # scheduling overhead at every iteration. Fixed-budget solves are hot
    # enough to justify a one-time flat capture of the exact recurrence.
    def submit_budget():
      for _ in range(iterations):
        self._submit_pcg_iteration(
          use_mas=use_mas, fixed_budget=True, tolerance=tolerance,
          reduction_grid=reduction_grid, vector_grid=vector_grid,
          vector_bytes=vector_bytes, stream=self._pcg_stream,
        )

    graph = CapturedGraph.capture(self._pcg_stream, submit_budget)
    self._pcg_graphs[key] = graph
    self.pcg_graph_build_count += 1
    self.pcg_graph_build_seconds += perf_counter() - started
    return graph

  def _pcg_conditional_graph(self, iteration_graph: CapturedGraph) -> CapturedGraph:
    key = (
      "conditional", id(iteration_graph),
      self._pcg_spmv_launch_signature,
    )
    graph = self._pcg_graphs.get(key)
    if graph is not None:
      self.pcg_graph_cache_hits += 1
      return graph
    started = perf_counter()

    def capture_condition(handle: int) -> CapturedGraph:
      return CapturedGraph.capture(
        self._pcg_stream,
        lambda: self.update_pcg_loop_kernel(
          np.uint64(handle), self._pcg_state,
          block=(1, 1, 1), grid=(1, 1, 1), stream=self._pcg_stream,
        ),
      )

    graph = CapturedGraph.conditional_while(iteration_graph, capture_condition)
    self._pcg_graphs[key] = graph
    self.pcg_graph_build_count += 1
    self.pcg_graph_build_seconds += perf_counter() - started
    return graph

  def pcg(
    self,
    right_hand_side,
    *,
    level_index: int = 0,
    use_mas: bool = True,
    initial_guess=None,
    tolerance: float = 1e-3,
    max_iterations: int = 20_000,
  ) -> PCGResult:
    if tolerance <= 0 or max_iterations < 0:
      raise ValueError("tolerance must be positive and max_iterations non-negative")
    started = perf_counter()
    level = self.hierarchy.levels[level_index]
    rhs = self._as_device(right_hand_side, np.float64)
    if int(rhs.size) != level.number_of_scalar_dofs:
      raise ValueError("device right-hand side has the wrong scalar size")
    # The persistent recurrence buffers cover the fine solve, which is the
    # performance-critical path. Reduced coarse solves retain the simple
    # GPUArray implementation until they receive dedicated sized buffers.
    if level_index != 0:
      self._finalize_numeric_update()
      return self._pcg_legacy(
        rhs,
        level_index=level_index,
        use_mas=use_mas,
        initial_guess=initial_guess,
        tolerance=tolerance,
        max_iterations=max_iterations,
      )
    x = self._pcg_solution
    residual = self._pcg_residual
    direction = self._pcg_direction
    preconditioned = self._preconditioned_output
    if initial_guess is not None and int(self._as_device(initial_guess, np.float64).size) != int(rhs.size):
      raise ValueError("device initial guess has the wrong scalar size")
    initial_control = self._pcg_host_control
    initial_control.fill(0.0)
    initial_control[9] = tolerance
    initial_control[11] = max_iterations
    self._pcg_initial_start_event.record(self._pcg_stream)
    self.cuda.memcpy_htod_async(
      self._pcg_state.gpudata, initial_control, self._pcg_stream
    )
    vector_bytes = self.fine_dofs * np.dtype(np.float64).itemsize
    if initial_guess is None:
      x.fill(0.0, stream=self._pcg_stream)
      self.cuda.memcpy_dtod_async(
        residual.gpudata, rhs.gpudata, vector_bytes,
        self._pcg_stream,
      )
    else:
      guess = self._as_device(initial_guess, np.float64)
      self.cuda.memcpy_dtod_async(
        x.gpudata, guess.gpudata, vector_bytes, self._pcg_stream
      )
      product = self.matvec(
        x, 0, reuse_workspace=True, stream=self._pcg_stream
      )
      self.residual_from_product_kernel(
        rhs, product, residual, np.uint32(self.fine_dofs),
        block=(256, 1, 1), grid=((self.fine_dofs + 255) // 256, 1, 1),
        stream=self._pcg_stream,
      )

    reduction_grid = (min(64, (self.fine_dofs + 255) // 256), 1, 1)
    residual_is_rhs = initial_guess is None
    if not residual_is_rhs:
      if use_mas:
        self.packed_residual[self.fine_dofs :].fill(
          0.0, stream=self._pcg_stream
        )
        self.precondition(
          rhs, reuse_workspace=True, stream=self._pcg_stream,
          clear_workspace=True,
        )
        reference_vector = preconditioned
      else:
        reference_vector = rhs
      self.dot_single_kernel(
        rhs, reference_vector, self._pcg_state, np.uint32(12),
        np.uint32(self.fine_dofs), block=(256, 1, 1),
        grid=reduction_grid, stream=self._pcg_stream,
      )
    if use_mas:
      self.packed_residual[self.fine_dofs :].fill(
        0.0, stream=self._pcg_stream
      )
      self.precondition(
        residual, reuse_workspace=True, stream=self._pcg_stream,
        clear_workspace=True,
      )
    else:
      self.cuda.memcpy_dtod_async(
        preconditioned.gpudata, residual.gpudata, vector_bytes,
        self._pcg_stream,
      )
    self.initialize_recurrence_kernel(
      self._pcg_state, rhs, residual, preconditioned, direction,
      np.uint32(self.fine_dofs),
      np.uint32(residual_is_rhs),
      block=(256, 1, 1), grid=reduction_grid,
      stream=self._pcg_stream,
    )
    # The first iteration consumes the same persistent product workspace
    # that later iterations clear while updating their search direction.
    self._matvec_output.fill(0.0, stream=self._pcg_stream)
    if use_mas:
      self.packed_residual[self.fine_dofs :].fill(
        0.0, stream=self._pcg_stream
      )
    self._pcg_initial_end_event.record(self._pcg_stream)
    self.cuda.memcpy_dtoh_async(
      self._pcg_host_initial, self._pcg_state.gpudata,
      self._pcg_stream,
    )
    self._pcg_stream.synchronize()
    self._finalize_numeric_update(synchronize=False)
    initial_state = self._pcg_host_initial
    initial_seconds = self._pcg_initial_start_event.time_till(
      self._pcg_initial_end_event
    ) * 1e-3
    rhs_norm = float(np.sqrt(max(initial_state[0], 0.0)))
    denominator = max(rhs_norm, np.finfo(np.float64).tiny)
    residual_norm = float(np.sqrt(max(initial_state[4], 0.0)))
    rz = float(initial_state[1])
    reference_rz = float(initial_state[12])
    if (np.isfinite(rz) and rz >= 0.0 and
        np.isfinite(reference_rz) and reference_rz > 0.0 and
        rz <= tolerance * reference_rz):
      return PCGResult(
        x, 0, residual_norm, residual_norm / denominator, True,
        initial_seconds,
      )
    if max_iterations == 0:
      return PCGResult(
        x, 0, residual_norm, residual_norm / denominator, False,
        initial_seconds,
      )
    if (not np.isfinite(rz) or rz <= 0.0 or
        not np.isfinite(reference_rz) or reference_rz <= 0.0):
      return PCGResult(
        x, 0, residual_norm, residual_norm / denominator, False,
        initial_seconds, "preconditioner is not positive definite",
      )
    status_address = int(self._pcg_state.gpudata) + 7 * np.dtype(np.float64).itemsize
    fixed_budget = tolerance <= 1e-20
    vector_grid = ((self.fine_dofs + 255) // 256, 1, 1)
    iteration_started = perf_counter()
    iteration_graph = self._pcg_iteration_graph(
      use_mas=use_mas, fixed_budget=fixed_budget, tolerance=tolerance,
      reduction_grid=reduction_grid, vector_grid=vector_grid,
      vector_bytes=vector_bytes,
    )
    if fixed_budget:
      budget_graph = self._pcg_budget_graph(
        iteration_graph, max_iterations, use_mas=use_mas,
        tolerance=tolerance, reduction_grid=reduction_grid,
        vector_grid=vector_grid, vector_bytes=vector_bytes,
      )
      budget_graph.launch()
      self.cuda.memcpy_dtoh_async(
        self._pcg_host_status, status_address, self._pcg_stream
      )
      self._pcg_stream.synchronize()
      status, status_value = map(float, self._pcg_host_status)
      residual_norm = float(np.sqrt(max(status_value, 0.0)))
      breakdown = None
      if status < 0.0 or not np.isfinite(status_value):
        breakdown = "matrix or preconditioner is not positive definite"
      return PCGResult(
        x, max_iterations, residual_norm, residual_norm / denominator,
        False, initial_seconds + perf_counter() - iteration_started,
        breakdown,
      )

    conditional_graph = self._pcg_conditional_graph(iteration_graph)
    conditional_graph.launch()
    self.cuda.memcpy_dtoh_async(
      self._pcg_host_completion, status_address, self._pcg_stream
    )
    self._pcg_stream.synchronize()
    status, status_value, _, completed_raw = map(
      float, self._pcg_host_completion
    )
    completed = int(completed_raw)
    residual_norm = float(np.sqrt(max(status_value, 0.0)))
    if status < 0.0:
      return PCGResult(
        x, max(completed - 1, 0), residual_norm,
        residual_norm / denominator, False,
        initial_seconds + perf_counter() - iteration_started,
        "matrix or preconditioner is not positive definite",
      )
    if status > 0.0:
      return PCGResult(
        x, completed, residual_norm, residual_norm / denominator, True,
        initial_seconds + perf_counter() - iteration_started,
      )
    return PCGResult(
      x, completed, residual_norm, residual_norm / denominator, False,
      initial_seconds + perf_counter() - iteration_started,
    )

  def _pcg_legacy(
    self,
    rhs,
    *,
    level_index: int,
    use_mas: bool,
    initial_guess,
    tolerance: float,
    max_iterations: int,
  ) -> PCGResult:
    """Compatibility path for the optional reduced coarse solve."""
    started = perf_counter()
    x = (
      self._zeros(rhs.shape, rhs.dtype)
      if initial_guess is None
      else self._as_device(initial_guess, np.float64).copy()
    )
    residual = rhs - self.matvec(x, level_index)
    rhs_norm = self._dot(rhs, rhs) ** 0.5
    residual_norm = self._dot(residual, residual) ** 0.5
    denominator = max(rhs_norm, np.finfo(np.float64).tiny)
    reference_rz = rhs_norm * rhs_norm
    z = residual.copy()
    rz = self._dot(residual, z)
    if rz <= tolerance * reference_rz or max_iterations == 0:
      return PCGResult(
        x, 0, residual_norm, residual_norm / denominator,
        rz <= tolerance * reference_rz, perf_counter() - started,
      )
    direction = z.copy()
    for iteration in range(1, max_iterations + 1):
      product = self.matvec(direction, level_index)
      curvature = self._dot(direction, product)
      if not np.isfinite(curvature) or curvature <= 0.0:
        return PCGResult(
          x, iteration - 1, residual_norm, residual_norm / denominator,
          False, perf_counter() - started, "matrix is not positive definite",
        )
      alpha = rz / curvature
      x = x + alpha * direction
      residual = residual - alpha * product
      residual_norm = self._dot(residual, residual) ** 0.5
      z = residual.copy()
      next_rz = self._dot(residual, z)
      if next_rz <= tolerance * reference_rz:
        return PCGResult(
          x, iteration, residual_norm, residual_norm / denominator,
          True, perf_counter() - started,
        )
      direction = z + (next_rz / rz) * direction
      rz = next_rz
    return PCGResult(
      x, max_iterations, residual_norm, residual_norm / denominator, False,
      perf_counter() - started,
    )

  @property
  def upload_capacities(self) -> dict[str, int]:
    return {key: slot.capacity for key, slot in sorted(self._uploads.items())}

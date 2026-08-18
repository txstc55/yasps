"""Multilevel additive Schwarz construction and application."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .hierarchy import Hierarchy, HierarchyLevel
from .local_inverse import InverseBuildResult, LocalInverseBackend
from .numeric_assembly import NumericHierarchy


@dataclass
class PreconditionerBuild:
  level_inverses: list[list[np.ndarray]]
  inverse_results: list[InverseBuildResult]
  seconds: float

  @property
  def device_bytes(self) -> int:
    return sum(result.device_bytes for result in self.inverse_results)


class MASPreconditioner:
  def __init__(self, hierarchy: Hierarchy, numeric: NumericHierarchy, inverse_backend: LocalInverseBackend, coarsest_level_weight: float = 1.0, level_weights: tuple[float, ...] | None = None):
    self.hierarchy = hierarchy
    self.numeric = numeric
    self.inverse_backend = inverse_backend
    self.coarsest_level_weight = float(coarsest_level_weight)
    self.level_weights = (
      (1.0,) * max(0, len(hierarchy.levels) - 1)
      + (self.coarsest_level_weight,)
      if level_weights is None else tuple(map(float, level_weights))
    )
    if (len(self.level_weights) != len(hierarchy.levels)
        or any(not np.isfinite(weight) or weight <= 0
           for weight in self.level_weights)):
      raise ValueError(
        "level_weights must contain one finite positive value per level"
      )
    started = perf_counter()
    results = [
      inverse_backend.build(numeric_level.local_matrices, level.domain_scalar_sizes)
      for level, numeric_level in zip(hierarchy.levels, numeric.levels)
    ]
    self.build = PreconditionerBuild(
      [result.inverses for result in results], results, perf_counter() - started
    )
    self._apply_plans = [
      self._make_apply_plan(level, inverses)
      for level, inverses in zip(hierarchy.levels, self.build.level_inverses)
    ]
    fine_level = hierarchy.levels[0]
    self._composed_scalar_maps = []
    for level_index, level in enumerate(hierarchy.levels):
      scalar_map = np.empty(fine_level.number_of_scalar_dofs, dtype=np.int64)
      node_map = hierarchy.composed_node_maps[level_index]
      for node, parent in enumerate(node_map):
        dimension = int(fine_level.node_dimensions[node])
        fine_start = int(fine_level.node_scalar_offsets[node])
        level_start = int(level.node_scalar_offsets[parent])
        scalar_map[fine_start : fine_start + dimension] = np.arange(
          level_start, level_start + dimension, dtype=np.int64
        )
      self._composed_scalar_maps.append(scalar_map)
    self.application_seconds = 0.0
    self.application_count = 0

  @staticmethod
  def _make_apply_plan(level: HierarchyLevel, inverses: list[np.ndarray]):
    """Bucket equal-size domains for batched BLAS during repeated applies."""
    buckets: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for domain, local_offsets, scalar_size, inverse in zip(
      level.domains, level.domain_scalar_offsets, level.domain_scalar_sizes, inverses
    ):
      scalar_indices = np.empty(int(scalar_size), dtype=np.int64)
      for node_index, node in enumerate(domain):
        dimension = int(level.node_dimensions[node])
        local_start = int(local_offsets[node_index])
        level_start = int(level.node_scalar_offsets[node])
        scalar_indices[local_start : local_start + dimension] = np.arange(
          level_start, level_start + dimension, dtype=np.int64
        )
      buckets.setdefault(int(scalar_size), []).append((scalar_indices, inverse))
    return [
      (
        np.stack([entry[0] for entry in entries]),
        np.stack([entry[1] for entry in entries]),
      )
      for _, entries in sorted(buckets.items())
    ]

  @staticmethod
  def _apply_plan(residual: np.ndarray, level: HierarchyLevel, plan) -> np.ndarray:
    correction = np.zeros(level.number_of_scalar_dofs, dtype=np.float64)
    for scalar_indices, inverses in plan:
      local_residuals = residual[scalar_indices]
      local_corrections = (inverses @ local_residuals[..., None])[..., 0]
      correction[scalar_indices] = local_corrections
    return correction

  @staticmethod
  def _apply_level(
    residual: np.ndarray,
    level: HierarchyLevel,
    inverses: list[np.ndarray],
  ) -> np.ndarray:
    correction = np.zeros(level.number_of_scalar_dofs, dtype=np.float64)
    for domain, local_offsets, scalar_size, inverse in zip(
      level.domains, level.domain_scalar_offsets, level.domain_scalar_sizes, inverses
    ):
      local_residual = np.empty(int(scalar_size), dtype=np.float64)
      for node_index, node in enumerate(domain):
        dim = int(level.node_dimensions[node])
        source = int(level.node_scalar_offsets[node])
        destination = int(local_offsets[node_index])
        local_residual[destination : destination + dim] = residual[source : source + dim]
      local_correction = inverse @ local_residual
      for node_index, node in enumerate(domain):
        dim = int(level.node_dimensions[node])
        source = int(local_offsets[node_index])
        destination = int(level.node_scalar_offsets[node])
        correction[destination : destination + dim] += local_correction[source : source + dim]
    return correction

  def apply(self, residual: np.ndarray) -> np.ndarray:
    started = perf_counter()
    fine_residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if fine_residual.size != self.hierarchy.levels[0].number_of_scalar_dofs:
      raise ValueError("residual has the wrong scalar size")
    result = np.zeros_like(fine_residual)
    for level_index, (level, plan, scalar_map) in enumerate(
      zip(self.hierarchy.levels, self._apply_plans, self._composed_scalar_maps)
    ):
      level_residual = (
        fine_residual
        if level_index == 0
        else np.bincount(
          scalar_map,
          weights=fine_residual,
          minlength=level.number_of_scalar_dofs,
        )
      )
      level_correction = self._apply_plan(level_residual, level, plan)
      weight = self.level_weights[level_index]
      result += weight * (
        level_correction if level_index == 0
        else level_correction[scalar_map]
      )
    self.application_seconds += perf_counter() - started
    self.application_count += 1
    return result

  def dense_reference(self) -> np.ndarray:
    """Materialize only the preconditioner operator for small test diagnostics."""
    size = self.hierarchy.levels[0].number_of_scalar_dofs
    return np.column_stack([self.apply(np.eye(size)[:, column]) for column in range(size)])

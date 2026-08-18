"""Per-solve block reduction and local principal-matrix assembly."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .hierarchy import Hierarchy, HierarchyLevel
from .matrix_view import BlockSparseMatrixView


@dataclass
class NumericLevel:
  blocks: dict[tuple[int, int], np.ndarray]
  local_matrices: list[np.ndarray]
  assembly_seconds: float

  def matvec(self, vector: np.ndarray, level: HierarchyLevel) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64).reshape(-1)
    if x.size != level.number_of_scalar_dofs:
      raise ValueError("vector has the wrong scalar size")
    result = np.zeros_like(x)
    for (row_node, col_node), block in self.blocks.items():
      row = int(level.node_scalar_offsets[row_node])
      col = int(level.node_scalar_offsets[col_node])
      result[row : row + block.shape[0]] += block @ x[col : col + block.shape[1]]
    return result


@dataclass
class NumericHierarchy:
  levels: list[NumericLevel]
  coarse_assembly_seconds: float
  local_assembly_seconds: float


def _add_block(blocks: dict[tuple[int, int], np.ndarray], key: tuple[int, int], block: np.ndarray) -> None:
  existing = blocks.get(key)
  if existing is None:
    blocks[key] = np.array(block, dtype=np.float64, copy=True)
  else:
    existing += block


def _assemble_locals(level: HierarchyLevel, blocks: dict[tuple[int, int], np.ndarray]) -> list[np.ndarray]:
  results: list[np.ndarray] = []
  for domain, local_offsets, scalar_size in zip(
    level.domains, level.domain_scalar_offsets, level.domain_scalar_sizes
  ):
    matrix = np.zeros((int(scalar_size), int(scalar_size)), dtype=np.float64)
    node_to_local = {node: index for index, node in enumerate(domain)}
    for (row_node, col_node), block in blocks.items():
      if row_node not in node_to_local or col_node not in node_to_local:
        continue
      row = int(local_offsets[node_to_local[row_node]])
      col = int(local_offsets[node_to_local[col_node]])
      matrix[row : row + block.shape[0], col : col + block.shape[1]] += block
    results.append(matrix)
  return results


def assemble_numeric_hierarchy(view: BlockSparseMatrixView, hierarchy: Hierarchy) -> NumericHierarchy:
  coarse_started = perf_counter()
  level_blocks: list[dict[tuple[int, int], np.ndarray]] = [dict() for _ in hierarchy.levels]
  for fine_row, fine_col, block in view.iter_blocks("both"):
    for level_index, mapping in enumerate(hierarchy.composed_node_maps):
      row, col = int(mapping[fine_row]), int(mapping[fine_col])
      _add_block(level_blocks[level_index], (row, col), block)
      if view.symmetric_storage and fine_row != fine_col:
        _add_block(level_blocks[level_index], (col, row), block.T)
  coarse_seconds = perf_counter() - coarse_started

  local_started = perf_counter()
  numeric_levels = [
    NumericLevel(blocks, _assemble_locals(level, blocks), 0.0)
    for level, blocks in zip(hierarchy.levels, level_blocks)
  ]
  local_seconds = perf_counter() - local_started
  for numeric in numeric_levels:
    numeric.assembly_seconds = local_seconds / max(1, len(numeric_levels))
  return NumericHierarchy(numeric_levels, coarse_seconds, local_seconds)

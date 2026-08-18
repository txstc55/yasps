"""Static variable-block graph construction and compaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matrix_view import BlockSparseMatrixView


@dataclass(frozen=True)
class BlockGraph:
  node_count: int
  edges: tuple[tuple[int, int], ...]
  adjacency: tuple[tuple[int, ...], ...]
  xadj: np.ndarray
  adjncy: np.ndarray

  @classmethod
  def from_edges(cls, node_count: int, edges) -> "BlockGraph":
    clean = {
      (min(int(i), int(j)), max(int(i), int(j)))
      for i, j in edges
      if int(i) != int(j)
    }
    if any(i < 0 or j >= node_count for i, j in clean):
      raise ValueError("graph edge node is out of range")
    neighbors = [set() for _ in range(node_count)]
    for i, j in clean:
      neighbors[i].add(j)
      neighbors[j].add(i)
    adjacency = tuple(tuple(sorted(row)) for row in neighbors)
    xadj = np.zeros(node_count + 1, dtype=np.int64)
    for i, row in enumerate(adjacency):
      xadj[i + 1] = xadj[i] + len(row)
    adjncy = np.fromiter((j for row in adjacency for j in row), dtype=np.int64)
    return cls(node_count, tuple(sorted(clean)), adjacency, xadj, adjncy)

  @classmethod
  def from_static_view(cls, view: BlockSparseMatrixView) -> "BlockGraph":
    return cls.from_edges(view.node_count, view.iter_block_coordinates("static"))

  def remap(self, fine_to_parent: np.ndarray, parent_count: int) -> "BlockGraph":
    return BlockGraph.from_edges(
      parent_count,
      ((fine_to_parent[i], fine_to_parent[j]) for i, j in self.edges),
    )

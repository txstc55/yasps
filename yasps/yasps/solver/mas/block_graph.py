"""Static variable-block graph construction and compaction."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from .matrix_view import BlockSparseMatrixView, to_host


class BlockSparsity:
  """Coordinates and per-block dimensions, with no numerical value storage.

  Both arrays contain exactly ``2 * num_blocks`` integers. Coordinates are
  global scalar starts, not variable IDs. Every variable must occur in at
  least one block (including isolated variables via their diagonal blocks),
  so its dimension and the complete contiguous DOF layout can be inferred.
  Duplicate/reversed coordinates are allowed; graph construction merges them.
  """

  def __init__(self, block_positions, block_dimensions, num_blocks):
    if isinstance(num_blocks, (bool, np.bool_)):
      raise TypeError("num_blocks must be an integer")
    num_blocks = index(num_blocks)
    if num_blocks < 0:
      raise ValueError("num_blocks must be non-negative")
    arrays = []
    for name, source in (("block_positions", block_positions), ("block_dimensions", block_dimensions)):
      values = to_host(source)
      if values.ndim != 1 or values.size != 2 * num_blocks:
        raise ValueError(f"{name} must be a flat array of length 2 * num_blocks")
      if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
      if values.size and (np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} contains an out-of-range index")
      arrays.append(values.astype(np.int64, copy=True))
    positions, sizes = arrays
    if np.any(sizes <= 0):
      raise ValueError("block dimensions must be positive")
    offsets, nodes = np.unique(positions, return_inverse=True)
    dimensions = np.zeros(offsets.size, dtype=np.int64)
    np.maximum.at(dimensions, nodes, sizes)
    if np.any(dimensions[nodes] != sizes):
      raise ValueError("inconsistent dimensions for the same variable offset")
    total_dofs = sum(map(int, dimensions))
    if total_dofs > np.iinfo(np.int64).max:
      raise ValueError("total variable dimensions exceed the supported index range")
    expected = np.zeros(offsets.size, dtype=np.int64)
    if offsets.size > 1:
      expected[1:] = np.cumsum(dimensions[:-1])
    if not np.array_equal(offsets, expected):
      raise ValueError("block coordinates must cover a contiguous variable layout")
    self.rows = self.cols = total_dofs
    self.variable_scalar_offsets = offsets
    self.variable_dimensions = dimensions
    self.variable_type_ids = None
    self.node_count = offsets.size
    self._block_nodes = nodes.reshape(-1, 2)

  def iter_block_coordinates(self, part="static"):
    if part not in ("static", "dynamic"):
      raise ValueError("unknown block part")
    if part == "static":
      yield from self._block_nodes

  def structure_signature(self):
    # Retained only as hierarchy metadata; rebuilds are explicit, not keyed
    # to this graph or to the numerical Hessian's static coordinates.
    return (self.rows, self.variable_dimensions.tobytes())


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

"""Static variable-block graph construction and compaction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
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

  def block_coordinates(self, part="static"):
    if part not in ("static", "dynamic"):
      raise ValueError("unknown block part")
    return self._block_nodes if part == "static" else np.empty((0, 2), dtype=np.int64)

  def iter_block_coordinates(self, part="static"):
    yield from self.block_coordinates(part)

  def structure_signature(self):
    # Retained only as hierarchy metadata; rebuilds are explicit, not keyed
    # to this graph or to the numerical Hessian's static coordinates.
    return (self.rows, self.variable_dimensions.tobytes())


@dataclass(frozen=True)
class BlockGraph:
  node_count: int
  xadj: np.ndarray
  adjncy: np.ndarray

  def __init__(self, node_count, edges, adjacency, xadj, adjncy):
    object.__setattr__(self, "node_count", node_count)
    object.__setattr__(self, "xadj", np.ascontiguousarray(xadj, dtype=np.int64))
    object.__setattr__(self, "adjncy", np.ascontiguousarray(adjncy, dtype=np.int64))
    if edges is not None:
      self.__dict__["edges"] = edges
    if adjacency is not None:
      self.__dict__["adjacency"] = adjacency

  @cached_property
  def edges(self) -> tuple[tuple[int, int], ...]:
    """Compatibility view; the solver itself only needs the compact CSR."""
    rows = np.repeat(np.arange(self.node_count), np.diff(self.xadj))
    upper = rows < self.adjncy
    node_ids = np.arange(self.node_count, dtype=object)
    return tuple(zip(node_ids[rows[upper]].tolist(), node_ids[self.adjncy[upper]].tolist()))

  @cached_property
  def adjacency(self) -> tuple[tuple[int, ...], ...]:
    node_ids = np.arange(self.node_count, dtype=object)
    neighbors = node_ids[self.adjncy].tolist()
    return tuple(tuple(neighbors[first:last]) for first, last in zip(self.xadj[:-1], self.xadj[1:]))

  @classmethod
  def from_edges(cls, node_count: int, edges) -> "BlockGraph":
    # Canonicalize and sort in arrays, rather than allocating a Python set
    # per node and a Python pair per raw (often repeated) block coordinate.
    pairs = np.asarray(edges, dtype=np.int64) if isinstance(edges, (np.ndarray, list, tuple)) else np.fromiter(edges, dtype=np.dtype((np.int64, 2)))
    if pairs.size and (pairs.ndim != 2 or pairs.shape[1] != 2):
      raise ValueError("graph edges must be node-index pairs")
    pairs = pairs.reshape(-1, 2)
    pairs = np.sort(pairs[pairs[:, 0] != pairs[:, 1]], axis=1)
    if pairs.size and (pairs.min() < 0 or pairs.max() >= node_count):
      raise ValueError("graph edge node is out of range")
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    if len(pairs):
      pairs = pairs[np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]]
    directed = np.concatenate((pairs, pairs[:, ::-1]))
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    xadj = np.r_[0, np.cumsum(np.bincount(directed[:, 0], minlength=node_count), dtype=np.int64)]
    adjncy = directed[:, 1].copy()
    del directed
    return cls(node_count, None, None, xadj, adjncy)

  @classmethod
  def from_static_view(cls, view: BlockSparseMatrixView) -> "BlockGraph":
    return cls.from_edges(view.node_count, view.block_coordinates("static"))

  def remap(self, fine_to_parent: np.ndarray, parent_count: int) -> "BlockGraph":
    rows = np.repeat(np.arange(self.node_count), np.diff(self.xadj))
    upper = rows < self.adjncy
    return BlockGraph.from_edges(
      parent_count,
      np.column_stack((fine_to_parent[rows[upper]], fine_to_parent[self.adjncy[upper]])),
    )

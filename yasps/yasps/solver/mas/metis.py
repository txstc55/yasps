"""Thin required-METIS connectivity-ordering wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .block_graph import BlockGraph


@dataclass(frozen=True)
class OrderingResult:
  order: np.ndarray
  groups: tuple[tuple[int, ...], ...]
  backend: str
  seconds: float


def _component_bfs_groups(graph: BlockGraph, nodes: list[int]) -> list[list[int]]:
  """Keep connected runs together inside one METIS partition."""
  allowed = set(nodes)
  remaining = set(nodes)
  groups: list[list[int]] = []
  while remaining:
    start = min(remaining)
    queue = [start]
    remaining.remove(start)
    cursor = 0
    while cursor < len(queue):
      node = queue[cursor]
      cursor += 1
      for neighbor in graph.adjacency[node]:
        if neighbor in allowed and neighbor in remaining:
          remaining.remove(neighbor)
          queue.append(neighbor)
    groups.append(queue)
  return groups


def metis_order(graph: BlockGraph, target_nodes_per_partition: int = 16) -> OrderingResult:
  start = perf_counter()
  if target_nodes_per_partition <= 0:
    raise ValueError("target_nodes_per_partition must be positive")
  if graph.node_count <= 1 or not graph.edges:
    order = np.arange(graph.node_count, dtype=np.int64)
    return OrderingResult(order, (tuple(map(int, order)),), "trivial", perf_counter() - start)
  try:
    import pymetis
  except ImportError as error:
    raise RuntimeError(
      "PyMetis/METIS is required for every nontrivial hierarchy graph; "
      "install the declared pymetis dependency"
    ) from error
  # Isolated variables have diagonal blocks, but no connectivity to partition.
  # Keep them as singleton domains rather than asking METIS to bisect them
  # again at every level. They remain in all DOF/transfer/numerical maps.
  degrees = np.diff(graph.xadj)
  active = np.flatnonzero(degrees)
  isolated = np.flatnonzero(degrees == 0)
  if isolated.size:
    compact = np.full(graph.node_count, -1, dtype=np.int64)
    compact[active] = np.arange(active.size)
    xadj = np.r_[0, np.cumsum(degrees[active])]
    adjncy = compact[graph.adjncy]
  else:
    xadj, adjncy = graph.xadj, graph.adjncy
  # Recursive partitioning produces locality-preserving groups.  The leaf
  # target is deliberately a node count, not a scalar-DOF weight: METIS must
  # see every variable block as one unweighted unit.  Real dimensions are
  # considered only by pack_domains after this ordering is returned.
  partition_count = max(1, (active.size + target_nodes_per_partition - 1) // target_nodes_per_partition)
  # Use the CSR we already built, without converting it to Python adjacency
  # lists and back. Older supported PyMetis versions take the arrays directly.
  csr = {"adjacency": pymetis.CSRAdjacency(xadj, adjncy)} if hasattr(pymetis, "CSRAdjacency") else {"xadj": xadj, "adjncy": adjncy}
  result = pymetis.part_graph(
    partition_count,
    recursive=True,
    **csr,
  )
  groups: dict[int, list[int]] = {}
  for node, partition in zip(active, result.vertex_part):
    groups.setdefault(int(partition), []).append(int(node))
  groups_in_graph_order = sorted(
    [
      component
      for group in groups.values()
      for component in _component_bfs_groups(graph, group)
    ] + [[int(node)] for node in isolated],
    key=min,
  )
  permutation = [
    node
    for group in groups_in_graph_order
    for node in group
  ]
  order = np.asarray(permutation, dtype=np.int64)
  if sorted(order.tolist()) != list(range(graph.node_count)):
    raise RuntimeError("METIS returned an invalid permutation")
  return OrderingResult(
    order,
    tuple(tuple(map(int, group)) for group in groups_in_graph_order),
    "pymetis-recursive-partition",
    perf_counter() - start,
  )

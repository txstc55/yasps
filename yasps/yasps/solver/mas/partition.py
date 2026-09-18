"""Real-scalar-capacity domain packing."""

from __future__ import annotations

from itertools import chain
from typing import Sequence

import numpy as np

from ._hierarchy_native import pack as _pack


def domain_arrays(domains):
  """Flatten disjoint domain lists without one NumPy allocation per domain."""
  counts = np.fromiter(map(len, domains), dtype=np.int64, count=len(domains))
  offsets = np.r_[0, np.cumsum(counts)]
  nodes = np.fromiter(chain.from_iterable(domains), dtype=np.int64, count=int(offsets[-1]))
  return nodes, offsets


def duplicate_domain_nodes(mapping, previous_node_domains, current_nodes, current_offsets):
  """Mark unchanged Schwarz domains, including multi-node domains.

  Every parent must have exactly one child, all children must belong to one
  previous domain, and their count must equal that whole previous domain.
  These conditions are equivalent to comparing the explicit child sets.
  """
  parent_count = current_nodes.size
  duplicate = np.zeros(parent_count, dtype=bool)
  if not parent_count or not mapping.size:
    return duplicate
  child_counts = np.bincount(mapping, minlength=parent_count)
  child = np.zeros(parent_count, dtype=np.int64)
  child[mapping] = np.arange(mapping.size)
  source = previous_node_domains[child[current_nodes]]
  starts = current_offsets[:-1]
  all_single = np.logical_and.reduceat(child_counts[current_nodes] == 1, starts)
  first = np.minimum.reduceat(source, starts)
  last = np.maximum.reduceat(source, starts)
  previous_counts = np.bincount(previous_node_domains)
  counts = np.diff(current_offsets)
  unchanged = all_single & (first == last) & (counts == previous_counts[first])
  duplicate[current_nodes] = np.repeat(unchanged, counts)
  return duplicate


def pack_domains(order: Sequence[int], node_dimensions: Sequence[int], max_domain_dofs: int) -> list[list[int]]:
  if max_domain_dofs <= 0:
    raise ValueError("max_domain_dofs must be positive")
  domains: list[list[int]] = []
  current_domain: list[int] = []
  current_dofs = 0
  seen: set[int] = set()
  for raw_node in order:
    node = int(raw_node)
    if node in seen or not 0 <= node < len(node_dimensions):
      raise ValueError("order must be a permutation of valid nodes")
    seen.add(node)
    node_dofs = int(node_dimensions[node])
    if node_dofs <= 0:
      raise ValueError("node dimensions must be positive")
    if current_domain and current_dofs + node_dofs > max_domain_dofs:
      domains.append(current_domain)
      current_domain = []
      current_dofs = 0
    # Oversized nodes are explicit singleton variable-size domains.
    current_domain.append(node)
    current_dofs += node_dofs
    if node_dofs > max_domain_dofs:
      domains.append(current_domain)
      current_domain = []
      current_dofs = 0
  if current_domain:
    domains.append(current_domain)
  if len(seen) != len(node_dimensions):
    raise ValueError("order does not contain every node")
  return domains


def pack_domain_groups(
  groups: Sequence[Sequence[int]],
  node_dimensions: Sequence[int],
  max_domain_dofs: int,
) -> list[list[int]]:
  """Pack each METIS connectivity group without crossing group boundaries."""
  flat, offsets = domain_arrays(groups)
  if not np.array_equal(np.sort(flat), np.arange(len(node_dimensions))):
    raise ValueError("partition groups must contain every node exactly once")
  dimensions = np.asarray(node_dimensions, dtype=np.int64)
  if np.any(dimensions <= 0):
    raise ValueError("node dimensions must be positive")
  return _pack(flat, offsets, np.ascontiguousarray(dimensions), max_domain_dofs)

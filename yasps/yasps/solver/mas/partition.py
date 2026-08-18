"""Real-scalar-capacity domain packing."""

from __future__ import annotations

from typing import Sequence


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
  flat = [int(node) for group in groups for node in group]
  if sorted(flat) != list(range(len(node_dimensions))):
    raise ValueError("partition groups must contain every node exactly once")
  domains: list[list[int]] = []
  for group in groups:
    # ``pack_domains`` validates full permutations, so use its exact greedy
    # rule locally while the global validation above owns uniqueness.
    current: list[int] = []
    current_dofs = 0
    for raw_node in group:
      node = int(raw_node)
      node_dofs = int(node_dimensions[node])
      if node_dofs <= 0:
        raise ValueError("node dimensions must be positive")
      if current and current_dofs + node_dofs > max_domain_dofs:
        domains.append(current)
        current = []
        current_dofs = 0
      current.append(node)
      current_dofs += node_dofs
      if node_dofs > max_domain_dofs:
        domains.append(current)
        current = []
        current_dofs = 0
    if current:
      domains.append(current)
  return domains

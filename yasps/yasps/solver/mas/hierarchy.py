"""One-time heterogeneous static hierarchy construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .block_graph import BlockGraph
from .matrix_view import BlockSparseMatrixView, to_host
from .metis import metis_order
from .partition import pack_domain_groups
from .transfer import TransferMap, make_transfer, prefix_offsets


@dataclass
class HierarchyLevel:
  index: int
  graph: BlockGraph
  node_dimensions: np.ndarray
  node_scalar_offsets: np.ndarray
  node_type_ids: np.ndarray | None
  domains: list[list[int]]
  domain_scalar_offsets: list[np.ndarray]
  domain_scalar_sizes: np.ndarray
  ordering: np.ndarray
  ordering_backend: str
  metis_seconds: float
  fine_to_parent: np.ndarray | None = None

  @property
  def number_of_nodes(self) -> int:
    return int(self.node_dimensions.size)

  @property
  def number_of_scalar_dofs(self) -> int:
    return int(self.node_dimensions.sum())


@dataclass
class Hierarchy:
  levels: list[HierarchyLevel]
  adjacent_maps: list[TransferMap]
  composed_node_maps: list[np.ndarray]
  build_seconds: float
  static_signature: tuple

  @property
  def number_of_levels(self) -> int:
    return len(self.levels)

  @property
  def nodes_per_level(self) -> list[int]:
    return [level.number_of_nodes for level in self.levels]

  @property
  def domains_per_level(self) -> list[int]:
    return [len(level.domains) for level in self.levels]

  def restrict_fine_to_level(self, vector: np.ndarray, level_index: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    fine = self.levels[0]
    if values.size != fine.number_of_scalar_dofs:
      raise ValueError("fine vector has the wrong scalar size")
    if level_index == 0:
      return values.copy()
    target = self.levels[level_index]
    result = np.zeros(target.number_of_scalar_dofs, dtype=np.float64)
    mapping = self.composed_node_maps[level_index]
    for node, parent in enumerate(mapping):
      dim = int(fine.node_dimensions[node])
      source = int(fine.node_scalar_offsets[node])
      destination = int(target.node_scalar_offsets[parent])
      result[destination : destination + dim] += values[source : source + dim]
    return result

  def prolong_level_to_fine(self, vector: np.ndarray, level_index: int) -> np.ndarray:
    target = self.levels[level_index]
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != target.number_of_scalar_dofs:
      raise ValueError("level vector has the wrong scalar size")
    if level_index == 0:
      return values.copy()
    fine = self.levels[0]
    result = np.empty(fine.number_of_scalar_dofs, dtype=np.float64)
    mapping = self.composed_node_maps[level_index]
    for node, parent in enumerate(mapping):
      dim = int(fine.node_dimensions[node])
      source = int(target.node_scalar_offsets[parent])
      destination = int(fine.node_scalar_offsets[node])
      result[destination : destination + dim] = values[source : source + dim]
    return result


def _domain_layout(domains: list[list[int]], dimensions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
  local_offsets, sizes = [], []
  for domain in domains:
    domain_dims = dimensions[np.asarray(domain, dtype=np.int64)]
    local_offsets.append(prefix_offsets(domain_dims))
    sizes.append(int(domain_dims.sum()))
  return local_offsets, np.asarray(sizes, dtype=np.int64)


def _collapse_compatible(
  graph: BlockGraph,
  domains: list[list[int]],
  dimensions: np.ndarray,
  type_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  count = graph.node_count
  parent = np.arange(count, dtype=np.int64)

  def find(node: int) -> int:
    while parent[node] != node:
      parent[node] = parent[parent[node]]
      node = int(parent[node])
    return node

  def union(left: int, right: int) -> None:
    a, b = find(left), find(right)
    if a != b:
      parent[max(a, b)] = min(a, b)

  node_domain = np.full(count, -1, dtype=np.int64)
  for domain_id, domain in enumerate(domains):
    node_domain[np.asarray(domain, dtype=np.int64)] = domain_id
  for left, right in graph.edges:
    compatible_type = type_ids is None or type_ids[left] == type_ids[right]
    if (
      node_domain[left] == node_domain[right]
      and dimensions[left] == dimensions[right]
      and compatible_type
    ):
      union(left, right)

  roots = [find(node) for node in range(count)]
  root_to_compact: dict[int, int] = {}
  mapping = np.empty(count, dtype=np.int64)
  for node, root in enumerate(roots):
    mapping[node] = root_to_compact.setdefault(root, len(root_to_compact))
  representative = np.empty(len(root_to_compact), dtype=np.int64)
  for root, compact in root_to_compact.items():
    representative[compact] = root
  coarse_dims = dimensions[representative].copy()
  coarse_types = None if type_ids is None else type_ids[representative].copy()
  return mapping, coarse_dims, coarse_types


def build_hierarchy(
  view: BlockSparseMatrixView,
  *,
  max_domain_dofs: int = 48,
  max_levels: int = 6,
  target_nodes_per_partition: int = 16,
  minimum_domains_for_next_level: int = 0,
  global_coarse_domain: bool = False,
  include_dynamic_edges: bool = False,
  domain_dof_schedule: tuple[int, ...] | None = None,
  target_node_schedule: tuple[int, ...] | None = None,
) -> Hierarchy:
  if max_levels <= 0:
    raise ValueError("max_levels must include at least level 0")
  if minimum_domains_for_next_level < 0:
    raise ValueError(
      "minimum_domains_for_next_level must be non-negative"
    )
  if (domain_dof_schedule is not None and
      (not domain_dof_schedule or any(
        int(value) <= 0 for value in domain_dof_schedule))):
    raise ValueError("domain DOF schedule must contain positive values")
  if (target_node_schedule is not None and
      (not target_node_schedule or any(
        int(value) <= 0 for value in target_node_schedule))):
    raise ValueError("target-node schedule must contain positive values")
  started = perf_counter()
  graph = (
    BlockGraph.from_edges(
      view.node_count,
      (*view.iter_block_coordinates("static"),
      *view.iter_block_coordinates("dynamic")),
    )
    if include_dynamic_edges else BlockGraph.from_static_view(view)
  )
  dimensions = to_host(view.variable_dimensions, np.int64).reshape(-1).copy()
  type_ids = None if view.variable_type_ids is None else to_host(view.variable_type_ids).reshape(-1).copy()
  levels: list[HierarchyLevel] = []
  transfers: list[TransferMap] = []
  composed = [np.arange(graph.node_count, dtype=np.int64)]
  seen_signatures: set[tuple] = set()

  while len(levels) < max_levels:
    signature = (tuple(dimensions.tolist()), graph.edges)
    if signature in seen_signatures:
      break
    seen_signatures.add(signature)
    level_index = len(levels)
    level_target_nodes = (
      target_nodes_per_partition if target_node_schedule is None else
      int(target_node_schedule[min(
        level_index, len(target_node_schedule) - 1
      )])
    )
    level_domain_dofs = (
      max_domain_dofs if domain_dof_schedule is None else
      int(domain_dof_schedule[min(
        level_index, len(domain_dof_schedule) - 1
      )])
    )
    ordering_result = metis_order(graph, level_target_nodes)
    domains = pack_domain_groups(
      ordering_result.groups, dimensions, level_domain_dofs
    )
    domain_offsets, domain_sizes = _domain_layout(domains, dimensions)
    level = HierarchyLevel(
      len(levels),
      graph,
      dimensions.copy(),
      prefix_offsets(dimensions),
      None if type_ids is None else type_ids.copy(),
      domains,
      domain_offsets,
      domain_sizes,
      ordering_result.order,
      ordering_result.backend,
      ordering_result.seconds,
    )
    levels.append(level)
    if (
      len(levels) >= max_levels or graph.node_count <= 1 or
      not graph.edges or
      (
        len(levels) >= 3 and minimum_domains_for_next_level and
        len(levels[-2].domains) < minimum_domains_for_next_level
      )
    ):
      break

    mapping, coarse_dims, coarse_types = _collapse_compatible(graph, domains, dimensions, type_ids)
    parent_count = int(coarse_dims.size)
    if parent_count == graph.node_count:
      break
    level.fine_to_parent = mapping
    transfers.append(make_transfer(mapping, dimensions, coarse_dims))
    composed.append(mapping[composed[-1]])
    coarse_graph = graph.remap(mapping, parent_count)
    coarse_signature = (tuple(coarse_dims.tolist()), coarse_graph.edges)
    if coarse_signature == signature:
      break
    graph, dimensions, type_ids = coarse_graph, coarse_dims, coarse_types

  # Disconnected static components can become coupled by collision blocks at
  # solve time. Once the collapsed system fits in one local bank, retaining
  # METIS component boundaries would silently drop all such cross-component
  # terms from the preconditioner. A single heterogeneous coarse bank is a
  # cheap, static-map catch-all and does not merge incompatible node types.
  if (global_coarse_domain and levels
      and levels[-1].number_of_scalar_dofs <= (
        max_domain_dofs if domain_dof_schedule is None else
        int(domain_dof_schedule[min(
          len(levels) - 1, len(domain_dof_schedule) - 1
        )])
      )
      and len(levels[-1].domains) > 1):
    coarse = levels[-1]
    coarse.domains = [coarse.ordering.astype(np.int64).tolist()]
    coarse.domain_scalar_offsets, coarse.domain_scalar_sizes = (
      _domain_layout(coarse.domains, coarse.node_dimensions)
    )

  return Hierarchy(levels, transfers, composed, perf_counter() - started, view.structure_signature())

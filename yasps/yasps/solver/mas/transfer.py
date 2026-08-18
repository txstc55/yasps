"""Adjacent and composed identity transfer maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransferMap:
  fine_node_to_parent: np.ndarray
  fine_node_scalar_offsets: np.ndarray
  parent_node_scalar_offsets: np.ndarray
  node_dimensions: np.ndarray
  parent_dimensions: np.ndarray

  def __post_init__(self) -> None:
    mapping = np.asarray(self.fine_node_to_parent, dtype=np.int64)
    fine_dims = np.asarray(self.node_dimensions, dtype=np.int64)
    parent_dims = np.asarray(self.parent_dimensions, dtype=np.int64)
    if mapping.size != fine_dims.size:
      raise ValueError("transfer map needs one parent per fine node")
    if np.any(mapping < 0) or np.any(mapping >= parent_dims.size):
      raise ValueError("transfer map parent is out of range")
    if any(fine_dims[i] != parent_dims[parent] for i, parent in enumerate(mapping)):
      raise ValueError("identity transfer cannot connect different dimensions")

  @property
  def fine_dofs(self) -> int:
    return int(np.asarray(self.node_dimensions).sum())

  @property
  def parent_dofs(self) -> int:
    return int(np.asarray(self.parent_dimensions).sum())

  def restrict(self, fine_vector: np.ndarray) -> np.ndarray:
    fine = np.asarray(fine_vector, dtype=np.float64).reshape(-1)
    if fine.size != self.fine_dofs:
      raise ValueError("fine vector has the wrong scalar size")
    coarse = np.zeros(self.parent_dofs, dtype=np.float64)
    for child, parent in enumerate(self.fine_node_to_parent):
      dim = int(self.node_dimensions[child])
      fine_start = int(self.fine_node_scalar_offsets[child])
      parent_start = int(self.parent_node_scalar_offsets[parent])
      coarse[parent_start : parent_start + dim] += fine[fine_start : fine_start + dim]
    return coarse

  def prolong(self, parent_vector: np.ndarray) -> np.ndarray:
    parent_values = np.asarray(parent_vector, dtype=np.float64).reshape(-1)
    if parent_values.size != self.parent_dofs:
      raise ValueError("parent vector has the wrong scalar size")
    fine = np.empty(self.fine_dofs, dtype=np.float64)
    for child, parent in enumerate(self.fine_node_to_parent):
      dim = int(self.node_dimensions[child])
      fine_start = int(self.fine_node_scalar_offsets[child])
      parent_start = int(self.parent_node_scalar_offsets[parent])
      fine[fine_start : fine_start + dim] = parent_values[parent_start : parent_start + dim]
    return fine


def prefix_offsets(dimensions: np.ndarray) -> np.ndarray:
  dims = np.asarray(dimensions, dtype=np.int64)
  return np.cumsum(np.r_[0, dims[:-1]], dtype=np.int64)


def make_transfer(mapping: np.ndarray, fine_dimensions: np.ndarray, parent_dimensions: np.ndarray) -> TransferMap:
  return TransferMap(
    np.asarray(mapping, dtype=np.int64),
    prefix_offsets(fine_dimensions),
    prefix_offsets(parent_dimensions),
    np.asarray(fine_dimensions, dtype=np.int64),
    np.asarray(parent_dimensions, dtype=np.int64),
  )

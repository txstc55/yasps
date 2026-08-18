"""Generic block-sparse matrix input, independent of YASPS.

Coordinates are scalar starts.  Values are row-major and grouped by block
shape.  This deliberately mirrors the small public surface exposed by YASPS,
without importing or depending on that package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Iterable, Iterator, Sequence

import numpy as np


def to_host(value, dtype=None) -> np.ndarray:
  """Return a NumPy view/copy from NumPy, PyCUDA, or array-like storage."""
  if value is None:
    result = np.empty(0)
  elif isinstance(value, np.ndarray):
    result = value
  elif hasattr(value, "get"):
    result = value.get()
  else:
    result = np.asarray(value)
  return np.asarray(result, dtype=dtype) if dtype is not None else np.asarray(result)


def _shape_pairs(dimensions: Sequence[int] | np.ndarray) -> np.ndarray:
  dims = np.asarray(dimensions, dtype=np.int64)
  if dims.size == 0:
    return np.empty((0, 2), dtype=np.int64)
  if dims.ndim == 2 and dims.shape[1] == 2:
    return dims
  if dims.ndim != 1 or dims.size % 2:
    raise ValueError("block dimensions must be row/column pairs")
  return dims.reshape(-1, 2)


def _digest_arrays(*arrays) -> bytes:
  digest = hashlib.blake2b(digest_size=20)
  for raw in arrays:
    array = np.ascontiguousarray(raw)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.uint64).tobytes())
    digest.update(array.view(np.uint8))
  return digest.digest()


@dataclass
class BlockSparseMatrixView:
  """A host/device-neutral view of a heterogeneous block-sparse matrix.

  ``*_category_starts`` index flattened values, while positions are grouped
  consecutively according to ``*_category_counts``.  Matrix block positions
  are scalar coordinates and must start exactly on variable boundaries.
  """

  rows: int
  cols: int
  variable_scalar_offsets: object
  variable_dimensions: object
  static_values: object = field(default_factory=lambda: np.empty(0, np.float64))
  static_positions: object = field(default_factory=lambda: np.empty((0, 2), np.int64))
  static_category_starts: object = field(default_factory=lambda: np.empty(0, np.int64))
  static_category_counts: object = field(default_factory=lambda: np.empty(0, np.int64))
  static_block_dimensions: object = field(default_factory=lambda: np.empty((0, 2), np.int64))
  dynamic_values: object = field(default_factory=lambda: np.empty(0, np.float64))
  dynamic_positions: object = field(default_factory=lambda: np.empty((0, 2), np.int64))
  dynamic_category_starts: object = field(default_factory=lambda: np.empty(0, np.int64))
  dynamic_category_counts: object = field(default_factory=lambda: np.empty(0, np.int64))
  dynamic_block_dimensions: object = field(default_factory=lambda: np.empty((0, 2), np.int64))
  variable_type_ids: object | None = None
  symmetric_storage: bool = True
  _layout_signature_cache: tuple | None = field(
    default=None, init=False, repr=False, compare=False
  )
  _structure_signature_cache: tuple | None = field(
    default=None, init=False, repr=False, compare=False
  )

  def __post_init__(self) -> None:
    self.rows, self.cols = int(self.rows), int(self.cols)
    if self.rows < 0 or self.cols < 0:
      raise ValueError("rows and cols must be non-negative")
    dims = to_host(self.variable_dimensions, np.int64).reshape(-1)
    offsets = to_host(self.variable_scalar_offsets, np.int64).reshape(-1)
    if dims.size != offsets.size:
      raise ValueError("variable dimensions and scalar offsets must have equal length")
    if np.any(dims <= 0):
      raise ValueError("variable dimensions must be positive")
    expected = np.cumsum(np.r_[0, dims[:-1]], dtype=np.int64)
    if not np.array_equal(offsets, expected):
      raise ValueError("variable scalar offsets must be contiguous prefix offsets")
    if dims.sum(initial=0) != self.rows or self.rows != self.cols:
      raise ValueError("variable layout must cover a square matrix exactly")
    types = None
    if self.variable_type_ids is not None:
      types = to_host(self.variable_type_ids).reshape(-1)
      if types.size != dims.size:
        raise ValueError("variable_type_ids must have one entry per variable")
    self._layout_signature_cache = (
      self.rows, self.cols, bool(self.symmetric_storage),
      _digest_arrays(
        offsets, dims,
        np.empty(0, np.int64) if types is None else types,
      ),
    )
    self._validate_part("static")
    self._validate_part("dynamic")

  @property
  def node_count(self) -> int:
    return int(to_host(self.variable_dimensions).size)

  @property
  def layout_signature(self) -> tuple:
    if self._layout_signature_cache is None:
      types = (
        np.empty(0, np.int64)
        if self.variable_type_ids is None
        else to_host(self.variable_type_ids)
      )
      self._layout_signature_cache = (
        self.rows, self.cols, bool(self.symmetric_storage),
        _digest_arrays(
          to_host(self.variable_scalar_offsets, np.int64),
          to_host(self.variable_dimensions, np.int64), types,
        ),
      )
    return self._layout_signature_cache

  def structure_signature(self) -> tuple:
    """The immutable hierarchy key; it intentionally excludes all values/dynamics."""
    if self._structure_signature_cache is None:
      self._structure_signature_cache = self.layout_signature + (
        _digest_arrays(
          to_host(self.static_positions, np.int64).reshape(-1),
          to_host(self.static_category_counts, np.int64),
          _shape_pairs(self.static_block_dimensions).reshape(-1),
        ),
      )
    return self._structure_signature_cache

  def invalidate_static_structure(self) -> None:
    """Invalidate the hierarchy key after intentionally replacing static topology.

    Static coordinates and category layout are otherwise an immutable
    contract. Numeric static values and every dynamic field may change
    without invalidating this cache.
    """
    self._layout_signature_cache = None
    self._structure_signature_cache = None

  def _validate_part(self, name: str) -> None:
    values = to_host(getattr(self, f"{name}_values"), np.float64).reshape(-1)
    positions = to_host(getattr(self, f"{name}_positions"), np.int64)
    positions = positions.reshape((-1, 2)) if positions.size else np.empty((0, 2), np.int64)
    starts = to_host(getattr(self, f"{name}_category_starts"), np.int64).reshape(-1)
    counts = to_host(getattr(self, f"{name}_category_counts"), np.int64).reshape(-1)
    shapes = _shape_pairs(getattr(self, f"{name}_block_dimensions"))
    if not (len(starts) == len(counts) == len(shapes)):
      raise ValueError(f"{name} category metadata lengths differ")
    if np.any(counts < 0) or counts.sum(initial=0) != len(positions):
      raise ValueError(f"{name} category counts do not cover positions")
    if np.any(shapes <= 0):
      raise ValueError(f"{name} block dimensions must be positive")
    boundary_to_node = {
      int(offset): index
      for index, offset in enumerate(to_host(self.variable_scalar_offsets, np.int64).reshape(-1))
    }
    variable_dims = to_host(self.variable_dimensions, np.int64).reshape(-1)
    position_index = 0
    for category, ((block_rows, block_cols), start, count) in enumerate(zip(shapes, starts, counts)):
      required = int(start + count * block_rows * block_cols)
      if start < 0 or required > values.size:
        raise ValueError(f"{name} category {category} exceeds its values buffer")
      for row, col in positions[position_index : position_index + count]:
        if int(row) not in boundary_to_node or int(col) not in boundary_to_node:
          raise ValueError(f"{name} block ({row}, {col}) starts inside or outside a variable")
        row_node, col_node = boundary_to_node[int(row)], boundary_to_node[int(col)]
        if variable_dims[row_node] != block_rows or variable_dims[col_node] != block_cols:
          raise ValueError(
            f"{name} block ({row}, {col}) shape {block_rows}x{block_cols} "
            "does not match variable dimensions"
          )
      position_index += int(count)

  def iter_blocks(self, part: str = "both") -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(row_node, col_node, block)`` using current numerical values."""
    names = ("static", "dynamic") if part == "both" else (part,)
    if any(name not in ("static", "dynamic") for name in names):
      raise ValueError("part must be 'static', 'dynamic', or 'both'")
    offsets = to_host(self.variable_scalar_offsets, np.int64).reshape(-1)
    boundary_to_node = {int(offset): i for i, offset in enumerate(offsets)}
    for name in names:
      self._validate_part(name)
      values = to_host(getattr(self, f"{name}_values"), np.float64).reshape(-1)
      positions = to_host(getattr(self, f"{name}_positions"), np.int64).reshape((-1, 2))
      starts = to_host(getattr(self, f"{name}_category_starts"), np.int64).reshape(-1)
      counts = to_host(getattr(self, f"{name}_category_counts"), np.int64).reshape(-1)
      shapes = _shape_pairs(getattr(self, f"{name}_block_dimensions"))
      position_start = 0
      for (block_rows, block_cols), value_start, count in zip(shapes, starts, counts):
        area = int(block_rows * block_cols)
        for local in range(int(count)):
          row, col = positions[position_start + local]
          begin = int(value_start + local * area)
          block = values[begin : begin + area].reshape(int(block_rows), int(block_cols))
          yield boundary_to_node[int(row)], boundary_to_node[int(col)], block
        position_start += int(count)

  def iter_block_coordinates(self, part: str = "static") -> Iterator[tuple[int, int]]:
    """Yield node coordinates without reading or depending on block values."""
    if part not in ("static", "dynamic"):
      raise ValueError("part must be 'static' or 'dynamic'")
    offsets = to_host(self.variable_scalar_offsets, np.int64).reshape(-1)
    boundary_to_node = {int(offset): i for i, offset in enumerate(offsets)}
    positions = to_host(getattr(self, f"{part}_positions"), np.int64)
    positions = positions.reshape((-1, 2)) if positions.size else np.empty((0, 2), np.int64)
    counts = to_host(getattr(self, f"{part}_category_counts"), np.int64).reshape(-1)
    if counts.sum(initial=0) != len(positions):
      raise ValueError(f"{part} category counts do not cover positions")
    for row, col in positions:
      if int(row) not in boundary_to_node or int(col) not in boundary_to_node:
        raise ValueError(f"{part} block ({row}, {col}) starts inside or outside a variable")
      yield boundary_to_node[int(row)], boundary_to_node[int(col)]

  @classmethod
  def from_blocks(
    cls,
    variable_dimensions: Sequence[int],
    static_blocks: Iterable[tuple[tuple[int, int], object]],
    dynamic_blocks: Iterable[tuple[tuple[int, int], object]] = (),
    *,
    variable_type_ids: Sequence[object] | None = None,
    symmetric_storage: bool = True,
    positions_are_nodes: bool = True,
  ) -> "BlockSparseMatrixView":
    """Convenience constructor used by standalone callers and tests."""
    dims = np.asarray(variable_dimensions, dtype=np.int64)
    offsets = np.cumsum(np.r_[0, dims[:-1]], dtype=np.int64)

    def pack(items):
      categories: dict[tuple[int, int], list[tuple[tuple[int, int], np.ndarray]]] = {}
      for (row_col, raw) in items:
        block = np.asarray(raw, dtype=np.float64)
        if block.ndim != 2:
          raise ValueError("each block must be a rank-2 array")
        row, col = map(int, row_col)
        if positions_are_nodes:
          if not (0 <= row < dims.size and 0 <= col < dims.size):
            raise ValueError("block node index is out of range")
          scalar_position = (int(offsets[row]), int(offsets[col]))
        else:
          scalar_position = (row, col)
        categories.setdefault(tuple(block.shape), []).append((scalar_position, block))
      values, positions, starts, counts, shapes = [], [], [], [], []
      cursor = 0
      for shape, entries in categories.items():
        shapes.append(shape)
        starts.append(cursor)
        counts.append(len(entries))
        for position, block in entries:
          positions.append(position)
          flat = block.reshape(-1)
          values.extend(flat)
          cursor += flat.size
      return (
        np.asarray(values, np.float64),
        np.asarray(positions, np.int64).reshape((-1, 2)),
        np.asarray(starts, np.int64),
        np.asarray(counts, np.int64),
        np.asarray(shapes, np.int64).reshape((-1, 2)),
      )

    static = pack(static_blocks)
    dynamic = pack(dynamic_blocks)
    total = int(dims.sum())
    return cls(
      total,
      total,
      offsets,
      dims,
      *static,
      *dynamic,
      variable_type_ids=variable_type_ids,
      symmetric_storage=symmetric_storage,
    )

"""Adapters from YASPS block matrices to the internal MAS matrix view."""

from __future__ import annotations

import numpy as np

from .mas.matrix_view import BlockSparseMatrixView


def _to_host(value, dtype=None):
  if hasattr(value, "get"):
    value = value.get()
  return np.asarray(value, dtype=dtype)


def _active_positions(positions, block_count: int):
  shape = getattr(positions, "shape", ())
  if len(shape) == 2:
    return positions[:block_count]
  return positions[:block_count * 2]


def _matrix_part(matrix, name: str):
  suffix = "" if name == "static" else "_dynamic"
  counts = tuple(int(value) for value in getattr(matrix, f"block_counts{suffix}"))
  category_count = len(counts)
  block_count = sum(counts)
  return {
    "values": getattr(matrix, f"blocks_flattened{suffix}"),
    "positions": _active_positions(
      getattr(matrix, f"block_positions{suffix}"), block_count
    ),
    "starts": np.asarray(
      getattr(matrix, f"blocks_start_indices{suffix}")[:category_count],
      dtype=np.int64,
    ),
    "counts": np.asarray(counts, dtype=np.int64),
    "dimensions": np.asarray(
      getattr(matrix, f"block_dimensions{suffix}"), dtype=np.int64
    ).reshape((-1, 2)),
  }


def _record_dimension(dimensions_by_offset, offset: int, dimension: int):
  previous = dimensions_by_offset.setdefault(int(offset), int(dimension))
  if previous != dimension:
    raise ValueError(
      f"inconsistent block dimensions at scalar offset {offset}: "
      f"{previous} and {dimension}"
    )


def _infer_matrix_layout(matrix, static, dynamic):
  if hasattr(matrix, "variable_dimensions"):
    dimensions = _to_host(matrix.variable_dimensions, np.int64).reshape(-1)
    if hasattr(matrix, "variable_scalar_offsets"):
      offsets = _to_host(
        matrix.variable_scalar_offsets, np.int64
      ).reshape(-1)
    else:
      offsets = np.cumsum(np.r_[0, dimensions[:-1]], dtype=np.int64)
    type_ids = (
      _to_host(matrix.variable_type_ids).reshape(-1)
      if hasattr(matrix, "variable_type_ids") and
      matrix.variable_type_ids is not None else None
    )
    return offsets, dimensions, type_ids

  if hasattr(matrix, "wrt"):
    dimensions = []
    type_ids = []
    for type_id, variable in enumerate(matrix.wrt):
      count = int(variable.correspondance.numInstances)
      dimensions.extend([int(variable.size)] * count)
      type_ids.extend([type_id] * count)
    dimensions = np.asarray(dimensions, dtype=np.int64)
    offsets = np.cumsum(np.r_[0, dimensions[:-1]], dtype=np.int64)
    return offsets, dimensions, np.asarray(type_ids, dtype=np.int64)

  dimensions_by_offset = {}
  for part in (static, dynamic):
    positions = _to_host(part["positions"], np.int64).reshape((-1, 2))
    cursor = 0
    for (rows, cols), count in zip(part["dimensions"], part["counts"]):
      for row, col in positions[cursor:cursor + int(count)]:
        _record_dimension(dimensions_by_offset, int(row), int(rows))
        _record_dimension(dimensions_by_offset, int(col), int(cols))
      cursor += int(count)

  if not dimensions_by_offset:
    raise ValueError(
      "masSolver cannot infer variable blocks from an empty yasps.matrix; "
      "provide variable_dimensions and variable_scalar_offsets"
    )
  offsets = np.asarray(sorted(dimensions_by_offset), dtype=np.int64)
  dimensions = np.asarray(
    [dimensions_by_offset[int(offset)] for offset in offsets], dtype=np.int64
  )
  expected = np.cumsum(np.r_[0, dimensions[:-1]], dtype=np.int64)
  if not np.array_equal(offsets, expected) or int(dimensions.sum()) != int(matrix.rows):
    raise ValueError(
      "masSolver inferred a non-contiguous variable layout; every variable "
      "in a generic yasps.matrix needs at least one stored block"
    )
  return offsets, dimensions, None


class YASPSMatrixView(BlockSparseMatrixView):
  """Device-resident view of either a :class:`matrix` or :class:`hessian`."""

  def __init__(self, matrix):
    required = (
      "rows", "cols", "symmetric_storage", "blocks_flattened",
      "block_positions", "blocks_start_indices", "block_counts",
      "block_dimensions", "blocks_flattened_dynamic",
      "block_positions_dynamic", "blocks_start_indices_dynamic",
      "block_counts_dynamic", "block_dimensions_dynamic",
    )
    missing = [name for name in required if not hasattr(matrix, name)]
    if missing:
      raise TypeError(
        "masSolver requires a yasps matrix-compatible object; missing: "
        + ", ".join(missing)
      )
    static = _matrix_part(matrix, "static")
    dynamic = _matrix_part(matrix, "dynamic")
    offsets, dimensions, type_ids = _infer_matrix_layout(
      matrix, static, dynamic
    )
    super().__init__(
      rows=int(matrix.rows),
      cols=int(matrix.cols),
      variable_scalar_offsets=offsets,
      variable_dimensions=dimensions,
      static_values=static["values"],
      static_positions=static["positions"],
      static_category_starts=static["starts"],
      static_category_counts=static["counts"],
      static_block_dimensions=static["dimensions"],
      dynamic_values=dynamic["values"],
      dynamic_positions=dynamic["positions"],
      dynamic_category_starts=dynamic["starts"],
      dynamic_category_counts=dynamic["counts"],
      dynamic_block_dimensions=dynamic["dimensions"],
      variable_type_ids=type_ids,
      symmetric_storage=bool(matrix.symmetric_storage),
    )
    self.matrix = matrix

  def update_numeric(self, matrix=None):
    """Refresh live values and collision storage without rebuilding METIS."""
    source = self.matrix if matrix is None else matrix
    if int(source.rows) != self.rows or int(source.cols) != self.cols:
      raise ValueError("matrix dimensions changed while refreshing MAS")
    static = _matrix_part(source, "static")
    dynamic = _matrix_part(source, "dynamic")
    self.static_values = static["values"]
    self.dynamic_values = dynamic["values"]
    self.dynamic_positions = dynamic["positions"]
    self.dynamic_category_starts = dynamic["starts"]
    self.dynamic_category_counts = dynamic["counts"]
    self.dynamic_block_dimensions = dynamic["dimensions"]
    self.matrix = source

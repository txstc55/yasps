# cython: language_level=3
"""Find exact block structure from a symbolic Jacobian's nonzero positions."""

from operator import index


def generate_jacobian_block_layout(rows, cols, nonzero_positions):
  """Return the finest diagonal blocks obtainable by row/column permutation.

  ``nonzero_positions`` is a flat sequence ``[row, col, row, col, ...]``.
  Positions must describe structural nonzeros: runtime values are deliberately
  never inspected. Each connected component of the bipartite row/column graph
  is one irreducible rectangular block.

  The returned dictionary describes a layout, without moving numeric data:
  - ``rows`` / ``cols`` retain the original Jacobian dimensions.
  - ``blocks`` lists each block's original row and column indices. These are
    local Jacobian indices, not global mesh vertex indices.
  - ``sizes`` / ``spans`` give each block's row / column count, respectively.
    Spans are counts, not offsets or index ranges.
  - ``row_permutation`` / ``column_permutation`` map reordered indices back
    to original indices. Thus ``J[row_permutation][:, column_permutation]``
    places the blocks on the diagonal.
  - ``inverse_row_permutation`` / ``inverse_column_permutation`` map original
    indices to reordered indices, reversing those mappings.
  - ``zero_rows`` / ``zero_columns`` list original axes with no structural
    nonzeros. They are appended to the permutations, outside the active blocks.

  For example, a 4x5 Jacobian with nonzeros at (0, 1), (0, 3), and (2, 0)
  produces blocks [{"rows": [0], "cols": [1, 3]}, {"rows": [2], "cols": [0]}],
  sizes [1, 1], and spans [2, 1]. The row permutation is [0, 2, 1, 3] and the
  column permutation is [1, 3, 0, 2, 4]. Column permutation entry 0 is 1:
  reordered column 0 was originally column 1, so inverse column entry 1 is 0.
  """
  rows = index(rows)
  cols = index(cols)
  if rows < 0 or cols < 0:
    raise ValueError("Jacobian dimensions must be nonnegative.")
  positions = list(nonzero_positions)
  if len(positions) % 2:
    raise ValueError("Nonzero positions must contain row/column pairs.")

  # A disjoint-set forest computes exact components in essentially linear time.
  parent = list(range(rows + cols))
  rank = [0] * len(parent)
  active_rows = set()
  active_cols = set()

  def find(node):
    while parent[node] != node:
      parent[node] = parent[parent[node]]
      node = parent[node]
    return node

  for offset in range(0, len(positions), 2):
    row = index(positions[offset])
    col = index(positions[offset + 1])
    if not 0 <= row < rows or not 0 <= col < cols:
      raise ValueError(f"Nonzero position ({row}, {col}) is outside ({rows}, {cols}).")
    active_rows.add(row)
    active_cols.add(col)
    left = find(row)
    right = find(rows + col)
    if left == right:
      continue
    if rank[left] < rank[right]:
      left, right = right, left
    parent[right] = left
    if rank[left] == rank[right]:
      rank[left] += 1

  components = {}
  for row in sorted(active_rows):
    components.setdefault(find(row), {"rows": [], "cols": []})["rows"].append(row)
  for col in sorted(active_cols):
    components[find(rows + col)]["cols"].append(col)
  blocks = sorted(components.values(), key=lambda block: (block["rows"][0], block["cols"][0]))
  zero_rows = [row for row in range(rows) if row not in active_rows]
  zero_columns = [col for col in range(cols) if col not in active_cols]
  row_permutation = [row for block in blocks for row in block["rows"]] + zero_rows
  column_permutation = [col for block in blocks for col in block["cols"]] + zero_columns
  inverse_row_permutation = [0] * rows
  inverse_column_permutation = [0] * cols
  for permuted, original in enumerate(row_permutation):
    inverse_row_permutation[original] = permuted
  for permuted, original in enumerate(column_permutation):
    inverse_column_permutation[original] = permuted

  return {
    "rows": rows,  # Original Jacobian row count.
    "cols": cols,  # Original Jacobian column count.
    "blocks": blocks,  # Original local row/column index lists for each block.
    "sizes": [len(block["rows"]) for block in blocks],  # Row count per block.
    "spans": [len(block["cols"]) for block in blocks],  # Column count per block, not an offset.
    "row_permutation": row_permutation,  # Reordered row -> original row.
    "column_permutation": column_permutation,  # Reordered column -> original column.
    "inverse_row_permutation": inverse_row_permutation,  # Original row -> reordered row.
    "inverse_column_permutation": inverse_column_permutation,  # Original column -> reordered column.
    "zero_rows": zero_rows,  # Structurally zero original rows, appended to the row permutation.
    "zero_columns": zero_columns,  # Structurally zero original columns, appended to the column permutation.
  }

"""CPU checks for structural blocks and permutation-aware Hessian contraction."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


# The layout utility is pure Python; these checks do not need a CUDA context.
utility_path = Path(__file__).resolve().parents[2] / "yasps/yasps/jacobianBlockLayout.py"
utility_spec = importlib.util.spec_from_file_location("jacobian_block_layout", utility_path)
utility = importlib.util.module_from_spec(utility_spec)
utility_spec.loader.exec_module(utility)
generate_layout = utility.generate_jacobian_block_layout


def layout_for(matrix):
    positions = np.column_stack(np.nonzero(matrix)).ravel().tolist()
    return generate_layout(*matrix.shape, positions)


def check_permutations_and_contraction(jacobian, layout, random):
    rows, cols = jacobian.shape
    row_order = layout["row_permutation"]
    col_order = layout["column_permutation"]
    assert sorted(row_order) == list(range(rows))
    assert sorted(col_order) == list(range(cols))
    np.testing.assert_array_equal(np.asarray(row_order)[layout["inverse_row_permutation"]], np.arange(rows))
    np.testing.assert_array_equal(np.asarray(col_order)[layout["inverse_column_permutation"]], np.arange(cols))

    permuted = jacobian[np.ix_(row_order, col_order)]
    expected_permuted = np.zeros_like(jacobian)
    row_offset = col_offset = 0
    for block in layout["blocks"]:
        block_rows = block["rows"]
        block_cols = block["cols"]
        expected_permuted[row_offset:row_offset + len(block_rows), col_offset:col_offset + len(block_cols)] = jacobian[np.ix_(block_rows, block_cols)]
        row_offset += len(block_rows)
        col_offset += len(block_cols)
    np.testing.assert_array_equal(permuted, expected_permuted)

    # Inner Hessian is deliberately dense/indefinite and couples all components.
    inner = random.normal(size=(rows, rows))
    inner = inner + inner.T
    assembled = np.zeros((cols, cols))
    for left in layout["blocks"]:
        for right in layout["blocks"]:
            left_jacobian = jacobian[np.ix_(left["rows"], left["cols"])]
            right_jacobian = jacobian[np.ix_(right["rows"], right["cols"])]
            inner_block = inner[np.ix_(left["rows"], right["rows"])]
            assembled[np.ix_(left["cols"], right["cols"])] = left_jacobian.T @ inner_block @ right_jacobian
    np.testing.assert_allclose(assembled, jacobian.T @ inner @ jacobian, atol=2e-12, rtol=2e-12)


def test_interleaved_qr_jacobian_has_three_rectangular_components():
    random = np.random.default_rng(521)
    q, _ = np.linalg.qr(random.normal(size=(8, 3)))
    jacobian = np.zeros((9, 24))
    for axis in range(3):
        jacobian[np.ix_(range(3 * axis, 3 * axis + 3), range(axis, 24, 3))] = q.T
    layout = layout_for(jacobian)
    assert layout["sizes"] == [3, 3, 3]
    assert layout["spans"] == [8, 8, 8]
    assert layout["row_permutation"] == list(range(9))
    assert layout["column_permutation"] == list(range(0, 24, 3)) + list(range(1, 24, 3)) + list(range(2, 24, 3))
    check_permutations_and_contraction(jacobian, layout, random)


def test_both_permutations_zero_axes_and_nonrectangular_pattern():
    jacobian = np.zeros((7, 9))
    jacobian[4, 1] = 2
    jacobian[4, 6] = 3
    jacobian[1, 6] = 4
    jacobian[2, 7] = 5
    jacobian[5, 2] = 6
    jacobian[5, 7] = 7
    layout = layout_for(jacobian)
    assert layout["blocks"] == [{"rows": [1, 4], "cols": [1, 6]}, {"rows": [2, 5], "cols": [2, 7]}]
    assert layout["zero_rows"] == [0, 3, 6]
    assert layout["zero_columns"] == [0, 3, 4, 5, 8]
    check_permutations_and_contraction(jacobian, layout, np.random.default_rng(7))


@pytest.mark.parametrize("rows,cols", [(0, 0), (0, 5), (4, 0), (6, 8)])
def test_empty_structure_keeps_every_axis(rows, cols):
    layout = generate_layout(rows, cols, [])
    assert layout["blocks"] == []
    assert layout["zero_rows"] == list(range(rows))
    assert layout["zero_columns"] == list(range(cols))
    assert layout["row_permutation"] == list(range(rows))
    assert layout["column_permutation"] == list(range(cols))


def test_duplicate_input_order_and_connected_component_are_exact():
    edges = [0, 1, 2, 1, 2, 3, 1, 2, 3, 0]
    reference = generate_layout(4, 4, edges)
    pairs = np.asarray(edges).reshape(-1, 2)
    reordered = np.concatenate([pairs[::-1], pairs]).ravel().tolist()
    assert generate_layout(4, 4, reordered) == reference
    assert reference["blocks"] == [{"rows": [0, 2], "cols": [1, 3]}, {"rows": [1], "cols": [2]}, {"rows": [3], "cols": [0]}]
    # A single bridge merges both previously separate components.
    bridged = generate_layout(4, 4, edges + [1, 1])
    assert bridged["blocks"] == [{"rows": [0, 1, 2], "cols": [1, 2, 3]}, {"rows": [3], "cols": [0]}]


@pytest.mark.parametrize("seed", range(12))
def test_random_rectangular_structure_and_dense_inner_hessian(seed):
    random = np.random.default_rng(seed)
    jacobian = random.normal(size=(13, 19))
    jacobian *= random.uniform(size=jacobian.shape) < 0.06
    # A structural nonzero must remain connected even when arbitrarily small.
    jacobian[0, 0] = 1e-300
    layout = layout_for(jacobian)
    check_permutations_and_contraction(jacobian, layout, random)


@pytest.mark.parametrize("rows,cols,positions", [(-1, 2, []), (1, 1, [0]), (1, 1, [1, 0]), (1, 1, [0, -1])])
def test_invalid_dimensions_and_positions_are_rejected(rows, cols, positions):
    with pytest.raises(ValueError):
        generate_layout(rows, cols, positions)


@pytest.mark.parametrize("seed", range(8))
def test_scalar_scatter_matches_dense_pullback_with_padding_and_repeated_vertices(seed):
    random = np.random.default_rng(101 + seed)
    # Original index order is neither global-coordinate order nor sparsity-
    # component order. Indices 0/1 retain padded/nontarget local columns.
    indices = np.array([13, 0, 2, 1, 5, 13, 2, 0, 9])
    sizes = np.array([3, 2, 3, 1, 4, 3, 3, 0, 4])
    permutations = np.where(indices >= 2, 1, np.where(indices == 1, -1, 0))
    outer = np.concatenate([[0], np.cumsum(sizes)])
    column_segment = np.repeat(np.arange(len(sizes)), sizes)
    valid = [i for i in range(len(sizes)) if permutations[i] > 0 and indices[i] >= 2]
    valid_rank = {segment: rank for rank, segment in enumerate(valid)}

    # Emulate coordinate generation independently with its nested segment loop,
    # then global compression. Multiple segment pairs can share a final block.
    lookups = []
    blocks = {}
    for local_i, i in enumerate(valid):
        for j in valid[local_i:]:
            low, high = (i, j) if indices[i] <= indices[j] else (j, i)
            key = (int(indices[low]), int(indices[high]))
            blocks.setdefault(key, np.zeros((sizes[low], sizes[high])))
            lookups.append(key)

    # Three nontrivial interleaved components; the final row and column are zero.
    rows, cols = 11, int(outer[-1])
    jacobian = np.zeros((rows, cols))
    row_labels = random.permutation(np.arange(rows - 1) % 3)
    col_labels = random.permutation(np.arange(cols - 1) % 3)
    for row in range(rows - 1):
        for col in range(cols - 1):
            if row_labels[row] == col_labels[col]:
                jacobian[row, col] = random.normal()
    layout = layout_for(jacobian)
    inner = random.normal(size=(rows, rows))
    inner += inner.T

    for component_i, left in enumerate(layout["blocks"]):
        for component_j in range(component_i, len(layout["blocks"])):
            right = layout["blocks"][component_j]
            values = jacobian[np.ix_(left["rows"], left["cols"])].T @ inner[np.ix_(left["rows"], right["rows"])] @ jacobian[np.ix_(right["rows"], right["cols"])]
            for a, original_a in enumerate(left["cols"]):
                for b in range(a if component_i == component_j else 0, len(right["cols"])):
                    original_b = right["cols"][b]
                    segment_a = int(column_segment[original_a])
                    segment_b = int(column_segment[original_b])
                    if segment_a not in valid_rank or segment_b not in valid_rank:
                        continue
                    first, last = sorted([valid_rank[segment_a], valid_rank[segment_b]])
                    lookup_index = first * len(valid) - first * (first + 1) // 2 + last
                    block = blocks[lookups[lookup_index]]
                    offset_a = original_a - outer[segment_a]
                    offset_b = original_b - outer[segment_b]
                    if indices[segment_a] <= indices[segment_b]:
                        block[offset_a, offset_b] += values[a, b]
                    else:
                        block[offset_b, offset_a] += values[a, b]
                    if indices[segment_a] == indices[segment_b] and original_a != original_b:
                        block[offset_b, offset_a] += values[a, b]

    assembled = np.zeros((14, 14))
    for (start_a, start_b), block in blocks.items():
        row, col = start_a - 2, start_b - 2
        height, width = block.shape
        assembled[row:row + height, col:col + width] += block
        if row != col:
            assembled[col:col + width, row:row + height] += block.T

    # Independent dense reference sums the full local matrix through a gather
    # map, naturally handling every repeated DOF and discarding padded axes.
    gather = np.zeros((cols, 14))
    for segment in valid:
        for offset in range(sizes[segment]):
            gather[outer[segment] + offset, indices[segment] - 2 + offset] = 1
    expected = gather.T @ jacobian.T @ inner @ jacobian @ gather
    np.testing.assert_allclose(assembled, expected, atol=2e-12, rtol=2e-12)

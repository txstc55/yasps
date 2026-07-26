"""Metal counterpart to ``hessianAndGradientKernel.pyx``."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import time

import mlx.core as mx

from yasps.backend import GPUArray
from yasps.codeGeneratorMetal import MetalProgram


_THREADS = 256
_STAGED_EXPRESSION_SOURCE_BYTES = 128 * 1024
_STAGED_EXPRESSION_BUFFER_BYTES = 16 * 1024 * 1024


@lru_cache(maxsize=1)
def _linalg_header():
  return (
    Path(__import__("yasps").__file__).parent
    / "kernel"
    / "metalLinalg.metal"
  ).read_text()


def _array(value):
  return value._array if isinstance(value, GPUArray) else value


def _atomic_add(output: str, index: str, value: str) -> str:
  return (
    "atomic_fetch_add_explicit("
    f"&{output}[{index}], {value}, memory_order_relaxed);"
  )


@lru_cache(maxsize=1)
def _accumulate_kernel():
  source = """
    const uint index = thread_position_in_grid.x;
    if (index < sizes[0]) output_0[index] = current_0[index] + added_0[index];
    if (index < sizes[1]) output_1[index] = current_1[index] + added_1[index];
    if (index < sizes[2]) output_2[index] = current_2[index] + added_2[index];
    if (index < sizes[3]) output_3[index] = current_3[index] + added_3[index];
  """
  return mx.fast.metal_kernel(
    name="yasps_accumulate_hessian_outputs",
    input_names=[
      "current_0",
      "current_1",
      "current_2",
      "current_3",
      "added_0",
      "added_1",
      "added_2",
      "added_3",
      "sizes",
    ],
    output_names=["output_0", "output_1", "output_2", "output_3"],
    source=source,
  )


def _accumulate(targets, additions):
  sizes = [target.size for target in targets]
  largest = max(sizes)
  if largest == 0:
    return
  outputs = _accumulate_kernel()(
    inputs=[
      *[_array(target) for target in targets],
      *additions,
      mx.array(sizes, dtype=mx.uint32),
    ],
    grid=(largest, 1, 1),
    threadgroup=(min(largest, _THREADS), 1, 1),
    output_shapes=[target.shape for target in targets],
    output_dtypes=[_array(target).dtype for target in targets],
  )
  for target, output in zip(targets, outputs):
    target._array = output


class MetalHessianProgram:
  """JIT one symbolic Hessian term using the CUDA assembly ABI."""

  def __init__(
    self,
    attribute,
    project_entire_hessian,
    projection_method,
    gradient_only,
    separate_hessian_jacobian,
    max_num_indices,
    jacobian_rows=0,
    jacobian_cols=0,
    hessian_row_size=0,
    jacobian_nonzero_count=0,
    jacobian_nonzero_positions=(),
    jacobian_children_sizes=(),
    jacobian_children_spans=(),
  ):
    self.attribute = attribute
    self.expression = MetalProgram(attribute)
    # Fusing a very large differentiated expression into every projection-size
    # variant makes Metal compile the same expression repeatedly.  Stage those
    # expressions once on the GPU and keep the assembly variants compact.  A
    # single local/separate projection has no variants to reuse.
    self.can_stage_expression = (
      project_entire_hessian
      and len(self.expression.header) >= _STAGED_EXPRESSION_SOURCE_BYTES
    )
    self._last_stage_expression = None
    if os.environ.get("YASPS_METAL_JIT_TRACE", "").strip().lower() in {
      "1",
      "true",
      "yes",
    }:
      print(
        "Metal Hessian JIT "
        f"{attribute.fullName}: expression source "
        f"{len(self.expression.header)} bytes, staging eligible "
        f"{self.can_stage_expression}"
      )
    self.project_entire_hessian = project_entire_hessian
    self.projection_method = projection_method
    self.gradient_only = gradient_only
    self.max_num_indices = max_num_indices
    self.separate_hessian_jacobian = (
      separate_hessian_jacobian and not project_entire_hessian
    )
    self.jacobian_rows = int(jacobian_rows)
    self.jacobian_cols = int(jacobian_cols)
    self.hessian_row_size = int(hessian_row_size)
    self.jacobian_nonzero_count = int(jacobian_nonzero_count)
    self.jacobian_nonzero_positions = tuple(
      int(position) for position in jacobian_nonzero_positions
    )
    self.jacobian_children_sizes = tuple(
      int(size) for size in jacobian_children_sizes
    )
    self.jacobian_children_spans = tuple(
      int(span) for span in jacobian_children_spans
    )
    if self.separate_hessian_jacobian:
      if len(self.jacobian_nonzero_positions) != 2 * self.jacobian_nonzero_count:
        raise ValueError("Separate Jacobian nonzero metadata is inconsistent.")
      if len(self.jacobian_children_sizes) != len(self.jacobian_children_spans):
        raise ValueError("Separate Jacobian block metadata is inconsistent.")
    self.kernels = {}

  def update(self, unique_gradient_sizes, stage_expression=None):
    if stage_expression is None:
      return
    sizes = [int(size) for size in unique_gradient_sizes if int(size) > 0]
    if not self.project_entire_hessian:
      sizes = [0]
    for size in sizes:
      key = (size, stage_expression)
      if key not in self.kernels:
        self.kernels[key] = self._compile(size, stage_expression)

  def _compile(self, gradient_size, stage_expression):
    source = self._source(gradient_size, stage_expression)
    header = (
      _linalg_header() if stage_expression else self.expression.header
    )
    digest = sha256((header + source).encode()).hexdigest()[:16]
    name = f"yasps_hessian_assembly_{digest}"
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    source_path = output_directory / f"{name}.metal"
    full_source = header + "\n\n" + source
    if not source_path.exists() or source_path.read_text() != full_source:
      source_path.write_text(full_source)
    expression_inputs = (
      ["expression_values"]
      if stage_expression
      else self.expression.resource_input_names
    )
    return mx.fast.metal_kernel(
      name=name,
      input_names=[
        *expression_inputs,
        "segment_indices",
        "segment_sizes",
        "local_permutations",
        "lookups",
        "coordinate_outer",
        "grouped_indices",
        "grouped_outer",
        "group_index",
        "projection_method",
        "diagonal_blocks_start",
        "gradient_segments_start",
        "attribute_count",
      ],
      output_names=[
        "gradient_contribution",
        "hessian_contribution",
        "diagonal_contribution",
        "diagonal_blocks_contribution",
      ],
      header=header,
      source=source,
      compile_options={"math_mode": "fast"},
      atomic_outputs=True,
    )

  def _source(self, gradient_size, stage_expression):
    lines = [
      "const uint local_index = thread_position_in_grid.x;",
      "const uint group_start = grouped_outer[group_index[0]];",
      "const uint group_end = grouped_outer[group_index[0] + 1u];",
      "if (local_index >= group_end - group_start) return;",
      "const uint instance = grouped_indices[group_start + local_index];",
      f"constexpr ushort K = {self.max_num_indices};",
    ]
    if stage_expression:
      lines.append(
        "const auto hg = expression_values "
        f"+ instance * {self.attribute.size}u;"
      )
    else:
      lines.extend(
        [
          f"float hg[{self.attribute.size}];",
          self.expression.root_call("instance", "hg"),
        ]
      )
    lines.extend(
      [
        "uint gradient_offset = 0u;",
        "for (ushort i = 0; i < K; ++i) {",
        "  const ushort segment_size = segment_sizes[instance * K + i];",
        "  uint placement = segment_indices[instance * K + i];",
        "  if (placement >= 2u) {",
      ]
    )
    if self.separate_hessian_jacobian:
      gradient_base = self.attribute.cols - self.jacobian_cols
    else:
      gradient_row = 0 if self.gradient_only else self.attribute.rows - 1
      gradient_base = gradient_row * self.attribute.cols
    gradient_value = f"hg[{gradient_base} + gradient_offset + j]"
    for_line = "    " + _atomic_add(
      "gradient_contribution", "placement - 2u + j", gradient_value
    )
    lines.extend(
      [
        "    for (ushort j = 0; j < segment_size; ++j) {",
        for_line,
        "    }",
        "  }",
        "  gradient_offset += segment_size;",
        "}",
      ]
    )
    if self.gradient_only:
      return "\n".join(lines)
    if self.project_entire_hessian:
      lines.extend(self._full_projection_source(gradient_size))
    elif self.separate_hessian_jacobian:
      lines.extend(self._separate_jacobian_source())
    else:
      lines.extend(self._local_projection_source())
    return "\n".join(lines)

  def _attribute_block_source(self, segment_index, segment_size):
    return [
      "ushort which_attribute = 0;",
      (
        "while (which_attribute + 1u < attribute_count[0] "
        f"&& {segment_index} >= gradient_segments_start[which_attribute + 1u]) {{"
      ),
      "  ++which_attribute;",
      "}",
      (
        f"const uint attribute_difference = {segment_index} "
        "- gradient_segments_start[which_attribute];"
      ),
      (
        f"const uint attribute_instance = attribute_difference / "
        f"uint({segment_size});"
      ),
      (
        "const uint diagonal_block_placement = "
        "diagonal_blocks_start[which_attribute] + attribute_instance "
        f"* uint({segment_size}) * uint({segment_size});"
      ),
    ]

  def _full_projection_source(self, gradient_size):
    n = int(gradient_size)
    lines = [
      f"float compressed_hessian[{n * n}];",
      f"for (ushort i = 0; i < {n * n}; ++i) compressed_hessian[i] = 0.0f;",
      "uint row_offset = 0u;",
      "for (ushort i = 0; i < K; ++i) {",
      "  const short signed_i = local_permutations[instance * K + i];",
      "  const ushort size_i = segment_sizes[instance * K + i];",
      "  uint column_offset = 0u;",
      "  if (signed_i != 0) {",
      "    const uint permutation_i = uint(signed_i < 0 ? -signed_i : signed_i) - 1u;",
      "    for (ushort j = 0; j < K; ++j) {",
      "      const short signed_j = local_permutations[instance * K + j];",
      "      const ushort size_j = segment_sizes[instance * K + j];",
      "      if (signed_j != 0) {",
      "        const uint permutation_j = uint(signed_j < 0 ? -signed_j : signed_j) - 1u;",
      "        for (ushort row = 0; row < size_i; ++row) {",
      "          for (ushort column = 0; column < size_j; ++column) {",
      (
        f"            compressed_hessian[(permutation_i + row) * {n} "
        "+ permutation_j + column] += "
        f"hg[(row_offset + row) * {self.attribute.cols} "
        "+ column_offset + column];"
      ),
      "          }",
      "        }",
      "      }",
      "      column_offset += size_j;",
      "    }",
      "  }",
      "  row_offset += size_i;",
      "}",
      (
        f"if (projection_method[0] >= 0) "
        f"yasps_spd_project<{n}>(compressed_hessian, projection_method[0]);"
      ),
      "const uint coordinate_start = coordinate_outer[instance];",
      "uint valid_block = 0u;",
      "for (ushort i = 0; i < K; ++i) {",
      "  short permutation_i = local_permutations[instance * K + i];",
      "  uint placement_i = segment_indices[instance * K + i];",
      "  if (permutation_i <= 0 || placement_i < 2u) continue;",
      "  --permutation_i;",
      "  const ushort size_i = segment_sizes[instance * K + i];",
      "  for (ushort j = i; j < K; ++j) {",
      "    short permutation_j = local_permutations[instance * K + j];",
      "    uint placement_j = segment_indices[instance * K + j];",
      "    if (permutation_j <= 0 || placement_j < 2u) continue;",
      "    --permutation_j;",
      "    const ushort size_j = segment_sizes[instance * K + j];",
      "    const uint output = lookups[coordinate_start + valid_block];",
      "    if (placement_i < placement_j) {",
      "      for (ushort row = 0; row < size_i; ++row) {",
      "        for (ushort column = 0; column < size_j; ++column) {",
      "          " + _atomic_add(
        "hessian_contribution",
        "output + row * size_j + column",
        (
          f"compressed_hessian[(uint(permutation_i) + row) * {n} "
          "+ uint(permutation_j) + column]"
        ),
      ),
      "        }",
      "      }",
      "    } else {",
      "      for (ushort row = 0; row < size_j; ++row) {",
      "        for (ushort column = 0; column < size_i; ++column) {",
      "          " + _atomic_add(
        "hessian_contribution",
        "output + row * size_i + column",
        (
          f"compressed_hessian[(uint(permutation_i) + column) * {n} "
          "+ uint(permutation_j) + row]"
        ),
      ),
      "        }",
      "      }",
      "    }",
      "    if (i == j) {",
      "      const uint segment_index = placement_i - 2u;",
      "      for (ushort row = 0; row < size_i; ++row) {",
      "        " + _atomic_add(
        "diagonal_contribution",
        "segment_index + row",
        (
          f"compressed_hessian[(uint(permutation_i) + row) * {n} "
          "+ uint(permutation_j) + row]"
        ),
      ),
      "      }",
      *["      " + line for line in self._attribute_block_source(
        "segment_index", "size_i"
      )],
      "      for (ushort row = 0; row < size_i; ++row) {",
      "        for (ushort column = 0; column < size_i; ++column) {",
      "          " + _atomic_add(
        "diagonal_blocks_contribution",
        "diagonal_block_placement + row * size_i + column",
        (
          f"compressed_hessian[(uint(permutation_i) + row) * {n} "
          "+ uint(permutation_j) + column]"
        ),
      ),
      "        }",
      "      }",
      "    }",
      "    ++valid_block;",
      "  }",
      "}",
    ]
    return lines

  def _local_projection_source(self):
    lines = [
      "uint raw_offsets[K + 1];",
      "raw_offsets[0] = 0u;",
      "for (ushort i = 0; i < K; ++i) {",
      "  raw_offsets[i + 1u] = raw_offsets[i] + segment_sizes[instance * K + i];",
      "}",
      "const uint coordinate_start = coordinate_outer[instance];",
      "uint valid_block = 0u;",
      "for (ushort i = 0; i < K; ++i) {",
      "  const short representative_i = local_permutations[instance * K + i];",
      "  const uint placement_i = segment_indices[instance * K + i];",
      "  if (representative_i <= 0 || placement_i < 2u) continue;",
      "  const ushort size_i = segment_sizes[instance * K + i];",
      "  for (ushort j = i; j < K; ++j) {",
      "    const short representative_j = local_permutations[instance * K + j];",
      "    const uint placement_j = segment_indices[instance * K + j];",
      "    if (representative_j <= 0 || placement_j < 2u) continue;",
      "    const ushort size_j = segment_sizes[instance * K + j];",
      "    const uint output = lookups[coordinate_start + valid_block];",
      "    for (ushort output_row = 0; output_row < size_i; ++output_row) {",
      "      for (ushort output_column = 0; output_column < size_j; ++output_column) {",
      "        float accumulated = 0.0f;",
      "        for (ushort source_i = 0; source_i < K; ++source_i) {",
      "          short source_permutation_i = local_permutations[instance * K + source_i];",
      "          if (source_permutation_i < 0) source_permutation_i = -source_permutation_i;",
      "          if (source_permutation_i != representative_i) continue;",
      "          for (ushort source_j = 0; source_j < K; ++source_j) {",
      "            short source_permutation_j = local_permutations[instance * K + source_j];",
      "            if (source_permutation_j < 0) source_permutation_j = -source_permutation_j;",
      "            if (source_permutation_j != representative_j) continue;",
      (
        f"            accumulated += hg[(raw_offsets[source_i] + output_row) "
        f"* {self.attribute.cols} + raw_offsets[source_j] + output_column];"
      ),
      "          }",
      "        }",
      "        if (placement_i < placement_j) {",
      "          " + _atomic_add(
        "hessian_contribution",
        "output + output_row * size_j + output_column",
        "accumulated",
      ),
      "        } else {",
      "          " + _atomic_add(
        "hessian_contribution",
        "output + output_column * size_i + output_row",
        "accumulated",
      ),
      "        }",
      "      }",
      "    }",
      "    if (i == j) {",
      "      const uint segment_index = placement_i - 2u;",
      *["      " + line for line in self._attribute_block_source(
        "segment_index", "size_i"
      )],
      "      for (ushort row = 0; row < size_i; ++row) {",
      "        for (ushort column = 0; column < size_i; ++column) {",
      "          float accumulated = 0.0f;",
      "          for (ushort source_i = 0; source_i < K; ++source_i) {",
      "            short source_permutation_i = local_permutations[instance * K + source_i];",
      "            if (source_permutation_i < 0) source_permutation_i = -source_permutation_i;",
      "            if (source_permutation_i != representative_i) continue;",
      "            for (ushort source_j = 0; source_j < K; ++source_j) {",
      "              short source_permutation_j = local_permutations[instance * K + source_j];",
      "              if (source_permutation_j < 0) source_permutation_j = -source_permutation_j;",
      "              if (source_permutation_j != representative_j) continue;",
      (
        f"              accumulated += hg[(raw_offsets[source_i] + row) "
        f"* {self.attribute.cols} + raw_offsets[source_j] + column];"
      ),
      "            }",
      "          }",
      "          " + _atomic_add(
        "diagonal_blocks_contribution",
        "diagonal_block_placement + row * size_i + column",
        "accumulated",
      ),
      "        }",
      "      }",
      "    }",
      "    ++valid_block;",
      "  }",
      "}",
    ]
    return lines

  def _separate_jacobian_source(self):
    hessian_rows = self.hessian_row_size
    jacobian_cols = self.jacobian_cols
    block_count = len(self.jacobian_children_sizes)
    hessian_outer = [0]
    span_outer = [0]
    for size, span in zip(
      self.jacobian_children_sizes, self.jacobian_children_spans
    ):
      hessian_outer.append(hessian_outer[-1] + size)
      span_outer.append(span_outer[-1] + span)
    lines = [
      f"constexpr ushort H = {hessian_rows};",
      f"constexpr ushort JCOLS = {jacobian_cols};",
      f"constexpr ushort LARGE_BLOCKS = {block_count};",
      (
        f"ushort span_outer[LARGE_BLOCKS + 1] = "
        f"{{{', '.join(f'{value}u' for value in span_outer)}}};"
      ),
      f"float jacobian[{hessian_rows * jacobian_cols}];",
      (
        f"for (ushort i = 0; i < {hessian_rows * jacobian_cols}; ++i) "
        "jacobian[i] = 0.0f;"
      ),
    ]
    for nonzero in range(self.jacobian_nonzero_count):
      row = self.jacobian_nonzero_positions[nonzero * 2]
      column = self.jacobian_nonzero_positions[nonzero * 2 + 1]
      lines.append(
        f"jacobian[{row * jacobian_cols + column}] = "
        f"hg[{hessian_rows * hessian_rows + nonzero}];"
      )
    lines.extend(
      [
        "ushort segment_outer[LARGE_BLOCKS + 1];",
        "for (ushort i = 0; i <= LARGE_BLOCKS; ++i) segment_outer[i] = 0;",
        "uint segment_length = 0u;",
        "ushort current_large_block = 0;",
        "for (ushort i = 0; i < K && current_large_block < LARGE_BLOCKS; ++i) {",
        "  segment_length += uint(segment_sizes[instance * K + i]);",
        "  if (segment_length == uint(span_outer[current_large_block + 1u])) {",
        "    segment_outer[current_large_block + 1u] = i + 1u;",
        "    ++current_large_block;",
        "  }",
        "}",
        "const uint coordinate_start = coordinate_outer[instance];",
        "uint valid_block = 0u;",
      ]
    )
    for large_i in range(block_count):
      for large_j in range(large_i, block_count):
        child_i = self.jacobian_children_sizes[large_i]
        child_j = self.jacobian_children_sizes[large_j]
        hessian_i = hessian_outer[large_i]
        hessian_j = hessian_outer[large_j]
        span_i = span_outer[large_i]
        span_j = span_outer[large_j]
        lines.extend(
          [
            "{",
            "  uint row_offset = 0u;",
            (
              f"  for (ushort local_i = segment_outer[{large_i}]; "
              f"local_i < segment_outer[{large_i + 1}]; ++local_i) {{"
            ),
            "    const short permutation_i = local_permutations[instance * K + local_i];",
            "    const ushort row_size = segment_sizes[instance * K + local_i];",
            "    if (permutation_i <= 0) { row_offset += row_size; continue; }",
            "    const uint placement_i = segment_indices[instance * K + local_i];",
            "    uint column_offset = 0u;",
            (
              f"    for (ushort local_j = segment_outer[{large_j}]; "
              f"local_j < segment_outer[{large_j + 1}]; ++local_j) {{"
            ),
            "      const short permutation_j = local_permutations[instance * K + local_j];",
            "      const ushort column_size = segment_sizes[instance * K + local_j];",
            "      if (permutation_j <= 0 || local_j < local_i) {",
            "        column_offset += column_size;",
            "        continue;",
            "      }",
            "      const uint placement_j = segment_indices[instance * K + local_j];",
            "      const uint output = lookups[coordinate_start + valid_block];",
            "      uint diagonal_block_placement = 0u;",
            "      if (placement_i == placement_j) {",
            "        const uint segment_index = placement_i - 2u;",
            *[
              "        "
              + line.replace(
                "const uint diagonal_block_placement =",
                "diagonal_block_placement =",
              )
              for line in self._attribute_block_source(
                "segment_index", "row_size"
              )
            ],
            "      }",
            "      for (ushort row = 0; row < row_size; ++row) {",
            "        for (ushort column = 0; column < column_size; ++column) {",
            "          float value = 0.0f;",
            f"          for (ushort inner_row = 0; inner_row < {child_i}; ++inner_row) {{",
            f"            const float left = jacobian[({hessian_i} + inner_row) * JCOLS + {span_i} + row_offset + row];",
            f"            for (ushort inner_column = 0; inner_column < {child_j}; ++inner_column) {{",
            (
              f"              value += left * hg[({hessian_i} + inner_row) "
              f"* H + {hessian_j} + inner_column] * "
              f"jacobian[({hessian_j} + inner_column) * JCOLS "
              f"+ {span_j} + column_offset + column];"
            ),
            "            }",
            "          }",
            "          if (placement_i <= placement_j) {",
            "            "
            + _atomic_add(
              "hessian_contribution",
              "output + row * column_size + column",
              "value",
            ),
            "          } else {",
            "            "
            + _atomic_add(
              "hessian_contribution",
              "output + column * row_size + row",
              "value",
            ),
            "          }",
            "          if (placement_i == placement_j && permutation_i != permutation_j) {",
            "            "
            + _atomic_add(
              "hessian_contribution",
              "output + column * row_size + row",
              "value",
            ),
            "          }",
            "          if (placement_i == placement_j) {",
            "            "
            + _atomic_add(
              "diagonal_blocks_contribution",
              "diagonal_block_placement + row * row_size + column",
              "value",
            ),
            "            if (permutation_i != permutation_j) {",
            "              "
            + _atomic_add(
              "diagonal_blocks_contribution",
              "diagonal_block_placement + column * row_size + row",
              "value",
            ),
            "            }",
            "          }",
            "        }",
            "      }",
            "      column_offset += column_size;",
            "      ++valid_block;",
            "    }",
            "    row_offset += row_size;",
            "  }",
            "}",
          ]
        )
    return lines

  def run(
    self,
    gi_kernel,
    lookups,
    gradient,
    hessian_blocks,
    diagonal,
    diagonal_blocks,
    diagonal_blocks_start,
    gradient_segments_start,
  ):
    trace = os.environ.get(
      "YASPS_METAL_HESSIAN_TRACE", ""
    ).strip().lower() in {"1", "true", "yes"}
    started = time.perf_counter() if trace else None
    unique_sizes = [
      int(size) for size in gi_kernel.outputUniqueGradientSizesCPU.tolist()
    ]
    instance_count = self.attribute.correspondance.numInstances
    if instance_count == 0:
      return
    expression_buffer_bytes = (
      instance_count * self.attribute.size * 4
    )
    stage_expression = (
      self.can_stage_expression
      and expression_buffer_bytes <= _STAGED_EXPRESSION_BUFFER_BYTES
    )
    self.update(unique_sizes, stage_expression)
    if (
      stage_expression != self._last_stage_expression
      and os.environ.get("YASPS_METAL_JIT_TRACE", "").strip().lower()
      in {"1", "true", "yes"}
    ):
      print(
        "Metal Hessian JIT "
        f"{self.attribute.fullName}: runtime expression buffer "
        f"{expression_buffer_bytes} bytes, staged {stage_expression}"
      )
    self._last_stage_expression = stage_expression
    targets = [gradient, hessian_blocks, diagonal, diagonal_blocks]
    output_shapes = [target.shape for target in targets]
    output_dtypes = [_array(target).dtype for target in targets]
    expression_inputs = (
      [self.expression.run()]
      if stage_expression
      else self.expression.resource_arrays()
    )
    for group_index, gradient_size in enumerate(unique_sizes):
      if gradient_size == 0:
        continue
      kernel_key = gradient_size if self.project_entire_hessian else 0
      kernel = self.kernels[(kernel_key, stage_expression)]
      inputs = [
        *expression_inputs,
        _array(gi_kernel.outputIndices),
        _array(gi_kernel.outputSizes),
        _array(gi_kernel.outputPermutations),
        _array(lookups),
        _array(gi_kernel.outputCompressedCoordinateCountsOuter),
        _array(gi_kernel.outputGroupedIndicesInner),
        _array(gi_kernel.outputGroupedIndicesOuter),
        mx.array([group_index], dtype=mx.uint32),
        mx.array([self.projection_method], dtype=mx.int32),
        _array(diagonal_blocks_start),
        _array(gradient_segments_start),
        mx.array(
          [gradient_segments_start.size - 1], dtype=mx.uint32
        ),
      ]
      contributions = kernel(
        inputs=inputs,
        grid=(instance_count, 1, 1),
        threadgroup=(min(instance_count, _THREADS), 1, 1),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        init_value=0.0,
      )
      _accumulate(targets, contributions)
    if trace:
      mx.eval(*[_array(target) for target in targets])
      elapsed_ms = (time.perf_counter() - started) * 1000.0
      mode = (
        "full"
        if self.project_entire_hessian
        else "separate"
        if self.separate_hessian_jacobian
        else "local"
      )
      print(
        "Metal Hessian materialization "
        f"{self.attribute.fullName}: {elapsed_ms:.3f} ms, "
        f"{instance_count} instances, projection {mode}, "
        f"gradient sizes {unique_sizes}"
      )

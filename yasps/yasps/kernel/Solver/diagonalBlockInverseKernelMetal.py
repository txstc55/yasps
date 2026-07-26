"""Metal counterpart to ``diagonalBlockInverseKernel.pyx``."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import mlx.core as mx


class MetalDiagonalBlockInverse:
  """Invert every diagonal block with the float32 symmetric EVD library."""

  def __init__(self, starts, counts, sizes):
    self.starts = tuple(int(value) for value in starts)
    self.counts = tuple(int(value) for value in counts)
    self.sizes = tuple(int(value) for value in sizes)
    total_blocks = sum(self.counts)
    lines = [
      "const uint block = thread_position_in_grid.x;",
      f"if (block >= {total_blocks}u) return;",
    ]
    block_start = 0
    for attribute, (start, count, size) in enumerate(
      zip(self.starts, self.counts, self.sizes)
    ):
      condition = "if" if attribute == 0 else "else if"
      lines.extend(
        [
          f"{condition} (block < {block_start + count}u) {{",
          f"  const uint local = block - {block_start}u;",
          f"  const uint offset = {start}u + local * {size * size}u;",
          f"  float input[{size * size}];",
          f"  float output[{size * size}];",
          (
            f"  for (ushort i = 0; i < {size * size}; ++i) "
            "input[i] = diagonal_blocks[offset + i];"
          ),
          f"  yasps_evd_abs_inverse<{size}>(input, output);",
          (
            f"  for (ushort i = 0; i < {size * size}; ++i) "
            "inverse_blocks[offset + i] = output[i];"
          ),
          "}",
        ]
      )
      block_start += count
    self.source = "\n".join(lines)
    self.header = (Path(__file__).parents[1] / "metalLinalg.metal").read_text()
    digest = sha256((self.header + self.source).encode()).hexdigest()[:16]
    self.name = f"yasps_diagonal_block_inverse_{digest}"
    self.kernel = mx.fast.metal_kernel(
      name=self.name,
      input_names=["diagonal_blocks"],
      output_names=["inverse_blocks"],
      header=self.header,
      source=self.source,
      compile_options={"math_mode": "fast"},
    )
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    source_path = output_directory / f"{self.name}.metal"
    full_source = self.header + "\n\n" + self.source
    if not source_path.exists() or source_path.read_text() != full_source:
      source_path.write_text(full_source)
    self.total_blocks = total_blocks

  def run(self, diagonal_blocks):
    if self.total_blocks == 0:
      return mx.zeros_like(diagonal_blocks._array)
    return self.kernel(
      inputs=[diagonal_blocks._array],
      grid=(self.total_blocks, 1, 1),
      threadgroup=(min(self.total_blocks, 256), 1, 1),
      output_shapes=[diagonal_blocks.shape],
      output_dtypes=[mx.float32],
      init_value=0.0,
    )[0]

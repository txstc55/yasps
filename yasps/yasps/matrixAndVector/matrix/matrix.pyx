from __future__ import annotations
import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from typing import List


from yasps.vector import vector
from yasps.context import context


_spmv_module = None
_spmv_kernel = None


def _getSpmvKernel():
  global _spmv_module, _spmv_kernel
  if _spmv_kernel is None:
    _spmv_module = SourceModule(r'''
extern "C" __global__ void spmv_blocks(
  const double* block_values,
  const unsigned int value_start,
  const unsigned int* block_positions,
  const unsigned int position_start,
  const unsigned int block_count,
  const unsigned int block_rows,
  const unsigned int block_cols,
  const double* x,
  double* y,
  const int symmetric_storage,
  const int transpose
) {
  const unsigned int block_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_index >= block_count) {
    return;
  }

  const unsigned int position_index = position_start + block_index;
  const unsigned int row = block_positions[position_index * 2];
  const unsigned int col = block_positions[position_index * 2 + 1];
  const double* block = block_values + value_start
    + block_index * block_rows * block_cols;

  if (!transpose) {
    for (unsigned int local_row = 0; local_row < block_rows; ++local_row) {
      double sum = 0.0;
      for (unsigned int local_col = 0; local_col < block_cols; ++local_col) {
        sum += block[local_row * block_cols + local_col] * x[col + local_col];
      }
      atomicAdd(y + row + local_row, sum);
    }

    if (symmetric_storage && row != col) {
      for (unsigned int local_col = 0; local_col < block_cols; ++local_col) {
        double sum = 0.0;
        for (unsigned int local_row = 0; local_row < block_rows; ++local_row) {
          sum += block[local_row * block_cols + local_col] * x[row + local_row];
        }
        atomicAdd(y + col + local_col, sum);
      }
    }
    return;
  }

  // Apply the transpose directly from the original block coordinates and
  // row-major values.  No transposed coordinate or block storage is needed.
  for (unsigned int local_col = 0; local_col < block_cols; ++local_col) {
    double sum = 0.0;
    for (unsigned int local_row = 0; local_row < block_rows; ++local_row) {
      sum += block[local_row * block_cols + local_col] * x[row + local_row];
    }
    atomicAdd(y + col + local_col, sum);
  }

  if (symmetric_storage && row != col) {
    // The transpose of the implicit mirrored block B^T is B.
    for (unsigned int local_row = 0; local_row < block_rows; ++local_row) {
      double sum = 0.0;
      for (unsigned int local_col = 0; local_col < block_cols; ++local_col) {
        sum += block[local_row * block_cols + local_col] * x[col + local_col];
      }
      atomicAdd(y + row + local_row, sum);
    }
  }
}
''')
    _spmv_kernel = _spmv_module.get_function("spmv_blocks")
  return _spmv_kernel


class matrix:
  def __init__(self, rows: int = 0, cols: int = 0, symmetric_storage: bool = False):
    if rows < 0 or cols < 0:
      raise ValueError("matrix.__init__: rows and cols must be non-negative.")
    self.__rows: int = rows
    self.__cols: int = cols
    self.__symmetric_storage: bool = bool(symmetric_storage)
    self.__context = context()

    # Block sparse representation, the static part
    self.__block_dimensions: List[int] = []  # record unique block dimensions (flattened)
    self.__blocks_flattened: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.float64)  # flattened block storage
    self.__blocks_start_indices: List[int] = []  # start index for each block size in the flattened storage
    self.__block_positions: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.uint32)  # coordinates for each block
    self.__block_counts: List[int] = []  # count for each block-size category

    # block sparse representation, the dynamic part
    self.__block_dimensions_dynamic: List[int] = []  # unique dynamic block dimensions (flattened)
    self.__blocks_flattened_dynamic: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.float64)  # flattened dynamic block storage
    self.__blocks_start_indices_dynamic: List[int] = []  # dynamic block category start indices
    self.__block_positions_dynamic: gpuarray.GPUArray = gpuarray.zeros(1, dtype=np.uint32)  # coordinates for dynamic blocks
    self.__block_counts_dynamic: List[int] = []  # count for each dynamic block-size category

  def __mul__(self, other: vector) -> vector:
    if not isinstance(other, vector):
      raise TypeError(f"matrix.__mul__: unsupported operand type(s) for *: 'matrix' and '{type(other).__name__}'")
    return self.spmv(other)

  def __matmul__(self, other: vector) -> vector:
    if not isinstance(other, vector):
      raise TypeError(f"matrix.__matmul__: unsupported operand type(s) for @: 'matrix' and '{type(other).__name__}'")
    return self.spmv(other)

  @property
  def rows(self) -> int:
    return self.__rows

  @property
  def cols(self) -> int:
    return self.__cols

  @property
  def symmetric_storage(self) -> bool:
    """Whether each stored off-diagonal block also represents its transpose."""
    return self.__symmetric_storage

  @property
  def block_dimensions(self) -> List[int]:
    """Static unique block sizes used by this matrix."""
    return self.__block_dimensions

  @block_dimensions.setter
  def block_dimensions(self, dims: List[int]) -> None:
    if not isinstance(dims, list):
      raise TypeError("matrix.block_dimensions: dims must be a list of int.")
    self.__block_dimensions = [int(x) for x in dims]

  @property
  def blocks_flattened(self) -> gpuarray.GPUArray:
    """Flattened static block storage buffer."""
    return self.__blocks_flattened

  @blocks_flattened.setter
  def blocks_flattened(self, values) -> None:
    if values is None:
      self.__blocks_flattened = gpuarray.empty(0, dtype=np.float64)
      return
    if isinstance(values, np.ndarray):
      values = gpuarray.to_gpu(np.asarray(values, dtype=np.float64).ravel())
    elif not isinstance(values, gpuarray.GPUArray):
      raise TypeError("matrix.blocks_flattened: values must be a numpy array or a GPUArray.")
    if values.dtype != np.float64:
      values = values.astype(np.float64)
    if values.ndim != 1:
      values = values.ravel()
    self.__blocks_flattened = values

  @property
  def blocks_start_indices(self) -> List[int]:
    """Start indices for each static block-size category."""
    return self.__blocks_start_indices

  @blocks_start_indices.setter
  def blocks_start_indices(self, starts: List[int]) -> None:
    if not isinstance(starts, list):
      raise TypeError("matrix.blocks_start_indices: starts must be a list of int.")
    self.__blocks_start_indices = [int(x) for x in starts]

  @property
  def block_positions(self) -> gpuarray.GPUArray:
    """Coordinates associated with each static block."""
    return self.__block_positions

  @block_positions.setter
  def block_positions(self, positions) -> None:
    if positions is None:
      self.__block_positions = gpuarray.empty(0, dtype=np.uint32)
      return
    if isinstance(positions, np.ndarray):
      positions = gpuarray.to_gpu(np.asarray(positions, dtype=np.uint32).ravel())
    elif not isinstance(positions, gpuarray.GPUArray):
      raise TypeError("matrix.block_positions: positions must be a numpy array or a GPUArray.")
    if positions.dtype != np.uint32:
      positions = positions.astype(np.uint32)
    if positions.ndim != 1:
      positions = positions.ravel()
    self.__block_positions = positions

  @property
  def block_counts(self) -> List[int]:
    """Number of blocks per static block-size category."""
    return self.__block_counts

  @block_counts.setter
  def block_counts(self, counts: List[int]) -> None:
    if not isinstance(counts, list):
      raise TypeError("matrix.block_counts: counts must be a list of int.")
    self.__block_counts = [int(x) for x in counts]

  @property
  def block_dimensions_dynamic(self) -> List[int]:
    """Dynamic unique block sizes used by this matrix."""
    return self.__block_dimensions_dynamic

  @block_dimensions_dynamic.setter
  def block_dimensions_dynamic(self, dims: List[int]) -> None:
    if not isinstance(dims, list):
      raise TypeError("matrix.block_dimensions_dynamic: dims must be a list of int.")
    self.__block_dimensions_dynamic = [int(x) for x in dims]

  @property
  def blocks_flattened_dynamic(self) -> gpuarray.GPUArray:
    """Flattened dynamic block storage buffer."""
    return self.__blocks_flattened_dynamic

  @blocks_flattened_dynamic.setter
  def blocks_flattened_dynamic(self, values) -> None:
    if values is None:
      self.__blocks_flattened_dynamic = gpuarray.empty(0, dtype=np.float64)
      return
    if isinstance(values, np.ndarray):
      values = gpuarray.to_gpu(np.asarray(values, dtype=np.float64).ravel())
    elif not isinstance(values, gpuarray.GPUArray):
      raise TypeError("matrix.blocks_flattened_dynamic: values must be a numpy array or a GPUArray.")
    if values.dtype != np.float64:
      values = values.astype(np.float64)
    if values.ndim != 1:
      values = values.ravel()
    self.__blocks_flattened_dynamic = values

  @property
  def blocks_start_indices_dynamic(self) -> List[int]:
    """Start indices for each dynamic block-size category."""
    return self.__blocks_start_indices_dynamic

  @blocks_start_indices_dynamic.setter
  def blocks_start_indices_dynamic(self, starts: List[int]) -> None:
    if not isinstance(starts, list):
      raise TypeError("matrix.blocks_start_indices_dynamic: starts must be a list of int.")
    self.__blocks_start_indices_dynamic = [int(x) for x in starts]

  @property
  def block_positions_dynamic(self) -> gpuarray.GPUArray:
    """Coordinates associated with each dynamic block."""
    return self.__block_positions_dynamic

  @block_positions_dynamic.setter
  def block_positions_dynamic(self, positions) -> None:
    if positions is None:
      self.__block_positions_dynamic = gpuarray.empty(0, dtype=np.uint32)
      return
    if isinstance(positions, np.ndarray):
      positions = gpuarray.to_gpu(np.asarray(positions, dtype=np.uint32).ravel())
    elif not isinstance(positions, gpuarray.GPUArray):
      raise TypeError("matrix.block_positions_dynamic: positions must be a numpy array or a GPUArray.")
    if positions.dtype != np.uint32:
      positions = positions.astype(np.uint32)
    if positions.ndim != 1:
      positions = positions.ravel()
    self.__block_positions_dynamic = positions

  @property
  def block_counts_dynamic(self) -> List[int]:
    """Number of blocks per dynamic block-size category."""
    return self.__block_counts_dynamic

  @block_counts_dynamic.setter
  def block_counts_dynamic(self, counts: List[int]) -> None:
    if not isinstance(counts, list):
      raise TypeError("matrix.block_counts_dynamic: counts must be a list of int.")
    self.__block_counts_dynamic = [int(x) for x in counts]

  # Backward-compatible camelCase aliases for matrix-wide representation attributes.
  @property
  def blockDimensions(self) -> List[int]:
    return self.__block_dimensions

  @blockDimensions.setter
  def blockDimensions(self, dims: List[int]) -> None:
    self.block_dimensions = dims

  @property
  def blocksFlattened(self) -> gpuarray.GPUArray:
    return self.__blocks_flattened

  @blocksFlattened.setter
  def blocksFlattened(self, values) -> None:
    self.blocks_flattened = values

  @property
  def blocksStartIndices(self) -> List[int]:
    return self.__blocks_start_indices

  @blocksStartIndices.setter
  def blocksStartIndices(self, starts: List[int]) -> None:
    self.blocks_start_indices = starts

  @property
  def blockPositions(self) -> gpuarray.GPUArray:
    return self.__block_positions

  @blockPositions.setter
  def blockPositions(self, positions) -> None:
    self.block_positions = positions

  @property
  def blockCounts(self) -> List[int]:
    return self.__block_counts

  @blockCounts.setter
  def blockCounts(self, counts: List[int]) -> None:
    self.block_counts = counts

  def __validateSpmvPart(
    self,
    block_dimensions: List[int],
    blocks_flattened: gpuarray.GPUArray,
    blocks_start_indices: List[int],
    block_positions: gpuarray.GPUArray,
    block_counts: List[int],
    part_name: str
  ) -> None:
    if len(block_dimensions) % 2 != 0:
      raise ValueError(f"matrix.spmv: {part_name} block_dimensions must contain row/column pairs.")
    num_dimensions = len(block_dimensions) // 2
    if len(block_counts) != num_dimensions:
      raise ValueError(f"matrix.spmv: {part_name} block_counts does not match block_dimensions.")
    if num_dimensions == 0:
      return
    if len(blocks_start_indices) < num_dimensions:
      raise ValueError(f"matrix.spmv: {part_name} blocks_start_indices is incomplete.")
    total_blocks = sum(block_counts)
    if block_positions.size < total_blocks * 2:
      raise ValueError(f"matrix.spmv: {part_name} block_positions is too small.")
    for index in range(num_dimensions):
      block_rows = block_dimensions[index * 2]
      block_cols = block_dimensions[index * 2 + 1]
      block_count = block_counts[index]
      if block_rows <= 0 or block_cols <= 0 or block_count < 0:
        raise ValueError(f"matrix.spmv: {part_name} block metadata must be positive.")
      required_values = (
        blocks_start_indices[index]
        + block_count * block_rows * block_cols
      )
      if required_values > blocks_flattened.size:
        raise ValueError(f"matrix.spmv: {part_name} blocks_flattened is too small.")

  def __launchSpmvPart(
    self,
    block_dimensions: List[int],
    blocks_flattened: gpuarray.GPUArray,
    blocks_start_indices: List[int],
    block_positions: gpuarray.GPUArray,
    block_counts: List[int],
    x_values: gpuarray.GPUArray,
    out_values: gpuarray.GPUArray,
    transpose: bool
  ) -> None:
    if len(block_dimensions) == 0:
      return
    kernel = _getSpmvKernel()
    position_start = 0
    for index, block_count in enumerate(block_counts):
      if block_count > 0:
        kernel(
          blocks_flattened,
          np.uint32(blocks_start_indices[index]),
          block_positions,
          np.uint32(position_start),
          np.uint32(block_count),
          np.uint32(block_dimensions[index * 2]),
          np.uint32(block_dimensions[index * 2 + 1]),
          x_values,
          out_values,
          np.int32(self.__symmetric_storage),
          np.int32(transpose),
          block=(128, 1, 1),
          grid=((block_count + 127) // 128, 1, 1)
        )
      position_start += block_count

  def spmv(self, x: vector, transpose: bool = False) -> vector:
    if not isinstance(x, vector):
      raise TypeError("matrix.spmv: x must be a yasps.vector.vector.")
    if not isinstance(transpose, (bool, np.bool_)):
      raise TypeError("matrix.spmv: transpose must be a bool.")
    input_size = self.__rows if transpose else self.__cols
    output_size = self.__cols if transpose else self.__rows
    if x.size != input_size:
      orientation = "transposed " if transpose else ""
      raise ValueError(
        f"matrix.spmv: input vector for {orientation}matrix has size "
        f"{x.size}, expected {input_size}."
      )
    result = vector(output_size)
    self.spmvInPlace(x, result, transpose)
    return result

  def spmvInPlace(self, x: vector, out: vector, transpose: bool = False) -> None:
    if not isinstance(x, vector) or not isinstance(out, vector):
      raise TypeError("matrix.spmvInPlace: x and out must be yasps.vector.vector.")
    if not isinstance(transpose, (bool, np.bool_)):
      raise TypeError("matrix.spmvInPlace: transpose must be a bool.")
    input_size = self.__rows if transpose else self.__cols
    output_size = self.__cols if transpose else self.__rows
    if x.size != input_size:
      orientation = "transposed " if transpose else ""
      raise ValueError(
        f"matrix.spmvInPlace: input vector for {orientation}matrix has size "
        f"{x.size}, expected {input_size}."
      )
    if out.size != output_size:
      orientation = "transposed " if transpose else ""
      raise ValueError(
        f"matrix.spmvInPlace: output vector for {orientation}matrix has size "
        f"{out.size}, expected {output_size}."
      )

    self.__context.useDefaultContext()
    if self.__rows == 0 or self.__cols == 0:
      out.value.fill(0)
      return

    self.__validateSpmvPart(
      self.__block_dimensions,
      self.__blocks_flattened,
      self.__blocks_start_indices,
      self.__block_positions,
      self.__block_counts,
      "static"
    )
    self.__validateSpmvPart(
      self.__block_dimensions_dynamic,
      self.__blocks_flattened_dynamic,
      self.__blocks_start_indices_dynamic,
      self.__block_positions_dynamic,
      self.__block_counts_dynamic,
      "dynamic"
    )

    x_values = x.value
    if int(x_values.gpudata) == int(out.value.gpudata):
      x_values = x_values.copy()
    out.value.fill(0)
    self.__launchSpmvPart(
      self.__block_dimensions,
      self.__blocks_flattened,
      self.__blocks_start_indices,
      self.__block_positions,
      self.__block_counts,
      x_values,
      out.value,
      transpose
    )
    self.__launchSpmvPart(
      self.__block_dimensions_dynamic,
      self.__blocks_flattened_dynamic,
      self.__blocks_start_indices_dynamic,
      self.__block_positions_dynamic,
      self.__block_counts_dynamic,
      x_values,
      out.value,
      transpose
    )

  def matVecProduct(self, x: vector, transpose: bool = False) -> vector:
    return self.spmv(x, transpose)

  def matVecProductInPlace(self, x: vector, out: vector, transpose: bool = False) -> None:
    self.spmvInPlace(x, out, transpose)

  def setDimensions(self, rows: int, cols: int) -> None:
    if rows < 0 or cols < 0:
      raise ValueError("matrix.setDimensions: rows and cols must be non-negative.")
    self.__rows = rows
    self.__cols = cols

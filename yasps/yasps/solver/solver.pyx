# cython: language_level=3
from __future__ import annotations
from typing import Optional

import numpy as np
import pycuda.gpuarray as gpuarray

from yasps.solverKernel import solverKernel
from yasps.diagonalBlockInverseKernel import diagonalBlockInverseKernel
from yasps.vector import vector


class solver:
  def __init__(self):
    self.__solverKernel: Optional[solverKernel] = None
    self.__d_p1_b: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_r: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_c: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_q: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_s: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__cg_scalars: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__solution: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__diagonalBlockInverseKernel: Optional[diagonalBlockInverseKernel] = None
    self.__diagonalBlockLayout = None

  @property
  def solution(self) -> gpuarray.GPUArray:
    return self.__solution

  def reset(self) -> None:
    if self.__solverKernel is not None:
      # Keep cleanup explicit: a Python finalizer may run after CUDA teardown.
      self.__solverKernel.cleanupStreams()
    self.__solverKernel = None
    self.__d_p1_b = gpuarray.empty(0, dtype=np.float64)
    self.__d_r = gpuarray.empty(0, dtype=np.float64)
    self.__d_c = gpuarray.empty(0, dtype=np.float64)
    self.__d_q = gpuarray.empty(0, dtype=np.float64)
    self.__d_s = gpuarray.empty(0, dtype=np.float64)
    self.__cg_scalars = gpuarray.empty(0, dtype=np.float64)
    self.__solution = gpuarray.empty(0, dtype=np.float64)
    self.__diagonalBlockInverseKernel = None
    self.__diagonalBlockLayout = None

  def __computeDiagonalBlockInverse(self, active_hessian) -> None:
    block_counts = tuple(item.correspondance.numInstances for item in active_hessian.wrt)
    block_sizes = tuple(item.size for item in active_hessian.wrt)
    block_starts = tuple(active_hessian.diagonal_blocks_start_cpu)
    layout = (block_starts, block_counts, block_sizes)
    if self.__diagonalBlockInverseKernel is None or self.__diagonalBlockLayout != layout:
      self.__diagonalBlockInverseKernel = diagonalBlockInverseKernel(set(block_sizes), list(block_starts), list(block_counts), list(block_sizes), len(block_sizes))
      self.__diagonalBlockLayout = layout
    if active_hessian.diagonal_blocks.size > 0:
      self.__diagonalBlockInverseKernel.computeDiagonalBlockInverse(active_hessian.diagonal_blocks, active_hessian.diagonal_blocks_inverse)

  def __ensureBuffers(self, active_hessian, right_hand_side: vector) -> None:
    values = right_hand_side.value
    if values.size == 0:
      return
    if self.__solution.size == values.size and self.__solverKernel is not None:
      return
    if active_hessian is None:
      return

    if self.__solverKernel is not None:
      self.__solverKernel.cleanupStreams()
    self.__solverKernel = solverKernel(active_hessian.block_dimensions + active_hessian.block_dimensions_dynamic)
    self.__d_p1_b = gpuarray.empty(values.shape, dtype=np.float64)
    self.__d_r = gpuarray.empty(values.shape, dtype=np.float64)
    self.__d_c = gpuarray.empty(values.shape, dtype=np.float64)
    self.__d_q = gpuarray.empty(values.shape, dtype=np.float64)
    self.__d_s = gpuarray.empty(values.shape, dtype=np.float64)
    # Persistent device-side CG recurrence state; see CgScalarSlot in solverKernel.
    self.__cg_scalars = gpuarray.empty(8, dtype=np.float64)
    self.__solution = gpuarray.empty(values.shape, dtype=np.float64)

  def computeSolution(self, active_hessian, right_hand_side: vector, initial_guess, tolerance = 1e-3, maxIterations = 20000, zero_initial_guess = False):
    if not isinstance(right_hand_side, vector):
      raise TypeError("solver.computeSolution: right_hand_side must be a yasps.vector.vector.")
    values = right_hand_side.value
    if values.size == 0:
      return 0
    if active_hessian is None:
      self.__ensureBuffers(active_hessian, right_hand_side)
      if self.__solution.size > 0:
        self.__solution.fill(0)
      return 0

    if right_hand_side.size != active_hessian.rows:
      raise ValueError(
        "solver.computeSolution: right_hand_side size must match the Hessian size."
      )

    self.__computeDiagonalBlockInverse(active_hessian)
    self.__ensureBuffers(active_hessian, right_hand_side)
    assert self.__solverKernel is not None
    self.__solverKernel.updateBlockDimensions(active_hessian.block_dimensions + active_hessian.block_dimensions_dynamic)

    return self.__solverKernel.computeSolution(
      maxIterations,
      tolerance,
      active_hessian.blocks_flattened,
      active_hessian.block_positions,
      active_hessian.blocks_start_indices,
      active_hessian.block_counts,
      active_hessian.block_dimensions,
      active_hessian.blocks_flattened_dynamic,
      active_hessian.block_positions_dynamic,
      active_hessian.blocks_start_indices_dynamic,
      active_hessian.block_counts_dynamic,
      active_hessian.block_dimensions_dynamic,
      active_hessian.diagonal,
      active_hessian.diagonal_blocks_inverse,
      active_hessian.diagonal_blocks_start_cpu,
      [item.correspondance.numInstances for item in active_hessian.wrt],
      [item.size for item in active_hessian.wrt],
      active_hessian.gradient_segments_start_cpu,
      len(active_hessian.wrt),
      values,
      self.__d_p1_b,
      self.__d_r,
      self.__d_c,
      self.__d_q,
      self.__d_s,
      self.__solution,
      initial_guess,
      zero_initial_guess=zero_initial_guess,
      cg_scalars=self.__cg_scalars
    )

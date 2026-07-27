# cython: language_level=3
from __future__ import annotations
from typing import Optional

import numpy as np
from yasps.backend import gpuarray

from yasps.solverKernel import solverKernel


class solver:
  def __init__(self):
    self.__solverKernel: Optional[solverKernel] = None
    self.__d_p1_b: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_r: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_c: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_q: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_s: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__solution: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)

  @property
  def solution(self) -> gpuarray.GPUArray:
    return self.__solution

  def reset(self) -> None:
    self.__solverKernel = None
    self.__d_p1_b = gpuarray.empty(0, dtype=np.float64)
    self.__d_r = gpuarray.empty(0, dtype=np.float64)
    self.__d_c = gpuarray.empty(0, dtype=np.float64)
    self.__d_q = gpuarray.empty(0, dtype=np.float64)
    self.__d_s = gpuarray.empty(0, dtype=np.float64)
    self.__solution = gpuarray.empty(0, dtype=np.float64)

  def __ensureBuffers(self, active_hessian, gradient_object) -> None:
    gradient = gradient_object.value
    if gradient.size == 0:
      return
    if self.__solution.size == gradient.size and self.__solverKernel is not None:
      return
    if active_hessian is None:
      return

    self.__solverKernel = solverKernel(active_hessian.block_dimensions + active_hessian.block_dimensions_dynamic)
    self.__d_p1_b = gpuarray.empty(gradient.shape, dtype=np.float64)
    self.__d_r = gpuarray.empty(gradient.shape, dtype=np.float64)
    self.__d_c = gpuarray.empty(gradient.shape, dtype=np.float64)
    self.__d_q = gpuarray.empty(gradient.shape, dtype=np.float64)
    self.__d_s = gpuarray.empty(gradient.shape, dtype=np.float64)
    self.__solution = gpuarray.empty(gradient.shape, dtype=np.float64)
    self.__solution.fill(0)

  def computeSolution(self, active_hessian, wrt, gradient_object, initial_guess, tolerance = 1e-3, maxIterations = 20000):
    gradient = gradient_object.value
    if gradient.size == 0:
      return 0
    if active_hessian is None:
      self.__ensureBuffers(active_hessian, gradient_object)
      if self.__solution.size > 0:
        self.__solution.fill(0)
      return 0

    self.__ensureBuffers(active_hessian, gradient_object)
    assert self.__solverKernel is not None
    self.__solverKernel.updateBlockDimensions(active_hessian.block_dimensions + active_hessian.block_dimensions_dynamic)

    self.__d_r.fill(0)
    self.__d_c.fill(0)
    self.__d_q.fill(0)
    self.__d_s.fill(0)
    self.__solution.fill(0)

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
      [item.correspondance.numInstances for item in wrt],
      [item.size for item in wrt],
      gradient_object.gradient_segments_start_cpu,
      len(wrt),
      gradient,
      self.__d_p1_b,
      self.__d_r,
      self.__d_c,
      self.__d_q,
      self.__d_s,
      self.__solution,
      initial_guess
    )

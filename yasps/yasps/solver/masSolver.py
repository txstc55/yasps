"""YASPS-facing multilevel additive Schwarz PCG solver."""

from __future__ import annotations

import numpy as np
import pycuda.gpuarray as gpuarray

from .mas.cuda_runtime import is_pycuda_array
from .mas.solver import MASSolver
from .yaspsMatrixView import YASPSMatrixView


class masSolver:
  """Solve a YASPS matrix or Hessian with MAS-preconditioned CG.

  The static block graph builds the METIS hierarchy once unless explicitly
  supplied via rebuildHierarchy. No static-sparsity comparison rebuilds it.
  Call rebuildHierarchy (or reset) when the desired partition changes. Static
  operator coordinates must remain fixed between these calls. Current static
  values and all dynamic blocks are assembled on every solve, so changing
  collision connectivity is reflected without rerunning static partitioning.
  """

  def __init__(self, **options):
    defaults = {
      "inverse_backend": "cooperative_gauss_jordan",
      "allow_cpu_fallback": False,
    }
    defaults.update(options)
    self.__solver = MASSolver(**defaults)
    self.__view = None
    self.__active_matrix = None
    self.__statistics = {}
    self.__empty_solution = gpuarray.empty(0, dtype=np.float64)

  @property
  def solution(self):
    solution = self.__solver.device_solution
    return self.__empty_solution if solution is None else solution

  @property
  def statistics(self) -> dict:
    return dict(self.__statistics)

  @property
  def hierarchy(self):
    return self.__solver.hierarchy

  @property
  def implementation(self):
    return self.__solver

  def reset(self):
    self.__solver.reset()
    self.__view = None
    self.__active_matrix = None
    self.__statistics = {}

  def rebuildHierarchy(self, block_positions, block_dimensions, num_blocks):
    """Build a partition from two GPU arrays, each of length 2*num_blocks.

    Positions contain global scalar (row, column) starts; dimensions contain
    one (rows, columns) pair per block, not one pair per compressed category.
    Include every variable, including isolated diagonal blocks. Only these
    coordinates are read; no dummy Hessian values are allocated or assembled.
    METIS still runs on the CPU after downloading the graph metadata.
    """
    if not is_pycuda_array(block_positions) or not is_pycuda_array(block_dimensions):
      raise TypeError("rebuildHierarchy requires two GPU arrays")
    hierarchy = self.__solver.rebuild_hierarchy_from_blocks(block_positions, block_dimensions, num_blocks)
    # The next real matrix must build a fresh view and numerical scatter maps.
    self.__view = None
    self.__active_matrix = None
    self.__statistics = {}
    return hierarchy

  def __updateView(self, active_matrix):
    if self.__view is None:
      self.__view = YASPSMatrixView(active_matrix)
    elif active_matrix is self.__active_matrix:
      self.__view.update_numeric(active_matrix)
    else:
      candidate = YASPSMatrixView(active_matrix)
      # A different numerical matrix needs fresh scatter descriptors, but it
      # must not replace an explicitly supplied partition graph.
      self.__solver.invalidate_numeric_state()
      self.__view = candidate
    self.__active_matrix = active_matrix
    return self.__view

  def computeSolution(
    self,
    active_matrix,
    right_hand_side,
    initial_guess,
    tolerance=1e-3,
    maxIterations=20_000,
    zero_initial_guess=False,
  ):
    rhs = (
      right_hand_side.value
      if hasattr(right_hand_side, "value") else right_hand_side
    )
    if not is_pycuda_array(rhs):
      raise TypeError("masSolver requires a device-resident right-hand side")
    if active_matrix is None:
      self.__empty_solution = gpuarray.zeros(rhs.shape, dtype=np.float64)
      self.__statistics = {
        "solver": "mas",
        "converged": True,
        "iterations": 0,
        "solve_seconds": 0.0,
      }
      return 0
    if int(rhs.size) != int(active_matrix.rows):
      raise ValueError(
        "masSolver right-hand side size must match the matrix size"
      )
    guess = None
    if not zero_initial_guess and initial_guess is not None:
      guess = (
        initial_guess.value
        if hasattr(initial_guess, "value") else initial_guess
      )
      if not is_pycuda_array(guess) or int(guess.size) != int(rhs.size):
        raise ValueError(
          "masSolver initial guess must be a matching device array"
        )

    view = self.__updateView(active_matrix)
    self.__solver.solve(
      view,
      rhs,
      initial_guess=guess,
      tolerance=float(tolerance),
      max_iterations=int(maxIterations),
    )
    stats = self.__solver.statistics
    compact = stats.as_dict()
    compact.pop("domain_scalar_sizes", None)
    result = 0 if stats.converged else -1000 - int(stats.iterations)
    if not stats.converged:
      reason = stats.breakdown or ""
      # These are public MAS codes, not the CUDA recurrence's status values.
      # Counts (including residual restarts) remain in statistics.iterations.
      if reason.startswith("preconditioned residual stagnated"):
        result = -5
      elif reason == "preconditioned residual diverged":
        result = -6
      elif "not positive definite" in reason or reason.startswith("non-positive curvature"):
        result = -4
      elif reason not in ("", "CG iteration limit reached"):
        result = -7  # Residual verification failure or another named breakdown.
    self.__statistics = compact | {
      "solver": "mas",
      "result": result,
      "metis_seconds": float(sum(stats.metis_seconds_per_level)),
      "matrix_size": int(active_matrix.rows),
      "tolerance": float(tolerance),
    }
    return result

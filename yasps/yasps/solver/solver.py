"""Runtime selection between YASPS linear solver implementations."""

from __future__ import annotations

from .jacobianPCGSolver import jacobianPCGSolver
from .masSolver import masSolver


class solver:
  """Linear solver dispatcher.

  Parameters
  ----------
  solver:
    ``"mas"`` (the default) for multilevel additive Schwarz PCG or
    ``"jacobian"`` for the original block-Jacobi PCG implementation.
  **options:
    Options forwarded to the selected implementation. The Jacobi solver has
    no configuration options; MAS options are accepted by :class:`masSolver`.
  """

  def __init__(self, solver="mas", **options):
    self.__solver_name = ""
    self.__implementation = None
    self.setSolver(solver, **options)

  @property
  def solverName(self) -> str:
    return self.__solver_name

  @property
  def implementation(self):
    return self.__implementation

  @property
  def solution(self):
    return self.__implementation.solution

  @property
  def statistics(self) -> dict:
    return self.__implementation.statistics

  def setSolver(self, solver="mas", **options):
    if not isinstance(solver, str):
      raise TypeError("solver must be either 'jacobian' or 'mas'")
    name = solver.strip().lower()
    if name == "jacobian":
      if options:
        unexpected = ", ".join(sorted(options))
        raise TypeError(
          f"jacobianPCGSolver does not accept options: {unexpected}"
        )
      implementation = jacobianPCGSolver()
    elif name == "mas":
      implementation = masSolver(**options)
    else:
      raise ValueError("solver must be either 'jacobian' or 'mas'")
    if self.__implementation is not None:
      self.__implementation.reset()
    self.__solver_name = name
    self.__implementation = implementation
    return self

  def reset(self):
    self.__implementation.reset()

  def rebuildHierarchy(self, block_positions, block_dimensions, num_blocks):
    """Explicit MAS graph rebuild from two GPU arrays and a block count."""
    if self.__solver_name != "mas":
      raise ValueError("rebuildHierarchy is only available for the MAS solver")
    return self.__implementation.rebuildHierarchy(block_positions, block_dimensions, num_blocks)

  def computeSolution(self, *args, **kwargs):
    """Return 0 on convergence; negative codes indicate failure.

    MAS: -4 is a definiteness/curvature breakdown, -5 stagnation, -6 divergence,
    -7 residual verification failure or another named breakdown, and -8 local
    block inversion failure (no solution; details in statistics.breakdown).
    Iteration limits return -1000-iteration. Counts include MAS residual restarts.
    Jacobi: iterative breakdowns and limits return -1000-iteration; -5 remains
    its invalid-initial-residual/tolerance code. Other setup/CUDA exceptions remain.
    Decode counts only for code <= -1000. Otherwise read statistics.iterations;
    statistics.breakdown records the reason for either solver.
    """
    return self.__implementation.computeSolution(*args, **kwargs)

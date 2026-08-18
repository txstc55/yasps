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

  def computeSolution(self, *args, **kwargs):
    return self.__implementation.computeSolution(*args, **kwargs)

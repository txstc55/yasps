"""Matrix-free preconditioned conjugate gradient."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PCGResult:
  solution: np.ndarray
  iterations: int
  final_residual: float
  relative_residual: float
  converged: bool
  seconds: float
  breakdown: str | None = None


def pcg(
  matvec: Callable[[np.ndarray], np.ndarray],
  right_hand_side: np.ndarray,
  precondition: Callable[[np.ndarray], np.ndarray] | None = None,
  initial_guess: np.ndarray | None = None,
  tolerance: float = 1e-3,
  max_iterations: int = 20_000,
) -> PCGResult:
  if tolerance <= 0 or max_iterations < 0:
    raise ValueError("tolerance must be positive and max_iterations non-negative")
  started = perf_counter()
  rhs = np.asarray(right_hand_side, dtype=np.float64).reshape(-1)
  x = np.zeros_like(rhs) if initial_guess is None else np.asarray(initial_guess, dtype=np.float64).reshape(-1).copy()
  if x.size != rhs.size:
    raise ValueError("initial guess has the wrong scalar size")
  residual = rhs - matvec(x)
  rhs_norm = float(np.linalg.norm(rhs))
  denominator = max(rhs_norm, np.finfo(np.float64).tiny)
  residual_norm = float(np.linalg.norm(residual))
  z = residual.copy() if precondition is None else precondition(residual)
  rz = float(np.dot(residual, z))
  if initial_guess is None:
    reference_rz = rz
  else:
    reference_z = rhs.copy() if precondition is None else precondition(rhs)
    reference_rz = float(np.dot(rhs, reference_z))
  if (np.isfinite(rz) and rz >= 0.0 and
      np.isfinite(reference_rz) and reference_rz > 0.0 and
      rz <= tolerance * reference_rz):
    return PCGResult(
      x, 0, residual_norm, residual_norm / denominator, True,
      perf_counter() - started,
    )
  if max_iterations == 0:
    return PCGResult(
      x, 0, residual_norm, residual_norm / denominator, False,
      perf_counter() - started,
    )
  if (not np.isfinite(rz) or rz <= 0.0 or
      not np.isfinite(reference_rz) or reference_rz <= 0.0):
    return PCGResult(
      x, 0, residual_norm, residual_norm / denominator, False,
      perf_counter() - started, "preconditioner is not positive definite"
    )
  direction = z.copy()
  for iteration in range(1, max_iterations + 1):
    product = matvec(direction)
    curvature = float(np.dot(direction, product))
    if not np.isfinite(curvature) or curvature <= 0:
      return PCGResult(
        x, iteration - 1, residual_norm, residual_norm / denominator, False,
        perf_counter() - started, "matrix is not positive definite"
      )
    alpha = rz / curvature
    x += alpha * direction
    residual -= alpha * product
    residual_norm = float(np.linalg.norm(residual))
    z = residual.copy() if precondition is None else precondition(residual)
    next_rz = float(np.dot(residual, z))
    if not np.isfinite(next_rz) or next_rz < 0.0:
      return PCGResult(
        x, iteration, residual_norm, residual_norm / denominator, False,
        perf_counter() - started, "preconditioner is not positive definite"
      )
    if next_rz <= tolerance * reference_rz:
      return PCGResult(
        x, iteration, residual_norm, residual_norm / denominator, True,
        perf_counter() - started,
      )
    if next_rz == 0.0:
      return PCGResult(
        x, iteration, residual_norm, residual_norm / denominator, False,
        perf_counter() - started, "preconditioner is not positive definite"
      )
    direction = z + (next_rz / rz) * direction
    rz = next_rz
  return PCGResult(
    x, max_iterations, residual_norm, residual_norm / denominator, False,
    perf_counter() - started
  )

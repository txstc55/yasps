"""Explicit local inverse backends for heterogeneous domain sizes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .jit import compile_cuda_library


class LocalInverseError(np.linalg.LinAlgError):
  """A local matrix is singular, indefinite, or failed device inversion."""


@dataclass
class InverseBuildResult:
  inverses: list[np.ndarray]
  seconds: float
  backend: str
  bucket_sizes: list[int]
  device_bytes: int = 0
  fallback_reason: str | None = None
  kernel_metrics: dict = None

  def __post_init__(self):
    if self.kernel_metrics is None:
      self.kernel_metrics = {}


class LocalInverseBackend(ABC):
  @abstractmethod
  def build(self, local_matrices: list[np.ndarray], local_sizes=None) -> InverseBuildResult:
    raise NotImplementedError


class CPUReferenceInverseBackend(LocalInverseBackend):
  """FP64 NumPy correctness oracle, not the production inverse builder."""

  def __init__(self, pivot_tolerance: float = 1e-12, require_spd: bool = True):
    self.pivot_tolerance = float(pivot_tolerance)
    self.require_spd = bool(require_spd)

  def build(self, local_matrices: list[np.ndarray], local_sizes=None) -> InverseBuildResult:
    started = perf_counter()
    inverses: list[np.ndarray] = []
    sizes: list[int] = []
    for domain, raw in enumerate(local_matrices):
      matrix = np.asarray(raw, dtype=np.float64)
      if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise LocalInverseError(f"domain {domain} is not a non-empty square matrix")
      if not np.all(np.isfinite(matrix)):
        raise LocalInverseError(f"domain {domain} contains non-finite values")
      symmetric = 0.5 * (matrix + matrix.T)
      scale = max(1.0, float(np.linalg.norm(symmetric, ord=np.inf)))
      try:
        if self.require_spd:
          if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=self.pivot_tolerance * scale):
            raise LocalInverseError(f"domain {domain} is not symmetric")
          factor = np.linalg.cholesky(symmetric)
          if np.min(np.diag(factor)) ** 2 <= self.pivot_tolerance * scale:
            raise LocalInverseError(f"domain {domain} has a numerically zero Cholesky pivot")
          identity = np.eye(matrix.shape[0], dtype=np.float64)
          inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
        else:
          if np.linalg.cond(matrix) >= 1.0 / self.pivot_tolerance:
            raise LocalInverseError(f"domain {domain} is singular to the configured tolerance")
          inverse = np.linalg.inv(matrix)
      except np.linalg.LinAlgError as error:
        raise LocalInverseError(f"domain {domain} inversion failed: {error}") from error
      if not np.all(np.isfinite(inverse)):
        raise LocalInverseError(f"domain {domain} inversion produced non-finite values")
      inverses.append(inverse)
      sizes.append(matrix.shape[0])
    return InverseBuildResult(inverses, perf_counter() - started, "cpu_reference", sizes)


def bucket_size(size: int) -> int:
  if size <= 0:
    raise ValueError("local matrix size must be positive")
  # Eight-wide storage keeps generated inverse kernels naturally aligned;
  # the bucket sequence is unbounded and contains no matrix-size policy.
  return ((size + 7) // 8) * 8


class CUDALocalInverseBackend(LocalInverseBackend):
  """Lazy PyCUDA wrapper for cooperative FP64 explicit inversion.

  ``algorithm='spd'`` uses cooperative Cholesky and solves against identity.
  ``algorithm='gauss_jordan'`` is the StiffGIPC-style comparison baseline.
  """

  def __init__(self, algorithm: str = "spd", threads_per_block: int = 96, pivot_tolerance: float = 1e-12):
    if algorithm not in ("spd", "gauss_jordan"):
      raise ValueError("CUDA inverse algorithm must be 'spd' or 'gauss_jordan'")
    if threads_per_block not in (64, 96, 128):
      raise ValueError("threads_per_block must be 64, 96, or 128")
    self.algorithm = algorithm
    self.threads_per_block = int(threads_per_block)
    self.pivot_tolerance = float(pivot_tolerance)
    self.device_batches = []

  def build(self, local_matrices: list[np.ndarray], local_sizes=None) -> InverseBuildResult:
    try:
      import pycuda.autoinit  # noqa: F401 - owns the lazy default context
      import pycuda.driver as cuda
      import pycuda.gpuarray as gpuarray
    except Exception as error:  # pragma: no cover - depends on the CUDA host
      raise RuntimeError(f"CUDA initialization failed: {error}") from error

    source_path = Path(__file__).resolve().parent / "cuda" / "local_inverse.cu"
    if not source_path.exists():  # pragma: no cover - wheel packaging safeguard
      raise RuntimeError(f"CUDA source is missing: {source_path}")
    source = source_path.read_text(encoding="utf8")
    kernel_name = "yasps_mas_inverse_spd" if self.algorithm == "spd" else "yasps_mas_inverse_gauss_jordan"
    module = compile_cuda_library(
      cuda, source, kernel_names=(kernel_name,),
      label=f"reference_inverse_{self.algorithm}",
    )
    kernel = module.kernel(kernel_name)
    function_attributes = {
      "registers_per_thread": 0,
      "static_shared_bytes": 0,
      "local_bytes_per_thread": 0,
      "jit_compile_seconds": float(module.compile_seconds),
      "jit_cache_hit": bool(module.cache_hit),
      "jit_library": str(module.path),
    }

    started = perf_counter()
    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for index, raw in enumerate(local_matrices):
      matrix = np.asarray(raw, dtype=np.float64)
      if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise LocalInverseError(f"domain {index} is not a non-empty square matrix")
      grouped.setdefault(bucket_size(matrix.shape[0]), []).append((index, matrix))
    results: list[np.ndarray | None] = [None] * len(local_matrices)
    total_device_bytes = 0
    self.device_batches = []
    any_failure = gpuarray.zeros(1, dtype=np.int32)
    for padded_size, entries in sorted(grouped.items()):
      count = len(entries)
      padded = np.zeros((count, padded_size, padded_size), dtype=np.float64)
      sizes = np.empty(count, dtype=np.int32)
      for batch, (_, matrix) in enumerate(entries):
        size = matrix.shape[0]
        sizes[batch] = size
        padded[batch, :size, :size] = matrix
        if size < padded_size:
          padded[batch, size:, size:] = np.eye(padded_size - size)
      device_input = gpuarray.to_gpu(padded.reshape(-1))
      device_output = gpuarray.empty_like(device_input)
      device_sizes = gpuarray.to_gpu(sizes)
      device_status = gpuarray.zeros(count, dtype=np.int32)
      shared_bytes = (2 * padded_size * padded_size + padded_size) * 8
      default_shared = int(cuda.Context.get_device().get_attribute(
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
      ))
      if shared_bytes > default_shared:
        opt_in_shared = int(cuda.Context.get_device().get_attribute(
          cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        ))
        if shared_bytes > opt_in_shared:
          raise ValueError(
            f"local inverse needs {shared_bytes} shared bytes, but "
            f"this GPU supports only {opt_in_shared} opt-in bytes"
          )
        kernel.set_max_dynamic_shared_bytes(shared_bytes)
      kernel(
        device_input,
        device_output,
        device_sizes,
        device_status,
        np.int32(padded_size),
        np.float64(self.pivot_tolerance),
        any_failure,
        block=(self.threads_per_block, 1, 1),
        grid=(count, 1, 1),
        shared=shared_bytes,
      )
      cuda.Context.synchronize()
      status = device_status.get()
      if np.any(status):
        failures = [entries[i][0] for i in np.flatnonzero(status)]
        raise LocalInverseError(f"CUDA {self.algorithm} inversion failed for domains {failures}")
      host_output = device_output.get().reshape(count, padded_size, padded_size)
      for batch, (original_index, matrix) in enumerate(entries):
        size = matrix.shape[0]
        results[original_index] = host_output[batch, :size, :size].copy()
      total_device_bytes += device_input.nbytes + device_output.nbytes + device_sizes.nbytes + device_status.nbytes
      self.device_batches.append((padded_size, entries, device_output, device_sizes))
    unique_buckets = sorted({bucket_size(np.asarray(matrix).shape[0]) for matrix in local_matrices})
    device = cuda.Context.get_device()
    max_threads_sm = int(device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR))
    max_shared_sm = int(device.get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
    max_registers_sm = int(device.get_attribute(cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR))
    max_blocks_sm = int(device.get_attribute(cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR))
    registers = function_attributes["registers_per_thread"]
    occupancy = {}
    for padded_size in unique_buckets:
      dynamic_shared = (2 * padded_size * padded_size + padded_size) * 8
      limits = [max_blocks_sm, max_threads_sm // self.threads_per_block]
      limits.append(max_shared_sm // max(1, dynamic_shared + function_attributes["static_shared_bytes"]))
      if registers:
        limits.append(max_registers_sm // (registers * self.threads_per_block))
      active_blocks = max(0, min(limits))
      occupancy[str(padded_size)] = min(1.0, active_blocks * self.threads_per_block / max_threads_sm)
    function_attributes.update({
      "dynamic_shared_bytes_by_bucket": {
        str(size): (2 * size * size + size) * 8 for size in unique_buckets
      },
      "synchronizations_per_matrix_by_bucket": {
        str(size): (3 * size + 1 if self.algorithm == "gauss_jordan" else 2 * size + 2)
        for size in unique_buckets
      },
      "occupancy_upper_bound_by_bucket": occupancy,
    })
    total_device_bytes += any_failure.nbytes
    return InverseBuildResult(
      [item for item in results if item is not None],
      perf_counter() - started,
      f"cuda_{self.algorithm}",
      [bucket_size(np.asarray(matrix).shape[0]) for matrix in local_matrices],
      total_device_bytes,
      kernel_metrics=function_attributes,
    )


class PreferredInverseBackend(LocalInverseBackend):
  """Production CUDA backend with an explicit, reported CPU fallback."""

  def __init__(self, algorithm: str, threads_per_block: int = 96, allow_cpu_fallback: bool = True):
    self.cuda = CUDALocalInverseBackend(algorithm, threads_per_block)
    self.cpu = CPUReferenceInverseBackend(require_spd=(algorithm == "spd"))
    self.allow_cpu_fallback = bool(allow_cpu_fallback)

  def build(self, local_matrices: list[np.ndarray], local_sizes=None) -> InverseBuildResult:
    try:
      return self.cuda.build(local_matrices, local_sizes)
    except (RuntimeError, OSError) as error:
      if not self.allow_cpu_fallback:
        raise
      result = self.cpu.build(local_matrices, local_sizes)
      result.backend = "cpu_reference_fallback"
      result.fallback_reason = str(error)
      return result


def make_inverse_backend(name: str, *, threads_per_block: int = 96, allow_cpu_fallback: bool = True) -> LocalInverseBackend:
  aliases = {
    "cpu_reference": lambda: CPUReferenceInverseBackend(),
    "cooperative_gauss_jordan": lambda: PreferredInverseBackend("gauss_jordan", threads_per_block, allow_cpu_fallback),
    "cooperative_spd_inverse": lambda: PreferredInverseBackend("spd", threads_per_block, allow_cpu_fallback),
  }
  try:
    return aliases[name]()
  except KeyError as error:
    raise ValueError(f"unknown inverse backend {name!r}; choose one of {sorted(aliases)}") from error

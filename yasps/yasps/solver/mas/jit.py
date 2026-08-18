"""Persistent NVCC JIT cache for dimension-specialized MAS kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter

import numpy as np


_LIBRARY_CACHE: dict[tuple[int, str], "CUDAJITLibrary"] = {}


def spmv_warps_for_shape(rows: int, cols: int) -> int:
  """Choose generated warp grouping from work and register pressure."""
  rows, cols = int(rows), int(cols)
  if rows <= 0 or cols <= 0:
    raise ValueError("block dimensions must be positive")
  available = min(8, max(1, 32 // max(rows, cols)))
  return 1 << (available.bit_length() - 1)


def _cache_root() -> Path:
  configured = os.environ.get("YASPS_MAS_JIT_CACHE")
  if configured:
    return Path(configured).expanduser().resolve()
  return Path.cwd() / ".yasps_constant" / "mas_jit"


def _device_architecture(cuda) -> str:
  major, minor = cuda.Context.get_device().compute_capability()
  return f"sm_{int(major)}{int(minor)}"


def _launcher_source(kernel_names: tuple[str, ...]) -> str:
  cases = "\n".join(
    f"    case {index}u:\n"
    f"      return static_cast<int>(cudaLaunchKernel(\n"
    f"          reinterpret_cast<const void*>({name}), grid, block, arguments,\n"
    f"          shared_bytes, reinterpret_cast<cudaStream_t>(stream)));"
    for index, name in enumerate(kernel_names)
  )
  attribute_cases = "\n".join(
    f"    case {index}u:\n"
    f"      return static_cast<int>(cudaFuncSetAttribute(\n"
    f"          reinterpret_cast<const void*>({name}),\n"
    f"          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));"
    for index, name in enumerate(kernel_names)
  )
  return f"""
#include <cuda_runtime.h>

extern "C" int yasps_mas_launch_kernel(
    unsigned int kernel_index,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int shared_bytes, void* stream, void** arguments) {{
  const dim3 grid(grid_x, grid_y, grid_z);
  const dim3 block(block_x, block_y, block_z);
  switch (kernel_index) {{
{cases}
    default:
      return static_cast<int>(cudaErrorInvalidDeviceFunction);
  }}
}}

extern "C" const char* yasps_mas_cuda_error_string(int error) {{
  return cudaGetErrorString(static_cast<cudaError_t>(error));
}}

extern "C" int yasps_mas_set_kernel_dynamic_shared(
    unsigned int kernel_index, int shared_bytes) {{
  switch (kernel_index) {{
{attribute_cases}
    default:
      return static_cast<int>(cudaErrorInvalidDeviceFunction);
  }}
}}
"""


def _argument_storage(value):
  if hasattr(value, "gpudata"):
    return ctypes.c_void_p(int(value.gpudata))
  if not isinstance(value, np.generic):
    try:
      return ctypes.c_void_p(int(value))
    except (TypeError, ValueError):
      pass
  if isinstance(value, np.generic):
    dtype = np.dtype(value.dtype)
    if dtype == np.dtype(np.uint8):
      return ctypes.c_uint8(int(value))
    if dtype == np.dtype(np.int8):
      return ctypes.c_int8(int(value))
    if dtype == np.dtype(np.uint16):
      return ctypes.c_uint16(int(value))
    if dtype == np.dtype(np.int16):
      return ctypes.c_int16(int(value))
    if dtype == np.dtype(np.uint32):
      return ctypes.c_uint32(int(value))
    if dtype == np.dtype(np.int32):
      return ctypes.c_int32(int(value))
    if dtype == np.dtype(np.uint64):
      return ctypes.c_uint64(int(value))
    if dtype == np.dtype(np.int64):
      return ctypes.c_int64(int(value))
    if dtype == np.dtype(np.float32):
      return ctypes.c_float(float(value))
    if dtype == np.dtype(np.float64):
      return ctypes.c_double(float(value))
  if isinstance(value, bool):
    return ctypes.c_int32(int(value))
  if isinstance(value, int):
    return ctypes.c_int32(value)
  if isinstance(value, float):
    return ctypes.c_double(value)
  raise TypeError(f"unsupported CUDA JIT kernel argument: {type(value)!r}")


class CUDAJITKernel:
  """PyCUDA-function-compatible view of one kernel in a cached library."""

  def __init__(self, library: "CUDAJITLibrary", index: int, name: str):
    self._library = library
    self._index = int(index)
    self.name = str(name)

  def __call__(
      self, *arguments, block, grid, stream=None, shared: int = 0,
  ) -> None:
    block = tuple(int(value) for value in block)
    grid = tuple(int(value) for value in grid)
    block = block + (1,) * (3 - len(block))
    grid = grid + (1,) * (3 - len(grid))
    if len(block) != 3 or len(grid) != 3:
      raise ValueError("CUDA launch block and grid must have at most 3 axes")
    storage = [_argument_storage(value) for value in arguments]
    pointers = (ctypes.c_void_p * len(storage))(*(
      ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
      for value in storage
    ))
    stream_handle = 0 if stream is None else int(stream.handle)
    error = self._library._launch(
      ctypes.c_uint32(self._index),
      ctypes.c_uint32(grid[0]), ctypes.c_uint32(grid[1]),
      ctypes.c_uint32(grid[2]), ctypes.c_uint32(block[0]),
      ctypes.c_uint32(block[1]), ctypes.c_uint32(block[2]),
      ctypes.c_uint32(int(shared)), ctypes.c_void_p(stream_handle), pointers,
    )
    if error:
      message = self._library._error_string(int(error))
      decoded = message.decode("utf8", errors="replace") if message else "unknown"
      raise RuntimeError(
        f"CUDA launch failed for {self.name}: {decoded} ({int(error)})"
      )

  def set_max_dynamic_shared_bytes(self, shared_bytes: int) -> None:
    error = self._library._set_dynamic_shared(
      ctypes.c_uint32(self._index), ctypes.c_int(int(shared_bytes)),
    )
    if error:
      message = self._library._error_string(int(error))
      decoded = message.decode("utf8", errors="replace") if message else "unknown"
      raise RuntimeError(
        f"CUDA shared-memory opt-in failed for {self.name}: "
        f"{decoded} ({int(error)})"
      )


@dataclass
class CUDAJITLibrary:
  path: Path
  names: tuple[str, ...]
  compile_seconds: float
  cache_hit: bool
  _handle: object
  _launch: object
  _error_string: object
  _set_dynamic_shared: object

  def kernel(self, name: str) -> CUDAJITKernel:
    try:
      index = self.names.index(name)
    except ValueError as error:
      raise KeyError(f"kernel {name!r} is not present in {self.path}") from error
    return CUDAJITKernel(self, index, name)


def compile_cuda_library(
    cuda,
    source: str,
    *,
    kernel_names: tuple[str, ...],
    label: str,
    options: tuple[str, ...] = (),
) -> CUDAJITLibrary:
  """Compile or load one persistent, architecture-specific CUDA library."""
  if not kernel_names or len(set(kernel_names)) != len(kernel_names):
    raise ValueError("kernel_names must contain unique CUDA kernel symbols")
  architecture = _device_architecture(cuda)
  full_source = source + _launcher_source(kernel_names)
  fingerprint = hashlib.sha256()
  for item in (
      architecture, label, *options, *kernel_names, full_source,
  ):
    fingerprint.update(item.encode("utf8"))
    fingerprint.update(b"\0")
  digest = fingerprint.hexdigest()
  context_key = int(cuda.Context.get_current().handle)
  memory_key = (context_key, digest)
  existing = _LIBRARY_CACHE.get(memory_key)
  if existing is not None:
    # Compilation telemetry is per request. Loading the same library again in
    # one process is a cache hit, not another copy of its original cold cost.
    return CUDAJITLibrary(
      existing.path, existing.names, 0.0, True,
      existing._handle, existing._launch, existing._error_string,
      existing._set_dynamic_shared,
    )

  root = _cache_root()
  root.mkdir(parents=True, exist_ok=True)
  stem = f"{label}_{architecture}_{digest[:20]}"
  source_path = root / f"{stem}.cu"
  library_path = root / f"{stem}.so"
  cache_hit = library_path.exists()
  compile_seconds = 0.0
  if not cache_hit:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
      raise RuntimeError("NVCC is required to build MAS JIT kernels")
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{stem}_", dir=root) as directory:
      temporary = Path(directory)
      temporary_source = temporary / source_path.name
      temporary_library = temporary / library_path.name
      temporary_source.write_text(full_source, encoding="utf8")
      command = [
        nvcc, "-shared", "-Xcompiler", "-fPIC", "-std=c++17", "-O3",
        f"-arch={architecture}", "-cudart=shared", *options,
        "-o", str(temporary_library), str(temporary_source),
      ]
      completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
      )
      if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"NVCC failed for MAS JIT {label}: {detail}")
      os.replace(temporary_source, source_path)
      os.replace(temporary_library, library_path)
    compile_seconds = perf_counter() - started

  handle = ctypes.CDLL(str(library_path))
  launch = handle.yasps_mas_launch_kernel
  launch.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
  ]
  launch.restype = ctypes.c_int
  error_string = handle.yasps_mas_cuda_error_string
  error_string.argtypes = [ctypes.c_int]
  error_string.restype = ctypes.c_char_p
  set_dynamic_shared = handle.yasps_mas_set_kernel_dynamic_shared
  set_dynamic_shared.argtypes = [ctypes.c_uint32, ctypes.c_int]
  set_dynamic_shared.restype = ctypes.c_int
  result = CUDAJITLibrary(
    library_path, kernel_names, compile_seconds, cache_hit,
    handle, launch, error_string, set_dynamic_shared,
  )
  _LIBRARY_CACHE[memory_key] = result
  return result

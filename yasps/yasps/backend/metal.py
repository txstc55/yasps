"""Small PyCUDA-GPUArray-compatible surface backed by shared Metal buffers."""

from __future__ import annotations

import atexit
import builtins
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterable

import numpy as np


_SOURCE = Path(__file__).with_name("metal_runtime.mm")
_CACHE_ROOT = Path(
  os.environ.get(
    "YASPS_METAL_CACHE",
    Path.home() / "Library" / "Caches" / "yasps" / "metal",
  )
)
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_RUNTIME_HASH = hashlib.sha256(
  _SOURCE.read_bytes() + platform.platform().encode()
).hexdigest()[:16]
_RUNTIME_PATH = _CACHE_ROOT / f"libyasps_metal_{_RUNTIME_HASH}.dylib"
_BUILD_LOCK = threading.Lock()
_ARRAY_KERNEL_LOCK = threading.Lock()
_ARRAY_KERNELS: dict[str, "MetalKernel"] = {}
_TIMING_LOCK = threading.Lock()
_TIMING_PATH = os.environ.get("YASPS_METAL_TIMING_JSON")
_KERNEL_TIMINGS: dict[str, dict[str, float | int | str]] = {}


class _Argument(ctypes.Structure):
  _fields_ = [
    ("kind", ctypes.c_uint32),
    ("data", ctypes.c_void_p),
    ("length", ctypes.c_size_t),
  ]


class _BatchDispatch(ctypes.Structure):
  _fields_ = [
    ("pipeline", ctypes.c_void_p),
    ("arguments", ctypes.POINTER(_Argument)),
    ("argument_count", ctypes.c_size_t),
    ("grid_size", ctypes.c_size_t),
    ("threadgroup_size", ctypes.c_size_t),
  ]


def _build_runtime() -> None:
  if _RUNTIME_PATH.exists():
    return
  with _BUILD_LOCK:
    if _RUNTIME_PATH.exists():
      return
    temporary = _RUNTIME_PATH.with_suffix(".tmp.dylib")
    command = [
      "clang++",
      "-std=c++17",
      "-O3",
      "-fobjc-arc",
      "-dynamiclib",
      str(_SOURCE),
      "-framework",
      "Foundation",
      "-framework",
      "Metal",
      "-o",
      str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(_RUNTIME_PATH)


_build_runtime()
_runtime = ctypes.CDLL(str(_RUNTIME_PATH))
_runtime.yasps_metal_device_name.restype = ctypes.c_char_p
_runtime.yasps_metal_current_allocated_size.restype = ctypes.c_uint64
_runtime.yasps_metal_recommended_working_set_size.restype = (
  ctypes.c_uint64
)
_runtime.yasps_metal_alloc.argtypes = [ctypes.c_size_t]
_runtime.yasps_metal_alloc.restype = ctypes.c_void_p
_runtime.yasps_metal_free.argtypes = [ctypes.c_void_p]
_runtime.yasps_metal_memcpy.argtypes = [
  ctypes.c_void_p,
  ctypes.c_void_p,
  ctypes.c_size_t,
]
_runtime.yasps_metal_pipeline_create.argtypes = [
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.c_char_p,
  ctypes.c_size_t,
]
_runtime.yasps_metal_pipeline_create.restype = ctypes.c_void_p
_runtime.yasps_metal_pipeline_destroy.argtypes = [ctypes.c_void_p]
_runtime.yasps_metal_dispatch.argtypes = [
  ctypes.c_void_p,
  ctypes.POINTER(_Argument),
  ctypes.c_size_t,
  ctypes.c_size_t,
  ctypes.c_size_t,
  ctypes.c_char_p,
  ctypes.c_size_t,
]
_runtime.yasps_metal_dispatch.restype = ctypes.c_int
_runtime.yasps_metal_dispatch_argument_buffer.argtypes = (
  _runtime.yasps_metal_dispatch.argtypes
)
_runtime.yasps_metal_dispatch_argument_buffer.restype = ctypes.c_int
_runtime.yasps_metal_dispatch_batch.argtypes = [
  ctypes.POINTER(_BatchDispatch),
  ctypes.c_size_t,
  ctypes.c_char_p,
  ctypes.c_size_t,
]
_runtime.yasps_metal_dispatch_batch.restype = ctypes.c_int
_runtime.yasps_metal_last_gpu_time_ms.restype = ctypes.c_double


class _Allocation:
  def __init__(self, nbytes: int):
    self.pointer = int(_runtime.yasps_metal_alloc(nbytes) or 0)
    if self.pointer == 0:
      raise MemoryError(f"Metal could not allocate {nbytes} bytes")
    self.nbytes = builtins.max(nbytes, 1)

  def __del__(self):
    pointer = getattr(self, "pointer", 0)
    if pointer:
      _runtime.yasps_metal_free(ctypes.c_void_p(pointer))
      self.pointer = 0


class GPUArray:
  __array_priority__ = 1000

  def __init__(self, shape: int | Iterable[int], dtype=np.float32):
    self.shape = _normalize_shape(shape)
    self.dtype = _metal_dtype(dtype)
    self.size = int(np.prod(self.shape, dtype=np.int64))
    self.nbytes = self.size * self.dtype.itemsize
    self._allocation = _Allocation(self.nbytes)
    self._offset = 0

  @classmethod
  def _view(
    cls,
    allocation: _Allocation,
    offset: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
  ) -> "GPUArray":
    result = cls.__new__(cls)
    result.shape = shape
    result.dtype = np.dtype(dtype)
    result.size = int(np.prod(shape, dtype=np.int64))
    result.nbytes = result.size * result.dtype.itemsize
    result._allocation = allocation
    result._offset = offset
    return result

  @property
  def gpudata(self) -> int:
    return self._allocation.pointer + self._offset

  @property
  def ptr(self) -> int:
    return self.gpudata

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def _numpy_view(self) -> np.ndarray:
    byte_count = builtins.max(self.nbytes, 1)
    storage = (ctypes.c_ubyte * byte_count).from_address(self.gpudata)
    return np.frombuffer(storage, dtype=self.dtype, count=self.size).reshape(
      self.shape
    )

  def get(self) -> np.ndarray:
    return self._numpy_view().copy()

  def set(self, value: Any) -> None:
    if isinstance(value, GPUArray):
      if value.size != self.size:
        raise ValueError(
          f"cannot copy {value.size} values into Metal array of size "
          f"{self.size}"
        )
      if value.dtype != self.dtype:
        value = value.astype(self.dtype)
      _runtime.yasps_metal_memcpy(
        ctypes.c_void_p(self.gpudata),
        ctypes.c_void_p(value.gpudata),
        self.nbytes,
      )
      return
    array = np.asarray(value, dtype=self.dtype)
    if array.size != self.size:
      raise ValueError(
        f"cannot copy {array.size} values into Metal array of size "
        f"{self.size}"
      )
    self._numpy_view()[...] = array.reshape(self.shape)

  def fill(self, value: Any) -> None:
    if self.size == 0:
      return
    suffix = _dtype_suffix(self.dtype)
    kernel = _array_kernel(f"yasps_fill_{suffix}")
    scalar = self.dtype.type(value)
    kernel.dispatch(
      [self, scalar, np.uint32(self.size)],
      self.size,
    )

  def copy(self) -> "GPUArray":
    result = empty(self.shape, self.dtype)
    _runtime.yasps_metal_memcpy(
      ctypes.c_void_p(result.gpudata),
      ctypes.c_void_p(self.gpudata),
      self.nbytes,
    )
    return result

  def astype(self, dtype) -> "GPUArray":
    target_dtype = _metal_dtype(dtype)
    if target_dtype == self.dtype:
      return self.copy()
    source_suffix = _dtype_suffix(self.dtype)
    target_suffix = _dtype_suffix(target_dtype)
    supported = {"float", "int", "uint"}
    if (
      source_suffix not in supported
      or target_suffix not in supported
    ):
      raise TypeError(
        f"Metal conversion from {self.dtype} to {target_dtype} "
        "is not supported"
      )
    result = empty(self.shape, target_dtype)
    _array_kernel(
      f"yasps_convert_{source_suffix}_to_{target_suffix}"
    ).dispatch(
      [self, result, np.uint32(self.size)],
      self.size,
    )
    return result

  def reshape(self, *shape: int) -> "GPUArray":
    normalized = _normalize_shape(shape[0] if len(shape) == 1 else shape)
    if int(np.prod(normalized, dtype=np.int64)) != self.size:
      raise ValueError("cannot reshape array to a different size")
    return self._view(
      self._allocation, self._offset, normalized, self.dtype
    )

  def ravel(self) -> "GPUArray":
    return self.reshape(self.size)

  def __len__(self) -> int:
    return self.shape[0]

  def __getitem__(self, key):
    view = self._numpy_view()[key]
    if not isinstance(view, np.ndarray):
      return view
    if not view.flags.c_contiguous:
      return to_gpu(np.ascontiguousarray(view))
    offset = int(view.ctypes.data) - self._allocation.pointer
    return self._view(self._allocation, offset, view.shape, view.dtype)

  def __setitem__(self, key, value) -> None:
    target = self._numpy_view()[key]
    source = value._numpy_view() if isinstance(value, GPUArray) else value
    target[...] = source

  def _binary(
    self,
    other,
    operation,
    reverse=False,
  ) -> "GPUArray":
    if self.size == 0:
      return empty(self.shape, self.dtype)
    suffix = _dtype_suffix(self.dtype)
    if suffix not in {"float", "int", "uint"}:
      raise TypeError(
        f"Metal arithmetic is not supported for {self.dtype}"
      )
    result = empty(self.shape, self.dtype)
    if isinstance(other, GPUArray):
      if other.shape != self.shape:
        raise ValueError(
          "Metal GPUArray arithmetic requires equal shapes"
        )
      if other.dtype != self.dtype:
        raise TypeError(
          "Metal GPUArray arithmetic requires equal dtypes"
        )
      name = f"yasps_{operation}_{suffix}_array"
      arguments = [
        other if reverse else self,
        self if reverse else other,
        result,
        np.uint32(self.size),
      ]
    else:
      scalar = self.dtype.type(other)
      direction = "reverse_scalar" if reverse else "scalar"
      name = f"yasps_{operation}_{suffix}_{direction}"
      arguments = (
        [scalar, self, result, np.uint32(self.size)]
        if reverse
        else [self, scalar, result, np.uint32(self.size)]
      )
    _array_kernel(name).dispatch(arguments, self.size)
    return result

  def __add__(self, other):
    return self._binary(other, "add")

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    return self._binary(other, "subtract")

  def __rsub__(self, other):
    return self._binary(other, "subtract", reverse=True)

  def __mul__(self, other):
    return self._binary(other, "multiply")

  def __rmul__(self, other):
    return self.__mul__(other)

  def __truediv__(self, other):
    return self._binary(other, "divide")

  def __rtruediv__(self, other):
    return self._binary(other, "divide", reverse=True)

  def __neg__(self):
    suffix = _dtype_suffix(self.dtype)
    if suffix not in {"float", "int"}:
      raise TypeError(f"Metal negation is not supported for {self.dtype}")
    result = empty(self.shape, self.dtype)
    _array_kernel(f"yasps_negate_{suffix}").dispatch(
      [self, result, np.uint32(self.size)],
      self.size,
    )
    return result

  def __abs__(self):
    suffix = _dtype_suffix(self.dtype)
    if suffix not in {"float", "int"}:
      raise TypeError(
        f"Metal absolute value is not supported for {self.dtype}"
      )
    result = empty(self.shape, self.dtype)
    _array_kernel(f"yasps_abs_{suffix}").dispatch(
      [self, result, np.uint32(self.size)],
      self.size,
    )
    return result


def _normalize_shape(shape: int | Iterable[int]) -> tuple[int, ...]:
  if isinstance(shape, (int, np.integer)):
    return (int(shape),)
  return tuple(int(item) for item in shape)


def _metal_dtype(dtype) -> np.dtype:
  result = np.dtype(dtype)
  if result.kind == "f":
    return np.dtype(np.float32)
  return result


def _dtype_suffix(dtype) -> str:
  suffixes = {
    np.dtype(np.float32): "float",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
    np.dtype(np.int64): "long",
    np.dtype(np.uint64): "ulong",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "ushort",
    np.dtype(np.int8): "char",
    np.dtype(np.uint8): "uchar",
  }
  try:
    return suffixes[np.dtype(dtype)]
  except KeyError as error:
    raise TypeError(f"unsupported Metal dtype: {dtype}") from error


def to_gpu(array: Any) -> GPUArray:
  host = np.asarray(array)
  result = GPUArray(host.shape, _metal_dtype(host.dtype))
  result.set(host)
  return result


def empty(shape, dtype=np.float32) -> GPUArray:
  return GPUArray(shape, dtype)


def zeros(shape, dtype=np.float32) -> GPUArray:
  result = empty(shape, dtype)
  result.fill(0)
  return result


def empty_like(array: GPUArray) -> GPUArray:
  return empty(array.shape, array.dtype)


def zeros_like(array: GPUArray) -> GPUArray:
  return zeros(array.shape, array.dtype)


def sum(array: GPUArray) -> GPUArray:
  return _reduce_float(array, "sum")


def max(array: GPUArray) -> GPUArray:
  return _reduce_float(array, "max")


def device_name() -> str:
  value = _runtime.yasps_metal_device_name()
  if value is None:
    raise RuntimeError("No usable Metal device")
  return value.decode()


class MetalKernel:
  """Compiled Metal function with typed buffer/constant dispatch."""

  def __init__(
    self,
    metallib: str | Path,
    function_name: str,
    argument_buffer: bool = False,
  ):
    self.metallib = Path(metallib)
    self.function_name = function_name
    self.argument_buffer = argument_buffer
    error = ctypes.create_string_buffer(8192)
    self._pipeline = _runtime.yasps_metal_pipeline_create(
      os.fsencode(self.metallib),
      function_name.encode(),
      error,
      len(error),
    )
    if not self._pipeline:
      raise RuntimeError(error.value.decode(errors="replace"))

  def __del__(self):
    pipeline = getattr(self, "_pipeline", None)
    if pipeline:
      _runtime.yasps_metal_pipeline_destroy(pipeline)
      self._pipeline = None

  @property
  def last_gpu_time_ms(self) -> float:
    return float(_runtime.yasps_metal_last_gpu_time_ms())

  def dispatch(
    self,
    arguments: Iterable[GPUArray | np.generic | int | float],
    grid_size: int,
    threadgroup_size: int = 0,
  ) -> None:
    argument_array, keepalive = _encode_arguments(arguments)
    error = ctypes.create_string_buffer(8192)
    dispatch_function = (
      _runtime.yasps_metal_dispatch_argument_buffer
      if self.argument_buffer
      else _runtime.yasps_metal_dispatch
    )
    start = time.perf_counter()
    result = dispatch_function(
      self._pipeline,
      argument_array,
      len(argument_array),
      grid_size,
      threadgroup_size,
      error,
      len(error),
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    if result != 0:
      raise RuntimeError(error.value.decode(errors="replace"))
    _record_kernel_timing(
      (
        f"{self.metallib.stem}::{self.function_name}"
        f"::threads_{grid_size}"
      ),
      self.metallib.name,
      self.function_name,
      wall_ms,
      self.last_gpu_time_ms,
      grid_size,
    )


def dispatch_batch(
  dispatches: Iterable[
    tuple[
      MetalKernel,
      Iterable[GPUArray | np.generic | int | float],
      int,
      int,
    ]
  ],
  label: str,
) -> None:
  items = list(dispatches)
  if not items:
    return

  encoded_dispatches: list[_BatchDispatch] = []
  argument_arrays = []
  keepalive = []
  total_grid_size = 0
  for kernel, arguments, grid_size, threadgroup_size in items:
    if kernel.argument_buffer:
      raise ValueError(
        "Metal batch dispatch does not support argument-buffer kernels"
      )
    argument_array, scalar_keepalive = _encode_arguments(arguments)
    argument_arrays.append(argument_array)
    keepalive.extend(scalar_keepalive)
    encoded_dispatches.append(
      _BatchDispatch(
        kernel._pipeline,
        argument_array,
        len(argument_array),
        grid_size,
        threadgroup_size,
      )
    )
    total_grid_size += grid_size

  dispatch_array = (
    _BatchDispatch * len(encoded_dispatches)
  )(*encoded_dispatches)
  error = ctypes.create_string_buffer(8192)
  start = time.perf_counter()
  result = _runtime.yasps_metal_dispatch_batch(
    dispatch_array,
    len(encoded_dispatches),
    error,
    len(error),
  )
  wall_ms = (time.perf_counter() - start) * 1000.0
  if result != 0:
    raise RuntimeError(error.value.decode(errors="replace"))
  gpu_ms = float(_runtime.yasps_metal_last_gpu_time_ms())
  _record_kernel_timing(
    (
      f"metal_batch::{label}::dispatches_{len(encoded_dispatches)}"
      f"::threads_{total_grid_size}"
    ),
    "metal_batch",
    label,
    wall_ms,
    gpu_ms,
    total_grid_size,
  )


def fill_batch(arrays: Iterable[GPUArray], value=0) -> None:
  dispatches = []
  for array in arrays:
    if array.size == 0:
      continue
    suffix = _dtype_suffix(array.dtype)
    dispatches.append((
      _array_kernel(f"yasps_fill_{suffix}"),
      [array, array.dtype.type(value), np.uint32(array.size)],
      array.size,
      0,
    ))
  dispatch_batch(dispatches, "fill")


def _encode_arguments(arguments):
  encoded: list[_Argument] = []
  keepalive: list[Any] = []
  for value in arguments:
    if isinstance(value, GPUArray):
      encoded.append(_Argument(0, value.gpudata, value.nbytes))
      continue
    scalar = _metal_scalar(value)
    keepalive.append(scalar)
    encoded.append(
      _Argument(
        1,
        ctypes.addressof(scalar),
        ctypes.sizeof(scalar),
      )
    )
  return (_Argument * len(encoded))(*encoded), keepalive


def _metal_scalar(value):
  if isinstance(value, (np.float32, float)):
    return ctypes.c_float(value)
  if isinstance(value, np.uint8):
    return ctypes.c_uint8(value)
  if isinstance(value, np.int8):
    return ctypes.c_int8(value)
  if isinstance(value, np.uint16):
    return ctypes.c_uint16(value)
  if isinstance(value, np.int16):
    return ctypes.c_int16(value)
  if isinstance(value, np.uint32):
    return ctypes.c_uint32(value)
  if isinstance(value, np.int32):
    return ctypes.c_int32(value)
  if isinstance(value, np.uint64):
    return ctypes.c_uint64(value)
  if isinstance(value, np.int64):
    return ctypes.c_int64(value)
  if isinstance(value, (int, np.integer)):
    return ctypes.c_uint32(value)
  raise TypeError(f"unsupported Metal constant type: {type(value).__name__}")


def compile_metal(
  sources: Iterable[str | Path],
  output: str | Path,
  include_dirs: Iterable[str | Path] = (),
  flags: Iterable[str] = (),
) -> Path:
  """Compile independent MSL translation units and link one metallib."""

  source_paths = [Path(source) for source in sources]
  output_path = Path(output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  compiler = shutil.which("xcrun")
  if compiler is None:
    raise RuntimeError("xcrun is required to compile Metal sources")

  with tempfile.TemporaryDirectory(
    prefix="yasps_metal_", dir=output_path.parent
  ) as temporary:
    def compile_source(index_and_source):
      index, source = index_and_source
      air = Path(temporary) / f"{index}_{source.stem}.air"
      command = [
        compiler,
        "-sdk",
        "macosx",
        "metal",
        "-std=metal3.1",
        "-O3",
        "-c",
        str(source),
        "-o",
        str(air),
      ]
      for include_dir in include_dirs:
        command.extend(["-I", str(include_dir)])
      command.extend(flags)
      subprocess.run(command, check=True)
      return air

    worker_count = min(
      len(source_paths),
      os.cpu_count() or 1,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
      air_files = list(
        executor.map(compile_source, enumerate(source_paths))
      )

      temporary_output = Path(temporary) / output_path.name
      subprocess.run(
        [
          compiler,
          "-sdk",
          "macosx",
          "metallib",
          *map(str, air_files),
          "-o",
          str(temporary_output),
        ],
        check=True,
      )
      shutil.copy2(temporary_output, output_path)
  return output_path


def _array_kernel(function_name: str) -> MetalKernel:
  with _ARRAY_KERNEL_LOCK:
    existing = _ARRAY_KERNELS.get(function_name)
    if existing is not None:
      return existing
    source = Path(__file__).with_name("metal_array.metal")
    digest = hashlib.sha256(
      source.read_bytes() + platform.platform().encode()
    ).hexdigest()[:16]
    library = _CACHE_ROOT / f"metal_array_{digest}.metallib"
    if not library.exists():
      compile_metal([source], library)
    kernel = MetalKernel(library, function_name)
    _ARRAY_KERNELS[function_name] = kernel
    return kernel


def _reduce_float(array: GPUArray, operation: str) -> GPUArray:
  if array.dtype != np.dtype(np.float32):
    raise TypeError(
      f"Metal {operation} reduction currently requires float32"
    )
  if array.size == 0:
    if operation == "sum":
      return zeros(1, np.float32)
    raise ValueError("zero-size array has no maximum")
  source = array
  count = array.size
  kernel = _array_kernel(f"yasps_reduce_{operation}_float")
  dispatches = []
  while True:
    group_count = (count + 255) // 256
    output = empty(group_count, np.float32)
    dispatches.append((
      kernel,
      [source, output, np.uint32(count)],
      group_count * 256,
      256,
    ))
    if group_count == 1:
      dispatch_batch(dispatches, f"reduce_{operation}")
      return output
    source = output
    count = group_count


def _record_kernel_timing(
  timing_name,
  library_name,
  function_name,
  wall_ms,
  gpu_ms,
  grid_size,
):
  if _TIMING_PATH is None:
    return
  with _TIMING_LOCK:
    timing = _KERNEL_TIMINGS.setdefault(
      timing_name,
      {
        "library": library_name,
        "function": function_name,
        "calls": 0,
        "threads": 0,
        "wall_ms": 0.0,
        "gpu_ms": 0.0,
        "min_gpu_ms": float("inf"),
        "max_gpu_ms": 0.0,
      },
    )
    timing["calls"] += 1
    timing["threads"] += int(grid_size)
    timing["wall_ms"] += float(wall_ms)
    timing["gpu_ms"] += float(gpu_ms)
    timing["min_gpu_ms"] = builtins.min(
      timing["min_gpu_ms"],
      gpu_ms,
    )
    timing["max_gpu_ms"] = builtins.max(
      timing["max_gpu_ms"],
      gpu_ms,
    )


def _write_kernel_timings():
  if _TIMING_PATH is None:
    return
  destination = Path(_TIMING_PATH)
  destination.parent.mkdir(parents=True, exist_ok=True)
  kernels = {}
  with _TIMING_LOCK:
    for name, timing in sorted(_KERNEL_TIMINGS.items()):
      calls = int(timing["calls"])
      kernels[name] = {
        **timing,
        "mean_gpu_ms": (
          float(timing["gpu_ms"]) / calls if calls else 0.0
        ),
        "mean_wall_ms": (
          float(timing["wall_ms"]) / calls if calls else 0.0
        ),
      }
  destination.write_text(
    json.dumps(
      {
        "backend": "metal",
        "device": device_name(),
        "kernels": kernels,
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )


atexit.register(_write_kernel_timings)


class _Context:
  @staticmethod
  def get_current():
    return _Context()

  def pop(self):
    return None

  def push(self):
    return None


class _Device:
  def __init__(self, index=0):
    if index != 0:
      raise ValueError("Metal backend exposes one default device")

  def make_context(self):
    return _Context()


class _CudaCompatibility:
  Context = _Context
  Device = _Device
  LogicError = RuntimeError

  @staticmethod
  def memcpy_dtod(destination, source, length):
    _runtime.yasps_metal_memcpy(
      ctypes.c_void_p(int(destination)),
      ctypes.c_void_p(int(source)),
      int(length),
    )

  @staticmethod
  def mem_get_info():
    total = int(
      _runtime.yasps_metal_recommended_working_set_size()
    )
    allocated = int(_runtime.yasps_metal_current_allocated_size())
    return builtins.max(total - allocated, 0), total


cuda = _CudaCompatibility()


__all__ = [
  "GPUArray",
  "MetalKernel",
  "compile_metal",
  "cuda",
  "device_name",
  "dispatch_batch",
  "empty",
  "empty_like",
  "fill_batch",
  "max",
  "sum",
  "to_gpu",
  "zeros",
  "zeros_like",
]

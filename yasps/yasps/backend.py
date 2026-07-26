"""Runtime backend selection and the small array API used by YASPS.

CUDA keeps using PyCUDA unchanged.  Apple silicon uses MLX, whose arrays and
custom kernels execute through Metal.  Set ``YASPS_BACKEND=cuda`` or
``YASPS_BACKEND=metal`` to override automatic selection.
"""

from __future__ import annotations

import os
import platform
from typing import Any

import numpy as np


def _cuda_available() -> bool:
  try:
    import pycuda.driver as cuda
    cuda.init()
    return cuda.Device.count() > 0
  except Exception:
    return False


def _metal_available() -> bool:
  if platform.system() != "Darwin" or platform.machine() != "arm64":
    return False
  try:
    import mlx.core as mx
    return bool(mx.metal.is_available())
  except Exception:
    return False


_requested_backend = os.environ.get("YASPS_BACKEND", "auto").strip().lower()
if _requested_backend not in {"auto", "cuda", "metal"}:
  raise ValueError("YASPS_BACKEND must be 'auto', 'cuda', or 'metal'.")

if _requested_backend == "cuda":
  if not _cuda_available():
    raise RuntimeError("YASPS_BACKEND=cuda was requested, but CUDA/PyCUDA is unavailable.")
  backend_name = "cuda"
elif _requested_backend == "metal":
  if not _metal_available():
    raise RuntimeError("YASPS_BACKEND=metal was requested, but MLX Metal is unavailable.")
  backend_name = "metal"
elif _cuda_available():
  backend_name = "cuda"
elif _metal_available():
  backend_name = "metal"
else:
  raise RuntimeError(
    "YASPS found neither CUDA/PyCUDA nor Metal/MLX. "
    "Install pycuda on NVIDIA systems or mlx on Apple silicon."
  )

is_cuda = backend_name == "cuda"
is_metal = backend_name == "metal"


if is_cuda:
  import pycuda.autoinit as _pycuda_autoinit  # noqa: F401
  import pycuda.driver as driver
  import pycuda.gpuarray as gpuarray

  real_dtype = np.float64

  def synchronize() -> None:
    driver.Context.synchronize()

else:
  import mlx.core as mx

  real_dtype = np.float32

  def _mlx_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float64):
      return mx.float32
    return {
      np.dtype(np.float16): mx.float16,
      np.dtype(np.float32): mx.float32,
      np.dtype(np.int8): mx.int8,
      np.dtype(np.int16): mx.int16,
      np.dtype(np.int32): mx.int32,
      np.dtype(np.int64): mx.int64,
      np.dtype(np.uint8): mx.uint8,
      np.dtype(np.uint16): mx.uint16,
      np.dtype(np.uint32): mx.uint32,
      np.dtype(np.uint64): mx.uint64,
      np.dtype(np.bool_): mx.bool_,
    }[dtype]

  def _numpy_dtype(dtype):
    if dtype == mx.float16:
      return np.dtype(np.float16)
    if dtype == mx.float32:
      return np.dtype(np.float32)
    if dtype == mx.int8:
      return np.dtype(np.int8)
    if dtype == mx.int16:
      return np.dtype(np.int16)
    if dtype == mx.int32:
      return np.dtype(np.int32)
    if dtype == mx.int64:
      return np.dtype(np.int64)
    if dtype == mx.uint8:
      return np.dtype(np.uint8)
    if dtype == mx.uint16:
      return np.dtype(np.uint16)
    if dtype == mx.uint32:
      return np.dtype(np.uint32)
    if dtype == mx.uint64:
      return np.dtype(np.uint64)
    if dtype == mx.bool_:
      return np.dtype(np.bool_)
    raise TypeError(f"Unsupported MLX dtype: {dtype}")

  def _unwrap(value):
    return value._array if isinstance(value, GPUArray) else value

  def _normalize_shape(shape):
    if isinstance(shape, (int, np.integer)):
      return int(shape)
    if isinstance(shape, (tuple, list)):
      return tuple(int(dimension) for dimension in shape)
    return shape

  class GPUArray:
    """PyCUDA-compatible facade over an MLX Metal array.

    MLX is functional, so mutating methods replace the wrapped array.  That is
    enough for YASPS' buffer ownership model and keeps all arithmetic on Metal.
    """

    __array_priority__ = 1000

    def __init__(self, shape=None, dtype=np.float32, _array=None, _parent=None, _key=None):
      self._parent = _parent
      self._key = _key
      self.__array = None
      if _parent is None:
        if _array is not None:
          self.__array = _array
        else:
          if shape is None:
            shape = 0
          self.__array = mx.empty(_normalize_shape(shape), dtype=_mlx_dtype(dtype))

    @property
    def _array(self):
      if self._parent is not None:
        return self._parent._array[self._key]
      return self.__array

    @_array.setter
    def _array(self, value):
      if self._parent is not None:
        parent_array = self._parent._array
        parent_array[self._key] = value
        self._parent._array = parent_array
      else:
        self.__array = value

    @classmethod
    def _wrap(cls, value):
      return value if isinstance(value, cls) else cls(_array=value)

    @property
    def shape(self):
      return self._array.shape

    @property
    def size(self):
      return self._array.size

    @property
    def ndim(self):
      return self._array.ndim

    @property
    def dtype(self):
      return _numpy_dtype(self._array.dtype)

    @property
    def nbytes(self):
      return self.size * self.dtype.itemsize

    @property
    def gpudata(self):
      raise RuntimeError("Raw CUDA pointers are unavailable on the Metal backend.")

    @property
    def ptr(self):
      raise RuntimeError("Raw CUDA pointers are unavailable on the Metal backend.")

    def get(self):
      mx.eval(self._array)
      return np.asarray(self._array)

    def item(self):
      return self._array.item()

    def set(self, value):
      value = _unwrap(value)
      if isinstance(value, mx.array):
        self._array = value.astype(self._array.dtype).reshape(self.shape)
      else:
        self._array = mx.array(np.asarray(value), dtype=self._array.dtype).reshape(self.shape)
      return self

    def fill(self, value):
      self._array = mx.full(self.shape, value, dtype=self._array.dtype)
      return self

    def copy(self):
      return GPUArray._wrap(mx.array(self._array))

    def astype(self, dtype):
      return GPUArray._wrap(self._array.astype(_mlx_dtype(dtype)))

    def reshape(self, *shape):
      if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
      return GPUArray._wrap(self._array.reshape(shape))

    def ravel(self):
      return GPUArray._wrap(self._array.reshape((-1,)))

    flatten = ravel

    def __len__(self):
      return len(self._array)

    def __array__(self, dtype=None):
      value = self.get()
      return value.astype(dtype, copy=False) if dtype is not None else value

    def __getitem__(self, key):
      # MLX indexing returns a copy.  Keep an explicit parent/key reference so
      # cached YASPS gradient and solution segments retain PyCUDA view semantics.
      return GPUArray(_parent=self, _key=key)

    def __setitem__(self, key, value):
      array = self._array
      array[key] = _unwrap(value)
      self._array = array

    def _binary(self, other, op):
      return GPUArray._wrap(op(self._array, _unwrap(other)))

    def __add__(self, other):
      return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
      return self.__add__(other)

    def __sub__(self, other):
      return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
      return GPUArray._wrap(_unwrap(other) - self._array)

    def __mul__(self, other):
      return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
      return self.__mul__(other)

    def __truediv__(self, other):
      return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
      return GPUArray._wrap(_unwrap(other) / self._array)

    def __pow__(self, other):
      return self._binary(other, lambda a, b: a ** b)

    def __neg__(self):
      return GPUArray._wrap(-self._array)

    def __abs__(self):
      return GPUArray._wrap(mx.abs(self._array))

    def __lt__(self, other):
      return self._binary(other, lambda a, b: a < b)

    def __le__(self, other):
      return self._binary(other, lambda a, b: a <= b)

    def __gt__(self, other):
      return self._binary(other, lambda a, b: a > b)

    def __ge__(self, other):
      return self._binary(other, lambda a, b: a >= b)

    def __eq__(self, other):
      return self._binary(other, lambda a, b: a == b)

    def __ne__(self, other):
      return self._binary(other, lambda a, b: a != b)

    def __repr__(self):
      return f"MetalGPUArray({self._array!r})"

  class _MetalGPUArrayModule:
    GPUArray = GPUArray

    @staticmethod
    def to_gpu(value):
      if isinstance(value, GPUArray):
        return value.copy()
      if isinstance(value, mx.array):
        return GPUArray._wrap(mx.array(value))
      array = np.asarray(value)
      return GPUArray._wrap(mx.array(array, dtype=_mlx_dtype(array.dtype)))

    @staticmethod
    def empty(shape, dtype=np.float32):
      return GPUArray._wrap(mx.empty(_normalize_shape(shape), dtype=_mlx_dtype(dtype)))

    @staticmethod
    def zeros(shape, dtype=np.float32):
      return GPUArray._wrap(mx.zeros(_normalize_shape(shape), dtype=_mlx_dtype(dtype)))

    @staticmethod
    def empty_like(value):
      value = _unwrap(value)
      return GPUArray._wrap(mx.empty(value.shape, dtype=value.dtype))

    @staticmethod
    def zeros_like(value):
      value = _unwrap(value)
      return GPUArray._wrap(mx.zeros_like(value))

    @staticmethod
    def sum(value, axis=None):
      return GPUArray._wrap(mx.sum(_unwrap(value), axis=axis))

    @staticmethod
    def max(value, axis=None):
      return GPUArray._wrap(mx.max(_unwrap(value), axis=axis))

    @staticmethod
    def min(value, axis=None):
      return GPUArray._wrap(mx.min(_unwrap(value), axis=axis))

  class _MetalDriver:
    @staticmethod
    def mem_get_info():
      # MLX exposes active/cache usage but not the driver's total allocation.
      active = int(mx.get_active_memory())
      limit = int(mx.get_wired_limit())
      return max(0, limit - active), limit

    @staticmethod
    def Context():
      return None

  gpuarray = _MetalGPUArrayModule()
  driver = _MetalDriver()

  def synchronize() -> None:
    mx.synchronize()


def backend_info() -> dict[str, Any]:
  info = {
    "backend": backend_name,
    "platform": platform.platform(),
    "real_dtype": np.dtype(real_dtype).name,
  }
  if is_metal:
    info["device"] = mx.device_info()
  return info

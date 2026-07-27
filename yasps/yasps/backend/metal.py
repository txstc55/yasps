"""Small PyCUDA-GPUArray-compatible surface backed by shared Metal buffers."""

from __future__ import annotations

import builtins
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
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


class _Argument(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
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
        self.dtype = np.dtype(dtype)
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

    def _numpy_view(self) -> np.ndarray:
        byte_count = builtins.max(self.nbytes, 1)
        storage = (ctypes.c_ubyte * byte_count).from_address(self.gpudata)
        return np.frombuffer(storage, dtype=self.dtype, count=self.size).reshape(
            self.shape
        )

    def get(self) -> np.ndarray:
        return self._numpy_view().copy()

    def set(self, value: Any) -> None:
        source = value._numpy_view() if isinstance(value, GPUArray) else value
        array = np.asarray(source, dtype=self.dtype)
        if array.size != self.size:
            raise ValueError(
                f"cannot copy {array.size} values into Metal array of size "
                f"{self.size}"
            )
        self._numpy_view()[...] = array.reshape(self.shape)

    def fill(self, value: Any) -> None:
        self._numpy_view().fill(value)

    def copy(self) -> "GPUArray":
        result = empty(self.shape, self.dtype)
        _runtime.yasps_metal_memcpy(
            ctypes.c_void_p(result.gpudata),
            ctypes.c_void_p(self.gpudata),
            self.nbytes,
        )
        return result

    def astype(self, dtype) -> "GPUArray":
        return to_gpu(self._numpy_view().astype(dtype))

    def reshape(self, *shape: int) -> "GPUArray":
        normalized = _normalize_shape(shape[0] if len(shape) == 1 else shape)
        if int(np.prod(normalized, dtype=np.int64)) != self.size:
            raise ValueError("cannot reshape array to a different size")
        return self._view(
            self._allocation, self._offset, normalized, self.dtype
        )

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

    def _binary(self, other, operation) -> "GPUArray":
        rhs = other._numpy_view() if isinstance(other, GPUArray) else other
        return to_gpu(operation(self._numpy_view(), rhs))

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return to_gpu(np.subtract(other, self._numpy_view()))

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __rtruediv__(self, other):
        return to_gpu(np.divide(other, self._numpy_view()))

    def __neg__(self):
        return to_gpu(np.negative(self._numpy_view()))


def _normalize_shape(shape: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(item) for item in shape)


def to_gpu(array: Any) -> GPUArray:
    host = np.asarray(array)
    result = GPUArray(host.shape, host.dtype)
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
    return to_gpu(np.asarray([np.sum(array._numpy_view())], dtype=array.dtype))


def max(array: GPUArray) -> GPUArray:
    return to_gpu(np.asarray([np.max(array._numpy_view())], dtype=array.dtype))


def device_name() -> str:
    value = _runtime.yasps_metal_device_name()
    if value is None:
        raise RuntimeError("No usable Metal device")
    return value.decode()


class MetalKernel:
    """Compiled Metal function with typed buffer/constant dispatch."""

    def __init__(self, metallib: str | Path, function_name: str):
        self.metallib = Path(metallib)
        self.function_name = function_name
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
        argument_array = (_Argument * len(encoded))(*encoded)
        error = ctypes.create_string_buffer(8192)
        result = _runtime.yasps_metal_dispatch(
            self._pipeline,
            argument_array,
            len(encoded),
            grid_size,
            threadgroup_size,
            error,
            len(error),
        )
        if result != 0:
            raise RuntimeError(error.value.decode(errors="replace"))


def _metal_scalar(value):
    if isinstance(value, (np.float32, float)):
        return ctypes.c_float(value)
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
        air_files = []
        for index, source in enumerate(source_paths):
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
            air_files.append(air)

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


cuda = _CudaCompatibility()


__all__ = [
    "GPUArray",
    "MetalKernel",
    "compile_metal",
    "cuda",
    "device_name",
    "empty",
    "empty_like",
    "max",
    "sum",
    "to_gpu",
    "zeros",
    "zeros_like",
]

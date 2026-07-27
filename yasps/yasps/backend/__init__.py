"""Backend selection shared by YASPS' Cython modules and examples."""

from __future__ import annotations

import os
import platform

import numpy as np


_requested_backend = os.environ.get("YASPS_BACKEND", "auto").lower()
if _requested_backend not in {"auto", "cuda", "metal"}:
    raise ValueError("YASPS_BACKEND must be one of: auto, cuda, metal")

if _requested_backend == "auto":
    _requested_backend = "metal" if platform.system() == "Darwin" else "cuda"

name = _requested_backend
real_dtype = np.float32

if name == "metal":
    from . import metal as gpuarray
    from .metal import cuda

    autoinit = None
else:
    import pycuda.autoinit as autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray


def is_metal() -> bool:
    return name == "metal"


def is_cuda() -> bool:
    return name == "cuda"


__all__ = [
    "autoinit",
    "cuda",
    "gpuarray",
    "is_cuda",
    "is_metal",
    "name",
    "real_dtype",
]

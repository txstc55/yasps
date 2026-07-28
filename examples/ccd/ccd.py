from __future__ import annotations

from collections.abc import Sequence
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray


_MLBVH_API_VERSION = 2
_CASE_WIDTHS = (2, 3, 4, 4)
_UINT32_MAX = np.iinfo(np.uint32).max
_INT32_MAX = np.iinfo(np.int32).max


def _library_is_stale(target: Path, sources: Sequence[Path]) -> bool:
  if not target.exists():
    return True
  target_mtime = target.stat().st_mtime_ns
  return any(source.stat().st_mtime_ns > target_mtime for source in sources)


def _cuda_architecture() -> str:
  override = os.environ.get("YASPS_CUDA_ARCH")
  if override:
    return override.removeprefix("sm_").removeprefix("compute_")
  major, minor = cuda.Context.get_device().compute_capability()
  return f"{major}{minor}"


def _build_libraries(module_dir: Path) -> tuple[Path, Path]:
  mlbvh_so = module_dir / "libmlbvh.so"
  accd_so = module_dir / "libaccd.so"
  architecture_file = module_dir / ".ccd-architecture"
  architecture = _cuda_architecture()

  sources = [
    module_dir / "mlbvh.cu",
    module_dir / "mlbvh.cuh",
    module_dir / "ACCD.cu",
    module_dir / "ACCD.cuh",
    module_dir / "gpu_eigen_libs.cu",
    module_dir / "gpu_eigen_libs.cuh",
    module_dir / "cuda_tools.h",
    module_dir / "eigen_data.h",
  ]
  try:
    architecture_matches = architecture_file.read_text(encoding="ascii").strip() == architecture
  except OSError:
    architecture_matches = False
  if architecture_matches and not (
    _library_is_stale(mlbvh_so, sources)
    or _library_is_stale(accd_so, sources)
  ):
    return mlbvh_so, accd_so

  common = [
    "nvcc",
    "-std=c++17",
    "-Xcompiler",
    "-fPIC",
    "-O3",
    "-DNDEBUG",
    "-I/usr/include/eigen",
    f"-I{module_dir}",
    "-gencode",
    f"arch=compute_{architecture},code=sm_{architecture}",
    "--relocatable-device-code=true",
  ]

  with tempfile.TemporaryDirectory(prefix=".ccd-build-", dir=module_dir) as build_dir_string:
    build_dir = Path(build_dir_string)
    mlbvh_o = build_dir / "mlbvh.o"
    accd_o = build_dir / "ACCD.o"
    eigen_o = build_dir / "gpu_eigen_libs.o"
    next_mlbvh_so = build_dir / "libmlbvh.so"
    next_accd_so = build_dir / "libaccd.so"

    compile_jobs = (
      (module_dir / "mlbvh.cu", mlbvh_o),
      (module_dir / "ACCD.cu", accd_o),
      (module_dir / "gpu_eigen_libs.cu", eigen_o),
    )
    for source, output in compile_jobs:
      subprocess.run(common + ["-c", str(source), "-o", str(output)], check=True)

    link_common = common + ["--shared"]
    subprocess.run(
      link_common + [str(mlbvh_o), str(eigen_o), "-o", str(next_mlbvh_so)],
      check=True,
    )
    subprocess.run(
      link_common + [str(accd_o), str(eigen_o), "-o", str(next_accd_so)],
      check=True,
    )
    os.replace(next_mlbvh_so, mlbvh_so)
    os.replace(next_accd_so, accd_so)
    architecture_file.write_text(architecture + "\n", encoding="ascii")

  return mlbvh_so, accd_so


class CCD:
  """GPU collision detection with a reusable swept broad-phase cache.

  ``mesh_indices == 0`` enables self-collision. A shared nonzero mesh ID
  suppresses pairs whose primitive vertices all belong to that same mesh.
  Distances and ``dhat`` are squared, matching the original wrapper.
  """

  def __init__(
    self,
    num_vertices: int,
    all_vertices: int,
    max_cd_pairs: int = 10_000_000,
    max_ccd_pairs: int = 100_000_000,
    mesh_indices: Sequence[int] | None = None,
    print_timings: bool = True,
  ):
    for name, value in (
      ("num_vertices", num_vertices),
      ("all_vertices", all_vertices),
      ("max_cd_pairs", max_cd_pairs),
      ("max_ccd_pairs", max_ccd_pairs),
    ):
      if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")

    num_vertices = int(num_vertices)
    all_vertices = int(all_vertices)
    max_cd_pairs = int(max_cd_pairs)
    max_ccd_pairs = int(max_ccd_pairs)
    if not 0 <= num_vertices <= all_vertices:
      raise ValueError("num_vertices must be between 0 and all_vertices")
    if not 1 <= all_vertices <= _INT32_MAX:
      raise ValueError(f"all_vertices must be in [1, {_INT32_MAX}]")
    for name, value in (
      ("max_cd_pairs", max_cd_pairs),
      ("max_ccd_pairs", max_ccd_pairs),
    ):
      if not 0 < value <= _INT32_MAX:
        raise ValueError(f"{name} must be in [1, {_INT32_MAX}]")

    self.__num_vertices = int(num_vertices)
    self.__all_vertices = int(all_vertices)
    self.__max_cd_pairs = int(max_cd_pairs)
    self.__max_ccd_pairs = int(max_ccd_pairs)
    self.__print_timings = bool(print_timings)

    module_dir = Path(__file__).resolve().parent
    mlbvh_so_path, accd_so_path = _build_libraries(module_dir)
    self.__mlbvh = ctypes.CDLL(str(mlbvh_so_path))
    self.__accd = ctypes.CDLL(str(accd_so_path))
    self.__bind_libraries()

    self.__bvh_f = self.__mlbvh.create_lbvh_f()
    self.__bvh_e = self.__mlbvh.create_lbvh_e()
    if not self.__bvh_f or not self.__bvh_e:
      raise RuntimeError("Failed to create MLBVH objects")

    self.__collision_pairs = gpuarray.empty((max_cd_pairs, 4), dtype=np.int32)
    self.__case_rank = gpuarray.empty(max_cd_pairs, dtype=np.uint32)
    self.__case_storage = gpuarray.empty(max_cd_pairs * 4, dtype=np.uint32)
    self.__cp_num = gpuarray.zeros(5, dtype=np.uint32)

    self.__ccd_candidates = gpuarray.empty((max_ccd_pairs, 2), dtype=np.int32)
    self.__expanded_ccd_pairs: gpuarray.GPUArray | None = None
    self.__candidate_num = gpuarray.zeros(1, dtype=np.uint32)
    self.__mqueue = gpuarray.empty(max_ccd_pairs, dtype=np.float64)
    self.__overflow = gpuarray.zeros(2, dtype=np.uint32)

    self.__btypes = gpuarray.zeros(all_vertices, dtype=np.int32)
    self.__mesh_indices = gpuarray.to_gpu(
      self.__validate_mesh_indices(mesh_indices, all_vertices)
    )

    self.__faces: gpuarray.GPUArray | None = None
    self.__edges: gpuarray.GPUArray | None = None
    self.__surface_vertices: gpuarray.GPUArray | None = None
    self.__rest_vertices: gpuarray.GPUArray | None = None
    self.__current_face_vertices: gpuarray.GPUArray | None = None
    self.__current_edge_vertices: gpuarray.GPUArray | None = None
    self.__faces_initialized = False
    self.__edges_initialized = False
    self.__face_num = 0
    self.__edge_num = 0
    self.__closed = False

    self.__counts = (0, 0, 0, 0)
    self.__candidate_count = 0
    self.__cache_valid = False
    self.__cached_faces = False
    self.__cached_edges = False
    self.__cached_dhat = 0.0
    self.__cached_alpha = 0.0
    self.__cached_vertex_ptr = 0
    self.__cached_direction_ptr = 0
    self.__cached_vertices: gpuarray.GPUArray | None = None
    self.__cached_directions: gpuarray.GPUArray | None = None

  @staticmethod
  def __validate_mesh_indices(
    mesh_indices: Sequence[int] | None, all_vertices: int
  ) -> np.ndarray:
    if mesh_indices is None or len(mesh_indices) == 0:
      return np.zeros(all_vertices, dtype=np.uint32)
    values = np.asarray(mesh_indices)
    if values.ndim != 1 or values.size != all_vertices:
      raise ValueError("Length of mesh_indices must be equal to all_vertices")
    if values.dtype.kind not in "iu":
      raise TypeError("mesh_indices must contain integers")
    if np.any(values < 0) or np.any(values > _UINT32_MAX):
      raise ValueError("mesh_indices values must fit in uint32")
    return np.ascontiguousarray(values, dtype=np.uint32)

  def __bind_libraries(self) -> None:
    pointer = ctypes.c_void_p
    uint32 = ctypes.c_uint32

    try:
      api_version = self.__mlbvh.mlbvh_api_version
    except AttributeError as error:
      raise RuntimeError(
        "libmlbvh.so is stale and does not expose mlbvh_api_version"
      ) from error
    api_version.argtypes = []
    api_version.restype = uint32
    found_version = int(api_version())
    if found_version != _MLBVH_API_VERSION:
      raise RuntimeError(
        f"Unsupported MLBVH API version {found_version}; expected {_MLBVH_API_VERSION}"
      )

    self.__mlbvh.create_lbvh_f.argtypes = []
    self.__mlbvh.create_lbvh_f.restype = pointer
    self.__mlbvh.create_lbvh_e.argtypes = []
    self.__mlbvh.create_lbvh_e.restype = pointer
    self.__mlbvh.destroy_lbvh_f.argtypes = [pointer]
    self.__mlbvh.destroy_lbvh_f.restype = None
    self.__mlbvh.destroy_lbvh_e.argtypes = [pointer]
    self.__mlbvh.destroy_lbvh_e.restype = None

    shared_init = [pointer] * 13 + [uint32] * 4
    self.__lbvh_f_init = self.__mlbvh.lbvh_f_init
    self.__lbvh_f_init.argtypes = shared_init
    self.__lbvh_f_init.restype = None
    self.__lbvh_e_init = self.__mlbvh.lbvh_e_init
    self.__lbvh_e_init.argtypes = shared_init
    self.__lbvh_e_init.restype = None

    self.__lbvh_f_construct = self.__mlbvh.lbvh_f_construct
    self.__lbvh_f_construct.argtypes = [pointer, pointer]
    self.__lbvh_f_construct.restype = None
    self.__lbvh_e_construct = self.__mlbvh.lbvh_e_construct
    self.__lbvh_e_construct.argtypes = [pointer, pointer]
    self.__lbvh_e_construct.restype = None

    self.__lbvh_f_construct_full_ccd = self.__mlbvh.lbvh_f_construct_full_ccd
    self.__lbvh_f_construct_full_ccd.argtypes = [pointer, pointer, pointer, ctypes.c_double]
    self.__lbvh_f_construct_full_ccd.restype = None
    self.__lbvh_e_construct_full_ccd = self.__mlbvh.lbvh_e_construct_full_ccd
    self.__lbvh_e_construct_full_ccd.argtypes = [pointer, pointer, pointer, ctypes.c_double]
    self.__lbvh_e_construct_full_ccd.restype = None

    self.__reset_candidate_cache = self.__mlbvh.lbvh_reset_candidate_cache
    self.__reset_candidate_cache.argtypes = [pointer, pointer]
    self.__reset_candidate_cache.restype = None
    self.__append_face_proximity = self.__mlbvh.lbvh_f_append_proximity_candidates
    self.__append_face_proximity.argtypes = [pointer, pointer, ctypes.c_double]
    self.__append_face_proximity.restype = None
    self.__append_edge_proximity = self.__mlbvh.lbvh_e_append_proximity_candidates
    self.__append_edge_proximity.argtypes = [pointer, pointer, ctypes.c_double]
    self.__append_edge_proximity.restype = None
    self.__append_face_swept = self.__mlbvh.lbvh_f_append_swept_candidates
    self.__append_face_swept.argtypes = [
      pointer, pointer, pointer, ctypes.c_double, ctypes.c_double
    ]
    self.__append_face_swept.restype = None
    self.__append_edge_swept = self.__mlbvh.lbvh_e_append_swept_candidates
    self.__append_edge_swept.argtypes = [
      pointer, pointer, pointer, ctypes.c_double, ctypes.c_double
    ]
    self.__append_edge_swept.restype = None
    self.__refilter_candidates = self.__mlbvh.lbvh_refilter_cached_candidates
    self.__refilter_candidates.argtypes = [pointer, pointer, pointer, ctypes.c_double]
    self.__refilter_candidates.restype = None
    self.__scatter_cases = self.__mlbvh.lbvh_scatter_packed_cases
    self.__scatter_cases.argtypes = [pointer, pointer]
    self.__scatter_cases.restype = None
    self.__expand_candidates = self.__mlbvh.lbvh_expand_cached_candidates
    self.__expand_candidates.argtypes = [pointer, pointer, pointer, uint32]
    self.__expand_candidates.restype = None

    self.__lbvh_f_scene_size = self.__mlbvh.scene_size_f
    self.__lbvh_f_scene_size.argtypes = [pointer]
    self.__lbvh_f_scene_size.restype = ctypes.c_double
    self.__lbvh_e_scene_size = self.__mlbvh.scene_size_e
    self.__lbvh_e_scene_size.argtypes = [pointer]
    self.__lbvh_e_scene_size.restype = ctypes.c_double

    try:
      self.__largest_step_compact = self.__accd.self_largestFeasibleStepSizeCompact
    except AttributeError as error:
      raise RuntimeError(
        "libaccd.so is stale and lacks compact-candidate support"
      ) from error
    self.__largest_step_compact.argtypes = [
      ctypes.c_double,
      pointer,
      pointer,
      pointer,
      pointer,
      pointer,
      pointer,
      ctypes.c_int,
    ]
    self.__largest_step_compact.restype = ctypes.c_double

  @staticmethod
  def __to_void_p(array: gpuarray.GPUArray | None) -> ctypes.c_void_p:
    if array is None or array.size == 0:
      return ctypes.c_void_p(None)
    return ctypes.c_void_p(int(array.gpudata))

  @staticmethod
  def __require_gpu_array(
    name: str,
    array: gpuarray.GPUArray,
    dtypes: tuple[np.dtype, ...],
    expected_size: int,
  ) -> None:
    if not isinstance(array, gpuarray.GPUArray):
      raise TypeError(f"{name} must be a pycuda.gpuarray.GPUArray")
    if np.dtype(array.dtype) not in dtypes:
      expected = ", ".join(str(dtype) for dtype in dtypes)
      raise TypeError(f"{name} must have dtype {expected}; got {array.dtype}")
    if array.size != expected_size:
      raise ValueError(f"{name} must contain {expected_size} values; got {array.size}")
    if not array.flags.c_contiguous:
      raise ValueError(f"{name} must be C-contiguous")

  def __require_indices(
    self,
    name: str,
    array: gpuarray.GPUArray,
    expected_size: int,
  ) -> None:
    self.__require_gpu_array(
      name,
      array,
      (np.dtype(np.uint32), np.dtype(np.int32)),
      expected_size,
    )
    if array.size == 0:
      return
    smallest = int(gpuarray.min(array).get())
    largest = int(gpuarray.max(array).get())
    if smallest < 0 or largest >= self.__all_vertices:
      raise ValueError(
        f"{name} indices must be in [0, {self.__all_vertices - 1}]; "
        f"got [{smallest}, {largest}]"
      )

  def __require_vertices(self, name: str, vertices: gpuarray.GPUArray) -> None:
    self.__require_gpu_array(
      name,
      vertices,
      (np.dtype(np.float64),),
      self.__all_vertices * 3,
    )

  @property
  def collision_pairs(self) -> gpuarray.GPUArray:
    return self.__collision_pairs

  @property
  def collision_pairs_ccd(self) -> gpuarray.GPUArray:
    """Active full-vertex ``int4`` candidates, materialized lazily.

    New code should use :attr:`ccd_candidates`, whose compact ``int2`` rows
    avoid this compatibility allocation.
    """
    self.__ensure_open()
    if not self.__cache_valid:
      raise RuntimeError("ccd() must create a candidate cache first")
    if self.__expanded_ccd_pairs is None:
      self.__expanded_ccd_pairs = gpuarray.empty(
        (self.__candidate_count, 4), dtype=np.int32
      )
      self.__expand_candidates(
        self.__face_handle(),
        self.__edge_handle(),
        self.__to_void_p(self.__expanded_ccd_pairs),
        ctypes.c_uint32(self.__candidate_count),
      )
    return self.__expanded_ccd_pairs

  @property
  def ccd_candidates(self) -> gpuarray.GPUArray:
    """Compact signed ``int2`` primitive IDs for the current swept cache."""
    return self.__ccd_candidates

  @property
  def candidate_count(self) -> int:
    return self.__candidate_count

  @property
  def cp_num(self) -> gpuarray.GPUArray:
    return self.__cp_num

  @property
  def separated_counts(self) -> list[int]:
    return list(self.__counts)

  @property
  def case_storage(self) -> gpuarray.GPUArray:
    return self.__case_storage

  @property
  def case_offsets(self) -> tuple[int, int, int, int]:
    pp, pe, pt, _ = self.__counts
    return (0, 2 * pp, 2 * pp + 3 * pe, 2 * pp + 3 * pe + 4 * pt)

  def __case_view(self, case: int) -> gpuarray.GPUArray:
    offset = self.case_offsets[case]
    size = _CASE_WIDTHS[case] * self.__counts[case]
    return self.__case_storage[offset:offset + size]

  @property
  def pp(self) -> gpuarray.GPUArray:
    return self.__case_view(0)

  @property
  def pe(self) -> gpuarray.GPUArray:
    return self.__case_view(1)

  @property
  def pt(self) -> gpuarray.GPUArray:
    return self.__case_view(2)

  @property
  def ee(self) -> gpuarray.GPUArray:
    return self.__case_view(3)

  def init_faces(
    self,
    vertices: gpuarray.GPUArray,
    faces: gpuarray.GPUArray,
    surface_vertices: gpuarray.GPUArray,
    face_num: int,
  ) -> None:
    self.__ensure_open()
    if self.__faces_initialized:
      raise RuntimeError("Face BVH is already initialized")
    if isinstance(face_num, (bool, np.bool_)) or not isinstance(face_num, (int, np.integer)):
      raise TypeError("face_num must be an integer")
    face_num = int(face_num)
    if not 0 <= face_num <= _INT32_MAX:
      raise ValueError(f"face_num must be in [0, {_INT32_MAX}]")
    self.__require_vertices("vertices", vertices)
    if face_num > 0:
      self.__require_indices("faces", faces, face_num * 3)
      self.__require_indices("surface_vertices", surface_vertices, self.__num_vertices)
    elif not isinstance(faces, gpuarray.GPUArray) or not isinstance(
      surface_vertices, gpuarray.GPUArray
    ):
      raise TypeError("faces and surface_vertices must be GPUArrays")
    self.__faces = faces
    self.__surface_vertices = surface_vertices
    self.__current_face_vertices = vertices
    self.__lbvh_f_init(
      self.__bvh_f,
      self.__to_void_p(self.__btypes),
      self.__to_void_p(vertices),
      self.__to_void_p(faces),
      self.__to_void_p(surface_vertices),
      self.__to_void_p(self.__collision_pairs),
      self.__to_void_p(self.__case_rank),
      self.__to_void_p(self.__cp_num),
      self.__to_void_p(self.__mesh_indices),
      self.__to_void_p(self.__ccd_candidates),
      self.__to_void_p(self.__candidate_num),
      self.__to_void_p(self.__case_storage),
      self.__to_void_p(self.__overflow),
      ctypes.c_uint32(self.__max_ccd_pairs),
      ctypes.c_uint32(self.__max_cd_pairs),
      ctypes.c_uint32(face_num),
      ctypes.c_uint32(self.__num_vertices),
    )
    self.__faces_initialized = True
    self.__face_num = face_num

  def init_edges(
    self,
    vertices: gpuarray.GPUArray,
    vertices_rest: gpuarray.GPUArray,
    edges: gpuarray.GPUArray,
    edge_num: int,
  ) -> None:
    self.__ensure_open()
    if self.__edges_initialized:
      raise RuntimeError("Edge BVH is already initialized")
    if isinstance(edge_num, (bool, np.bool_)) or not isinstance(edge_num, (int, np.integer)):
      raise TypeError("edge_num must be an integer")
    edge_num = int(edge_num)
    if not 0 <= edge_num <= _INT32_MAX:
      raise ValueError(f"edge_num must be in [0, {_INT32_MAX}]")
    self.__require_vertices("vertices", vertices)
    if edge_num > 0:
      self.__require_vertices("vertices_rest", vertices_rest)
      self.__require_indices("edges", edges, edge_num * 2)
    elif not isinstance(vertices_rest, gpuarray.GPUArray) or not isinstance(
      edges, gpuarray.GPUArray
    ):
      raise TypeError("vertices_rest and edges must be GPUArrays")
    self.__edges = edges
    self.__rest_vertices = vertices_rest
    self.__current_edge_vertices = vertices
    self.__lbvh_e_init(
      self.__bvh_e,
      self.__to_void_p(self.__btypes),
      self.__to_void_p(vertices),
      self.__to_void_p(vertices_rest),
      self.__to_void_p(edges),
      self.__to_void_p(self.__collision_pairs),
      self.__to_void_p(self.__case_rank),
      self.__to_void_p(self.__cp_num),
      self.__to_void_p(self.__mesh_indices),
      self.__to_void_p(self.__ccd_candidates),
      self.__to_void_p(self.__candidate_num),
      self.__to_void_p(self.__case_storage),
      self.__to_void_p(self.__overflow),
      ctypes.c_uint32(self.__max_ccd_pairs),
      ctypes.c_uint32(self.__max_cd_pairs),
      ctypes.c_uint32(edge_num),
      ctypes.c_uint32(self.__num_vertices),
    )
    self.__edges_initialized = True
    self.__edge_num = edge_num

  def construct_faces(self, vertices: gpuarray.GPUArray) -> None:
    self.__require_face_bvh()
    self.__require_vertices("vertices", vertices)
    self.__invalidate_cache()
    self.__current_face_vertices = vertices
    self.__lbvh_f_construct(self.__bvh_f, self.__to_void_p(vertices))

  def construct_edges(self, vertices: gpuarray.GPUArray) -> None:
    self.__require_edge_bvh()
    self.__require_vertices("vertices", vertices)
    self.__invalidate_cache()
    self.__current_edge_vertices = vertices
    self.__lbvh_e_construct(self.__bvh_e, self.__to_void_p(vertices))

  def construct_full_ccd_faces(
    self,
    vertices: gpuarray.GPUArray,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
  ) -> None:
    self.__require_face_bvh()
    self.__validate_sweep(vertices, moving_directions, alpha, 0.0)
    self.__invalidate_cache()
    self.__current_face_vertices = vertices
    self.__lbvh_f_construct_full_ccd(
      self.__bvh_f,
      self.__to_void_p(vertices),
      self.__to_void_p(moving_directions),
      ctypes.c_double(alpha),
    )

  def construct_full_ccd_edges(
    self,
    vertices: gpuarray.GPUArray,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
  ) -> None:
    self.__require_edge_bvh()
    self.__validate_sweep(vertices, moving_directions, alpha, 0.0)
    self.__invalidate_cache()
    self.__current_edge_vertices = vertices
    self.__lbvh_e_construct_full_ccd(
      self.__bvh_e,
      self.__to_void_p(vertices),
      self.__to_void_p(moving_directions),
      ctypes.c_double(alpha),
    )

  def __ensure_open(self) -> None:
    if self.__closed:
      raise RuntimeError("CCD instance is closed")

  def __face_handle(self):
    return self.__bvh_f if self.__faces_initialized else None

  def __edge_handle(self):
    return self.__bvh_e if self.__edges_initialized else None

  def __invalidate_cache(self) -> None:
    self.__cache_valid = False
    self.__cached_faces = False
    self.__cached_edges = False
    self.__expanded_ccd_pairs = None
    self.__cached_dhat = 0.0
    self.__cached_alpha = 0.0
    self.__cached_vertex_ptr = 0
    self.__cached_direction_ptr = 0
    self.__cached_vertices = None
    self.__cached_directions = None

  def __require_face_bvh(self) -> None:
    self.__ensure_open()
    if not self.__faces_initialized:
      raise RuntimeError("init_faces must be called first")

  def __require_edge_bvh(self) -> None:
    self.__ensure_open()
    if not self.__edges_initialized:
      raise RuntimeError("init_edges must be called first")

  @staticmethod
  def __validate_dhat(dhat: float) -> None:
    if not np.isfinite(dhat) or dhat < 0.0:
      raise ValueError("dhat must be finite and nonnegative")

  def __validate_sweep(
    self,
    vertices: gpuarray.GPUArray,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
    dhat: float,
  ) -> None:
    self.__require_vertices("vertices", vertices)
    self.__require_vertices("moving_directions", moving_directions)
    self.__validate_dhat(dhat)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
      raise ValueError("alpha must be finite and in (0, 1]")

  def __reset_work(self) -> None:
    self.__candidate_num.fill(0)
    self.__cp_num.fill(0)
    self.__overflow.fill(0)
    self.__candidate_count = 0
    self.__counts = (0, 0, 0, 0)

  def __read_candidate_count(self) -> int:
    candidate_count = int(self.__candidate_num.get()[0])
    overflow = self.__overflow.get()
    if int(overflow[0]) or candidate_count > self.__max_ccd_pairs:
      self.__invalidate_cache()
      raise OverflowError(
        "CCD broad-phase candidate capacity exceeded: "
        f"needed more than {self.__max_ccd_pairs} entries"
      )
    self.__candidate_count = candidate_count
    return candidate_count

  def __finish_active_pairs(self) -> tuple[int, int, int, int]:
    counts = self.__cp_num.get()
    total = int(counts[0])
    active_overflow = int(self.__overflow.get()[1])
    if active_overflow or total > self.__max_cd_pairs:
      raise OverflowError(
        "Active collision-pair capacity exceeded: "
        f"needed more than {self.__max_cd_pairs} entries"
      )
    case_counts = tuple(int(value) for value in counts[1:5])
    if sum(case_counts) != total:
      raise RuntimeError(
        f"MLBVH case counts {case_counts} do not sum to total {total}"
      )
    self.__scatter_cases(self.__face_handle(), self.__edge_handle())
    cuda.Context.synchronize()
    self.__counts = case_counts
    return case_counts

  def __run_fresh_cd(
    self,
    vertices: gpuarray.GPUArray,
    dhat: float,
    include_faces: bool,
    include_edges: bool,
  ) -> tuple[int, int, int, int]:
    self.__ensure_open()
    if include_faces:
      self.__require_face_bvh()
    if include_edges:
      self.__require_edge_bvh()
    if not include_faces and not include_edges:
      raise ValueError("At least one primitive type must be included")
    self.__require_vertices("vertices", vertices)
    self.__validate_dhat(dhat)
    self.__reset_work()
    self.__invalidate_cache()
    self.__reset_candidate_cache(self.__face_handle(), self.__edge_handle())
    if include_faces:
      self.__current_face_vertices = vertices
      self.__append_face_proximity(
        self.__bvh_f, self.__to_void_p(vertices), ctypes.c_double(dhat)
      )
    if include_edges:
      self.__current_edge_vertices = vertices
      self.__append_edge_proximity(
        self.__bvh_e, self.__to_void_p(vertices), ctypes.c_double(dhat)
      )
    self.__read_candidate_count()
    self.__refilter_candidates(
      self.__face_handle(),
      self.__edge_handle(),
      self.__to_void_p(vertices),
      ctypes.c_double(dhat),
    )
    return self.__finish_active_pairs()

  def cd_faces(self, vertices: gpuarray.GPUArray, dhat: float) -> list[int]:
    started = time.perf_counter()
    counts = self.__run_fresh_cd(vertices, dhat, include_faces=True, include_edges=False)
    self.__print_elapsed("Face collision detection", started)
    return list(counts)

  def cd_edges(self, vertices: gpuarray.GPUArray, dhat: float) -> list[int]:
    started = time.perf_counter()
    counts = self.__run_fresh_cd(vertices, dhat, include_faces=False, include_edges=True)
    self.__print_elapsed("Edge collision detection", started)
    return list(counts)

  def cd(self, vertices: gpuarray.GPUArray, dhat: float) -> list[int]:
    started = time.perf_counter()
    counts = self.__run_fresh_cd(vertices, dhat, include_faces=True, include_edges=True)
    self.__print_elapsed("Collision detection", started)
    return list(counts)

  def __run_ccd(
    self,
    vertices: gpuarray.GPUArray,
    dhat: float,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
    include_faces: bool,
    include_edges: bool,
  ) -> int:
    self.__ensure_open()
    if include_faces:
      self.__require_face_bvh()
    if include_edges:
      self.__require_edge_bvh()
    if not include_faces and not include_edges:
      raise ValueError("At least one primitive type must be included")
    self.__validate_sweep(vertices, moving_directions, alpha, dhat)
    self.__reset_work()
    self.__invalidate_cache()
    self.__reset_candidate_cache(self.__face_handle(), self.__edge_handle())
    if include_faces:
      self.__current_face_vertices = vertices
      self.__append_face_swept(
        self.__bvh_f,
        self.__to_void_p(vertices),
        self.__to_void_p(moving_directions),
        ctypes.c_double(alpha),
        ctypes.c_double(dhat),
      )
    if include_edges:
      self.__current_edge_vertices = vertices
      self.__append_edge_swept(
        self.__bvh_e,
        self.__to_void_p(vertices),
        self.__to_void_p(moving_directions),
        ctypes.c_double(alpha),
        ctypes.c_double(dhat),
      )
    candidate_count = self.__read_candidate_count()
    self.__cp_num.set(np.array([candidate_count, 0, 0, 0, 0], dtype=np.uint32))
    self.__cache_valid = True
    self.__cached_faces = include_faces
    self.__cached_edges = include_edges
    self.__cached_dhat = float(dhat)
    self.__cached_alpha = float(alpha)
    self.__cached_vertex_ptr = int(vertices.gpudata)
    self.__cached_direction_ptr = int(moving_directions.gpudata)
    self.__cached_vertices = vertices
    self.__cached_directions = moving_directions
    return candidate_count

  def ccd_faces(
    self,
    vertices: gpuarray.GPUArray,
    dhat: float,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
  ) -> int:
    started = time.perf_counter()
    count = self.__run_ccd(
      vertices, dhat, moving_directions, alpha, include_faces=True, include_edges=False
    )
    self.__print_elapsed("Face continuous collision detection", started)
    return count

  def ccd_edges(
    self,
    vertices: gpuarray.GPUArray,
    dhat: float,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
  ) -> int:
    started = time.perf_counter()
    count = self.__run_ccd(
      vertices, dhat, moving_directions, alpha, include_faces=False, include_edges=True
    )
    self.__print_elapsed("Edge continuous collision detection", started)
    return count

  def ccd(
    self,
    vertices: gpuarray.GPUArray,
    dhat: float,
    moving_directions: gpuarray.GPUArray,
    alpha: float,
  ) -> int:
    started = time.perf_counter()
    count = self.__run_ccd(
      vertices, dhat, moving_directions, alpha, include_faces=True, include_edges=True
    )
    self.__print_elapsed("Continuous collision detection", started)
    return count

  def cd_from_cached_ccd(
    self,
    vertices: gpuarray.GPUArray,
    dhat: float,
    alpha: float,
  ) -> list[int]:
    """Exactly filter a prior swept cache at line-search trial positions.

    ``vertices`` must represent a point on the cached trajectory at ``alpha``.
    Reuse is valid only inside the cached alpha interval and at an equal or
    smaller squared distance threshold.
    """
    self.__ensure_open()
    if not self.__cache_valid:
      raise RuntimeError("ccd() must create a candidate cache before cached filtering")
    if (self.__face_num > 0 and not self.__cached_faces) or (
      self.__edge_num > 0 and not self.__cached_edges
    ):
      raise RuntimeError(
        "Cached filtering requires a sweep over every initialized primitive type; "
        "call ccd() before cd_from_cached_ccd()"
      )
    self.__require_vertices("vertices", vertices)
    self.__validate_dhat(dhat)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > self.__cached_alpha:
      raise ValueError(
        f"Trial alpha {alpha} is outside cached sweep [0, {self.__cached_alpha}]"
      )
    if dhat > self.__cached_dhat:
      raise ValueError(
        f"Trial dhat {dhat} exceeds cached dhat {self.__cached_dhat}"
      )

    started = time.perf_counter()
    self.__counts = (0, 0, 0, 0)
    self.__refilter_candidates(
      self.__face_handle(),
      self.__edge_handle(),
      self.__to_void_p(vertices),
      ctypes.c_double(dhat),
    )
    if self.__faces_initialized:
      self.__current_face_vertices = vertices
    if self.__edges_initialized:
      self.__current_edge_vertices = vertices
    counts = self.__finish_active_pairs()
    self.__print_elapsed("Cached collision filtering", started)
    return list(counts)

  def compute_largest_step_size(
    self,
    slackness: float,
    vertices: gpuarray.GPUArray,
    moving_directions: gpuarray.GPUArray,
  ) -> float:
    self.__ensure_open()
    if not self.__cache_valid:
      raise RuntimeError("ccd() must be called before compute_largest_step_size()")
    if not np.isfinite(slackness) or not 0.0 <= slackness < 1.0:
      raise ValueError("slackness must be finite and in [0, 1)")
    self.__require_vertices("vertices", vertices)
    self.__require_vertices("moving_directions", moving_directions)
    if int(vertices.gpudata) != self.__cached_vertex_ptr:
      raise ValueError("vertices must be the same GPU array used to build the CCD cache")
    if int(moving_directions.gpudata) != self.__cached_direction_ptr:
      raise ValueError(
        "moving_directions must be the same GPU array used to build the CCD cache"
      )

    started = time.perf_counter()
    step = self.__largest_step_compact(
      ctypes.c_double(slackness),
      self.__to_void_p(vertices),
      self.__to_void_p(self.__ccd_candidates),
      self.__to_void_p(self.__faces),
      self.__to_void_p(self.__edges),
      self.__to_void_p(moving_directions),
      self.__to_void_p(self.__mqueue),
      ctypes.c_int(self.__candidate_count),
    )
    cuda.Context.synchronize()
    step = min(float(step), self.__cached_alpha)
    self.__print_elapsed("Computing largest step size", started)
    return step

  def reset(self) -> None:
    self.__ensure_open()
    self.__reset_work()
    self.__invalidate_cache()

  def get_scene_size_faces(self) -> float:
    self.__require_face_bvh()
    return float(self.__lbvh_f_scene_size(self.__bvh_f))

  def get_scene_size_edges(self) -> float:
    self.__require_edge_bvh()
    return float(self.__lbvh_e_scene_size(self.__bvh_e))

  def __print_elapsed(self, label: str, started: float) -> None:
    if self.__print_timings:
      elapsed_ms = (time.perf_counter() - started) * 1000.0
      print(f"{label} took {elapsed_ms:.2f} ms")

  def close(self) -> None:
    if getattr(self, "_CCD__closed", False):
      return
    mlbvh = getattr(self, "_CCD__mlbvh", None)
    bvh_f = getattr(self, "_CCD__bvh_f", None)
    bvh_e = getattr(self, "_CCD__bvh_e", None)
    if mlbvh is not None and bvh_f:
      mlbvh.destroy_lbvh_f(bvh_f)
    if mlbvh is not None and bvh_e:
      mlbvh.destroy_lbvh_e(bvh_e)
    self.__bvh_f = None
    self.__bvh_e = None
    self.__invalidate_cache()
    self.__faces_initialized = False
    self.__edges_initialized = False
    self.__face_num = 0
    self.__edge_num = 0
    for attribute in (
      "_CCD__collision_pairs",
      "_CCD__case_rank",
      "_CCD__case_storage",
      "_CCD__cp_num",
      "_CCD__ccd_candidates",
      "_CCD__expanded_ccd_pairs",
      "_CCD__candidate_num",
      "_CCD__mqueue",
      "_CCD__overflow",
      "_CCD__btypes",
      "_CCD__mesh_indices",
      "_CCD__faces",
      "_CCD__edges",
      "_CCD__surface_vertices",
      "_CCD__rest_vertices",
      "_CCD__current_face_vertices",
      "_CCD__current_edge_vertices",
    ):
      if hasattr(self, attribute):
        setattr(self, attribute, None)
    self.__candidate_count = 0
    self.__counts = (0, 0, 0, 0)
    self.__closed = True
    self.__mlbvh = None
    self.__accd = None

  def __enter__(self) -> CCD:
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.close()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass

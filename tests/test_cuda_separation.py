from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
  return (ROOT / path).read_text(encoding="utf-8")


def _inclusive_slice(source: str, start: str, end: str) -> str:
  start_index = source.index(start)
  end_index = source.index(end, start_index) + len(end)
  if source[end_index:end_index + 1] == "\n":
    end_index += 1
  return source[start_index:end_index]


def _sha256(value: str) -> str:
  return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_generated_cuda_solver_matches_pre_metal_snapshot():
  source = _source(
    "yasps/yasps/kernel/Solver/solverKernel.pyx"
  )
  cuda_codegen = _inclusive_slice(
    source,
    "      kernelString: str = '''",
    '} // close the extern "C"',
  )

  assert _sha256(cuda_codegen) == (
    "d7fd72959a5c39dcd89b987e174f221c"
    "40a89fff9092ad8b2c8a2a361e9717e3"
  )
  assert "relativeTolerance = threshold * h_delta_0" in cuda_codegen
  assert "useIdentityPreconditioner" not in cuda_codegen
  assert "relative_residual_v1" not in source


def test_generated_cuda_gradient_indices_match_pre_metal_snapshot():
  source = _source(
    "yasps/yasps/kernel/Coordinate/gradientIndicesKernel.pyx"
  )
  cuda_codegen = _inclusive_slice(
    source,
    "    self.__kernelString += '''",
    (
      "    self.__kernelString = "
      "prune_duplicate_functions(self.__kernelString)"
      " # just in case we have duplicated functions"
    ),
  )

  assert _sha256(cuda_codegen) == (
    "76f523e3425fa28f0451428a0acb1926"
    "3c7a05d2269cc124b1d5e794d3b453c5"
  )


def test_cuda_keeps_original_reset_and_allocation_paths():
  solver_source = _source("yasps/yasps/solver/solver.pyx")
  assert """
    if not is_metal():
      self.__d_p1_b.fill(0)
      self.__d_r.fill(0)
      self.__d_c.fill(0)
      self.__d_q.fill(0)
      self.__d_s.fill(0)
      self.__solution.fill(0)
""" in solver_source

  indices_source = _source(
    "yasps/yasps/kernel/Coordinate/gradientIndicesKernel.pyx"
  )
  assert """
      else:
        # Keep the original CUDA allocation sizes and ordering.
        self.__outputIndices = gpuarray.zeros(self.maxNumIndicesNeeded * newNumInstances, dtype=np.uint32)
        self.__outputIndexSizes = gpuarray.zeros(self.maxNumIndicesNeeded * newNumInstances, dtype=np.uint16)
        self.__outputPermutations = gpuarray.zeros(self.maxNumIndicesNeeded * newNumInstances, dtype=np.int16)
        self.__outputGradientSizes = gpuarray.zeros(newNumInstances, dtype=np.uint16)
        self.__outputGroupedIndicesInner = gpuarray.zeros(self.maxNumIndicesNeeded * newNumInstances, dtype=np.uint32)
        self.__outputCompressedCoordinateCountsOuter = gpuarray.zeros(newNumInstances + 1, dtype=np.uint32)
""" in indices_source


def test_cuda_dtype_and_eigen_codegen_are_backend_specific():
  backend_source = _source("yasps/yasps/backend/__init__.py")
  assert (
    'real_dtype = np.float32 if name == "metal" else np.float64'
    in backend_source
  )

  separate_source = _source(
    "yasps/yasps/kernel/Hessian/"
    "hessianKernelSeparateJacobian.pyx"
  )
  assert 'map_const = "const " if is_metal() else ""' in separate_source


def test_optimized_metal_batch_labels_are_reported(monkeypatch):
  monkeypatch.setattr(sys, "argv", ["summarize.py"])
  module_path = ROOT / "metal_evaluation/summarize.py"
  spec = importlib.util.spec_from_file_location(
    "yasps_evaluation_summary",
    module_path,
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  assert module.classify_kernel("cg_iterations") == "linear_solver"
  assert (
    module.classify_kernel("diagonal_block_inverse")
    == "linear_solver"
  )
  assert (
    module.classify_kernel("coordinate_sort_unique")
    == "sparse_indices"
  )
  assert (
    module.classify_kernel("gradient_index_compression")
    == "sparse_indices"
  )
  assert module.classify_kernel("ccd_continuous") == "ccd"
  assert module.classify_kernel("reduce_sum") == "array_runtime"

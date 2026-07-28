# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from typing import List
from yasps.helper import prune_duplicate_functions
import os
import ctypes
from yasps.helper import timed
import pycuda.gpuarray as gpuarray
from yasps.primitiveUnion import primitiveUnion
import subprocess
import hashlib
import json
from yasps.context import context

GLOBAL_KERNEL_CACHE_VERSION = "v5_exact_source_sync"


def _stable_content_signature(payload, length: int = 24) -> str:
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()[:length]


def _write_generated_source(path: str, source: str, build_token: str) -> None:
  temporary_path = f"{path}.tmp_{build_token}"
  with open(temporary_path, "w") as f:
    f.write(source)
  os.replace(temporary_path, path)


class globalKernel:
  @timed("globalKernel.__init__")
  def __init__(self, att: attribute):
    self.__kernelString: str = ""
    self.__headerFileString: str = ""
    self.__att = att
    self.__kernel = None
    self.__additional_compile_flags = []  # --ptxas-options=-v,-warn-spills,-warn-lmem-usage  use this for memory checking
    self.__generateKernel()
    self.__context = context()

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  def __buildHeaderString(self, sorted_dependency: List[deviceKernel]) -> str:
    header_string = '''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

// For small matrix < 4
template <unsigned int N>
__device__ void spd_projection_small(const double *A, double* output, int choice) {
  if (choice == 0){
    for (int i = 0; i < N * N; i++) {
      output[i] = A[i];
    }
    return;
  }
  if (N == 1){
    output[0] = choice == 1 ? abs(A[0]) : (A[0] < 1e-6 ? 1e-6: A[0]);
    return;
  }

  const int M = 4;
  // Initialize an M x M matrix with zeros
  Eigen::Matrix<double, M, M> symMtr = Eigen::Matrix<double, M, M>::Identity();

  // Copy the input N x N matrix into the top-left corner of the M x M matrix
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      symMtr(row, col) = A[row * N + col];
    }
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, M, M>> eigenSolver(symMtr);
  const Eigen::Matrix<double, M, M>& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, M, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < M; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1e-6;
    }
  }

  Eigen::Matrix<double, M, M> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();

  // Copy the top-left N x N submatrix back to A
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      output[row * N + col] = A_reconstructed(row, col);
    }
  }
  return;
}

template <unsigned int N>
__device__ void spd_projection(const double *A, double* output, int choice) {
  if (choice == 0){
    for (int i = 0; i < N * N; i++) {
      output[i] = A[i];
    }
    return;
  }

  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();

  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 1e-6;
    }
  }

  Eigen::Matrix<double, N, N> A_reconstructed;
  A_reconstructed.noalias() = B * eigenValues.asDiagonal() * B.transpose();

  Eigen::Map<Eigen::Matrix<double, N, N, Eigen::RowMajor>> outputMap(output);
  outputMap = A_reconstructed;
  return;
}

template <unsigned int N>
__device__ void spd_projection_inplace(double *A, int choice) {
  if (choice == 0){
    return;
  }
  // Map A to an N x N Eigen matrix without copying
  Eigen::Map<const Eigen::Matrix<double, N, N>> mappedA(A);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> eigenSolver(mappedA);
  const auto& B = eigenSolver.eigenvectors();
  Eigen::Matrix<double, N, 1> eigenValues = eigenSolver.eigenvalues();
  for (int i = 0; i < N; i++) {
    if (eigenValues[i] < 0) {
      eigenValues[i] = choice == 1 ? abs(eigenValues[i]) : 0.0;
    }
  }

  // Reconstruct the matrix directly without using an intermediate matrix
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += B(i, k) * eigenValues[k] * B(j, k);
      }
      A[i * N + j] = sum;
    }
  }
  return;
}
'''
    for item in (sorted_dependency + [self.__att.deviceKernel]):
      header_string += f'''
extern "C" {{
{item.kernelHeader};
}}'''
    return header_string

  def __buildHostSource(
    self,
    sorted_datas: List[attribute],
    sorted_connectivities: List[connectivity],
    sorted_primitive_unions: List[primitiveUnion],
    header_basename: str,
  ) -> str:
    attribute_name = self.__att.fullName.replace("-", "_neg_")
    kernel_raw_name = f'''
__global__ void {attribute_name}_global_function({
  "".join([f"const double* {x.code_generation_data_name}, " for x in sorted_datas])
}  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sorted_connectivities])
}  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])
}  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sorted_primitive_unions])
}  double* result,
  unsigned int MAX_INDEX
)'''
    host_source = f'''
#include "{header_basename}"
extern "C" {{
{kernel_raw_name}{{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  {attribute_name}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sorted_datas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sorted_connectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
    {"".join([f'{x.code_generation_counts_name},' for x in sorted_primitive_unions])}
    index,
    result + index * {self.__att.size}
  );
}}
}}

extern "C"
int compute(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sorted_datas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sorted_connectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in sorted_primitive_unions])}
  double* result,
  unsigned int MAX_INDEX
){{
  {attribute_name}_global_function<<<(MAX_INDEX + 31) / 32, 32>>>(
    {"".join([f"{x.code_generation_data_name}, " for x in sorted_datas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sorted_connectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}," for x in sorted_primitive_unions])}
    result,
    MAX_INDEX
  );
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {{
    fprintf(stderr, "CUDA launch error: %s\\n", cudaGetErrorString(err));
    return -1;
  }}
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {{
    fprintf(stderr, "CUDA execution error: %s\\n", cudaGetErrorString(err));
    return -1;
  }}
  return 0;
}}
'''
    return prune_duplicate_functions(host_source)

  def __loadKernel(
    self,
    file_name: str,
    sorted_datas: List[attribute],
    sorted_connectivities: List[connectivity],
    sorted_primitive_unions: List[primitiveUnion],
  ) -> None:
    self.__kernel = ctypes.CDLL(f"{file_name}.so").compute
    self.__kernel.argtypes = [
      *[ctypes.c_void_p for _ in sorted_datas],
      *[ctypes.c_void_p for _ in sorted_connectivities],
      *[ctypes.c_void_p for x in sorted_connectivities if x.dimension == 0],
      *[ctypes.c_void_p for _ in sorted_primitive_unions],
      ctypes.c_void_p,
      ctypes.c_uint,
    ]
    self.__kernel.restype = ctypes.c_int

  @timed("globalKernel.__generateKernel")
  def __generateKernel(self) -> None:
    assert self.__att.deviceKernel is not None
    sorted_dependency: List[deviceKernel] = self.__att.deviceKernel.dependents
    sorted_datas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sorted_connectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sorted_primitive_unions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    os.makedirs(".yasps_tmp", exist_ok=True)

    compile_flags = [
      "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-arch=sm_89",
      "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
    ] + self.__additional_compile_flags
    dlink_prefix_flags = ["-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89"]
    dlink_suffix_flags = list(self.__additional_compile_flags)
    link_prefix_flags = ["-shared", "-Xcompiler", "-fPIC", "-arch=sm_89"]
    link_suffix_flags = ["-lcudart", "-lcuda"] + self.__additional_compile_flags

    self.__headerFileString = self.__buildHeaderString(sorted_dependency)
    header_signature = _stable_content_signature({
      "cache_version": GLOBAL_KERNEL_CACHE_VERSION,
      "header": self.__headerFileString,
    }, length=32)
    header_basename = f"global_headers_{header_signature}.cuh"
    header_path = f".yasps_tmp/{header_basename}"
    self.__kernelString = self.__buildHostSource(
      sorted_datas,
      sorted_connectivities,
      sorted_primitive_unions,
      header_basename,
    )

    device_units = []
    seen_device_unit_signatures = set()
    for item in (sorted_dependency + [self.__att.deviceKernel]):
      device_unit_source = f'''
#include "{header_basename}"
extern "C"{{
{item.kernelString}
}}
'''
      device_unit_signature = _stable_content_signature({
        "cache_version": GLOBAL_KERNEL_CACHE_VERSION,
        "compiler_flags": compile_flags,
        "header": self.__headerFileString,
        "source": device_unit_source,
      }, length=32)
      if device_unit_signature in seen_device_unit_signatures:
        continue
      seen_device_unit_signatures.add(device_unit_signature)
      device_units.append((device_unit_signature, device_unit_source))

    abi_metadata = {
      "attribute": {
        "full_name": self.__att.fullName,
        "rows": int(self.__att.rows),
        "cols": int(self.__att.cols),
        "size": int(self.__att.size),
      },
      "datas": [
        (item.fullName, item.fullNameWithHash, item.code_generation_data_name)
        for item in sorted_datas
      ],
      "connectivities": [
        (
          item.fullName,
          int(item.dimension),
          item.code_generation_index_name,
          item.code_generation_csr_name,
        )
        for item in sorted_connectivities
      ],
      "primitive_unions": [
        (item.fullName, item.code_generation_counts_name)
        for item in sorted_primitive_unions
      ],
    }
    generation_signature = _stable_content_signature({
      "cache_version": GLOBAL_KERNEL_CACHE_VERSION,
      "compiler": "nvcc",
      "compile_flags": compile_flags,
      "dlink_prefix_flags": dlink_prefix_flags,
      "dlink_suffix_flags": dlink_suffix_flags,
      "link_prefix_flags": link_prefix_flags,
      "link_suffix_flags": link_suffix_flags,
      "abi": abi_metadata,
      "header": self.__headerFileString,
      "host": self.__kernelString,
      "device_units": device_units,
    }, length=32)
    file_name = f".yasps_tmp/global_kernel_{generation_signature}"

    if not os.path.exists(f"{file_name}.so"):
      print(f"File {file_name}.so does not exist, compiling")
      build_token = f"{os.getpid()}_{id(self)}"
      _write_generated_source(header_path, self.__headerFileString, build_token)

      compile_jobs = []
      obj_files = []
      for device_unit_signature, device_unit_source in device_units:
        cu_file = f".yasps_tmp/global_device_{device_unit_signature}.cu"
        obj_file = f".yasps_tmp/global_device_{device_unit_signature}.o"
        obj_files.append(obj_file)
        if os.path.exists(obj_file):
          continue
        _write_generated_source(cu_file, device_unit_source, build_token)
        temporary_obj_file = f"{obj_file}.tmp_{build_token}"
        compile_cmd = [
          "nvcc", *compile_flags,
          "-c", cu_file, "-o", temporary_obj_file,
        ]
        print("Command is")
        print(" ".join(compile_cmd))
        compile_jobs.append((
          compile_cmd,
          subprocess.Popen(compile_cmd),
          temporary_obj_file,
          obj_file,
        ))

      kernel_cu_file = f"{file_name}.cu"
      _write_generated_source(kernel_cu_file, self.__kernelString, build_token)
      kernel_obj_file = f"{file_name}.o"
      temporary_kernel_obj_file = f"{kernel_obj_file}.tmp_{build_token}"
      kernel_compile_cmd = [
        "nvcc", *compile_flags,
        "-c", kernel_cu_file, "-o", temporary_kernel_obj_file,
      ]
      print("Kernel compile command: ")
      print(" ".join(kernel_compile_cmd))
      compile_jobs.append((
        kernel_compile_cmd,
        subprocess.Popen(kernel_compile_cmd),
        temporary_kernel_obj_file,
        kernel_obj_file,
      ))

      compile_failures = []
      for compile_cmd, job, temporary_obj_file, obj_file in compile_jobs:
        return_code = job.wait()
        if return_code != 0:
          compile_failures.append((return_code, compile_cmd))
        else:
          os.replace(temporary_obj_file, obj_file)
      if compile_failures:
        return_code, compile_cmd = compile_failures[0]
        raise subprocess.CalledProcessError(return_code, compile_cmd)

      device_link_obj = f"{file_name}_device_link.o"
      temporary_device_link_obj = f"{device_link_obj}.tmp_{build_token}"
      dlink_cmd = [
        "nvcc", *dlink_prefix_flags,
        *(obj_files + [kernel_obj_file]),
        "-o", temporary_device_link_obj,
        *dlink_suffix_flags,
      ]
      print("Device link command: ")
      print(" ".join(dlink_cmd))
      subprocess.run(dlink_cmd, check=True)
      os.replace(temporary_device_link_obj, device_link_obj)

      temporary_shared_object = f"{file_name}.so.tmp_{build_token}"
      final_link_cmd = [
        "nvcc", *link_prefix_flags,
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", temporary_shared_object,
        *link_suffix_flags,
      ]
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)
      os.replace(temporary_shared_object, f"{file_name}.so")
    else:
      print(f"File {file_name}.so does exist, linking")

    self.__loadKernel(
      file_name,
      sorted_datas,
      sorted_connectivities,
      sorted_primitive_unions,
    )

  @timed("globalKernel.compute")
  def compute(self, output):
    assert self.__kernel is not None
    if self.__att.correspondance.numInstances == 0:
      return # there is nothing to compute
    counts_gpu = [x.children_primitive_counts_gpu for x in self.__att.deviceKernel.kernelPrimitiveUnions]
    args = [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelDatas]
    args += [self.__to_void_p(x.value) for x in self.__att.deviceKernel.kernelConnectivity]
    args += [self.__to_void_p(x.compressedRows) for x in self.__att.deviceKernel.kernelConnectivity if x.dimension == 0]
    args += [self.__to_void_p(x) for x in counts_gpu]
    args += [self.__to_void_p(output)]
    args += [ctypes.c_uint32(self.__att.correspondance.numInstances)]
    self.__context.useDefaultContext()
    error_code = self.__kernel(*args)
    if error_code != 0:
      raise RuntimeError(f"globalKernel.compute: Kernel execution failed with error code {error_code}")

  @property
  def kernelString(self) -> str:
    return self.__kernelString

  @property
  def kernel(self):
    return self.__kernel

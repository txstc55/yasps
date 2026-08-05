# cython: language_level=3
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pycuda.gpuarray as gpuarray

from yasps.attribute import attribute
from yasps.connectivity import connectivity
from yasps.context import context
from yasps.deviceKernel import deviceKernel
from yasps.hessianKernelHeader import hessianKernelHeader
from yasps.helper import prune_duplicate_functions, timed
from yasps.primitiveUnion import primitiveUnion
from yasps.secondOrderJacobianIndicesKernel import (
  secondOrderJacobianIndicesKernel,
)
from yasps.secondOrderJacobianKernelNoProject import (
  secondOrderJacobianKernelNoProject,
)


MAX_PARALLEL_COMPILATIONS = 32


def availableCPUCount() -> int:
  if hasattr(os, "sched_getaffinity"):
    try:
      return max(1, len(os.sched_getaffinity(0)))
    except OSError:
      pass
  return max(1, os.cpu_count() or 1)


class secondOrderJacobianKernel:
  """Compile and run a fused rectangular second-derivative block kernel."""

  def __init__(self, att: attribute, dynamic_term: bool = False):
    self.__att = att
    self.__dynamic_term = bool(dynamic_term)
    self.__kernel = None
    self.__row_stride = 0
    self.__column_stride = 0
    self.__context = context()
    self.__additional_compile_flags = []

  def __to_void_p(self, value):
    if value is None or value.size == 0:
      return ctypes.c_void_p(None)
    assert value.gpudata is not None
    return ctypes.c_void_p(int(value.gpudata))

  def __compile(self, command: List[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)

  def __compileMany(self, commands: List[List[str]]) -> None:
    if len(commands) == 0:
      return
    for command in commands:
      print(" ".join(command))
    num_workers = min(
      MAX_PARALLEL_COMPILATIONS,
      availableCPUCount(),
      len(commands)
    )
    print(
      f"Compiling {len(commands)} CUDA translation units with at most "
      f"{num_workers} concurrent NVCC processes."
    )
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
      futures = [
        executor.submit(subprocess.run, command, check=True)
        for command in commands
      ]
      for future in futures:
        future.result()

  def __hostKernelString(
    self,
    sorted_datas: List[attribute],
    sorted_connectivities: List[connectivity],
    sorted_unions: List[primitiveUnion]
  ) -> str:
    return f'''
extern "C"
int compute_second_order_jacobian_no_project(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sorted_datas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sorted_connectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
  {"".join([f"const unsigned int* {x.code_generation_counts_name}, " for x in sorted_unions])}
  const unsigned int* row_indices,
  const unsigned short int* row_sizes,
  const short int* row_permutations,
  const unsigned int* column_indices,
  const unsigned short int* column_sizes,
  const short int* column_permutations,
  const unsigned int* coordinates_outer,
  const unsigned int* lookups,
  double* jacobian_blocks,
  const unsigned int num_instances
) {{
  if (num_instances == 0) {{
    return 0;
  }}
  compute_second_order_jacobian_no_project_global<<<
    (num_instances + 31) / 32, 32
  >>>(
    {"".join([f"{x.code_generation_data_name}, " for x in sorted_datas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sorted_connectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sorted_unions])}
    row_indices,
    row_sizes,
    row_permutations,
    column_indices,
    column_sizes,
    column_permutations,
    coordinates_outer,
    lookups,
    jacobian_blocks,
    num_instances
  );
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {{
    fprintf(
      stderr,
      "CUDA error in second-order Jacobian computation: %s\\n",
      cudaGetErrorString(error)
    );
    return -1;
  }}
  return 0;
}}
'''

  def __setKernelSignature(
    self,
    sorted_datas: List[attribute],
    sorted_connectivities: List[connectivity],
    sorted_unions: List[primitiveUnion]
  ) -> None:
    assert self.__kernel is not None
    self.__kernel.restype = ctypes.c_int
    self.__kernel.argtypes = [
      *(ctypes.c_void_p for _ in sorted_datas),
      *(ctypes.c_void_p for _ in sorted_connectivities),
      *(
        ctypes.c_void_p
        for item in sorted_connectivities
        if item.dimension == 0
      ),
      *(ctypes.c_void_p for _ in sorted_unions),
      *(ctypes.c_void_p for _ in range(9)),
      ctypes.c_uint32,
    ]

  @timed("secondOrderJacobianKernel.generateKernel")
  def generateKernel(self, row_stride: int, column_stride: int) -> None:
    if row_stride <= 0 or column_stride <= 0:
      raise ValueError(
        "secondOrderJacobianKernel.generateKernel: strides must be positive."
      )
    if self.__kernel is not None:
      if (
        self.__row_stride != row_stride
        or self.__column_stride != column_stride
      ):
        raise ValueError(
          "secondOrderJacobianKernel.generateKernel: an existing kernel "
          "cannot be reused with different rectangular strides."
        )
      return

    self.__row_stride = row_stride
    self.__column_stride = column_stride
    sorted_dependencies: List[deviceKernel] = self.__att.deviceKernel.dependents
    sorted_datas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sorted_connectivities: List[connectivity] = (
      self.__att.deviceKernel.kernelConnectivity
    )
    sorted_unions: List[primitiveUnion] = (
      self.__att.deviceKernel.kernelPrimitiveUnions
    )
    if self.__att.name == "":
      attribute_name = f"attr_{self.__att.hash}".replace("-", "_neg_")
    else:
      attribute_name = self.__att.fullName.replace("-", "_neg_")

    # Include the raw-layout contract version in the cache key.  In
    # particular, this prevents loading the old implementation that treated
    # permutation values as raw offsets and therefore read UNION padding
    # incorrectly.
    signature = "|".join([
      "rectangular_no_project_raw_offsets_v2",
      self.__att.fullNameWithHash,
      str(self.__att.rows),
      str(self.__att.cols),
      str(row_stride),
      str(column_stride),
    ])
    signature_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    file_name = (
      ".yasps_tmp/compute_second_order_jacobian_for_"
      f"{signature_hash}"
    )

    if not os.path.exists(f"{file_name}.so"):
      header = hessianKernelHeader(
        self.__att,
        set(),
        sorted_dependencies
      ).kernelString
      with open(".yasps_tmp/allHeaders.cuh", "w") as output:
        output.write(header)

      dependency_objects: List[str] = []
      compile_commands: List[List[str]] = []
      seen_objects = set()
      for item in sorted_dependencies + [self.__att.deviceKernel]:
        object_file = f".yasps_tmp/{item.attributeName}.o"
        if object_file in seen_objects:
          continue
        seen_objects.add(object_file)
        dependency_objects.append(object_file)
        if os.path.exists(object_file):
          continue
        source_file = f".yasps_tmp/{item.attributeName}.cu"
        with open(source_file, "w") as output:
          output.write(f'''
#include "allHeaders.cuh"
extern "C" {{
{item.kernelString}
}}
''')
        compile_commands.append([
          "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17",
          "-O3", "-arch=sm_89", "-c", source_file, "-o", object_file,
          "-I/usr/include/eigen3", "--expt-relaxed-constexpr",
          "--disable-warnings", "--relocatable-device-code=true",
          *self.__additional_compile_flags,
        ])

      generated_kernel = secondOrderJacobianKernelNoProject(
        self.__att,
        row_stride,
        column_stride,
        attribute_name
      ).kernelString
      generated_kernel += self.__hostKernelString(
        sorted_datas,
        sorted_connectivities,
        sorted_unions
      )
      generated_kernel = prune_duplicate_functions(generated_kernel)
      kernel_source = f"{file_name}.cu"
      kernel_object = f"{file_name}.o"
      with open(kernel_source, "w") as output:
        output.write(generated_kernel)
      compile_commands.append([
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17",
        "-O3", "-arch=sm_89", "-c", kernel_source, "-o",
        kernel_object, "-I/usr/include/eigen3",
        "--expt-relaxed-constexpr", "--disable-warnings",
        "--relocatable-device-code=true",
        *self.__additional_compile_flags,
      ])
      self.__compileMany(compile_commands)

      device_link_object = f"{file_name}_device_link.o"
      self.__compile([
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3", *dependency_objects, kernel_object, "-o",
        device_link_object, "--relocatable-device-code=true",
        *self.__additional_compile_flags,
      ])
      self.__compile([
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3", kernel_object, device_link_object, *dependency_objects,
        "-o", f"{file_name}.so", "-lcudart", "-lcuda",
        *self.__additional_compile_flags,
      ])

    library = ctypes.CDLL(f"{file_name}.so")
    self.__kernel = library.compute_second_order_jacobian_no_project
    self.__setKernelSignature(
      sorted_datas,
      sorted_connectivities,
      sorted_unions
    )

  @timed("secondOrderJacobianKernel.compute")
  def compute(
    self,
    indices_kernel: secondOrderJacobianIndicesKernel,
    lookup: gpuarray.GPUArray,
    blocks: gpuarray.GPUArray
  ) -> None:
    if indices_kernel.numTotalCoordinates == 0:
      return
    if self.__kernel is None:
      raise RuntimeError(
        "secondOrderJacobianKernel.compute: generateKernel must run first."
      )
    if (
      indices_kernel.rowIndicesKernel.maxNumIndicesNeeded
      != self.__row_stride
      or indices_kernel.columnIndicesKernel.maxNumIndicesNeeded
      != self.__column_stride
    ):
      raise ValueError(
        "secondOrderJacobianKernel.compute: coordinate strides do not match "
        "the compiled computation kernel."
      )
    if self.__att.correspondance.numInstances != indices_kernel.numInstances:
      raise ValueError(
        "secondOrderJacobianKernel.compute: local derivative and coordinate "
        "streams have different instance counts."
      )

    self.__context.useDefaultContext()
    row = indices_kernel.rowIndicesKernel
    column = indices_kernel.columnIndicesKernel
    counts_gpu = [
      item.children_primitive_counts_gpu
      for item in self.__att.deviceKernel.kernelPrimitiveUnions
    ]
    arguments = [item.value for item in self.__att.deviceKernel.kernelDatas]
    arguments += [
      item.value for item in self.__att.deviceKernel.kernelConnectivity
    ]
    arguments += [
      item.compressedRows
      for item in self.__att.deviceKernel.kernelConnectivity
      if item.dimension == 0
    ]
    arguments += counts_gpu
    error_code = self.__kernel(
      *[self.__to_void_p(item) for item in arguments],
      self.__to_void_p(row.outputIndices),
      self.__to_void_p(row.outputSizes),
      self.__to_void_p(row.outputPermutations),
      self.__to_void_p(column.outputIndices),
      self.__to_void_p(column.outputSizes),
      self.__to_void_p(column.outputPermutations),
      self.__to_void_p(indices_kernel.coordinateCountsOuter),
      self.__to_void_p(lookup),
      self.__to_void_p(blocks),
      ctypes.c_uint32(indices_kernel.numInstances)
    )
    if error_code != 0:
      raise RuntimeError(
        "secondOrderJacobianKernel.compute: CUDA kernel execution failed "
        f"with error code {error_code}."
      )

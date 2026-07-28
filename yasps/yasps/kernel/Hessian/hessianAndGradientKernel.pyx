# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
import pycuda.gpuarray as gpuarray
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.primitiveUnion import primitiveUnion
from yasps.helper import prune_duplicate_functions, timed
import os
import ctypes
from typing import List, Set
import hashlib
import json
import subprocess
from yasps.context import context
import numpy as np
from yasps.hessianKernelHeader import hessianKernelHeader
from yasps.hessianKernelFullProject import hessianKernelFullProject
from yasps.hessianKernelNoProject import hessianKernelNoProject
from yasps.hessianKernelHost import hessianKernelHost
from yasps.hessianKernelSeparateJacobian import (
  hessianKernelSeparateJacobian,
  USE_DIRECT_SEPARATED_JACOBIAN_CONTRACTION,
)

HESSIAN_KERNEL_CACHE_VERSION = (
  "v3_direct_compact_jacobian" if USE_DIRECT_SEPARATED_JACOBIAN_CONTRACTION
  else "v3_symbolic_jacobian_helpers"
)

def _stable_content_signature(payload, length: int = 24) -> str:
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()[:length]

def _bind_hessian_header(source: str, header_basename: str) -> str:
  marker = '#include "allHeaders.cuh"'
  if marker not in source:
    raise ValueError("Hessian CUDA source is missing the generated-header include.")
  return source.replace(marker, f'#include "{header_basename}"', 1)

def _write_generated_source(path: str, source: str, build_token: str) -> None:
  temporary_path = f"{path}.tmp_{build_token}"
  with open(temporary_path, "w") as f:
    f.write(source)
  os.replace(temporary_path, path)

class hessianAndGradientKernel:
  att_name_to_gradient_sizes: dict[str, Set[int]] = {}  # maps attribute names to their unique gradient sizes, this way we only need to generate unique gradient sizes once
  att_name_to_kernel: dict[str, hessianAndGradientKernel] = {}  # maps attribute names to their hessian and gradient kernel instances, this way we can just return the previous existing kernel


  def __init__(self, att: attribute, project_entire_hessian: bool, projection_method: int = 1, gradeient_only: bool = False, clear_separation: bool = True, jacobian_rows = 0, jacobian_cols = 0, hessian_row_size = 0, dynamic_term = False):
    self.__kernelString: str = ""
    self.__headerFileString: str = ""
    self.__kernel = None # the kernel for computhing the gradient and hessians
    self.__unique_gradient_sizes: Set[int] = set([]) # this will tell us the unique gradient sizes, we will use this to generate and regenerate kernel when there are new gradient sizes
    self.__project_entire_hessian = project_entire_hessian
    self.__projection_method = projection_method
    self.__gradient_only = gradeient_only
    self.__att = att
    self.__clear_separation = clear_separation
    self.__jacobian_rows = jacobian_rows
    self.__jacobian_cols = jacobian_cols
    self.__hessian_row_size = hessian_row_size
    self.__additional_compile_flags = []  # --ptxas-options=-v,-warn-spills,-warn-lmem-usage  use this for memory checking
    self.__dynamic_terms = dynamic_term
    self.__context = context()
    # self.__generateKernel(att)

  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  @timed("hessianAndGradientKernel.generateKernel")
  def generateKernel(
    self,
    unique_gradient_sizes: List[int],
    max_child_gradient_size: int,  # the maximum gradient segment size
    wrt: List[attribute], # wrt
    max_num_indices: int, # max number of indices for each local hessian
    global_jacobian_block_nonzero_attributes: List[attribute] = [],
    global_jacobian_block_nonzero_local_positions: List[int] = [],
    global_jacobian_children_sizes: List[int] = [],
    global_jacobian_children_spans: List[int] = [],
  ) -> None:
    # check if our unique gradient sizes contains the input gradient sizes
    # print("unique gradient sizes are", unique_gradient_sizes)
    if (set(unique_gradient_sizes).issubset(self.__unique_gradient_sizes) and self.__project_entire_hessian) or (max_child_gradient_size in self.__unique_gradient_sizes and not self.__project_entire_hessian):
      return
    if self.__project_entire_hessian:
      print("Unique gradient sizes before is", unique_gradient_sizes)
      self.__unique_gradient_sizes.update(unique_gradient_sizes)
      print("Unique gradient sizes after is", self.__unique_gradient_sizes)
    else:
      print("Max child gradient size before is", max_child_gradient_size)
      self.__unique_gradient_sizes.add(max_child_gradient_size)
      print("Unique gradient sizes after is", self.__unique_gradient_sizes)


    # if we need to separate the jacobian and hessian, the first thing we need to do is reconstruct the jacobian and hessian symbolically
    # which we will use to figure out how to compute the final hessian block by performing J_i^T H_ij J_j for each block
    separate_jacobian_kernel: hessianKernelSeparateJacobian = hessianKernelSeparateJacobian(self.__att, self.__gradient_only)
    if self.__clear_separation:
      separate_jacobian_kernel.create_multiplied_blocks(
        global_jacobian_block_nonzero_attributes,
        global_jacobian_block_nonzero_local_positions,
        global_jacobian_children_sizes,
        global_jacobian_children_spans
      )



    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = self.__att.deviceKernel.dependents + separate_jacobian_kernel.dependents
    sortedDatas: List[attribute] = self.__att.deviceKernel.kernelDatas
    sortedConnectivities: List[connectivity] = self.__att.deviceKernel.kernelConnectivity
    sortedPrimitiveUnions: List[primitiveUnion] = self.__att.deviceKernel.kernelPrimitiveUnions
    sorted_unique_gradient_sizes = sorted(
      int(size) for size in self.__unique_gradient_sizes if int(size) != 0
    )
    if self.__att.name == "":
      attributeName = f'attr_{self.__att.hash}'.replace("-", "_neg_")
    else:
      attributeName = self.__att.fullName

    hessian_header_string = hessianKernelHeader(
      self.__att,
      sorted_unique_gradient_sizes,
      sortedDependency,
    ).kernelString
    header_signature = _stable_content_signature({"header": hessian_header_string})
    header_basename = f"hessian_headers_{header_signature}.cuh"
    header_path = f".yasps_tmp/{header_basename}"

    generated_gradient_kernels = []
    for unique_gradient_size in sorted_unique_gradient_sizes:
      if self.__project_entire_hessian:
        kernel_source = hessianKernelFullProject(
          self.__att,
          unique_gradient_size,
          self.__gradient_only,
          max_num_indices,
          attributeName,
        ).kernelString
      elif not self.__clear_separation:
        kernel_source = hessianKernelNoProject(
          self.__att,
          unique_gradient_size,
          self.__gradient_only,
          max_num_indices,
          attributeName,
        ).kernelString
      else:
        kernel_source = separate_jacobian_kernel.generateKernelString(
          unique_gradient_size,
          max_num_indices,
          attributeName,
          len(wrt),
        )
      generated_gradient_kernels.append((
        unique_gradient_size,
        _bind_hessian_header(kernel_source, header_basename),
      ))

    self.__kernelString = _bind_hessian_header(
      prune_duplicate_functions(
        hessianKernelHost(
          self.__att,
          sorted_unique_gradient_sizes,
          max_child_gradient_size,
          self.__project_entire_hessian,
        ).kernelString
      ),
      header_basename,
    )

    compiler_identity = {
      "arch": "sm_89",
      "optimization": "-O3",
      "std": "c++17",
      "relocatable_device_code": True,
      "extra_flags": list(self.__additional_compile_flags),
    }
    device_units = []
    for item in (sortedDependency + [self.__att.deviceKernel]):
      device_unit_source = f'''
#include "{header_basename}"
extern "C"{{
{item.kernelString}
}}
'''
      device_unit_signature = _stable_content_signature({
        "cache_version": HESSIAN_KERNEL_CACHE_VERSION,
        "compiler": compiler_identity,
        "header": hessian_header_string,
        "source": device_unit_source,
      })
      device_units.append((device_unit_signature, device_unit_source))

    generation_metadata = {
      "cache_version": HESSIAN_KERNEL_CACHE_VERSION,
      "compiler": compiler_identity,
      "attribute": {
        "full_name": self.__att.fullName,
        "rows": int(self.__att.rows),
        "cols": int(self.__att.cols),
      },
      "wrt": [
        {
          "full_name": item.fullName,
          "rows": int(item.rows),
          "cols": int(item.cols),
          "size": int(item.size),
          "is_dynamic": bool(item.isDynamic),
        }
        for item in wrt
      ],
      "unique_gradient_sizes": sorted_unique_gradient_sizes,
      "max_child_gradient_size": int(max_child_gradient_size),
      "max_num_indices": int(max_num_indices),
      "project_entire_hessian": bool(self.__project_entire_hessian),
      "projection_method": int(self.__projection_method),
      "gradient_only": bool(self.__gradient_only),
      "clear_separation": bool(self.__clear_separation),
      "direct_separated_contraction_enabled": bool(USE_DIRECT_SEPARATED_JACOBIAN_CONTRACTION),
      "jacobian_rows": int(self.__jacobian_rows),
      "jacobian_cols": int(self.__jacobian_cols),
      "hessian_row_size": int(self.__hessian_row_size),
      "dynamic_term": bool(self.__dynamic_terms),
      "jacobian_nonzero_attributes": [
        item.fullNameWithHash for item in global_jacobian_block_nonzero_attributes
      ],
      "jacobian_nonzero_local_positions": [
        int(position) for position in global_jacobian_block_nonzero_local_positions
      ],
      "jacobian_children_sizes": [int(size) for size in global_jacobian_children_sizes],
      "jacobian_children_spans": [int(span) for span in global_jacobian_children_spans],
    }
    generation_signature = _stable_content_signature({
      "metadata": generation_metadata,
      "header": hessian_header_string,
      "host": self.__kernelString,
      "gradient_kernels": generated_gradient_kernels,
      "device_units": device_units,
    }, length=32)
    file_name = f".yasps_tmp/compute_hessian_and_gradient_{generation_signature}"

    if not os.path.exists(f'{file_name}.so'):
      build_token = f"{os.getpid()}_{id(self)}"
      if not os.path.exists(header_path):
        _write_generated_source(header_path, hessian_header_string, build_token)

      compile_jobs = []
      obj_files = []
      seen_obj_files = set()
      for device_unit_signature, device_unit_source in device_units:
        cu_file = f".yasps_tmp/hessian_device_{device_unit_signature}.cu"
        obj_file = f".yasps_tmp/hessian_device_{device_unit_signature}.o"
        if obj_file in seen_obj_files:
          continue
        seen_obj_files.add(obj_file)
        obj_files.append(obj_file)
        if os.path.exists(obj_file):
          continue
        _write_generated_source(cu_file, device_unit_source, build_token)
        temporary_obj_file = f"{obj_file}.tmp_{build_token}"
        compile_cmd = [
          "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
          "-O3",
          "-c", cu_file, "-o", temporary_obj_file,
          "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
          "--relocatable-device-code=true",
        ] + self.__additional_compile_flags
        print("Command is")
        print(" ".join(compile_cmd))
        compile_jobs.append((
          compile_cmd,
          subprocess.Popen(compile_cmd),
          temporary_obj_file,
          obj_file,
        ))

      gradient_obj_files = []
      for unique_gradient_size, kernel_source in generated_gradient_kernels:
        cu_file = f".yasps_tmp/hessian_gradient_{generation_signature}_{unique_gradient_size}.cu"
        obj_file = f".yasps_tmp/hessian_gradient_{generation_signature}_{unique_gradient_size}.o"
        gradient_obj_files.append(obj_file)
        _write_generated_source(cu_file, kernel_source, build_token)
        temporary_obj_file = f"{obj_file}.tmp_{build_token}"
        compile_cmd = [
          "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
          "-O3",
          "-c", cu_file, "-o", temporary_obj_file,
          "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
          "--relocatable-device-code=true",
        ] + self.__additional_compile_flags
        print("Command is")
        print(" ".join(compile_cmd))
        compile_jobs.append((
          compile_cmd,
          subprocess.Popen(compile_cmd),
          temporary_obj_file,
          obj_file,
        ))
      obj_files.extend(gradient_obj_files)

      kernel_cu_file = f"{file_name}.cu"
      _write_generated_source(kernel_cu_file, self.__kernelString, build_token)
      kernel_obj_file = f"{file_name}.o"
      temporary_kernel_obj_file = f"{kernel_obj_file}.tmp_{build_token}"
      kernel_compile_cmd = [
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
        "-O3",
        "-c", kernel_cu_file, "-o", temporary_kernel_obj_file,
        "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
        "--relocatable-device-code=true",
      ] + self.__additional_compile_flags
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
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3",
        *(obj_files + [kernel_obj_file]), "-o", temporary_device_link_obj,
        "--relocatable-device-code=true",
      ] + self.__additional_compile_flags
      print("Device link command: ")
      print(" ".join(dlink_cmd))
      subprocess.run(dlink_cmd, check=True)
      os.replace(temporary_device_link_obj, device_link_obj)

      temporary_shared_object = f"{file_name}.so.tmp_{build_token}"
      final_link_cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3",
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", temporary_shared_object,
        "-lcudart", "-lcuda",
      ] + self.__additional_compile_flags
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)
      os.replace(temporary_shared_object, f"{file_name}.so")


      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute_hessian_and_gradient_with_compression # get the compiled kernel
      self.__kernel.restype = ctypes.c_int # set the return type to None
      self.__kernel.argtypes = [
        # Data arrays
        *(ctypes.c_void_p for _ in sortedDatas),             # const double* for each data array
        *(ctypes.c_void_p for _ in sortedConnectivities),     # const unsigned int* for each connectivity index array
        *(ctypes.c_void_p for _ in (x for x in sortedConnectivities if x.dimension == 0)),  # const unsigned int* for CSR arrays
        *(ctypes.c_void_p for x in sortedPrimitiveUnions),    # const unsigned int* for each primitive union counts array
        # Other inputs
        ctypes.c_void_p,    # segment_indices
        ctypes.c_void_p,    # segment_sizes
        ctypes.c_void_p,    # local_permutations
        ctypes.c_void_p,    # lookups
        ctypes.c_void_p,    # coordinatesOuter
        ctypes.c_void_p,    # groupedIndicesInner
        ctypes.c_void_p,    # groupedIndicesOuter
        # Scalars
        ctypes.c_uint32,      # nth_gradient_size
        ctypes.c_uint32,      # projection_method
        # Outputs
        ctypes.c_void_p,    # gradient
        ctypes.c_void_p,    # hessian_blocks
        ctypes.c_void_p,    # diagonal
        ctypes.c_void_p,    # diagonal blocks
        ctypes.c_void_p,    # diagonal blocks start
        ctypes.c_void_p,    # gradient segments start
        # Other CPU arrays
        ctypes.c_void_p,    # unique_gradient_sizes
        ctypes.c_uint,      # num_unique_gradient_sizes
      ]
    else:
      self.__kernel = ctypes.CDLL(f"{file_name}.so").compute_hessian_and_gradient_with_compression # get the compiled kernel
      self.__kernel.restype = ctypes.c_int # set the return type to None
      self.__kernel.argtypes = [
        # Data arrays
        *(ctypes.c_void_p for _ in sortedDatas),             # const double* for each data array
        *(ctypes.c_void_p for _ in sortedConnectivities),     # const unsigned int* for each connectivity index array
        *(ctypes.c_void_p for _ in (x for x in sortedConnectivities if x.dimension == 0)),  # const unsigned int* for CSR arrays
        *(ctypes.c_void_p for x in sortedPrimitiveUnions),    # const unsigned int* for each primitive union counts array
        # Other inputs
        ctypes.c_void_p,    # segment_indices
        ctypes.c_void_p,    # segment_sizes
        ctypes.c_void_p,    # local_permutations
        ctypes.c_void_p,    # lookups
        ctypes.c_void_p,    # coordinatesOuter
        ctypes.c_void_p,    # groupedIndicesInner
        ctypes.c_void_p,    # groupedIndicesOuter
        # Scalars
        ctypes.c_uint32,      # nth_gradient_size
        ctypes.c_uint32,      # projection_method
        # Outputs
        ctypes.c_void_p,    # gradient
        ctypes.c_void_p,    # hessian_blocks
        ctypes.c_void_p,    # diagonal
        ctypes.c_void_p,    # diagonal blocks
        ctypes.c_void_p,    # diagonal blocks start
        ctypes.c_void_p,    # gradient segments start
        # Other CPU arrays
        ctypes.c_void_p,    # unique_gradient_sizes
        ctypes.c_uint,      # num_unique_gradient_sizes
      ]

  @timed("hessianAndGradientKernel.compute")
  def compute(
    self,
    attributeArgs: List[gpuarray.GPUArray],
    giKernel: gradientIndicesKernel,
    lookups: gpuarray.GPUArray,
    gradient: gpuarray.GPUArray,
    hessian_blocks: gpuarray.GPUArray,
    diagonal: gpuarray.GPUArray,
    diagonal_blocks: gpuarray.GPUArray,
    diagonal_blocks_start: gpuarray.GPUArray,
    gradient_segments_start: gpuarray.GPUArray
  ):
    # print("Unique gradient sizes cpu before hessian kernel:", giKernel.outputUniqueGradientSizesCPU)
    # print("Num unique gradient sizes cpu before hessian kernel:", giKernel.numUniqueGradientSizesCPU)
    if giKernel.numUniqueGradientSizesCPU == 0:
      # there is nothing to compute
      return
    assert self.__kernel is not None
    # self.__context.useNamedContext("dynamic_hessian" if self.__dynamic_terms else "static_hessian")
    self.__context.useDefaultContext()
    # self.__context.useNamedContext("hessian")
    error_code = self.__kernel(
      *[self.__to_void_p(x) for x in attributeArgs],
      self.__to_void_p(giKernel.outputIndices),
      self.__to_void_p(giKernel.outputSizes),
      self.__to_void_p(giKernel.outputPermutations),
      self.__to_void_p(lookups),
      self.__to_void_p(giKernel.outputCompressedCoordinateCountsOuter),
      self.__to_void_p(giKernel.outputGroupedIndicesInner),
      self.__to_void_p(giKernel.outputGroupedIndicesOuter),
      ctypes.c_uint32(0),
      ctypes.c_uint32(self.__projection_method),
      self.__to_void_p(gradient),
      self.__to_void_p(hessian_blocks),
      self.__to_void_p(diagonal),
      self.__to_void_p(diagonal_blocks),
      self.__to_void_p(diagonal_blocks_start),
      self.__to_void_p(gradient_segments_start),
      giKernel.outputUniqueGradientSizesCPU.ctypes.data_as(ctypes.c_void_p),
      ctypes.c_uint32(giKernel.numUniqueGradientSizesCPU)
    )
    if error_code != 0:
      raise RuntimeError(f"HessianAndGradientKernel.compute: Kernel execution failed with error code {error_code}")

  @property
  def kernelString(self) -> str:
    return self.__kernelString

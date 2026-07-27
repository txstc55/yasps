# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from yasps.backend import gpuarray
from yasps.gradientIndicesKernel import gradientIndicesKernel
from yasps.primitiveUnion import primitiveUnion
from yasps.helper import prune_duplicate_functions, timed
import os
import ctypes
from typing import List, Set
import hashlib
import subprocess
from yasps.context import context
import numpy as np
from yasps.hessianKernelHeader import hessianKernelHeader
from yasps.hessianKernelFullProject import hessianKernelFullProject
from yasps.hessianKernelNoProject import hessianKernelNoProject
from yasps.hessianKernelHost import hessianKernelHost
from yasps.hessianKernelSeparateJacobian import hessianKernelSeparateJacobian
from yasps.backend import is_metal
from pathlib import Path

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
    self.__metal_kernels = {}
    self.__metal_default_size = 0
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
    global_jacobian_block_nonzero_local_positions: List[attribute] = [],
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
    wrt_names = "_".join([att.fullName for att in wrt])
    size_names = "_".join([str(size) for size in unique_gradient_sizes])
    full_file_name = f"compute_hessian_and_gradient_for_{self.__att.fullName}_wrt_{wrt_names}_with_sizes_{size_names}"
    full_file_name_hashed = int(hashlib.sha256(full_file_name.encode('utf-8')).hexdigest(), 16)
    file_name = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}" + ("" if self.__project_entire_hessian else "_no_proj")
    if is_metal():
      self.__generateMetalKernel(
        sortedDependency,
        separate_jacobian_kernel,
        max_child_gradient_size,
        max_num_indices,
        attributeName=(
          f'att_{self.__att.hash}'.replace("-", "_neg_")
          if self.__att.name == ""
          else self.__att.fullName
        )
      )
      return
    # print(f"full file name: {full_file_name}\nhashed: {file_name}.cu")
    # print(f"hashed: {file_name}.cu")
    if not os.path.exists(f'{file_name}.so'):
      hessian_header_file = hessianKernelHeader(self.__att, self.__unique_gradient_sizes, sortedDependency)
      with open(".yasps_tmp/allHeaders.cuh", 'w') as f:
        f.write(hessian_header_file.kernelString)
        f.close()

      compile_jobs = []
      obj_files = []
      seen_obj_files = set([])
      for item in (sortedDependency + [self.__att.deviceKernel]):
        # we check if the .o file exists
        cu_file = f".yasps_tmp/{item.attributeName}.cu"
        obj_file = f".yasps_tmp/{item.attributeName}.o"
        if not obj_file in seen_obj_files:
          obj_files.append(obj_file)
        if (not os.path.exists(obj_file)) and (not obj_file in seen_obj_files):
          with open(cu_file, 'w') as f:
            f.write(f'''
#include "allHeaders.cuh"
extern "C"{{
{item.kernelString}
}}
''')
            f.close()
          compile_cmd = [
            "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
            "-O3",
            "-c", cu_file, "-o", obj_file,
            "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
            "--relocatable-device-code=true",
          ] + self.__additional_compile_flags
          print("Command is")
          print(" ".join(compile_cmd))
          job = subprocess.Popen(compile_cmd)
          compile_jobs.append(job)
        seen_obj_files.add(obj_file)

      # now actually generate the global kernel
      attributeName: str = ""
      if self.__att.name == "":
        attributeName = f'attr_{self.__att.hash}'.replace("-", "_neg_")
      else:
        attributeName = self.__att.fullName
      for unique_gradient_size in self.__unique_gradient_sizes:
        if unique_gradient_size == 0:
          continue
        cu_file = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}_fgs_{unique_gradient_size}" + (".cu" if self.__project_entire_hessian else "_no_proj.cu")
        obj_file = f".yasps_tmp/compute_hessian_and_gradient_for_{full_file_name_hashed}_fgs_{unique_gradient_size}" + (".o" if self.__project_entire_hessian else "_no_proj.o")
        obj_files.append(obj_file)
        # if not os.path.exists(obj_file):
        if True: # always regenerate the kernel because header has been replaced
          with open(cu_file, 'w') as f:
            if self.__project_entire_hessian:
              f.write(hessianKernelFullProject(self.__att, unique_gradient_size, self.__gradient_only, max_num_indices, attributeName).kernelString)
            elif not self.__clear_separation:
              f.write(hessianKernelNoProject(self.__att, unique_gradient_size, self.__gradient_only, max_num_indices, attributeName).kernelString)
            else:
              f.write(separate_jacobian_kernel.generateKernelString(unique_gradient_size, max_num_indices, attributeName))
            f.close()
          compile_cmd = [
            "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
            "-O3",
            "-c", cu_file, "-o", obj_file,
            "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
            "--relocatable-device-code=true",
          ] + self.__additional_compile_flags
          print("Command is")
          print(" ".join(compile_cmd))
          job = subprocess.Popen(compile_cmd)
          compile_jobs.append(job)

      # now we add the c functions that will go over all the unique gradient sizes
      self.__kernelString = hessianKernelHost(self.__att, self.__unique_gradient_sizes, max_child_gradient_size, self.__project_entire_hessian).kernelString
      # prune duplicate functions
      self.__kernelString = prune_duplicate_functions(self.__kernelString)
      # generate the code to check
      f = open(f"{file_name}.cu", "w")
      f.write(self.__kernelString)
      f.close()
      # Generate global kernel .o file
      kernel_cu_file = f"{file_name}.cu"
      kernel_obj_file = f"{file_name}.o"
      kernel_compile_cmd = [
        "nvcc", "-dc", "-Xcompiler", "-fPIC", "-std=c++17", "-arch=sm_89",
        "-O3",
        "-c", kernel_cu_file, "-o", kernel_obj_file,
        "-I/usr/include/eigen3", "--expt-relaxed-constexpr", "--disable-warnings",
        "--relocatable-device-code=true",
      ] + self.__additional_compile_flags
      print("Kernel compile command: ")
      print(" ".join(kernel_compile_cmd))
      job = subprocess.Popen(kernel_compile_cmd)
      compile_jobs.append(job)
      # Wait for all compilation jobs
      for job in compile_jobs:
        job.wait()


      # obj_files = list(set(obj_files))
      # Device link step: critical for CUDA separable compilation
      device_link_obj = f"{file_name}_device_link.o"
      dlink_cmd = [
        "nvcc", "-dlink", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3",
        *(obj_files + [kernel_obj_file]), "-o", device_link_obj,
        "--relocatable-device-code=true",
      ] + self.__additional_compile_flags
      print("Device link command: ")
      print(" ".join(dlink_cmd))
      subprocess.run(dlink_cmd, check=True)
      # Final shared object linking
      final_link_cmd = [
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-arch=sm_89",
        "-O3",
        kernel_obj_file, device_link_obj, *obj_files,
        "-o", f"{file_name}.so",
        "-lcudart", "-lcuda",
      ] + self.__additional_compile_flags
      print("Final link command: ")
      print(" ".join(final_link_cmd))
      subprocess.run(final_link_cmd, check=True)


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

  def __generateMetalKernel(
    self,
    sorted_dependency,
    separate_jacobian_kernel,
    max_child_gradient_size,
    max_num_indices,
    attributeName,
  ):
    from yasps.backend import metal_codegen
    from yasps.backend.metal_hessian_codegen import (
      translate_hessian_kernel,
    )

    if self.__project_entire_hessian:
      specialized_sizes = sorted(
        size for size in self.__unique_gradient_sizes if size != 0
      )
    else:
      specialized_sizes = [max_child_gradient_size]
      self.__metal_default_size = max_child_gradient_size

    global_sources = {}
    for specialized_size in specialized_sizes:
      function_name = (
        "compute_hessian_and_gradient_global_function_"
        f"final_gradient_size_{specialized_size}"
      )
      if self.__project_entire_hessian:
        cuda_source = hessianKernelFullProject(
          self.__att,
          specialized_size,
          self.__gradient_only,
          max_num_indices,
          attributeName
        ).kernelString
      elif not self.__clear_separation:
        cuda_source = hessianKernelNoProject(
          self.__att,
          specialized_size,
          self.__gradient_only,
          max_num_indices,
          attributeName
        ).kernelString
      else:
        cuda_source = separate_jacobian_kernel.generateKernelString(
          specialized_size,
          max_num_indices,
          attributeName
        )
      global_sources[specialized_size] = translate_hessian_kernel(
        cuda_source,
        function_name
      )

    all_device_kernels = (
      list(sorted_dependency) + [self.__att.deviceKernel]
    )
    for item in all_device_kernels:
      if not item.metalKernelString or not item.metalKernelHeader:
        raise RuntimeError(
          "hessianAndGradientKernel: generated Metal dependency "
          f"{item.attributeName!r} is incomplete"
        )

    matrix_path = (
      Path(metal_codegen.__file__).resolve().parents[1]
      / "kernel"
      / "Compute"
      / "metalMatrix.metal"
    )
    source_material = [
      matrix_path.read_text(encoding="utf-8"),
      *(item.metalKernelString for item in all_device_kernels),
      *(global_sources[size] for size in specialized_sizes),
    ]
    source_digest = hashlib.sha256(
      "\n".join(source_material).encode("utf-8")
    ).hexdigest()[:16]
    file_name = Path(
      f".yasps_tmp/hessian_{self.__att.fullNameWithHash}_"
      f"{source_digest}"
    )
    library_path = file_name.with_suffix(".metallib")
    header_path = file_name.parent / "yasps_hessian_headers.metal"

    if not library_path.exists():
      header_lines = ['#include "metalMatrix.metal"', ""]
      seen_headers = set()
      for item in all_device_kernels:
        if item.metalKernelHeader in seen_headers:
          continue
        header_lines.append(f"extern {item.metalKernelHeader};")
        seen_headers.add(item.metalKernelHeader)
      header_path.write_text(
        "\n".join(header_lines) + "\n",
        encoding="utf-8"
      )

      source_paths = []
      seen_sources = set()
      for item in all_device_kernels:
        if item.metalKernelString in seen_sources:
          continue
        item_digest = hashlib.sha256(
          item.metalKernelString.encode("utf-8")
        ).hexdigest()[:12]
        source_path = Path(
          f"{file_name}_{item.attributeName}_{item_digest}.metal"
        )
        source_path.write_text(
          f'#include "{header_path.name}"\n'
          f"{item.metalKernelString}\n",
          encoding="utf-8"
        )
        source_paths.append(source_path)
        seen_sources.add(item.metalKernelString)

      for specialized_size, source in global_sources.items():
        source_path = Path(
          f"{file_name}_size_{specialized_size}.metal"
        )
        source_path.write_text(source, encoding="utf-8")
        source_paths.append(source_path)

      gpuarray.compile_metal(
        source_paths,
        library_path,
        include_dirs=[file_name.parent, matrix_path.parent],
      )

    for specialized_size in specialized_sizes:
      function_name = (
        "compute_hessian_and_gradient_global_function_"
        f"final_gradient_size_{specialized_size}"
      )
      self.__metal_kernels[specialized_size] = gpuarray.MetalKernel(
        library_path,
        function_name,
        argument_buffer=True
      )
    self.__kernel = self.__metal_kernels

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
    if is_metal():
      unique_sizes = giKernel.outputUniqueGradientSizesCPU
      outer_indices = giKernel.outputGroupedIndicesOuter.get()[
        :giKernel.numUniqueGradientSizesCPU + 1
      ]
      for size_index, unique_size in enumerate(unique_sizes):
        instance_count = int(
          outer_indices[size_index + 1] - outer_indices[size_index]
        )
        if instance_count == 0:
          continue
        specialized_size = (
          int(unique_size)
          if self.__project_entire_hessian
          else self.__metal_default_size
        )
        kernel = self.__metal_kernels.get(specialized_size)
        if kernel is None:
          raise RuntimeError(
            "hessianAndGradientKernel: no Metal specialization for "
            f"gradient size {specialized_size}"
          )
        kernel.dispatch(
          attributeArgs + [
            giKernel.outputIndices,
            giKernel.outputSizes,
            giKernel.outputPermutations,
            lookups,
            giKernel.outputCompressedCoordinateCountsOuter,
            giKernel.outputGroupedIndicesInner,
            giKernel.outputGroupedIndicesOuter,
            np.uint32(size_index),
            np.uint32(self.__projection_method),
            gradient,
            hessian_blocks,
            diagonal,
            diagonal_blocks,
            diagonal_blocks_start,
            gradient_segments_start,
          ],
          instance_count,
          32
        )
      return
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

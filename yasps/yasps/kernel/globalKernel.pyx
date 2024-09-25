# cython: language_level=3
from __future__ import annotations
from yasps.deviceKernel import deviceKernel
from yasps.attribute import attribute
from yasps.connectivity import connectivity
from pycuda.compiler import SourceModule
import pycuda.driver as pd
from typing import Optional, List


def mangle_function_name(func_name, arg_types):
    """
    Generate the mangled name for a C++ function according to the Itanium C++ ABI.

    Parameters:
    - func_name: The name of the function (string).
    - arg_types: A list of argument types (list of strings).

    Returns:
    - The mangled function name (string).
    """
    mangled_name = '_Z' + str(len(func_name)) + func_name
    type_encodings = []
    types_seen = []

    base_type_encodings = {
        'void': 'v',
        'bool': 'b',
        'char': 'c',
        'signed char': 'a',
        'unsigned char': 'h',
        'short': 's',
        'unsigned short': 't',
        'int': 'i',
        'unsigned int': 'j',
        'long': 'l',
        'unsigned long': 'm',
        'long long': 'x',
        'unsigned long long': 'y',
        'float': 'f',
        'double': 'd',
        'long double': 'e',
        'wchar_t': 'w',
        'char16_t': 'Ds',
        'char32_t': 'Di',
        'nullptr_t': 'Dn',
    }

    def encode_type(arg_type):
        arg_type = arg_type.strip()

        # Handle pointers and references recursively
        pointer_prefix = ''
        while arg_type.endswith(('*', '&')):
            if arg_type.endswith('*'):
                pointer_prefix += 'P'
                arg_type = arg_type[:-1].strip()
            elif arg_type.endswith('&'):
                pointer_prefix += 'R'
                arg_type = arg_type[:-1].strip()

        # Handle const and volatile qualifiers
        cv_qualifiers = ''
        while arg_type.startswith(('const ', 'volatile ')):
            if arg_type.startswith('const '):
                cv_qualifiers += 'K'
                arg_type = arg_type[6:].strip()
            elif arg_type.startswith('volatile '):
                cv_qualifiers += 'V'
                arg_type = arg_type[9:].strip()

        # Encode the base type
        if arg_type in base_type_encodings:
            encoding = pointer_prefix + cv_qualifiers + base_type_encodings[arg_type]
        else:
            # For user-defined types
            parts = arg_type.split('::')
            encoding = 'N' + ''.join(f'{len(part)}{part}' for part in parts) + 'E'
            encoding = pointer_prefix + cv_qualifiers + encoding

        # Handle type substitutions
        if encoding in types_seen:
            index = types_seen.index(encoding)
            substitution = f'S{index}_'
            return substitution
        else:
            types_seen.append(encoding)
            return encoding

    for arg in arg_types:
        type_encoding = encode_type(arg)
        type_encodings.append(type_encoding)

    mangled_name += ''.join(type_encodings)
    return mangled_name


class globalKernel:
  def __init__(self, att: attribute):
    self.__kernelString: str = ""
    self.__kernel: Optional[pd.Function] = None
    self.__generateKernel(att)


  def __generateKernel(self, attr: attribute) -> None:
    ## first we get all the header functions
    sortedDependency: List[deviceKernel] = sorted(attr.deviceKernel.dependents, key = lambda x: x.kernelHeader)
    sortedDatas: List[attribute] = sorted(attr.deviceKernel.kernelDatas, key = lambda x: x.fullName)
    sortedConnectivities: List[connectivity] = sorted(attr.deviceKernel.kernelConnectivity, key = lambda x: x.fullName)
    # add the includes
    self.__kernelString += f'''
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define EIGEN_USE_GPU
#include <Eigen/Core>
#include <Eigen/Dense>
'''

    for item in sortedDependency:
      self.__kernelString += f"{item.kernelHeader};"
    self.__kernelString += f"{attr.deviceKernel.kernelHeader};"

    for item in sortedDependency:
      self.__kernelString += item.kernelString
    self.__kernelString += attr.deviceKernel.kernelString

    # now actually generate the global kernel
    attributeName: str = ""
    if attr.name == "":
      attributeName = f'attr_{attr.hash}'
    else:
      attributeName = attr.fullName

    self.__kernelString += f'''
__global__ void {attributeName}_global_function({"".join([f"const double* {x.fullName}_global_data, " for x in sortedDatas])}{"".join([f"const unsigned int* {x.fullName}_global_indices, " for x in sortedConnectivities])}double* result, unsigned int MAX_INDEX){{
  // first we get the index
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= MAX_INDEX){{
    return;
  }}
  // now we call the device function
  {attributeName}_device_function({"".join([f"{x.fullName}_global_data, " for x in sortedDatas])}{"".join([f"{x.fullName}_global_indices, " for x in sortedConnectivities])}index, result + index * {attr.size});
}}
'''
    # compile the code with eigen library and get the function
    mod = SourceModule(
      self.__kernelString,
      options = ["-std=c++11", '-O3', '-I/usr/include/eigen3', "--expt-relaxed-constexpr"],
      no_extern_c = True
    )
    print(self.__kernelString)

    # get the mangled name
    input_types = []
    for _ in sortedDatas:
      input_types.append("const double*")
    for _ in sortedConnectivities:
      input_types.append("const unsigned int*")
    input_types.append("double*")
    input_types.append("unsigned int")
    kernel_name: str = mangle_function_name(f"{attributeName}_global_function", input_types)
    print(f"kernel name: {kernel_name}")
    print(f"name gpt: _Z55scene0_mesh1_vertices_weighted_position_global_functionPKdS0_PKjPdj")
    self.__kernel = mod.get_function(kernel_name)



  @property
  def kernelString(self) -> str:
    return self.__kernelString

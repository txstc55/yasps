# def mangle_function_name(func_name, arg_types):
#     """
#     Generate the mangled name for a C++ function according to the Itanium C++ ABI.

#     Parameters:
#     - func_name: The name of the function (string).
#     - arg_types: A list of argument types (list of strings).

#     Returns:
#     - The mangled function name (string).
#     """
#     mangled_name = '_Z' + str(len(func_name)) + func_name
#     type_encodings = []
#     types_seen = []

#     base_type_encodings = {
#         'void': 'v',
#         'bool': 'b',
#         'char': 'c',
#         'signed char': 'a',
#         'unsigned char': 'h',
#         'short': 's',
#         'unsigned short': 't',
#         'int': 'i',
#         'unsigned int': 'j',
#         'long': 'l',
#         'unsigned long': 'm',
#         'long long': 'x',
#         'unsigned long long': 'y',
#         'float': 'f',
#         'double': 'd',
#         'long double': 'e',
#         'wchar_t': 'w',
#         'char16_t': 'Ds',
#         'char32_t': 'Di',
#         'nullptr_t': 'Dn',
#     }

#     def encode_type(arg_type):
#         arg_type = arg_type.strip()

#         # Handle pointers and references recursively
#         pointer_prefix = ''
#         while arg_type.endswith(('*', '&')):
#             if arg_type.endswith('*'):
#                 pointer_prefix += 'P'
#                 arg_type = arg_type[:-1].strip()
#             elif arg_type.endswith('&'):
#                 pointer_prefix += 'R'
#                 arg_type = arg_type[:-1].strip()

#         # Handle const and volatile qualifiers
#         cv_qualifiers = ''
#         while arg_type.startswith(('const ', 'volatile ')):
#             if arg_type.startswith('const '):
#                 cv_qualifiers += 'K'
#                 arg_type = arg_type[6:].strip()
#             elif arg_type.startswith('volatile '):
#                 cv_qualifiers += 'V'
#                 arg_type = arg_type[9:].strip()

#         # Encode the base type
#         if arg_type in base_type_encodings:
#             encoding = pointer_prefix + cv_qualifiers + base_type_encodings[arg_type]
#         else:
#             # For user-defined types
#             parts = arg_type.split('::')
#             encoding = 'N' + ''.join(f'{len(part)}{part}' for part in parts) + 'E'
#             encoding = pointer_prefix + cv_qualifiers + encoding

#         # Handle type substitutions
#         if encoding in types_seen:
#             index = types_seen.index(encoding)
#             substitution = f'S{index}_'
#             types_seen.append(encoding) # add the type regardless
#             return substitution
#         else:
#             types_seen.append(encoding)
#             return encoding

#     for arg in arg_types:
#         type_encoding = encode_type(arg)
#         type_encodings.append(type_encoding)

#     mangled_name += ''.join(type_encodings)
#     return mangled_name

import os
def get_mangled_name(kernel_header: str, kernel_name: str) -> str:
  f = open(kernel_name+".cu", 'w')
  f.write(f'''
#include <cuda.h>
{kernel_header}{{
}}
''')
  f.close()
  # compile the code
  os.system(f"nvcc -c {kernel_name}.cu -o {kernel_name}.so ")
  # get the mangled name
  os.system(f"nm {kernel_name}.so | grep {kernel_name} > {kernel_name}.nm")
  f = open(f"{kernel_name}.nm", 'r')
  lines = f.readlines()
  f.close()

  # remove files
  os.system(f"rm {kernel_name}.cu")
  os.system(f"rm {kernel_name}.so")
  os.system(f"rm {kernel_name}.nm")
  print(lines)
  if len(lines) > 0:
    return (lines[0].split(" ")[2].strip())
  return ""

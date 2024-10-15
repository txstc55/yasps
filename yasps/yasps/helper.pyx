import os
def get_mangled_name(kernel_header: str, kernel_name: str) -> str:
  f = open("tmp_compile.cu", 'w')
  f.write(f'''
#include <cuda.h>
{kernel_header}{{
}}
''')
  f.close()
  # compile the code
  os.system(f"nvcc -c tmp_compile.cu -o tmp_compile.so ")
  # get the mangled name
  os.system(f"nm tmp_compile.so | grep {kernel_name} > tmp_compile.nm")
  f = open(f"tmp_compile.nm", 'r')
  lines = f.readlines()
  f.close()

  # remove files
  os.system(f"rm tmp_compile.cu")
  os.system(f"rm tmp_compile.so")
  os.system(f"rm tmp_compile.nm")
  ## sort lines by length of the line
  lines.sort(key=len)
  if len(lines) > 0:
    return (lines[0].split(" ")[2].strip())
  return ""

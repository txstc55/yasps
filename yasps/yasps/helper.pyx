import os
# for solving the name mangling issue when we compile a file
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


def extract_block(flattened_mat, rows: int, cols: int, block_start_row: int, block_start_col: int, block_row_size: int, block_col_size: int):
  # extract the block of a matrix
  if rows * cols != len(flattened_mat):
    raise ValueError("yasps.Helper: The number of elements in the matrix should be equal to rows * cols")
  if block_start_row + block_row_size > rows or block_start_col + block_col_size > cols:
    raise ValueError(f"yasps.Helper: The block is out of the matrix, rows: {rows}, cols: {cols}, block_start_row: {block_start_row}, block_start_col: {block_start_col}, block_row_size: {block_row_size}, block_col_size: {block_col_size}")
  block = []
  for i in range(block_start_row, block_start_row + block_row_size):
    block.extend(flattened_mat[i * cols + block_start_col: i * cols + block_start_col + block_col_size])
  return block


import re

def prune_duplicate_functions(code_string):
  # Pattern to match function signatures
  pattern = r'__device__\s+__inline__\s+void\s+\w+\s*\(.*?\)\s*(;|\{)'
  matches = list(re.finditer(pattern, code_string, re.DOTALL))
  functions_seen = set()
  output_code = ''
  pos = 0
  for i, match in enumerate(matches):
    start = match.start()
    # Append code before the function
    output_code += code_string[pos:start]
    # Determine if it's a declaration or implementation
    signature = match.group()
    if signature.strip().endswith(';'):
      # Function declaration
      # Find the semicolon
      end = code_string.find(';', match.end()-1) + 1
      function_code = code_string[match.start():end]
    else:
      # Function implementation
      # Need to find the matching closing brace
      brace_count = 0
      idx = match.end()-1  # position of the '{'
      while idx < len(code_string):
        if code_string[idx] == '{':
          brace_count +=1
        elif code_string[idx] == '}':
          brace_count -=1
          if brace_count ==0:
            idx +=1
            break
        idx +=1
      end = idx
      function_code = code_string[match.start():end]
    # Check if signature is already seen
    if signature.strip() not in functions_seen:
      functions_seen.add(signature.strip())
      output_code += function_code
    else:
      # Skip the function, do not add to output_code
      pass
    # Update position
    pos = end

  # Append the remaining code
  output_code += code_string[pos:]
  return output_code

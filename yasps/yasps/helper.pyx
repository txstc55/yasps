import os
import numpy as np
# for solving the name mangling issue when we compile a file
def get_mangled_name(kernel_header: str, kernel_name: str) -> str:
  f = open(".yasps_tmp/tmp_compile.cu", 'w')
  f.write(f'''
#include <cuda.h>
{kernel_header}{{
}}
''')
  f.close()
  # compile the code
  os.system("nvcc -c .yasps_tmp/tmp_compile.cu -o .yasps_tmp/tmp_compile.so ")
  # get the mangled name
  os.system(f"nm .yasps_tmp/tmp_compile.so | grep {kernel_name} > .yasps_tmp/tmp_compile.nm")
  f = open(".yasps_tmp/tmp_compile.nm", 'r')
  lines = f.readlines()
  f.close()

  # remove files
  os.system("rm .yasps_tmp/tmp_compile.cu")
  os.system("rm .yasps_tmp/tmp_compile.so")
  os.system("rm .yasps_tmp/tmp_compile.nm")
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
  pattern = r'__device__\s+void\s+\w+\s*\(.*?\)\s*(;|\{)'
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

def energy_process_one_path_vector(
    path,
    start_i,
    end_i,
    indicesCPU,
    wrtStartIndicesAndSize
):
  N = end_i - start_i
  currentIndices = np.arange(start_i, end_i, dtype=np.uint32)
  data_steps = [(h, op) for (h, op) in path if op < 0]  # operation == -1 => data
  D = len(data_steps)
  appended = np.empty((D, N), dtype=np.uint32)
  data_step_idx = 0
  for (hashValue, operation) in path:
    if operation >= 0:
      rowIndex = operation
      currentIndices = indicesCPU[hashValue][currentIndices, rowIndex]
    else:
      (start_pos, size, is_primitive) = wrtStartIndicesAndSize[hashValue]
      if is_primitive:
        updated = start_pos + currentIndices * size
        appended[data_step_idx] = updated
        currentIndices = updated
      else:
        appended[data_step_idx].fill(start_pos)
        currentIndices.fill(start_pos)
      data_step_idx += 1
  path_results = [[] for _ in range(N)]
  for i_local in range(N):
    for d in range(D):
      path_results[i_local].append(appended[d, i_local])
  return path_results


def energy_process_work(
  start_index: int,
  end_index: int,
  max_index: int,
  duplicatedPaths,
  indicesCPU,
  wrtStartIndicesAndSize
):
  end_index = min(end_index, max_index)
  N = end_index - start_index
  if N <= 0:
    return []
  all_results = [[] for _ in range(N)]
  for path in duplicatedPaths:
    path_results = energy_process_one_path_vector(path, start_index, end_index, indicesCPU, wrtStartIndicesAndSize)
    for i_local in range(N):
      all_results[i_local].extend(path_results[i_local])
  current_process_all_indices = []
  for i_local_list in all_results:
    current_process_all_indices.extend(i_local_list)
  return current_process_all_indices

# create a wrapper function for timing
import time
DEBUG_TIME = True
from functools import wraps
def timed(msg="Function"):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      if DEBUG_TIME:
        print(f"[{msg}] Starting...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed_ms = (end - start) * 1000
        print(f"[{msg}] Finished in {elapsed_ms:.2f} ms")
        return result
      else:
        return func(*args, **kwargs)
    return wrapper
  return decorator

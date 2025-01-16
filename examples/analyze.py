file_path = "cloth/cloth_1000.log"
file = open(file_path, "r")

import re
code_generation_time = 0.0
total_nnz = 0
gradient_size = 0
sparse_index_time = 0.0
autodiff_computation_time = 0.0
compile_time = 0.0
kernel_execution_time = 0.0
solver_iterations = 0
solver_time = 0.0
total_iterations = 0
for line in file:
  if line.startswith("Code generation time:"):
    code_generation_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Total NNZ is"):
    total_nnz += int(re.findall("\d+", line)[0])
  elif line.startswith("The size of the gradient is"):
    gradient_size += int(re.findall("\d+", line)[0])
  elif line.startswith("Sparse indices generation"):
    sparse_index_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Autodiff computation"):
    autodiff_computation_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Compilation time"):
    compile_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Kernel execution time"):
    kernel_execution_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Solver converged"):
    solver_iterations += int(re.findall("\d+", line)[0])
    total_iterations += 1
  elif line.startswith("Solver time"):
    solver_time += float(re.findall("\d+\.\d+", line)[0])


print(f"{gradient_size} & {total_nnz} & {sparse_index_time:.5g} & {code_generation_time:.5g} & {compile_time:.5g} & {autodiff_computation_time:.5g} & {kernel_execution_time:.5g} & {solver_time:.5g} & {solver_iterations} & {total_iterations} \\\\")

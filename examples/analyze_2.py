file_path = "rolling_ball/rolling_ball.log"
file = open(file_path, "r")

import re
code_generation_time = 0.0
total_nnz0 = 0
total_nnz1 = 0
gradient_size0 = 0
gradient_size1 = 0
sparse_index_time = 0.0
autodiff_computation_time0 = 0.0
autodiff_computation_time1 = 0.0
compile_time = 0.0
kernel_execution_time = 0.0
solver_iterations0 = 0
solver_iterations1 = 0
solver_time0 = 0.0
solver_time1 = 0.0

nnz_which = 0
gradient_which = 0
autodiff_which = 0
solver_iteration_which = 0
solver_time_which = 0
for line in file:
  if line.startswith("Code generation time:"):
    code_generation_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Total NNZ is"):
    if nnz_which == 0:
      total_nnz0 += int(re.findall("\d+", line)[0])
      nnz_which = 1
    else:
      total_nnz1 += int(re.findall("\d+", line)[0])
      nnz_which = 0
  elif line.startswith("The size of the gradient is"):
    if gradient_which == 0:
      gradient_size0 += int(re.findall("\d+", line)[0])
      gradient_which = 1
    else:
      gradient_size1 += int(re.findall("\d+", line)[0])
      gradient_which = 0
  elif line.startswith("Sparse indices generation"):
    sparse_index_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Autodiff computation"):
    if autodiff_which == 0:
      autodiff_computation_time0 += float(re.findall("\d+\.\d+", line)[0])
      autodiff_which = 1
    else:
      autodiff_computation_time1 += float(re.findall("\d+\.\d+", line)[0])
      autodiff_which = 0
  elif line.startswith("Compilation time"):
    compile_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Kernel execution time"):
    kernel_execution_time += float(re.findall("\d+\.\d+", line)[0])
  elif line.startswith("Solver converged"):
    if solver_iteration_which == 0:
      solver_iterations0 += int(re.findall("\d+", line)[0])
      solver_iteration_which = 1
    else:
      solver_iterations1 += int(re.findall("\d+", line)[0])
      solver_iteration_which = 0
  elif line.startswith("Solver time"):
    if solver_time_which == 0:
      solver_time0 += float(re.findall("\d+\.\d+", line)[0])
      solver_time_which = 1
    else:
      solver_time1 += float(re.findall("\d+\.\d+", line)[0])
      solver_time_which = 0


print(f"{gradient_size0}, {gradient_size1} & {total_nnz0}, {total_nnz1} & {sparse_index_time:.5g} & {code_generation_time:.5g} & {compile_time:.5g} & {autodiff_computation_time0:.5g}, {autodiff_computation_time1:.5g} & {kernel_execution_time:.5g} & {solver_time0:.5g}, {solver_time1:.5g} & {solver_iterations0}, {solver_iterations1} & 500 \\\\")

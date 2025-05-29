import numpy as np
import ctypes
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
def to_void_p(x: gpuarray.GPUArray):
  if x is None or x.size == 0:
    # Return a NULL pointer if array is empty
    return ctypes.c_void_p(None)
  assert x.gpudata is not None
  return ctypes.c_void_p(int(x.gpudata))

## REMINDER
# THE REASON WHY THIS HAPPENS
# IS BECAUSE YOU DIDN'T DO SOMETHING LIKE DOUBLE ARR[42] = {0};
# ALWAYS FUCKING DO THIS I GUESS
data = np.load("dumped_kernel_data.npz", allow_pickle=True)
attributes=data["attributes"]
gikernel_stuffs=data["gikernel_stuffs"]
extra_cint_args=data["extra_cint_args"]
outputs=data["outputs"]
unique_gradient_sizes_cpu=data["unique_gradient_sizes_cpu"]
num_unique_gradient_sizes_cpu=data["num_unique_gradient_sizes_cpu"]
# print(attributes)
attributes_gpu = [gpuarray.to_gpu(attr) for attr in attributes]
gikernel_stuffs_gpu = [gpuarray.to_gpu(stuff) for stuff in gikernel_stuffs]
outputs_gpu = [gpuarray.to_gpu(out) for out in outputs]


kernel = ctypes.CDLL(".yasps_tmp/compute_hessian_and_gradient_for_15534483392287658020200411281031352894001573481165142211208581455804831666452.so").compute_hessian_and_gradient_with_compression
kernel.restype = None
kernel.argtypes = [
  # Data arrays
  *(ctypes.c_void_p for _ in attributes),             # const double* for each data array
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
  ctypes.c_uint32,      # max_num_indices
  ctypes.c_uint32,      # projection_method
  # Outputs
  ctypes.c_void_p,    # gradient
  ctypes.c_void_p,    # hessian_blocks
  ctypes.c_void_p,    # diagonal
  # Other CPU arrays
  ctypes.c_void_p,    # unique_gradient_sizes
  ctypes.c_uint,      # num_unique_gradient_sizes
]
kernel(
  *[to_void_p(x) for x in attributes_gpu],
  *[to_void_p(x) for x in gikernel_stuffs_gpu],
  ctypes.c_uint32(0),
  ctypes.c_uint32(extra_cint_args[1]),
  ctypes.c_uint32(extra_cint_args[2]),
  *[to_void_p(x) for x in outputs_gpu],
  unique_gradient_sizes_cpu.ctypes.data_as(ctypes.c_void_p),
  ctypes.c_uint32(num_unique_gradient_sizes_cpu)
)

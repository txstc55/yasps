import pycuda.autoinit  # Automatically initializes CUDA driver
from yasps import attribute
from pycuda import gpuarray
from yasps import scene
import numpy as np
x = attribute("x", value = [1, 2, 3, 4, 5, 6])
print(x.rows)
print(x.value.get())
# x = gpuarray.to_gpu(np.array([1, 2, 3]))
# print(x.get())
#

x = scene("abc")
x.addMesh("abc")
print(x.abc)

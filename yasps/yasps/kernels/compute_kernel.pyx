# the compute kernel class
# it does only the computation, does not care about how the data is stored
# or fetched
# we will only store the string here and not the compiled code

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class kernel:
  def __init__(self, kernel_string: str):
    self.__kernel_string = kernel_string
    self.__compiled_kernel = None

  @property
  def kernel_string(self)->str:
    return self.__kernel_string

  @property
  def compiled_kernel(self)->pycuda.driver.Function:
    return self.__compiled_kernel

  @compiled_kernel.setter
  def compiled_kernel(self, compiled_kernel: pycuda.driver.Function):
    self.__compiled_kernel = compiled_kernel

  def __str__(self)->str:
    return f"kernel({self.kernel_string})"

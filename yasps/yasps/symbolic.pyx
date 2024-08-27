import pycuda
import numpy as np


class symbolic:
  def __init__(self):
    self.value = 0
    self.count = 0
    self.data_size = 0
    self.total_size = 0

from yasps import gpu_data
import numpy as np
if __name__ == "__main__":
  x = gpu_data.gpu_data()
  x.cpu_to_gpu(np.array([1,2,3]))
  print(f"value: {x.value}")
  print(f"count: {x.count}")
  print(f"data_size: {x.data_size}")

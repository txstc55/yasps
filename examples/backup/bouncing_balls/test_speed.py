import time

blockDimensions = [7, 5, 8, 4, 25, 6, 19, 2] * 100

# Original
r1 = 0
start = time.time()
for _ in range(1000):
  r1 = max(blockDimensions[::2]) // 3 * 3 + 3
print("Original:", r1, "Time:", time.time() - start)

# Optimized
r2 = 0
start = time.time()
for _ in range(1000):
  r2 = (max(blockDimensions[::2]) // 3) * 3 + 3
print("Optimized:", r2, "Time:", time.time() - start)

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time

# Define the CUDA kernel code as a string
kernel_code = """
#include <Eigen/Dense>

__global__ void matrix_multiplys(double *A_data, double *B_data, double *C_data, int num_matrices) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_matrices) {
        // Each thread works on its corresponding matrix
        int matrix_offset = idx * 9; // Each 3x3 matrix has 9 elements

        // Map the input data to Eigen matrices (RowMajor to match NumPy)
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> A(A_data + matrix_offset);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> B(B_data + matrix_offset);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> C(C_data + matrix_offset);
        Eigen::Matrix<double, 3, 3> F;
        F.resize(A.rows(), B.cols());
        F = A * B;
        // First matrix multiplication: D = A * B
        // Eigen::Matrix<double, 3, 3> D = A * B;

        // Second matrix multiplication: E = D * A
        Eigen::Matrix<double, 3, 3> E = F * A;

        // Store the result in the output matrix
        C = E;
    }
}
"""

# Update the Eigen include directory path as needed
eigen_include_dir = "/usr/include/eigen3"  # Modify this path if Eigen is installed elsewhere

# Compiler options
compiler_options = [
    f"-I{eigen_include_dir}",  # Include path for Eigen headers
    "-std=c++11",  # Use C++11 standard
    "--expt-relaxed-constexpr",
]

# Compile the CUDA kernel without wrapping in extern "C"
mod = SourceModule(kernel_code, options=compiler_options, no_extern_c=True)

# Retrieve the kernel function

try:
    matrix_multiply = mod.get_function("matrix_multiply")
except cuda.LogicError:
    matrix_multiply = mod.get_function("_Z16matrix_multiplysPdS_S_i")

# Number of matrices
num_matrices = 1000000  # 1 million matrices

# Prepare input matrices (1M matrices of 3x3)
A_host = np.random.rand(num_matrices, 3, 3).astype(np.float64)
B_host = np.random.rand(num_matrices, 3, 3).astype(np.float64)

# Flatten the matrices into 1D arrays for GPU processing
A_host_flat = A_host.flatten()
B_host_flat = B_host.flatten()

# Allocate memory for the output matrices
C_host = np.empty_like(A_host_flat)

# Allocate device memory and copy input data to the device
A_device = cuda.mem_alloc(A_host_flat.nbytes)
B_device = cuda.mem_alloc(B_host_flat.nbytes)
C_device = cuda.mem_alloc(C_host.nbytes)

cuda.memcpy_htod(A_device, A_host_flat)
cuda.memcpy_htod(B_device, B_host_flat)

# Define block and grid dimensions
block_dim = (256, 1, 1)  # 256 threads per block (this can be tuned)
grid_dim = ((num_matrices + block_dim[0] - 1) // block_dim[0], 1, 1)  # Enough blocks to cover 1M matrices

# Create CUDA events to measure time
start = cuda.Event()
end = cuda.Event()

# Record the start event
start.record()

# Perform matrix multiplication on 1 million matrices
matrix_multiply(A_device, B_device, C_device, np.int32(num_matrices), block=block_dim, grid=grid_dim)

# Record the end event
end.record()

# Wait for the event to complete
end.synchronize()

# Calculate the elapsed time in milliseconds
elapsed_time = start.time_till(end)

# Copy the result back to the host
cuda.memcpy_dtoh(C_host, C_device)

# Reshape the result back to (num_matrices, 3, 3)
C_host = C_host.reshape(num_matrices, 3, 3)

print("Sample result of the matrix multiplication (C = (A * B) * A):")
print("A:\n", A_host[0])
print("B:\n", B_host[0])
print("Ground truth:\n" , np.dot(np.dot(A_host[0], B_host[0]), A_host[0]))
print("C:\n", C_host[0])  # Print the result of the first matrix multiplication

# Log the total time
print(f"Total time for 1 million matrix multiplications: {elapsed_time / 1000:.6f} seconds")

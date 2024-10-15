import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import time

def generate_large_symmetric_blocks(total_size, num_blocks):
    # Use small block sizes for memory efficiency, e.g., 2x2 blocks
    block_size = (2, 2)
    block_rows, block_cols = block_size

    # Randomly generate positions for the upper triangle
    np.random.seed(0)  # For reproducibility
    block_row_indices = np.random.randint(0, total_size - block_rows + 1, size=num_blocks)
    block_col_indices = np.random.randint(0, total_size - block_cols + 1, size=num_blocks)

    # Keep only upper triangular positions
    mask = block_row_indices <= block_col_indices
    block_row_indices = block_row_indices[mask]
    block_col_indices = block_col_indices[mask]
    num_blocks = len(block_row_indices)

    blocks_data = []
    for i in range(num_blocks):
        # Generate symmetric positive definite blocks
        block = np.random.rand(block_rows, block_cols).astype(np.float64)
        block = (block + block.T) / 2  # Make symmetric
        block += block_rows * np.eye(block_rows, dtype=np.float64)  # Make positive definite
        blocks_data.append(block)
    block_row_sizes = np.full(num_blocks, block_rows, dtype=np.int32)
    block_col_sizes = np.full(num_blocks, block_cols, dtype=np.int32)
    return (blocks_data, block_row_indices, block_col_indices,
            block_row_sizes, block_col_sizes)

def assemble_data_on_gpu(blocks_data, block_row_indices, block_col_indices,
                         block_row_sizes, block_col_sizes):
    # Flatten all block data and create pointers
    data_list = []
    ptr_list = []
    offset = 0
    for block in blocks_data:
        data_list.extend(block.flatten())
        ptr_list.append(offset)
        offset += block.size
    data_array = np.array(data_list, dtype=np.float64)
    ptr_array = np.array(ptr_list, dtype=np.int32)

    # Transfer data to GPU
    data_gpu = gpuarray.to_gpu(data_array)
    ptr_gpu = gpuarray.to_gpu(ptr_array)
    row_indices_gpu = gpuarray.to_gpu(block_row_indices.astype(np.int32))
    col_indices_gpu = gpuarray.to_gpu(block_col_indices.astype(np.int32))
    row_sizes_gpu = gpuarray.to_gpu(block_row_sizes)
    col_sizes_gpu = gpuarray.to_gpu(block_col_sizes)
    num_blocks = len(blocks_data)

    return (data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
            row_sizes_gpu, col_sizes_gpu, num_blocks)

def spmv_symmetric_cuda(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                        row_sizes_gpu, col_sizes_gpu, num_blocks, x_gpu, y_gpu, total_size):
    # CUDA kernel for symmetric SpMV: y += A * x + A^T * x
    mod = SourceModule("""
    __global__ void spmv_symmetric(
        const double *data,
        const int *ptr,
        const int *row_indices,
        const int *col_indices,
        const int *row_sizes,
        const int *col_sizes,
        const int num_blocks,
        const double *x,
        double *y)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_blocks)
        {
            int data_start = ptr[idx];
            int row_start = row_indices[idx];
            int col_start = col_indices[idx];
            int row_size = row_sizes[idx];
            int col_size = col_sizes[idx];

            // Compute y += A_block * x (upper triangle)
            for (int i = 0; i < row_size; ++i)
            {
                double y_row = 0.0f;
                for (int j = 0; j < col_size; ++j)
                {
                    double val = data[data_start + i * col_size + j];
                    double x_col = x[col_start + j];
                    y_row += val * x_col;
                }
                atomicAdd(&y[row_start + i], y_row);
            }

            // If not on the diagonal, compute y += A_block^T * x (lower triangle)
            if (row_start != col_start)
            {
                for (int j = 0; j < col_size; ++j)
                {
                    double y_col = 0.0f;
                    for (int i = 0; i < row_size; ++i)
                    {
                        double val = data[data_start + i * col_size + j];
                        double x_row = x[row_start + i];
                        y_col += val * x_row;
                    }
                    atomicAdd(&y[col_start + j], y_col);
                }
            }
        }
    }
    """)
    spmv_symmetric = mod.get_function("spmv_symmetric")
    block_dim = 256
    grid_dim = (num_blocks + block_dim - 1) // block_dim
    spmv_symmetric(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                   row_sizes_gpu, col_sizes_gpu, np.int32(num_blocks),
                   x_gpu, y_gpu,
                   block=(block_dim, 1, 1), grid=(grid_dim, 1))

def spmv_wrapper(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                 row_sizes_gpu, col_sizes_gpu, num_blocks, x_gpu, total_size):
    y_gpu = gpuarray.zeros(total_size, dtype=np.float64)
    spmv_symmetric_cuda(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                        row_sizes_gpu, col_sizes_gpu, num_blocks, x_gpu, y_gpu, total_size)
    return y_gpu

def conjugate_gradient_gpu(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                           row_sizes_gpu, col_sizes_gpu, num_blocks, b_gpu, x_gpu, total_size,
                           tol=1e-6, maxiter=1000):
    # Initialize
    r_gpu = b_gpu - spmv_wrapper(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                                 row_sizes_gpu, col_sizes_gpu, num_blocks, x_gpu, total_size)
    p_gpu = r_gpu.copy()
    rsold_gpu = gpuarray.dot(r_gpu, r_gpu)
    for i in range(maxiter):
        Ap_gpu = spmv_wrapper(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                              row_sizes_gpu, col_sizes_gpu, num_blocks, p_gpu, total_size)
        dot_pAp = gpuarray.dot(p_gpu, Ap_gpu)
        alpha_gpu = rsold_gpu / dot_pAp
        x_gpu = x_gpu + alpha_gpu * p_gpu
        r_gpu = r_gpu - alpha_gpu * Ap_gpu
        rsnew_gpu = gpuarray.dot(r_gpu, r_gpu)
        rsnew = rsnew_gpu.get()
        if np.sqrt(rsnew) < tol:
            print(f"Converged at iteration {i}")
            break
        beta_gpu = rsnew_gpu / rsold_gpu
        p_gpu = r_gpu + beta_gpu * p_gpu
        rsold_gpu = rsnew_gpu
    return x_gpu

def main():
    total_size = 1000000  # Matrix size of 1 million
    num_blocks = 5000000  # 5 million blocks
    print(f"Generating a sparse symmetric matrix of size {total_size} with {num_blocks} blocks.")

    # Generate blocks and positions
    start_time = time.time()
    (blocks_data, block_row_indices, block_col_indices,
     block_row_sizes, block_col_sizes) = generate_large_symmetric_blocks(total_size, num_blocks)
    gen_time = time.time() - start_time
    print(f"Matrix generation time: {gen_time:.2f} seconds.")

    # Assemble data on GPU
    start_time = time.time()
    (data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
     row_sizes_gpu, col_sizes_gpu, num_blocks_actual) = assemble_data_on_gpu(
        blocks_data, block_row_indices, block_col_indices,
        block_row_sizes, block_col_sizes)
    assemble_time = time.time() - start_time
    print(f"Data assembled on GPU in {assemble_time:.2f} seconds.")

    # Generate a random true solution and compute b = A * x_true
    x_true_cpu = np.random.rand(total_size).astype(np.float64)
    x_true_gpu = gpuarray.to_gpu(x_true_cpu)
    print("Computing b = A * x_true...")
    start_time = time.time()
    b_gpu = spmv_wrapper(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                         row_sizes_gpu, col_sizes_gpu, num_blocks_actual, x_true_gpu, total_size)
    b_compute_time = time.time() - start_time
    print(f"Computed b in {b_compute_time:.2f} seconds.")

    # Initialize x_gpu to zeros
    x_gpu = gpuarray.zeros(total_size, dtype=np.float64)

    # Solve Ax = b using CG on GPU
    print("Starting Conjugate Gradient solver...")
    start_time = time.time()
    x_gpu = conjugate_gradient_gpu(data_gpu, ptr_gpu, row_indices_gpu, col_indices_gpu,
                                   row_sizes_gpu, col_sizes_gpu, num_blocks_actual, b_gpu, x_gpu, total_size)
    cg_time = time.time() - start_time
    print(f"Conjugate Gradient solver completed in {cg_time:.2f} seconds.")

    # Verify the solution (optional, can be time-consuming for large vectors)
    """
    x_computed = x_gpu.get()
    error = np.linalg.norm(x_computed - x_true_cpu)
    print("Error in solution:", error)
    """

if __name__ == "__main__":
    main()

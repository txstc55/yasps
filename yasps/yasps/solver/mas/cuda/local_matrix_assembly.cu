#include <cstdint>

static __device__ double yasps_mas_atomic_add(double* address, double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits, assumed;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

// Scatter one already-remapped dense block into a domain principal matrix.
// Entries whose endpoint has domain_local_start == UINT32_MAX are cross-domain
// and intentionally remain only in the global SpMV.
extern "C" __global__ void yasps_mas_scatter_local_block(
    const double* block, double* local_matrix,
    std::uint32_t block_rows, std::uint32_t block_cols,
    std::uint32_t local_row, std::uint32_t local_col,
    std::uint32_t local_leading_dimension) {
  const std::uint32_t entry = blockIdx.x * blockDim.x + threadIdx.x;
  if (entry >= block_rows * block_cols) return;
  const std::uint32_t row = entry / block_cols;
  const std::uint32_t col = entry % block_cols;
  yasps_mas_atomic_add(
      local_matrix + (local_row + row) * local_leading_dimension + local_col + col,
      block[entry]);
}

static __device__ unsigned long long yasps_mas_hash_key(
    unsigned long long key, unsigned int capacity) {
  return (key * 11400714819323198485ull) & (capacity - 1);
}

// Map scalar-start input coordinates through a fixed composed node map and
// reduce equal parent blocks into an open-addressed device hash table.
extern "C" __global__ void yasps_mas_hash_assemble(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, unsigned int block_rows, unsigned int block_cols,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_dimensions, unsigned int fine_scalar_dofs,
    const unsigned int* fine_to_level, unsigned int parent_node_count,
    bool symmetric_storage, bool transpose,
    unsigned long long* keys, double* reduced_values,
    unsigned int capacity, int* status) {
  const unsigned int source_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (source_id >= count) return;
  const unsigned long long position_id = position_start + source_id;
  const unsigned int scalar_row = positions[2 * position_id];
  const unsigned int scalar_col = positions[2 * position_id + 1];
  if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
    atomicExch(status, 1);
    return;
  }
  const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
  const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
  if (fine_row == 0xffffffffu || fine_col == 0xffffffffu) {
    atomicExch(status, 1);
    return;
  }
  if (fine_dimensions[fine_row] != block_rows ||
      fine_dimensions[fine_col] != block_cols) {
    atomicExch(status, 3);
    return;
  }
  if (transpose && (!symmetric_storage || fine_row == fine_col)) return;
  unsigned int parent_row = fine_to_level[fine_row];
  unsigned int parent_col = fine_to_level[fine_col];
  if (transpose) {
    const unsigned int temporary = parent_row;
    parent_row = parent_col;
    parent_col = temporary;
  }
  const unsigned long long encoded =
      static_cast<unsigned long long>(parent_row) * parent_node_count + parent_col + 1;
  unsigned int slot = static_cast<unsigned int>(yasps_mas_hash_key(encoded, capacity));
  bool inserted = false;
  for (unsigned int probe = 0; probe < capacity; ++probe) {
    const unsigned long long previous = atomicCAS(keys + slot, 0ull, encoded);
    if (previous == 0ull || previous == encoded) {
      inserted = true;
      break;
    }
    slot = (slot + 1) & (capacity - 1);
  }
  if (!inserted) {
    atomicExch(status, 2);
    return;
  }
  const unsigned int target_rows = transpose ? block_cols : block_rows;
  const unsigned int target_cols = transpose ? block_rows : block_cols;
  const unsigned int area = target_rows * target_cols;
  const double* source = values + value_start +
                         static_cast<unsigned long long>(source_id) * block_rows * block_cols;
  for (unsigned int entry = 0; entry < area; ++entry) {
    const unsigned int target_row = entry / target_cols;
    const unsigned int target_col = entry % target_cols;
    const double value = transpose
        ? source[target_col * block_cols + target_row]
        : source[target_row * block_cols + target_col];
    yasps_mas_atomic_add(reduced_values + static_cast<unsigned long long>(slot) * area + entry,
                         value);
  }
}

extern "C" __global__ void yasps_mas_initialize_padded_domains(
    double* matrices, const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets, const unsigned char* packed_active,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    unsigned int domain_count) {
  const unsigned int domain = blockIdx.x;
  if (domain >= domain_count) return;
  const unsigned int size = sizes[domain];
  const unsigned int padded = padded_sizes[domain];
  double* matrix = matrices + matrix_offsets[domain];
  for (unsigned int entry = threadIdx.x; entry < padded * padded; entry += blockDim.x) {
    const unsigned int row = entry / padded;
    const unsigned int col = entry % padded;
    const bool inactive = row >= size || !packed_active[vector_offsets[domain] + row];
    matrix[entry] = (row == col && inactive) ? 1.0 : 0.0;
  }
}

// Numerical assembly and adjacent propagation retain only the canonical
// upper triangle.  Filling the lower triangle once here is cheaper than a
// second atomic write for every source contribution at every hierarchy level.
extern "C" __global__ void yasps_mas_symmetrize_padded_domains(
    double* matrices, const unsigned long long* matrix_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    unsigned int domain_count) {
  const unsigned int domain = blockIdx.x;
  if (domain >= domain_count) return;
  const unsigned int size = sizes[domain];
  const unsigned int padded = padded_sizes[domain];
  double* matrix = matrices + matrix_offsets[domain];
  const unsigned int area = size * size;
  for (unsigned int entry = threadIdx.x; entry < area;
       entry += blockDim.x) {
    const unsigned int row = entry / size;
    const unsigned int col = entry % size;
    if (row > col) matrix[row * padded + col] = matrix[col * padded + row];
  }
}

// Scatter each occupied reduced block into its domain matrix. Directed hash
// entries already include symmetric transposes, including the collapse-to-
// diagonal B + B^T case.
extern "C" __global__ void yasps_mas_hash_scatter_domains(
    const unsigned long long* keys, const double* reduced_values,
    unsigned int capacity, unsigned int parent_node_count,
    unsigned int block_rows, unsigned int block_cols,
    const unsigned int* node_domains, const unsigned int* node_local_offsets,
    const unsigned long long* matrix_offsets, const unsigned int* padded_sizes,
    double* matrices) {
  const unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= capacity || keys[slot] == 0ull) return;
  const unsigned long long raw = keys[slot] - 1;
  const unsigned int row_node = static_cast<unsigned int>(raw / parent_node_count);
  const unsigned int col_node = static_cast<unsigned int>(raw % parent_node_count);
  const unsigned int domain = node_domains[row_node];
  if (domain != node_domains[col_node]) return;
  const unsigned int local_row = node_local_offsets[row_node];
  const unsigned int local_col = node_local_offsets[col_node];
  const unsigned int padded = padded_sizes[domain];
  double* matrix = matrices + matrix_offsets[domain];
  const double* block = reduced_values +
                        static_cast<unsigned long long>(slot) * block_rows * block_cols;
  for (unsigned int row = 0; row < block_rows; ++row)
    for (unsigned int col = 0; col < block_cols; ++col)
      yasps_mas_atomic_add(matrix + (local_row + row) * padded + local_col + col,
                           block[row * block_cols + col]);
}

extern "C" __global__ void yasps_mas_hash_spmv(
    const unsigned long long* keys, const double* reduced_values,
    unsigned int capacity, unsigned int node_count,
    unsigned int block_rows, unsigned int block_cols,
    const unsigned int* node_scalar_offsets,
    const double* x, double* y) {
  const unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= capacity || keys[slot] == 0ull) return;
  const unsigned long long raw = keys[slot] - 1;
  const unsigned int row_node = static_cast<unsigned int>(raw / node_count);
  const unsigned int col_node = static_cast<unsigned int>(raw % node_count);
  const unsigned int row_start = node_scalar_offsets[row_node];
  const unsigned int col_start = node_scalar_offsets[col_node];
  const double* block = reduced_values +
                        static_cast<unsigned long long>(slot) * block_rows * block_cols;
  for (unsigned int row = 0; row < block_rows; ++row) {
    double value = 0.0;
    for (unsigned int col = 0; col < block_cols; ++col)
      value += block[row * block_cols + col] * x[col_start + col];
    yasps_mas_atomic_add(y + row_start + row, value);
  }
}

// GIPC-style numerical assembly: one CUDA block owns one input Hessian block
// and walks that block through every fixed static merge level.  No dynamically
// sized collapsed sparse matrix is needed; collision contributions land
// directly in the fixed domain-matrix arena.
extern "C" __global__ void yasps_mas_fused_assemble_domains(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, unsigned int block_rows, unsigned int block_cols,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_dimensions, unsigned int fine_scalar_dofs,
    const unsigned int* fine_node_domains,
    const unsigned int* fine_node_local_offsets,
    unsigned int level_count, unsigned int fine_node_count,
    const unsigned long long* matrix_offsets,
    const unsigned int* padded_sizes, double* matrices,
    bool symmetric_storage, int* status) {
  const unsigned int source_id = blockIdx.x;
  if (source_id >= count) return;
  const unsigned long long position_id = position_start + source_id;
  const unsigned int scalar_row = positions[2 * position_id];
  const unsigned int scalar_col = positions[2 * position_id + 1];
  if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
    if (threadIdx.x == 0) atomicExch(status, 1);
    return;
  }
  const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
  const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
  if (fine_row == 0xffffffffu || fine_col == 0xffffffffu) {
    if (threadIdx.x == 0) atomicExch(status, 1);
    return;
  }
  if (fine_dimensions[fine_row] != block_rows ||
      fine_dimensions[fine_col] != block_cols) {
    if (threadIdx.x == 0) atomicExch(status, 3);
    return;
  }
  const double* source = values + value_start +
      static_cast<unsigned long long>(source_id) * block_rows * block_cols;
  const unsigned int area = block_rows * block_cols;
  // Adjacent propagation carries an entry while its endpoints remain in the
  // same domain. Unlike GIPC's strictly nested banks, independently computed
  // METIS levels may split a pair and merge it again later. Re-inject on every
  // false->true transition so that a later coarse bank never loses that block.
  bool present_in_previous_level = false;
  for (unsigned int level = 0; level < level_count; ++level) {
    const unsigned int map_row = level * fine_node_count + fine_row;
    const unsigned int map_col = level * fine_node_count + fine_col;
    const unsigned int domain = fine_node_domains[map_row];
    const bool internal = domain == fine_node_domains[map_col];
    if (!internal || present_in_previous_level) {
      present_in_previous_level = internal;
      continue;
    }
    const unsigned int local_row = fine_node_local_offsets[map_row];
    const unsigned int local_col = fine_node_local_offsets[map_col];
    const unsigned int padded = padded_sizes[domain];
    double* matrix = matrices + matrix_offsets[domain];
    for (unsigned int entry = threadIdx.x; entry < area; entry += blockDim.x) {
      const unsigned int row = entry / block_cols;
      const unsigned int col = entry % block_cols;
      const double value = source[entry];
      unsigned int target_row = local_row + row;
      unsigned int target_col = local_col + col;
      if (symmetric_storage) {
        if (fine_row == fine_col && target_row > target_col) continue;
        if (target_row > target_col) {
          const unsigned int temporary = target_row;
          target_row = target_col;
          target_col = temporary;
        }
      }
      yasps_mas_atomic_add(
          matrix + target_row * padded + target_col, value);
    }
    present_in_previous_level = true;
  }
}

// Propagate already-aggregated fine-domain matrices to every coarser local
// system. One CUDA block owns one fine domain; threads walk dense scalar
// entries in chunks, avoiding a second traversal of all original Hessian
// blocks at every level.
extern "C" __global__ void yasps_mas_propagate_fine_domains(
    const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    const unsigned int* level0_packed_to_fine,
    const unsigned long long* fine_to_packed,
    const unsigned int* packed_scalar_domains,
    const unsigned int* packed_scalar_local_offsets,
    unsigned int fine_domain_count, unsigned int fine_scalar_dofs,
    unsigned int level_count, double* matrices) {
  const unsigned int fine_domain = blockIdx.x;
  if (fine_domain >= fine_domain_count || level_count <= 1) return;
  const unsigned int size = sizes[fine_domain];
  const unsigned int padded = padded_sizes[fine_domain];
  const unsigned long long source_vector = vector_offsets[fine_domain];
  const double* source_matrix = matrices + matrix_offsets[fine_domain];
  const unsigned int area = size * size;
  for (unsigned int entry = threadIdx.x; entry < area;
       entry += blockDim.x) {
    const unsigned int row = entry / size;
    const unsigned int col = entry % size;
    const double value = source_matrix[row * padded + col];
    if (value == 0.0) continue;
    const unsigned int fine_row =
        level0_packed_to_fine[source_vector + row];
    const unsigned int fine_col =
        level0_packed_to_fine[source_vector + col];
    for (unsigned int level = 1; level < level_count; ++level) {
      const unsigned long long map_base =
          static_cast<unsigned long long>(level) * fine_scalar_dofs;
      const unsigned long long packed_row = fine_to_packed[map_base + fine_row];
      const unsigned long long packed_col = fine_to_packed[map_base + fine_col];
      const unsigned int row_domain = packed_scalar_domains[packed_row];
      if (row_domain != packed_scalar_domains[packed_col]) continue;
      unsigned int local_row = packed_scalar_local_offsets[packed_row];
      unsigned int local_col = packed_scalar_local_offsets[packed_col];
      if (local_row > local_col) {
        const unsigned int temporary = local_row;
        local_row = local_col;
        local_col = temporary;
      }
      const unsigned int target_padded = padded_sizes[row_domain];
      const double collapsed_value =
          (local_row == local_col && row != col) ? 2.0 * value : value;
      yasps_mas_atomic_add(
          matrices + matrix_offsets[row_domain] +
              static_cast<unsigned long long>(local_row) * target_padded + local_col,
          collapsed_value);
    }
  }
}

// Assemble both the static and dynamic category streams with one launch. The
// category tables are tiny, device-resident descriptors; the Hessian blocks
// and every hierarchy-level scatter remain entirely on the GPU. This is the
// heterogeneous equivalent of GIPC's single PrepareHessian_bcoo traversal.
extern "C" __global__ void yasps_mas_fused_assemble_all_categories(
    const double* static_values, const unsigned int* static_positions,
    const unsigned int* static_counts,
    const unsigned long long* static_value_starts,
    const unsigned long long* static_position_offsets,
    const unsigned int* static_shapes, unsigned int static_category_count,
    unsigned int static_block_count,
    const double* dynamic_values, const unsigned int* dynamic_positions,
    const unsigned int* dynamic_counts,
    const unsigned long long* dynamic_value_starts,
    const unsigned long long* dynamic_position_offsets,
    const unsigned int* dynamic_shapes, unsigned int dynamic_category_count,
    unsigned int dynamic_block_count,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_dimensions, unsigned int fine_scalar_dofs,
    const unsigned int* fine_node_domains,
    const unsigned int* fine_node_local_offsets,
    unsigned int level_count, unsigned int fine_node_count,
    const unsigned long long* matrix_offsets,
    const unsigned int* padded_sizes, double* matrices,
    bool symmetric_storage, int* status) {
  const unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  // The active dynamic count lives in the device descriptor so a CUDA graph
  // remains reusable while collision counts vary within reserved capacity.
  const unsigned int active_dynamic_count = dynamic_category_count
      ? static_cast<unsigned int>(
            dynamic_position_offsets[dynamic_category_count])
      : 0u;
  const unsigned int total_count = static_block_count + active_dynamic_count;
  if (global_id >= total_count) return;

  const bool is_dynamic = global_id >= static_block_count;
  const unsigned int stream_id = is_dynamic
      ? global_id - static_block_count : global_id;
  const double* values = is_dynamic ? dynamic_values : static_values;
  const unsigned int* positions = is_dynamic
      ? dynamic_positions : static_positions;
  const unsigned int* counts = is_dynamic ? dynamic_counts : static_counts;
  const unsigned long long* value_starts = is_dynamic
      ? dynamic_value_starts : static_value_starts;
  const unsigned long long* position_offsets = is_dynamic
      ? dynamic_position_offsets : static_position_offsets;
  const unsigned int* shapes = is_dynamic ? dynamic_shapes : static_shapes;
  const unsigned int category_count = is_dynamic
      ? dynamic_category_count : static_category_count;
  (void)dynamic_block_count;

  unsigned int category = 0;
  while (category + 1 < category_count &&
         stream_id >= position_offsets[category + 1])
    ++category;
  if (category >= category_count ||
      stream_id >= position_offsets[category] + counts[category]) {
    atomicExch(status, 4);
    return;
  }
  const unsigned int source_id = static_cast<unsigned int>(
      stream_id - position_offsets[category]);
  const unsigned int block_rows = shapes[2 * category];
  const unsigned int block_cols = shapes[2 * category + 1];
  const unsigned long long position_id =
      position_offsets[category] + source_id;
  const unsigned int scalar_row = positions[2 * position_id];
  const unsigned int scalar_col = positions[2 * position_id + 1];
  if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
    atomicExch(status, 1);
    return;
  }
  const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
  const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
  if (fine_row == 0xffffffffu || fine_col == 0xffffffffu) {
    atomicExch(status, 1);
    return;
  }
  if (fine_dimensions[fine_row] != block_rows ||
      fine_dimensions[fine_col] != block_cols) {
    atomicExch(status, 3);
    return;
  }
  const double* source = values + value_starts[category] +
      static_cast<unsigned long long>(source_id) * block_rows * block_cols;
  const unsigned int area = block_rows * block_cols;
  bool present_in_previous_level = false;
  for (unsigned int level = 0; level < level_count; ++level) {
    const unsigned int map_row = level * fine_node_count + fine_row;
    const unsigned int map_col = level * fine_node_count + fine_col;
    const unsigned int domain = fine_node_domains[map_row];
    const bool internal = domain == fine_node_domains[map_col];
    if (!internal || present_in_previous_level) {
      present_in_previous_level = internal;
      continue;
    }
    const unsigned int local_row = fine_node_local_offsets[map_row];
    const unsigned int local_col = fine_node_local_offsets[map_col];
    const unsigned int padded = padded_sizes[domain];
    double* matrix = matrices + matrix_offsets[domain];
    for (unsigned int entry = 0; entry < area; ++entry) {
      const unsigned int row = entry / block_cols;
      const unsigned int col = entry % block_cols;
      const double value = source[entry];
      yasps_mas_atomic_add(
          matrix + (local_row + row) * padded + local_col + col, value);
      if (symmetric_storage && fine_row != fine_col)
        yasps_mas_atomic_add(
            matrix + (local_col + col) * padded + local_row + row, value);
    }
    present_in_previous_level = true;
  }
}

// Propagate one complete hierarchy level to the next. Launches for adjacent
// levels are ordered on one stream (and can be captured as one graph), so a
// coarse bank consumes the fully aggregated result of its children rather than
// rereading every fine bank for every level.
extern "C" __global__ void yasps_mas_propagate_adjacent_domains(
    const unsigned long long* matrix_offsets,
    const unsigned long long* vector_offsets,
    const unsigned int* sizes, const unsigned int* padded_sizes,
    const unsigned long long* packed_to_next_packed,
    const unsigned char* packed_active,
    const unsigned int* packed_scalar_domains,
    const unsigned int* packed_scalar_local_offsets,
    unsigned int source_domain_begin, unsigned int source_domain_count,
    double* matrices) {
  const unsigned int local_domain = blockIdx.x;
  if (local_domain >= source_domain_count) return;
  const unsigned int domain = source_domain_begin + local_domain;
  const unsigned int size = sizes[domain];
  const unsigned int padded = padded_sizes[domain];
  const unsigned long long source_vector = vector_offsets[domain];
  const double* source_matrix = matrices + matrix_offsets[domain];
  const unsigned int area = size * size;
  for (unsigned int entry = threadIdx.x; entry < area;
       entry += blockDim.x) {
    const unsigned int row = entry / size;
    const unsigned int col = entry % size;
    if (row > col) continue;
    if (!packed_active[source_vector + row] ||
        !packed_active[source_vector + col]) continue;
    const double value = source_matrix[row * padded + col];
    if (value == 0.0) continue;
    const unsigned long long packed_row =
        packed_to_next_packed[source_vector + row];
    const unsigned long long packed_col =
        packed_to_next_packed[source_vector + col];
    if (packed_row == 0xffffffffffffffffull ||
        packed_col == 0xffffffffffffffffull) continue;
    const unsigned int row_domain = packed_scalar_domains[packed_row];
    if (row_domain != packed_scalar_domains[packed_col]) continue;
    unsigned int local_row = packed_scalar_local_offsets[packed_row];
    unsigned int local_col = packed_scalar_local_offsets[packed_col];
    if (local_row > local_col) {
      const unsigned int temporary = local_row;
      local_row = local_col;
      local_col = temporary;
    }
    const unsigned int target_padded = padded_sizes[row_domain];
    const double collapsed_value =
        (local_row == local_col && row != col) ? 2.0 * value : value;
    yasps_mas_atomic_add(
        matrices + matrix_offsets[row_domain]
            + static_cast<unsigned long long>(local_row) * target_padded
            + local_col,
        collapsed_value);
  }
}

// Build one overlapping two-node Schwarz matrix for every dynamic off-diagonal
// block. This makes collision couplings that cross every static partition
// visible to the preconditioner without rebuilding METIS. The diagonal pieces
// are read from the already assembled fine domains, so they include current
// static and dynamic diagonal values. One CUDA block owns one edge matrix.
extern "C" __global__ void yasps_mas_assemble_dynamic_edge_domains(
    const double* dynamic_values, const unsigned int* dynamic_positions,
    const unsigned int* counts, const unsigned long long* value_starts,
    const unsigned long long* position_offsets, const unsigned int* shapes,
    unsigned int category_count, unsigned int block_count,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_dimensions, unsigned int fine_scalar_dofs,
    const unsigned int* fine_node_domains,
    const unsigned int* fine_node_local_offsets,
    const unsigned long long* matrix_offsets,
    const unsigned int* padded_sizes, const double* fine_matrices,
    unsigned int edge_padded_size, double* edge_matrices,
    unsigned int* edge_node_counts, int* status) {
  const unsigned int edge = blockIdx.x;
  if (edge >= block_count) return;
  unsigned int category = 0;
  while (category + 1 < category_count &&
         edge >= position_offsets[category + 1])
    ++category;
  if (category >= category_count ||
      edge >= position_offsets[category] + counts[category]) return;
  const unsigned int source_id = static_cast<unsigned int>(
      edge - position_offsets[category]);
  const unsigned int rows = shapes[2 * category];
  const unsigned int cols = shapes[2 * category + 1];
  const unsigned int scalar_row = dynamic_positions[2 * edge];
  const unsigned int scalar_col = dynamic_positions[2 * edge + 1];
  if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
    if (threadIdx.x == 0) atomicExch(status, 1);
    return;
  }
  const unsigned int row_node = scalar_boundary_to_node[scalar_row];
  const unsigned int col_node = scalar_boundary_to_node[scalar_col];
  if (row_node == 0xffffffffu || col_node == 0xffffffffu ||
      fine_dimensions[row_node] != rows || fine_dimensions[col_node] != cols) {
    if (threadIdx.x == 0) atomicExch(status, 3);
    return;
  }
  double* destination = edge_matrices
      + static_cast<unsigned long long>(edge) * edge_padded_size * edge_padded_size;
  const bool active = row_node != col_node;
  if (active && threadIdx.x == 0) {
    atomicAdd(edge_node_counts + row_node, 1u);
    atomicAdd(edge_node_counts + col_node, 1u);
  }
  const unsigned int n = rows + cols;
  const unsigned int row_domain = fine_node_domains[row_node];
  const unsigned int col_domain = fine_node_domains[col_node];
  const unsigned int row_local = fine_node_local_offsets[row_node];
  const unsigned int col_local = fine_node_local_offsets[col_node];
  const unsigned int row_padded = padded_sizes[row_domain];
  const unsigned int col_padded = padded_sizes[col_domain];
  const double* row_matrix = fine_matrices + matrix_offsets[row_domain];
  const double* col_matrix = fine_matrices + matrix_offsets[col_domain];
  const double* dynamic = dynamic_values + value_starts[category]
      + static_cast<unsigned long long>(source_id) * rows * cols;
  const bool same_domain = row_domain == col_domain;
  const unsigned int area = edge_padded_size * edge_padded_size;
  for (unsigned int entry = threadIdx.x; entry < area;
       entry += blockDim.x) {
    const unsigned int local_row = entry / edge_padded_size;
    const unsigned int local_col = entry % edge_padded_size;
    double value = local_row == local_col ? 1.0 : 0.0;
    if (active && local_row < n && local_col < n) {
      if (local_row < rows && local_col < rows) {
        value = row_matrix[(row_local + local_row) * row_padded
                           + row_local + local_col];
      } else if (local_row >= rows && local_col >= rows) {
        value = col_matrix[(col_local + local_row - rows) * col_padded
                           + col_local + local_col - rows];
      } else if (local_row < rows) {
        value = same_domain
            ? row_matrix[(row_local + local_row) * row_padded
                         + col_local + local_col - rows]
            : dynamic[local_row * cols + local_col - rows];
      } else {
        value = same_domain
            ? row_matrix[(col_local + local_row - rows) * row_padded
                         + row_local + local_col]
            : dynamic[local_col * cols + local_row - rows];
      }
    }
    destination[entry] = value;
  }
}

// Build one collision patch from a short, contiguous chunk of same-shape
// dynamic blocks. Repeated endpoints share one coordinate, so contact stars
// and chains are inverted together instead of as unrelated two-node edges.
// The chunk size is conservatively chosen from (rows + cols), guaranteeing
// that even a chunk with no repeated endpoint fits the configured bank.
extern "C" __global__ void yasps_mas_assemble_dynamic_group_domains(
    const double* dynamic_values, const unsigned int* dynamic_positions,
    const unsigned int* counts, const unsigned long long* value_starts,
    const unsigned long long* position_offsets, const unsigned int* shapes,
    const unsigned long long* group_offsets, const unsigned int* chunk_sizes,
    unsigned int category_count, unsigned int group_capacity,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_dimensions, unsigned int fine_scalar_dofs,
    const unsigned int* fine_node_scalar_offsets,
    const unsigned int* fine_node_domains,
    const unsigned int* fine_node_local_offsets,
    const unsigned long long* matrix_offsets,
    const unsigned int* padded_sizes, const double* fine_matrices,
    unsigned int group_stride, double* group_matrices,
    unsigned int* group_active_sizes, unsigned int* group_scalar_indices,
    unsigned int* group_scalar_nodes, unsigned int* group_node_counts,
    int* status) {
  const unsigned int group = blockIdx.x;
  const unsigned int active_groups = category_count
      ? static_cast<unsigned int>(group_offsets[category_count]) : 0u;
  if (group >= group_capacity) return;
  double* destination = group_matrices
      + static_cast<unsigned long long>(group) * group_stride * group_stride;
  if (group >= active_groups) {
    for (unsigned int entry = threadIdx.x;
         entry < group_stride * group_stride; entry += blockDim.x) {
      const unsigned int row = entry / group_stride;
      const unsigned int col = entry - row * group_stride;
      destination[entry] = row == col ? 1.0 : 0.0;
    }
    if (threadIdx.x == 0) group_active_sizes[group] = 0;
    return;
  }

  unsigned int category = 0;
  while (category + 1 < category_count &&
         group >= group_offsets[category + 1])
    ++category;
  const unsigned int local_group = static_cast<unsigned int>(
      group - group_offsets[category]);
  const unsigned int chunk = chunk_sizes[category];
  const unsigned int category_count_blocks = counts[category];
  const unsigned int local_begin = local_group * chunk;
  const unsigned int edge_count = min(
      chunk, category_count_blocks - local_begin);
  const unsigned int block_rows = shapes[2 * category];
  const unsigned int block_cols = shapes[2 * category + 1];
  const unsigned long long position_begin =
      position_offsets[category] + local_begin;

  __shared__ unsigned int nodes[64];
  __shared__ unsigned int node_scalar_offsets[64];
  __shared__ unsigned int node_count;
  __shared__ unsigned int active_size;
  if (threadIdx.x == 0) {
    node_count = 0;
    active_size = 0;
    for (unsigned int edge = 0; edge < edge_count; ++edge) {
      const unsigned long long position = position_begin + edge;
      const unsigned int scalar_row = dynamic_positions[2 * position];
      const unsigned int scalar_col = dynamic_positions[2 * position + 1];
      if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
        atomicExch(status, 1);
        continue;
      }
      const unsigned int row_node = scalar_boundary_to_node[scalar_row];
      const unsigned int col_node = scalar_boundary_to_node[scalar_col];
      if (row_node == 0xffffffffu || col_node == 0xffffffffu ||
          fine_dimensions[row_node] != block_rows ||
          fine_dimensions[col_node] != block_cols) {
        atomicExch(status, 3);
        continue;
      }
      // Diagonal collision blocks are already present in the fine matrices.
      // They do not define an edge or add a new patch endpoint.
      if (row_node == col_node) continue;
      const unsigned int endpoints[2] = {row_node, col_node};
      for (unsigned int endpoint = 0; endpoint < 2; ++endpoint) {
        const unsigned int node = endpoints[endpoint];
        bool found = false;
        for (unsigned int index = 0; index < node_count; ++index)
          found = found || nodes[index] == node;
        if (!found) {
          const unsigned int dimension = fine_dimensions[node];
          if (node_count >= 64 || active_size + dimension > group_stride) {
            atomicExch(status, 6);
            continue;
          }
          nodes[node_count] = node;
          node_scalar_offsets[node_count] = active_size;
          active_size += dimension;
          ++node_count;
        }
      }
    }
    group_active_sizes[group] = active_size;
  }
  __syncthreads();

  for (unsigned int entry = threadIdx.x;
       entry < group_stride * group_stride; entry += blockDim.x) {
    const unsigned int row = entry / group_stride;
    const unsigned int col = entry - row * group_stride;
    destination[entry] = (row == col && row >= active_size) ? 1.0 : 0.0;
  }
  for (unsigned int node_index = threadIdx.x;
       node_index < node_count; node_index += blockDim.x)
    atomicAdd(group_node_counts + nodes[node_index], 1u);
  __syncthreads();

  for (unsigned int node_index = 0; node_index < node_count; ++node_index) {
    const unsigned int node = nodes[node_index];
    const unsigned int dimension = fine_dimensions[node];
    const unsigned int target = node_scalar_offsets[node_index];
    const unsigned int domain = fine_node_domains[node];
    const unsigned int local = fine_node_local_offsets[node];
    const unsigned int leading = padded_sizes[domain];
    const double* source = fine_matrices + matrix_offsets[domain];
    for (unsigned int entry = threadIdx.x; entry < dimension * dimension;
         entry += blockDim.x) {
      const unsigned int row = entry / dimension;
      const unsigned int col = entry - row * dimension;
      destination[(target + row) * group_stride + target + col] =
          source[(local + row) * leading + local + col];
    }
    for (unsigned int component = threadIdx.x; component < dimension;
         component += blockDim.x) {
      group_scalar_indices[group * group_stride + target + component] =
          fine_node_scalar_offsets[node] + component;
      group_scalar_nodes[group * group_stride + target + component] = node;
    }
  }
  __syncthreads();

  // Materialize every already-assembled coupling between patch nodes that
  // share a fine Schwarz domain. This includes static edges and dynamic edges
  // already present there. Copying only diagonal blocks can make an arbitrary
  // collision chunk indefinite even when the global Hessian is SPD.
  for (unsigned int left = 0; left < node_count; ++left) {
    const unsigned int left_node = nodes[left];
    const unsigned int left_domain = fine_node_domains[left_node];
    const unsigned int left_dimension = fine_dimensions[left_node];
    const unsigned int left_local = fine_node_local_offsets[left_node];
    for (unsigned int right = left + 1; right < node_count; ++right) {
      const unsigned int right_node = nodes[right];
      if (fine_node_domains[right_node] != left_domain) continue;
      const unsigned int right_dimension = fine_dimensions[right_node];
      const unsigned int right_local = fine_node_local_offsets[right_node];
      const unsigned int leading = padded_sizes[left_domain];
      const double* source = fine_matrices + matrix_offsets[left_domain];
      const unsigned int left_target = node_scalar_offsets[left];
      const unsigned int right_target = node_scalar_offsets[right];
      const unsigned int pair_area = left_dimension * right_dimension;
      for (unsigned int entry = threadIdx.x; entry < pair_area;
           entry += blockDim.x) {
        const unsigned int row = entry / right_dimension;
        const unsigned int col = entry - row * right_dimension;
        destination[(left_target + row) * group_stride + right_target + col] =
            source[(left_local + row) * leading + right_local + col];
        destination[(right_target + col) * group_stride + left_target + row] =
            source[(right_local + col) * leading + left_local + row];
      }
    }
  }
  __syncthreads();

}

// Complete each collision patch with every dynamic Hessian block whose two
// endpoints are members. Dynamic categories are regrouped by block shape, so
// merely copying a contiguous input chunk can split one contact clique and
// destroy positive definiteness. This device-only completion restores the
// true principal collision submatrix without any Python/CPU block traversal.
extern "C" __global__ void yasps_mas_complete_dynamic_group_domains(
    const double* dynamic_values, const unsigned int* dynamic_positions,
    const unsigned int* counts, const unsigned long long* value_starts,
    const unsigned long long* position_offsets, const unsigned int* shapes,
    unsigned int category_count, unsigned int group_count,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_node_domains,
    const unsigned int* group_active_sizes,
    const unsigned int* group_scalar_nodes,
    unsigned int group_stride, double* group_matrices) {
  const unsigned int group = blockIdx.x;
  if (group >= group_count) return;
  const unsigned int n = group_active_sizes[group];
  if (!n) return;
  const unsigned int active_blocks = category_count
      ? static_cast<unsigned int>(position_offsets[category_count]) : 0u;
  double* destination = group_matrices
      + static_cast<unsigned long long>(group) * group_stride * group_stride;
  const unsigned int* scalar_nodes = group_scalar_nodes + group * group_stride;
  for (unsigned int block = threadIdx.x; block < active_blocks;
       block += blockDim.x) {
    unsigned int category = 0;
    while (category + 1 < category_count &&
           block >= position_offsets[category + 1])
      ++category;
    const unsigned int scalar_row = dynamic_positions[2ull * block];
    const unsigned int scalar_col = dynamic_positions[2ull * block + 1];
    const unsigned int row_node = scalar_boundary_to_node[scalar_row];
    const unsigned int col_node = scalar_boundary_to_node[scalar_col];
    if (row_node == col_node ||
        fine_node_domains[row_node] == fine_node_domains[col_node])
      continue;
    unsigned int local_row = 0xffffffffu;
    unsigned int local_col = 0xffffffffu;
    for (unsigned int scalar = 0; scalar < n; ++scalar) {
      if (local_row == 0xffffffffu && scalar_nodes[scalar] == row_node)
        local_row = scalar;
      if (local_col == 0xffffffffu && scalar_nodes[scalar] == col_node)
        local_col = scalar;
    }
    if (local_row == 0xffffffffu || local_col == 0xffffffffu) continue;
    const unsigned int rows = shapes[2 * category];
    const unsigned int cols = shapes[2 * category + 1];
    const unsigned int area = rows * cols;
    const unsigned int source_id = static_cast<unsigned int>(
        block - position_offsets[category]);
    const double* source = dynamic_values + value_starts[category]
        + static_cast<unsigned long long>(source_id) * area;
    for (unsigned int entry = 0; entry < area; ++entry) {
      const unsigned int row = entry / cols;
      const unsigned int col = entry - row * cols;
      const double value = source[entry];
      yasps_mas_atomic_add(destination
          + (local_row + row) * group_stride + local_col + col, value);
      yasps_mas_atomic_add(destination
          + (local_col + col) * group_stride + local_row + row, value);
    }
  }
}

extern "C" __global__ void yasps_mas_regularize_dynamic_group_domains(
    double* matrices, const unsigned int* active_sizes,
    unsigned int group_stride, unsigned int group_count) {
  const unsigned int group = blockIdx.x;
  if (group >= group_count) return;
  const unsigned int n = active_sizes[group];
  double* matrix = matrices
      + static_cast<unsigned long long>(group) * group_stride * group_stride;
  __shared__ double lower_bounds[64];
  __shared__ double diagonal_scales[64];
  __shared__ double diagonal_shift;
  if (threadIdx.x < n) {
    const unsigned int row = threadIdx.x;
    double radius = 0.0;
    for (unsigned int col = 0; col < n; ++col)
      if (col != row) radius += fabs(matrix[row * group_stride + col]);
    const double diagonal = matrix[row * group_stride + row];
    lower_bounds[row] = diagonal - radius;
    diagonal_scales[row] = fabs(diagonal);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double minimum = n ? lower_bounds[0] : 1.0;
    double scale = n ? diagonal_scales[0] : 1.0;
    for (unsigned int row = 1; row < n; ++row) {
      minimum = fmin(minimum, lower_bounds[row]);
      scale = fmax(scale, diagonal_scales[row]);
    }
    diagonal_shift = minimum > 0.0
        ? 0.0 : (-minimum + fmax(1.0, scale) * 1.0e-6);
  }
  __syncthreads();
  for (unsigned int row = threadIdx.x; row < n; row += blockDim.x)
    matrix[row * group_stride + row] += diagonal_shift;
}

// Apply an input block category at any hierarchy level without materializing
// a collapsed sparse matrix. The fine-node-to-level-scalar-start map is static.
extern "C" __global__ void yasps_mas_mapped_block_spmv(
    const double* values, unsigned long long value_start,
    const unsigned int* positions, unsigned long long position_start,
    unsigned int count, unsigned int block_rows, unsigned int block_cols,
    const unsigned int* scalar_boundary_to_node,
    const unsigned int* fine_dimensions, unsigned int fine_scalar_dofs,
    const unsigned int* fine_node_scalar_starts,
    unsigned int level_index, unsigned int fine_node_count,
    const double* x, double* y, bool symmetric_storage, int* status) {
  const unsigned int source_id = blockIdx.x;
  if (source_id >= count) return;
  const unsigned long long position_id = position_start + source_id;
  const unsigned int scalar_row = positions[2 * position_id];
  const unsigned int scalar_col = positions[2 * position_id + 1];
  if (scalar_row >= fine_scalar_dofs || scalar_col >= fine_scalar_dofs) {
    if (threadIdx.x == 0) atomicExch(status, 1);
    return;
  }
  const unsigned int fine_row = scalar_boundary_to_node[scalar_row];
  const unsigned int fine_col = scalar_boundary_to_node[scalar_col];
  if (fine_row == 0xffffffffu || fine_col == 0xffffffffu) {
    if (threadIdx.x == 0) atomicExch(status, 1);
    return;
  }
  if (fine_dimensions[fine_row] != block_rows ||
      fine_dimensions[fine_col] != block_cols) {
    if (threadIdx.x == 0) atomicExch(status, 3);
    return;
  }
  const unsigned int map_offset = level_index * fine_node_count;
  const unsigned int row_start = fine_node_scalar_starts[map_offset + fine_row];
  const unsigned int col_start = fine_node_scalar_starts[map_offset + fine_col];
  const double* block = values + value_start +
      static_cast<unsigned long long>(source_id) * block_rows * block_cols;
  for (unsigned int row = threadIdx.x; row < block_rows; row += blockDim.x) {
    double value = 0.0;
    for (unsigned int col = 0; col < block_cols; ++col)
      value += block[row * block_cols + col] * x[col_start + col];
    yasps_mas_atomic_add(y + row_start + row, value);
  }
  if (symmetric_storage && fine_row != fine_col) {
    for (unsigned int col = threadIdx.x; col < block_cols; col += blockDim.x) {
      double value = 0.0;
      for (unsigned int row = 0; row < block_rows; ++row)
        value += block[row * block_cols + col] * x[row_start + row];
      yasps_mas_atomic_add(y + col_start + col, value);
    }
  }
}

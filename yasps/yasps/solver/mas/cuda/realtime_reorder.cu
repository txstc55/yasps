#include <cstdint>

// Realtime collision-aware collapse inside the immutable METIS banks.  METIS
// supplies only an ordering/bank layout; collision edges update connected
// components and scalar aliases on the device for every numerical solve.

// Collision topology is transient.  GIPC reconstructs ``_goingNext`` from
// its static maps on every preconditioner rebuild; the immutable-map MAS path
// must likewise discard representatives left by the preceding solve.  This
// kernel is deliberately separate from the mask clear because representatives
// are uint32 while masks are uint64 and both buffers are already contiguous.
extern "C" __global__ void yasps_mas_reset_runtime_representatives(
    unsigned int* representatives, unsigned int node_count) {
  const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node < node_count) representatives[node] = node;
}

extern "C" __global__ void yasps_mas_build_collision_masks(
    const unsigned int* positions, const unsigned int* counts,
    const unsigned long long* position_offsets, unsigned int category_count,
    const unsigned int* scalar_boundary_to_node, unsigned int fine_scalar_dofs,
    const unsigned int* fine_to_level_node,
    const unsigned long long* level_node_bases,
    const unsigned int* representatives, const unsigned int* node_to_next,
    const unsigned int* node_domains, const unsigned int* node_domain_ordinals,
    const unsigned int* node_dimensions, const long long* node_type_ids,
    bool merge_across_types,
    unsigned int source_level, unsigned int fine_node_count,
    unsigned long long* connection_masks, int* status) {
  const unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int active_count = category_count
      ? static_cast<unsigned int>(position_offsets[category_count]) : 0u;
  if (edge >= active_count) return;

  unsigned int category = 0;
  while (category + 1 < category_count &&
         edge >= position_offsets[category + 1])
    ++category;
  if (category >= category_count ||
      edge >= position_offsets[category] + counts[category]) {
    atomicExch(status, 4);
    return;
  }
  const unsigned int scalar_row = positions[2ull * edge];
  const unsigned int scalar_col = positions[2ull * edge + 1];
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

  const unsigned int row_local =
      fine_to_level_node[source_level * fine_node_count + fine_row];
  const unsigned int col_local =
      fine_to_level_node[source_level * fine_node_count + fine_col];
  const unsigned long long base = level_node_bases[source_level];
  const unsigned int row = representatives[base + row_local];
  const unsigned int col = representatives[base + col_local];
  if (row == col) return;

  // Connections in a level's bank construct the *next* level supernodes;
  // the current Schwarz matrix retains all of its coordinates.  This is the
  // same distinction made by GIPC's BuildConnectMaskLx/ComputeNextLevel.
  const unsigned int target_row = node_to_next[row];
  const unsigned int target_col = node_to_next[col];
  if (target_row == 0xffffffffu || target_col == 0xffffffffu ||
      target_row == target_col ||
      node_domains[target_row] != node_domains[target_col] ||
      node_dimensions[target_row] != node_dimensions[target_col] ||
      (!merge_across_types
       && node_type_ids[target_row] != node_type_ids[target_col]))
    return;
  const unsigned int row_ordinal = node_domain_ordinals[target_row];
  const unsigned int col_ordinal = node_domain_ordinals[target_col];
  if (row_ordinal >= 64u || col_ordinal >= 64u) {
    atomicExch(status, 5);
    return;
  }
  atomicOr(connection_masks + target_row, 1ull << col_ordinal);
  atomicOr(connection_masks + target_col, 1ull << row_ordinal);
}

// One block owns one variable-size bank. The bit-mask representation permits
// at most 64 nodes. A bounded
// shared-memory transitive closure is faster and deterministic for these tiny
// graphs than a device-wide union/find convergence loop.
extern "C" __global__ void yasps_mas_close_collision_components(
    const unsigned long long* domain_node_offsets,
    const unsigned int* domain_nodes, unsigned int domain_begin,
    unsigned int domain_count,
    const unsigned long long* connection_masks,
    unsigned int* representatives, int* status) {
  const unsigned int local_domain = blockIdx.x;
  if (local_domain >= domain_count) return;
  const unsigned int domain = domain_begin + local_domain;
  const unsigned long long begin = domain_node_offsets[domain];
  const unsigned int count = static_cast<unsigned int>(
      domain_node_offsets[domain + 1] - begin);
  if (count > 64u) {
    if (threadIdx.x == 0) atomicExch(status, 5);
    return;
  }
  __shared__ unsigned long long masks[64];
  __shared__ unsigned long long expanded[64];
  __shared__ unsigned int nodes[64];
  const unsigned int lane = threadIdx.x;
  if (lane < count) {
    const unsigned int node = domain_nodes[begin + lane];
    nodes[lane] = node;
    masks[lane] = connection_masks[node] | (1ull << lane);
  }
  __syncthreads();
  for (unsigned int iteration = 0; iteration < count; ++iteration) {
    if (lane < count) {
      unsigned long long value = masks[lane];
      unsigned long long todo = value;
      while (todo) {
        const unsigned int neighbor = static_cast<unsigned int>(__ffsll(todo) - 1);
        value |= masks[neighbor];
        todo &= todo - 1;
      }
      expanded[lane] = value;
    }
    __syncthreads();
    if (lane < count) masks[lane] = expanded[lane];
    __syncthreads();
  }
  if (lane < count) {
    const unsigned int root = static_cast<unsigned int>(__ffsll(masks[lane]) - 1);
    representatives[nodes[lane]] = nodes[root];
  }
}

// A realtime merge changes the source-to-parent map for every later level.
// Carry each merged component into the next immutable METIS bank before that
// bank is closed.  GIPC gets this property from rebuilding _goingNext; the
// explicit propagation is the equivalent operation for our reusable maps.
extern "C" __global__ void yasps_mas_propagate_runtime_components(
    const unsigned int* representatives,
    const unsigned int* node_to_next,
    const unsigned int* node_domains,
    const unsigned int* node_domain_ordinals,
    const unsigned int* node_dimensions,
    const long long* node_type_ids,
    bool merge_across_types,
    unsigned int source_begin, unsigned int source_count,
    unsigned long long* connection_masks, int* status) {
  const unsigned int local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= source_count) return;
  const unsigned int node = source_begin + local;
  const unsigned int root = representatives[node];
  if (root == node) return;
  const unsigned int target = node_to_next[node];
  const unsigned int target_root = node_to_next[root];
  if (target == 0xffffffffu || target_root == 0xffffffffu ||
      target == target_root ||
      node_domains[target] != node_domains[target_root] ||
      node_dimensions[target] != node_dimensions[target_root] ||
      (!merge_across_types &&
       node_type_ids[target] != node_type_ids[target_root]))
    return;
  const unsigned int target_ordinal = node_domain_ordinals[target];
  const unsigned int root_ordinal = node_domain_ordinals[target_root];
  if (target_ordinal >= 64u || root_ordinal >= 64u) {
    atomicExch(status, 5);
    return;
  }
  atomicOr(connection_masks + target, 1ull << root_ordinal);
  atomicOr(connection_masks + target_root, 1ull << target_ordinal);
}

extern "C" __global__ void yasps_mas_build_runtime_scalar_maps(
    const unsigned int* fine_to_level_node,
    const unsigned long long* level_node_bases,
    const unsigned int* representatives,
    const unsigned int* node_local_offsets,
    const unsigned long long* packed_node_starts,
    unsigned int level_count, unsigned int fine_node_count,
    unsigned int* fine_node_local_offsets,
    unsigned long long* fine_node_to_packed_starts,
    unsigned long long* fine_node_level_keys) {
  const unsigned int item = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int count = level_count * fine_node_count;
  if (item >= count) return;
  const unsigned int level = item / fine_node_count;
  const unsigned int local_node = fine_to_level_node[item];
  const unsigned int node = static_cast<unsigned int>(
      level_node_bases[level] + local_node);
  const unsigned int root = representatives[node];
  fine_node_local_offsets[item] = node_local_offsets[root];
  const unsigned long long packed_start = packed_node_starts[root];
  fine_node_to_packed_starts[item] = packed_start;
  fine_node_level_keys[item] =
      (static_cast<unsigned long long>(root) << 32) | packed_start;
}

extern "C" __global__ void yasps_mas_build_runtime_transfer_maps(
    const unsigned int* representatives,
    const unsigned int* node_to_next,
    const unsigned long long* packed_node_starts,
    const unsigned int* node_dimensions,
    unsigned int packed_node_count,
    unsigned long long invalid,
    unsigned long long* packed_to_next,
    unsigned char* packed_active) {
  const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= packed_node_count) return;
  const unsigned long long source = packed_node_starts[node];
  const unsigned int dimension = node_dimensions[node];
  const bool active = representatives[node] == node;
  for (unsigned int component = 0; component < dimension; ++component) {
    packed_active[source + component] = active ? 1u : 0u;
    packed_to_next[source + component] = invalid;
  }
  if (!active) return;
  const unsigned int target = node_to_next[node];
  if (target == 0xffffffffu) return;
  const unsigned int target_root = representatives[target];
  const unsigned long long target_start = packed_node_starts[target_root];
  for (unsigned int component = 0; component < dimension; ++component)
    packed_to_next[source + component] = target_start + component;
}

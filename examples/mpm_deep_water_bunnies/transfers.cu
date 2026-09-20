// Transfers, topology metadata, and safeguards. YASPS differentiates energies.
#include <cmath>
#include <assert.h>

#ifndef GRID_N
#define GRID_N 48
#endif
#ifndef GRID_DX
#define GRID_DX (0.5 / (GRID_N - 2.0))
#endif
#ifndef GRID_ORIGIN_X
#define GRID_ORIGIN_X (-0.25 - 0.5 * GRID_DX)
#define GRID_ORIGIN_Y (-0.5 * GRID_DX)
#define GRID_ORIGIN_Z (-0.25 - 0.5 * GRID_DX)
#endif
constexpr int GRID_COUNT = GRID_N * GRID_N * GRID_N;

// Each compressed wall block belongs to the material's complete 27-node
// stencil graph. Reuse that storage instead of adding duplicate MAS blocks.
// Both lists contain unique, lexicographically sorted scalar or 3x3 blocks.
extern "C" __global__ void add_particle_wall_blocks(const unsigned int* wall_coordinates, const double* wall_values, unsigned int wall_count, const unsigned int* material_coordinates, double* material_values, unsigned int material_count, unsigned int block_size) {
  unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= wall_count) return;
  unsigned int row = wall_coordinates[2 * p], col = wall_coordinates[2 * p + 1];
  unsigned int first = 0, last = material_count;
  while (first < last) {
    unsigned int middle = first + (last - first) / 2;
    unsigned int r = material_coordinates[2 * middle], c = material_coordinates[2 * middle + 1];
    if (r < row || (r == row && c < col)) first = middle + 1;
    else last = middle;
  }
  assert(first < material_count && material_coordinates[2 * first] == row && material_coordinates[2 * first + 1] == col);
  // Unique source coordinates imply unique destinations: no atomics needed.
  for (unsigned int k = 0; k < block_size; ++k) material_values[block_size * first + k] += wall_values[block_size * p + k];
}

// Compact per-wall active particle IDs and their weights on the GPU. IDs are
// relative to the currently bound batch, matching the one-to-one YASPS JOIN.
extern "C" __global__ void select_particle_walls(
    const double* positions, const double* particle_weights,
    unsigned int* ids, double* weights, unsigned int* counts,
    unsigned int count, unsigned int capacity, double distance_squared,
    double xmin, double xmax, double ymin, double zmin, double zmax) {
  const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  const double x = positions[3 * p], y = positions[3 * p + 1], z = positions[3 * p + 2];
  const double gaps[5] = {x - xmin, xmax - x, y - ymin, z - zmin, zmax - z};
  for (unsigned int wall = 0; wall < 5; ++wall) {
    if (gaps[wall] * gaps[wall] < distance_squared) {
      const unsigned int slot = atomicAdd(counts + wall, 1u);
      ids[wall * capacity + slot] = p;
      weights[wall * capacity + slot] = particle_weights[p];
    }
  }
}

__device__ double grid_origin(int axis) {
  return axis == 0 ? GRID_ORIGIN_X : (axis == 1 ? GRID_ORIGIN_Y : GRID_ORIGIN_Z);
}

__device__ double grid_coordinate(unsigned int index, int axis) {
  const int node = axis == 0 ? index / (GRID_N * GRID_N) : (axis == 1 ? (index / GRID_N) % GRID_N : index % GRID_N);
  return grid_origin(axis) + node * GRID_DX;
}

extern "C" __global__ void p2g(
    const double* x, const double* v, const double* F, const double* C,
    const double* mass, unsigned int* indices, double* weights, double* B,
    double* grid_mass, double* momentum, int* invalid, int count, int liquid) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  int base[3];
  double w[3][3], dw[3][3];
  for (int a = 0; a < 3; ++a) {
    const double xi = (x[3*p+a] - grid_origin(a)) / GRID_DX;
    base[a] = (int)floor(xi - 0.5);
    if (base[a] < 0 || base[a] + 2 >= GRID_N) { atomicExch(invalid, 1); return; }
    const double f = xi - base[a];
    w[a][0] = 0.5 * (1.5-f) * (1.5-f);
    w[a][1] = 0.75 - (f-1.0) * (f-1.0);
    w[a][2] = 0.5 * (f-0.5) * (f-0.5);
    dw[a][0] = (f-1.5)/GRID_DX; dw[a][1] = -2.0*(f-1.0)/GRID_DX; dw[a][2] = (f-0.5)/GRID_DX;
  }
  for (int i = 0; i < 27; ++i) {
    const int ox = i/9, oy = (i/3)%3, oz = i%3;
    const unsigned int node = ((base[0]+ox)*GRID_N + base[1]+oy)*GRID_N + base[2]+oz;
    const double weight = w[0][ox]*w[1][oy]*w[2][oz];
    const double grad[3] = {dw[0][ox]*w[1][oy]*w[2][oz], w[0][ox]*dw[1][oy]*w[2][oz], w[0][ox]*w[1][oy]*dw[2][oz]};
    indices[p*27+i] = node; weights[p*27+i] = weight;
    atomicAdd(grid_mass+node, mass[p]*weight);
    for (int a = 0; a < 3; ++a) {
      double velocity = v[p*3+a], b = 0.0;
      for (int j = 0; j < 3; ++j) {
        velocity += C[p*9+a*3+j]*(grid_coordinate(node,j)-x[p*3+j]);
        if (!liquid) b += grad[j]*F[p*9+j*3+a];
      }
      atomicAdd(momentum+node*3+a, mass[p]*weight*velocity);
      // Liquid energy uses the raw gradients, never cof(F0)*F0 products.
      B[p*81+i*3+a] = liquid ? grad[a] : b;
    }
  }
}

extern "C" __global__ void grid_targets(const double* mass, const double* momentum, double* target, double h) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= GRID_COUNT) return;
  for (int a = 0; a < 3; ++a)
    target[node*3+a] = grid_coordinate(node,a) + (mass[node] > 0.0 ? h*momentum[node*3+a]/mass[node] - (a == 1 ? h*h*9.8 : 0.0) : 0.0);
}

extern "C" __global__ void mask_inactive(double* direction, const double* mass, int separate_coordinates) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node < GRID_COUNT && mass[node] == 0.0)
    for (int a = 0; a < 3; ++a) direction[separate_coordinates ? a*GRID_COUNT+node : node*3+a] = 0.0;
}

// Quadratic P2G supports have fixed offsets: their first node is an exact key
// for all 27 connectivity entries. Keep a real particle representative for it.
extern "C" __global__ void topology_stencil_representatives(
    const unsigned int* indices, unsigned int* representatives, unsigned int count) {
  const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < count) atomicMin(representatives + indices[27 * p], p);
}

extern "C" __global__ void topology_stencil_marks(
    const unsigned int* representatives, unsigned int* ordinals) {
  const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node < GRID_COUNT) ordinals[node] = representatives[node] != 0xffffffffu;
}

extern "C" __global__ void topology_gather_stencils(
    const unsigned int* indices, const unsigned int* representatives,
    const unsigned int* ordinals, unsigned int* compact_indices) {
  const unsigned int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= GRID_COUNT || representatives[node] == 0xffffffffu) return;
  const unsigned int p = representatives[node], output = ordinals[node] - 1;
  for (int i = 0; i < 27; ++i) compact_indices[27 * output + i] = indices[27 * p + i];
}

extern "C" __global__ void topology_block_dimensions(
    unsigned int* dimensions, unsigned int first, unsigned int count,
    unsigned int rows, unsigned int cols) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    dimensions[2 * (first + i)] = rows;
    dimensions[2 * (first + i) + 1] = cols;
  }
}

extern "C" __global__ void wall_step_bounds(
    const double* position, const double* displacement,
    const unsigned int* vertex_ids, double* bounds,
    int count, double lower_x, double lower_y, double lower_z,
    double upper_x, double upper_z, double slackness) {
  const int query = blockIdx.x * blockDim.x + threadIdx.x;
  if (query >= count) return;
  const unsigned int vertex = vertex_ids[query];
  double crossing = INFINITY;
  const double x = position[3*vertex], y = position[3*vertex+1], z = position[3*vertex+2];
  const double dx = displacement[3*vertex], dy = displacement[3*vertex+1], dz = displacement[3*vertex+2];
  if (dx < 0.0) crossing = fmin(crossing, (x-lower_x)/(-dx));
  if (dx > 0.0) crossing = fmin(crossing, (upper_x-x)/dx);
  if (dy < 0.0) crossing = fmin(crossing, (y-lower_y)/(-dy));
  if (dz < 0.0) crossing = fmin(crossing, (z-lower_z)/(-dz));
  if (dz > 0.0) crossing = fmin(crossing, (upper_z-z)/dz);
  bounds[query] = crossing <= 1.0 ? fmax(0.0, slackness*crossing) : 1.0;
}

extern "C" __global__ void g2p(
    const double* q, double h,
    const double* old_x, const double* old_F, const unsigned int* indices,
    const double* weights, const double* B, double* x, double* v, double* F, double* C, int count, const double* old_J, double* J, int liquid) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  double incremental_F[9];
  for (int a = 0; a < 3; ++a) {
    double dx = 0.0, df[3] = {0,0,0}, c[3] = {0,0,0};
    for (int i = 0; i < 27; ++i) {
      const unsigned int node = indices[p*27+i];
      const double d = q[node*3+a] - grid_coordinate(node,a);
      const double w = weights[p*27+i];
      dx += w*d;
      for (int b = 0; b < 3; ++b) {
        df[b] += d*B[p*81+i*3+b];
        c[b] += w*(d/h)*(grid_coordinate(node,b)-old_x[p*3+b]);
      }
    }
    x[p*3+a] = old_x[p*3+a]+dx; v[p*3+a] = dx/h;
    for (int b = 0; b < 3; ++b) {
      if (liquid) {
        incremental_F[a*3+b] = (a == b ? 1.0 : 0.0) + df[b];
        double value = old_F[p*9+a*3+b];
        for (int j = 0; j < 3; ++j) value += df[j]*old_F[p*9+j*3+b];
        F[p*9+a*3+b] = value;
      } else {
        F[p*9+a*3+b] = old_F[p*9+a*3+b]+df[b];
      }
      C[p*9+a*3+b] = 4.0/(GRID_DX*GRID_DX)*c[b];
    }
  }
  if (liquid) {
    // Commit volume once, after Newton: det((I+D)*F0) = det(I+D)*J0.
    const double* a = incremental_F;
    const double determinant = a[0]*(a[4]*a[8]-a[5]*a[7]) - a[1]*(a[3]*a[8]-a[5]*a[6]) + a[2]*(a[3]*a[7]-a[4]*a[6]);
    J[p] = old_J[p]*determinant;
  }
}

// Keep the CCD row ordering while separating the AV point from the SV feature.
extern "C" __global__ void split_point_feature_pairs(const unsigned int* pairs, unsigned int* points, unsigned int* surface, int arity, int count) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  points[p] = pairs[p * arity];
  for (int j = 1; j < arity; ++j) surface[p * (arity - 1) + j - 1] = pairs[p * arity + j];
}

extern "C" __global__ void contact_weights(const unsigned int* pairs, const double* vertex_weight, double* output, int arity, int count) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  double weight = 1.0;
  for (int i = 0; i < arity; ++i) {
    const double candidate = vertex_weight[pairs[p*arity+i]];
    if (candidate >= 0.0) weight = candidate;
  }
  output[p] = weight;
}

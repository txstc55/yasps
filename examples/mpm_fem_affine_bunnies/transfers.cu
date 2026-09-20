// Transfers, topology metadata, and safeguards. YASPS differentiates energies.
#include <cmath>

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

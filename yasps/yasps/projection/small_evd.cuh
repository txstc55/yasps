// Small real symmetric EVD, adapted from Eigen's SelfAdjointEigenSolver,
// Tridiagonalization, Householder and Jacobi implementations.
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2008-2010 Benoit Jacob <benoitjacob@googlemail.com>
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See https://mozilla.org/MPL/2.0/.
#pragma once

#include <cmath>
#include <cfloat>

#if defined(__CUDACC__)
#define SMALL_EVD_HD __host__ __device__ __forceinline__
#else
#define SMALL_EVD_HD inline
#endif

namespace small_evd {

// Full-double norm/reciprocal path; CUDA uses its double rsqrt (1 ULP
// documented observed error). Eigen's ratio form preserves tiny rotations
// and handles squaring outside the normal range. No float intermediates.
SMALL_EVD_HD void givens(double p, double q, double& c, double& s) {
  if (q == 0.0) { c = p < 0.0 ? -1.0 : 1.0; s = 0.0; }
  else if (p == 0.0) { c = 0.0; s = q < 0.0 ? 1.0 : -1.0; }
  // Tiny rotations retain Eigen's ratio form: it rounds the dominant
  // coefficient to exactly +/-1 rather than p * reciprocal(abs(p)).
  else if (::fabs(q) > 0x1p-26 * ::fabs(p) && ::fabs(p) > 0x1p-26 * ::fabs(q)
           && p * p + q * q >= DBL_MIN && p * p + q * q <= DBL_MAX) {
    // CUDA's double rsqrt retains double precision (documented 1 ULP).
#if defined(__CUDA_ARCH__)
    const double inverse_norm = ::rsqrt(p * p + q * q);
#else
    const double inverse_norm = 1.0 / ::sqrt(p * p + q * q);
#endif
    c = p * inverse_norm;
    s = -q * inverse_norm;
  }
  else if (::fabs(p) > ::fabs(q)) {
    const double t = q / p;
    // At |t| <= 2^-26, round-to-nearest binary64 sqrt(1+t*t) is
    // exactly one, including rounding of the sum. Preserve signed zeros
    // and skip both the square root and reciprocal in this exact case.
    if (::fabs(t) <= 0x1p-26) {
      c = p < 0.0 ? -1.0 : 1.0;
      s = -t * c;
      return;
    }
    double u = ::sqrt(1.0 + t * t);
    if (p < 0.0) u = -u;
    c = 1.0 / u; s = -t * c;
  } else {
    const double t = p / q;
    // The same exact tiny-ratio limit applies when q is dominant.
    if (::fabs(t) <= 0x1p-26) {
      s = q < 0.0 ? 1.0 : -1.0;
      c = -t * s;
      return;
    }
    double u = ::sqrt(1.0 + t * t);
    if (q < 0.0) u = -u;
    s = -1.0 / u; c = -t * s;
  }
}

// NVVM otherwise recognizes the fixed-index selection as a dynamic array
// lookup and allocates a local diagonal array. A register select makes the
// intended storage explicit while preserving every bit of the selected value.
SMALL_EVD_HD double register_select(bool condition, double yes, double no) {
#if defined(__CUDA_ARCH__)
  double result;
  asm("{ .reg .pred p; setp.ne.u32 p, %3, 0; selp.f64 %0, %1, %2, p; }"
      : "=d"(result) : "d"(yes), "d"(no), "r"(int(condition)));
  return result;
#else
  return condition ? yes : no;
#endif
}

// Eigenvectors are column-major. Input is column-major by default; set
// RowMajorInput to read row-major lower storage directly, without a mirror pass.
// Only the input's lower triangle is read. Input may alias vectors.
// Eigenvalues are intentionally unsorted:
// projection needs eigenpairs, not the extra matrix traffic of sorting.
// Supports positive compile-time dimensions N. QR uses fixed indices to keep its
// diagonal and subdiagonal in registers on CUDA.
// Householder uses the unused upper triangle, with no extra matrix workspace.
// A finite lower triangle is interpreted as symmetric; the upper is ignored.
// values must not overlap vectors/input;
// input and vectors may be identical, but must not otherwise partially overlap.
// Returns false at Eigen's 30*N QR-iteration limit; vectors/values are then
// partially modified and must not be used as a decomposition.
template<int N, bool RowMajorInput = false>
SMALL_EVD_HD bool decompose(const double* input, double* vectors, double* values) {
  static_assert(N >= 1, "small_evd requires a positive matrix dimension");
  if (N == 1) { values[0] = input[0]; vectors[0] = 1.0; return true; }
  if constexpr (N == 2) {
    // One Jacobi rotation diagonalizes a symmetric 2x2 matrix directly.
    // Capture the lower triangle before writing Q, so exact aliasing is safe.
    const double a = input[0], b = input[RowMajorInput ? 2 : 1], d = input[3];
    double t = 0.0;
    if (b != 0.0) {
      const double scale = ::fmax(::fabs(a), ::fmax(::fabs(b), ::fabs(d)));
      const bool normal_reciprocal = scale >= DBL_MIN && scale <= 1.0 / DBL_MIN;
      const double inverse_scale = normal_reciprocal ? 1.0 / scale : 0.0;
      const double as = normal_reciprocal ? a * inverse_scale : a / scale;
      const double bs = normal_reciprocal ? b * inverse_scale : b / scale;
      const double ds = normal_reciprocal ? d * inverse_scale : d / scale;
      if (bs != 0.0) {
        const double delta = 0.5 * (ds - as);
        // The same-sign denominator avoids cancellation; hypot also retains
        // tiny couplings when delta is zero. The resulting |t| is at most one.
        t = bs / (delta + ::copysign(::hypot(delta, bs), delta));
      }
    }
    const double c = 1.0 / ::sqrt(1.0 + t * t);
    const double s = t * c;
    // Use original entries rather than rescaling rounded eigenvalues, retaining
    // tiny diagonal entries. FMA avoids an extra rounding in each correction.
    values[0] = ::fma(-t, b, a);
    values[1] = ::fma(t, b, d);
    vectors[0] = c;  vectors[1] = -s;
    vectors[2] = s;  vectors[3] = c;
    return true;
  }

  double scale = 0.0;
  for (int j = 0; j < N; ++j)
    for (int i = j; i < N; ++i) {
      const double a = ::fabs(input[RowMajorInput ? i * N + j : i + j * N]);
      if (a > scale) scale = a;
    }
  if (scale == 0.0) scale = 1.0;
  // One correctly-rounded reciprocal replaces the triangular set of FP64
  // divisions. Preserve division when the reciprocal is not normal.
  const bool normal_reciprocal = scale >= DBL_MIN && scale <= 1.0 / DBL_MIN;
  const double inverse_scale = normal_reciprocal ? 1.0 / scale : 0.0;
  // Fuse the input-layout read into scaling. With row-major exact aliasing,
  // off-diagonal writes land in the unused input upper triangle, so no unread
  // lower entry is overwritten; diagonal entries are scaled in place.
  for (int j = 0; j < N; ++j)
    for (int i = j; i < N; ++i) {
      const double value = input[RowMajorInput ? i * N + j : i + j * N];
      vectors[i + j * N] = normal_reciprocal
          ? value * inverse_scale : value / scale;
    }

  double diagonal[N], subdiag[N];
  // Eigen's exact 3x3 tridiagonalization needs only one plane transform.
  // It saves the generic rank-two update and Householder-Q expansion.
  if constexpr (N == 3) {
    diagonal[0] = vectors[0];
    const double a10 = vectors[1];
    const double a20 = vectors[2];
    const double a11 = vectors[4];
    const double a21 = vectors[5];
    const double a22 = vectors[8];
    const double v1norm2 = a20 * a20;
    if (v1norm2 <= DBL_MIN) {
      diagonal[1] = a11;
      diagonal[2] = a22;
      subdiag[0] = a10;
      subdiag[1] = a21;
      for (int i = 0; i < 9; ++i) vectors[i] = i % 4 == 0 ? 1.0 : 0.0;
    } else {
      const double beta = ::sqrt(a10 * a10 + v1norm2);
      const double inv_beta = 1.0 / beta;
      const double m01 = a10 * inv_beta;
      const double m02 = a20 * inv_beta;
      const double q = 2.0 * m01 * a21 + m02 * (a22 - a11);
      diagonal[1] = a11 + m02 * q;
      diagonal[2] = a22 - m02 * q;
      subdiag[0] = beta;
      subdiag[1] = a21 - m01 * q;
      vectors[0] = 1.0;
      vectors[1] = vectors[2] = vectors[3] = vectors[6] = 0.0;
      vectors[4] = m01;
      vectors[5] = vectors[7] = m02;
      vectors[8] = -m01;
    }
  } else {
    // The unused upper row holds this reflector's temporary matvec.
    // Its first entry then holds tau until reverse Q expansion consumes it.
    for (int k = 0; k < N - 1; ++k) {
      double tail_norm = 0.0;
      for (int i = k + 2; i < N; ++i) {
        const double v = vectors[i + k * N];
        tail_norm += v * v;
      }
      const double first = vectors[k + 1 + k * N];
      double beta = first;
      double tau = 0.0;
      if (tail_norm > DBL_MIN) {
        beta = ::sqrt(first * first + tail_norm);
        if (first >= 0.0) beta = -beta;
        // Nonzero tail norm bounds this reciprocal away from overflow.
        const double inverse_denominator = 1.0 / (first - beta);
        for (int i = k + 2; i < N; ++i) vectors[i + k * N] *= inverse_denominator;
        tau = (beta - first) / beta;
      } else {
        for (int i = k + 2; i < N; ++i) vectors[i + k * N] = 0.0;
      }
      vectors[k + 1 + k * N] = 1.0;
      if (tau != 0.0) {
        for (int i = k + 1; i < N; ++i) {
          double sum = 0.0;
          for (int j = k + 1; j <= i; ++j)
            sum += vectors[i + j * N] * vectors[j + k * N];
          for (int j = i + 1; j < N; ++j)
            sum += vectors[j + i * N] * vectors[j + k * N];
          vectors[k + i * N] = tau * sum;
        }
        double dot = 0.0;
        for (int i = k + 1; i < N; ++i) dot += vectors[k + i * N] * vectors[i + k * N];
        const double correction = -0.5 * tau * dot;
        for (int i = k + 1; i < N; ++i) vectors[k + i * N] += correction * vectors[i + k * N];
        for (int j = k + 1; j < N; ++j)
          for (int i = j; i < N; ++i)
            vectors[i + j * N] -= vectors[i + k * N] * vectors[k + j * N] + vectors[k + i * N] * vectors[j + k * N];
      }
      vectors[k + 1 + k * N] = beta;
      vectors[k + (k + 1) * N] = tau;
    }
    // Fixed indices let the CUDA compiler scalarize the tridiagonal.
#pragma unroll
    for (int i = 0; i < N; ++i) diagonal[i] = vectors[i + i * N];
#pragma unroll
    for (int i = 0; i < N - 1; ++i) subdiag[i] = vectors[i + 1 + i * N];

    // Expand the reflectors backwards in their original matrix storage.
    // Only the already-expanded trailing block is touched by each reflector.
    vectors[N * N - 1] = 1.0;
    for (int k = N - 2; k >= 0; --k) {
      const double tau = vectors[k + (k + 1) * N];
      vectors[k + 1 + k * N] = 1.0;
      if (tau != 0.0) {
        for (int j = k + 1; j < N; ++j) {
          double dot = 0.0;
          for (int i = k + 1; i < N; ++i) dot += vectors[i + k * N] * vectors[i + j * N];
          dot *= tau;
          for (int i = k + 1; i < N; ++i) vectors[i + j * N] -= vectors[i + k * N] * dot;
        }
      }
      for (int i = k + 1; i < N; ++i) {
        vectors[i + k * N] = 0.0;
        vectors[k + i * N] = 0.0;
      }
      vectors[k + k * N] = 1.0;
    }
  }

  int end = N - 1;
  int start = 0;
  int iterations = 0;
  while (end > 0) {
#pragma unroll
    for (int i = 0; i < N - 1; ++i) if (i >= start && i < end) {
      if (::fabs(subdiag[i]) < DBL_MIN) subdiag[i] = 0.0;
      else {
        const double scaled = subdiag[i] / DBL_EPSILON;
        if (scaled * scaled <= ::fabs(diagonal[i]) + ::fabs(diagonal[i + 1])) subdiag[i] = 0.0;
      }
    }
    // Fixed-index scans reproduce the dynamic trailing-zero/block scans.
    int next_end = 0;
#pragma unroll
    for (int i = 0; i < N - 1; ++i)
      if (i < end && subdiag[i] != 0.0) next_end = i + 1;
    end = next_end;
    if (end == 0) break;
    if (++iterations > 30 * N) {
#pragma unroll
      for (int i = 0; i < N; ++i) values[i] = diagonal[i];
      return false;
    }
    start = 0;
#pragma unroll
    for (int i = 0; i < N - 1; ++i)
      if (i < end - 1 && subdiag[i] == 0.0) start = i + 1;

    // Underflow-safe Wilkinson shift, matching Eigen's QR step.
    double trailing0 = 0.0, trailing1 = 0.0, e = 0.0;
    double leading = 0.0, z = 0.0;
#pragma unroll
    for (int i = 0; i < N - 1; ++i) {
      trailing0 = register_select(i == end - 1, diagonal[i], trailing0);
      trailing1 = register_select(i == end - 1, diagonal[i + 1], trailing1);
      e = register_select(i == end - 1, subdiag[i], e);
      leading = register_select(i == start, diagonal[i], leading);
      z = register_select(i == start, subdiag[i], z);
    }
    const double td = 0.5 * (trailing0 - trailing1);
    double mu = trailing1;
    if (td == 0.0) mu -= ::fabs(e);
    else if (e != 0.0) {
      const double e2 = e * e;
      // Input scaling bounds these values by O(N), so their squares cannot
      // overflow for the supported small dimensions. Keep hypot's ratio
      // scaling only when squares are subnormal; its general overflow
      // machinery is otherwise unnecessary in every QR iteration.
      const double square_sum = td * td + e2;
      const double h = square_sum >= DBL_MIN ? ::sqrt(square_sum) : ::hypot(td, e);
      const double denominator = td + (td > 0.0 ? h : -h);
      mu -= e2 == 0.0 ? e / (denominator / e) : e2 / denominator;
    }
    double x = leading - mu;
#pragma unroll
    for (int k = 0; k < N - 1; ++k) if (k >= start && k < end && z != 0.0) {
      double c, s;
      givens(x, z, c, s);
      const double d0 = diagonal[k];
      const double d1 = diagonal[k + 1];
      const double off = subdiag[k];
      // Algebraically equivalent symmetric similarity update. The paired
      // diagonal correction also avoids accumulating trace roundoff.
      const double delta = d1 - d0;
      const double correction = s * (s * delta - 2.0 * c * off);
      diagonal[k] = d0 + correction;
      diagonal[k + 1] = d1 - correction;
      subdiag[k] = -c * s * delta + (c * c - s * s) * off;
      if (k > start) subdiag[k - 1] = c * subdiag[k - 1] - s * z;
      x = subdiag[k];
      if (k < end - 1) {
        z = -s * subdiag[k + 1];
        subdiag[k + 1] *= c;
      }
      // Eight rows expose independent work without fully expanding this
      // loop inside each already-unrolled QR column.
#pragma unroll 8
      for (int i = 0; i < N; ++i) {
        const double a = vectors[i + k * N];
        const double b = vectors[i + (k + 1) * N];
        vectors[i + k * N] = c * a - s * b;
        vectors[i + (k + 1) * N] = s * a + c * b;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < N; ++i) values[i] = diagonal[i] * scale;
  return true;
}

// The source remains untouched unless input==output; both forms are supported.
// choice=0 copies, choice=1 reflects negative eigenvalues, all others clamp.
// No positive-semidefinite precheck is performed. On nonconvergence, returns
// false without writing output. Exact alias is supported; partial overlap is not.
template<int N>
SMALL_EVD_HD bool project(const double* input, double* output, int choice) {
  if (choice == 0) {
    for (int i = 0; i < N * N; ++i) output[i] = input[i];
    return true;
  }
  double vectors[N * N];
  double values[N];
  if (!decompose<N>(input, vectors, values)) return false;
  for (int k = 0; k < N; ++k)
    if (values[k] < 0.0) values[k] = choice == 1 ? -values[k] : 0.0;
  for (int j = 0; j < N; ++j)
    for (int i = j; i < N; ++i) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) sum += vectors[i + k * N] * values[k] * vectors[j + k * N];
      output[i + j * N] = sum;
      output[j + i * N] = sum;
    }
  return true;
}

// Lower-stack alternative: use the output as the EVD matrix workspace.
// Reconstruction consumes one eigenvector row at a time, writing only that
// row's upper triangle; all later eigenvector rows remain intact. Mirroring
// happens only after the last eigenvector row has been consumed.
// Local storage is O(N), at the cost of accessing output throughout the EVD.
// Prefer this when output is an existing thread-local array. Direct global
// output can be slower because its accesses are not interleaved across threads.
// Returns false on nonconvergence with output partially modified. Same choices
// and alias rules as project; check the return value before using the result.
// RowMajorInput has the same meaning as in decompose; symmetric output can be
// consumed in either layout. choice=0 copies the original bytes unchanged.
template<int N, bool RowMajorInput = false>
SMALL_EVD_HD bool project_reuse_output(const double* input, double* output, int choice) {
  if (choice == 0) {
    for (int i = 0; i < N * N; ++i) output[i] = input[i];
    return true;
  }
  double values[N];
  if (!decompose<N, RowMajorInput>(input, output, values)) return false;
  for (int k = 0; k < N; ++k)
    if (values[k] < 0.0) values[k] = choice == 1 ? -values[k] : 0.0;
  double row[N];
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) row[k] = output[i + k * N];
    double diagonal = 0.0;
    for (int k = 0; k < N; ++k) diagonal += row[k] * values[k] * row[k];
    output[i + i * N] = diagonal;
    for (int j = i + 1; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) sum += row[k] * values[k] * output[j + k * N];
      output[i + j * N] = sum;
    }
  }
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) output[j + i * N] = output[i + j * N];
  return true;
}

} // namespace small_evd

template <ushort N>
METAL_FUNC float yasps_determinant(thread const float* input) {
  float a[N * N];
  for (ushort i = 0; i < N * N; ++i) a[i] = input[i];

  float determinant = 1.0f;
  for (ushort column = 0; column < N; ++column) {
    ushort pivot = column;
    float pivot_abs = metal::abs(a[column * N + column]);
    for (ushort row = column + 1; row < N; ++row) {
      const float candidate = metal::abs(a[row * N + column]);
      if (candidate > pivot_abs) {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (pivot_abs <= 1.0e-20f) return 0.0f;
    if (pivot != column) {
      for (ushort j = column; j < N; ++j) {
        const float temporary = a[column * N + j];
        a[column * N + j] = a[pivot * N + j];
        a[pivot * N + j] = temporary;
      }
      determinant = -determinant;
    }
    const float diagonal = a[column * N + column];
    determinant *= diagonal;
    for (ushort row = column + 1; row < N; ++row) {
      const float scale = a[row * N + column] / diagonal;
      for (ushort j = column + 1; j < N; ++j) {
        a[row * N + j] -= scale * a[column * N + j];
      }
    }
  }
  return determinant;
}

template <ushort N>
METAL_FUNC void yasps_inverse(
    thread const float* input,
    thread float* output) {
  float augmented[N * N * 2];
  for (ushort row = 0; row < N; ++row) {
    for (ushort column = 0; column < N; ++column) {
      augmented[row * (2 * N) + column] = input[row * N + column];
      augmented[row * (2 * N) + N + column] =
          row == column ? 1.0f : 0.0f;
    }
  }

  for (ushort column = 0; column < N; ++column) {
    ushort pivot = column;
    float pivot_abs = metal::abs(augmented[column * (2 * N) + column]);
    for (ushort row = column + 1; row < N; ++row) {
      const float candidate =
          metal::abs(augmented[row * (2 * N) + column]);
      if (candidate > pivot_abs) {
        pivot = row;
        pivot_abs = candidate;
      }
    }
    if (pivot != column) {
      for (ushort j = 0; j < 2 * N; ++j) {
        const float temporary = augmented[column * (2 * N) + j];
        augmented[column * (2 * N) + j] =
            augmented[pivot * (2 * N) + j];
        augmented[pivot * (2 * N) + j] = temporary;
      }
    }

    float diagonal = augmented[column * (2 * N) + column];
    if (metal::abs(diagonal) <= 1.0e-20f) {
      diagonal = diagonal < 0.0f ? -1.0e-20f : 1.0e-20f;
    }
    for (ushort j = 0; j < 2 * N; ++j) {
      augmented[column * (2 * N) + j] /= diagonal;
    }
    for (ushort row = 0; row < N; ++row) {
      if (row == column) continue;
      const float scale = augmented[row * (2 * N) + column];
      for (ushort j = 0; j < 2 * N; ++j) {
        augmented[row * (2 * N) + j] -=
            scale * augmented[column * (2 * N) + j];
      }
    }
  }

  for (ushort row = 0; row < N; ++row) {
    for (ushort column = 0; column < N; ++column) {
      output[row * N + column] =
          augmented[row * (2 * N) + N + column];
    }
  }
}

// Batched YASPS kernels invoke this once per simulation primitive. Each Metal
// thread therefore diagonalizes one small Hessian while the GPU processes the
// full batch in parallel. The matrix never leaves thread-local storage.
template <ushort N>
METAL_FUNC void yasps_spd_project(thread float* a, int choice) {
  if (choice == 0) return;
  if (N == 1) {
    a[0] = choice == 1 ? metal::abs(a[0]) : metal::max(a[0], 0.0f);
    return;
  }

  float eigenvectors[N * N];
  for (ushort row = 0; row < N; ++row) {
    for (ushort column = 0; column < N; ++column) {
      const float lhs = a[row * N + column];
      const float rhs = a[column * N + row];
      a[row * N + column] = 0.5f * (lhs + rhs);
      eigenvectors[row * N + column] = row == column ? 1.0f : 0.0f;
    }
  }

  for (ushort iteration = 0; iteration < N * N * 8; ++iteration) {
    ushort p = 0;
    ushort q = 1;
    float largest = 0.0f;
    float diagonal_scale = 0.0f;
    for (ushort row = 0; row < N; ++row) {
      diagonal_scale =
          metal::max(diagonal_scale, metal::abs(a[row * N + row]));
      for (ushort column = row + 1; column < N; ++column) {
        const float candidate = metal::abs(a[row * N + column]);
        if (candidate > largest) {
          largest = candidate;
          p = row;
          q = column;
        }
      }
    }
    if (largest <= 2.0e-6f * metal::max(diagonal_scale, 1.0f)) break;

    const float app = a[p * N + p];
    const float aqq = a[q * N + q];
    const float apq = a[p * N + q];
    const float tau = (aqq - app) / (2.0f * apq);
    const float tangent =
        (tau >= 0.0f ? 1.0f : -1.0f) /
        (metal::abs(tau) + metal::sqrt(1.0f + tau * tau));
    const float cosine = metal::rsqrt(1.0f + tangent * tangent);
    const float sine = tangent * cosine;

    for (ushort k = 0; k < N; ++k) {
      if (k == p || k == q) continue;
      const float akp = a[k * N + p];
      const float akq = a[k * N + q];
      const float rotated_p = cosine * akp - sine * akq;
      const float rotated_q = sine * akp + cosine * akq;
      a[k * N + p] = rotated_p;
      a[p * N + k] = rotated_p;
      a[k * N + q] = rotated_q;
      a[q * N + k] = rotated_q;
    }
    a[p * N + p] =
        cosine * cosine * app - 2.0f * sine * cosine * apq
        + sine * sine * aqq;
    a[q * N + q] =
        sine * sine * app + 2.0f * sine * cosine * apq
        + cosine * cosine * aqq;
    a[p * N + q] = 0.0f;
    a[q * N + p] = 0.0f;

    for (ushort row = 0; row < N; ++row) {
      const float vp = eigenvectors[row * N + p];
      const float vq = eigenvectors[row * N + q];
      eigenvectors[row * N + p] = cosine * vp - sine * vq;
      eigenvectors[row * N + q] = sine * vp + cosine * vq;
    }
  }

  float eigenvalues[N];
  for (ushort i = 0; i < N; ++i) {
    const float value = a[i * N + i];
    eigenvalues[i] =
        choice == 1 ? metal::abs(value) : metal::max(value, 0.0f);
  }
  float projected[N * N];
  for (ushort row = 0; row < N; ++row) {
    for (ushort column = 0; column < N; ++column) {
      float value = 0.0f;
      for (ushort k = 0; k < N; ++k) {
        value += eigenvectors[row * N + k] * eigenvalues[k]
            * eigenvectors[column * N + k];
      }
      projected[row * N + column] = value;
    }
  }
  for (ushort i = 0; i < N * N; ++i) a[i] = projected[i];
}

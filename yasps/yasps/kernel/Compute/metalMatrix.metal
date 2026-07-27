#ifndef YASPS_METAL_MATRIX
#define YASPS_METAL_MATRIX

#include <metal_stdlib>
using namespace metal;

// Fixed-size row-major matrices used by generated YASPS device functions.
// Their storage stays private to one Metal thread, matching the former Eigen
// temporaries in CUDA kernels.
template <uint Rows, uint Cols>
struct YaspsMatrix {
  float values[Rows * Cols];

  thread float *data() thread {
    return values;
  }

  thread const float *data() const thread {
    return values;
  }

  thread float &operator()(uint row, uint col) thread {
    return values[row * Cols + col];
  }

  float operator()(uint row, uint col) const thread {
    return values[row * Cols + col];
  }

  float value() const thread {
    static_assert(Rows * Cols == 1, "value() requires a scalar matrix");
    return values[0];
  }

  YaspsMatrix<Rows, Cols> array() const thread {
    return *this;
  }

  thread YaspsMatrix<Rows, Cols> &noalias() thread {
    return *this;
  }

  YaspsMatrix<Rows, Cols> derived() const thread {
    return *this;
  }

  YaspsMatrix<Rows, Cols> sin() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = metal::sin(values[index]);
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> cos() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = metal::cos(values[index]);
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> tan() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = metal::tan(values[index]);
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> abs() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = metal::fabs(values[index]);
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> log() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = metal::log(values[index]);
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> sqrt() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = metal::sqrt(values[index]);
    }
    return result;
  }

  YaspsMatrix<Cols, Rows> transpose() const thread {
    YaspsMatrix<Cols, Rows> result = {};
    for (uint row = 0; row < Rows; ++row) {
      for (uint col = 0; col < Cols; ++col) {
        result(col, row) = (*this)(row, col);
      }
    }
    return result;
  }

  YaspsMatrix<1, Cols> row(uint selected) const thread {
    YaspsMatrix<1, Cols> result = {};
    for (uint col = 0; col < Cols; ++col) {
      result(0, col) = (*this)(selected, col);
    }
    return result;
  }

  YaspsMatrix<Rows, 1> col(uint selected) const thread {
    YaspsMatrix<Rows, 1> result = {};
    for (uint row_index = 0; row_index < Rows; ++row_index) {
      result(row_index, 0) = (*this)(row_index, selected);
    }
    return result;
  }

  float dot(thread const YaspsMatrix<Rows, Cols> &other) const thread {
    float result = 0.0f;
    for (uint index = 0; index < Rows * Cols; ++index) {
      result += values[index] * other.values[index];
    }
    return result;
  }

  float norm() const thread {
    return metal::sqrt(dot(*this));
  }

  YaspsMatrix<Rows, Cols> cross(
      thread const YaspsMatrix<Rows, Cols> &other) const thread {
    static_assert(Rows * Cols == 3, "cross() requires three elements");
    YaspsMatrix<Rows, Cols> result = {};
    result.values[0] = values[1] * other.values[2] -
                       values[2] * other.values[1];
    result.values[1] = values[2] * other.values[0] -
                       values[0] * other.values[2];
    result.values[2] = values[0] * other.values[1] -
                       values[1] * other.values[0];
    return result;
  }

  float determinant() const thread {
    static_assert(Rows == Cols, "determinant() requires a square matrix");
    YaspsMatrix<Rows, Cols> work = *this;
    float determinant_value = 1.0f;
    for (uint pivot = 0; pivot < Rows; ++pivot) {
      uint best_row = pivot;
      float best_value = fabs(work(pivot, pivot));
      for (uint row_index = pivot + 1; row_index < Rows; ++row_index) {
        float candidate = fabs(work(row_index, pivot));
        if (candidate > best_value) {
          best_value = candidate;
          best_row = row_index;
        }
      }
      if (best_value <= 1.0e-20f) {
        return 0.0f;
      }
      if (best_row != pivot) {
        for (uint col_index = 0; col_index < Cols; ++col_index) {
          float temporary = work(pivot, col_index);
          work(pivot, col_index) = work(best_row, col_index);
          work(best_row, col_index) = temporary;
        }
        determinant_value = -determinant_value;
      }
      float diagonal = work(pivot, pivot);
      determinant_value *= diagonal;
      for (uint row_index = pivot + 1; row_index < Rows; ++row_index) {
        float factor = work(row_index, pivot) / diagonal;
        for (uint col_index = pivot + 1; col_index < Cols; ++col_index) {
          work(row_index, col_index) -=
              factor * work(pivot, col_index);
        }
      }
    }
    return determinant_value;
  }

  YaspsMatrix<Rows, Cols> inverse() const thread {
    static_assert(Rows == Cols, "inverse() requires a square matrix");
    YaspsMatrix<Rows, Cols> work = *this;
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows; ++index) {
      result(index, index) = 1.0f;
    }
    for (uint pivot = 0; pivot < Rows; ++pivot) {
      uint best_row = pivot;
      float best_value = fabs(work(pivot, pivot));
      for (uint row_index = pivot + 1; row_index < Rows; ++row_index) {
        float candidate = fabs(work(row_index, pivot));
        if (candidate > best_value) {
          best_value = candidate;
          best_row = row_index;
        }
      }
      if (best_row != pivot) {
        for (uint col_index = 0; col_index < Cols; ++col_index) {
          float temporary = work(pivot, col_index);
          work(pivot, col_index) = work(best_row, col_index);
          work(best_row, col_index) = temporary;
          temporary = result(pivot, col_index);
          result(pivot, col_index) = result(best_row, col_index);
          result(best_row, col_index) = temporary;
        }
      }
      float divisor = work(pivot, pivot);
      float safe_divisor =
          fabs(divisor) < 1.0e-20f ? copysign(1.0e-20f, divisor) : divisor;
      for (uint col_index = 0; col_index < Cols; ++col_index) {
        work(pivot, col_index) /= safe_divisor;
        result(pivot, col_index) /= safe_divisor;
      }
      for (uint row_index = 0; row_index < Rows; ++row_index) {
        if (row_index == pivot) {
          continue;
        }
        float factor = work(row_index, pivot);
        for (uint col_index = 0; col_index < Cols; ++col_index) {
          work(row_index, col_index) -=
              factor * work(pivot, col_index);
          result(row_index, col_index) -=
              factor * result(pivot, col_index);
        }
      }
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator-() const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = -values[index];
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator+(
      thread const YaspsMatrix<Rows, Cols> &other) const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = values[index] + other.values[index];
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator-(
      thread const YaspsMatrix<Rows, Cols> &other) const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = values[index] - other.values[index];
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator+(float scalar) const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = values[index] + scalar;
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator-(float scalar) const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = values[index] - scalar;
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator*(float scalar) const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = values[index] * scalar;
    }
    return result;
  }

  YaspsMatrix<Rows, Cols> operator/(float scalar) const thread {
    YaspsMatrix<Rows, Cols> result = {};
    for (uint index = 0; index < Rows * Cols; ++index) {
      result.values[index] = values[index] / scalar;
    }
    return result;
  }

  template <uint OtherCols>
  YaspsMatrix<Rows, OtherCols> operator*(
      thread const YaspsMatrix<Cols, OtherCols> &other) const thread {
    YaspsMatrix<Rows, OtherCols> result = {};
    for (uint row_index = 0; row_index < Rows; ++row_index) {
      for (uint col_index = 0; col_index < OtherCols; ++col_index) {
        float element = 0.0f;
        for (uint inner = 0; inner < Cols; ++inner) {
          element += (*this)(row_index, inner) *
                     other(inner, col_index);
        }
        result(row_index, col_index) = element;
      }
    }
    return result;
  }
};

// Pointer-backed equivalent of Eigen::Map<const Matrix<...>>. Generated
// helper functions use this for read-only inputs so large local matrices are
// not copied into every helper's private storage.
template <uint Rows, uint Cols>
struct YaspsMatrixView {
  thread const float *values;

  thread const float *data() const thread {
    return values;
  }

  float operator()(uint row, uint col) const thread {
    return values[row * Cols + col];
  }
};

template <uint Rows, uint Cols>
YaspsMatrixView<Rows, Cols> yasps_matrix_view(
    thread const float *source) {
  return {source};
}

inline float yasps_scalar_value(float value) {
  return value;
}

inline float yasps_scalar_value(YaspsMatrix<1, 1> value) {
  return value.values[0];
}

template <uint Rows, uint Cols>
YaspsMatrix<Rows, Cols> operator*(
    float scalar, thread const YaspsMatrix<Rows, Cols> &matrix) {
  return matrix * scalar;
}

template <uint Rows, uint Cols>
YaspsMatrix<Rows, Cols> operator+(
    float scalar, thread const YaspsMatrix<Rows, Cols> &matrix) {
  return matrix + scalar;
}

template <uint Rows, uint Cols>
YaspsMatrix<Rows, Cols> operator-(
    float scalar, thread const YaspsMatrix<Rows, Cols> &matrix) {
  YaspsMatrix<Rows, Cols> result = {};
  for (uint index = 0; index < Rows * Cols; ++index) {
    result.values[index] = scalar - matrix.values[index];
  }
  return result;
}

template <uint Rows, uint Cols>
YaspsMatrix<Rows, Cols> yasps_matrix_from_pointer(
    thread const float *source) {
  YaspsMatrix<Rows, Cols> result = {};
  for (uint index = 0; index < Rows * Cols; ++index) {
    result.values[index] = source[index];
  }
  return result;
}

#define YASPS_UNARY_MATRIX_FUNCTION(Name, Function)                         \
  template <uint Rows, uint Cols>                                           \
  YaspsMatrix<Rows, Cols> Name(                                             \
      thread const YaspsMatrix<Rows, Cols> &input) {                        \
    YaspsMatrix<Rows, Cols> result = {};                                    \
    for (uint index = 0; index < Rows * Cols; ++index) {                    \
      result.values[index] = Function(input.values[index]);                 \
    }                                                                       \
    return result;                                                          \
  }

YASPS_UNARY_MATRIX_FUNCTION(yasps_sin, sin)
YASPS_UNARY_MATRIX_FUNCTION(yasps_cos, cos)
YASPS_UNARY_MATRIX_FUNCTION(yasps_tan, tan)
YASPS_UNARY_MATRIX_FUNCTION(yasps_abs, fabs)
YASPS_UNARY_MATRIX_FUNCTION(yasps_log, log)
YASPS_UNARY_MATRIX_FUNCTION(yasps_sqrt, sqrt)

#undef YASPS_UNARY_MATRIX_FUNCTION

template <uint Size>
void yasps_symmetric_jacobi_eigendecomposition(
    thread YaspsMatrix<Size, Size> &matrix,
    thread YaspsMatrix<Size, Size> &eigenvectors) {
  for (uint sweep = 0; sweep < 12; ++sweep) {
    float off_diagonal = 0.0f;
    for (uint p = 0; p + 1 < Size; ++p) {
      for (uint q = p + 1; q < Size; ++q) {
        float apq = matrix(p, q);
        off_diagonal = max(off_diagonal, fabs(apq));
        if (fabs(apq) <= 1.0e-7f) {
          continue;
        }
        float app = matrix(p, p);
        float aqq = matrix(q, q);
        float tau = (aqq - app) / (2.0f * apq);
        float tangent =
            (tau >= 0.0f ? 1.0f : -1.0f) /
            (fabs(tau) + sqrt(1.0f + tau * tau));
        float cosine = rsqrt(1.0f + tangent * tangent);
        float sine = tangent * cosine;

        for (uint index = 0; index < Size; ++index) {
          float aip = matrix(index, p);
          float aiq = matrix(index, q);
          matrix(index, p) = cosine * aip - sine * aiq;
          matrix(index, q) = sine * aip + cosine * aiq;
        }
        for (uint index = 0; index < Size; ++index) {
          float api = matrix(p, index);
          float aqi = matrix(q, index);
          matrix(p, index) = cosine * api - sine * aqi;
          matrix(q, index) = sine * api + cosine * aqi;
        }
        for (uint index = 0; index < Size; ++index) {
          float vip = eigenvectors(index, p);
          float viq = eigenvectors(index, q);
          eigenvectors(index, p) = cosine * vip - sine * viq;
          eigenvectors(index, q) = sine * vip + cosine * viq;
        }
      }
    }
    if (off_diagonal <= 1.0e-6f) {
      break;
    }
  }
}

template <uint Size>
void yasps_spd_projection_inplace(thread float *source, int choice) {
  if (choice == 0) {
    return;
  }
  for (uint index = 0; index < Size * Size; ++index) {
    if (!isfinite(source[index])) {
      for (uint output_index = 0;
           output_index < Size * Size;
           ++output_index) {
        source[output_index] = 0.0f;
      }
      return;
    }
  }

  YaspsMatrix<Size, Size> matrix =
      yasps_matrix_from_pointer<Size, Size>(source);
  YaspsMatrix<Size, Size> eigenvectors = {};
  for (uint index = 0; index < Size; ++index) {
    eigenvectors(index, index) = 1.0f;
  }
  yasps_symmetric_jacobi_eigendecomposition<Size>(
      matrix, eigenvectors);

  for (uint index = 0; index < Size; ++index) {
    float eigenvalue = matrix(index, index);
    if (eigenvalue < 0.0f) {
      matrix(index, index) =
          choice == 1 ? fabs(eigenvalue) : 0.0f;
    }
  }
  bool finite_projection = true;
  for (uint row = 0; row < Size; ++row) {
    for (uint col = row; col < Size; ++col) {
      float value = 0.0f;
      for (uint inner = 0; inner < Size; ++inner) {
        value += eigenvectors(row, inner) * matrix(inner, inner) *
                 eigenvectors(col, inner);
      }
      source[row * Size + col] = value;
      source[col * Size + row] = value;
      finite_projection = finite_projection && isfinite(value);
    }
  }
  if (!finite_projection) {
    for (uint index = 0; index < Size * Size; ++index) {
      source[index] = 0.0f;
    }
  }
}

template <uint Size>
void spd_projection_inplace(thread float *source, int choice) {
  yasps_spd_projection_inplace<Size>(source, choice);
}

template <uint Size>
void spd_projection_small(
    thread const float *source,
    thread float *output,
    int choice) {
  for (uint index = 0; index < Size * Size; ++index) {
    output[index] = source[index];
  }
  yasps_spd_projection_inplace<Size>(output, choice);
}

template <uint Size>
void yasps_symmetric_pseudoinverse(
    thread const float *source,
    thread float *output) {
  for (uint index = 0; index < Size * Size; ++index) {
    if (!isfinite(source[index])) {
      for (uint row = 0; row < Size; ++row) {
        for (uint col = 0; col < Size; ++col) {
          output[row * Size + col] = row == col ? 1.0f : 0.0f;
        }
      }
      return;
    }
  }
  YaspsMatrix<Size, Size> matrix =
      yasps_matrix_from_pointer<Size, Size>(source);
  YaspsMatrix<Size, Size> eigenvectors = {};
  for (uint index = 0; index < Size; ++index) {
    eigenvectors(index, index) = 1.0f;
  }
  yasps_symmetric_jacobi_eigendecomposition<Size>(
      matrix, eigenvectors);

  float inverse_eigenvalues[Size];
  for (uint index = 0; index < Size; ++index) {
    float eigenvalue = matrix(index, index);
    inverse_eigenvalues[index] =
        fabs(eigenvalue) < 1.0e-6f
        ? fabs(eigenvalue)
        : fabs(1.0f / eigenvalue);
  }
  bool finite_inverse = true;
  for (uint row = 0; row < Size; ++row) {
    for (uint col = row; col < Size; ++col) {
      float value = 0.0f;
      for (uint inner = 0; inner < Size; ++inner) {
        value += eigenvectors(row, inner)
          * inverse_eigenvalues[inner]
          * eigenvectors(col, inner);
      }
      output[row * Size + col] = value;
      output[col * Size + row] = value;
      finite_inverse = finite_inverse && isfinite(value);
    }
  }
  if (!finite_inverse) {
    for (uint row = 0; row < Size; ++row) {
      for (uint col = 0; col < Size; ++col) {
        output[row * Size + col] = row == col ? 1.0f : 0.0f;
      }
    }
  }
}

#endif

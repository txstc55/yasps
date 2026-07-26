#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "metalSolverExtension.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_metal_solver_ext, module) {
  module.doc() = "Compiled YASPS MLX/Metal solver primitives";
  module.def(
      "solve_pcg",
      &yasps::metal::solve_pcg,
      "max_iterations"_a,
      "threshold"_a,
      "block_values"_a,
      "block_positions"_a,
      "block_value_starts"_a,
      "block_counts"_a,
      "block_dimensions"_a,
      "dynamic_values"_a,
      "dynamic_positions"_a,
      "dynamic_value_starts"_a,
      "dynamic_counts"_a,
      "dynamic_dimensions"_a,
      "inverse_blocks"_a,
      "diagonal_starts"_a,
      "diagonal_counts"_a,
      "diagonal_sizes"_a,
      "gradient_starts"_a,
      "gradient"_a,
      "p1_b"_a,
      "residual"_a,
      "direction"_a,
      "product"_a,
      "preconditioned"_a,
      "solution"_a,
      "initial_guess"_a,
      "source"_a,
      "library_name"_a,
      nb::kw_only(),
      "trace"_a = false,
      "stream"_a = nb::none());
}

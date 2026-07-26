#include "metalSolverExtension.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/transforms.h"

namespace yasps::metal {

namespace {

constexpr size_t kVectorThreads = 256;
constexpr size_t kBlockThreads = 32;
constexpr size_t kReductionThreads = 256;
constexpr size_t kReductionWidth = 2 * kReductionThreads;

void require_float32_vector(const mx::array& value, const char* name) {
  if (value.dtype() != mx::float32 || value.ndim() != 1 ||
      !value.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(name) + " must be a contiguous float32 vector");
  }
}

void require_uint32_vector(const mx::array& value, const char* name) {
  if (value.dtype() != mx::uint32 || value.ndim() != 1 ||
      !value.flags().row_contiguous) {
    throw std::invalid_argument(
        std::string(name) + " must be a contiguous uint32 vector");
  }
}

void require_same_size(
    const mx::array& value,
    const mx::array& reference,
    const char* name) {
  if (value.size() != reference.size()) {
    throw std::invalid_argument(
        std::string(name) + " must match the gradient size");
  }
}

void require_metadata(
    const std::vector<uint32_t>& starts,
    const std::vector<uint32_t>& counts,
    const std::vector<uint32_t>& dimensions,
    const char* name) {
  if (starts.size() < counts.size() ||
      dimensions.size() != 2 * counts.size()) {
    throw std::invalid_argument(
        std::string(name) + " block metadata has inconsistent lengths");
  }
}

mx::array allocate_like(const mx::array& reference) {
  return mx::array(
      mx::allocator::malloc(reference.nbytes()),
      reference.shape(),
      reference.dtype());
}

mx::array allocate_float_vector(size_t size) {
  return mx::array(
      mx::allocator::malloc(size * sizeof(float)),
      mx::Shape{static_cast<mx::ShapeElem>(size)},
      mx::float32);
}

void dispatch_vector(
    mx::metal::CommandEncoder& encoder,
    MTL::ComputePipelineState* kernel,
    const mx::array& left,
    const mx::array* right,
    mx::array& output,
    float scale = 0.0f) {
  const uint32_t count = static_cast<uint32_t>(output.size());
  if (count == 0) {
    return;
  }
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(left, 0);
  int output_index = 1;
  if (right != nullptr) {
    encoder.set_input_array(*right, 1);
    encoder.set_bytes(scale, 2);
    output_index = 3;
  }
  encoder.set_output_array(output, output_index);
  encoder.set_bytes(count, output_index + 1);
  encoder.dispatch_threads(
      MTL::Size(count, 1, 1),
      MTL::Size(std::min<size_t>(count, kVectorThreads), 1, 1));
}

void dispatch_clear(
    mx::metal::CommandEncoder& encoder,
    MTL::ComputePipelineState* kernel,
    mx::array& output) {
  const uint32_t count = static_cast<uint32_t>(output.size());
  if (count == 0) {
    return;
  }
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(output, 0);
  encoder.set_bytes(count, 1);
  encoder.dispatch_threads(
      MTL::Size(count, 1, 1),
      MTL::Size(std::min<size_t>(count, kVectorThreads), 1, 1));
}

std::string dimension_kernel_name(
    const char* prefix,
    uint32_t rows,
    uint32_t columns = 0) {
  std::ostringstream name;
  name << prefix << rows;
  if (columns != 0) {
    name << "x" << columns;
  }
  return name.str();
}

}  // namespace

std::tuple<mx::array, int, float> solve_pcg(
    uint32_t max_iterations,
    float threshold,
    const mx::array& block_values,
    const mx::array& block_positions,
    const std::vector<uint32_t>& block_value_starts,
    const std::vector<uint32_t>& block_counts,
    const std::vector<uint32_t>& block_dimensions,
    const mx::array& dynamic_values,
    const mx::array& dynamic_positions,
    const std::vector<uint32_t>& dynamic_value_starts,
    const std::vector<uint32_t>& dynamic_counts,
    const std::vector<uint32_t>& dynamic_dimensions,
    const mx::array& inverse_blocks,
    const std::vector<uint32_t>& diagonal_starts,
    const std::vector<uint32_t>& diagonal_counts,
    const std::vector<uint32_t>& diagonal_sizes,
    const std::vector<uint32_t>& gradient_starts,
    const mx::array& gradient,
    mx::array p1_b,
    mx::array residual,
    mx::array direction,
    mx::array product,
    mx::array preconditioned,
    mx::array solution,
    const mx::array& initial_guess,
    const std::string& source,
    const std::string& library_name,
    bool trace,
    mx::StreamOrDevice stream) {
  require_float32_vector(block_values, "block_values");
  require_uint32_vector(block_positions, "block_positions");
  require_float32_vector(dynamic_values, "dynamic_values");
  require_uint32_vector(dynamic_positions, "dynamic_positions");
  require_float32_vector(inverse_blocks, "inverse_blocks");
  require_float32_vector(gradient, "gradient");
  require_float32_vector(initial_guess, "initial_guess");
  require_float32_vector(p1_b, "p1_b");
  require_float32_vector(residual, "residual");
  require_float32_vector(direction, "direction");
  require_float32_vector(product, "product");
  require_float32_vector(preconditioned, "preconditioned");
  require_float32_vector(solution, "solution");
  require_same_size(initial_guess, gradient, "initial_guess");
  require_same_size(p1_b, gradient, "p1_b");
  require_same_size(residual, gradient, "residual");
  require_same_size(direction, gradient, "direction");
  require_same_size(product, gradient, "product");
  require_same_size(preconditioned, gradient, "preconditioned");
  require_same_size(solution, gradient, "solution");
  require_metadata(
      block_value_starts, block_counts, block_dimensions, "static");
  require_metadata(
      dynamic_value_starts,
      dynamic_counts,
      dynamic_dimensions,
      "dynamic");
  if (diagonal_counts.size() != diagonal_sizes.size() ||
      diagonal_starts.size() < diagonal_counts.size() ||
      gradient_starts.size() < diagonal_counts.size()) {
    throw std::invalid_argument(
        "diagonal block metadata has inconsistent lengths");
  }
  if (gradient.size() == 0) {
    return {solution, 0, 0.0f};
  }
  if (gradient.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("Metal PCG vectors exceed uint32 indexing");
  }

  mx::eval(std::vector<mx::array>{
      block_values,
      block_positions,
      dynamic_values,
      dynamic_positions,
      inverse_blocks,
      gradient,
      p1_b,
      residual,
      direction,
      product,
      preconditioned,
      solution,
      initial_guess,
  });

  const auto selected_stream = mx::to_stream(stream);
  auto& metal_device = mx::metal::device(selected_stream.device);
  mx::CompileOptions compile_options;
  compile_options.math_mode = mx::MathMode::Fast;
  auto* library = metal_device.get_library(
      library_name, compile_options, [&source]() { return source; });
  auto& encoder = mx::metal::get_command_encoder(selected_stream);

  auto* clear_kernel = metal_device.get_kernel(
      "yasps_solver_clear", library);
  auto* copy_kernel = metal_device.get_kernel(
      "yasps_solver_copy", library);
  auto* combine_kernel = metal_device.get_kernel(
      "yasps_solver_combine", library);
  auto* dot_kernel = metal_device.get_kernel(
      "yasps_solver_dot_first", library);
  auto* reduce_kernel = metal_device.get_kernel(
      "yasps_solver_reduce_pairs", library);

  std::unordered_map<uint64_t, MTL::ComputePipelineState*> spmv_kernels;
  auto register_spmv_kernels =
      [&](const std::vector<uint32_t>& dimensions) {
        for (size_t i = 0; i < dimensions.size(); i += 2) {
          const uint32_t rows = dimensions[i];
          const uint32_t columns = dimensions[i + 1];
          const uint64_t key =
              (static_cast<uint64_t>(rows) << 32) | columns;
          if (spmv_kernels.find(key) == spmv_kernels.end()) {
            spmv_kernels.emplace(
                key,
                metal_device.get_kernel(
                    dimension_kernel_name(
                        "yasps_solver_spmv_", rows, columns),
                    library));
          }
        }
      };
  register_spmv_kernels(block_dimensions);
  register_spmv_kernels(dynamic_dimensions);

  std::unordered_map<uint32_t, MTL::ComputePipelineState*>
      preconditioner_kernels;
  for (uint32_t size : diagonal_sizes) {
    if (preconditioner_kernels.find(size) ==
        preconditioner_kernels.end()) {
      preconditioner_kernels.emplace(
          size,
          metal_device.get_kernel(
              dimension_kernel_name(
                  "yasps_solver_precondition_", size),
              library));
    }
  }

  const size_t maximum_reduction_groups =
      (gradient.size() + kReductionWidth - 1) / kReductionWidth;
  auto reduction_hi_a = allocate_float_vector(maximum_reduction_groups);
  auto reduction_lo_a = allocate_float_vector(maximum_reduction_groups);
  auto reduction_hi_b = allocate_float_vector(maximum_reduction_groups);
  auto reduction_lo_b = allocate_float_vector(maximum_reduction_groups);
  auto best_solution = allocate_like(solution);

  auto copy = [&](const mx::array& input, mx::array& output) {
    dispatch_vector(encoder, copy_kernel, input, nullptr, output);
  };
  auto combine = [&](
                     const mx::array& left,
                     const mx::array& right,
                     float scale,
                     mx::array& output) {
    dispatch_vector(
        encoder, combine_kernel, left, &right, output, scale);
  };

  auto dispatch_spmv_family = [&](
                                  const mx::array& values,
                                  const mx::array& positions,
                                  const std::vector<uint32_t>& starts,
                                  const std::vector<uint32_t>& counts,
                                  const std::vector<uint32_t>& dimensions,
                                  const mx::array& input,
                                  mx::array& output) {
    uint32_t position_start = 0;
    for (size_t block_group = 0; block_group < counts.size();
         ++block_group) {
      const uint32_t count = counts[block_group];
      if (count == 0) {
        continue;
      }
      const uint32_t rows = dimensions[2 * block_group];
      const uint32_t columns = dimensions[2 * block_group + 1];
      const uint64_t key =
          (static_cast<uint64_t>(rows) << 32) | columns;
      auto* kernel = spmv_kernels.at(key);
      encoder.set_compute_pipeline_state(kernel);
      encoder.set_input_array(values, 0);
      encoder.set_input_array(positions, 1);
      encoder.set_input_array(input, 2);
      encoder.set_output_array(output, 3);
      encoder.set_bytes(starts[block_group], 4);
      encoder.set_bytes(position_start, 5);
      encoder.set_bytes(count, 6);
      const size_t groups =
          (static_cast<size_t>(count) + kBlockThreads - 1) /
          kBlockThreads;
      encoder.dispatch_threads(
          MTL::Size(groups * kBlockThreads, 1, 1),
          MTL::Size(kBlockThreads, 1, 1));
      position_start += count;
    }
  };

  auto spmv = [&](const mx::array& input, mx::array& output) {
    dispatch_clear(encoder, clear_kernel, output);
    {
      auto concurrent = encoder.start_concurrent();
      dispatch_spmv_family(
          block_values,
          block_positions,
          block_value_starts,
          block_counts,
          block_dimensions,
          input,
          output);
      dispatch_spmv_family(
          dynamic_values,
          dynamic_positions,
          dynamic_value_starts,
          dynamic_counts,
          dynamic_dimensions,
          input,
          output);
    }
  };

  auto precondition = [&](const mx::array& input, mx::array& output) {
    dispatch_clear(encoder, clear_kernel, output);
    auto concurrent = encoder.start_concurrent();
    for (size_t attribute = 0; attribute < diagonal_counts.size();
         ++attribute) {
      const uint32_t count = diagonal_counts[attribute];
      if (count == 0) {
        continue;
      }
      auto* kernel =
          preconditioner_kernels.at(diagonal_sizes[attribute]);
      encoder.set_compute_pipeline_state(kernel);
      encoder.set_input_array(inverse_blocks, 0);
      encoder.set_input_array(input, 1);
      encoder.set_output_array(output, 2);
      encoder.set_bytes(diagonal_starts[attribute], 3);
      encoder.set_bytes(gradient_starts[attribute], 4);
      encoder.set_bytes(count, 5);
      encoder.dispatch_threads(
          MTL::Size(count, 1, 1),
          MTL::Size(
              std::min<size_t>(count, kVectorThreads), 1, 1));
    }
  };

  auto dot = [&](const mx::array& left, const mx::array& right) {
    uint32_t count = static_cast<uint32_t>(left.size());
    size_t groups =
        (static_cast<size_t>(count) + kReductionWidth - 1) /
        kReductionWidth;
    encoder.set_compute_pipeline_state(dot_kernel);
    encoder.set_input_array(left, 0);
    encoder.set_input_array(right, 1);
    encoder.set_output_array(reduction_hi_a, 2);
    encoder.set_output_array(reduction_lo_a, 3);
    encoder.set_bytes(count, 4);
    encoder.dispatch_threads(
        MTL::Size(groups * kReductionThreads, 1, 1),
        MTL::Size(kReductionThreads, 1, 1));

    mx::array* input_hi = &reduction_hi_a;
    mx::array* input_lo = &reduction_lo_a;
    mx::array* output_hi = &reduction_hi_b;
    mx::array* output_lo = &reduction_lo_b;
    count = static_cast<uint32_t>(groups);
    while (count > 1) {
      groups =
          (static_cast<size_t>(count) + kReductionWidth - 1) /
          kReductionWidth;
      encoder.set_compute_pipeline_state(reduce_kernel);
      encoder.set_input_array(*input_hi, 0);
      encoder.set_input_array(*input_lo, 1);
      encoder.set_output_array(*output_hi, 2);
      encoder.set_output_array(*output_lo, 3);
      encoder.set_bytes(count, 4);
      encoder.dispatch_threads(
          MTL::Size(groups * kReductionThreads, 1, 1),
          MTL::Size(kReductionThreads, 1, 1));
      count = static_cast<uint32_t>(groups);
      std::swap(input_hi, output_hi);
      std::swap(input_lo, output_lo);
    }
    encoder.synchronize();
    return input_hi->data<float>()[0] + input_lo->data<float>()[0];
  };

  float relative_tolerance = 0.0f;
  auto finish = [&](int status, float residual_value, bool restore_best) {
    if (restore_best) {
      copy(best_solution, solution);
    }
    encoder.synchronize();
    if (trace) {
      std::cout << "Metal PCG status " << status
                << ", preconditioned residual " << residual_value
                << ", threshold " << relative_tolerance << std::endl;
    }
    return std::make_tuple(solution, status, residual_value);
  };

  copy(initial_guess, solution);
  precondition(gradient, p1_b);
  const float delta_zero = dot(p1_b, gradient);
  spmv(initial_guess, residual);
  combine(gradient, residual, -1.0f, residual);
  precondition(residual, direction);
  float delta_new = dot(residual, direction);
  relative_tolerance = threshold * delta_zero;
  if (delta_new <= relative_tolerance) {
    return finish(0, delta_new, false);
  }

  float best_delta = delta_new;
  copy(solution, best_solution);
  uint32_t stagnant_restarts = 0;

  for (uint32_t iteration = 1; iteration <= max_iterations;
       ++iteration) {
    spmv(direction, product);
    float denominator = dot(direction, product);
    if (!std::isfinite(denominator) || denominator <= 0.0f) {
      spmv(solution, residual);
      combine(gradient, residual, -1.0f, residual);
      precondition(residual, direction);
      delta_new = dot(residual, direction);
      spmv(direction, product);
      denominator = dot(direction, product);
      if (std::isfinite(delta_new) && delta_new < best_delta) {
        best_delta = delta_new;
        copy(solution, best_solution);
      }
      if (!std::isfinite(denominator) || denominator <= 0.0f) {
        return finish(
            -static_cast<int>(iteration) - 4, best_delta, true);
      }
    }

    const float alpha = delta_new / denominator;
    combine(solution, direction, alpha, solution);
    combine(residual, product, -alpha, residual);
    precondition(residual, preconditioned);
    const float delta_old = delta_new;
    delta_new = dot(residual, preconditioned);

    if (trace && iteration % 512 == 0) {
      std::cout << "Metal PCG iteration " << iteration
                << ", recursive residual " << delta_new
                << ", best true residual " << best_delta << std::endl;
    }

    if (delta_new <= relative_tolerance) {
      spmv(solution, residual);
      combine(gradient, residual, -1.0f, residual);
      precondition(residual, preconditioned);
      delta_new = dot(residual, preconditioned);
      if (delta_new <= relative_tolerance) {
        return finish(
            static_cast<int>(iteration), delta_new, false);
      }
      if (std::isfinite(delta_new) && delta_new < best_delta) {
        best_delta = delta_new;
        copy(solution, best_solution);
      }
      copy(preconditioned, direction);
      continue;
    }

    if (iteration % 32 == 0) {
      spmv(solution, residual);
      combine(gradient, residual, -1.0f, residual);
      precondition(residual, preconditioned);
      delta_new = dot(residual, preconditioned);
      if (delta_new <= relative_tolerance) {
        return finish(
            static_cast<int>(iteration), delta_new, false);
      }
      const float previous_best = best_delta;
      if (std::isfinite(delta_new) && delta_new < best_delta) {
        best_delta = delta_new;
        copy(solution, best_solution);
      }
      if (std::isfinite(delta_new) &&
          delta_new < previous_best * 0.99f) {
        stagnant_restarts = 0;
      } else {
        ++stagnant_restarts;
      }
      if (!std::isfinite(delta_new) || stagnant_restarts >= 8) {
        return finish(
            -static_cast<int>(iteration) - 4, best_delta, true);
      }
      copy(preconditioned, direction);
    } else {
      combine(
          preconditioned,
          direction,
          delta_new / delta_old,
          direction);
    }
  }

  return finish(
      -static_cast<int>(max_iterations) - 5, best_delta, true);
}

}  // namespace yasps::metal

#include "metalSolverExtension.h"

#include <algorithm>
#include <chrono>
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
constexpr uint32_t kMaximumStagnantRestarts = 8;

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

mx::array allocate_int_vector(size_t size) {
  return mx::array(
      mx::allocator::malloc(size * sizeof(int32_t)),
      mx::Shape{static_cast<mx::ShapeElem>(size)},
      mx::int32);
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

  const auto materialization_started =
      std::chrono::steady_clock::now();
  if (trace) {
    auto eval_stage = [](
                          const char* name,
                          const std::vector<mx::array>& arrays) {
      const auto started = std::chrono::steady_clock::now();
      mx::eval(arrays);
      const auto finished = std::chrono::steady_clock::now();
      std::cout
          << "Metal PCG materialization " << name << " "
          << std::chrono::duration<double, std::milli>(
                 finished - started)
                 .count()
          << " ms" << std::endl;
    };
    eval_stage(
        "metadata",
        {block_positions, dynamic_positions});
    eval_stage(
        "Hessian/gradient",
        {block_values, dynamic_values, gradient});
    eval_stage("block inverse", {inverse_blocks});
    eval_stage(
        "workspace",
        {
            p1_b,
            residual,
            direction,
            product,
            preconditioned,
            solution,
            initial_guess,
        });
  } else {
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
  }
  const auto solve_started = std::chrono::steady_clock::now();

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
  auto* store_pair_kernel = metal_device.get_kernel(
      "yasps_solver_store_pair", library);
  auto* prepare_alpha_kernel = metal_device.get_kernel(
      "yasps_solver_prepare_alpha", library);
  auto* update_solution_residual_kernel = metal_device.get_kernel(
      "yasps_solver_update_solution_residual", library);
  auto* finish_iteration_kernel = metal_device.get_kernel(
      "yasps_solver_finish_iteration", library);
  auto* update_direction_kernel = metal_device.get_kernel(
      "yasps_solver_update_direction", library);

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
  auto state = allocate_float_vector(7);
  auto iteration_status = allocate_int_vector(1);

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

  auto encode_dot = [&](const mx::array& left, const mx::array& right) {
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
    return std::make_pair(input_hi, input_lo);
  };

  auto dot = [&](const mx::array& left, const mx::array& right) {
    const auto [input_hi, input_lo] = encode_dot(left, right);
    encoder.synchronize();
    return input_hi->data<float>()[0] + input_lo->data<float>()[0];
  };

  auto dot_to_state = [&](
                          const mx::array& left,
                          const mx::array& right,
                          uint32_t slot) {
    const auto [input_hi, input_lo] = encode_dot(left, right);
    encoder.set_compute_pipeline_state(store_pair_kernel);
    encoder.set_input_array(*input_hi, 0);
    encoder.set_input_array(*input_lo, 1);
    encoder.set_output_array(state, 2);
    encoder.set_bytes(slot, 3);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  };

  auto prepare_alpha = [&](uint32_t iteration) {
    encoder.set_compute_pipeline_state(prepare_alpha_kernel);
    encoder.set_output_array(state, 0);
    encoder.set_output_array(iteration_status, 1);
    encoder.set_bytes(iteration, 2);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  };

  auto update_solution_residual = [&]() {
    const uint32_t count = static_cast<uint32_t>(solution.size());
    encoder.set_compute_pipeline_state(update_solution_residual_kernel);
    encoder.set_output_array(solution, 0);
    encoder.set_input_array(direction, 1);
    encoder.set_output_array(residual, 2);
    encoder.set_input_array(product, 3);
    encoder.set_input_array(state, 4);
    encoder.set_input_array(iteration_status, 5);
    encoder.set_bytes(count, 6);
    encoder.dispatch_threads(
        MTL::Size(count, 1, 1),
        MTL::Size(std::min<size_t>(count, kVectorThreads), 1, 1));
  };

  auto finish_iteration = [&](uint32_t iteration) {
    encoder.set_compute_pipeline_state(finish_iteration_kernel);
    encoder.set_output_array(state, 0);
    encoder.set_output_array(iteration_status, 1);
    encoder.set_bytes(iteration, 2);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  };

  auto update_direction = [&]() {
    const uint32_t count = static_cast<uint32_t>(direction.size());
    encoder.set_compute_pipeline_state(update_direction_kernel);
    encoder.set_input_array(preconditioned, 0);
    encoder.set_output_array(direction, 1);
    encoder.set_input_array(state, 2);
    encoder.set_input_array(iteration_status, 3);
    encoder.set_bytes(count, 4);
    encoder.dispatch_threads(
        MTL::Size(count, 1, 1),
        MTL::Size(std::min<size_t>(count, kVectorThreads), 1, 1));
  };

  float relative_tolerance = 0.0f;
  auto finish = [&](
                    int status,
                    float residual_value,
                    bool restore_best,
                    const char* outcome = "") {
    if (restore_best) {
      copy(best_solution, solution);
    }
    encoder.synchronize();
    if (trace) {
      const auto finished = std::chrono::steady_clock::now();
      const double materialization_ms =
          std::chrono::duration<double, std::milli>(
              solve_started - materialization_started)
              .count();
      const double solve_ms =
          std::chrono::duration<double, std::milli>(
              finished - solve_started)
              .count();
      std::cout << "Metal PCG status " << status
                << ", preconditioned residual " << residual_value
                << ", threshold " << relative_tolerance
                << ", input materialization " << materialization_ms
                << " ms, compiled recurrence " << solve_ms << " ms"
                << outcome
                << std::endl;
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
  const float initial_delta = delta_new;
  relative_tolerance = threshold * delta_zero;
  if (delta_new <= relative_tolerance) {
    return finish(0, delta_new, false);
  }

  float best_delta = delta_new;
  copy(solution, best_solution);
  uint32_t stagnant_restarts = 0;
  state.data<float>()[0] = delta_zero;
  state.data<float>()[1] = relative_tolerance;
  state.data<float>()[2] = delta_new;
  iteration_status.data<int32_t>()[0] = 0;

  constexpr uint32_t recurrence_chunk = 32;
  uint32_t last_iteration = 0;
  while (last_iteration < max_iterations) {
    const uint32_t chunk_end = std::min(
        max_iterations, last_iteration + recurrence_chunk);
    for (uint32_t iteration = last_iteration + 1;
         iteration <= chunk_end;
         ++iteration) {
      spmv(direction, product);
      dot_to_state(direction, product, 3);
      prepare_alpha(iteration);
      update_solution_residual();
      precondition(residual, preconditioned);
      dot_to_state(residual, preconditioned, 2);
      finish_iteration(iteration);
      update_direction();
    }
    encoder.synchronize();
    const int chunk_status = iteration_status.data<int32_t>()[0];
    last_iteration = chunk_end;

    // Confirm convergence and bound float32 recurrence drift against b - A*x.
    spmv(solution, residual);
    combine(gradient, residual, -1.0f, residual);
    precondition(residual, preconditioned);
    delta_new = dot(residual, preconditioned);
    if (delta_new <= relative_tolerance) {
      const int converged_iteration =
          chunk_status > 0 ? chunk_status : static_cast<int>(chunk_end);
      return finish(converged_iteration, delta_new, false);
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
    if (!std::isfinite(delta_new)) {
      return finish(
          chunk_status < 0
              ? chunk_status
              : -static_cast<int>(chunk_end) - 4,
          best_delta,
          true);
    }
    if (stagnant_restarts >= kMaximumStagnantRestarts) {
      // CUDA's float64 recurrence can keep reducing tolerances that are below
      // the useful float32 floor.  Exact residual replacements show whether
      // Metal has instead reached a stable best iterate.  Treat that as a
      // numerical stopping condition only when the solve made real progress;
      // invalid arithmetic and non-positive curvature remain hard failures.
      const bool made_progress =
          std::isfinite(best_delta) && best_delta < initial_delta;
      return finish(
          made_progress
              ? static_cast<int>(chunk_end)
              : -static_cast<int>(chunk_end) - 4,
          best_delta,
          true,
          made_progress ? ", float32 residual floor" : "");
    }

    copy(preconditioned, direction);
    if (chunk_status < 0) {
      spmv(direction, product);
      const float denominator = dot(direction, product);
      if (!std::isfinite(denominator) || denominator <= 0.0f) {
        return finish(chunk_status, best_delta, true);
      }
    }
    state.data<float>()[2] = delta_new;
    iteration_status.data<int32_t>()[0] = 0;

    if (trace && last_iteration % 512 == 0) {
      std::cout << "Metal PCG iteration " << last_iteration
                << ", true residual " << delta_new
                << ", best true residual " << best_delta << std::endl;
    }
  }

  return finish(
      -static_cast<int>(max_iterations) - 5, best_delta, true);
}

}  // namespace yasps::metal

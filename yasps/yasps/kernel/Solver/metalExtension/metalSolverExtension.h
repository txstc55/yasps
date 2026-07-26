#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "mlx/ops.h"

namespace mx = mlx::core;

namespace yasps::metal {

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
    bool trace = false,
    mx::StreamOrDevice stream = {});

}  // namespace yasps::metal

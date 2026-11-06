#pragma once

// Internal header: Weight initialization schemes.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <random>

namespace brutal_mlp::detail {

[[nodiscard]] WeightInitialization resolve_weight_initialization(WeightInitialization initialization,
                                                                 Activation activation);

void initialize_weights(Vector& weights,
                        WeightInitialization initialization,
                        std::size_t fan_in,
                        std::size_t fan_out,
                        std::mt19937_64& rng);

} // namespace brutal_mlp::detail

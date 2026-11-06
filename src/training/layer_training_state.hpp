#pragma once

// Internal header: Per-layer optimizer state persisted in checkpoints.

#include "brutal_mlp/brutal_mlp.hpp"

namespace brutal_mlp::detail {

struct LayerTrainingState {
    Vector grad_weights;
    Vector grad_biases;
    Vector velocity_weights;
    Vector velocity_biases;
    Vector first_moment_weights;
    Vector first_moment_biases;
    Vector second_moment_weights;
    Vector second_moment_biases;
};

void validate_layer_training_state(const LayerTrainingState& state, const LayerParameters& layer);

} // namespace brutal_mlp::detail

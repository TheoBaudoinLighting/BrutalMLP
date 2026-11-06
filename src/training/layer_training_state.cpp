// Per-layer optimizer state persisted in checkpoints.

#include "training/layer_training_state.hpp"

#include "core/checks.hpp"

#include <cstddef>

namespace brutal_mlp::detail {

void validate_layer_training_state(const LayerTrainingState& state, const LayerParameters& layer) {
    const std::size_t weight_count = checked_product(layer.input_size, layer.output_size, "checkpoint weight");
    const std::size_t bias_count = layer.output_size;
    require(state.grad_weights.size() == weight_count, "checkpoint grad weight count mismatch");
    require(state.velocity_weights.size() == weight_count, "checkpoint velocity weight count mismatch");
    require(state.first_moment_weights.size() == weight_count, "checkpoint first moment weight count mismatch");
    require(state.second_moment_weights.size() == weight_count, "checkpoint second moment weight count mismatch");
    require(state.grad_biases.size() == bias_count, "checkpoint grad bias count mismatch");
    require(state.velocity_biases.size() == bias_count, "checkpoint velocity bias count mismatch");
    require(state.first_moment_biases.size() == bias_count, "checkpoint first moment bias count mismatch");
    require(state.second_moment_biases.size() == bias_count, "checkpoint second moment bias count mismatch");
}

} // namespace brutal_mlp::detail

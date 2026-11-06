#pragma once

// Internal header: Validation of optimizer, schedule, initialization, training and loss configs.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <string>

namespace brutal_mlp::detail {

void validate_optimizer(const OptimizerConfig& optimizer);

void validate_initialization_config(const InitializationConfig& initialization);

void validate_training_options(const TrainingOptions& options);

void validate_dropout_probability(Scalar probability, const std::string& name);

[[nodiscard]] bool is_classification_loss(Loss loss) noexcept;

void validate_loss_config(const LossConfig& loss, std::size_t output_size);

} // namespace brutal_mlp::detail

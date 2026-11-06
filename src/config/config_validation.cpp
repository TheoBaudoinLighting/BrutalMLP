// Validation of optimizer, schedule, initialization, training and loss configs.

#include "config/config_validation.hpp"

#include "core/checks.hpp"

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

namespace brutal_mlp::detail {

namespace {

void validate_learning_rate_schedule_config(const LearningRateScheduleConfig& schedule) {
    require_finite(schedule.base_learning_rate, "base_learning_rate");
    require(schedule.base_learning_rate >= Scalar{0}, "base_learning_rate must be non-negative");
    require_finite(schedule.minimum_learning_rate, "minimum_learning_rate");
    require(schedule.minimum_learning_rate >= Scalar{0}, "minimum_learning_rate must be non-negative");
    require_finite(schedule.warmup_start_learning_rate, "warmup_start_learning_rate");
    require(schedule.warmup_start_learning_rate >= Scalar{0},
            "warmup_start_learning_rate must be non-negative");
    require(schedule.step_size > 0, "step_size must be positive");
    require_finite(schedule.step_decay_factor, "step_decay_factor");
    require(schedule.step_decay_factor > Scalar{0} && schedule.step_decay_factor <= Scalar{1},
            "step_decay_factor must be in (0, 1]");
    require_finite(schedule.exponential_decay_rate, "exponential_decay_rate");
    require(schedule.exponential_decay_rate > Scalar{0} && schedule.exponential_decay_rate <= Scalar{1},
            "exponential_decay_rate must be in (0, 1]");
    require_finite(schedule.reduce_on_plateau_factor, "reduce_on_plateau_factor");
    require(schedule.reduce_on_plateau_factor > Scalar{0} && schedule.reduce_on_plateau_factor < Scalar{1},
            "reduce_on_plateau_factor must be in (0, 1)");
    require_finite(schedule.reduce_on_plateau_min_delta, "reduce_on_plateau_min_delta");
    require(schedule.reduce_on_plateau_min_delta >= Scalar{0},
            "reduce_on_plateau_min_delta must be non-negative");
}

} // namespace

void validate_optimizer(const OptimizerConfig& optimizer) {
    require_finite(optimizer.learning_rate, "learning_rate");
    require(optimizer.learning_rate > Scalar{0}, "learning_rate must be positive");
    require_finite(optimizer.beta1, "beta1");
    require_finite(optimizer.beta2, "beta2");
    require(optimizer.beta1 >= Scalar{0} && optimizer.beta1 < Scalar{1}, "beta1 must be in [0, 1)");
    require(optimizer.beta2 >= Scalar{0} && optimizer.beta2 < Scalar{1}, "beta2 must be in [0, 1)");
    require_finite(optimizer.epsilon, "epsilon");
    require(optimizer.epsilon > Scalar{0}, "epsilon must be positive");
    require_finite(optimizer.momentum, "momentum");
    require(optimizer.momentum >= Scalar{0} && optimizer.momentum < Scalar{1}, "momentum must be in [0, 1)");
    require_finite(optimizer.l2, "l2");
    require(optimizer.l2 >= Scalar{0}, "l2 must be non-negative");
    require_finite(optimizer.l1, "l1");
    require(optimizer.l1 >= Scalar{0}, "l1 must be non-negative");
    require_finite(optimizer.decoupled_weight_decay, "decoupled_weight_decay");
    require(optimizer.decoupled_weight_decay >= Scalar{0}, "decoupled_weight_decay must be non-negative");
    require_finite(optimizer.max_norm, "max_norm");
    require(optimizer.max_norm >= Scalar{0}, "max_norm must be non-negative");
    require_finite(optimizer.gradient_clip_norm, "gradient_clip_norm");
    require(optimizer.gradient_clip_norm >= Scalar{0}, "gradient_clip_norm must be non-negative");
    require_finite(optimizer.gradient_clip_value, "gradient_clip_value");
    require(optimizer.gradient_clip_value >= Scalar{0}, "gradient_clip_value must be non-negative");
    require_finite(optimizer.layer_gradient_clip_norm, "layer_gradient_clip_norm");
    require(optimizer.layer_gradient_clip_norm >= Scalar{0},
            "layer_gradient_clip_norm must be non-negative");
}

void validate_initialization_config(const InitializationConfig& initialization) {
    require_finite(initialization.bias, "initialization bias");
}

void validate_training_options(const TrainingOptions& options) {
    require(options.epochs > 0, "epochs must be positive");
    require(options.batch_size > 0, "batch_size must be positive");
    require_finite(options.validation_split, "validation_split");
    require(options.validation_split >= Scalar{0} && options.validation_split < Scalar{1},
            "validation_split must be in [0, 1)");
    require_finite(options.test_split, "test_split");
    require(options.test_split >= Scalar{0} && options.test_split < Scalar{1}, "test_split must be in [0, 1)");
    require(options.validation_split + options.test_split < Scalar{1},
            "validation_split + test_split must be less than 1");
    require_finite(options.min_delta, "min_delta");
    require(options.min_delta >= Scalar{0}, "min_delta must be non-negative");
    validate_learning_rate_schedule_config(options.learning_rate_schedule);
    require_finite(options.gradient_noise_stddev, "gradient_noise_stddev");
    require(options.gradient_noise_stddev >= Scalar{0}, "gradient_noise_stddev must be non-negative");
    require_finite(options.debug.loss_explosion_factor, "debug.loss_explosion_factor");
    require(options.debug.loss_explosion_factor >= Scalar{1},
            "debug.loss_explosion_factor must be at least 1");
    require_finite(options.debug.loss_explosion_threshold, "debug.loss_explosion_threshold");
    require(options.debug.loss_explosion_threshold > Scalar{0},
            "debug.loss_explosion_threshold must be positive");
    require_finite(options.debug.gradient_norm_explosion_factor, "debug.gradient_norm_explosion_factor");
    require(options.debug.gradient_norm_explosion_factor >= Scalar{1},
            "debug.gradient_norm_explosion_factor must be at least 1");
    require_finite(options.debug.gradient_norm_explosion_threshold, "debug.gradient_norm_explosion_threshold");
    require(options.debug.gradient_norm_explosion_threshold > Scalar{0},
            "debug.gradient_norm_explosion_threshold must be positive");
}

void validate_dropout_probability(Scalar probability, const std::string& name) {
    require_finite(probability, name);
    require(probability >= Scalar{0} && probability < Scalar{1}, name + " must be in [0, 1)");
}

[[nodiscard]] bool is_classification_loss(Loss loss) noexcept {
    return loss == Loss::binary_cross_entropy || loss == Loss::categorical_cross_entropy;
}

void validate_loss_config(const LossConfig& loss, std::size_t output_size) {
    require_finite(loss.huber_delta, "huber_delta");
    require(loss.huber_delta > Scalar{0}, "huber_delta must be positive");
    require_finite(loss.relative_epsilon, "relative_epsilon");
    require(loss.relative_epsilon > Scalar{0}, "relative_epsilon must be positive");

    for (Scalar weight : loss.weights) {
        require_finite(weight, "loss weight");
        require(weight >= Scalar{0}, "loss weights must be non-negative");
    }

    switch (loss.type) {
    case Loss::mean_squared_error:
    case Loss::mean_absolute_error:
    case Loss::huber:
    case Loss::relative_mean_squared_error:
    case Loss::relative_l2:
    case Loss::log_cosh:
    case Loss::binary_cross_entropy:
    case Loss::categorical_cross_entropy:
        require(loss.weights.empty(), "loss weights are only supported with weighted_mean_squared_error");
        return;
    case Loss::weighted_mean_squared_error: {
        require(loss.weights.size() == output_size, "weighted_mean_squared_error weight count mismatch");
        const Scalar total_weight = std::accumulate(loss.weights.begin(), loss.weights.end(), Scalar{0});
        require(total_weight > Scalar{0}, "weighted_mean_squared_error requires at least one positive weight");
        return;
    }
    case Loss::custom:
        require(loss.weights.empty(), "custom loss does not use built-in weights");
        require(loss.custom_loss.value != nullptr, "custom loss value callback is required");
        require(loss.custom_loss.gradient != nullptr, "custom loss gradient callback is required");
        return;
    }

    throw std::invalid_argument("unknown loss");
}

} // namespace brutal_mlp::detail

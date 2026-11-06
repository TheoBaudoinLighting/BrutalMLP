// Factory helpers of the public configuration structs.

#include "brutal_mlp/brutal_mlp.hpp"

#include "config/config_validation.hpp"

#include <cstddef>
#include <stdexcept>

namespace brutal_mlp {

using namespace detail;

LossConfig LossConfig::from_loss(Loss loss) {
    switch (loss) {
    case Loss::mean_squared_error:
        return mean_squared_error();
    case Loss::mean_absolute_error:
        return mean_absolute_error();
    case Loss::huber:
        return huber();
    case Loss::relative_mean_squared_error:
        return relative_mean_squared_error();
    case Loss::log_cosh:
        return log_cosh();
    case Loss::weighted_mean_squared_error:
        throw std::invalid_argument("weighted_mean_squared_error requires explicit weights");
    case Loss::binary_cross_entropy:
        return binary_cross_entropy();
    case Loss::categorical_cross_entropy:
        return categorical_cross_entropy();
    case Loss::custom:
        throw std::invalid_argument("custom loss requires callbacks");
    case Loss::relative_l2:
        return relative_l2();
    }

    throw std::invalid_argument("unknown loss");
}

LossConfig LossConfig::mean_squared_error() {
    LossConfig config;
    config.type = Loss::mean_squared_error;
    return config;
}

LossConfig LossConfig::mean_absolute_error() {
    LossConfig config;
    config.type = Loss::mean_absolute_error;
    return config;
}

LossConfig LossConfig::huber(Scalar delta) {
    LossConfig config;
    config.type = Loss::huber;
    config.huber_delta = delta;
    validate_loss_config(config, 0);
    return config;
}

LossConfig LossConfig::relative_mean_squared_error(Scalar epsilon) {
    LossConfig config;
    config.type = Loss::relative_mean_squared_error;
    config.relative_epsilon = epsilon;
    validate_loss_config(config, 0);
    return config;
}

LossConfig LossConfig::relative_l2(Scalar epsilon) {
    LossConfig config;
    config.type = Loss::relative_l2;
    config.relative_epsilon = epsilon;
    validate_loss_config(config, 0);
    return config;
}

LossConfig LossConfig::log_cosh() {
    LossConfig config;
    config.type = Loss::log_cosh;
    return config;
}

LossConfig LossConfig::weighted_mean_squared_error(const Vector& weights) {
    LossConfig config;
    config.type = Loss::weighted_mean_squared_error;
    config.weights = weights;
    validate_loss_config(config, weights.size());
    return config;
}

LossConfig LossConfig::binary_cross_entropy() {
    LossConfig config;
    config.type = Loss::binary_cross_entropy;
    return config;
}

LossConfig LossConfig::categorical_cross_entropy() {
    LossConfig config;
    config.type = Loss::categorical_cross_entropy;
    return config;
}

LossConfig LossConfig::custom(CustomLossValueFunction value,
                              CustomLossGradientFunction gradient,
                              void* context) {
    LossConfig config;
    config.type = Loss::custom;
    config.custom_loss.value = value;
    config.custom_loss.gradient = gradient;
    config.custom_loss.context = context;
    validate_loss_config(config, 0);
    return config;
}

OptimizerConfig OptimizerConfig::adam(Scalar learning_rate) {
    OptimizerConfig config;
    config.type = OptimizerType::adam;
    config.learning_rate = learning_rate;
    return config;
}

OptimizerConfig OptimizerConfig::adamw(Scalar learning_rate, Scalar weight_decay) {
    OptimizerConfig config;
    config.type = OptimizerType::adamw;
    config.learning_rate = learning_rate;
    config.decoupled_weight_decay = weight_decay;
    return config;
}

OptimizerConfig OptimizerConfig::sgd(Scalar learning_rate, Scalar momentum) {
    OptimizerConfig config;
    config.type = OptimizerType::sgd;
    config.learning_rate = learning_rate;
    config.momentum = momentum;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::constant() {
    return LearningRateScheduleConfig{};
}

LearningRateScheduleConfig LearningRateScheduleConfig::step_decay(std::size_t step_size,
                                                                  Scalar decay_factor) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::step_decay;
    config.step_size = step_size;
    config.step_decay_factor = decay_factor;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::exponential_decay(Scalar decay_rate) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::exponential_decay;
    config.exponential_decay_rate = decay_rate;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::cosine_annealing(std::size_t cosine_epochs,
                                                                        Scalar minimum_learning_rate) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::cosine_annealing;
    config.cosine_epochs = cosine_epochs;
    config.minimum_learning_rate = minimum_learning_rate;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::reduce_on_plateau(std::size_t patience,
                                                                         Scalar factor) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::reduce_on_plateau;
    config.reduce_on_plateau_patience = patience;
    config.reduce_on_plateau_factor = factor;
    return config;
}

InitializationConfig InitializationConfig::automatic(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::automatic;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::he_normal(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::he_normal;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::he_uniform(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::he_uniform;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::xavier_normal(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::xavier_normal;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::xavier_uniform(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::xavier_uniform;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::lecun_normal(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::lecun_normal;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::lecun_uniform(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::lecun_uniform;
    config.bias = bias;
    return config;
}

} // namespace brutal_mlp

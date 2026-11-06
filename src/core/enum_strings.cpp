// String conversions for public enums.

#include "brutal_mlp/brutal_mlp.hpp"

#include "math/activations.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace brutal_mlp {

using namespace detail;

std::string to_string(Activation activation) {
    switch (activation) {
    case Activation::linear:
        return "linear";
    case Activation::relu:
        return "relu";
    case Activation::sigmoid:
        return "sigmoid";
    case Activation::tanh:
        return "tanh";
    case Activation::softmax:
        return "softmax";
    }

    throw std::invalid_argument("unknown activation");
}

std::string to_string(Loss loss) {
    switch (loss) {
    case Loss::mean_squared_error:
        return "mean_squared_error";
    case Loss::mean_absolute_error:
        return "mean_absolute_error";
    case Loss::huber:
        return "huber";
    case Loss::relative_mean_squared_error:
        return "relative_mean_squared_error";
    case Loss::log_cosh:
        return "log_cosh";
    case Loss::weighted_mean_squared_error:
        return "weighted_mean_squared_error";
    case Loss::binary_cross_entropy:
        return "binary_cross_entropy";
    case Loss::categorical_cross_entropy:
        return "categorical_cross_entropy";
    case Loss::custom:
        return "custom";
    case Loss::relative_l2:
        return "relative_l2";
    }

    throw std::invalid_argument("unknown loss");
}

std::string to_string(Metric metric) {
    switch (metric) {
    case Metric::mean_squared_error:
        return "mean_squared_error";
    case Metric::mean_absolute_error:
        return "mean_absolute_error";
    case Metric::root_mean_squared_error:
        return "root_mean_squared_error";
    case Metric::r2_score:
        return "r2_score";
    case Metric::accuracy:
        return "accuracy";
    case Metric::precision:
        return "precision";
    case Metric::recall:
        return "recall";
    case Metric::f1_score:
        return "f1_score";
    case Metric::confusion_matrix:
        return "confusion_matrix";
    case Metric::custom:
        return "custom";
    }

    throw std::invalid_argument("unknown metric");
}

std::string to_string(Averaging averaging) {
    switch (averaging) {
    case Averaging::binary:
        return "binary";
    case Averaging::macro:
        return "macro";
    case Averaging::micro:
        return "micro";
    }

    throw std::invalid_argument("unknown averaging");
}

std::string to_string(OptimizerType optimizer) {
    switch (optimizer) {
    case OptimizerType::sgd:
        return "sgd";
    case OptimizerType::adam:
        return "adam";
    case OptimizerType::adamw:
        return "adamw";
    }

    throw std::invalid_argument("unknown optimizer");
}

std::string to_string(NormalizationMode mode) {
    switch (mode) {
    case NormalizationMode::none:
        return "none";
    case NormalizationMode::standard_score:
        return "standard_score";
    case NormalizationMode::min_max:
        return "min_max";
    }

    throw std::invalid_argument("unknown normalization mode");
}

std::string to_string(InferenceStatus status) {
    switch (status) {
    case InferenceStatus::ok:
        return "ok";
    case InferenceStatus::null_input:
        return "null_input";
    case InferenceStatus::null_output:
        return "null_output";
    case InferenceStatus::invalid_input_size:
        return "invalid_input_size";
    case InferenceStatus::invalid_output_size:
        return "invalid_output_size";
    case InferenceStatus::invalid_input_stride:
        return "invalid_input_stride";
    case InferenceStatus::invalid_output_stride:
        return "invalid_output_stride";
    case InferenceStatus::null_scratch:
        return "null_scratch";
    case InferenceStatus::insufficient_scratch:
        return "insufficient_scratch";
    }

    throw std::invalid_argument("unknown inference status");
}

std::string to_string(TrainingStopReason reason) {
    switch (reason) {
    case TrainingStopReason::completed_epochs:
        return "completed_epochs";
    case TrainingStopReason::early_stopping:
        return "early_stopping";
    case TrainingStopReason::invalid_training_data:
        return "invalid_training_data";
    case TrainingStopReason::non_finite_loss:
        return "non_finite_loss";
    case TrainingStopReason::non_finite_gradient:
        return "non_finite_gradient";
    case TrainingStopReason::non_finite_weights:
        return "non_finite_weights";
    case TrainingStopReason::loss_explosion:
        return "loss_explosion";
    case TrainingStopReason::gradient_explosion:
        return "gradient_explosion";
    }

    throw std::invalid_argument("unknown training stop reason");
}

std::string to_string(TrainingMonitor monitor) {
    switch (monitor) {
    case TrainingMonitor::automatic:
        return "automatic";
    case TrainingMonitor::training_loss:
        return "training_loss";
    case TrainingMonitor::validation_loss:
        return "validation_loss";
    case TrainingMonitor::test_loss:
        return "test_loss";
    }

    throw std::invalid_argument("unknown training monitor");
}

std::string to_string(TrainingMonitorMode mode) {
    switch (mode) {
    case TrainingMonitorMode::minimize:
        return "minimize";
    case TrainingMonitorMode::maximize:
        return "maximize";
    }

    throw std::invalid_argument("unknown training monitor mode");
}

std::string to_string(LearningRateSchedule schedule) {
    switch (schedule) {
    case LearningRateSchedule::constant:
        return "constant";
    case LearningRateSchedule::step_decay:
        return "step_decay";
    case LearningRateSchedule::exponential_decay:
        return "exponential_decay";
    case LearningRateSchedule::cosine_annealing:
        return "cosine_annealing";
    case LearningRateSchedule::reduce_on_plateau:
        return "reduce_on_plateau";
    }

    throw std::invalid_argument("unknown learning rate schedule");
}

std::string to_string(WeightInitialization initialization) {
    switch (initialization) {
    case WeightInitialization::automatic:
        return "automatic";
    case WeightInitialization::he_normal:
        return "he_normal";
    case WeightInitialization::he_uniform:
        return "he_uniform";
    case WeightInitialization::xavier_normal:
        return "xavier_normal";
    case WeightInitialization::xavier_uniform:
        return "xavier_uniform";
    case WeightInitialization::lecun_normal:
        return "lecun_normal";
    case WeightInitialization::lecun_uniform:
        return "lecun_uniform";
    }

    throw std::invalid_argument("unknown weight initialization");
}

std::string to_string(ParallelExecution execution) {
    switch (execution) {
    case ParallelExecution::serial:
        return "serial";
    case ParallelExecution::worker_threads:
        return "worker_threads";
    }

    throw std::invalid_argument("unknown parallel execution");
}

std::string to_string(InferenceBackend backend) {
    switch (backend) {
    case InferenceBackend::scalar:
        return "scalar";
    case InferenceBackend::ispc:
        return "ispc";
    }

    throw std::invalid_argument("unknown inference backend");
}

Activation activation_from_string(std::string_view value) {
    if (value == "linear") {
        return Activation::linear;
    }
    if (value == "relu") {
        return Activation::relu;
    }
    if (value == "sigmoid") {
        return Activation::sigmoid;
    }
    if (value == "tanh") {
        return Activation::tanh;
    }
    if (value == "softmax") {
        return Activation::softmax;
    }
    throw std::invalid_argument("unknown activation: " + std::string(value));
}

Loss loss_from_string(std::string_view value) {
    if (value == "mean_squared_error") {
        return Loss::mean_squared_error;
    }
    if (value == "mean_absolute_error") {
        return Loss::mean_absolute_error;
    }
    if (value == "huber") {
        return Loss::huber;
    }
    if (value == "relative_mean_squared_error") {
        return Loss::relative_mean_squared_error;
    }
    if (value == "relative_l2") {
        return Loss::relative_l2;
    }
    if (value == "log_cosh") {
        return Loss::log_cosh;
    }
    if (value == "weighted_mean_squared_error") {
        return Loss::weighted_mean_squared_error;
    }
    if (value == "binary_cross_entropy") {
        return Loss::binary_cross_entropy;
    }
    if (value == "categorical_cross_entropy") {
        return Loss::categorical_cross_entropy;
    }
    if (value == "custom") {
        return Loss::custom;
    }
    throw std::invalid_argument("unknown loss: " + std::string(value));
}

Metric metric_from_string(std::string_view value) {
    if (value == "mean_squared_error") {
        return Metric::mean_squared_error;
    }
    if (value == "mean_absolute_error") {
        return Metric::mean_absolute_error;
    }
    if (value == "root_mean_squared_error") {
        return Metric::root_mean_squared_error;
    }
    if (value == "r2_score") {
        return Metric::r2_score;
    }
    if (value == "accuracy") {
        return Metric::accuracy;
    }
    if (value == "precision") {
        return Metric::precision;
    }
    if (value == "recall") {
        return Metric::recall;
    }
    if (value == "f1_score") {
        return Metric::f1_score;
    }
    if (value == "confusion_matrix") {
        return Metric::confusion_matrix;
    }
    if (value == "custom") {
        return Metric::custom;
    }
    throw std::invalid_argument("unknown metric: " + std::string(value));
}

Averaging averaging_from_string(std::string_view value) {
    if (value == "binary") {
        return Averaging::binary;
    }
    if (value == "macro") {
        return Averaging::macro;
    }
    if (value == "micro") {
        return Averaging::micro;
    }
    throw std::invalid_argument("unknown averaging: " + std::string(value));
}

OptimizerType optimizer_type_from_string(std::string_view value) {
    if (value == "sgd") {
        return OptimizerType::sgd;
    }
    if (value == "adam") {
        return OptimizerType::adam;
    }
    if (value == "adamw") {
        return OptimizerType::adamw;
    }
    throw std::invalid_argument("unknown optimizer: " + std::string(value));
}

NormalizationMode normalization_mode_from_string(std::string_view value) {
    if (value == "none") {
        return NormalizationMode::none;
    }
    if (value == "standard_score") {
        return NormalizationMode::standard_score;
    }
    if (value == "min_max") {
        return NormalizationMode::min_max;
    }
    throw std::invalid_argument("unknown normalization mode: " + std::string(value));
}

ParallelExecution parallel_execution_from_string(std::string_view value) {
    if (value == "serial") {
        return ParallelExecution::serial;
    }
    if (value == "worker_threads") {
        return ParallelExecution::worker_threads;
    }
    throw std::invalid_argument("unknown parallel execution: " + std::string(value));
}

InferenceBackend inference_backend_from_string(std::string_view value) {
    if (value == "scalar") {
        return InferenceBackend::scalar;
    }
    if (value == "ispc") {
        return InferenceBackend::ispc;
    }
    throw std::invalid_argument("unknown inference backend: " + std::string(value));
}

} // namespace brutal_mlp

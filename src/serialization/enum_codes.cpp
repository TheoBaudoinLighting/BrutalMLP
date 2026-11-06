// Stable binary codes of serialized enums.

#include "serialization/enum_codes.hpp"

#include "math/activations.hpp"

#include <cstdint>
#include <stdexcept>

namespace brutal_mlp::detail {

[[nodiscard]] BinaryScalarType current_binary_scalar_type() noexcept {
#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
    return BinaryScalarType::float64;
#else
    return BinaryScalarType::float32;
#endif
}

[[nodiscard]] std::uint32_t scalar_type_code(BinaryScalarType type) {
    switch (type) {
    case BinaryScalarType::float32:
        return 1;
    case BinaryScalarType::float64:
        return 2;
    }
    throw std::invalid_argument("unknown binary scalar type");
}

[[nodiscard]] BinaryScalarType scalar_type_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return BinaryScalarType::float32;
    case 2:
        return BinaryScalarType::float64;
    }
    throw std::runtime_error("unsupported binary scalar type");
}

[[nodiscard]] std::uint32_t model_kind_code(BinaryModelKind kind) {
    switch (kind) {
    case BinaryModelKind::training:
        return 1;
    case BinaryModelKind::inference:
        return 2;
    case BinaryModelKind::checkpoint:
        return 3;
    }
    throw std::invalid_argument("unknown binary model kind");
}

[[nodiscard]] BinaryModelKind model_kind_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return BinaryModelKind::training;
    case 2:
        return BinaryModelKind::inference;
    case 3:
        return BinaryModelKind::checkpoint;
    }
    throw std::runtime_error("unsupported binary model kind");
}

[[nodiscard]] std::uint32_t activation_code(Activation activation) {
    switch (activation) {
    case Activation::linear:
        return 1;
    case Activation::relu:
        return 2;
    case Activation::sigmoid:
        return 3;
    case Activation::tanh:
        return 4;
    case Activation::softmax:
        return 5;
    }
    throw std::invalid_argument("unknown activation");
}

[[nodiscard]] Activation activation_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return Activation::linear;
    case 2:
        return Activation::relu;
    case 3:
        return Activation::sigmoid;
    case 4:
        return Activation::tanh;
    case 5:
        return Activation::softmax;
    }
    throw std::runtime_error("unsupported activation code");
}

[[nodiscard]] std::uint32_t loss_code(Loss loss) {
    switch (loss) {
    case Loss::mean_squared_error:
        return 1;
    case Loss::mean_absolute_error:
        return 2;
    case Loss::huber:
        return 3;
    case Loss::relative_mean_squared_error:
        return 4;
    case Loss::log_cosh:
        return 5;
    case Loss::weighted_mean_squared_error:
        return 6;
    case Loss::binary_cross_entropy:
        return 7;
    case Loss::categorical_cross_entropy:
        return 8;
    case Loss::custom:
        return 9;
    case Loss::relative_l2:
        return 10;
    }
    throw std::invalid_argument("unknown loss");
}

[[nodiscard]] Loss loss_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return Loss::mean_squared_error;
    case 2:
        return Loss::mean_absolute_error;
    case 3:
        return Loss::huber;
    case 4:
        return Loss::relative_mean_squared_error;
    case 5:
        return Loss::log_cosh;
    case 6:
        return Loss::weighted_mean_squared_error;
    case 7:
        return Loss::binary_cross_entropy;
    case 8:
        return Loss::categorical_cross_entropy;
    case 9:
        return Loss::custom;
    case 10:
        return Loss::relative_l2;
    }
    throw std::runtime_error("unsupported loss code");
}

[[nodiscard]] std::uint32_t optimizer_code(OptimizerType optimizer) {
    switch (optimizer) {
    case OptimizerType::sgd:
        return 1;
    case OptimizerType::adam:
        return 2;
    case OptimizerType::adamw:
        return 3;
    }
    throw std::invalid_argument("unknown optimizer");
}

[[nodiscard]] OptimizerType optimizer_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return OptimizerType::sgd;
    case 2:
        return OptimizerType::adam;
    case 3:
        return OptimizerType::adamw;
    }
    throw std::runtime_error("unsupported optimizer code");
}

[[nodiscard]] std::uint32_t learning_rate_schedule_code(LearningRateSchedule schedule) {
    switch (schedule) {
    case LearningRateSchedule::constant:
        return 1;
    case LearningRateSchedule::step_decay:
        return 2;
    case LearningRateSchedule::exponential_decay:
        return 3;
    case LearningRateSchedule::cosine_annealing:
        return 4;
    case LearningRateSchedule::reduce_on_plateau:
        return 5;
    }
    throw std::invalid_argument("unknown learning rate schedule");
}

[[nodiscard]] LearningRateSchedule learning_rate_schedule_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return LearningRateSchedule::constant;
    case 2:
        return LearningRateSchedule::step_decay;
    case 3:
        return LearningRateSchedule::exponential_decay;
    case 4:
        return LearningRateSchedule::cosine_annealing;
    case 5:
        return LearningRateSchedule::reduce_on_plateau;
    }
    throw std::runtime_error("unsupported learning rate schedule code");
}

[[nodiscard]] std::uint32_t normalization_mode_code(NormalizationMode mode) {
    switch (mode) {
    case NormalizationMode::none:
        return 1;
    case NormalizationMode::standard_score:
        return 2;
    case NormalizationMode::min_max:
        return 3;
    }
    throw std::invalid_argument("unknown normalization mode");
}

[[nodiscard]] NormalizationMode normalization_mode_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return NormalizationMode::none;
    case 2:
        return NormalizationMode::standard_score;
    case 3:
        return NormalizationMode::min_max;
    }
    throw std::runtime_error("unsupported normalization mode code");
}

[[nodiscard]] std::uint32_t training_monitor_code(TrainingMonitor monitor) {
    switch (monitor) {
    case TrainingMonitor::automatic:
        return 1;
    case TrainingMonitor::training_loss:
        return 2;
    case TrainingMonitor::validation_loss:
        return 3;
    case TrainingMonitor::test_loss:
        return 4;
    }
    throw std::invalid_argument("unknown training monitor");
}

[[nodiscard]] TrainingMonitor training_monitor_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return TrainingMonitor::automatic;
    case 2:
        return TrainingMonitor::training_loss;
    case 3:
        return TrainingMonitor::validation_loss;
    case 4:
        return TrainingMonitor::test_loss;
    }
    throw std::runtime_error("unsupported training monitor code");
}

[[nodiscard]] std::uint32_t training_monitor_mode_code(TrainingMonitorMode mode) {
    switch (mode) {
    case TrainingMonitorMode::minimize:
        return 1;
    case TrainingMonitorMode::maximize:
        return 2;
    }
    throw std::invalid_argument("unknown training monitor mode");
}

[[nodiscard]] TrainingMonitorMode training_monitor_mode_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return TrainingMonitorMode::minimize;
    case 2:
        return TrainingMonitorMode::maximize;
    }
    throw std::runtime_error("unsupported training monitor mode code");
}

[[nodiscard]] std::uint32_t parallel_execution_code(ParallelExecution execution) {
    switch (execution) {
    case ParallelExecution::serial:
        return 1;
    case ParallelExecution::worker_threads:
        return 2;
    }
    throw std::invalid_argument("unknown parallel execution");
}

[[nodiscard]] ParallelExecution parallel_execution_from_code(std::uint32_t code) {
    switch (code) {
    case 1:
        return ParallelExecution::serial;
    case 2:
        return ParallelExecution::worker_threads;
    }
    throw std::runtime_error("unsupported parallel execution code");
}

} // namespace brutal_mlp::detail

#pragma once

// Internal header: Stable binary codes of serialized enums.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstdint>

namespace brutal_mlp::detail {

[[nodiscard]] BinaryScalarType current_binary_scalar_type() noexcept;

[[nodiscard]] std::uint32_t scalar_type_code(BinaryScalarType type);

[[nodiscard]] BinaryScalarType scalar_type_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t model_kind_code(BinaryModelKind kind);

[[nodiscard]] BinaryModelKind model_kind_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t activation_code(Activation activation);

[[nodiscard]] Activation activation_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t loss_code(Loss loss);

[[nodiscard]] Loss loss_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t optimizer_code(OptimizerType optimizer);

[[nodiscard]] OptimizerType optimizer_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t learning_rate_schedule_code(LearningRateSchedule schedule);

[[nodiscard]] LearningRateSchedule learning_rate_schedule_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t normalization_mode_code(NormalizationMode mode);

[[nodiscard]] NormalizationMode normalization_mode_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t training_monitor_code(TrainingMonitor monitor);

[[nodiscard]] TrainingMonitor training_monitor_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t training_monitor_mode_code(TrainingMonitorMode mode);

[[nodiscard]] TrainingMonitorMode training_monitor_mode_from_code(std::uint32_t code);

[[nodiscard]] std::uint32_t parallel_execution_code(ParallelExecution execution);

[[nodiscard]] ParallelExecution parallel_execution_from_code(std::uint32_t code);

} // namespace brutal_mlp::detail

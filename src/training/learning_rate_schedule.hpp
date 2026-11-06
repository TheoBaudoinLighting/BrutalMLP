#pragma once

// Internal header: Learning-rate schedules and reduce-on-plateau state.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>

namespace brutal_mlp::detail {

[[nodiscard]] LearningRateScheduleConfig resolve_learning_rate_schedule(
    const LearningRateScheduleConfig& schedule,
    Scalar optimizer_learning_rate);

[[nodiscard]] Scalar floor_learning_rate(const LearningRateScheduleConfig& schedule,
                                         Scalar learning_rate);

[[nodiscard]] Scalar learning_rate_for_epoch(const LearningRateScheduleConfig& schedule,
                                             const LearningRateSchedulerState& state,
                                             std::size_t epoch,
                                             std::size_t total_epochs);

[[nodiscard]] LearningRateSchedulerState initial_learning_rate_state(
    const LearningRateScheduleConfig& schedule);

bool update_reduce_on_plateau_learning_rate(const LearningRateScheduleConfig& schedule,
                                            LearningRateSchedulerState& state,
                                            std::size_t epoch,
                                            Scalar metric,
                                            bool finite_epoch);

} // namespace brutal_mlp::detail

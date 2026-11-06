// Learning-rate schedules and reduce-on-plateau state.

#include "training/learning_rate_schedule.hpp"

#include "core/checks.hpp"
#include "training/training_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brutal_mlp::detail {

namespace {

[[nodiscard]] Scalar warmup_learning_rate(const LearningRateScheduleConfig& schedule,
                                          std::size_t epoch) {
    const Scalar progress = static_cast<Scalar>(epoch + 1) / static_cast<Scalar>(schedule.warmup_epochs);
    return schedule.warmup_start_learning_rate +
           (schedule.base_learning_rate - schedule.warmup_start_learning_rate) * progress;
}

[[nodiscard]] Scalar deterministic_learning_rate_for_epoch(const LearningRateScheduleConfig& schedule,
                                                           std::size_t epoch,
                                                           std::size_t total_epochs) {
    if (schedule.warmup_epochs > 0 && epoch < schedule.warmup_epochs) {
        return warmup_learning_rate(schedule, epoch);
    }

    const std::size_t main_epoch = epoch >= schedule.warmup_epochs ? epoch - schedule.warmup_epochs : 0;
    switch (schedule.type) {
    case LearningRateSchedule::constant:
    case LearningRateSchedule::reduce_on_plateau:
        return floor_learning_rate(schedule, schedule.base_learning_rate);
    case LearningRateSchedule::step_decay: {
        const std::size_t steps = main_epoch / schedule.step_size;
        const Scalar factor =
            static_cast<Scalar>(std::pow(schedule.step_decay_factor, static_cast<Scalar>(steps)));
        return floor_learning_rate(schedule, schedule.base_learning_rate * factor);
    }
    case LearningRateSchedule::exponential_decay: {
        const Scalar factor =
            static_cast<Scalar>(std::pow(schedule.exponential_decay_rate, static_cast<Scalar>(main_epoch)));
        return floor_learning_rate(schedule, schedule.base_learning_rate * factor);
    }
    case LearningRateSchedule::cosine_annealing: {
        constexpr double pi = 3.141592653589793238462643383279502884;
        const std::size_t remaining_epochs =
            total_epochs > schedule.warmup_epochs ? total_epochs - schedule.warmup_epochs : 1;
        const std::size_t cosine_epochs =
            schedule.cosine_epochs > 0 ? schedule.cosine_epochs : remaining_epochs;
        const std::size_t clamped_epoch = std::min(main_epoch, cosine_epochs);
        const Scalar progress = static_cast<Scalar>(clamped_epoch) / static_cast<Scalar>(cosine_epochs);
        const Scalar cosine = static_cast<Scalar>(std::cos(pi * static_cast<double>(progress)));
        return schedule.minimum_learning_rate +
               (schedule.base_learning_rate - schedule.minimum_learning_rate) *
                   (Scalar{0.5} * (Scalar{1} + cosine));
    }
    }

    throw std::invalid_argument("unknown learning rate schedule");
}

} // namespace

[[nodiscard]] LearningRateScheduleConfig resolve_learning_rate_schedule(
    const LearningRateScheduleConfig& schedule,
    Scalar optimizer_learning_rate) {
    LearningRateScheduleConfig resolved = schedule;
    if (resolved.base_learning_rate == Scalar{0}) {
        resolved.base_learning_rate = optimizer_learning_rate;
    }
    require_finite(resolved.base_learning_rate, "resolved base_learning_rate");
    require(resolved.base_learning_rate > Scalar{0}, "resolved base_learning_rate must be positive");
    require(resolved.minimum_learning_rate <= resolved.base_learning_rate,
            "minimum_learning_rate must be less than or equal to base_learning_rate");
    require(resolved.warmup_start_learning_rate <= resolved.base_learning_rate,
            "warmup_start_learning_rate must be less than or equal to base_learning_rate");
    return resolved;
}

[[nodiscard]] Scalar floor_learning_rate(const LearningRateScheduleConfig& schedule,
                                         Scalar learning_rate) {
    return std::max(schedule.minimum_learning_rate, learning_rate);
}

[[nodiscard]] Scalar learning_rate_for_epoch(const LearningRateScheduleConfig& schedule,
                                             const LearningRateSchedulerState& state,
                                             std::size_t epoch,
                                             std::size_t total_epochs) {
    if (schedule.type == LearningRateSchedule::reduce_on_plateau) {
        if (schedule.warmup_epochs > 0 && epoch < schedule.warmup_epochs) {
            return warmup_learning_rate(schedule, epoch);
        }
        return floor_learning_rate(schedule, state.current_learning_rate);
    }
    return deterministic_learning_rate_for_epoch(schedule, epoch, total_epochs);
}

[[nodiscard]] LearningRateSchedulerState initial_learning_rate_state(
    const LearningRateScheduleConfig& schedule) {
    LearningRateSchedulerState state;
    state.initialized = true;
    state.base_learning_rate = schedule.base_learning_rate;
    state.current_learning_rate = floor_learning_rate(schedule, schedule.base_learning_rate);
    return state;
}

bool update_reduce_on_plateau_learning_rate(const LearningRateScheduleConfig& schedule,
                                            LearningRateSchedulerState& state,
                                            std::size_t epoch,
                                            Scalar metric,
                                            bool finite_epoch) {
    if (schedule.type != LearningRateSchedule::reduce_on_plateau || !finite_epoch) {
        return false;
    }
    if (schedule.warmup_epochs > 0 && epoch < schedule.warmup_epochs) {
        return false;
    }

    if (!state.has_best_metric ||
        metric_improved(metric,
                        state.best_metric,
                        schedule.reduce_on_plateau_mode,
                        schedule.reduce_on_plateau_min_delta)) {
        state.best_metric = metric;
        state.has_best_metric = true;
        state.stale_epochs = 0;
        state.cooldown_remaining = schedule.reduce_on_plateau_cooldown;
        return false;
    }

    if (state.cooldown_remaining > 0) {
        --state.cooldown_remaining;
        return false;
    }

    ++state.stale_epochs;
    if (state.stale_epochs < schedule.reduce_on_plateau_patience) {
        return false;
    }

    const Scalar previous = state.current_learning_rate;
    state.current_learning_rate =
        floor_learning_rate(schedule, state.current_learning_rate * schedule.reduce_on_plateau_factor);
    state.stale_epochs = 0;
    state.cooldown_remaining = schedule.reduce_on_plateau_cooldown;
    return state.current_learning_rate < previous;
}

} // namespace brutal_mlp::detail

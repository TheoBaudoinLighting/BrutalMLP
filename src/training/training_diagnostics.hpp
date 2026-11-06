#pragma once

// Internal header: Training debug checks, explosion tracking and clipping statistics.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace brutal_mlp::detail {

struct TrainingDebugFailure : public std::runtime_error {
    TrainingStopReason reason;

    TrainingDebugFailure(TrainingStopReason stop_reason, std::string message)
        : std::runtime_error(std::move(message)),
          reason(stop_reason) {}
};

struct TrainingDebugScalarTracker {
    bool has_value{false};
    Scalar value{Scalar{0}};
};

[[nodiscard]] bool debug_exploded(Scalar current,
                                  const TrainingDebugScalarTracker& previous,
                                  Scalar absolute_threshold,
                                  Scalar relative_factor) noexcept;

[[nodiscard]] std::string debug_explosion_message(const char* label,
                                                  Scalar current,
                                                  const TrainingDebugScalarTracker& previous,
                                                  Scalar absolute_threshold,
                                                  Scalar relative_factor,
                                                  std::size_t epoch,
                                                  const std::string& extra = {});

void update_debug_tracker_if_finite(TrainingDebugScalarTracker& tracker, Scalar value) noexcept;

void merge_gradient_clipping(GradientClippingDiagnostics& total,
                             const GradientClippingDiagnostics& batch) noexcept;

void finalize_gradient_clipping(GradientClippingDiagnostics& diagnostics) noexcept;

struct TrainingDebugContext {
    const char* phase{"training"};
    std::size_t epoch{0};
    bool has_epoch{false};
    std::size_t sample_index{0};
    bool has_sample_index{false};
};

void debug_check_sample_values(const Scalar* input,
                               std::size_t input_size,
                               const Scalar* target,
                               std::size_t target_size,
                               const TrainingDebugContext& context);

} // namespace brutal_mlp::detail

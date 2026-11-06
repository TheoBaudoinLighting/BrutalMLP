// Training debug checks, explosion tracking and clipping statistics.

#include "training/training_diagnostics.hpp"

#include "core/checks.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

namespace brutal_mlp::detail {

namespace {

[[noreturn]] void throw_training_debug_failure(TrainingStopReason reason, const std::string& message) {
    throw TrainingDebugFailure(reason, message);
}

[[nodiscard]] std::string debug_context_prefix(const TrainingDebugContext& context) {
    std::ostringstream output;
    output << "debug check failed";
    if (context.phase != nullptr && context.phase[0] != '\0') {
        output << " during " << context.phase;
    }
    if (context.has_epoch) {
        output << " at epoch " << context.epoch;
    }
    if (context.has_sample_index) {
        output << " sample " << context.sample_index;
    }
    return output.str();
}

void debug_check_finite_buffer(const Scalar* values,
                               std::size_t size,
                               const char* label,
                               const TrainingDebugContext& context) {
    if (values == nullptr && size != 0) {
        throw_invalid_argument(std::string(label) + " buffer must not be null");
    }
    for (std::size_t i = 0; i < size; ++i) {
        if (!is_finite(values[i])) {
            throw_training_debug_failure(
                TrainingStopReason::invalid_training_data,
                debug_context_prefix(context) + ": " + non_finite_kind(values[i]) +
                    " in " + label + "[" + std::to_string(i) + "]=" +
                    scalar_debug_string(values[i]));
        }
    }
}

} // namespace

[[nodiscard]] bool debug_exploded(Scalar current,
                                  const TrainingDebugScalarTracker& previous,
                                  Scalar absolute_threshold,
                                  Scalar relative_factor) noexcept {
    if (current > absolute_threshold) {
        return true;
    }
    return previous.has_value && previous.value > Scalar{0} && current > previous.value * relative_factor;
}

[[nodiscard]] std::string debug_explosion_message(const char* label,
                                                  Scalar current,
                                                  const TrainingDebugScalarTracker& previous,
                                                  Scalar absolute_threshold,
                                                  Scalar relative_factor,
                                                  std::size_t epoch,
                                                  const std::string& extra) {
    std::ostringstream output;
    output << "debug check failed: " << label << " explosion at epoch " << epoch
           << " value=" << scalar_debug_string(current)
           << " threshold=" << scalar_debug_string(absolute_threshold);
    if (previous.has_value) {
        output << " previous=" << scalar_debug_string(previous.value)
               << " factor=" << scalar_debug_string(relative_factor);
    }
    if (!extra.empty()) {
        output << " " << extra;
    }
    return output.str();
}

void update_debug_tracker_if_finite(TrainingDebugScalarTracker& tracker, Scalar value) noexcept {
    if (is_finite(value)) {
        tracker.has_value = true;
        tracker.value = value;
    }
}

void merge_gradient_clipping(GradientClippingDiagnostics& total,
                             const GradientClippingDiagnostics& batch) noexcept {
    total.batch_count += batch.batch_count;
    total.global_clip_count += batch.global_clip_count;
    total.layer_count += batch.layer_count;
    total.layer_clip_count += batch.layer_clip_count;
    total.gradient_value_count += batch.gradient_value_count;
    total.gradient_value_clip_count += batch.gradient_value_clip_count;
    total.minimum_clip_scale = std::min(total.minimum_clip_scale, batch.minimum_clip_scale);
}

void finalize_gradient_clipping(GradientClippingDiagnostics& diagnostics) noexcept {
    if (diagnostics.batch_count > 0) {
        diagnostics.global_clip_rate =
            static_cast<Scalar>(diagnostics.global_clip_count) /
            static_cast<Scalar>(diagnostics.batch_count);
    }
    if (diagnostics.layer_count > 0) {
        diagnostics.layer_clip_rate =
            static_cast<Scalar>(diagnostics.layer_clip_count) /
            static_cast<Scalar>(diagnostics.layer_count);
    }
    if (diagnostics.gradient_value_count > 0) {
        diagnostics.value_clip_rate =
            static_cast<Scalar>(diagnostics.gradient_value_clip_count) /
            static_cast<Scalar>(diagnostics.gradient_value_count);
    }
}

void debug_check_sample_values(const Scalar* input,
                               std::size_t input_size,
                               const Scalar* target,
                               std::size_t target_size,
                               const TrainingDebugContext& context) {
    debug_check_finite_buffer(input, input_size, "input", context);
    debug_check_finite_buffer(target, target_size, "target", context);
}

} // namespace brutal_mlp::detail

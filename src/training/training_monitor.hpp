#pragma once

// Internal header: Monitored-metric selection and improvement tests.

#include "brutal_mlp/brutal_mlp.hpp"

namespace brutal_mlp::detail {

[[nodiscard]] TrainingMonitor resolve_training_monitor(TrainingMonitor monitor,
                                                       bool has_validation_loss,
                                                       bool has_test_loss);

[[nodiscard]] Scalar select_monitored_metric(TrainingMonitor monitor,
                                             Scalar training_loss,
                                             bool has_validation_loss,
                                             Scalar validation_loss,
                                             bool has_test_loss,
                                             Scalar test_loss);

[[nodiscard]] Scalar initial_best_metric(TrainingMonitorMode mode);

[[nodiscard]] bool metric_improved(Scalar current,
                                   Scalar best,
                                   TrainingMonitorMode mode,
                                   Scalar min_delta);

} // namespace brutal_mlp::detail

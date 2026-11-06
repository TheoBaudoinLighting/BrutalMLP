// Monitored-metric selection and improvement tests.

#include "training/training_monitor.hpp"

#include "core/checks.hpp"

#include <limits>
#include <stdexcept>

namespace brutal_mlp::detail {

[[nodiscard]] TrainingMonitor resolve_training_monitor(TrainingMonitor monitor,
                                                       bool has_validation_loss,
                                                       bool has_test_loss) {
    if (monitor == TrainingMonitor::automatic) {
        return has_validation_loss ? TrainingMonitor::validation_loss : TrainingMonitor::training_loss;
    }
    if (monitor == TrainingMonitor::validation_loss) {
        require(has_validation_loss, "validation_loss monitor requires a non-empty validation split");
    }
    if (monitor == TrainingMonitor::test_loss) {
        require(has_test_loss, "test_loss monitor requires a non-empty test split");
    }
    return monitor;
}

[[nodiscard]] Scalar select_monitored_metric(TrainingMonitor monitor,
                                             Scalar training_loss,
                                             bool has_validation_loss,
                                             Scalar validation_loss,
                                             bool has_test_loss,
                                             Scalar test_loss) {
    switch (monitor) {
    case TrainingMonitor::automatic:
        return select_monitored_metric(resolve_training_monitor(monitor, has_validation_loss, has_test_loss),
                                       training_loss,
                                       has_validation_loss,
                                       validation_loss,
                                       has_test_loss,
                                       test_loss);
    case TrainingMonitor::training_loss:
        return training_loss;
    case TrainingMonitor::validation_loss:
        require(has_validation_loss, "validation_loss monitor requires a non-empty validation split");
        return validation_loss;
    case TrainingMonitor::test_loss:
        require(has_test_loss, "test_loss monitor requires a non-empty test split");
        return test_loss;
    }

    throw std::invalid_argument("unknown training monitor");
}

[[nodiscard]] Scalar initial_best_metric(TrainingMonitorMode mode) {
    switch (mode) {
    case TrainingMonitorMode::minimize:
        return std::numeric_limits<Scalar>::infinity();
    case TrainingMonitorMode::maximize:
        return -std::numeric_limits<Scalar>::infinity();
    }

    throw std::invalid_argument("unknown training monitor mode");
}

[[nodiscard]] bool metric_improved(Scalar current,
                                   Scalar best,
                                   TrainingMonitorMode mode,
                                   Scalar min_delta) {
    switch (mode) {
    case TrainingMonitorMode::minimize:
        return current + min_delta < best;
    case TrainingMonitorMode::maximize:
        return current > best + min_delta;
    }

    throw std::invalid_argument("unknown training monitor mode");
}

} // namespace brutal_mlp::detail

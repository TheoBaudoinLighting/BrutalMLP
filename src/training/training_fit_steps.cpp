// TrainingModel::fit_steps: step-based training on mini-batches drawn at random from a dataset.

#include "training/training_model_impl.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "core/random_utils.hpp"
#include "data/data_validation.hpp"
#include "training/learning_rate_schedule.hpp"
#include "training/training_diagnostics.hpp"
#include "training/training_monitor.hpp"
#include "training/training_run_config.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace brutal_mlp {

using namespace detail;

namespace {

// Independent random streams derived from the run seed.
constexpr std::uint64_t kBatchIndexStream = 0x5B1F0E3C9A2D4871ULL;
constexpr std::uint64_t kStepRandomStream = 0xC2B2AE3D27D4EB4FULL;

// TrainingOptions equivalent of a step run. One "epoch" is one evaluation period. Used to
// validate the shared fields, to describe the run and to size the worker pool.
[[nodiscard]] TrainingOptions equivalent_training_options(const StepTrainingOptions& options, std::size_t periods) {
    TrainingOptions equivalent;
    equivalent.epochs = periods;
    equivalent.batch_size = options.batch_size;
    equivalent.shuffle = true;
    equivalent.seed = options.seed;
    equivalent.learning_rate_schedule = options.learning_rate_schedule;
    equivalent.gradient_noise_stddev = options.gradient_noise_stddev;
    equivalent.parallelism = options.parallelism;
    equivalent.debug = options.debug;
    equivalent.restore_best_weights = false;
    return equivalent;
}

} // namespace

TrainingHistory TrainingModel::fit_steps(const Dataset& training,
                                         const StepTrainingOptions& options,
                                         const Dataset* validation) {
    require(options.steps > 0, "steps must be positive");
    require(options.batch_size > 0, "batch_size must be positive");
    const std::uint64_t interval =
        options.evaluation_interval == 0 ? options.steps : std::min(options.evaluation_interval, options.steps);
    const std::size_t periods = static_cast<std::size_t>((options.steps + interval - 1) / interval);
    TrainingOptions runtime_options = equivalent_training_options(options, periods);
    validate_training_options(runtime_options);
    validate_dataset_shape(training.sample_count(), training.input_size(), training.output_size(), input_size(), output_size());
    if (validation != nullptr) {
        validate_dataset_shape(validation->sample_count(),
                               validation->input_size(),
                               validation->output_size(),
                               input_size(),
                               output_size());
    }
    runtime_options.learning_rate_schedule =
        resolve_learning_rate_schedule(options.learning_rate_schedule, impl_->optimizer.learning_rate);
    OptimizerConfig run_optimizer = impl_->optimizer;
    run_optimizer.learning_rate = runtime_options.learning_rate_schedule.base_learning_rate;
    impl_->last_training_options = runtime_options;
    impl_->has_last_training_options = true;
    const std::uint64_t seed = non_zero_seed(options.seed, impl_->seed);

    TrainingHistory history;
    history.training_loss.reserve(periods);
    history.validation_loss.reserve(periods);
    history.epochs.reserve(periods);
    // No index list is materialized: the split only carries counts for the run description.
    history.run_config = impl_->make_training_run_config(run_optimizer, runtime_options, DatasetSplit{},
                                                         training.sample_count(), true);
    history.run_config.dataset.training_sample_count = training.sample_count();
    history.run_config.dataset.validation_sample_count = validation != nullptr ? validation->sample_count() : 0;
    history.stop_reason = TrainingStopReason::completed_epochs;
    history.stop_message = "completed requested steps";
    history.monitor = validation != nullptr ? TrainingMonitor::validation_loss : TrainingMonitor::training_loss;
    history.monitor_mode = TrainingMonitorMode::minimize;

    // A continued run (first_step > 0) keeps the scheduler state of the previous call.
    LearningRateSchedulerState learning_rate_state =
        options.first_step > 0 && impl_->learning_rate_state.initialized
            ? impl_->learning_rate_state
            : initial_learning_rate_state(runtime_options.learning_rate_schedule);
    learning_rate_state.base_learning_rate = runtime_options.learning_rate_schedule.base_learning_rate;
    if (learning_rate_state.current_learning_rate == Scalar{0}) {
        learning_rate_state.current_learning_rate =
            floor_learning_rate(runtime_options.learning_rate_schedule,
                                runtime_options.learning_rate_schedule.base_learning_rate);
    }

    // Everything below is sized by the batch (or the validation set), never by `training`.
    TrainingModel::Impl::TrainingWorkspace workspace = impl_->make_workspace(options.batch_size);
    std::unique_ptr<TrainingModel::Impl::ParallelTraining> parallel;
    if (options.parallelism.execution == ParallelExecution::worker_threads) {
        parallel = std::make_unique<TrainingModel::Impl::ParallelTraining>(impl_->make_parallel_training(
            runtime_options, validation != nullptr ? validation->sample_count() : options.batch_size));
    }
    std::vector<std::size_t> batch_indices(options.batch_size);
    const std::vector<std::size_t> validation_indices =
        validation != nullptr ? make_indices(validation->sample_count()) : std::vector<std::size_t>{};
    MiniBatch raw_batch;
    MiniBatch normalized_batch;
    TrainingDebugScalarTracker debug_training_loss;
    TrainingDebugScalarTracker debug_validation_loss;
    TrainingDebugScalarTracker debug_gradient_norm;
    Scalar best_metric = initial_best_metric(TrainingMonitorMode::minimize);
    const std::size_t schedule_epochs =
        static_cast<std::size_t>((options.first_step + options.steps + interval - 1) / interval);

    std::uint64_t step = 0;
    for (std::size_t period = 0; period < periods; ++period) {
        const auto period_start = std::chrono::steady_clock::now();
        const std::size_t schedule_epoch =
            static_cast<std::size_t>((options.first_step + static_cast<std::uint64_t>(period) * interval) / interval);
        const Scalar learning_rate = learning_rate_for_epoch(runtime_options.learning_rate_schedule,
                                                             learning_rate_state,
                                                             schedule_epoch,
                                                             schedule_epochs);
        if (runtime_options.learning_rate_schedule.type != LearningRateSchedule::reduce_on_plateau) {
            learning_rate_state.current_learning_rate = learning_rate;
        }

        double period_loss = 0.0;
        std::uint64_t period_samples = 0;
        Scalar period_gradient_norm = Scalar{0};
        GradientClippingDiagnostics period_clipping;
        bool non_finite_gradient = false;
        bool gradient_explosion = false;
        std::string stop_message;
        const std::uint64_t period_end = std::min<std::uint64_t>(options.steps, (period + 1) * interval);

        for (; step < period_end; ++step) {
            const std::uint64_t global_step = options.first_step + step;
            for (std::size_t slot = 0; slot < options.batch_size; ++slot) {
                batch_indices[slot] =
                    counter_based_index(seed ^ kBatchIndexStream, global_step, slot, training.sample_count());
            }
            TrainingDebugContext debug_context;
            debug_context.phase = "training batch";
            debug_context.epoch = period;
            debug_context.has_epoch = true;
            try {
                if (parallel) {
                    impl_->fill_dataset_batch_parallel(training, batch_indices, 0, options.batch_size, raw_batch,
                                                       normalized_batch, *parallel, &runtime_options.debug,
                                                       debug_context);
                } else {
                    impl_->fill_dataset_batch(training, batch_indices, 0, options.batch_size, raw_batch,
                                              normalized_batch, &runtime_options.debug, debug_context);
                }
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                impl_->learning_rate_state = learning_rate_state;
                refresh_training_run_result(history, history.epochs.size());
                return history;
            }

            // Dropout and gradient noise of a step depend only on (seed, step index).
            std::mt19937_64 step_rng(mix64(seed ^ kStepRandomStream) + global_step);
            Scalar batch_loss = Scalar{0};
            impl_->zero_gradients();
            if (parallel) {
                impl_->accumulate_batch_gradients_parallel(normalized_batch, *parallel, &step_rng, &batch_loss);
            } else {
                impl_->accumulate_batch_gradients(normalized_batch, workspace, &step_rng, &batch_loss);
            }
            const auto gradient_result = impl_->apply_gradients(options.batch_size,
                                                                learning_rate,
                                                                options.gradient_noise_stddev,
                                                                &step_rng);
            if (!is_finite(gradient_result.gradient_norm)) {
                period_gradient_norm = gradient_result.gradient_norm;
                non_finite_gradient = true;
                stop_message = "stopped because gradient norm became non-finite at step " +
                               std::to_string(global_step) + ": norm=" +
                               scalar_debug_string(gradient_result.gradient_norm);
                if (!gradient_result.non_finite_location.empty()) {
                    stop_message += " first_non_finite=" + gradient_result.non_finite_location;
                }
                ++step;
                break;
            }
            if (runtime_options.debug.enabled &&
                debug_exploded(gradient_result.gradient_norm,
                               debug_gradient_norm,
                               runtime_options.debug.gradient_norm_explosion_threshold,
                               runtime_options.debug.gradient_norm_explosion_factor)) {
                period_gradient_norm = gradient_result.gradient_norm;
                gradient_explosion = true;
                stop_message = debug_explosion_message("gradient norm",
                                                       gradient_result.gradient_norm,
                                                       debug_gradient_norm,
                                                       runtime_options.debug.gradient_norm_explosion_threshold,
                                                       runtime_options.debug.gradient_norm_explosion_factor,
                                                       period,
                                                       "step=" + std::to_string(global_step));
                ++step;
                break;
            }
            update_debug_tracker_if_finite(debug_gradient_norm, gradient_result.gradient_norm);
            merge_gradient_clipping(period_clipping, gradient_result.clipping);
            period_gradient_norm = std::max(period_gradient_norm, gradient_result.gradient_norm);
            period_loss += static_cast<double>(batch_loss);
            period_samples += options.batch_size;
        }
        finalize_gradient_clipping(period_clipping);

        const Scalar training_loss =
            period_samples > 0 ? static_cast<Scalar>(period_loss / static_cast<double>(period_samples)) : Scalar{0};
        history.training_loss.push_back(training_loss);
        bool has_validation_loss = false;
        Scalar validation_loss = Scalar{0};
        if (validation != nullptr && !non_finite_gradient && !gradient_explosion) {
            try {
                validation_loss = parallel ? impl_->loss_for_dataset_indices_parallel(*validation,
                                                                                      validation_indices,
                                                                                      *parallel,
                                                                                      &runtime_options.debug,
                                                                                      "validation loss",
                                                                                      period,
                                                                                      true)
                                           : impl_->loss_for_dataset_indices(*validation,
                                                                             validation_indices,
                                                                             workspace,
                                                                             &runtime_options.debug,
                                                                             "validation loss",
                                                                             period,
                                                                             true);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                impl_->learning_rate_state = learning_rate_state;
                refresh_training_run_result(history, history.epochs.size());
                return history;
            }
            history.validation_loss.push_back(validation_loss);
            has_validation_loss = true;
        }

        const std::size_t non_finite_parameters = impl_->non_finite_parameter_count();
        const bool non_finite_loss = (period_samples > 0 && !is_finite(training_loss)) ||
                                     (has_validation_loss && !is_finite(validation_loss));
        bool loss_explosion = false;
        if (runtime_options.debug.enabled && period_samples > 0 && is_finite(training_loss) &&
            debug_exploded(training_loss,
                           debug_training_loss,
                           runtime_options.debug.loss_explosion_threshold,
                           runtime_options.debug.loss_explosion_factor)) {
            loss_explosion = true;
            stop_message = debug_explosion_message("training loss",
                                                   training_loss,
                                                   debug_training_loss,
                                                   runtime_options.debug.loss_explosion_threshold,
                                                   runtime_options.debug.loss_explosion_factor,
                                                   period);
        }
        update_debug_tracker_if_finite(debug_training_loss, training_loss);
        if (has_validation_loss) {
            update_debug_tracker_if_finite(debug_validation_loss, validation_loss);
        }
        const bool finite_period = !non_finite_gradient && !gradient_explosion && !non_finite_loss &&
                                   non_finite_parameters == 0 && !loss_explosion;
        const Scalar monitored_metric = has_validation_loss ? validation_loss : training_loss;
        bool improved = false;
        if (finite_period && metric_improved(monitored_metric, best_metric, TrainingMonitorMode::minimize, Scalar{0})) {
            best_metric = monitored_metric;
            improved = true;
            history.has_best_checkpoint = true;
            history.best_epoch = period;
            history.best_metric = monitored_metric;
        }
        const bool learning_rate_reduced = update_reduce_on_plateau_learning_rate(runtime_options.learning_rate_schedule,
                                                                                  learning_rate_state,
                                                                                  schedule_epoch,
                                                                                  monitored_metric,
                                                                                  finite_period);

        TrainingEpochDiagnostics diagnostics;
        diagnostics.epoch = period;
        diagnostics.training_loss = training_loss;
        diagnostics.has_validation_loss = has_validation_loss;
        diagnostics.validation_loss = validation_loss;
        diagnostics.monitor = history.monitor;
        diagnostics.monitored_metric = monitored_metric;
        diagnostics.learning_rate_schedule = runtime_options.learning_rate_schedule.type;
        diagnostics.learning_rate = learning_rate;
        diagnostics.next_learning_rate = learning_rate_for_epoch(runtime_options.learning_rate_schedule,
                                                                 learning_rate_state,
                                                                 schedule_epoch + 1,
                                                                 schedule_epochs);
        diagnostics.learning_rate_reduced = learning_rate_reduced;
        diagnostics.gradient_norm = period_gradient_norm;
        diagnostics.clipping = period_clipping;
        diagnostics.weights = impl_->weight_statistics();
        diagnostics.finite = finite_period;
        diagnostics.non_finite_parameter_count = non_finite_parameters;
        diagnostics.epoch_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - period_start).count();
        diagnostics.best_checkpoint = improved;
        diagnostics.improved = improved;
        history.epochs.push_back(diagnostics);

        if (non_finite_gradient) {
            history.stop_reason = TrainingStopReason::non_finite_gradient;
            history.stop_message = stop_message;
            break;
        }
        if (gradient_explosion) {
            history.stop_reason = TrainingStopReason::gradient_explosion;
            history.stop_message = stop_message;
            break;
        }
        if (non_finite_parameters > 0) {
            history.stop_reason = TrainingStopReason::non_finite_weights;
            history.stop_message = "stopped because model parameters became non-finite in period " +
                                   std::to_string(period) + ": " + impl_->first_non_finite_parameter_location();
            break;
        }
        if (non_finite_loss) {
            history.stop_reason = TrainingStopReason::non_finite_loss;
            history.stop_message = "stopped because loss became non-finite in period " + std::to_string(period) +
                                   " training_loss=" + scalar_debug_string(training_loss);
            break;
        }
        if (loss_explosion) {
            history.stop_reason = TrainingStopReason::loss_explosion;
            history.stop_message = stop_message;
            break;
        }
    }

    impl_->learning_rate_state = learning_rate_state;
    refresh_training_run_result(history, history.epochs.size());
    return history;
}

} // namespace brutal_mlp

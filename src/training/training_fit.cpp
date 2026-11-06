// TrainingModel::fit for indexed datasets.

#include "brutal_mlp/brutal_mlp.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "core/random_utils.hpp"
#include "data/data_validation.hpp"
#include "training/learning_rate_schedule.hpp"
#include "training/training_diagnostics.hpp"
#include "training/training_model_impl.hpp"
#include "training/training_monitor.hpp"
#include "training/training_run_config.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace brutal_mlp {

using namespace detail;

TrainingHistory TrainingModel::fit(const Dataset& dataset, const TrainingOptions& options) {
    validate_training_options(options);
    validate_dataset_shape(dataset.sample_count(), dataset.input_size(), dataset.output_size(), input_size(), output_size());
    TrainingOptions runtime_options = options;
    runtime_options.learning_rate_schedule =
        resolve_learning_rate_schedule(options.learning_rate_schedule, impl_->optimizer.learning_rate);
    OptimizerConfig run_optimizer = impl_->optimizer;
    run_optimizer.learning_rate = runtime_options.learning_rate_schedule.base_learning_rate;
    impl_->last_training_options = runtime_options;
    impl_->has_last_training_options = true;

    DatasetSplitOptions split_options;
    split_options.validation_split = options.validation_split;
    split_options.test_split = options.test_split;
    split_options.shuffle = options.shuffle;
    split_options.seed = non_zero_seed(options.seed, impl_->seed);
    DatasetSplit split = make_dataset_split(dataset.sample_count(), split_options);
    std::vector<std::size_t> training_indices = split.training_indices;

    TrainingHistory history;
    history.training_loss.reserve(options.epochs);
    history.validation_loss.reserve(options.epochs);
    history.test_loss.reserve(options.epochs);
    history.epochs.reserve(options.epochs);
    history.run_config = impl_->make_training_run_config(run_optimizer,
                                                         runtime_options,
                                                         split,
                                                         dataset.sample_count(),
                                                         false);
    history.stop_reason = TrainingStopReason::completed_epochs;
    history.stop_message = "completed requested epochs";
    const TrainingMonitor resolved_monitor = resolve_training_monitor(options.early_stopping_monitor,
                                                                       !split.validation_indices.empty(),
                                                                       !split.test_indices.empty());
    const TrainingMonitor resolved_learning_rate_monitor =
        runtime_options.learning_rate_schedule.type == LearningRateSchedule::reduce_on_plateau
            ? resolve_training_monitor(runtime_options.learning_rate_schedule.reduce_on_plateau_monitor,
                                       !split.validation_indices.empty(),
                                       !split.test_indices.empty())
            : resolved_monitor;
    history.monitor = resolved_monitor;
    history.monitor_mode = options.early_stopping_mode;
    auto finish_history = [&]() -> TrainingHistory {
        refresh_training_run_result(history, impl_->completed_epochs);
        return history;
    };

    const bool resume_training = impl_->has_resume_epoch;
    const std::size_t first_epoch = resume_training ? std::min(impl_->resume_epoch, options.epochs) : 0;
    impl_->has_resume_epoch = false;
    impl_->completed_epochs = first_epoch;
    if (first_epoch >= options.epochs) {
        history.stop_message = "checkpoint already reached requested epochs";
        return finish_history();
    }

    Scalar best_metric = initial_best_metric(options.early_stopping_mode);
    TrainingModel::Impl::TrainingSnapshot best_snapshot;
    std::size_t stale_epochs = 0;
    std::size_t cooldown_remaining = 0;
    LearningRateSchedulerState learning_rate_state =
        resume_training && impl_->learning_rate_state.initialized
            ? impl_->learning_rate_state
            : initial_learning_rate_state(runtime_options.learning_rate_schedule);
    learning_rate_state.base_learning_rate = runtime_options.learning_rate_schedule.base_learning_rate;
    if (learning_rate_state.current_learning_rate == Scalar{0}) {
        learning_rate_state.current_learning_rate =
            floor_learning_rate(runtime_options.learning_rate_schedule,
                                runtime_options.learning_rate_schedule.base_learning_rate);
    }
    impl_->learning_rate_state = learning_rate_state;

    // Forward/backward buffers allocated once for the whole fit, reused by every sample.
    TrainingModel::Impl::TrainingWorkspace workspace = impl_->make_workspace(options.batch_size);
    // With ParallelExecution::worker_threads, batches and loss evaluations are computed by a
    // worker pool created once for this call. Results do not depend on the thread count.
    std::unique_ptr<TrainingModel::Impl::ParallelTraining> parallel;
    if (options.parallelism.execution == ParallelExecution::worker_threads) {
        parallel = std::make_unique<TrainingModel::Impl::ParallelTraining>(impl_->make_parallel_training(options, dataset.sample_count()));
    }
    auto evaluate_loss = [&](const std::vector<std::size_t>& indices, const char* phase, std::size_t epoch) {
        return parallel ? impl_->loss_for_dataset_indices_parallel(
                              dataset, indices, *parallel, &runtime_options.debug, phase, epoch, true)
                        : impl_->loss_for_dataset_indices(
                              dataset, indices, workspace, &runtime_options.debug, phase, epoch, true);
    };
    MiniBatch raw_batch;
    MiniBatch normalized_batch;
    TrainingDebugScalarTracker debug_training_loss;
    TrainingDebugScalarTracker debug_validation_loss;
    TrainingDebugScalarTracker debug_test_loss;
    TrainingDebugScalarTracker debug_gradient_norm;

    for (std::size_t epoch = first_epoch; epoch < options.epochs; ++epoch) {
        const auto epoch_start = std::chrono::steady_clock::now();
        const Scalar scheduled_learning_rate = learning_rate_for_epoch(runtime_options.learning_rate_schedule,
                                                                       learning_rate_state,
                                                                       epoch,
                                                                       options.epochs);
        if (runtime_options.learning_rate_schedule.type != LearningRateSchedule::reduce_on_plateau) {
            learning_rate_state.current_learning_rate = scheduled_learning_rate;
        }
        Scalar epoch_gradient_norm = Scalar{0};
        GradientClippingDiagnostics epoch_clipping;
        bool non_finite_gradient = false;
        bool gradient_explosion = false;
        std::string non_finite_gradient_message;
        std::string gradient_explosion_message;

        std::mt19937_64 epoch_rng(non_zero_seed(options.seed, impl_->seed) + epoch);
        if (options.shuffle) {
            std::shuffle(training_indices.begin(), training_indices.end(), epoch_rng);
        }

        for (std::size_t begin = 0; begin < training_indices.size(); begin += options.batch_size) {
            const std::size_t batch_count = std::min(options.batch_size, training_indices.size() - begin);
            TrainingDebugContext debug_context;
            debug_context.phase = "training batch";
            debug_context.epoch = epoch;
            debug_context.has_epoch = true;
            try {
                impl_->fill_dataset_batch(dataset,
                                          training_indices,
                                          begin,
                                          batch_count,
                                          raw_batch,
                                          normalized_batch,
                                          &runtime_options.debug,
                                          debug_context);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && best_snapshot.valid()) {
                    impl_->restore_snapshot(best_snapshot);
                }
                return finish_history();
            }
            impl_->zero_gradients();
            if (parallel) {
                impl_->accumulate_batch_gradients_parallel(normalized_batch, *parallel, &epoch_rng);
            } else {
                impl_->accumulate_batch_gradients(normalized_batch, workspace, &epoch_rng);
            }
            const auto gradient_result =
                impl_->apply_gradients(batch_count,
                                       scheduled_learning_rate,
                                       runtime_options.gradient_noise_stddev,
                                       &epoch_rng);
            if (!is_finite(gradient_result.gradient_norm)) {
                epoch_gradient_norm = gradient_result.gradient_norm;
                non_finite_gradient = true;
                non_finite_gradient_message =
                    "stopped because gradient norm became non-finite at epoch " +
                    std::to_string(epoch) + " batch_begin " + std::to_string(begin) +
                    ": norm=" + scalar_debug_string(gradient_result.gradient_norm);
                if (!gradient_result.non_finite_location.empty()) {
                    non_finite_gradient_message += " first_non_finite=" + gradient_result.non_finite_location;
                }
                break;
            }
            if (runtime_options.debug.enabled &&
                debug_exploded(gradient_result.gradient_norm,
                               debug_gradient_norm,
                               runtime_options.debug.gradient_norm_explosion_threshold,
                               runtime_options.debug.gradient_norm_explosion_factor)) {
                epoch_gradient_norm = gradient_result.gradient_norm;
                gradient_explosion = true;
                gradient_explosion_message =
                    debug_explosion_message("gradient norm",
                                            gradient_result.gradient_norm,
                                            debug_gradient_norm,
                                            runtime_options.debug.gradient_norm_explosion_threshold,
                                            runtime_options.debug.gradient_norm_explosion_factor,
                                            epoch,
                                            "batch_begin=" + std::to_string(begin));
                break;
            }
            update_debug_tracker_if_finite(debug_gradient_norm, gradient_result.gradient_norm);
            merge_gradient_clipping(epoch_clipping, gradient_result.clipping);
            epoch_gradient_norm = std::max(epoch_gradient_norm, gradient_result.gradient_norm);
        }
        finalize_gradient_clipping(epoch_clipping);

        Scalar training_loss = Scalar{0};
        try {
            training_loss = evaluate_loss(training_indices, "training loss", epoch);
        } catch (const TrainingDebugFailure& failure) {
            history.stop_reason = failure.reason;
            history.stop_message = failure.what();
            if (options.restore_best_weights && best_snapshot.valid()) {
                impl_->restore_snapshot(best_snapshot);
            }
            return finish_history();
        }
        history.training_loss.push_back(training_loss);

        bool has_validation_loss = false;
        Scalar validation_loss = Scalar{0};
        if (!split.validation_indices.empty()) {
            try {
                validation_loss = evaluate_loss(split.validation_indices, "validation loss", epoch);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && best_snapshot.valid()) {
                    impl_->restore_snapshot(best_snapshot);
                }
                return finish_history();
            }
            history.validation_loss.push_back(validation_loss);
            has_validation_loss = true;
        }
        bool has_test_loss = false;
        Scalar test_loss = Scalar{0};
        if (!split.test_indices.empty()) {
            try {
                test_loss = evaluate_loss(split.test_indices, "test loss", epoch);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && best_snapshot.valid()) {
                    impl_->restore_snapshot(best_snapshot);
                }
                return finish_history();
            }
            history.test_loss.push_back(test_loss);
            has_test_loss = true;
        }

        const WeightStatistics weight_stats = impl_->weight_statistics();
        const std::size_t non_finite_parameters = impl_->non_finite_parameter_count();
        bool loss_explosion = false;
        std::string loss_explosion_message;
        if (runtime_options.debug.enabled && is_finite(training_loss) &&
            debug_exploded(training_loss,
                           debug_training_loss,
                           runtime_options.debug.loss_explosion_threshold,
                           runtime_options.debug.loss_explosion_factor)) {
            loss_explosion = true;
            loss_explosion_message =
                debug_explosion_message("training loss",
                                        training_loss,
                                        debug_training_loss,
                                        runtime_options.debug.loss_explosion_threshold,
                                        runtime_options.debug.loss_explosion_factor,
                                        epoch);
        }
        if (!loss_explosion && runtime_options.debug.enabled && has_validation_loss && is_finite(validation_loss) &&
            debug_exploded(validation_loss,
                           debug_validation_loss,
                           runtime_options.debug.loss_explosion_threshold,
                           runtime_options.debug.loss_explosion_factor)) {
            loss_explosion = true;
            loss_explosion_message =
                debug_explosion_message("validation loss",
                                        validation_loss,
                                        debug_validation_loss,
                                        runtime_options.debug.loss_explosion_threshold,
                                        runtime_options.debug.loss_explosion_factor,
                                        epoch);
        }
        if (!loss_explosion && runtime_options.debug.enabled && has_test_loss && is_finite(test_loss) &&
            debug_exploded(test_loss,
                           debug_test_loss,
                           runtime_options.debug.loss_explosion_threshold,
                           runtime_options.debug.loss_explosion_factor)) {
            loss_explosion = true;
            loss_explosion_message =
                debug_explosion_message("test loss",
                                        test_loss,
                                        debug_test_loss,
                                        runtime_options.debug.loss_explosion_threshold,
                                        runtime_options.debug.loss_explosion_factor,
                                        epoch);
        }
        update_debug_tracker_if_finite(debug_training_loss, training_loss);
        if (has_validation_loss) {
            update_debug_tracker_if_finite(debug_validation_loss, validation_loss);
        }
        if (has_test_loss) {
            update_debug_tracker_if_finite(debug_test_loss, test_loss);
        }
        const bool non_finite_loss = !is_finite(training_loss) ||
                                     (has_validation_loss && !is_finite(validation_loss)) ||
                                     (has_test_loss && !is_finite(test_loss));
        const bool non_finite_weights = non_finite_parameters > 0;
        const bool finite_epoch = !non_finite_gradient &&
                                  !non_finite_loss &&
                                  !non_finite_weights &&
                                  !gradient_explosion &&
                                  !loss_explosion;
        const Scalar monitored_metric = select_monitored_metric(resolved_monitor,
                                                                training_loss,
                                                                has_validation_loss,
                                                                validation_loss,
                                                                has_test_loss,
                                                                test_loss);
        const Scalar learning_rate_metric =
            runtime_options.learning_rate_schedule.type == LearningRateSchedule::reduce_on_plateau
                ? select_monitored_metric(resolved_learning_rate_monitor,
                                          training_loss,
                                          has_validation_loss,
                                          validation_loss,
                                          has_test_loss,
                                          test_loss)
                : monitored_metric;
        bool best_checkpoint = false;
        bool improved = false;

        if (finite_epoch && metric_improved(monitored_metric,
                                            best_metric,
                                            options.early_stopping_mode,
                                            options.min_delta)) {
            best_metric = monitored_metric;
            stale_epochs = 0;
            cooldown_remaining = options.early_stopping_cooldown;
            best_checkpoint = true;
            improved = true;
            history.has_best_checkpoint = true;
            history.best_epoch = epoch;
            history.best_metric = monitored_metric;
            if (options.restore_best_weights) {
                best_snapshot = impl_->snapshot();
            }
        } else if (finite_epoch) {
            if (cooldown_remaining > 0) {
                --cooldown_remaining;
            } else {
                ++stale_epochs;
            }
        }

        const bool learning_rate_reduced =
            update_reduce_on_plateau_learning_rate(runtime_options.learning_rate_schedule,
                                                   learning_rate_state,
                                                   epoch,
                                                   learning_rate_metric,
                                                   finite_epoch);
        impl_->learning_rate_state = learning_rate_state;
        const Scalar next_learning_rate = learning_rate_for_epoch(runtime_options.learning_rate_schedule,
                                                                  learning_rate_state,
                                                                  epoch + 1,
                                                                  options.epochs);

        const auto epoch_end = std::chrono::steady_clock::now();
        TrainingEpochDiagnostics diagnostics;
        diagnostics.epoch = epoch;
        diagnostics.training_loss = training_loss;
        diagnostics.has_validation_loss = has_validation_loss;
        diagnostics.validation_loss = validation_loss;
        diagnostics.has_test_loss = has_test_loss;
        diagnostics.test_loss = test_loss;
        diagnostics.monitor = resolved_monitor;
        diagnostics.monitored_metric = monitored_metric;
        diagnostics.learning_rate_schedule = runtime_options.learning_rate_schedule.type;
        diagnostics.learning_rate = scheduled_learning_rate;
        diagnostics.next_learning_rate = next_learning_rate;
        diagnostics.learning_rate_reduced = learning_rate_reduced;
        diagnostics.gradient_norm = epoch_gradient_norm;
        diagnostics.clipping = epoch_clipping;
        diagnostics.weights = weight_stats;
        diagnostics.finite = finite_epoch;
        diagnostics.non_finite_parameter_count = non_finite_parameters;
        diagnostics.epoch_seconds = std::chrono::duration<double>(epoch_end - epoch_start).count();
        diagnostics.best_checkpoint = best_checkpoint;
        diagnostics.improved = improved;
        diagnostics.stale_epochs = stale_epochs;
        diagnostics.cooldown_remaining = cooldown_remaining;
        history.epochs.push_back(diagnostics);
        impl_->completed_epochs = epoch + 1;

        if (diagnostics.finite) {
            TrainingCheckpointInfo checkpoint_info;
            checkpoint_info.completed_epochs = impl_->completed_epochs;
            checkpoint_info.has_metric = true;
            checkpoint_info.metric = monitored_metric;
            checkpoint_info.learning_rate_state = learning_rate_state;
            checkpoint_info.training_options = runtime_options;
            if (best_checkpoint && !options.best_checkpoint_path.empty()) {
                checkpoint_info.best_checkpoint = true;
                save_checkpoint(options.best_checkpoint_path, checkpoint_info);
            }
            if (!options.latest_checkpoint_path.empty()) {
                checkpoint_info.best_checkpoint = false;
                save_checkpoint(options.latest_checkpoint_path, checkpoint_info);
            }
        }

        if (non_finite_gradient) {
            history.stop_reason = TrainingStopReason::non_finite_gradient;
            history.stop_message = non_finite_gradient_message.empty()
                                       ? "stopped because gradient norm became non-finite"
                                       : non_finite_gradient_message;
            break;
        }
        if (non_finite_weights) {
            history.stop_reason = TrainingStopReason::non_finite_weights;
            history.stop_message = "stopped because model parameters became non-finite at epoch " +
                                   std::to_string(epoch) + ": " +
                                   impl_->first_non_finite_parameter_location();
            break;
        }
        if (non_finite_loss) {
            history.stop_reason = TrainingStopReason::non_finite_loss;
            history.stop_message = "stopped because loss became non-finite at epoch " +
                                   std::to_string(epoch) +
                                   " training_loss=" + scalar_debug_string(training_loss);
            if (has_validation_loss) {
                history.stop_message += " validation_loss=" + scalar_debug_string(validation_loss);
            }
            if (has_test_loss) {
                history.stop_message += " test_loss=" + scalar_debug_string(test_loss);
            }
            break;
        }
        if (gradient_explosion) {
            history.stop_reason = TrainingStopReason::gradient_explosion;
            history.stop_message = gradient_explosion_message;
            break;
        }
        if (loss_explosion) {
            history.stop_reason = TrainingStopReason::loss_explosion;
            history.stop_message = loss_explosion_message;
            break;
        }
        if (options.early_stopping_patience > 0 && stale_epochs >= options.early_stopping_patience) {
            history.stop_reason = TrainingStopReason::early_stopping;
            history.stop_message = "stopped by early stopping patience";
            break;
        }
    }

    if (options.restore_best_weights && best_snapshot.valid()) {
        impl_->restore_snapshot(best_snapshot);
    }

    return finish_history();
}

} // namespace brutal_mlp

#include "brutal_mlp/brutal_mlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#ifndef BRUTAL_MLP_VERSION_STRING
#define BRUTAL_MLP_VERSION_STRING "0.0.0"
#endif

namespace brutal_mlp {
namespace {

#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
constexpr Scalar kProbabilityEpsilon = Scalar{1e-12};
constexpr Scalar kTargetSumTolerance = Scalar{1e-8};
#else
constexpr Scalar kProbabilityEpsilon = Scalar{1e-7f};
constexpr Scalar kTargetSumTolerance = Scalar{1e-4f};
#endif
constexpr std::size_t kMaxSerializedLayers = 1000000;
constexpr std::size_t kMaxSerializedScalars = 268435456;

[[nodiscard]] bool is_finite(Scalar value) {
    return std::isfinite(value);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void require_finite(Scalar value, const std::string& name) {
    require(is_finite(value), name + " must be finite");
}

[[nodiscard]] std::string utc_timestamp(std::uint64_t unix_time) {
    const std::time_t raw_time = static_cast<std::time_t>(unix_time);
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &raw_time) != 0) {
        return {};
    }
#else
    if (gmtime_r(&raw_time, &utc) == nullptr) {
        return {};
    }
#endif

    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

[[nodiscard]] std::string scalar_debug_string(Scalar value) {
    std::ostringstream output;
    output << std::setprecision(9) << value;
    return output.str();
}

[[nodiscard]] std::string non_finite_kind(Scalar value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value > Scalar{0} ? "+Inf" : "-Inf";
    }
    return "non-finite";
}

[[nodiscard]] std::string scalar_type_string() {
    return sizeof(Scalar) == sizeof(double) ? "float64" : "float32";
}

void write_json_escaped(std::ostream& output, std::string_view value) {
    output << '"';
    for (char ch : value) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        switch (ch) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (byte < 0x20) {
                output << "\\u"
                       << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(byte)
                       << std::dec << std::setfill(' ');
            } else {
                output << ch;
            }
            break;
        }
    }
    output << '"';
}

void write_json_indent(std::ostream& output, int spaces) {
    for (int i = 0; i < spaces; ++i) {
        output << ' ';
    }
}

void write_json_key(std::ostream& output, int indent, std::string_view key) {
    write_json_indent(output, indent);
    write_json_escaped(output, key);
    output << ": ";
}

void write_json_scalar(std::ostream& output, Scalar value) {
    if (!is_finite(value)) {
        output << "null";
        return;
    }
    output << std::setprecision(std::numeric_limits<Scalar>::max_digits10) << value;
}

void write_json_scalar_array(std::ostream& output, const Vector& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        write_json_scalar(output, values[i]);
    }
    output << ']';
}

void write_json_feature(std::ostream& output, const FeatureNormalization& feature, int indent) {
    output << "{\n";
    write_json_key(output, indent + 2, "mode");
    write_json_escaped(output, to_string(feature.mode));
    output << ",\n";
    write_json_key(output, indent + 2, "mean");
    write_json_scalar(output, feature.mean);
    output << ",\n";
    write_json_key(output, indent + 2, "stddev");
    write_json_scalar(output, feature.stddev);
    output << ",\n";
    write_json_key(output, indent + 2, "minimum");
    write_json_scalar(output, feature.minimum);
    output << ",\n";
    write_json_key(output, indent + 2, "maximum");
    write_json_scalar(output, feature.maximum);
    output << ",\n";
    write_json_key(output, indent + 2, "normalized_min");
    write_json_scalar(output, feature.normalized_min);
    output << ",\n";
    write_json_key(output, indent + 2, "normalized_max");
    write_json_scalar(output, feature.normalized_max);
    output << ",\n";
    write_json_key(output, indent + 2, "clamp");
    output << (feature.clamp ? "true" : "false");
    output << ",\n";
    write_json_key(output, indent + 2, "clamp_min");
    write_json_scalar(output, feature.clamp_min);
    output << ",\n";
    write_json_key(output, indent + 2, "clamp_max");
    write_json_scalar(output, feature.clamp_max);
    output << '\n';
    write_json_indent(output, indent);
    output << '}';
}

void write_json_features(std::ostream& output,
                         const std::vector<FeatureNormalization>& features,
                         int indent) {
    output << "[\n";
    for (std::size_t i = 0; i < features.size(); ++i) {
        write_json_indent(output, indent + 2);
        write_json_feature(output, features[i], indent + 2);
        if (i + 1 < features.size()) {
            output << ',';
        }
        output << '\n';
    }
    write_json_indent(output, indent);
    output << ']';
}

void write_json_optimizer(std::ostream& output, const OptimizerConfig& optimizer, int indent) {
    output << "{\n";
    write_json_key(output, indent + 2, "type");
    write_json_escaped(output, to_string(optimizer.type));
    output << ",\n";
    write_json_key(output, indent + 2, "learning_rate");
    write_json_scalar(output, optimizer.learning_rate);
    output << ",\n";
    write_json_key(output, indent + 2, "beta1");
    write_json_scalar(output, optimizer.beta1);
    output << ",\n";
    write_json_key(output, indent + 2, "beta2");
    write_json_scalar(output, optimizer.beta2);
    output << ",\n";
    write_json_key(output, indent + 2, "epsilon");
    write_json_scalar(output, optimizer.epsilon);
    output << ",\n";
    write_json_key(output, indent + 2, "momentum");
    write_json_scalar(output, optimizer.momentum);
    output << ",\n";
    write_json_key(output, indent + 2, "l1");
    write_json_scalar(output, optimizer.l1);
    output << ",\n";
    write_json_key(output, indent + 2, "l2");
    write_json_scalar(output, optimizer.l2);
    output << ",\n";
    write_json_key(output, indent + 2, "decoupled_weight_decay");
    write_json_scalar(output, optimizer.decoupled_weight_decay);
    output << ",\n";
    write_json_key(output, indent + 2, "max_norm");
    write_json_scalar(output, optimizer.max_norm);
    output << ",\n";
    write_json_key(output, indent + 2, "gradient_clip_norm");
    write_json_scalar(output, optimizer.gradient_clip_norm);
    output << ",\n";
    write_json_key(output, indent + 2, "gradient_clip_value");
    write_json_scalar(output, optimizer.gradient_clip_value);
    output << ",\n";
    write_json_key(output, indent + 2, "layer_gradient_clip_norm");
    write_json_scalar(output, optimizer.layer_gradient_clip_norm);
    output << '\n';
    write_json_indent(output, indent);
    output << '}';
}

void write_json_learning_rate_schedule(std::ostream& output,
                                       const LearningRateScheduleConfig& schedule,
                                       int indent) {
    output << "{\n";
    write_json_key(output, indent + 2, "type");
    write_json_escaped(output, to_string(schedule.type));
    output << ",\n";
    write_json_key(output, indent + 2, "base_learning_rate");
    write_json_scalar(output, schedule.base_learning_rate);
    output << ",\n";
    write_json_key(output, indent + 2, "minimum_learning_rate");
    write_json_scalar(output, schedule.minimum_learning_rate);
    output << ",\n";
    write_json_key(output, indent + 2, "warmup_epochs");
    output << schedule.warmup_epochs;
    output << ",\n";
    write_json_key(output, indent + 2, "warmup_start_learning_rate");
    write_json_scalar(output, schedule.warmup_start_learning_rate);
    output << ",\n";
    write_json_key(output, indent + 2, "step_size");
    output << schedule.step_size;
    output << ",\n";
    write_json_key(output, indent + 2, "step_decay_factor");
    write_json_scalar(output, schedule.step_decay_factor);
    output << ",\n";
    write_json_key(output, indent + 2, "exponential_decay_rate");
    write_json_scalar(output, schedule.exponential_decay_rate);
    output << ",\n";
    write_json_key(output, indent + 2, "cosine_epochs");
    output << schedule.cosine_epochs;
    output << ",\n";
    write_json_key(output, indent + 2, "reduce_on_plateau_monitor");
    write_json_escaped(output, to_string(schedule.reduce_on_plateau_monitor));
    output << ",\n";
    write_json_key(output, indent + 2, "reduce_on_plateau_mode");
    write_json_escaped(output, to_string(schedule.reduce_on_plateau_mode));
    output << ",\n";
    write_json_key(output, indent + 2, "reduce_on_plateau_patience");
    output << schedule.reduce_on_plateau_patience;
    output << ",\n";
    write_json_key(output, indent + 2, "reduce_on_plateau_factor");
    write_json_scalar(output, schedule.reduce_on_plateau_factor);
    output << ",\n";
    write_json_key(output, indent + 2, "reduce_on_plateau_min_delta");
    write_json_scalar(output, schedule.reduce_on_plateau_min_delta);
    output << ",\n";
    write_json_key(output, indent + 2, "reduce_on_plateau_cooldown");
    output << schedule.reduce_on_plateau_cooldown;
    output << '\n';
    write_json_indent(output, indent);
    output << '}';
}

void write_json_loss(std::ostream& output, const LossConfig& loss, int indent) {
    output << "{\n";
    write_json_key(output, indent + 2, "type");
    write_json_escaped(output, to_string(loss.type));
    output << ",\n";
    write_json_key(output, indent + 2, "huber_delta");
    write_json_scalar(output, loss.huber_delta);
    output << ",\n";
    write_json_key(output, indent + 2, "relative_epsilon");
    write_json_scalar(output, loss.relative_epsilon);
    output << ",\n";
    write_json_key(output, indent + 2, "weights");
    write_json_scalar_array(output, loss.weights);
    output << ",\n";
    write_json_key(output, indent + 2, "custom");
    output << (loss.type == Loss::custom ? "true" : "false");
    output << '\n';
    write_json_indent(output, indent);
    output << '}';
}

void write_json_normalization(std::ostream& output,
                              const NormalizationSpec& normalization,
                              int indent) {
    output << "{\n";
    write_json_key(output, indent + 2, "input_features");
    write_json_features(output, normalization.input_features, indent + 2);
    output << ",\n";
    write_json_key(output, indent + 2, "output_features");
    write_json_features(output, normalization.output_features, indent + 2);
    output << '\n';
    write_json_indent(output, indent);
    output << '}';
}

void write_json_training_options(std::ostream& output,
                                 const TrainingOptions& options,
                                 int indent) {
    output << "{\n";
    write_json_key(output, indent + 2, "epochs");
    output << options.epochs;
    output << ",\n";
    write_json_key(output, indent + 2, "batch_size");
    output << options.batch_size;
    output << ",\n";
    write_json_key(output, indent + 2, "shuffle");
    output << (options.shuffle ? "true" : "false");
    output << ",\n";
    write_json_key(output, indent + 2, "seed");
    output << options.seed;
    output << ",\n";
    write_json_key(output, indent + 2, "validation_split");
    write_json_scalar(output, options.validation_split);
    output << ",\n";
    write_json_key(output, indent + 2, "test_split");
    write_json_scalar(output, options.test_split);
    output << ",\n";
    write_json_key(output, indent + 2, "streaming_shuffle_buffer_size");
    output << options.streaming_shuffle_buffer_size;
    output << ",\n";
    write_json_key(output, indent + 2, "early_stopping_patience");
    output << options.early_stopping_patience;
    output << ",\n";
    write_json_key(output, indent + 2, "min_delta");
    write_json_scalar(output, options.min_delta);
    output << ",\n";
    write_json_key(output, indent + 2, "early_stopping_monitor");
    write_json_escaped(output, to_string(options.early_stopping_monitor));
    output << ",\n";
    write_json_key(output, indent + 2, "early_stopping_mode");
    write_json_escaped(output, to_string(options.early_stopping_mode));
    output << ",\n";
    write_json_key(output, indent + 2, "early_stopping_cooldown");
    output << options.early_stopping_cooldown;
    output << ",\n";
    write_json_key(output, indent + 2, "gradient_noise_stddev");
    write_json_scalar(output, options.gradient_noise_stddev);
    output << ",\n";
    write_json_key(output, indent + 2, "parallelism");
    output << "{\n";
    write_json_key(output, indent + 4, "execution");
    write_json_escaped(output, to_string(options.parallelism.execution));
    output << ",\n";
    write_json_key(output, indent + 4, "thread_count");
    output << options.parallelism.thread_count;
    output << ",\n";
    write_json_key(output, indent + 4, "minimum_parallel_samples");
    output << options.parallelism.minimum_parallel_samples;
    output << '\n';
    write_json_indent(output, indent + 2);
    output << "},\n";
    write_json_key(output, indent + 2, "debug");
    output << "{\n";
    write_json_key(output, indent + 4, "enabled");
    output << (options.debug.enabled ? "true" : "false");
    output << ",\n";
    write_json_key(output, indent + 4, "loss_explosion_factor");
    write_json_scalar(output, options.debug.loss_explosion_factor);
    output << ",\n";
    write_json_key(output, indent + 4, "loss_explosion_threshold");
    write_json_scalar(output, options.debug.loss_explosion_threshold);
    output << ",\n";
    write_json_key(output, indent + 4, "gradient_norm_explosion_factor");
    write_json_scalar(output, options.debug.gradient_norm_explosion_factor);
    output << ",\n";
    write_json_key(output, indent + 4, "gradient_norm_explosion_threshold");
    write_json_scalar(output, options.debug.gradient_norm_explosion_threshold);
    output << '\n';
    write_json_indent(output, indent + 2);
    output << "},\n";
    write_json_key(output, indent + 2, "restore_best_weights");
    output << (options.restore_best_weights ? "true" : "false");
    output << ",\n";
    write_json_key(output, indent + 2, "best_checkpoint_path");
    write_json_escaped(output, options.best_checkpoint_path.generic_string());
    output << ",\n";
    write_json_key(output, indent + 2, "latest_checkpoint_path");
    write_json_escaped(output, options.latest_checkpoint_path.generic_string());
    output << '\n';
    write_json_indent(output, indent);
    output << '}';
}

void refresh_training_run_result(TrainingHistory& history, std::size_t completed_epochs) {
    TrainingRunResultConfig& result = history.run_config.result;
    result.completed_epochs = completed_epochs;
    result.has_best_score = history.has_best_checkpoint;
    result.best_epoch = history.best_epoch;
    result.best_score = history.best_metric;
    result.monitor = history.monitor;
    result.monitor_mode = history.monitor_mode;
    result.stop_reason = history.stop_reason;
    result.stop_message = history.stop_message;

    result.has_final_training_loss = !history.training_loss.empty();
    result.final_training_loss = result.has_final_training_loss
                                     ? history.training_loss.back()
                                     : Scalar{0};
    result.has_final_validation_loss = !history.validation_loss.empty();
    result.final_validation_loss = result.has_final_validation_loss
                                       ? history.validation_loss.back()
                                       : Scalar{0};
    result.has_final_test_loss = !history.test_loss.empty();
    result.final_test_loss = result.has_final_test_loss
                                 ? history.test_loss.back()
                                 : Scalar{0};
}

struct TrainingDebugFailure : public std::runtime_error {
    TrainingStopReason reason;

    TrainingDebugFailure(TrainingStopReason stop_reason, std::string message)
        : std::runtime_error(std::move(message)),
          reason(stop_reason) {}
};

[[noreturn]] void throw_training_debug_failure(TrainingStopReason reason, const std::string& message) {
    throw TrainingDebugFailure(reason, message);
}

[[nodiscard]] Scalar clamp_probability(Scalar value) {
    if (value < kProbabilityEpsilon) {
        return kProbabilityEpsilon;
    }
    if (value > Scalar{1} - kProbabilityEpsilon) {
        return Scalar{1} - kProbabilityEpsilon;
    }
    return value;
}

[[nodiscard]] Scalar sigmoid(Scalar value) noexcept {
    if (value >= Scalar{0}) {
        const Scalar z = static_cast<Scalar>(std::exp(-value));
        return Scalar{1} / (Scalar{1} + z);
    }
    const Scalar z = static_cast<Scalar>(std::exp(value));
    return z / (Scalar{1} + z);
}

[[nodiscard]] Vector apply_activation(const Vector& values, Activation activation) {
    Vector result(values.size());

    switch (activation) {
    case Activation::linear:
        return values;
    case Activation::relu:
        std::transform(values.begin(), values.end(), result.begin(), [](Scalar v) {
            return std::max<Scalar>(Scalar{0}, v);
        });
        return result;
    case Activation::sigmoid:
        std::transform(values.begin(), values.end(), result.begin(), sigmoid);
        return result;
    case Activation::tanh:
        std::transform(values.begin(), values.end(), result.begin(), [](Scalar v) {
            return static_cast<Scalar>(std::tanh(v));
        });
        return result;
    case Activation::softmax: {
        const Scalar max_value = *std::max_element(values.begin(), values.end());
        Scalar sum = Scalar{0};
        for (std::size_t i = 0; i < values.size(); ++i) {
            result[i] = static_cast<Scalar>(std::exp(values[i] - max_value));
            sum += result[i];
        }
        for (Scalar& value : result) {
            value /= sum;
        }
        return result;
    }
    }

    throw std::invalid_argument("unknown activation");
}

[[nodiscard]] Scalar activation_derivative(Activation activation, Scalar pre_activation, Scalar activation_value) {
    switch (activation) {
    case Activation::linear:
        return Scalar{1};
    case Activation::relu:
        return pre_activation > Scalar{0} ? Scalar{1} : Scalar{0};
    case Activation::sigmoid:
        return activation_value * (Scalar{1} - activation_value);
    case Activation::tanh:
        return Scalar{1} - activation_value * activation_value;
    case Activation::softmax:
        throw std::invalid_argument("softmax derivative requires the full Jacobian");
    }

    throw std::invalid_argument("unknown activation");
}

[[nodiscard]] Scalar activation_derivative_from_pre_activation(Activation activation, Scalar pre_activation) {
    switch (activation) {
    case Activation::linear:
        return Scalar{1};
    case Activation::relu:
        return pre_activation > Scalar{0} ? Scalar{1} : Scalar{0};
    case Activation::sigmoid: {
        const Scalar value = sigmoid(pre_activation);
        return value * (Scalar{1} - value);
    }
    case Activation::tanh: {
        const Scalar value = static_cast<Scalar>(std::tanh(pre_activation));
        return Scalar{1} - value * value;
    }
    case Activation::softmax:
        throw std::invalid_argument("softmax derivative requires the full Jacobian");
    }

    throw std::invalid_argument("unknown activation");
}

void validate_optimizer(const OptimizerConfig& optimizer) {
    require_finite(optimizer.learning_rate, "learning_rate");
    require(optimizer.learning_rate > Scalar{0}, "learning_rate must be positive");
    require_finite(optimizer.beta1, "beta1");
    require_finite(optimizer.beta2, "beta2");
    require(optimizer.beta1 >= Scalar{0} && optimizer.beta1 < Scalar{1}, "beta1 must be in [0, 1)");
    require(optimizer.beta2 >= Scalar{0} && optimizer.beta2 < Scalar{1}, "beta2 must be in [0, 1)");
    require_finite(optimizer.epsilon, "epsilon");
    require(optimizer.epsilon > Scalar{0}, "epsilon must be positive");
    require_finite(optimizer.momentum, "momentum");
    require(optimizer.momentum >= Scalar{0} && optimizer.momentum < Scalar{1}, "momentum must be in [0, 1)");
    require_finite(optimizer.l2, "l2");
    require(optimizer.l2 >= Scalar{0}, "l2 must be non-negative");
    require_finite(optimizer.l1, "l1");
    require(optimizer.l1 >= Scalar{0}, "l1 must be non-negative");
    require_finite(optimizer.decoupled_weight_decay, "decoupled_weight_decay");
    require(optimizer.decoupled_weight_decay >= Scalar{0}, "decoupled_weight_decay must be non-negative");
    require_finite(optimizer.max_norm, "max_norm");
    require(optimizer.max_norm >= Scalar{0}, "max_norm must be non-negative");
    require_finite(optimizer.gradient_clip_norm, "gradient_clip_norm");
    require(optimizer.gradient_clip_norm >= Scalar{0}, "gradient_clip_norm must be non-negative");
    require_finite(optimizer.gradient_clip_value, "gradient_clip_value");
    require(optimizer.gradient_clip_value >= Scalar{0}, "gradient_clip_value must be non-negative");
    require_finite(optimizer.layer_gradient_clip_norm, "layer_gradient_clip_norm");
    require(optimizer.layer_gradient_clip_norm >= Scalar{0},
            "layer_gradient_clip_norm must be non-negative");
}

void validate_learning_rate_schedule_config(const LearningRateScheduleConfig& schedule) {
    require_finite(schedule.base_learning_rate, "base_learning_rate");
    require(schedule.base_learning_rate >= Scalar{0}, "base_learning_rate must be non-negative");
    require_finite(schedule.minimum_learning_rate, "minimum_learning_rate");
    require(schedule.minimum_learning_rate >= Scalar{0}, "minimum_learning_rate must be non-negative");
    require_finite(schedule.warmup_start_learning_rate, "warmup_start_learning_rate");
    require(schedule.warmup_start_learning_rate >= Scalar{0},
            "warmup_start_learning_rate must be non-negative");
    require(schedule.step_size > 0, "step_size must be positive");
    require_finite(schedule.step_decay_factor, "step_decay_factor");
    require(schedule.step_decay_factor > Scalar{0} && schedule.step_decay_factor <= Scalar{1},
            "step_decay_factor must be in (0, 1]");
    require_finite(schedule.exponential_decay_rate, "exponential_decay_rate");
    require(schedule.exponential_decay_rate > Scalar{0} && schedule.exponential_decay_rate <= Scalar{1},
            "exponential_decay_rate must be in (0, 1]");
    require_finite(schedule.reduce_on_plateau_factor, "reduce_on_plateau_factor");
    require(schedule.reduce_on_plateau_factor > Scalar{0} && schedule.reduce_on_plateau_factor < Scalar{1},
            "reduce_on_plateau_factor must be in (0, 1)");
    require_finite(schedule.reduce_on_plateau_min_delta, "reduce_on_plateau_min_delta");
    require(schedule.reduce_on_plateau_min_delta >= Scalar{0},
            "reduce_on_plateau_min_delta must be non-negative");
}

void validate_initialization_config(const InitializationConfig& initialization) {
    require_finite(initialization.bias, "initialization bias");
}

[[nodiscard]] std::size_t resolve_parallel_worker_count(const ParallelOptions& options,
                                                        std::size_t sample_count) noexcept {
    if (sample_count == 0 || options.execution != ParallelExecution::worker_threads) {
        return sample_count == 0 ? 0 : 1;
    }
    if (options.minimum_parallel_samples > 0 && sample_count < options.minimum_parallel_samples) {
        return 1;
    }

    std::size_t worker_count = options.thread_count;
    if (worker_count == 0) {
        worker_count = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (worker_count == 0) {
        worker_count = 1;
    }
    return std::min(worker_count, sample_count);
}

[[nodiscard]] std::size_t required_batch_scratch_size(std::size_t single_sample_scratch,
                                                      std::size_t sample_count,
                                                      const ParallelOptions& options) noexcept {
    const std::size_t worker_count = resolve_parallel_worker_count(options, sample_count);
    if (single_sample_scratch != 0 &&
        worker_count > std::numeric_limits<std::size_t>::max() / single_sample_scratch) {
        return std::numeric_limits<std::size_t>::max();
    }
    return single_sample_scratch * worker_count;
}

void join_workers_noexcept(std::vector<std::thread>& workers) noexcept {
    for (std::thread& worker : workers) {
        if (!worker.joinable()) {
            continue;
        }
        try {
            worker.join();
        } catch (...) {
            if (worker.joinable()) {
                try {
                    worker.detach();
                } catch (...) {
                }
            }
        }
    }
}

void validate_training_options(const TrainingOptions& options) {
    require(options.epochs > 0, "epochs must be positive");
    require(options.batch_size > 0, "batch_size must be positive");
    require_finite(options.validation_split, "validation_split");
    require(options.validation_split >= Scalar{0} && options.validation_split < Scalar{1},
            "validation_split must be in [0, 1)");
    require_finite(options.test_split, "test_split");
    require(options.test_split >= Scalar{0} && options.test_split < Scalar{1}, "test_split must be in [0, 1)");
    require(options.validation_split + options.test_split < Scalar{1},
            "validation_split + test_split must be less than 1");
    require_finite(options.min_delta, "min_delta");
    require(options.min_delta >= Scalar{0}, "min_delta must be non-negative");
    validate_learning_rate_schedule_config(options.learning_rate_schedule);
    require_finite(options.gradient_noise_stddev, "gradient_noise_stddev");
    require(options.gradient_noise_stddev >= Scalar{0}, "gradient_noise_stddev must be non-negative");
    require_finite(options.debug.loss_explosion_factor, "debug.loss_explosion_factor");
    require(options.debug.loss_explosion_factor >= Scalar{1},
            "debug.loss_explosion_factor must be at least 1");
    require_finite(options.debug.loss_explosion_threshold, "debug.loss_explosion_threshold");
    require(options.debug.loss_explosion_threshold > Scalar{0},
            "debug.loss_explosion_threshold must be positive");
    require_finite(options.debug.gradient_norm_explosion_factor, "debug.gradient_norm_explosion_factor");
    require(options.debug.gradient_norm_explosion_factor >= Scalar{1},
            "debug.gradient_norm_explosion_factor must be at least 1");
    require_finite(options.debug.gradient_norm_explosion_threshold, "debug.gradient_norm_explosion_threshold");
    require(options.debug.gradient_norm_explosion_threshold > Scalar{0},
            "debug.gradient_norm_explosion_threshold must be positive");
}

void validate_dropout_probability(Scalar probability, const std::string& name) {
    require_finite(probability, name);
    require(probability >= Scalar{0} && probability < Scalar{1}, name + " must be in [0, 1)");
}

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

struct TrainingDebugScalarTracker {
    bool has_value{false};
    Scalar value{Scalar{0}};
};

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
                                                  const std::string& extra = {}) {
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

[[nodiscard]] WeightInitialization resolve_weight_initialization(WeightInitialization initialization,
                                                                 Activation activation) {
    if (initialization != WeightInitialization::automatic) {
        return initialization;
    }
    return activation == Activation::relu ? WeightInitialization::he_uniform
                                          : WeightInitialization::xavier_uniform;
}

struct WeightInitializationParameters {
    bool normal{false};
    Scalar scale{static_cast<Scalar>(0)};
};

[[nodiscard]] WeightInitializationParameters weight_initialization_parameters(
    WeightInitialization initialization,
    std::size_t fan_in,
    std::size_t fan_out) {
    require(fan_in > 0, "fan_in must be positive");
    require(fan_out > 0, "fan_out must be positive");

    const Scalar fan_in_scalar = static_cast<Scalar>(fan_in);
    const Scalar fan_sum = static_cast<Scalar>(fan_in + fan_out);
    switch (initialization) {
    case WeightInitialization::automatic:
        throw std::invalid_argument("automatic initialization must be resolved before use");
    case WeightInitialization::he_normal:
        return WeightInitializationParameters{true,
                                              static_cast<Scalar>(std::sqrt(Scalar{2} / fan_in_scalar))};
    case WeightInitialization::he_uniform:
        return WeightInitializationParameters{false,
                                              static_cast<Scalar>(std::sqrt(Scalar{6} / fan_in_scalar))};
    case WeightInitialization::xavier_normal:
        return WeightInitializationParameters{true,
                                              static_cast<Scalar>(std::sqrt(Scalar{2} / fan_sum))};
    case WeightInitialization::xavier_uniform:
        return WeightInitializationParameters{false,
                                              static_cast<Scalar>(std::sqrt(Scalar{6} / fan_sum))};
    case WeightInitialization::lecun_normal:
        return WeightInitializationParameters{true,
                                              static_cast<Scalar>(std::sqrt(Scalar{1} / fan_in_scalar))};
    case WeightInitialization::lecun_uniform:
        return WeightInitializationParameters{false,
                                              static_cast<Scalar>(std::sqrt(Scalar{3} / fan_in_scalar))};
    }

    throw std::invalid_argument("unknown weight initialization");
}

[[nodiscard]] Scalar sign_or_zero(Scalar value) noexcept {
    if (value > Scalar{0}) {
        return Scalar{1};
    }
    if (value < Scalar{0}) {
        return -Scalar{1};
    }
    return Scalar{0};
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

void initialize_weights(Vector& weights,
                        WeightInitialization initialization,
                        std::size_t fan_in,
                        std::size_t fan_out,
                        std::mt19937_64& rng) {
    const WeightInitializationParameters parameters =
        weight_initialization_parameters(initialization, fan_in, fan_out);
    if (parameters.normal) {
        std::normal_distribution<Scalar> distribution(Scalar{0}, parameters.scale);
        for (Scalar& weight : weights) {
            weight = distribution(rng);
        }
        return;
    }

    std::uniform_real_distribution<Scalar> distribution(-parameters.scale, parameters.scale);
    for (Scalar& weight : weights) {
        weight = distribution(rng);
    }
}

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

[[nodiscard]] Scalar warmup_learning_rate(const LearningRateScheduleConfig& schedule,
                                          std::size_t epoch) {
    const Scalar progress = static_cast<Scalar>(epoch + 1) / static_cast<Scalar>(schedule.warmup_epochs);
    return schedule.warmup_start_learning_rate +
           (schedule.base_learning_rate - schedule.warmup_start_learning_rate) * progress;
}

[[nodiscard]] Scalar floor_learning_rate(const LearningRateScheduleConfig& schedule,
                                         Scalar learning_rate) {
    return std::max(schedule.minimum_learning_rate, learning_rate);
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

[[nodiscard]] bool is_classification_loss(Loss loss) noexcept {
    return loss == Loss::binary_cross_entropy || loss == Loss::categorical_cross_entropy;
}

void validate_loss_config(const LossConfig& loss, std::size_t output_size) {
    require_finite(loss.huber_delta, "huber_delta");
    require(loss.huber_delta > Scalar{0}, "huber_delta must be positive");
    require_finite(loss.relative_epsilon, "relative_epsilon");
    require(loss.relative_epsilon > Scalar{0}, "relative_epsilon must be positive");

    for (Scalar weight : loss.weights) {
        require_finite(weight, "loss weight");
        require(weight >= Scalar{0}, "loss weights must be non-negative");
    }

    switch (loss.type) {
    case Loss::mean_squared_error:
    case Loss::mean_absolute_error:
    case Loss::huber:
    case Loss::relative_mean_squared_error:
    case Loss::log_cosh:
    case Loss::binary_cross_entropy:
    case Loss::categorical_cross_entropy:
        require(loss.weights.empty(), "loss weights are only supported with weighted_mean_squared_error");
        return;
    case Loss::weighted_mean_squared_error: {
        require(loss.weights.size() == output_size, "weighted_mean_squared_error weight count mismatch");
        const Scalar total_weight = std::accumulate(loss.weights.begin(), loss.weights.end(), Scalar{0});
        require(total_weight > Scalar{0}, "weighted_mean_squared_error requires at least one positive weight");
        return;
    }
    case Loss::custom:
        require(loss.weights.empty(), "custom loss does not use built-in weights");
        require(loss.custom_loss.value != nullptr, "custom loss value callback is required");
        require(loss.custom_loss.gradient != nullptr, "custom loss gradient callback is required");
        return;
    }

    throw std::invalid_argument("unknown loss");
}

void validate_matrix(const Matrix& matrix,
                     std::size_t expected_columns,
                     const std::string& name,
                     bool allow_empty = false) {
    require(allow_empty || !matrix.empty(), name + " must not be empty");
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        require(matrix[row].size() == expected_columns,
                name + " row " + std::to_string(row) + " has an invalid column count");
        for (std::size_t column = 0; column < matrix[row].size(); ++column) {
            require_finite(matrix[row][column],
                           name + " row " + std::to_string(row) +
                               " column " + std::to_string(column));
        }
    }
}

void validate_dense_matrix(const DenseMatrix& matrix,
                           std::size_t expected_columns,
                           const std::string& name,
                           bool allow_empty = false) {
    require(allow_empty || !matrix.empty(), name + " must not be empty");
    if (matrix.empty()) {
        require(matrix.columns() == expected_columns || matrix.columns() == 0,
                name + " has an invalid column count");
        return;
    }
    require(matrix.columns() == expected_columns, name + " has an invalid column count");
    require(matrix.scalar_count() == matrix.rows() * matrix.columns(), name + " storage size mismatch");
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            require_finite(matrix(row, column),
                           name + " row " + std::to_string(row) +
                               " column " + std::to_string(column));
        }
    }
}

void validate_target_for_loss(const Scalar* target, std::size_t target_size, Loss loss) {
    require(target != nullptr || target_size == 0, "target buffer must not be null");
    switch (loss) {
    case Loss::mean_squared_error:
    case Loss::mean_absolute_error:
    case Loss::huber:
    case Loss::relative_mean_squared_error:
    case Loss::log_cosh:
    case Loss::weighted_mean_squared_error:
    case Loss::custom:
        return;
    case Loss::binary_cross_entropy:
        for (std::size_t i = 0; i < target_size; ++i) {
            const Scalar value = target[i];
            require(value >= Scalar{0} && value <= Scalar{1},
                    "binary_cross_entropy targets must be in [0, 1]");
        }
        return;
    case Loss::categorical_cross_entropy: {
        Scalar sum = Scalar{0};
        for (std::size_t i = 0; i < target_size; ++i) {
            const Scalar value = target[i];
            require(value >= Scalar{0}, "categorical_cross_entropy targets must be non-negative");
            sum += value;
        }
        require(std::abs(sum - Scalar{1}) <= kTargetSumTolerance,
                "categorical_cross_entropy targets must sum to 1");
        return;
    }
    }

    throw std::invalid_argument("unknown loss");
}

void validate_target_for_loss(const Vector& target, Loss loss) {
    validate_target_for_loss(target.data(), target.size(), loss);
}

void validate_targets_for_loss(const Matrix& targets, Loss loss) {
    for (const Vector& target : targets) {
        validate_target_for_loss(target, loss);
    }
}

void validate_targets_for_loss(const DenseMatrix& targets, Loss loss) {
    for (std::size_t row = 0; row < targets.rows(); ++row) {
        validate_target_for_loss(targets.row_data(row), targets.columns(), loss);
    }
}

void validate_training_data(const Matrix& inputs,
                            const Matrix& targets,
                            std::size_t input_size,
                            std::size_t output_size,
                            const LossConfig& loss) {
    validate_matrix(inputs, input_size, "inputs");
    validate_matrix(targets, output_size, "targets");
    require(inputs.size() == targets.size(), "inputs and targets must have the same number of rows");
    validate_targets_for_loss(targets, loss.type);
}

void validate_training_data(const DenseMatrix& inputs,
                            const DenseMatrix& targets,
                            std::size_t input_size,
                            std::size_t output_size,
                            const LossConfig& loss) {
    validate_dense_matrix(inputs, input_size, "inputs");
    validate_dense_matrix(targets, output_size, "targets");
    require(inputs.rows() == targets.rows(), "inputs and targets must have the same number of rows");
    validate_targets_for_loss(targets, loss.type);
}

void validate_dataset_shape(std::size_t sample_count,
                            std::size_t dataset_input_size,
                            std::size_t dataset_output_size,
                            std::size_t expected_input_size,
                            std::size_t expected_output_size) {
    require(sample_count > 0, "dataset must not be empty");
    require(dataset_input_size == expected_input_size, "dataset input_size mismatch");
    require(dataset_output_size == expected_output_size, "dataset output_size mismatch");
}

void validate_sample_values(const Vector& input, const Vector& target, Loss loss) {
    for (Scalar value : input) {
        require_finite(value, "dataset input value");
    }
    for (Scalar value : target) {
        require_finite(value, "dataset target value");
    }
    validate_target_for_loss(target, loss);
}

void validate_sample_values(const Scalar* input,
                            std::size_t input_size,
                            const Scalar* target,
                            std::size_t target_size,
                            Loss loss) {
    require(input != nullptr || input_size == 0, "dataset input buffer must not be null");
    require(target != nullptr || target_size == 0, "dataset target buffer must not be null");
    for (std::size_t i = 0; i < input_size; ++i) {
        require_finite(input[i], "dataset input value");
    }
    for (std::size_t i = 0; i < target_size; ++i) {
        require_finite(target[i], "dataset target value");
    }
    validate_target_for_loss(target, target_size, loss);
}

struct TrainingDebugContext {
    const char* phase{"training"};
    std::size_t epoch{0};
    bool has_epoch{false};
    std::size_t sample_index{0};
    bool has_sample_index{false};
};

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
    require(values != nullptr || size == 0, std::string(label) + " buffer must not be null");
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

void debug_check_sample_values(const Scalar* input,
                               std::size_t input_size,
                               const Scalar* target,
                               std::size_t target_size,
                               const TrainingDebugContext& context) {
    debug_check_finite_buffer(input, input_size, "input", context);
    debug_check_finite_buffer(target, target_size, "target", context);
}

void validate_dataset_split_options(const DatasetSplitOptions& options) {
    require_finite(options.validation_split, "validation_split");
    require_finite(options.test_split, "test_split");
    require(options.validation_split >= Scalar{0} && options.validation_split < Scalar{1},
            "validation_split must be in [0, 1)");
    require(options.test_split >= Scalar{0} && options.test_split < Scalar{1}, "test_split must be in [0, 1)");
    require(options.validation_split + options.test_split < Scalar{1},
            "validation_split + test_split must be less than 1");
}

[[nodiscard]] bool is_regression_metric(Metric metric) noexcept {
    return metric == Metric::mean_squared_error ||
           metric == Metric::mean_absolute_error ||
           metric == Metric::root_mean_squared_error ||
           metric == Metric::r2_score;
}

[[nodiscard]] bool is_classification_metric(Metric metric) noexcept {
    return metric == Metric::accuracy ||
           metric == Metric::precision ||
           metric == Metric::recall ||
           metric == Metric::f1_score ||
           metric == Metric::confusion_matrix;
}

[[nodiscard]] bool contains_metric(const std::vector<Metric>& metrics, Metric metric) {
    return std::find(metrics.begin(), metrics.end(), metric) != metrics.end();
}

void validate_evaluation_options(const EvaluationOptions& options) {
    require_finite(options.classification_threshold, "classification_threshold");
    require(options.classification_threshold >= Scalar{0} && options.classification_threshold <= Scalar{1},
            "classification_threshold must be in [0, 1]");

    for (std::size_t i = 0; i < options.metrics.size(); ++i) {
        require(options.metrics[i] != Metric::custom, "custom metrics must be registered with add_custom_metric");
        for (std::size_t j = i + 1; j < options.metrics.size(); ++j) {
            require(options.metrics[i] != options.metrics[j], "duplicate metric requested");
        }
    }

    for (std::size_t i = 0; i < options.custom_metrics.size(); ++i) {
        const CustomMetric& metric = options.custom_metrics[i];
        require(!metric.name.empty(), "custom metric name must not be empty");
        require(metric.evaluate != nullptr, "custom metric callback is required");
        for (std::size_t j = i + 1; j < options.custom_metrics.size(); ++j) {
            require(metric.name != options.custom_metrics[j].name, "duplicate custom metric name");
        }
    }
}

[[nodiscard]] std::vector<Metric> selected_metrics(const EvaluationOptions& options) {
    if (!options.metrics.empty() || !options.custom_metrics.empty()) {
        return options.metrics;
    }
    return EvaluationOptions::regression().metrics;
}

void validate_evaluation_data(const Matrix& predictions,
                              const Matrix& targets) {
    validate_matrix(predictions, predictions.empty() ? 0 : predictions.front().size(), "predictions");
    require(!predictions.front().empty(), "predictions must have at least one column");
    validate_matrix(targets, predictions.front().size(), "targets");
    require(predictions.size() == targets.size(), "predictions and targets must have the same number of rows");
}

void validate_evaluation_data(const DenseMatrix& predictions,
                              const DenseMatrix& targets) {
    validate_dense_matrix(predictions, predictions.columns(), "predictions");
    require(predictions.columns() > 0, "predictions must have at least one column");
    validate_dense_matrix(targets, predictions.columns(), "targets");
    require(predictions.rows() == targets.rows(), "predictions and targets must have the same number of rows");
}

[[nodiscard]] std::size_t argmax_index(const Vector& values) {
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

[[nodiscard]] std::size_t argmax_index(const Scalar* values, std::size_t size) {
    require(values != nullptr, "values buffer must not be null");
    return static_cast<std::size_t>(std::distance(values, std::max_element(values, values + size)));
}

[[nodiscard]] std::size_t class_count_for_output_size(std::size_t output_size) noexcept {
    return output_size == 1 ? std::size_t{2} : output_size;
}

[[nodiscard]] std::size_t class_label_from_row(const Vector& row, Scalar threshold) {
    if (row.size() == 1) {
        return row[0] >= threshold ? std::size_t{1} : std::size_t{0};
    }
    return argmax_index(row);
}

[[nodiscard]] std::size_t class_label_from_row(const Scalar* row, std::size_t size, Scalar threshold) {
    require(row != nullptr, "classification row buffer must not be null");
    if (size == 1) {
        return row[0] >= threshold ? std::size_t{1} : std::size_t{0};
    }
    return argmax_index(row, size);
}

[[nodiscard]] ConfusionMatrix make_confusion_matrix(const Matrix& predictions,
                                                    const Matrix& targets,
                                                    Scalar threshold) {
    const std::size_t class_count = class_count_for_output_size(predictions.front().size());
    ConfusionMatrix matrix;
    matrix.counts.assign(class_count, std::vector<std::size_t>(class_count, 0));

    for (std::size_t row = 0; row < predictions.size(); ++row) {
        const std::size_t actual = class_label_from_row(targets[row], threshold);
        const std::size_t predicted = class_label_from_row(predictions[row], threshold);
        ++matrix.counts[actual][predicted];
    }
    return matrix;
}

[[nodiscard]] ConfusionMatrix make_confusion_matrix(const DenseMatrix& predictions,
                                                    const DenseMatrix& targets,
                                                    Scalar threshold) {
    const std::size_t output_size = predictions.columns();
    const std::size_t class_count = class_count_for_output_size(output_size);
    ConfusionMatrix matrix;
    matrix.counts.assign(class_count, std::vector<std::size_t>(class_count, 0));

    for (std::size_t row = 0; row < predictions.rows(); ++row) {
        const std::size_t actual = class_label_from_row(targets.row_data(row), output_size, threshold);
        const std::size_t predicted = class_label_from_row(predictions.row_data(row), output_size, threshold);
        ++matrix.counts[actual][predicted];
    }
    return matrix;
}

[[nodiscard]] Scalar ratio_or_zero(std::size_t numerator, std::size_t denominator) noexcept {
    return denominator == 0 ? Scalar{0} : static_cast<Scalar>(numerator) / static_cast<Scalar>(denominator);
}

struct ClassificationSummary {
    Scalar accuracy{Scalar{0}};
    Scalar precision{Scalar{0}};
    Scalar recall{Scalar{0}};
    Scalar f1_score{Scalar{0}};
};

struct ClassCounts {
    std::size_t true_positive{0};
    std::size_t false_positive{0};
    std::size_t false_negative{0};
};

[[nodiscard]] ClassCounts counts_for_class(const ConfusionMatrix& matrix, std::size_t class_index) {
    ClassCounts counts;
    counts.true_positive = matrix.counts[class_index][class_index];

    for (std::size_t i = 0; i < matrix.counts.size(); ++i) {
        if (i != class_index) {
            counts.false_positive += matrix.counts[i][class_index];
            counts.false_negative += matrix.counts[class_index][i];
        }
    }
    return counts;
}

[[nodiscard]] Scalar precision_from_counts(const ClassCounts& counts) noexcept {
    return ratio_or_zero(counts.true_positive, counts.true_positive + counts.false_positive);
}

[[nodiscard]] Scalar recall_from_counts(const ClassCounts& counts) noexcept {
    return ratio_or_zero(counts.true_positive, counts.true_positive + counts.false_negative);
}

[[nodiscard]] Scalar f1_from_precision_recall(Scalar precision, Scalar recall) noexcept {
    return precision + recall == Scalar{0} ? Scalar{0} : Scalar{2} * precision * recall / (precision + recall);
}

[[nodiscard]] ClassificationSummary summarize_classification(const ConfusionMatrix& matrix,
                                                             const EvaluationOptions& options) {
    const std::size_t class_count = matrix.counts.size();
    require(class_count > 0, "confusion matrix must not be empty");
    require(options.positive_class < class_count, "positive_class is out of range");

    std::size_t correct = 0;
    std::size_t total = 0;
    for (std::size_t actual = 0; actual < class_count; ++actual) {
        for (std::size_t predicted = 0; predicted < class_count; ++predicted) {
            const std::size_t count = matrix.counts[actual][predicted];
            total += count;
            if (actual == predicted) {
                correct += count;
            }
        }
    }

    ClassificationSummary summary;
    summary.accuracy = ratio_or_zero(correct, total);

    if (options.averaging == Averaging::binary) {
        const ClassCounts counts = counts_for_class(matrix, options.positive_class);
        summary.precision = precision_from_counts(counts);
        summary.recall = recall_from_counts(counts);
        summary.f1_score = f1_from_precision_recall(summary.precision, summary.recall);
        return summary;
    }

    if (options.averaging == Averaging::micro) {
        ClassCounts counts;
        for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
            const ClassCounts current = counts_for_class(matrix, class_index);
            counts.true_positive += current.true_positive;
            counts.false_positive += current.false_positive;
            counts.false_negative += current.false_negative;
        }
        summary.precision = precision_from_counts(counts);
        summary.recall = recall_from_counts(counts);
        summary.f1_score = f1_from_precision_recall(summary.precision, summary.recall);
        return summary;
    }

    Scalar precision_total = Scalar{0};
    Scalar recall_total = Scalar{0};
    Scalar f1_total = Scalar{0};
    for (std::size_t class_index = 0; class_index < class_count; ++class_index) {
        const ClassCounts counts = counts_for_class(matrix, class_index);
        const Scalar precision = precision_from_counts(counts);
        const Scalar recall = recall_from_counts(counts);
        precision_total += precision;
        recall_total += recall;
        f1_total += f1_from_precision_recall(precision, recall);
    }

    const Scalar scale = Scalar{1} / static_cast<Scalar>(class_count);
    summary.precision = precision_total * scale;
    summary.recall = recall_total * scale;
    summary.f1_score = f1_total * scale;
    return summary;
}

struct RegressionSummary {
    Scalar mean_squared_error{Scalar{0}};
    Scalar mean_absolute_error{Scalar{0}};
    Scalar root_mean_squared_error{Scalar{0}};
    Scalar r2_score{Scalar{0}};
};

[[nodiscard]] RegressionSummary summarize_regression(const Matrix& predictions, const Matrix& targets) {
    Scalar squared_error = Scalar{0};
    Scalar absolute_error = Scalar{0};
    Scalar target_sum = Scalar{0};
    std::size_t count = 0;

    for (std::size_t row = 0; row < predictions.size(); ++row) {
        for (std::size_t column = 0; column < predictions[row].size(); ++column) {
            const Scalar delta = predictions[row][column] - targets[row][column];
            squared_error += delta * delta;
            absolute_error += static_cast<Scalar>(std::abs(delta));
            target_sum += targets[row][column];
            ++count;
        }
    }

    const Scalar count_scalar = static_cast<Scalar>(count);
    const Scalar mean_target = target_sum / count_scalar;
    Scalar target_variance_sum = Scalar{0};
    for (const Vector& target : targets) {
        for (Scalar value : target) {
            const Scalar centered = value - mean_target;
            target_variance_sum += centered * centered;
        }
    }

    RegressionSummary summary;
    summary.mean_squared_error = squared_error / count_scalar;
    summary.mean_absolute_error = absolute_error / count_scalar;
    summary.root_mean_squared_error = static_cast<Scalar>(std::sqrt(summary.mean_squared_error));
    if (target_variance_sum == Scalar{0}) {
        summary.r2_score = squared_error == Scalar{0} ? Scalar{1} : Scalar{0};
    } else {
        summary.r2_score = Scalar{1} - squared_error / target_variance_sum;
    }
    return summary;
}

[[nodiscard]] RegressionSummary summarize_regression(const DenseMatrix& predictions, const DenseMatrix& targets) {
    Scalar squared_error = Scalar{0};
    Scalar absolute_error = Scalar{0};
    Scalar target_sum = Scalar{0};
    const std::size_t count = predictions.scalar_count();

    for (std::size_t i = 0; i < count; ++i) {
        const Scalar delta = predictions.values()[i] - targets.values()[i];
        squared_error += delta * delta;
        absolute_error += static_cast<Scalar>(std::abs(delta));
        target_sum += targets.values()[i];
    }

    const Scalar count_scalar = static_cast<Scalar>(count);
    const Scalar mean_target = target_sum / count_scalar;
    Scalar target_variance_sum = Scalar{0};
    for (Scalar value : targets.values()) {
        const Scalar centered = value - mean_target;
        target_variance_sum += centered * centered;
    }

    RegressionSummary summary;
    summary.mean_squared_error = squared_error / count_scalar;
    summary.mean_absolute_error = absolute_error / count_scalar;
    summary.root_mean_squared_error = static_cast<Scalar>(std::sqrt(summary.mean_squared_error));
    if (target_variance_sum == Scalar{0}) {
        summary.r2_score = squared_error == Scalar{0} ? Scalar{1} : Scalar{0};
    } else {
        summary.r2_score = Scalar{1} - squared_error / target_variance_sum;
    }
    return summary;
}

[[nodiscard]] std::vector<std::size_t> make_indices(std::size_t size) {
    std::vector<std::size_t> indices(size);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    return indices;
}

[[nodiscard]] std::uint64_t non_zero_seed(std::uint64_t requested, std::uint64_t fallback) {
    return requested == 0 ? fallback : requested;
}

[[nodiscard]] std::string read_token(std::istream& stream, const std::string& name) {
    std::string token;
    if (!(stream >> token)) {
        throw std::runtime_error("failed to read " + name);
    }
    return token;
}

template <typename T>
T read_value(std::istream& stream, const std::string& name) {
    T value{};
    if (!(stream >> value)) {
        throw std::runtime_error("failed to read " + name);
    }
    return value;
}

template <>
Scalar read_value<Scalar>(std::istream& stream, const std::string& name) {
    Scalar value{};
    if (!(stream >> value)) {
        throw std::runtime_error("failed to read " + name);
    }
    require_finite(value, name);
    return value;
}

void require_text_stream_consumed(std::istream& stream, const std::string& format_name) {
    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error(format_name + " file has trailing data");
    }
    if (stream.bad()) {
        throw std::runtime_error("failed while checking " + format_name + " file ending");
    }
}

[[nodiscard]] bool is_supported_activation(Activation activation) noexcept {
    switch (activation) {
    case Activation::linear:
    case Activation::relu:
    case Activation::sigmoid:
    case Activation::tanh:
    case Activation::softmax:
        return true;
    }
    return false;
}

[[nodiscard]] std::size_t checked_product(std::size_t lhs, std::size_t rhs, const std::string& name) {
    require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
            name + " dimensions overflow");
    return lhs * rhs;
}

void validate_serialized_layer_count(std::size_t layer_count, const std::string& name) {
    if (layer_count == 0) {
        throw std::runtime_error(name + " layer count must be positive");
    }
    if (layer_count > kMaxSerializedLayers) {
        throw std::runtime_error(name + " layer count is unreasonably large");
    }
}

void validate_serialized_scalar_count(std::size_t count, const std::string& name) {
    if (count > kMaxSerializedScalars) {
        throw std::runtime_error(name + " count is unreasonably large");
    }
}

void require_exact_serialized_count(std::size_t actual, std::size_t expected, const std::string& name) {
    if (actual != expected) {
        throw std::runtime_error(name + " count mismatch");
    }
}

void apply_activation_in_place(Scalar* values, std::size_t size, Activation activation) noexcept {
    switch (activation) {
    case Activation::linear:
        return;
    case Activation::relu:
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = values[i] > Scalar{0} ? values[i] : Scalar{0};
        }
        return;
    case Activation::sigmoid:
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = sigmoid(values[i]);
        }
        return;
    case Activation::tanh:
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = static_cast<Scalar>(std::tanh(values[i]));
        }
        return;
    case Activation::softmax: {
        Scalar max_value = values[0];
        for (std::size_t i = 1; i < size; ++i) {
            if (values[i] > max_value) {
                max_value = values[i];
            }
        }

        Scalar sum = Scalar{0};
        for (std::size_t i = 0; i < size; ++i) {
            values[i] = static_cast<Scalar>(std::exp(values[i] - max_value));
            sum += values[i];
        }
        for (std::size_t i = 0; i < size; ++i) {
            values[i] /= sum;
        }
        return;
    }
    }
}

void validate_layer_parameters(const std::vector<LayerParameters>& parameters) {
    require(!parameters.empty(), "parameters must contain at least one layer");

    std::size_t expected_input_size = parameters.front().input_size;
    require(expected_input_size > 0, "input_size must be positive");

    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const LayerParameters& layer = parameters[i];
        require(layer.input_size == expected_input_size, "parameter input_size mismatch");
        require(layer.output_size > 0, "parameter output_size must be positive");
        require(is_supported_activation(layer.activation), "parameter activation is invalid");
        const std::size_t expected_weight_count =
            checked_product(layer.input_size, layer.output_size, "parameter weight");
        require(layer.weights.size() == expected_weight_count, "parameter weights size mismatch");
        require(layer.biases.size() == layer.output_size, "parameter biases size mismatch");
        validate_dropout_probability(layer.dropout_probability, "layer dropout_probability");
        if (i + 1 == parameters.size()) {
            require(layer.dropout_probability == Scalar{0}, "output layer dropout must be zero");
        }
        if (layer.activation == Activation::softmax) {
            require(i + 1 == parameters.size(), "softmax is only supported on the output layer");
            require(layer.output_size >= 2, "softmax output layer must contain at least two neurons");
        }
        for (Scalar weight : layer.weights) {
            require_finite(weight, "weight");
        }
        for (Scalar bias : layer.biases) {
            require_finite(bias, "bias");
        }
        expected_input_size = layer.output_size;
    }
}

[[nodiscard]] bool feature_changes_value(const FeatureNormalization& feature) noexcept {
    return feature.mode != NormalizationMode::none || feature.clamp;
}

[[nodiscard]] bool has_feature_normalization(const std::vector<FeatureNormalization>& features) noexcept {
    return std::any_of(features.begin(), features.end(), feature_changes_value);
}

void validate_feature_normalization(const FeatureNormalization& feature) {
    require_finite(feature.mean, "normalization mean");
    require_finite(feature.stddev, "normalization stddev");
    require_finite(feature.minimum, "normalization minimum");
    require_finite(feature.maximum, "normalization maximum");
    require_finite(feature.normalized_min, "normalization normalized_min");
    require_finite(feature.normalized_max, "normalization normalized_max");
    require_finite(feature.clamp_min, "normalization clamp_min");
    require_finite(feature.clamp_max, "normalization clamp_max");
    require(feature.clamp_min <= feature.clamp_max, "normalization clamp range is invalid");

    switch (feature.mode) {
    case NormalizationMode::none:
        break;
    case NormalizationMode::standard_score:
        require(feature.stddev > Scalar{0}, "standard_score stddev must be positive");
        break;
    case NormalizationMode::min_max:
        require(feature.maximum > feature.minimum, "min_max range must be positive");
        require(feature.normalized_max != feature.normalized_min, "min_max normalized range must be non-zero");
        break;
    }
}

void validate_normalization_features(const std::vector<FeatureNormalization>& features,
                                     std::size_t expected_size,
                                     const std::string& name) {
    if (features.empty()) {
        return;
    }
    require(features.size() == expected_size, name + " normalization feature count mismatch");
    for (const FeatureNormalization& feature : features) {
        validate_feature_normalization(feature);
    }
}

void validate_normalization_spec(const NormalizationSpec& normalization,
                                 std::size_t input_size,
                                 std::size_t output_size) {
    validate_normalization_features(normalization.input_features, input_size, "input");
    validate_normalization_features(normalization.output_features, output_size, "output");
}

void validate_normalization_for_loss(const NormalizationSpec& normalization, const LossConfig& loss) {
    if (normalization.has_output_normalization()) {
        require(!is_classification_loss(loss.type), "output normalization is not supported with classification losses");
    }
}

[[nodiscard]] Scalar clamp_value(Scalar value, Scalar minimum, Scalar maximum) noexcept {
    return std::min(std::max(value, minimum), maximum);
}

[[nodiscard]] Scalar normalize_value(Scalar value, const FeatureNormalization& feature) noexcept {
    Scalar result = value;
    switch (feature.mode) {
    case NormalizationMode::none:
        break;
    case NormalizationMode::standard_score:
        result = (value - feature.mean) / feature.stddev;
        break;
    case NormalizationMode::min_max:
        result = feature.normalized_min +
                 (value - feature.minimum) * (feature.normalized_max - feature.normalized_min) /
                     (feature.maximum - feature.minimum);
        break;
    }
    if (feature.clamp) {
        result = clamp_value(result, feature.clamp_min, feature.clamp_max);
    }
    return result;
}

[[nodiscard]] Scalar denormalize_value(Scalar value, const FeatureNormalization& feature) noexcept {
    Scalar adjusted = feature.clamp ? clamp_value(value, feature.clamp_min, feature.clamp_max) : value;
    switch (feature.mode) {
    case NormalizationMode::none:
        return adjusted;
    case NormalizationMode::standard_score:
        return adjusted * feature.stddev + feature.mean;
    case NormalizationMode::min_max:
        return feature.minimum +
               (adjusted - feature.normalized_min) * (feature.maximum - feature.minimum) /
                   (feature.normalized_max - feature.normalized_min);
    }
    return adjusted;
}

[[nodiscard]] Vector normalize_vector(const Vector& values, const std::vector<FeatureNormalization>& features) {
    if (!has_feature_normalization(features)) {
        return values;
    }
    Vector result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = normalize_value(values[i], features[i]);
    }
    return result;
}

void normalize_buffer_into(const Scalar* values,
                           std::size_t size,
                           const std::vector<FeatureNormalization>& features,
                           Scalar* result) {
    require(values != nullptr || size == 0, "normalization input buffer must not be null");
    require(result != nullptr || size == 0, "normalization output buffer must not be null");
    require(features.empty() || features.size() == size, "normalization feature count mismatch");
    if (size == 0) {
        return;
    }
    if (!has_feature_normalization(features)) {
        std::copy(values, values + size, result);
        return;
    }
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = normalize_value(values[i], features[i]);
    }
}

void normalize_vector_into(const Vector& values,
                           const std::vector<FeatureNormalization>& features,
                           Vector& result) {
    result.resize(values.size());
    normalize_buffer_into(values.data(), values.size(), features, result.data());
}

[[nodiscard]] Vector denormalize_vector(const Vector& values, const std::vector<FeatureNormalization>& features) {
    if (!has_feature_normalization(features)) {
        return values;
    }
    Vector result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = denormalize_value(values[i], features[i]);
    }
    return result;
}

[[nodiscard]] Matrix normalize_matrix(const Matrix& values, const std::vector<FeatureNormalization>& features) {
    if (!has_feature_normalization(features)) {
        return values;
    }
    Matrix result;
    result.reserve(values.size());
    for (const Vector& row : values) {
        result.push_back(normalize_vector(row, features));
    }
    return result;
}

void write_normalization_features(std::ostream& output, const std::vector<FeatureNormalization>& features) {
    output << features.size() << '\n';
    for (const FeatureNormalization& feature : features) {
        output << to_string(feature.mode) << ' '
               << feature.mean << ' '
               << feature.stddev << ' '
               << feature.minimum << ' '
               << feature.maximum << ' '
               << feature.normalized_min << ' '
               << feature.normalized_max << ' '
               << (feature.clamp ? 1 : 0) << ' '
               << feature.clamp_min << ' '
               << feature.clamp_max << '\n';
    }
}

void write_normalization(std::ostream& output, const NormalizationSpec& normalization) {
    output << "NORMALIZATION_V1\n";
    write_normalization_features(output, normalization.input_features);
    write_normalization_features(output, normalization.output_features);
}

[[nodiscard]] std::vector<FeatureNormalization> read_normalization_features(std::istream& input,
                                                                            const std::string& name) {
    const std::size_t count = read_value<std::size_t>(input, name + "_feature_count");
    validate_serialized_scalar_count(count, name + " feature");
    std::vector<FeatureNormalization> features;
    features.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        FeatureNormalization feature;
        feature.mode = normalization_mode_from_string(read_token(input, name + "_mode"));
        feature.mean = read_value<Scalar>(input, name + "_mean");
        feature.stddev = read_value<Scalar>(input, name + "_stddev");
        feature.minimum = read_value<Scalar>(input, name + "_minimum");
        feature.maximum = read_value<Scalar>(input, name + "_maximum");
        feature.normalized_min = read_value<Scalar>(input, name + "_normalized_min");
        feature.normalized_max = read_value<Scalar>(input, name + "_normalized_max");
        feature.clamp = read_value<int>(input, name + "_clamp") != 0;
        feature.clamp_min = read_value<Scalar>(input, name + "_clamp_min");
        feature.clamp_max = read_value<Scalar>(input, name + "_clamp_max");
        validate_feature_normalization(feature);
        features.push_back(feature);
    }
    return features;
}

[[nodiscard]] NormalizationSpec read_normalization(std::istream& input) {
    const std::string magic = read_token(input, "normalization_magic");
    if (magic != "NORMALIZATION_V1") {
        throw std::runtime_error("unsupported normalization format");
    }

    NormalizationSpec normalization;
    normalization.input_features = read_normalization_features(input, "input_normalization");
    normalization.output_features = read_normalization_features(input, "output_normalization");
    return normalization;
}

void write_loss_config(std::ostream& output, const LossConfig& loss) {
    if (loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be serialized");
    }

    output << "LOSS_CONFIG_V1\n";
    output << to_string(loss.type) << '\n';
    output << loss.huber_delta << ' '
           << loss.relative_epsilon << ' '
           << loss.weights.size();
    for (Scalar weight : loss.weights) {
        output << ' ' << weight;
    }
    output << '\n';
}

[[nodiscard]] LossConfig read_loss_config(std::istream& input) {
    const std::string magic = read_token(input, "loss_config_magic");
    if (magic != "LOSS_CONFIG_V1") {
        throw std::runtime_error("unsupported loss config format");
    }

    LossConfig loss;
    loss.type = loss_from_string(read_token(input, "loss"));
    loss.huber_delta = read_value<Scalar>(input, "huber_delta");
    loss.relative_epsilon = read_value<Scalar>(input, "relative_epsilon");
    const std::size_t weight_count = read_value<std::size_t>(input, "loss_weight_count");
    validate_serialized_scalar_count(weight_count, "loss weight");
    loss.weights.resize(weight_count);
    for (Scalar& weight : loss.weights) {
        weight = read_value<Scalar>(input, "loss_weight");
    }
    return loss;
}

constexpr unsigned char kBinaryMagic[8] = {'B', 'M', 'L', 'P', 'B', 'I', 'N', '\0'};
constexpr std::uint32_t kBinaryVersion = 7;

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

[[nodiscard]] std::uint64_t current_unix_time() noexcept {
    const std::time_t now = std::time(nullptr);
    return now < static_cast<std::time_t>(0) ? 0 : static_cast<std::uint64_t>(now);
}

[[nodiscard]] std::size_t checked_size(std::uint64_t value, const std::string& name) {
    require(value <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
            name + " is too large for this platform");
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = static_cast<std::uint32_t>(-(static_cast<int>(crc & 1u)));
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

struct BinaryWriter {
    std::vector<unsigned char> bytes;

    void write_u8(std::uint8_t value) {
        bytes.push_back(value);
    }

    void write_bool(bool value) {
        write_u8(value ? 1u : 0u);
    }

    void write_u32(std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xFFu));
        }
    }

    void write_u64(std::uint64_t value) {
        for (int shift = 0; shift < 64; shift += 8) {
            bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xFFu));
        }
    }

    void write_size(std::size_t value) {
        write_u64(static_cast<std::uint64_t>(value));
    }

    void write_scalar(Scalar value) {
        require_finite(value, "binary scalar");
#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
        static_assert(sizeof(Scalar) == sizeof(std::uint64_t), "double binary scalar size mismatch");
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        write_u64(bits);
#else
        static_assert(sizeof(Scalar) == sizeof(std::uint32_t), "float binary scalar size mismatch");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        write_u32(bits);
#endif
    }

    void write_string(const std::string& value) {
        write_size(value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }

    void write_scalars(const Vector& values) {
        write_size(values.size());
        for (Scalar value : values) {
            write_scalar(value);
        }
    }
};

struct BinaryReader {
    const std::vector<unsigned char>* bytes{nullptr};
    std::size_t cursor{0};

    explicit BinaryReader(const std::vector<unsigned char>& data)
        : bytes(&data) {}

    void require_available(std::size_t count, const std::string& name) const {
        require(bytes != nullptr, "binary reader is not initialized");
        if (count > bytes->size() || cursor > bytes->size() - count) {
            throw std::runtime_error("truncated binary model while reading " + name);
        }
    }

    [[nodiscard]] std::uint8_t read_u8(const std::string& name) {
        require_available(1, name);
        return (*bytes)[cursor++];
    }

    [[nodiscard]] bool read_bool(const std::string& name) {
        const std::uint8_t value = read_u8(name);
        if (value > 1u) {
            throw std::runtime_error("invalid binary boolean: " + name);
        }
        return value != 0u;
    }

    [[nodiscard]] std::uint32_t read_u32(const std::string& name) {
        require_available(4, name);
        std::uint32_t value = 0;
        for (int shift = 0; shift < 32; shift += 8) {
            value |= static_cast<std::uint32_t>((*bytes)[cursor++]) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t read_u64(const std::string& name) {
        require_available(8, name);
        std::uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>((*bytes)[cursor++]) << shift;
        }
        return value;
    }

    [[nodiscard]] std::size_t read_size(const std::string& name) {
        return checked_size(read_u64(name), name);
    }

    [[nodiscard]] Scalar read_scalar(const std::string& name) {
        Scalar value{};
#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
        static_assert(sizeof(Scalar) == sizeof(std::uint64_t), "double binary scalar size mismatch");
        const std::uint64_t bits = read_u64(name);
        std::memcpy(&value, &bits, sizeof(value));
#else
        static_assert(sizeof(Scalar) == sizeof(std::uint32_t), "float binary scalar size mismatch");
        const std::uint32_t bits = read_u32(name);
        std::memcpy(&value, &bits, sizeof(value));
#endif
        require_finite(value, name);
        return value;
    }

    [[nodiscard]] std::string read_string(const std::string& name) {
        const std::size_t size = read_size(name + "_size");
        require_available(size, name);
        std::string value;
        value.assign(reinterpret_cast<const char*>(bytes->data() + cursor), size);
        cursor += size;
        return value;
    }

    [[nodiscard]] Vector read_scalars(const std::string& name) {
        const std::size_t count = read_size(name + "_count");
        validate_serialized_scalar_count(count, name);
        Vector values(count);
        for (Scalar& value : values) {
            value = read_scalar(name);
        }
        return values;
    }

    [[nodiscard]] Vector read_scalars_exact(const std::string& name, std::size_t expected_count) {
        const std::size_t count = read_size(name + "_count");
        require_exact_serialized_count(count, expected_count, name);
        Vector values(count);
        for (Scalar& value : values) {
            value = read_scalar(name);
        }
        return values;
    }

    [[nodiscard]] bool at_end() const noexcept {
        return bytes != nullptr && cursor == bytes->size();
    }
};

void write_binary_metadata(BinaryWriter& writer, BinaryMetadata metadata) {
    if (metadata.created_unix_time == 0) {
        metadata.created_unix_time = current_unix_time();
    }
    writer.write_u64(metadata.created_unix_time);
    writer.write_string(metadata.description);
    writer.write_size(metadata.entries.size());
    for (const BinaryMetadataEntry& entry : metadata.entries) {
        writer.write_string(entry.key);
        writer.write_string(entry.value);
    }
}

[[nodiscard]] BinaryMetadata read_binary_metadata(BinaryReader& reader) {
    BinaryMetadata metadata;
    metadata.created_unix_time = reader.read_u64("created_unix_time");
    metadata.description = reader.read_string("metadata_description");
    const std::size_t entry_count = reader.read_size("metadata_entry_count");
    metadata.entries.reserve(entry_count);
    for (std::size_t i = 0; i < entry_count; ++i) {
        BinaryMetadataEntry entry;
        entry.key = reader.read_string("metadata_key");
        entry.value = reader.read_string("metadata_value");
        metadata.entries.push_back(std::move(entry));
    }
    return metadata;
}

void write_binary_training_options(BinaryWriter& writer, const TrainingOptions& options) {
    writer.write_size(options.epochs);
    writer.write_size(options.batch_size);
    writer.write_bool(options.shuffle);
    writer.write_u64(options.seed);
    writer.write_scalar(options.validation_split);
    writer.write_scalar(options.test_split);
    writer.write_size(options.streaming_shuffle_buffer_size);
    writer.write_size(options.early_stopping_patience);
    writer.write_scalar(options.min_delta);
    writer.write_u32(training_monitor_code(options.early_stopping_monitor));
    writer.write_u32(training_monitor_mode_code(options.early_stopping_mode));
    writer.write_size(options.early_stopping_cooldown);
    writer.write_u32(learning_rate_schedule_code(options.learning_rate_schedule.type));
    writer.write_scalar(options.learning_rate_schedule.base_learning_rate);
    writer.write_scalar(options.learning_rate_schedule.minimum_learning_rate);
    writer.write_size(options.learning_rate_schedule.warmup_epochs);
    writer.write_scalar(options.learning_rate_schedule.warmup_start_learning_rate);
    writer.write_size(options.learning_rate_schedule.step_size);
    writer.write_scalar(options.learning_rate_schedule.step_decay_factor);
    writer.write_scalar(options.learning_rate_schedule.exponential_decay_rate);
    writer.write_size(options.learning_rate_schedule.cosine_epochs);
    writer.write_u32(training_monitor_code(options.learning_rate_schedule.reduce_on_plateau_monitor));
    writer.write_u32(training_monitor_mode_code(options.learning_rate_schedule.reduce_on_plateau_mode));
    writer.write_size(options.learning_rate_schedule.reduce_on_plateau_patience);
    writer.write_scalar(options.learning_rate_schedule.reduce_on_plateau_factor);
    writer.write_scalar(options.learning_rate_schedule.reduce_on_plateau_min_delta);
    writer.write_size(options.learning_rate_schedule.reduce_on_plateau_cooldown);
    writer.write_scalar(options.gradient_noise_stddev);
    writer.write_u32(parallel_execution_code(options.parallelism.execution));
    writer.write_size(options.parallelism.thread_count);
    writer.write_size(options.parallelism.minimum_parallel_samples);
    writer.write_bool(options.debug.enabled);
    writer.write_scalar(options.debug.loss_explosion_factor);
    writer.write_scalar(options.debug.loss_explosion_threshold);
    writer.write_scalar(options.debug.gradient_norm_explosion_factor);
    writer.write_scalar(options.debug.gradient_norm_explosion_threshold);
    writer.write_bool(options.restore_best_weights);
    writer.write_string(options.best_checkpoint_path.string());
    writer.write_string(options.latest_checkpoint_path.string());
}

[[nodiscard]] TrainingOptions read_binary_training_options(BinaryReader& reader) {
    TrainingOptions options;
    options.epochs = reader.read_size("training_epochs");
    options.batch_size = reader.read_size("training_batch_size");
    options.shuffle = reader.read_bool("training_shuffle");
    options.seed = reader.read_u64("training_seed");
    options.validation_split = reader.read_scalar("training_validation_split");
    options.test_split = reader.read_scalar("training_test_split");
    options.streaming_shuffle_buffer_size = reader.read_size("training_streaming_shuffle_buffer_size");
    options.early_stopping_patience = reader.read_size("training_early_stopping_patience");
    options.min_delta = reader.read_scalar("training_min_delta");
    options.early_stopping_monitor = training_monitor_from_code(reader.read_u32("training_monitor"));
    options.early_stopping_mode = training_monitor_mode_from_code(reader.read_u32("training_monitor_mode"));
    options.early_stopping_cooldown = reader.read_size("training_early_stopping_cooldown");
    options.learning_rate_schedule.type =
        learning_rate_schedule_from_code(reader.read_u32("learning_rate_schedule"));
    options.learning_rate_schedule.base_learning_rate = reader.read_scalar("base_learning_rate");
    options.learning_rate_schedule.minimum_learning_rate = reader.read_scalar("minimum_learning_rate");
    options.learning_rate_schedule.warmup_epochs = reader.read_size("warmup_epochs");
    options.learning_rate_schedule.warmup_start_learning_rate =
        reader.read_scalar("warmup_start_learning_rate");
    options.learning_rate_schedule.step_size = reader.read_size("step_size");
    options.learning_rate_schedule.step_decay_factor = reader.read_scalar("step_decay_factor");
    options.learning_rate_schedule.exponential_decay_rate = reader.read_scalar("exponential_decay_rate");
    options.learning_rate_schedule.cosine_epochs = reader.read_size("cosine_epochs");
    options.learning_rate_schedule.reduce_on_plateau_monitor =
        training_monitor_from_code(reader.read_u32("reduce_on_plateau_monitor"));
    options.learning_rate_schedule.reduce_on_plateau_mode =
        training_monitor_mode_from_code(reader.read_u32("reduce_on_plateau_mode"));
    options.learning_rate_schedule.reduce_on_plateau_patience =
        reader.read_size("reduce_on_plateau_patience");
    options.learning_rate_schedule.reduce_on_plateau_factor =
        reader.read_scalar("reduce_on_plateau_factor");
    options.learning_rate_schedule.reduce_on_plateau_min_delta =
        reader.read_scalar("reduce_on_plateau_min_delta");
    options.learning_rate_schedule.reduce_on_plateau_cooldown =
        reader.read_size("reduce_on_plateau_cooldown");
    options.gradient_noise_stddev = reader.read_scalar("gradient_noise_stddev");
    options.parallelism.execution =
        parallel_execution_from_code(reader.read_u32("parallel_execution"));
    options.parallelism.thread_count = reader.read_size("parallel_thread_count");
    options.parallelism.minimum_parallel_samples = reader.read_size("parallel_minimum_samples");
    options.debug.enabled = reader.read_bool("debug_enabled");
    options.debug.loss_explosion_factor = reader.read_scalar("debug_loss_explosion_factor");
    options.debug.loss_explosion_threshold = reader.read_scalar("debug_loss_explosion_threshold");
    options.debug.gradient_norm_explosion_factor = reader.read_scalar("debug_gradient_norm_explosion_factor");
    options.debug.gradient_norm_explosion_threshold = reader.read_scalar("debug_gradient_norm_explosion_threshold");
    options.restore_best_weights = reader.read_bool("training_restore_best_weights");
    options.best_checkpoint_path = reader.read_string("training_best_checkpoint_path");
    options.latest_checkpoint_path = reader.read_string("training_latest_checkpoint_path");
    validate_training_options(options);
    return options;
}

void write_binary_learning_rate_scheduler_state(BinaryWriter& writer,
                                                const LearningRateSchedulerState& state) {
    writer.write_bool(state.initialized);
    writer.write_scalar(state.base_learning_rate);
    writer.write_scalar(state.current_learning_rate);
    writer.write_bool(state.has_best_metric);
    if (state.has_best_metric) {
        writer.write_scalar(state.best_metric);
    }
    writer.write_size(state.stale_epochs);
    writer.write_size(state.cooldown_remaining);
}

[[nodiscard]] LearningRateSchedulerState read_binary_learning_rate_scheduler_state(BinaryReader& reader) {
    LearningRateSchedulerState state;
    state.initialized = reader.read_bool("learning_rate_state_initialized");
    state.base_learning_rate = reader.read_scalar("learning_rate_state_base");
    state.current_learning_rate = reader.read_scalar("learning_rate_state_current");
    state.has_best_metric = reader.read_bool("learning_rate_state_has_best_metric");
    if (state.has_best_metric) {
        state.best_metric = reader.read_scalar("learning_rate_state_best_metric");
    }
    state.stale_epochs = reader.read_size("learning_rate_state_stale_epochs");
    state.cooldown_remaining = reader.read_size("learning_rate_state_cooldown_remaining");
    require(state.base_learning_rate >= Scalar{0}, "learning_rate_state_base must be non-negative");
    require(state.current_learning_rate >= Scalar{0}, "learning_rate_state_current must be non-negative");
    return state;
}

void write_binary_optimizer(BinaryWriter& writer, const OptimizerConfig& optimizer) {
    writer.write_u32(optimizer_code(optimizer.type));
    writer.write_scalar(optimizer.learning_rate);
    writer.write_scalar(optimizer.beta1);
    writer.write_scalar(optimizer.beta2);
    writer.write_scalar(optimizer.epsilon);
    writer.write_scalar(optimizer.momentum);
    writer.write_scalar(optimizer.l2);
    writer.write_scalar(optimizer.l1);
    writer.write_scalar(optimizer.decoupled_weight_decay);
    writer.write_scalar(optimizer.max_norm);
    writer.write_scalar(optimizer.gradient_clip_norm);
    writer.write_scalar(optimizer.gradient_clip_value);
    writer.write_scalar(optimizer.layer_gradient_clip_norm);
}

[[nodiscard]] OptimizerConfig read_binary_optimizer(BinaryReader& reader) {
    OptimizerConfig optimizer;
    optimizer.type = optimizer_from_code(reader.read_u32("optimizer"));
    optimizer.learning_rate = reader.read_scalar("learning_rate");
    optimizer.beta1 = reader.read_scalar("beta1");
    optimizer.beta2 = reader.read_scalar("beta2");
    optimizer.epsilon = reader.read_scalar("epsilon");
    optimizer.momentum = reader.read_scalar("momentum");
    optimizer.l2 = reader.read_scalar("l2");
    optimizer.l1 = reader.read_scalar("l1");
    optimizer.decoupled_weight_decay = reader.read_scalar("decoupled_weight_decay");
    optimizer.max_norm = reader.read_scalar("max_norm");
    optimizer.gradient_clip_norm = reader.read_scalar("gradient_clip_norm");
    optimizer.gradient_clip_value = reader.read_scalar("gradient_clip_value");
    optimizer.layer_gradient_clip_norm = reader.read_scalar("layer_gradient_clip_norm");
    validate_optimizer(optimizer);
    return optimizer;
}

void write_binary_loss_config(BinaryWriter& writer, const LossConfig& loss) {
    if (loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be serialized");
    }
    writer.write_u32(loss_code(loss.type));
    writer.write_scalar(loss.huber_delta);
    writer.write_scalar(loss.relative_epsilon);
    writer.write_scalars(loss.weights);
}

[[nodiscard]] LossConfig read_binary_loss_config(BinaryReader& reader, std::size_t output_size) {
    LossConfig loss;
    loss.type = loss_from_code(reader.read_u32("loss"));
    if (loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be loaded from binary");
    }
    loss.huber_delta = reader.read_scalar("huber_delta");
    loss.relative_epsilon = reader.read_scalar("relative_epsilon");
    const std::size_t expected_weight_count =
        loss.type == Loss::weighted_mean_squared_error ? output_size : std::size_t{0};
    loss.weights = reader.read_scalars_exact("loss_weights", expected_weight_count);
    validate_loss_config(loss, output_size);
    return loss;
}

void write_binary_normalization_feature(BinaryWriter& writer, const FeatureNormalization& feature) {
    writer.write_u32(normalization_mode_code(feature.mode));
    writer.write_scalar(feature.mean);
    writer.write_scalar(feature.stddev);
    writer.write_scalar(feature.minimum);
    writer.write_scalar(feature.maximum);
    writer.write_scalar(feature.normalized_min);
    writer.write_scalar(feature.normalized_max);
    writer.write_bool(feature.clamp);
    writer.write_scalar(feature.clamp_min);
    writer.write_scalar(feature.clamp_max);
}

[[nodiscard]] FeatureNormalization read_binary_normalization_feature(BinaryReader& reader) {
    FeatureNormalization feature;
    feature.mode = normalization_mode_from_code(reader.read_u32("normalization_mode"));
    feature.mean = reader.read_scalar("normalization_mean");
    feature.stddev = reader.read_scalar("normalization_stddev");
    feature.minimum = reader.read_scalar("normalization_minimum");
    feature.maximum = reader.read_scalar("normalization_maximum");
    feature.normalized_min = reader.read_scalar("normalization_normalized_min");
    feature.normalized_max = reader.read_scalar("normalization_normalized_max");
    feature.clamp = reader.read_bool("normalization_clamp");
    feature.clamp_min = reader.read_scalar("normalization_clamp_min");
    feature.clamp_max = reader.read_scalar("normalization_clamp_max");
    validate_feature_normalization(feature);
    return feature;
}

void write_binary_normalization_features(BinaryWriter& writer, const std::vector<FeatureNormalization>& features) {
    writer.write_size(features.size());
    for (const FeatureNormalization& feature : features) {
        write_binary_normalization_feature(writer, feature);
    }
}

[[nodiscard]] std::vector<FeatureNormalization> read_binary_normalization_features(BinaryReader& reader,
                                                                                   const std::string& name,
                                                                                   std::size_t expected_size) {
    const std::size_t count = reader.read_size(name + "_count");
    if (count != 0 && count != expected_size) {
        throw std::runtime_error(name + " feature count mismatch");
    }
    std::vector<FeatureNormalization> features;
    features.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        features.push_back(read_binary_normalization_feature(reader));
    }
    return features;
}

void write_binary_normalization(BinaryWriter& writer, const NormalizationSpec& normalization) {
    write_binary_normalization_features(writer, normalization.input_features);
    write_binary_normalization_features(writer, normalization.output_features);
}

[[nodiscard]] NormalizationSpec read_binary_normalization(BinaryReader& reader,
                                                          std::size_t input_size,
                                                          std::size_t output_size) {
    NormalizationSpec normalization;
    normalization.input_features = read_binary_normalization_features(reader, "input_normalization", input_size);
    normalization.output_features = read_binary_normalization_features(reader, "output_normalization", output_size);
    validate_normalization_spec(normalization, input_size, output_size);
    return normalization;
}

void write_binary_layer(BinaryWriter& writer, const LayerParameters& layer) {
    writer.write_size(layer.input_size);
    writer.write_size(layer.output_size);
    writer.write_u32(activation_code(layer.activation));
    writer.write_scalar(layer.dropout_probability);
    writer.write_scalars(layer.weights);
    writer.write_scalars(layer.biases);
}

[[nodiscard]] LayerParameters read_binary_layer(BinaryReader& reader, std::size_t expected_input_size) {
    LayerParameters layer;
    layer.input_size = reader.read_size("layer_input_size");
    layer.output_size = reader.read_size("layer_output_size");
    layer.activation = activation_from_code(reader.read_u32("layer_activation"));
    layer.dropout_probability = reader.read_scalar("layer_dropout_probability");
    require(layer.input_size > 0, "binary layer input_size must be positive");
    require(layer.output_size > 0, "binary layer output_size must be positive");
    require(layer.input_size == expected_input_size, "binary topology is inconsistent");
    const std::size_t expected_weight_count =
        checked_product(layer.input_size, layer.output_size, "binary weight");
    layer.weights = reader.read_scalars_exact("weights", expected_weight_count);
    layer.biases = reader.read_scalars_exact("biases", layer.output_size);
    return layer;
}

struct LayerTrainingState {
    Vector grad_weights;
    Vector grad_biases;
    Vector velocity_weights;
    Vector velocity_biases;
    Vector first_moment_weights;
    Vector first_moment_biases;
    Vector second_moment_weights;
    Vector second_moment_biases;
};

void validate_layer_training_state(const LayerTrainingState& state, const LayerParameters& layer) {
    const std::size_t weight_count = checked_product(layer.input_size, layer.output_size, "checkpoint weight");
    const std::size_t bias_count = layer.output_size;
    require(state.grad_weights.size() == weight_count, "checkpoint grad weight count mismatch");
    require(state.velocity_weights.size() == weight_count, "checkpoint velocity weight count mismatch");
    require(state.first_moment_weights.size() == weight_count, "checkpoint first moment weight count mismatch");
    require(state.second_moment_weights.size() == weight_count, "checkpoint second moment weight count mismatch");
    require(state.grad_biases.size() == bias_count, "checkpoint grad bias count mismatch");
    require(state.velocity_biases.size() == bias_count, "checkpoint velocity bias count mismatch");
    require(state.first_moment_biases.size() == bias_count, "checkpoint first moment bias count mismatch");
    require(state.second_moment_biases.size() == bias_count, "checkpoint second moment bias count mismatch");
}

void write_binary_layer_training_state(BinaryWriter& writer, const LayerTrainingState& state) {
    writer.write_scalars(state.grad_weights);
    writer.write_scalars(state.grad_biases);
    writer.write_scalars(state.velocity_weights);
    writer.write_scalars(state.velocity_biases);
    writer.write_scalars(state.first_moment_weights);
    writer.write_scalars(state.first_moment_biases);
    writer.write_scalars(state.second_moment_weights);
    writer.write_scalars(state.second_moment_biases);
}

[[nodiscard]] LayerTrainingState read_binary_layer_training_state(BinaryReader& reader,
                                                                  const LayerParameters& layer) {
    const std::size_t weight_count = checked_product(layer.input_size, layer.output_size, "checkpoint weight");
    const std::size_t bias_count = layer.output_size;
    LayerTrainingState state;
    state.grad_weights = reader.read_scalars_exact("grad_weights", weight_count);
    state.grad_biases = reader.read_scalars_exact("grad_biases", bias_count);
    state.velocity_weights = reader.read_scalars_exact("velocity_weights", weight_count);
    state.velocity_biases = reader.read_scalars_exact("velocity_biases", bias_count);
    state.first_moment_weights = reader.read_scalars_exact("first_moment_weights", weight_count);
    state.first_moment_biases = reader.read_scalars_exact("first_moment_biases", bias_count);
    state.second_moment_weights = reader.read_scalars_exact("second_moment_weights", weight_count);
    state.second_moment_biases = reader.read_scalars_exact("second_moment_biases", bias_count);
    validate_layer_training_state(state, layer);
    return state;
}

[[nodiscard]] std::vector<unsigned char> read_binary_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary model file for reading: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to read binary model file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);

    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!input && !bytes.empty()) {
        throw std::runtime_error("failed while reading binary model file: " + path.string());
    }
    return bytes;
}

void write_binary_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open binary model file for writing: " + path.string());
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!output) {
        throw std::runtime_error("failed while writing binary model file: " + path.string());
    }
}

struct BinaryEnvelope {
    std::uint32_t version{0};
    BinaryScalarType scalar_type{BinaryScalarType::float32};
    BinaryModelKind kind{BinaryModelKind::training};
    std::vector<unsigned char> payload;
    std::uint32_t checksum{0};
};

[[nodiscard]] std::vector<unsigned char> build_binary_envelope(BinaryModelKind kind,
                                                               const std::vector<unsigned char>& payload) {
    BinaryWriter writer;
    writer.bytes.insert(writer.bytes.end(), kBinaryMagic, kBinaryMagic + 8);
    writer.write_u32(kBinaryVersion);
    writer.write_u32(scalar_type_code(current_binary_scalar_type()));
    writer.write_u32(model_kind_code(kind));
    writer.write_size(payload.size());
    writer.bytes.insert(writer.bytes.end(), payload.begin(), payload.end());

    const std::uint32_t checksum = crc32(writer.bytes.data(), writer.bytes.size());
    writer.write_u32(checksum);
    return writer.bytes;
}

[[nodiscard]] BinaryEnvelope parse_binary_envelope(const std::vector<unsigned char>& bytes) {
    constexpr std::size_t header_size = 8 + 4 + 4 + 4 + 8;
    constexpr std::size_t checksum_size = 4;
    if (bytes.size() < header_size + checksum_size) {
        throw std::runtime_error("binary model file is too small");
    }

    for (std::size_t i = 0; i < 8; ++i) {
        if (bytes[i] != kBinaryMagic[i]) {
            throw std::runtime_error("unsupported binary model magic");
        }
    }

    std::vector<unsigned char> header_and_payload(bytes.begin(), bytes.end() - checksum_size);
    BinaryReader file_reader(bytes);
    file_reader.cursor = 8;

    BinaryEnvelope envelope;
    envelope.version = file_reader.read_u32("binary_version");
    if (envelope.version != kBinaryVersion) {
        throw std::runtime_error("unsupported binary model version");
    }
    envelope.scalar_type = scalar_type_from_code(file_reader.read_u32("binary_scalar_type"));
    if (envelope.scalar_type != current_binary_scalar_type()) {
        throw std::runtime_error("binary model scalar type does not match this build");
    }
    envelope.kind = model_kind_from_code(file_reader.read_u32("binary_model_kind"));
    const std::size_t payload_size = file_reader.read_size("binary_payload_size");
    if (payload_size != bytes.size() - header_size - checksum_size) {
        throw std::runtime_error("binary model payload size mismatch");
    }

    envelope.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(header_size),
                            bytes.end() - static_cast<std::ptrdiff_t>(checksum_size));

    BinaryReader checksum_reader(bytes);
    checksum_reader.cursor = bytes.size() - checksum_size;
    envelope.checksum = checksum_reader.read_u32("binary_checksum");
    const std::uint32_t actual_checksum = crc32(header_and_payload.data(), header_and_payload.size());
    if (envelope.checksum != actual_checksum) {
        throw std::runtime_error("binary model checksum mismatch");
    }
    return envelope;
}

struct ParsedBinaryModel {
    BinaryModelInfo info;
    LossConfig loss{};
    OptimizerConfig optimizer{OptimizerConfig::adam()};
    NormalizationSpec normalization{};
    std::uint64_t optimizer_step{0};
    std::vector<LayerParameters> parameters;
    std::vector<LayerTrainingState> training_states;
};

void finalize_binary_info(ParsedBinaryModel& parsed) {
    parsed.info.layer_count = parsed.parameters.size();
    parsed.info.output_size = parsed.parameters.empty() ? 0 : parsed.parameters.back().output_size;
    parsed.info.optimizer_step = parsed.optimizer_step;
    parsed.info.weight_count = 0;
    parsed.info.bias_count = 0;
    for (const LayerParameters& layer : parsed.parameters) {
        parsed.info.weight_count += layer.weights.size();
        parsed.info.bias_count += layer.biases.size();
    }
}

[[nodiscard]] ParsedBinaryModel parse_binary_payload(const BinaryEnvelope& envelope) {
    BinaryReader reader(envelope.payload);
    ParsedBinaryModel parsed;
    parsed.info.version = envelope.version;
    parsed.info.scalar_type = envelope.scalar_type;
    parsed.info.model_kind = envelope.kind;
    parsed.info.checksum = envelope.checksum;
    parsed.info.metadata = read_binary_metadata(reader);
    parsed.info.input_size = reader.read_size("input_size");
    require(parsed.info.input_size > 0, "binary input_size must be positive");

    if (envelope.kind == BinaryModelKind::training) {
        const std::size_t declared_output_size = reader.read_size("declared_output_size");
        require(declared_output_size > 0, "binary declared output_size must be positive");
        parsed.loss = read_binary_loss_config(reader, declared_output_size);
        parsed.optimizer = read_binary_optimizer(reader);
        parsed.info.seed = reader.read_u64("seed");
        parsed.optimizer_step = reader.read_u64("optimizer_step");
        parsed.info.has_training_options = reader.read_bool("has_training_options");
        if (parsed.info.has_training_options) {
            parsed.info.training_options = read_binary_training_options(reader);
        }
        parsed.normalization = read_binary_normalization(reader, parsed.info.input_size, declared_output_size);

        const std::size_t layer_count = reader.read_size("layer_count");
        validate_serialized_layer_count(layer_count, "binary training model");
        parsed.parameters.reserve(layer_count);
        std::size_t expected_input_size = parsed.info.input_size;
        for (std::size_t i = 0; i < layer_count; ++i) {
            LayerParameters layer = read_binary_layer(reader, expected_input_size);
            expected_input_size = layer.output_size;
            parsed.parameters.push_back(std::move(layer));
        }
        require(!parsed.parameters.empty(), "binary training model has no layers");
        require(parsed.parameters.back().output_size == declared_output_size,
                "binary training output size does not match topology");
        validate_loss_config(parsed.loss, declared_output_size);
        validate_normalization_for_loss(parsed.normalization, parsed.loss);
    } else if (envelope.kind == BinaryModelKind::inference) {
        const std::size_t declared_output_size = reader.read_size("declared_output_size");
        require(declared_output_size > 0, "binary declared output_size must be positive");
        parsed.info.seed = reader.read_u64("seed");
        parsed.info.has_training_options = reader.read_bool("has_training_options");
        if (parsed.info.has_training_options) {
            parsed.info.training_options = read_binary_training_options(reader);
        }
        parsed.normalization = read_binary_normalization(reader, parsed.info.input_size, declared_output_size);

        const std::size_t layer_count = reader.read_size("layer_count");
        validate_serialized_layer_count(layer_count, "binary inference model");
        parsed.parameters.reserve(layer_count);
        std::size_t expected_input_size = parsed.info.input_size;
        for (std::size_t i = 0; i < layer_count; ++i) {
            LayerParameters layer = read_binary_layer(reader, expected_input_size);
            expected_input_size = layer.output_size;
            parsed.parameters.push_back(std::move(layer));
        }
        require(!parsed.parameters.empty(), "binary inference model has no layers");
        require(parsed.parameters.back().output_size == declared_output_size,
                "binary inference output size does not match topology");
    } else if (envelope.kind == BinaryModelKind::checkpoint) {
        const std::size_t declared_output_size = reader.read_size("declared_output_size");
        require(declared_output_size > 0, "binary declared output_size must be positive");
        parsed.loss = read_binary_loss_config(reader, declared_output_size);
        parsed.optimizer = read_binary_optimizer(reader);
        parsed.info.seed = reader.read_u64("seed");
        parsed.optimizer_step = reader.read_u64("optimizer_step");
        parsed.info.completed_epochs = reader.read_size("completed_epochs");
        parsed.info.checkpoint_best = reader.read_bool("checkpoint_best");
        parsed.info.has_checkpoint_metric = reader.read_bool("has_checkpoint_metric");
        if (parsed.info.has_checkpoint_metric) {
            parsed.info.checkpoint_metric = reader.read_scalar("checkpoint_metric");
        }
        parsed.info.has_learning_rate_state = reader.read_bool("has_learning_rate_state");
        require(parsed.info.has_learning_rate_state, "checkpoint requires learning rate scheduler state");
        parsed.info.learning_rate_state = read_binary_learning_rate_scheduler_state(reader);
        parsed.info.has_training_options = reader.read_bool("has_training_options");
        require(parsed.info.has_training_options, "checkpoint requires training options");
        parsed.info.training_options = read_binary_training_options(reader);
        parsed.normalization = read_binary_normalization(reader, parsed.info.input_size, declared_output_size);

        const std::size_t layer_count = reader.read_size("layer_count");
        validate_serialized_layer_count(layer_count, "checkpoint");
        parsed.parameters.reserve(layer_count);
        parsed.training_states.reserve(layer_count);
        std::size_t expected_input_size = parsed.info.input_size;
        for (std::size_t i = 0; i < layer_count; ++i) {
            LayerParameters layer = read_binary_layer(reader, expected_input_size);
            expected_input_size = layer.output_size;
            LayerTrainingState state = read_binary_layer_training_state(reader, layer);
            parsed.parameters.push_back(std::move(layer));
            parsed.training_states.push_back(std::move(state));
        }
        require(!parsed.parameters.empty(), "checkpoint has no layers");
        require(parsed.parameters.back().output_size == declared_output_size,
                "checkpoint output size does not match topology");
        validate_loss_config(parsed.loss, declared_output_size);
        validate_normalization_for_loss(parsed.normalization, parsed.loss);
    }

    if (!reader.at_end()) {
        throw std::runtime_error("binary model payload has trailing bytes");
    }
    validate_layer_parameters(parsed.parameters);
    finalize_binary_info(parsed);
    return parsed;
}

[[nodiscard]] ParsedBinaryModel parse_binary_model_file(const std::filesystem::path& path) {
    return parse_binary_payload(parse_binary_envelope(read_binary_file(path)));
}

void write_common_binary_model_payload(BinaryWriter& writer,
                                       const BinaryMetadata& metadata,
                                       std::size_t input_size,
                                       std::size_t output_size,
                                       std::uint64_t seed,
                                       bool has_training_options,
                                       const TrainingOptions& training_options,
                                       const NormalizationSpec& normalization,
                                       const std::vector<LayerParameters>& parameters) {
    write_binary_metadata(writer, metadata);
    writer.write_size(input_size);
    writer.write_size(output_size);
    writer.write_u64(seed);
    writer.write_bool(has_training_options);
    if (has_training_options) {
        write_binary_training_options(writer, training_options);
    }
    write_binary_normalization(writer, normalization);
    writer.write_size(parameters.size());
    for (const LayerParameters& layer : parameters) {
        write_binary_layer(writer, layer);
    }
}

} // namespace

BinaryModelInfo inspect_binary_model(const std::filesystem::path& path) {
    return parse_binary_model_file(path).info;
}

std::string_view version_string() noexcept {
    return BRUTAL_MLP_VERSION_STRING;
}

std::string TrainingRunConfig::to_json() const {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
    output << "{\n";

    write_json_key(output, 2, "library");
    output << "{\n";
    write_json_key(output, 4, "name");
    write_json_escaped(output, library_name);
    output << ",\n";
    write_json_key(output, 4, "version");
    write_json_escaped(output, library_version);
    output << ",\n";
    write_json_key(output, 4, "scalar_type");
    write_json_escaped(output, scalar_type);
    output << '\n';
    write_json_indent(output, 2);
    output << "},\n";

    write_json_key(output, 2, "date");
    output << "{\n";
    write_json_key(output, 4, "unix_time");
    output << created_unix_time;
    output << ",\n";
    write_json_key(output, 4, "utc");
    write_json_escaped(output, created_utc);
    output << '\n';
    write_json_indent(output, 2);
    output << "},\n";

    write_json_key(output, 2, "seed");
    output << "{\n";
    write_json_key(output, 4, "model");
    output << model_seed;
    output << ",\n";
    write_json_key(output, 4, "training");
    output << training_seed;
    output << '\n';
    write_json_indent(output, 2);
    output << "},\n";

    write_json_key(output, 2, "architecture");
    output << "{\n";
    write_json_key(output, 4, "input_size");
    output << input_size;
    output << ",\n";
    write_json_key(output, 4, "output_size");
    output << output_size;
    output << ",\n";
    write_json_key(output, 4, "layers");
    output << "[\n";
    for (std::size_t i = 0; i < architecture.size(); ++i) {
        const TrainingRunLayerConfig& layer = architecture[i];
        write_json_indent(output, 6);
        output << "{\n";
        write_json_key(output, 8, "input_size");
        output << layer.input_size;
        output << ",\n";
        write_json_key(output, 8, "output_size");
        output << layer.output_size;
        output << ",\n";
        write_json_key(output, 8, "activation");
        write_json_escaped(output, to_string(layer.activation));
        output << ",\n";
        write_json_key(output, 8, "dropout_probability");
        write_json_scalar(output, layer.dropout_probability);
        output << '\n';
        write_json_indent(output, 6);
        output << '}';
        if (i + 1 < architecture.size()) {
            output << ',';
        }
        output << '\n';
    }
    write_json_indent(output, 4);
    output << "]\n";
    write_json_indent(output, 2);
    output << "},\n";

    write_json_key(output, 2, "optimizer");
    write_json_optimizer(output, optimizer, 2);
    output << ",\n";
    write_json_key(output, 2, "learning_rate_schedule");
    write_json_learning_rate_schedule(output, learning_rate_schedule, 2);
    output << ",\n";
    write_json_key(output, 2, "loss");
    write_json_loss(output, loss, 2);
    output << ",\n";
    write_json_key(output, 2, "normalization");
    write_json_normalization(output, normalization, 2);
    output << ",\n";

    write_json_key(output, 2, "dataset");
    output << "{\n";
    write_json_key(output, 4, "sample_count");
    output << dataset.sample_count;
    output << ",\n";
    write_json_key(output, 4, "input_size");
    output << dataset.input_size;
    output << ",\n";
    write_json_key(output, 4, "output_size");
    output << dataset.output_size;
    output << ",\n";
    write_json_key(output, 4, "training_sample_count");
    output << dataset.training_sample_count;
    output << ",\n";
    write_json_key(output, 4, "validation_sample_count");
    output << dataset.validation_sample_count;
    output << ",\n";
    write_json_key(output, 4, "test_sample_count");
    output << dataset.test_sample_count;
    output << ",\n";
    write_json_key(output, 4, "validation_split");
    write_json_scalar(output, dataset.validation_split);
    output << ",\n";
    write_json_key(output, 4, "test_split");
    write_json_scalar(output, dataset.test_split);
    output << ",\n";
    write_json_key(output, 4, "shuffle");
    output << (dataset.shuffle ? "true" : "false");
    output << ",\n";
    write_json_key(output, 4, "split_seed");
    output << dataset.split_seed;
    output << ",\n";
    write_json_key(output, 4, "streaming");
    output << (dataset.streaming ? "true" : "false");
    output << ",\n";
    write_json_key(output, 4, "streaming_shuffle_buffer_size");
    output << dataset.streaming_shuffle_buffer_size;
    output << '\n';
    write_json_indent(output, 2);
    output << "},\n";

    write_json_key(output, 2, "training_options");
    write_json_training_options(output, options, 2);
    output << ",\n";

    write_json_key(output, 2, "result");
    output << "{\n";
    write_json_key(output, 4, "completed_epochs");
    output << result.completed_epochs;
    output << ",\n";
    write_json_key(output, 4, "has_best_score");
    output << (result.has_best_score ? "true" : "false");
    output << ",\n";
    write_json_key(output, 4, "best_epoch");
    if (result.has_best_score) {
        output << result.best_epoch;
    } else {
        output << "null";
    }
    output << ",\n";
    write_json_key(output, 4, "best_score");
    if (result.has_best_score) {
        write_json_scalar(output, result.best_score);
    } else {
        output << "null";
    }
    output << ",\n";
    write_json_key(output, 4, "monitor");
    write_json_escaped(output, to_string(result.monitor));
    output << ",\n";
    write_json_key(output, 4, "monitor_mode");
    write_json_escaped(output, to_string(result.monitor_mode));
    output << ",\n";
    write_json_key(output, 4, "stop_reason");
    write_json_escaped(output, to_string(result.stop_reason));
    output << ",\n";
    write_json_key(output, 4, "stop_message");
    write_json_escaped(output, result.stop_message);
    output << ",\n";
    write_json_key(output, 4, "final_training_loss");
    if (result.has_final_training_loss) {
        write_json_scalar(output, result.final_training_loss);
    } else {
        output << "null";
    }
    output << ",\n";
    write_json_key(output, 4, "final_validation_loss");
    if (result.has_final_validation_loss) {
        write_json_scalar(output, result.final_validation_loss);
    } else {
        output << "null";
    }
    output << ",\n";
    write_json_key(output, 4, "final_test_loss");
    if (result.has_final_test_loss) {
        write_json_scalar(output, result.final_test_loss);
    } else {
        output << "null";
    }
    output << '\n';
    write_json_indent(output, 2);
    output << "}\n";
    output << "}\n";
    return output.str();
}

std::string to_string(Activation activation) {
    switch (activation) {
    case Activation::linear:
        return "linear";
    case Activation::relu:
        return "relu";
    case Activation::sigmoid:
        return "sigmoid";
    case Activation::tanh:
        return "tanh";
    case Activation::softmax:
        return "softmax";
    }

    throw std::invalid_argument("unknown activation");
}

std::string to_string(Loss loss) {
    switch (loss) {
    case Loss::mean_squared_error:
        return "mean_squared_error";
    case Loss::mean_absolute_error:
        return "mean_absolute_error";
    case Loss::huber:
        return "huber";
    case Loss::relative_mean_squared_error:
        return "relative_mean_squared_error";
    case Loss::log_cosh:
        return "log_cosh";
    case Loss::weighted_mean_squared_error:
        return "weighted_mean_squared_error";
    case Loss::binary_cross_entropy:
        return "binary_cross_entropy";
    case Loss::categorical_cross_entropy:
        return "categorical_cross_entropy";
    case Loss::custom:
        return "custom";
    }

    throw std::invalid_argument("unknown loss");
}

std::string to_string(Metric metric) {
    switch (metric) {
    case Metric::mean_squared_error:
        return "mean_squared_error";
    case Metric::mean_absolute_error:
        return "mean_absolute_error";
    case Metric::root_mean_squared_error:
        return "root_mean_squared_error";
    case Metric::r2_score:
        return "r2_score";
    case Metric::accuracy:
        return "accuracy";
    case Metric::precision:
        return "precision";
    case Metric::recall:
        return "recall";
    case Metric::f1_score:
        return "f1_score";
    case Metric::confusion_matrix:
        return "confusion_matrix";
    case Metric::custom:
        return "custom";
    }

    throw std::invalid_argument("unknown metric");
}

std::string to_string(Averaging averaging) {
    switch (averaging) {
    case Averaging::binary:
        return "binary";
    case Averaging::macro:
        return "macro";
    case Averaging::micro:
        return "micro";
    }

    throw std::invalid_argument("unknown averaging");
}

std::string to_string(OptimizerType optimizer) {
    switch (optimizer) {
    case OptimizerType::sgd:
        return "sgd";
    case OptimizerType::adam:
        return "adam";
    case OptimizerType::adamw:
        return "adamw";
    }

    throw std::invalid_argument("unknown optimizer");
}

std::string to_string(NormalizationMode mode) {
    switch (mode) {
    case NormalizationMode::none:
        return "none";
    case NormalizationMode::standard_score:
        return "standard_score";
    case NormalizationMode::min_max:
        return "min_max";
    }

    throw std::invalid_argument("unknown normalization mode");
}

std::string to_string(InferenceStatus status) {
    switch (status) {
    case InferenceStatus::ok:
        return "ok";
    case InferenceStatus::null_input:
        return "null_input";
    case InferenceStatus::null_output:
        return "null_output";
    case InferenceStatus::invalid_input_size:
        return "invalid_input_size";
    case InferenceStatus::invalid_output_size:
        return "invalid_output_size";
    case InferenceStatus::invalid_input_stride:
        return "invalid_input_stride";
    case InferenceStatus::invalid_output_stride:
        return "invalid_output_stride";
    case InferenceStatus::null_scratch:
        return "null_scratch";
    case InferenceStatus::insufficient_scratch:
        return "insufficient_scratch";
    }

    throw std::invalid_argument("unknown inference status");
}

std::string to_string(TrainingStopReason reason) {
    switch (reason) {
    case TrainingStopReason::completed_epochs:
        return "completed_epochs";
    case TrainingStopReason::early_stopping:
        return "early_stopping";
    case TrainingStopReason::invalid_training_data:
        return "invalid_training_data";
    case TrainingStopReason::non_finite_loss:
        return "non_finite_loss";
    case TrainingStopReason::non_finite_gradient:
        return "non_finite_gradient";
    case TrainingStopReason::non_finite_weights:
        return "non_finite_weights";
    case TrainingStopReason::loss_explosion:
        return "loss_explosion";
    case TrainingStopReason::gradient_explosion:
        return "gradient_explosion";
    }

    throw std::invalid_argument("unknown training stop reason");
}

std::string to_string(TrainingMonitor monitor) {
    switch (monitor) {
    case TrainingMonitor::automatic:
        return "automatic";
    case TrainingMonitor::training_loss:
        return "training_loss";
    case TrainingMonitor::validation_loss:
        return "validation_loss";
    case TrainingMonitor::test_loss:
        return "test_loss";
    }

    throw std::invalid_argument("unknown training monitor");
}

std::string to_string(TrainingMonitorMode mode) {
    switch (mode) {
    case TrainingMonitorMode::minimize:
        return "minimize";
    case TrainingMonitorMode::maximize:
        return "maximize";
    }

    throw std::invalid_argument("unknown training monitor mode");
}

std::string to_string(LearningRateSchedule schedule) {
    switch (schedule) {
    case LearningRateSchedule::constant:
        return "constant";
    case LearningRateSchedule::step_decay:
        return "step_decay";
    case LearningRateSchedule::exponential_decay:
        return "exponential_decay";
    case LearningRateSchedule::cosine_annealing:
        return "cosine_annealing";
    case LearningRateSchedule::reduce_on_plateau:
        return "reduce_on_plateau";
    }

    throw std::invalid_argument("unknown learning rate schedule");
}

std::string to_string(WeightInitialization initialization) {
    switch (initialization) {
    case WeightInitialization::automatic:
        return "automatic";
    case WeightInitialization::he_normal:
        return "he_normal";
    case WeightInitialization::he_uniform:
        return "he_uniform";
    case WeightInitialization::xavier_normal:
        return "xavier_normal";
    case WeightInitialization::xavier_uniform:
        return "xavier_uniform";
    case WeightInitialization::lecun_normal:
        return "lecun_normal";
    case WeightInitialization::lecun_uniform:
        return "lecun_uniform";
    }

    throw std::invalid_argument("unknown weight initialization");
}

std::string to_string(ParallelExecution execution) {
    switch (execution) {
    case ParallelExecution::serial:
        return "serial";
    case ParallelExecution::worker_threads:
        return "worker_threads";
    }

    throw std::invalid_argument("unknown parallel execution");
}

Activation activation_from_string(std::string_view value) {
    if (value == "linear") {
        return Activation::linear;
    }
    if (value == "relu") {
        return Activation::relu;
    }
    if (value == "sigmoid") {
        return Activation::sigmoid;
    }
    if (value == "tanh") {
        return Activation::tanh;
    }
    if (value == "softmax") {
        return Activation::softmax;
    }
    throw std::invalid_argument("unknown activation: " + std::string(value));
}

Loss loss_from_string(std::string_view value) {
    if (value == "mean_squared_error") {
        return Loss::mean_squared_error;
    }
    if (value == "mean_absolute_error") {
        return Loss::mean_absolute_error;
    }
    if (value == "huber") {
        return Loss::huber;
    }
    if (value == "relative_mean_squared_error") {
        return Loss::relative_mean_squared_error;
    }
    if (value == "log_cosh") {
        return Loss::log_cosh;
    }
    if (value == "weighted_mean_squared_error") {
        return Loss::weighted_mean_squared_error;
    }
    if (value == "binary_cross_entropy") {
        return Loss::binary_cross_entropy;
    }
    if (value == "categorical_cross_entropy") {
        return Loss::categorical_cross_entropy;
    }
    if (value == "custom") {
        return Loss::custom;
    }
    throw std::invalid_argument("unknown loss: " + std::string(value));
}

Metric metric_from_string(std::string_view value) {
    if (value == "mean_squared_error") {
        return Metric::mean_squared_error;
    }
    if (value == "mean_absolute_error") {
        return Metric::mean_absolute_error;
    }
    if (value == "root_mean_squared_error") {
        return Metric::root_mean_squared_error;
    }
    if (value == "r2_score") {
        return Metric::r2_score;
    }
    if (value == "accuracy") {
        return Metric::accuracy;
    }
    if (value == "precision") {
        return Metric::precision;
    }
    if (value == "recall") {
        return Metric::recall;
    }
    if (value == "f1_score") {
        return Metric::f1_score;
    }
    if (value == "confusion_matrix") {
        return Metric::confusion_matrix;
    }
    if (value == "custom") {
        return Metric::custom;
    }
    throw std::invalid_argument("unknown metric: " + std::string(value));
}

Averaging averaging_from_string(std::string_view value) {
    if (value == "binary") {
        return Averaging::binary;
    }
    if (value == "macro") {
        return Averaging::macro;
    }
    if (value == "micro") {
        return Averaging::micro;
    }
    throw std::invalid_argument("unknown averaging: " + std::string(value));
}

OptimizerType optimizer_type_from_string(std::string_view value) {
    if (value == "sgd") {
        return OptimizerType::sgd;
    }
    if (value == "adam") {
        return OptimizerType::adam;
    }
    if (value == "adamw") {
        return OptimizerType::adamw;
    }
    throw std::invalid_argument("unknown optimizer: " + std::string(value));
}

NormalizationMode normalization_mode_from_string(std::string_view value) {
    if (value == "none") {
        return NormalizationMode::none;
    }
    if (value == "standard_score") {
        return NormalizationMode::standard_score;
    }
    if (value == "min_max") {
        return NormalizationMode::min_max;
    }
    throw std::invalid_argument("unknown normalization mode: " + std::string(value));
}

ParallelExecution parallel_execution_from_string(std::string_view value) {
    if (value == "serial") {
        return ParallelExecution::serial;
    }
    if (value == "worker_threads") {
        return ParallelExecution::worker_threads;
    }
    throw std::invalid_argument("unknown parallel execution: " + std::string(value));
}

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t column_count, Scalar value) {
    resize(row_count, column_count, value);
}

DenseMatrix DenseMatrix::from_matrix(const Matrix& matrix) {
    if (matrix.empty()) {
        return {};
    }
    const std::size_t column_count = matrix.front().size();
    validate_matrix(matrix, column_count, "matrix");
    DenseMatrix dense(matrix.size(), column_count);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        dense.set_row(row, matrix[row]);
    }
    return dense;
}

Matrix DenseMatrix::to_matrix() const {
    Matrix matrix(rows_, Vector(columns_, Scalar{0}));
    for (std::size_t row = 0; row < rows_; ++row) {
        const Scalar* source = row_data(row);
        std::copy(source, source + columns_, matrix[row].begin());
    }
    return matrix;
}

void DenseMatrix::resize(std::size_t row_count, std::size_t column_count, Scalar value) {
    require(column_count > 0 || row_count == 0, "dense matrix columns must be positive");
    if (column_count != 0) {
        require(row_count <= std::numeric_limits<std::size_t>::max() / column_count,
                "dense matrix dimensions overflow");
    }
    rows_ = row_count;
    columns_ = column_count;
    values_.assign(rows_ * columns_, value);
}

void DenseMatrix::clear() noexcept {
    rows_ = 0;
    columns_ = 0;
    values_.clear();
}

bool DenseMatrix::empty() const noexcept {
    return rows_ == 0;
}

std::size_t DenseMatrix::rows() const noexcept {
    return rows_;
}

std::size_t DenseMatrix::columns() const noexcept {
    return columns_;
}

std::size_t DenseMatrix::size() const noexcept {
    return rows_;
}

std::size_t DenseMatrix::scalar_count() const noexcept {
    return values_.size();
}

const Vector& DenseMatrix::values() const noexcept {
    return values_;
}

Scalar* DenseMatrix::data() noexcept {
    return values_.data();
}

const Scalar* DenseMatrix::data() const noexcept {
    return values_.data();
}

Scalar* DenseMatrix::row_data(std::size_t row) {
    require(row < rows_, "dense matrix row index is out of range");
    return values_.data() + row * columns_;
}

const Scalar* DenseMatrix::row_data(std::size_t row) const {
    require(row < rows_, "dense matrix row index is out of range");
    return values_.data() + row * columns_;
}

Scalar& DenseMatrix::operator()(std::size_t row, std::size_t column) {
    require(row < rows_, "dense matrix row index is out of range");
    require(column < columns_, "dense matrix column index is out of range");
    return values_[row * columns_ + column];
}

Scalar DenseMatrix::operator()(std::size_t row, std::size_t column) const {
    require(row < rows_, "dense matrix row index is out of range");
    require(column < columns_, "dense matrix column index is out of range");
    return values_[row * columns_ + column];
}

Vector DenseMatrix::row(std::size_t row_index) const {
    const Scalar* source = row_data(row_index);
    return Vector(source, source + columns_);
}

void DenseMatrix::set_row(std::size_t row_index, const Vector& values) {
    require(values.size() == columns_, "dense matrix row has an invalid column count");
    const Scalar* source = values.data();
    Scalar* destination = row_data(row_index);
    std::copy(source, source + columns_, destination);
}

LossConfig LossConfig::from_loss(Loss loss) {
    switch (loss) {
    case Loss::mean_squared_error:
        return mean_squared_error();
    case Loss::mean_absolute_error:
        return mean_absolute_error();
    case Loss::huber:
        return huber();
    case Loss::relative_mean_squared_error:
        return relative_mean_squared_error();
    case Loss::log_cosh:
        return log_cosh();
    case Loss::weighted_mean_squared_error:
        throw std::invalid_argument("weighted_mean_squared_error requires explicit weights");
    case Loss::binary_cross_entropy:
        return binary_cross_entropy();
    case Loss::categorical_cross_entropy:
        return categorical_cross_entropy();
    case Loss::custom:
        throw std::invalid_argument("custom loss requires callbacks");
    }

    throw std::invalid_argument("unknown loss");
}

LossConfig LossConfig::mean_squared_error() {
    LossConfig config;
    config.type = Loss::mean_squared_error;
    return config;
}

LossConfig LossConfig::mean_absolute_error() {
    LossConfig config;
    config.type = Loss::mean_absolute_error;
    return config;
}

LossConfig LossConfig::huber(Scalar delta) {
    LossConfig config;
    config.type = Loss::huber;
    config.huber_delta = delta;
    validate_loss_config(config, 0);
    return config;
}

LossConfig LossConfig::relative_mean_squared_error(Scalar epsilon) {
    LossConfig config;
    config.type = Loss::relative_mean_squared_error;
    config.relative_epsilon = epsilon;
    validate_loss_config(config, 0);
    return config;
}

LossConfig LossConfig::log_cosh() {
    LossConfig config;
    config.type = Loss::log_cosh;
    return config;
}

LossConfig LossConfig::weighted_mean_squared_error(const Vector& weights) {
    LossConfig config;
    config.type = Loss::weighted_mean_squared_error;
    config.weights = weights;
    validate_loss_config(config, weights.size());
    return config;
}

LossConfig LossConfig::binary_cross_entropy() {
    LossConfig config;
    config.type = Loss::binary_cross_entropy;
    return config;
}

LossConfig LossConfig::categorical_cross_entropy() {
    LossConfig config;
    config.type = Loss::categorical_cross_entropy;
    return config;
}

LossConfig LossConfig::custom(CustomLossValueFunction value,
                              CustomLossGradientFunction gradient,
                              void* context) {
    LossConfig config;
    config.type = Loss::custom;
    config.custom_loss.value = value;
    config.custom_loss.gradient = gradient;
    config.custom_loss.context = context;
    validate_loss_config(config, 0);
    return config;
}

std::size_t ConfusionMatrix::class_count() const noexcept {
    return counts.size();
}

std::size_t ConfusionMatrix::total() const noexcept {
    std::size_t result = 0;
    for (const auto& row : counts) {
        result += std::accumulate(row.begin(), row.end(), std::size_t{0});
    }
    return result;
}

EvaluationOptions EvaluationOptions::regression() {
    EvaluationOptions options;
    options.metrics = {
        Metric::mean_squared_error,
        Metric::mean_absolute_error,
        Metric::root_mean_squared_error,
        Metric::r2_score,
    };
    return options;
}

EvaluationOptions EvaluationOptions::binary_classification(Scalar threshold, std::size_t positive_class_value) {
    EvaluationOptions options;
    options.metrics = {
        Metric::accuracy,
        Metric::precision,
        Metric::recall,
        Metric::f1_score,
        Metric::confusion_matrix,
    };
    options.classification_threshold = threshold;
    options.positive_class = positive_class_value;
    options.averaging = Averaging::binary;
    validate_evaluation_options(options);
    return options;
}

EvaluationOptions EvaluationOptions::multiclass_classification(Averaging averaging_value) {
    EvaluationOptions options;
    options.metrics = {
        Metric::accuracy,
        Metric::precision,
        Metric::recall,
        Metric::f1_score,
        Metric::confusion_matrix,
    };
    options.averaging = averaging_value;
    validate_evaluation_options(options);
    return options;
}

EvaluationOptions EvaluationOptions::all() {
    EvaluationOptions options;
    options.metrics = {
        Metric::mean_squared_error,
        Metric::mean_absolute_error,
        Metric::root_mean_squared_error,
        Metric::r2_score,
        Metric::accuracy,
        Metric::precision,
        Metric::recall,
        Metric::f1_score,
        Metric::confusion_matrix,
    };
    return options;
}

EvaluationOptions& EvaluationOptions::include(Metric metric) {
    require(metric != Metric::custom, "custom metrics must be registered with add_custom_metric");
    require(!contains_metric(metrics, metric), "duplicate metric requested");
    metrics.push_back(metric);
    return *this;
}

EvaluationOptions& EvaluationOptions::add_custom_metric(std::string metric_name,
                                                        CustomMetricFunction evaluate,
                                                        void* context) {
    require(!metric_name.empty(), "custom metric name must not be empty");
    require(evaluate != nullptr, "custom metric callback is required");
    for (const CustomMetric& metric : custom_metrics) {
        require(metric.name != metric_name, "duplicate custom metric name");
    }
    custom_metrics.push_back(CustomMetric{std::move(metric_name), evaluate, context});
    return *this;
}

bool EvaluationResult::has_metric(Metric metric_value) const noexcept {
    if (metric_value == Metric::confusion_matrix) {
        return has_confusion_matrix;
    }
    return std::any_of(values.begin(), values.end(), [metric_value](const MetricValue& value) {
        return value.metric == metric_value;
    });
}

bool EvaluationResult::has_custom_metric(std::string_view metric_name) const noexcept {
    return std::any_of(values.begin(), values.end(), [metric_name](const MetricValue& value) {
        return value.metric == Metric::custom && value.name == metric_name;
    });
}

Scalar EvaluationResult::metric(Metric metric_value) const {
    require(metric_value != Metric::custom, "use custom_metric to read custom metric values");
    require(metric_value != Metric::confusion_matrix, "confusion_matrix is not a scalar metric");
    for (const MetricValue& value : values) {
        if (value.metric == metric_value) {
            return value.value;
        }
    }
    throw std::out_of_range("metric is not present: " + to_string(metric_value));
}

Scalar EvaluationResult::custom_metric(std::string_view metric_name) const {
    for (const MetricValue& value : values) {
        if (value.metric == Metric::custom && value.name == metric_name) {
            return value.value;
        }
    }
    throw std::out_of_range("custom metric is not present: " + std::string(metric_name));
}

EvaluationResult evaluate_predictions(const Matrix& predictions,
                                      const Matrix& targets,
                                      const EvaluationOptions& options) {
    validate_evaluation_data(predictions, targets);
    validate_evaluation_options(options);

    const std::vector<Metric> metrics = selected_metrics(options);
    const bool needs_regression = std::any_of(metrics.begin(), metrics.end(), is_regression_metric);
    const bool needs_classification = std::any_of(metrics.begin(), metrics.end(), is_classification_metric);

    EvaluationResult result;
    RegressionSummary regression;
    if (needs_regression) {
        regression = summarize_regression(predictions, targets);
    }

    ClassificationSummary classification;
    if (needs_classification) {
        result.confusion_matrix = make_confusion_matrix(predictions, targets, options.classification_threshold);
        result.has_confusion_matrix = true;
        classification = summarize_classification(result.confusion_matrix, options);
    }

    for (Metric metric : metrics) {
        switch (metric) {
        case Metric::mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_squared_error});
            break;
        case Metric::mean_absolute_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_absolute_error});
            break;
        case Metric::root_mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.root_mean_squared_error});
            break;
        case Metric::r2_score:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.r2_score});
            break;
        case Metric::accuracy:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.accuracy});
            break;
        case Metric::precision:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.precision});
            break;
        case Metric::recall:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.recall});
            break;
        case Metric::f1_score:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.f1_score});
            break;
        case Metric::confusion_matrix:
            break;
        case Metric::custom:
            throw std::invalid_argument("custom metrics must be registered with add_custom_metric");
        }
    }

    for (const CustomMetric& metric : options.custom_metrics) {
        const Scalar value = metric.evaluate(predictions, targets, metric.context);
        require_finite(value, "custom metric");
        result.values.push_back(MetricValue{Metric::custom, metric.name, value});
    }

    return result;
}

EvaluationResult evaluate_predictions(std::initializer_list<Vector> predictions,
                                      std::initializer_list<Vector> targets,
                                      const EvaluationOptions& options) {
    return evaluate_predictions(Matrix(predictions), Matrix(targets), options);
}

EvaluationResult evaluate_predictions(const DenseMatrix& predictions,
                                      const DenseMatrix& targets,
                                      const EvaluationOptions& options) {
    validate_evaluation_data(predictions, targets);
    validate_evaluation_options(options);

    const std::vector<Metric> metrics = selected_metrics(options);
    const bool needs_regression = std::any_of(metrics.begin(), metrics.end(), is_regression_metric);
    const bool needs_classification = std::any_of(metrics.begin(), metrics.end(), is_classification_metric);

    EvaluationResult result;
    RegressionSummary regression;
    if (needs_regression) {
        regression = summarize_regression(predictions, targets);
    }

    ClassificationSummary classification;
    if (needs_classification) {
        result.confusion_matrix = make_confusion_matrix(predictions, targets, options.classification_threshold);
        result.has_confusion_matrix = true;
        classification = summarize_classification(result.confusion_matrix, options);
    }

    for (Metric metric : metrics) {
        switch (metric) {
        case Metric::mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_squared_error});
            break;
        case Metric::mean_absolute_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.mean_absolute_error});
            break;
        case Metric::root_mean_squared_error:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.root_mean_squared_error});
            break;
        case Metric::r2_score:
            result.values.push_back(MetricValue{metric, to_string(metric), regression.r2_score});
            break;
        case Metric::accuracy:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.accuracy});
            break;
        case Metric::precision:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.precision});
            break;
        case Metric::recall:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.recall});
            break;
        case Metric::f1_score:
            result.values.push_back(MetricValue{metric, to_string(metric), classification.f1_score});
            break;
        case Metric::confusion_matrix:
            break;
        case Metric::custom:
            throw std::invalid_argument("custom metrics must be registered with add_custom_metric");
        }
    }

    if (!options.custom_metrics.empty()) {
        const Matrix prediction_matrix = predictions.to_matrix();
        const Matrix target_matrix = targets.to_matrix();
        for (const CustomMetric& metric : options.custom_metrics) {
            const Scalar value = metric.evaluate(prediction_matrix, target_matrix, metric.context);
            require_finite(value, "custom metric");
            result.values.push_back(MetricValue{Metric::custom, metric.name, value});
        }
    }

    return result;
}

Dataset::~Dataset() = default;

MatrixDataset::MatrixDataset(const Matrix& inputs, const Matrix& targets)
    : inputs_(&inputs),
      targets_(&targets),
      input_size_(inputs.empty() ? 0 : inputs.front().size()),
      output_size_(targets.empty() ? 0 : targets.front().size()) {
    validate_matrix(inputs, input_size_, "inputs");
    validate_matrix(targets, output_size_, "targets");
    require(inputs.size() == targets.size(), "inputs and targets must have the same number of rows");
    require(input_size_ > 0, "inputs must have at least one column");
    require(output_size_ > 0, "targets must have at least one column");
}

std::size_t MatrixDataset::sample_count() const noexcept {
    return inputs_ ? inputs_->size() : 0;
}

std::size_t MatrixDataset::input_size() const noexcept {
    return input_size_;
}

std::size_t MatrixDataset::output_size() const noexcept {
    return output_size_;
}

void MatrixDataset::sample(std::size_t index,
                           Scalar* input,
                           std::size_t provided_input_size,
                           Scalar* target,
                           std::size_t provided_target_size) const {
    require(inputs_ != nullptr && targets_ != nullptr, "matrix dataset is not initialized");
    require(index < sample_count(), "dataset sample index is out of range");
    require(input != nullptr, "dataset input buffer must not be null");
    require(target != nullptr, "dataset target buffer must not be null");
    require(provided_input_size == input_size_, "dataset input buffer size mismatch");
    require(provided_target_size == output_size_, "dataset target buffer size mismatch");

    const Vector& source_input = (*inputs_)[index];
    const Vector& source_target = (*targets_)[index];
    std::copy(source_input.begin(), source_input.end(), input);
    std::copy(source_target.begin(), source_target.end(), target);
}

DenseMatrixDataset::DenseMatrixDataset(const DenseMatrix& inputs, const DenseMatrix& targets)
    : inputs_(&inputs),
      targets_(&targets) {
    validate_dense_matrix(inputs, inputs.columns(), "inputs");
    validate_dense_matrix(targets, targets.columns(), "targets");
    require(inputs.rows() == targets.rows(), "inputs and targets must have the same number of rows");
    require(inputs.columns() > 0, "inputs must have at least one column");
    require(targets.columns() > 0, "targets must have at least one column");
}

std::size_t DenseMatrixDataset::sample_count() const noexcept {
    return inputs_ ? inputs_->rows() : 0;
}

std::size_t DenseMatrixDataset::input_size() const noexcept {
    return inputs_ ? inputs_->columns() : 0;
}

std::size_t DenseMatrixDataset::output_size() const noexcept {
    return targets_ ? targets_->columns() : 0;
}

void DenseMatrixDataset::sample(std::size_t index,
                                Scalar* input,
                                std::size_t provided_input_size,
                                Scalar* target,
                                std::size_t provided_target_size) const {
    require(inputs_ != nullptr && targets_ != nullptr, "dense matrix dataset is not initialized");
    require(index < sample_count(), "dataset sample index is out of range");
    require(input != nullptr, "dataset input buffer must not be null");
    require(target != nullptr, "dataset target buffer must not be null");
    require(provided_input_size == input_size(), "dataset input buffer size mismatch");
    require(provided_target_size == output_size(), "dataset target buffer size mismatch");

    const Scalar* source_input = inputs_->row_data(index);
    const Scalar* source_target = targets_->row_data(index);
    std::copy(source_input, source_input + input_size(), input);
    std::copy(source_target, source_target + output_size(), target);
}

GeneratedDataset::GeneratedDataset(std::size_t sample_count_value,
                                   std::size_t input_size_value,
                                   std::size_t output_size_value,
                                   IndexedSampleFunction generate,
                                   void* context)
    : sample_count_(sample_count_value),
      input_size_(input_size_value),
      output_size_(output_size_value),
      generate_(generate),
      context_(context) {
    require(sample_count_ > 0, "generated dataset sample_count must be positive");
    require(input_size_ > 0, "generated dataset input_size must be positive");
    require(output_size_ > 0, "generated dataset output_size must be positive");
    require(generate_ != nullptr, "generated dataset callback is required");
}

std::size_t GeneratedDataset::sample_count() const noexcept {
    return sample_count_;
}

std::size_t GeneratedDataset::input_size() const noexcept {
    return input_size_;
}

std::size_t GeneratedDataset::output_size() const noexcept {
    return output_size_;
}

void GeneratedDataset::sample(std::size_t index,
                              Scalar* input,
                              std::size_t provided_input_size,
                              Scalar* target,
                              std::size_t provided_target_size) const {
    require(index < sample_count_, "generated dataset sample index is out of range");
    require(input != nullptr, "generated dataset input buffer must not be null");
    require(target != nullptr, "generated dataset target buffer must not be null");
    require(provided_input_size == input_size_, "generated dataset input buffer size mismatch");
    require(provided_target_size == output_size_, "generated dataset target buffer size mismatch");
    generate_(index, input, input_size_, target, output_size_, context_);
}

DatasetSplit make_dataset_split(std::size_t sample_count, const DatasetSplitOptions& options) {
    require(sample_count > 0, "sample_count must be positive");
    validate_dataset_split_options(options);

    std::vector<std::size_t> indices = make_indices(sample_count);
    if (options.shuffle) {
        std::mt19937_64 rng(non_zero_seed(options.seed, 5489u));
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    const std::size_t validation_count =
        static_cast<std::size_t>(std::floor(static_cast<Scalar>(sample_count) * options.validation_split));
    const std::size_t test_count =
        static_cast<std::size_t>(std::floor(static_cast<Scalar>(sample_count) * options.test_split));
    require(validation_count + test_count < sample_count,
            "dataset split leaves no training samples");

    DatasetSplit split;
    const auto test_begin = indices.end() - static_cast<std::ptrdiff_t>(test_count);
    const auto validation_begin = test_begin - static_cast<std::ptrdiff_t>(validation_count);
    split.training_indices.assign(indices.begin(), validation_begin);
    split.validation_indices.assign(validation_begin, test_begin);
    split.test_indices.assign(test_begin, indices.end());
    return split;
}

DatasetView::DatasetView(const Dataset& source, std::vector<std::size_t> indices)
    : source_(&source),
      indices_(std::move(indices)) {
    require(source_ != nullptr, "dataset view source must not be null");
    for (std::size_t index : indices_) {
        require(index < source.sample_count(), "dataset view index is out of range");
    }
}

std::size_t DatasetView::sample_count() const noexcept {
    return indices_.size();
}

std::size_t DatasetView::input_size() const noexcept {
    return source_ ? source_->input_size() : 0;
}

std::size_t DatasetView::output_size() const noexcept {
    return source_ ? source_->output_size() : 0;
}

const std::vector<std::size_t>& DatasetView::indices() const noexcept {
    return indices_;
}

void DatasetView::sample(std::size_t index,
                         Scalar* input,
                         std::size_t provided_input_size,
                         Scalar* target,
                         std::size_t provided_target_size) const {
    require(source_ != nullptr, "dataset view source must not be null");
    require(index < indices_.size(), "dataset view sample index is out of range");
    source_->sample(indices_[index], input, provided_input_size, target, provided_target_size);
}

void MiniBatch::resize(std::size_t samples, std::size_t input_width, std::size_t output_width) {
    require(input_width > 0 || samples == 0, "mini-batch input width must be positive");
    require(output_width > 0 || samples == 0, "mini-batch output width must be positive");
    sample_count = samples;
    input_size = input_width;
    output_size = output_width;
    inputs.resize(sample_count * input_size);
    targets.resize(sample_count * output_size);
}

std::size_t MiniBatch::size() const noexcept {
    return sample_count;
}

bool MiniBatch::empty() const noexcept {
    return sample_count == 0;
}

void MiniBatch::clear() noexcept {
    sample_count = 0;
    input_size = 0;
    output_size = 0;
    inputs.clear();
    targets.clear();
}

Scalar* MiniBatch::input_data(std::size_t sample) {
    require(sample < sample_count, "mini-batch input sample index is out of range");
    return inputs.data() + sample * input_size;
}

const Scalar* MiniBatch::input_data(std::size_t sample) const {
    require(sample < sample_count, "mini-batch input sample index is out of range");
    return inputs.data() + sample * input_size;
}

Scalar* MiniBatch::target_data(std::size_t sample) {
    require(sample < sample_count, "mini-batch target sample index is out of range");
    return targets.data() + sample * output_size;
}

const Scalar* MiniBatch::target_data(std::size_t sample) const {
    require(sample < sample_count, "mini-batch target sample index is out of range");
    return targets.data() + sample * output_size;
}

Vector MiniBatch::input(std::size_t sample) const {
    const Scalar* row = input_data(sample);
    return Vector(row, row + input_size);
}

Vector MiniBatch::target(std::size_t sample) const {
    const Scalar* row = target_data(sample);
    return Vector(row, row + output_size);
}

struct BatchGenerator::Impl {
    const Dataset* dataset{nullptr};
    std::size_t batch_size{0};
    bool shuffle{false};
    std::uint64_t seed{0};
    std::size_t cursor{0};
    std::vector<std::size_t> indices;
};

BatchGenerator::BatchGenerator(const Dataset& dataset,
                               std::size_t batch_size,
                               bool shuffle,
                               std::uint64_t seed)
    : impl_(std::make_unique<Impl>()) {
    require(batch_size > 0, "batch_size must be positive");
    require(dataset.sample_count() > 0, "dataset must not be empty");
    impl_->dataset = &dataset;
    impl_->batch_size = batch_size;
    impl_->shuffle = shuffle;
    impl_->seed = seed;
    reset();
}

BatchGenerator::BatchGenerator(const BatchGenerator& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

BatchGenerator& BatchGenerator::operator=(const BatchGenerator& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

BatchGenerator::BatchGenerator(BatchGenerator&& other) noexcept = default;

BatchGenerator& BatchGenerator::operator=(BatchGenerator&& other) noexcept = default;

BatchGenerator::~BatchGenerator() = default;

void BatchGenerator::reset(std::uint64_t epoch) {
    require(impl_ && impl_->dataset, "batch generator is not initialized");
    impl_->indices = make_indices(impl_->dataset->sample_count());
    impl_->cursor = 0;
    if (impl_->shuffle) {
        std::mt19937_64 rng(non_zero_seed(impl_->seed, 5489u) + epoch);
        std::shuffle(impl_->indices.begin(), impl_->indices.end(), rng);
    }
}

bool BatchGenerator::next(MiniBatch& batch) {
    require(impl_ && impl_->dataset, "batch generator is not initialized");
    batch.clear();
    if (impl_->cursor >= impl_->indices.size()) {
        return false;
    }

    const std::size_t count = std::min(impl_->batch_size, impl_->indices.size() - impl_->cursor);
    batch.resize(count, impl_->dataset->input_size(), impl_->dataset->output_size());
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t row = impl_->indices[impl_->cursor + i];
        impl_->dataset->sample(row,
                               batch.input_data(i),
                               batch.input_size,
                               batch.target_data(i),
                               batch.output_size);
    }
    impl_->cursor += count;
    return true;
}

StreamingDataset::~StreamingDataset() = default;

FunctionStreamingDataset::FunctionStreamingDataset(std::size_t sample_count_value,
                                                   std::size_t input_size_value,
                                                   std::size_t output_size_value,
                                                   StreamingSampleFunction next,
                                                   StreamingResetFunction reset,
                                                   void* context)
    : sample_count_(sample_count_value),
      input_size_(input_size_value),
      output_size_(output_size_value),
      next_(next),
      reset_(reset),
      context_(context) {
    require(sample_count_ > 0, "streaming dataset sample_count must be positive");
    require(input_size_ > 0, "streaming dataset input_size must be positive");
    require(output_size_ > 0, "streaming dataset output_size must be positive");
    require(next_ != nullptr, "streaming dataset callback is required");
}

std::size_t FunctionStreamingDataset::sample_count() const noexcept {
    return sample_count_;
}

std::size_t FunctionStreamingDataset::input_size() const noexcept {
    return input_size_;
}

std::size_t FunctionStreamingDataset::output_size() const noexcept {
    return output_size_;
}

void FunctionStreamingDataset::reset() {
    if (reset_) {
        reset_(context_);
    }
}

bool FunctionStreamingDataset::next(Scalar* input,
                                    std::size_t provided_input_size,
                                    Scalar* target,
                                    std::size_t provided_target_size) {
    require(input != nullptr, "streaming dataset input buffer must not be null");
    require(target != nullptr, "streaming dataset target buffer must not be null");
    require(provided_input_size == input_size_, "streaming dataset input buffer size mismatch");
    require(provided_target_size == output_size_, "streaming dataset target buffer size mismatch");
    return next_(input, input_size_, target, output_size_, context_);
}

namespace {

[[nodiscard]] std::size_t count_csv_samples(const std::filesystem::path& path, const CsvStreamingOptions& options) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open csv dataset for reading: " + path.string());
    }

    std::string line;
    if (options.has_header) {
        std::getline(input, line);
    }

    std::size_t count = 0;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            ++count;
        }
    }
    return count;
}

void parse_csv_line(const std::string& line,
                    char delimiter,
                    Scalar* input,
                    std::size_t input_size,
                    Scalar* target,
                    std::size_t target_size) {
    std::stringstream stream(line);
    std::string token;
    std::size_t column = 0;
    const std::size_t total_columns = input_size + target_size;

    while (std::getline(stream, token, delimiter)) {
        require(column < total_columns, "csv row has too many columns");
        Scalar value{};
        try {
            value = static_cast<Scalar>(std::stod(token));
        } catch (const std::exception&) {
            throw std::runtime_error("failed to parse csv numeric value");
        }
        require_finite(value, "csv value");
        if (column < input_size) {
            input[column] = value;
        } else {
            target[column - input_size] = value;
        }
        ++column;
    }

    require(column == total_columns, "csv row has an invalid column count");
}

} // namespace

struct CsvStreamingDataset::Impl {
    std::filesystem::path path;
    std::size_t input_size{0};
    std::size_t output_size{0};
    CsvStreamingOptions options{};
    std::size_t sample_count{0};
    std::ifstream stream;
};

CsvStreamingDataset::CsvStreamingDataset(const std::filesystem::path& path,
                                         std::size_t input_size,
                                         std::size_t output_size,
                                         CsvStreamingOptions options)
    : impl_(std::make_unique<Impl>()) {
    require(input_size > 0, "csv dataset input_size must be positive");
    require(output_size > 0, "csv dataset output_size must be positive");
    impl_->path = path;
    impl_->input_size = input_size;
    impl_->output_size = output_size;
    impl_->options = options;
    impl_->sample_count = count_csv_samples(path, options);
    require(impl_->sample_count > 0, "csv dataset must not be empty");
    reset();
}

CsvStreamingDataset::CsvStreamingDataset(const CsvStreamingDataset& other)
    : impl_(std::make_unique<Impl>()) {
    impl_->path = other.impl_->path;
    impl_->input_size = other.impl_->input_size;
    impl_->output_size = other.impl_->output_size;
    impl_->options = other.impl_->options;
    impl_->sample_count = other.impl_->sample_count;
    reset();
}

CsvStreamingDataset& CsvStreamingDataset::operator=(const CsvStreamingDataset& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>();
        impl_->path = other.impl_->path;
        impl_->input_size = other.impl_->input_size;
        impl_->output_size = other.impl_->output_size;
        impl_->options = other.impl_->options;
        impl_->sample_count = other.impl_->sample_count;
        reset();
    }
    return *this;
}

CsvStreamingDataset::CsvStreamingDataset(CsvStreamingDataset&& other) noexcept = default;

CsvStreamingDataset& CsvStreamingDataset::operator=(CsvStreamingDataset&& other) noexcept = default;

CsvStreamingDataset::~CsvStreamingDataset() = default;

std::size_t CsvStreamingDataset::sample_count() const noexcept {
    return impl_ ? impl_->sample_count : 0;
}

std::size_t CsvStreamingDataset::input_size() const noexcept {
    return impl_ ? impl_->input_size : 0;
}

std::size_t CsvStreamingDataset::output_size() const noexcept {
    return impl_ ? impl_->output_size : 0;
}

void CsvStreamingDataset::reset() {
    require(impl_ != nullptr, "csv dataset is not initialized");
    impl_->stream.close();
    impl_->stream.clear();
    impl_->stream.open(impl_->path);
    if (!impl_->stream) {
        throw std::runtime_error("failed to open csv dataset for reading: " + impl_->path.string());
    }
    if (impl_->options.has_header) {
        std::string header;
        std::getline(impl_->stream, header);
    }
}

bool CsvStreamingDataset::next(Scalar* input,
                               std::size_t provided_input_size,
                               Scalar* target,
                               std::size_t provided_target_size) {
    require(impl_ != nullptr, "csv dataset is not initialized");
    require(input != nullptr, "csv dataset input buffer must not be null");
    require(target != nullptr, "csv dataset target buffer must not be null");
    require(provided_input_size == impl_->input_size, "csv dataset input buffer size mismatch");
    require(provided_target_size == impl_->output_size, "csv dataset target buffer size mismatch");

    std::string line;
    while (std::getline(impl_->stream, line)) {
        if (line.empty()) {
            continue;
        }
        parse_csv_line(line, impl_->options.delimiter, input, impl_->input_size, target, impl_->output_size);
        return true;
    }
    return false;
}

OptimizerConfig OptimizerConfig::adam(Scalar learning_rate) {
    OptimizerConfig config;
    config.type = OptimizerType::adam;
    config.learning_rate = learning_rate;
    return config;
}

OptimizerConfig OptimizerConfig::adamw(Scalar learning_rate, Scalar weight_decay) {
    OptimizerConfig config;
    config.type = OptimizerType::adamw;
    config.learning_rate = learning_rate;
    config.decoupled_weight_decay = weight_decay;
    return config;
}

OptimizerConfig OptimizerConfig::sgd(Scalar learning_rate, Scalar momentum) {
    OptimizerConfig config;
    config.type = OptimizerType::sgd;
    config.learning_rate = learning_rate;
    config.momentum = momentum;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::constant() {
    return LearningRateScheduleConfig{};
}

LearningRateScheduleConfig LearningRateScheduleConfig::step_decay(std::size_t step_size,
                                                                  Scalar decay_factor) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::step_decay;
    config.step_size = step_size;
    config.step_decay_factor = decay_factor;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::exponential_decay(Scalar decay_rate) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::exponential_decay;
    config.exponential_decay_rate = decay_rate;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::cosine_annealing(std::size_t cosine_epochs,
                                                                        Scalar minimum_learning_rate) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::cosine_annealing;
    config.cosine_epochs = cosine_epochs;
    config.minimum_learning_rate = minimum_learning_rate;
    return config;
}

LearningRateScheduleConfig LearningRateScheduleConfig::reduce_on_plateau(std::size_t patience,
                                                                         Scalar factor) {
    LearningRateScheduleConfig config;
    config.type = LearningRateSchedule::reduce_on_plateau;
    config.reduce_on_plateau_patience = patience;
    config.reduce_on_plateau_factor = factor;
    return config;
}

InitializationConfig InitializationConfig::automatic(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::automatic;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::he_normal(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::he_normal;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::he_uniform(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::he_uniform;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::xavier_normal(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::xavier_normal;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::xavier_uniform(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::xavier_uniform;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::lecun_normal(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::lecun_normal;
    config.bias = bias;
    return config;
}

InitializationConfig InitializationConfig::lecun_uniform(Scalar bias) {
    InitializationConfig config;
    config.weights = WeightInitialization::lecun_uniform;
    config.bias = bias;
    return config;
}

FeatureNormalization FeatureNormalization::none() {
    return FeatureNormalization{};
}

FeatureNormalization FeatureNormalization::standard_score(Scalar mean,
                                                          Scalar stddev,
                                                          bool clamp,
                                                          Scalar clamp_min,
                                                          Scalar clamp_max) {
    FeatureNormalization feature;
    feature.mode = NormalizationMode::standard_score;
    feature.mean = mean;
    feature.stddev = stddev;
    feature.clamp = clamp;
    feature.clamp_min = clamp_min;
    feature.clamp_max = clamp_max;
    validate_feature_normalization(feature);
    return feature;
}

FeatureNormalization FeatureNormalization::min_max(Scalar minimum,
                                                   Scalar maximum,
                                                   Scalar normalized_min,
                                                   Scalar normalized_max,
                                                   bool clamp) {
    FeatureNormalization feature;
    feature.mode = NormalizationMode::min_max;
    feature.minimum = minimum;
    feature.maximum = maximum;
    feature.normalized_min = normalized_min;
    feature.normalized_max = normalized_max;
    feature.clamp = clamp;
    feature.clamp_min = std::min(normalized_min, normalized_max);
    feature.clamp_max = std::max(normalized_min, normalized_max);
    validate_feature_normalization(feature);
    return feature;
}

NormalizationSpec NormalizationSpec::none() {
    return NormalizationSpec{};
}

NormalizationSpec NormalizationSpec::standard_score(const Vector& input_means,
                                                    const Vector& input_stddevs,
                                                    const Vector& output_means,
                                                    const Vector& output_stddevs) {
    require(input_means.size() == input_stddevs.size(), "input standard_score vector size mismatch");
    require(output_means.size() == output_stddevs.size(), "output standard_score vector size mismatch");

    NormalizationSpec spec;
    spec.input_features.reserve(input_means.size());
    for (std::size_t i = 0; i < input_means.size(); ++i) {
        spec.input_features.push_back(FeatureNormalization::standard_score(input_means[i], input_stddevs[i]));
    }

    spec.output_features.reserve(output_means.size());
    for (std::size_t i = 0; i < output_means.size(); ++i) {
        spec.output_features.push_back(FeatureNormalization::standard_score(output_means[i], output_stddevs[i]));
    }
    return spec;
}

NormalizationSpec NormalizationSpec::min_max(const Vector& input_minimums,
                                             const Vector& input_maximums,
                                             const Vector& output_minimums,
                                             const Vector& output_maximums,
                                             Scalar normalized_min,
                                             Scalar normalized_max,
                                             bool clamp) {
    require(input_minimums.size() == input_maximums.size(), "input min_max vector size mismatch");
    require(output_minimums.size() == output_maximums.size(), "output min_max vector size mismatch");

    NormalizationSpec spec;
    spec.input_features.reserve(input_minimums.size());
    for (std::size_t i = 0; i < input_minimums.size(); ++i) {
        spec.input_features.push_back(
            FeatureNormalization::min_max(input_minimums[i], input_maximums[i], normalized_min, normalized_max, clamp));
    }

    spec.output_features.reserve(output_minimums.size());
    for (std::size_t i = 0; i < output_minimums.size(); ++i) {
        spec.output_features.push_back(FeatureNormalization::min_max(
            output_minimums[i], output_maximums[i], normalized_min, normalized_max, clamp));
    }
    return spec;
}

bool NormalizationSpec::has_input_normalization() const noexcept {
    return has_feature_normalization(input_features);
}

bool NormalizationSpec::has_output_normalization() const noexcept {
    return has_feature_normalization(output_features);
}

struct InferenceModel::Impl {
    struct Layer {
        std::size_t input_size{0};
        std::size_t output_size{0};
        std::size_t weight_offset{0};
        std::size_t bias_offset{0};
        Activation activation{Activation::linear};
    };

    std::size_t input_size{0};
    std::size_t output_size{0};
    std::size_t scratch_stride{0};
    std::size_t input_scratch_size{0};
    NormalizationSpec normalization{};
    std::vector<Layer> layers;
    Vector weights;
    Vector biases;

    [[nodiscard]] std::size_t scratch_size() const noexcept {
        return input_scratch_size + (scratch_stride == 0 ? 0 : scratch_stride * 2);
    }

    void predict_unchecked_to(const Scalar* input, Scalar* output, Scalar* scratch) const noexcept {
        Scalar* scratch_cursor = scratch;
        const Scalar* previous = input;
        if (normalization.has_input_normalization()) {
            for (std::size_t i = 0; i < input_size; ++i) {
                scratch_cursor[i] = normalize_value(input[i], normalization.input_features[i]);
            }
            previous = scratch_cursor;
            scratch_cursor += input_scratch_size;
        }

        Scalar* scratch_a = scratch_cursor;
        Scalar* scratch_b = scratch ? scratch + scratch_stride : nullptr;
        if (scratch_b && input_scratch_size > 0) {
            scratch_b += input_scratch_size;
        }
        bool write_a = true;

        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const Layer& layer = layers[layer_index];
            const bool output_layer = layer_index + 1 == layers.size();
            Scalar* target = output_layer ? output : (write_a ? scratch_a : scratch_b);

            for (std::size_t out = 0; out < layer.output_size; ++out) {
                Scalar value = biases[layer.bias_offset + out];
                const std::size_t weight_offset = layer.weight_offset + out * layer.input_size;
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    value += weights[weight_offset + in] * previous[in];
                }
                target[out] = value;
            }

            apply_activation_in_place(target, layer.output_size, layer.activation);
            previous = target;
            write_a = !write_a;
        }

        if (normalization.has_output_normalization()) {
            for (std::size_t i = 0; i < output_size; ++i) {
                output[i] = denormalize_value(output[i], normalization.output_features[i]);
            }
        }
    }

    void predict_batch_range_unchecked_to(const Scalar* inputs,
                                          std::size_t sample_begin,
                                          std::size_t sample_end,
                                          std::size_t input_stride,
                                          Scalar* outputs,
                                          std::size_t output_stride,
                                          Scalar* scratch) const noexcept {
        for (std::size_t sample = sample_begin; sample < sample_end; ++sample) {
            predict_unchecked_to(inputs + sample * input_stride,
                                 outputs + sample * output_stride,
                                 scratch);
        }
    }

    void predict_batch_unchecked_to(const Scalar* inputs,
                                    std::size_t sample_count,
                                    std::size_t input_stride,
                                    Scalar* outputs,
                                    std::size_t output_stride,
                                    Scalar* scratch) const noexcept {
        predict_batch_range_unchecked_to(inputs, 0, sample_count, input_stride, outputs, output_stride, scratch);
    }
};

InferenceModel::InferenceModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CompiledModel::CompiledModel(InferenceModel model)
    : InferenceModel(std::move(model)) {}

CompiledModel CompiledModel::from_parameters(const std::vector<LayerParameters>& parameters,
                                             const NormalizationSpec& normalization) {
    return CompiledModel(InferenceModel::from_parameters(parameters, normalization));
}

CompiledModel CompiledModel::load(const std::filesystem::path& path) {
    return CompiledModel(InferenceModel::load(path));
}

CompiledModel CompiledModel::load_binary(const std::filesystem::path& path) {
    return CompiledModel(InferenceModel::load_binary(path));
}

InferenceModel::InferenceModel(const InferenceModel& other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr) {}

InferenceModel& InferenceModel::operator=(const InferenceModel& other) {
    if (this != &other) {
        impl_ = other.impl_ ? std::make_unique<Impl>(*other.impl_) : nullptr;
    }
    return *this;
}

InferenceModel::InferenceModel(InferenceModel&& other) noexcept = default;

InferenceModel& InferenceModel::operator=(InferenceModel&& other) noexcept = default;

InferenceModel::~InferenceModel() = default;

InferenceModel InferenceModel::from_parameters(const std::vector<LayerParameters>& parameters,
                                               const NormalizationSpec& normalization) {
    validate_layer_parameters(parameters);
    validate_normalization_spec(normalization, parameters.front().input_size, parameters.back().output_size);

    auto impl = std::make_unique<Impl>();
    impl->input_size = parameters.front().input_size;
    impl->output_size = parameters.back().output_size;
    impl->normalization = normalization;
    impl->layers.reserve(parameters.size());

    std::size_t weight_count = 0;
    std::size_t bias_count = 0;
    std::size_t max_hidden_width = 0;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const LayerParameters& layer = parameters[i];
        weight_count += layer.weights.size();
        bias_count += layer.biases.size();
        if (i + 1 < parameters.size()) {
            max_hidden_width = std::max(max_hidden_width, layer.output_size);
        }
    }

    impl->weights.reserve(weight_count);
    impl->biases.reserve(bias_count);
    impl->input_scratch_size = normalization.has_input_normalization() ? impl->input_size : 0;
    impl->scratch_stride = max_hidden_width;

    for (const LayerParameters& source : parameters) {
        const std::size_t weight_offset = impl->weights.size();
        const std::size_t bias_offset = impl->biases.size();
        impl->weights.insert(impl->weights.end(), source.weights.begin(), source.weights.end());
        impl->biases.insert(impl->biases.end(), source.biases.begin(), source.biases.end());
        impl->layers.push_back(Impl::Layer{
            source.input_size,
            source.output_size,
            weight_offset,
            bias_offset,
            source.activation,
        });
    }

    return InferenceModel(std::move(impl));
}

bool InferenceModel::empty() const noexcept {
    return !impl_ || impl_->layers.empty();
}

std::size_t InferenceModel::input_size() const noexcept {
    return empty() ? 0 : impl_->input_size;
}

std::size_t InferenceModel::output_size() const noexcept {
    return empty() ? 0 : impl_->output_size;
}

std::size_t InferenceModel::layer_count() const noexcept {
    return empty() ? 0 : impl_->layers.size();
}

std::size_t InferenceModel::scratch_size() const noexcept {
    return empty() ? 0 : impl_->scratch_size();
}

std::size_t InferenceModel::batch_scratch_size(std::size_t sample_count,
                                               const ParallelOptions& parallelism) const noexcept {
    return empty() ? 0 : required_batch_scratch_size(impl_->scratch_size(), sample_count, parallelism);
}

std::size_t InferenceModel::weight_count() const noexcept {
    return empty() ? 0 : impl_->weights.size();
}

std::size_t InferenceModel::bias_count() const noexcept {
    return empty() ? 0 : impl_->biases.size();
}

const Scalar* InferenceModel::weights_data() const noexcept {
    return empty() ? nullptr : impl_->weights.data();
}

const Scalar* InferenceModel::biases_data() const noexcept {
    return empty() ? nullptr : impl_->biases.data();
}

NormalizationSpec InferenceModel::normalization() const {
    return empty() ? NormalizationSpec{} : impl_->normalization;
}

InferenceStatus InferenceModel::predict_to(const Scalar* input,
                                           std::size_t provided_input_size,
                                           Scalar* output,
                                           std::size_t provided_output_size,
                                           Scalar* scratch,
                                           std::size_t provided_scratch_size) const noexcept {
    if (!input) {
        return InferenceStatus::null_input;
    }
    if (!output) {
        return InferenceStatus::null_output;
    }
    if (empty() || provided_input_size != impl_->input_size) {
        return InferenceStatus::invalid_input_size;
    }
    if (provided_output_size != impl_->output_size) {
        return InferenceStatus::invalid_output_size;
    }
    if (impl_->scratch_size() > 0 && !scratch) {
        return InferenceStatus::null_scratch;
    }
    if (provided_scratch_size < impl_->scratch_size()) {
        return InferenceStatus::insufficient_scratch;
    }

    impl_->predict_unchecked_to(input, output, scratch);
    return InferenceStatus::ok;
}

InferenceStatus InferenceModel::predict_to(const Scalar* input,
                                           std::size_t provided_input_size,
                                           InferenceWorkspace& workspace) const noexcept {
    return predict_to(input,
                      provided_input_size,
                      workspace.output_data(),
                      workspace.output_size(),
                      workspace.scratch_data(),
                      workspace.scratch_size());
}

InferenceStatus InferenceModel::predict_batch_to(const Scalar* inputs,
                                                 std::size_t sample_count,
                                                 std::size_t input_stride,
                                                 Scalar* outputs,
                                                 std::size_t output_stride,
                                                 Scalar* scratch,
                                                 std::size_t provided_scratch_size) const noexcept {
    if (empty()) {
        return InferenceStatus::invalid_input_size;
    }
    if (sample_count == 0) {
        return InferenceStatus::ok;
    }
    if (!inputs) {
        return InferenceStatus::null_input;
    }
    if (!outputs) {
        return InferenceStatus::null_output;
    }
    if (input_stride < impl_->input_size) {
        return InferenceStatus::invalid_input_stride;
    }
    if (output_stride < impl_->output_size) {
        return InferenceStatus::invalid_output_stride;
    }
    if (impl_->scratch_size() > 0 && !scratch) {
        return InferenceStatus::null_scratch;
    }
    if (provided_scratch_size < impl_->scratch_size()) {
        return InferenceStatus::insufficient_scratch;
    }

    impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
    return InferenceStatus::ok;
}

InferenceStatus InferenceModel::predict_batch_to(const Scalar* inputs,
                                                 std::size_t sample_count,
                                                 std::size_t input_stride,
                                                 Scalar* outputs,
                                                 std::size_t output_stride,
                                                 Scalar* scratch,
                                                 std::size_t provided_scratch_size,
                                                 const ParallelOptions& parallelism) const noexcept {
    if (empty()) {
        return InferenceStatus::invalid_input_size;
    }
    if (sample_count == 0) {
        return InferenceStatus::ok;
    }
    if (!inputs) {
        return InferenceStatus::null_input;
    }
    if (!outputs) {
        return InferenceStatus::null_output;
    }
    if (input_stride < impl_->input_size) {
        return InferenceStatus::invalid_input_stride;
    }
    if (output_stride < impl_->output_size) {
        return InferenceStatus::invalid_output_stride;
    }

    const std::size_t worker_count = resolve_parallel_worker_count(parallelism, sample_count);
    const std::size_t single_scratch_size = impl_->scratch_size();
    const std::size_t required_scratch_size = required_batch_scratch_size(single_scratch_size,
                                                                         sample_count,
                                                                         parallelism);
    if (required_scratch_size == std::numeric_limits<std::size_t>::max()) {
        return InferenceStatus::insufficient_scratch;
    }
    if (required_scratch_size > 0 && !scratch) {
        return InferenceStatus::null_scratch;
    }
    if (provided_scratch_size < required_scratch_size) {
        return InferenceStatus::insufficient_scratch;
    }

    if (worker_count <= 1) {
        impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
        return InferenceStatus::ok;
    }

    std::vector<std::thread> workers;
    try {
        workers.reserve(worker_count - 1);
        const std::size_t base_count = sample_count / worker_count;
        const std::size_t remainder = sample_count % worker_count;
        std::size_t begin = 0;
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            const std::size_t count = base_count + (worker_index < remainder ? 1 : 0);
            const std::size_t end = begin + count;
            Scalar* worker_scratch = single_scratch_size == 0
                                         ? nullptr
                                         : scratch + worker_index * single_scratch_size;
            if (worker_index == 0) {
                begin = end;
                continue;
            }
            workers.emplace_back([impl = impl_.get(),
                                  inputs,
                                  begin,
                                  end,
                                  input_stride,
                                  outputs,
                                  output_stride,
                                  worker_scratch] {
                impl->predict_batch_range_unchecked_to(inputs,
                                                       begin,
                                                       end,
                                                       input_stride,
                                                       outputs,
                                                       output_stride,
                                                       worker_scratch);
            });
            begin = end;
        }

        const std::size_t first_end = base_count + (remainder > 0 ? 1 : 0);
        impl_->predict_batch_range_unchecked_to(inputs,
                                               0,
                                               first_end,
                                               input_stride,
                                               outputs,
                                               output_stride,
                                               scratch);
    } catch (...) {
        join_workers_noexcept(workers);
        impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
        return InferenceStatus::ok;
    }

    join_workers_noexcept(workers);
    return InferenceStatus::ok;
}

void InferenceModel::predict_unchecked_to(const Scalar* input,
                                          Scalar* output,
                                          Scalar* scratch) const noexcept {
    impl_->predict_unchecked_to(input, output, scratch);
}

void InferenceModel::predict_unchecked_to(const Scalar* input,
                                          InferenceWorkspace& workspace) const noexcept {
    predict_unchecked_to(input, workspace.output_data(), workspace.scratch_data());
}

void InferenceModel::predict_batch_unchecked_to(const Scalar* inputs,
                                                std::size_t sample_count,
                                                std::size_t input_stride,
                                                Scalar* outputs,
                                                std::size_t output_stride,
                                                Scalar* scratch) const noexcept {
    impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
}

Vector InferenceModel::predict(const Vector& input) const {
    Vector output(output_size(), Scalar{0});
    Vector scratch(scratch_size(), Scalar{0});
    const InferenceStatus status =
        predict_to(input.data(), input.size(), output.data(), output.size(), scratch.data(), scratch.size());
    if (status != InferenceStatus::ok) {
        throw std::invalid_argument("inference failed: " + to_string(status));
    }
    return output;
}

Matrix InferenceModel::predict_batch(const Matrix& inputs) const {
    validate_matrix(inputs, input_size(), "inputs", true);
    Matrix result;
    result.reserve(inputs.size());
    Vector scratch(scratch_size(), Scalar{0});

    for (const Vector& input : inputs) {
        Vector output(output_size(), Scalar{0});
        const InferenceStatus status =
            predict_to(input.data(), input.size(), output.data(), output.size(), scratch.data(), scratch.size());
        if (status != InferenceStatus::ok) {
            throw std::invalid_argument("inference failed: " + to_string(status));
        }
        result.push_back(std::move(output));
    }

    return result;
}

Matrix InferenceModel::predict_batch(std::initializer_list<Vector> inputs) const {
    return predict_batch(Matrix(inputs));
}

DenseMatrix InferenceModel::predict_batch(const DenseMatrix& inputs) const {
    validate_dense_matrix(inputs, input_size(), "inputs", true);
    DenseMatrix result(inputs.rows(), output_size());
    if (inputs.empty()) {
        return result;
    }

    Vector scratch(scratch_size(), Scalar{0});
    const InferenceStatus status = predict_batch_to(inputs.data(),
                                                    inputs.rows(),
                                                    inputs.columns(),
                                                    result.data(),
                                                    result.columns(),
                                                    scratch.data(),
                                                    scratch.size());
    if (status != InferenceStatus::ok) {
        throw std::invalid_argument("inference failed: " + to_string(status));
    }
    return result;
}

EvaluationResult InferenceModel::evaluate_metrics(const Matrix& inputs,
                                                  const Matrix& targets,
                                                  const EvaluationOptions& options) const {
    validate_matrix(inputs, input_size(), "inputs");
    validate_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

EvaluationResult InferenceModel::evaluate_metrics(std::initializer_list<Vector> inputs,
                                                  std::initializer_list<Vector> targets,
                                                  const EvaluationOptions& options) const {
    return evaluate_metrics(Matrix(inputs), Matrix(targets), options);
}

EvaluationResult InferenceModel::evaluate_metrics(const DenseMatrix& inputs,
                                                  const DenseMatrix& targets,
                                                  const EvaluationOptions& options) const {
    validate_dense_matrix(inputs, input_size(), "inputs");
    validate_dense_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

InferenceWorkspace::InferenceWorkspace(const InferenceModel& model) {
    resize_for(model);
}

void InferenceWorkspace::resize_for(const InferenceModel& model) {
    output_.assign(model.output_size(), Scalar{0});
    scratch_.assign(model.scratch_size(), Scalar{0});
}

void InferenceWorkspace::clear() noexcept {
    output_.clear();
    scratch_.clear();
}

std::size_t InferenceWorkspace::output_size() const noexcept {
    return output_.size();
}

std::size_t InferenceWorkspace::scratch_size() const noexcept {
    return scratch_.size();
}

Scalar* InferenceWorkspace::output_data() noexcept {
    return output_.data();
}

const Scalar* InferenceWorkspace::output_data() const noexcept {
    return output_.data();
}

Scalar* InferenceWorkspace::scratch_data() noexcept {
    return scratch_.data();
}

const Scalar* InferenceWorkspace::scratch_data() const noexcept {
    return scratch_.data();
}

const Vector& InferenceWorkspace::output() const noexcept {
    return output_;
}

std::vector<LayerParameters> InferenceModel::parameters() const {
    require(!empty(), "inference model is empty");
    std::vector<LayerParameters> result;
    result.reserve(impl_->layers.size());

    for (const Impl::Layer& layer : impl_->layers) {
        LayerParameters parameters;
        parameters.input_size = layer.input_size;
        parameters.output_size = layer.output_size;
        parameters.activation = layer.activation;
        parameters.weights.assign(impl_->weights.begin() + static_cast<std::ptrdiff_t>(layer.weight_offset),
                                  impl_->weights.begin() + static_cast<std::ptrdiff_t>(layer.weight_offset +
                                                                                      layer.input_size *
                                                                                          layer.output_size));
        parameters.biases.assign(impl_->biases.begin() + static_cast<std::ptrdiff_t>(layer.bias_offset),
                                 impl_->biases.begin() + static_cast<std::ptrdiff_t>(layer.bias_offset +
                                                                                    layer.output_size));
        result.push_back(std::move(parameters));
    }

    return result;
}

void InferenceModel::save(const std::filesystem::path& path) const {
    require(!empty(), "inference model is empty");

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open inference model file for writing: " + path.string());
    }

    output << std::setprecision(17);
    output << "BRUTAL_MLP_INFERENCE_V2\n";
    output << impl_->input_size << '\n';
    write_normalization(output, impl_->normalization);
    output << impl_->layers.size() << '\n';

    for (const Impl::Layer& layer : impl_->layers) {
        output << layer.input_size << ' ' << layer.output_size << ' ' << to_string(layer.activation) << '\n';

        const std::size_t weight_count = layer.input_size * layer.output_size;
        output << weight_count;
        for (std::size_t i = 0; i < weight_count; ++i) {
            output << ' ' << impl_->weights[layer.weight_offset + i];
        }
        output << '\n';

        output << layer.output_size;
        for (std::size_t i = 0; i < layer.output_size; ++i) {
            output << ' ' << impl_->biases[layer.bias_offset + i];
        }
        output << '\n';
    }

    if (!output) {
        throw std::runtime_error("failed while writing inference model file: " + path.string());
    }
}

void InferenceModel::save_binary(const std::filesystem::path& path, const BinaryMetadata& metadata) const {
    require(!empty(), "inference model is empty");

    const std::vector<LayerParameters> model_parameters = parameters();
    BinaryWriter payload;
    write_common_binary_model_payload(payload,
                                      metadata,
                                      input_size(),
                                      output_size(),
                                      0,
                                      false,
                                      TrainingOptions{},
                                      impl_->normalization,
                                      model_parameters);
    write_binary_file(path, build_binary_envelope(BinaryModelKind::inference, payload.bytes));
}

InferenceModel InferenceModel::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open inference model file for reading: " + path.string());
    }

    const std::string magic = read_token(input, "magic");
    const bool has_normalization = magic == "BRUTAL_MLP_INFERENCE_V2";
    if (!has_normalization && magic != "BRUTAL_MLP_INFERENCE_V1") {
        throw std::runtime_error("unsupported inference model file format");
    }

    const std::size_t input_size = read_value<std::size_t>(input, "input_size");
    require(input_size > 0, "inference input_size must be positive");
    NormalizationSpec normalization;
    if (has_normalization) {
        normalization = read_normalization(input);
    }
    const std::size_t layer_count = read_value<std::size_t>(input, "layer_count");
    validate_serialized_layer_count(layer_count, "inference model");
    std::vector<LayerParameters> parameters;
    parameters.reserve(layer_count);

    std::size_t expected_input_size = input_size;
    for (std::size_t i = 0; i < layer_count; ++i) {
        LayerParameters layer;
        layer.input_size = read_value<std::size_t>(input, "layer_input_size");
        layer.output_size = read_value<std::size_t>(input, "layer_output_size");
        layer.activation = activation_from_string(read_token(input, "layer_activation"));
        require(layer.input_size > 0, "inference layer input_size must be positive");
        require(layer.output_size > 0, "inference layer output_size must be positive");
        require(layer.input_size == expected_input_size, "inference file topology is inconsistent");
        const std::size_t expected_weight_count =
            checked_product(layer.input_size, layer.output_size, "inference weight");

        const std::size_t weight_count = read_value<std::size_t>(input, "weight_count");
        require_exact_serialized_count(weight_count, expected_weight_count, "inference weight");
        layer.weights.resize(weight_count);
        for (Scalar& weight : layer.weights) {
            weight = read_value<Scalar>(input, "weight");
        }

        const std::size_t bias_count = read_value<std::size_t>(input, "bias_count");
        require_exact_serialized_count(bias_count, layer.output_size, "inference bias");
        layer.biases.resize(bias_count);
        for (Scalar& bias : layer.biases) {
            bias = read_value<Scalar>(input, "bias");
        }

        expected_input_size = layer.output_size;
        parameters.push_back(std::move(layer));
    }

    require_text_stream_consumed(input, "inference model");
    validate_layer_parameters(parameters);
    validate_normalization_spec(normalization, input_size, parameters.back().output_size);
    return InferenceModel::from_parameters(parameters, normalization);
}

InferenceModel InferenceModel::load_binary(const std::filesystem::path& path) {
    ParsedBinaryModel parsed = parse_binary_model_file(path);
    if (parsed.info.model_kind != BinaryModelKind::inference) {
        throw std::runtime_error("binary model is not an inference model");
    }
    return InferenceModel::from_parameters(parsed.parameters, parsed.normalization);
}

struct TrainingModel::Impl {
    struct Layer {
        std::size_t input_size{0};
        std::size_t output_size{0};
        Activation activation{Activation::linear};
        Scalar dropout_probability{static_cast<Scalar>(0)};
        Vector weights;
        Vector biases;
        Vector grad_weights;
        Vector grad_biases;
        Vector velocity_weights;
        Vector velocity_biases;
        Vector first_moment_weights;
        Vector first_moment_biases;
        Vector second_moment_weights;
        Vector second_moment_biases;
    };

    std::size_t input_size{0};
    LossConfig loss{};
    OptimizerConfig optimizer{OptimizerConfig::adam()};
    NormalizationSpec normalization{};
    std::uint64_t seed{5489u};
    std::uint64_t optimizer_step{0};
    bool has_last_training_options{false};
    TrainingOptions last_training_options{};
    LearningRateSchedulerState learning_rate_state{};
    std::size_t completed_epochs{0};
    bool has_resume_epoch{false};
    std::size_t resume_epoch{0};
    std::vector<Layer> layers;

    [[nodiscard]] std::size_t output_size() const {
        return layers.back().output_size;
    }

    [[nodiscard]] TrainingRunConfig make_training_run_config(const OptimizerConfig& run_optimizer,
                                                             const TrainingOptions& runtime_options,
                                                             const DatasetSplit& split,
                                                             std::size_t sample_count,
                                                             bool streaming) const {
        TrainingRunConfig config;
        config.library_version = std::string(version_string());
        config.scalar_type = scalar_type_string();
        config.created_unix_time = current_unix_time();
        config.created_utc = utc_timestamp(config.created_unix_time);
        config.model_seed = seed;
        config.training_seed = non_zero_seed(runtime_options.seed, seed);
        config.input_size = input_size;
        config.output_size = output_size();

        config.architecture.reserve(layers.size());
        for (const Layer& layer : layers) {
            TrainingRunLayerConfig layer_config;
            layer_config.input_size = layer.input_size;
            layer_config.output_size = layer.output_size;
            layer_config.activation = layer.activation;
            layer_config.dropout_probability = layer.dropout_probability;
            config.architecture.push_back(layer_config);
        }

        config.optimizer = run_optimizer;
        config.learning_rate_schedule = runtime_options.learning_rate_schedule;
        config.loss = loss;
        config.normalization = normalization;
        config.options = runtime_options;

        config.dataset.sample_count = sample_count;
        config.dataset.input_size = input_size;
        config.dataset.output_size = output_size();
        config.dataset.training_sample_count = split.training_indices.size();
        config.dataset.validation_sample_count = split.validation_indices.size();
        config.dataset.test_sample_count = split.test_indices.size();
        config.dataset.validation_split = runtime_options.validation_split;
        config.dataset.test_split = runtime_options.test_split;
        config.dataset.shuffle = runtime_options.shuffle;
        config.dataset.split_seed = config.training_seed;
        config.dataset.streaming = streaming;
        config.dataset.streaming_shuffle_buffer_size = runtime_options.streaming_shuffle_buffer_size;

        config.result.monitor = runtime_options.early_stopping_monitor;
        config.result.monitor_mode = runtime_options.early_stopping_mode;
        return config;
    }

    [[nodiscard]] Vector forward(const Vector& input) const {
        Vector activations = input;
        for (const Layer& layer : layers) {
            Vector z(layer.output_size, Scalar{0});
            for (std::size_t out = 0; out < layer.output_size; ++out) {
                Scalar value = layer.biases[out];
                const std::size_t offset = out * layer.input_size;
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    value += layer.weights[offset + in] * activations[in];
                }
                z[out] = value;
            }
            activations = apply_activation(z, layer.activation);
        }
        return activations;
    }

    [[nodiscard]] Vector predict_raw(const Vector& input) const {
        return denormalize_vector(forward(normalize_vector(input, normalization.input_features)),
                                  normalization.output_features);
    }

    [[nodiscard]] Vector predict_raw(const Scalar* input, std::size_t input_count) const {
        require(input != nullptr, "input buffer must not be null");
        require(input_count == input_size, "input has an invalid size");
        Vector normalized(input_count, Scalar{0});
        normalize_buffer_into(input, input_count, normalization.input_features, normalized.data());
        return denormalize_vector(forward(normalized), normalization.output_features);
    }

    [[nodiscard]] Vector forward_cached(const Scalar* input,
                                        std::size_t input_count,
                                        std::vector<Vector>& activations,
                                        std::vector<Vector>& pre_activations) const {
        require(input != nullptr, "input buffer must not be null");
        require(input_count == input_size, "input has an invalid size");
        activations.clear();
        pre_activations.clear();
        activations.reserve(layers.size() + 1);
        pre_activations.reserve(layers.size());
        activations.emplace_back(input, input + input_count);

        for (const Layer& layer : layers) {
            const Vector& previous = activations.back();
            Vector z(layer.output_size, Scalar{0});
            for (std::size_t out = 0; out < layer.output_size; ++out) {
                Scalar value = layer.biases[out];
                const std::size_t offset = out * layer.input_size;
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    value += layer.weights[offset + in] * previous[in];
                }
                z[out] = value;
            }
            pre_activations.push_back(z);
            activations.push_back(apply_activation(z, layer.activation));
        }

        return activations.back();
    }

    [[nodiscard]] Vector forward_cached(const Vector& input,
                                        std::vector<Vector>& activations,
                                        std::vector<Vector>& pre_activations) const {
        return forward_cached(input.data(), input.size(), activations, pre_activations);
    }

    [[nodiscard]] Vector forward_cached_training(const Scalar* input,
                                                 std::size_t input_count,
                                                 std::vector<Vector>& activations,
                                                 std::vector<Vector>& pre_activations,
                                                 std::vector<Vector>& dropout_masks,
                                                 std::mt19937_64& rng) const {
        require(input != nullptr, "input buffer must not be null");
        require(input_count == input_size, "input has an invalid size");
        activations.clear();
        pre_activations.clear();
        dropout_masks.clear();
        activations.reserve(layers.size() + 1);
        pre_activations.reserve(layers.size());
        dropout_masks.reserve(layers.size());
        activations.emplace_back(input, input + input_count);

        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const Layer& layer = layers[layer_index];
            const Vector& previous = activations.back();
            Vector z(layer.output_size, Scalar{0});
            for (std::size_t out = 0; out < layer.output_size; ++out) {
                Scalar value = layer.biases[out];
                const std::size_t offset = out * layer.input_size;
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    value += layer.weights[offset + in] * previous[in];
                }
                z[out] = value;
            }

            Vector activation = apply_activation(z, layer.activation);
            Vector dropout_mask(layer.output_size, Scalar{1});
            const bool output_layer = layer_index + 1 == layers.size();
            if (!output_layer && layer.dropout_probability > Scalar{0}) {
                const Scalar keep_probability = Scalar{1} - layer.dropout_probability;
                const Scalar scale = Scalar{1} / keep_probability;
                std::bernoulli_distribution keep(static_cast<double>(keep_probability));
                for (std::size_t i = 0; i < activation.size(); ++i) {
                    if (keep(rng)) {
                        dropout_mask[i] = scale;
                        activation[i] *= scale;
                    } else {
                        dropout_mask[i] = Scalar{0};
                        activation[i] = Scalar{0};
                    }
                }
            }

            pre_activations.push_back(std::move(z));
            dropout_masks.push_back(std::move(dropout_mask));
            activations.push_back(std::move(activation));
        }

        return activations.back();
    }

    [[nodiscard]] Scalar sample_loss(const Vector& prediction, const Scalar* target, std::size_t target_count) const {
        require(target != nullptr, "target buffer must not be null");
        require(target_count == prediction.size(), "target has an invalid size");
        Scalar total = Scalar{0};

        switch (loss.type) {
        case Loss::mean_squared_error:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar delta = prediction[i] - target[i];
                total += delta * delta;
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::mean_absolute_error:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                total += static_cast<Scalar>(std::abs(prediction[i] - target[i]));
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::huber:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar abs_delta = static_cast<Scalar>(std::abs(prediction[i] - target[i]));
                if (abs_delta <= loss.huber_delta) {
                    total += Scalar{0.5} * abs_delta * abs_delta;
                } else {
                    total += loss.huber_delta * (abs_delta - Scalar{0.5} * loss.huber_delta);
                }
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::relative_mean_squared_error:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar delta = prediction[i] - target[i];
                const Scalar denominator = std::max(static_cast<Scalar>(std::abs(target[i])), loss.relative_epsilon);
                total += (delta * delta) / (denominator * denominator);
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::log_cosh:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar abs_delta = static_cast<Scalar>(std::abs(prediction[i] - target[i]));
                total += abs_delta +
                         static_cast<Scalar>(std::log1p(std::exp(-Scalar{2} * abs_delta))) -
                         static_cast<Scalar>(std::log(Scalar{2}));
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::weighted_mean_squared_error: {
            const Scalar total_weight = std::accumulate(loss.weights.begin(), loss.weights.end(), Scalar{0});
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar delta = prediction[i] - target[i];
                total += loss.weights[i] * delta * delta;
            }
            return total / total_weight;
        }
        case Loss::binary_cross_entropy:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar p = clamp_probability(prediction[i]);
                total += static_cast<Scalar>(-target[i] * std::log(p) -
                                             (Scalar{1} - target[i]) * std::log(Scalar{1} - p));
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::categorical_cross_entropy:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                total += static_cast<Scalar>(-target[i] * std::log(clamp_probability(prediction[i])));
            }
            return total;
        case Loss::custom: {
            const Scalar value = loss.custom_loss.value(prediction.data(),
                                                        target,
                                                        prediction.size(),
                                                        loss.custom_loss.context);
            require_finite(value, "custom loss");
            return value;
        }
        }

        throw std::invalid_argument("unknown loss");
    }

    [[nodiscard]] Scalar sample_loss(const Vector& prediction, const Vector& target) const {
        return sample_loss(prediction, target.data(), target.size());
    }

    [[nodiscard]] Vector output_delta(const Vector& prediction,
                                      const Scalar* target,
                                      std::size_t target_count,
                                      const Vector& output_pre_activation) const {
        require(target != nullptr, "target buffer must not be null");
        require(target_count == prediction.size(), "target has an invalid size");
        const Layer& output_layer = layers.back();
        Vector delta(prediction.size(), Scalar{0});

        if (loss.type == Loss::categorical_cross_entropy && output_layer.activation == Activation::softmax) {
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                delta[i] = prediction[i] - target[i];
            }
            return delta;
        }

        if (loss.type == Loss::binary_cross_entropy && output_layer.activation == Activation::sigmoid) {
            const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                delta[i] = (prediction[i] - target[i]) * scale;
            }
            return delta;
        }

        Vector activation_gradient(prediction.size(), Scalar{0});
        switch (loss.type) {
        case Loss::mean_squared_error: {
            const Scalar scale = Scalar{2} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                activation_gradient[i] = (prediction[i] - target[i]) * scale;
            }
            break;
        }
        case Loss::mean_absolute_error: {
            const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar delta_value = prediction[i] - target[i];
                if (delta_value > Scalar{0}) {
                    activation_gradient[i] = scale;
                } else if (delta_value < Scalar{0}) {
                    activation_gradient[i] = -scale;
                }
            }
            break;
        }
        case Loss::huber: {
            const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar delta_value = prediction[i] - target[i];
                const Scalar abs_delta = static_cast<Scalar>(std::abs(delta_value));
                if (abs_delta <= loss.huber_delta) {
                    activation_gradient[i] = delta_value * scale;
                } else {
                    activation_gradient[i] = (delta_value > Scalar{0} ? loss.huber_delta : -loss.huber_delta) * scale;
                }
            }
            break;
        }
        case Loss::relative_mean_squared_error: {
            const Scalar scale = Scalar{2} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar denominator = std::max(static_cast<Scalar>(std::abs(target[i])), loss.relative_epsilon);
                activation_gradient[i] =
                    (prediction[i] - target[i]) * scale / (denominator * denominator);
            }
            break;
        }
        case Loss::log_cosh: {
            const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                activation_gradient[i] = static_cast<Scalar>(std::tanh(prediction[i] - target[i])) * scale;
            }
            break;
        }
        case Loss::weighted_mean_squared_error: {
            const Scalar total_weight = std::accumulate(loss.weights.begin(), loss.weights.end(), Scalar{0});
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                activation_gradient[i] =
                    Scalar{2} * loss.weights[i] * (prediction[i] - target[i]) / total_weight;
            }
            break;
        }
        case Loss::binary_cross_entropy: {
            const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar p = clamp_probability(prediction[i]);
                activation_gradient[i] = ((Scalar{1} - target[i]) / (Scalar{1} - p) - target[i] / p) * scale;
            }
            break;
        }
        case Loss::categorical_cross_entropy:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                activation_gradient[i] = -target[i] / clamp_probability(prediction[i]);
            }
            break;
        case Loss::custom:
            loss.custom_loss.gradient(prediction.data(),
                                      target,
                                      prediction.size(),
                                      activation_gradient.data(),
                                      loss.custom_loss.context);
            for (Scalar value : activation_gradient) {
                require_finite(value, "custom loss gradient");
            }
            break;
        }

        if (output_layer.activation == Activation::softmax) {
            const Scalar dot = std::inner_product(activation_gradient.begin(),
                                                  activation_gradient.end(),
                                                  prediction.begin(),
                                                  Scalar{0});
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                delta[i] = prediction[i] * (activation_gradient[i] - dot);
            }
            return delta;
        }

        for (std::size_t i = 0; i < prediction.size(); ++i) {
            delta[i] = activation_gradient[i] *
                       activation_derivative(output_layer.activation, output_pre_activation[i], prediction[i]);
        }
        return delta;
    }

    [[nodiscard]] Vector output_delta(const Vector& prediction,
                                      const Vector& target,
                                      const Vector& output_pre_activation) const {
        return output_delta(prediction, target.data(), target.size(), output_pre_activation);
    }

    void accumulate_gradients(const Vector& input, const Vector& target) {
        accumulate_gradients(input.data(), input.size(), target.data(), target.size());
    }

    void accumulate_gradients(const Scalar* input,
                              std::size_t input_count,
                              const Scalar* target,
                              std::size_t target_count,
                              std::mt19937_64* rng = nullptr) {
        std::vector<Vector> activations;
        std::vector<Vector> pre_activations;
        std::vector<Vector> dropout_masks;
        const Vector prediction =
            rng == nullptr
                ? forward_cached(input, input_count, activations, pre_activations)
                : forward_cached_training(input, input_count, activations, pre_activations, dropout_masks, *rng);
        Vector delta = output_delta(prediction, target, target_count, pre_activations.back());

        for (std::size_t layer_offset = layers.size(); layer_offset-- > 0;) {
            Layer& layer = layers[layer_offset];
            const Vector& previous_activation = activations[layer_offset];

            for (std::size_t out = 0; out < layer.output_size; ++out) {
                layer.grad_biases[out] += delta[out];
                const std::size_t weight_offset = out * layer.input_size;
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    layer.grad_weights[weight_offset + in] += delta[out] * previous_activation[in];
                }
            }

            if (layer_offset == 0) {
                continue;
            }

            const Layer& previous_layer = layers[layer_offset - 1];
            Vector previous_delta(layer.input_size, Scalar{0});
            for (std::size_t in = 0; in < layer.input_size; ++in) {
                Scalar value = Scalar{0};
                for (std::size_t out = 0; out < layer.output_size; ++out) {
                    value += layer.weights[out * layer.input_size + in] * delta[out];
                }
                if (rng != nullptr) {
                    value *= dropout_masks[layer_offset - 1][in];
                    previous_delta[in] =
                        value * activation_derivative_from_pre_activation(
                                    previous_layer.activation,
                                    pre_activations[layer_offset - 1][in]);
                } else {
                    previous_delta[in] =
                        value * activation_derivative(previous_layer.activation,
                                                      pre_activations[layer_offset - 1][in],
                                                      activations[layer_offset][in]);
                }
            }
            delta = std::move(previous_delta);
        }
    }

    void accumulate_batch_gradients(const MiniBatch& batch, std::mt19937_64* rng = nullptr) {
        require(batch.input_size == input_size, "mini-batch input width mismatch");
        require(batch.output_size == output_size(), "mini-batch output width mismatch");
        for (std::size_t sample = 0; sample < batch.size(); ++sample) {
            accumulate_gradients(batch.input_data(sample),
                                 batch.input_size,
                                 batch.target_data(sample),
                                 batch.output_size,
                                 rng);
        }
    }

    void zero_gradients() {
        for (Layer& layer : layers) {
            std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), Scalar{0});
            std::fill(layer.grad_biases.begin(), layer.grad_biases.end(), Scalar{0});
        }
    }

    [[nodiscard]] Scalar accumulated_gradient_norm(std::size_t batch_size) const {
        const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
        Scalar norm_squared = Scalar{0};
        for (const Layer& layer : layers) {
            for (std::size_t i = 0; i < layer.weights.size(); ++i) {
                Scalar gradient = layer.grad_weights[i] * batch_scale;
                gradient += optimizer.l2 * layer.weights[i];
                gradient += optimizer.l1 * sign_or_zero(layer.weights[i]);
                norm_squared += gradient * gradient;
            }
            for (Scalar gradient_value : layer.grad_biases) {
                const Scalar gradient = gradient_value * batch_scale;
                norm_squared += gradient * gradient;
            }
        }

        return static_cast<Scalar>(std::sqrt(norm_squared));
    }

    [[nodiscard]] Scalar layer_gradient_norm(const Layer& layer, Scalar batch_scale) const {
        Scalar norm_squared = Scalar{0};
        for (std::size_t i = 0; i < layer.weights.size(); ++i) {
            Scalar gradient = layer.grad_weights[i] * batch_scale;
            gradient += optimizer.l2 * layer.weights[i];
            gradient += optimizer.l1 * sign_or_zero(layer.weights[i]);
            norm_squared += gradient * gradient;
        }
        for (Scalar gradient_value : layer.grad_biases) {
            const Scalar gradient = gradient_value * batch_scale;
            norm_squared += gradient * gradient;
        }
        return static_cast<Scalar>(std::sqrt(norm_squared));
    }

    [[nodiscard]] WeightStatistics weight_statistics() const {
        WeightStatistics statistics;
        bool has_finite_weight = false;
        Scalar total = Scalar{0};

        for (const Layer& layer : layers) {
            for (Scalar weight : layer.weights) {
                ++statistics.count;
                if (!is_finite(weight)) {
                    statistics.finite = false;
                    ++statistics.non_finite_count;
                    continue;
                }
                if (!has_finite_weight) {
                    statistics.minimum = weight;
                    statistics.maximum = weight;
                    has_finite_weight = true;
                } else {
                    statistics.minimum = std::min(statistics.minimum, weight);
                    statistics.maximum = std::max(statistics.maximum, weight);
                }
                total += weight;
            }
        }

        const std::size_t finite_count = statistics.count - statistics.non_finite_count;
        if (finite_count > 0) {
            statistics.mean = total / static_cast<Scalar>(finite_count);
        }
        return statistics;
    }

    [[nodiscard]] std::size_t non_finite_parameter_count() const {
        std::size_t count = 0;
        for (const Layer& layer : layers) {
            for (Scalar weight : layer.weights) {
                if (!is_finite(weight)) {
                    ++count;
                }
            }
            for (Scalar bias : layer.biases) {
                if (!is_finite(bias)) {
                    ++count;
                }
            }
        }
        return count;
    }

    [[nodiscard]] std::string first_non_finite_parameter_location() const {
        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const Layer& layer = layers[layer_index];
            for (std::size_t i = 0; i < layer.weights.size(); ++i) {
                const Scalar value = layer.weights[i];
                if (!is_finite(value)) {
                    return "layer " + std::to_string(layer_index) + " weight[" +
                           std::to_string(i) + "]=" + scalar_debug_string(value) +
                           " (" + non_finite_kind(value) + ")";
                }
            }
            for (std::size_t i = 0; i < layer.biases.size(); ++i) {
                const Scalar value = layer.biases[i];
                if (!is_finite(value)) {
                    return "layer " + std::to_string(layer_index) + " bias[" +
                           std::to_string(i) + "]=" + scalar_debug_string(value) +
                           " (" + non_finite_kind(value) + ")";
                }
            }
        }
        return "unknown parameter";
    }

    [[nodiscard]] std::string first_non_finite_gradient_location(std::size_t batch_size) const {
        const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const Layer& layer = layers[layer_index];
            for (std::size_t i = 0; i < layer.grad_weights.size(); ++i) {
                const Scalar raw_gradient = layer.grad_weights[i];
                if (!is_finite(raw_gradient)) {
                    return "layer " + std::to_string(layer_index) + " grad_weight[" +
                           std::to_string(i) + "]=" + scalar_debug_string(raw_gradient) +
                           " (" + non_finite_kind(raw_gradient) + ")";
                }
                Scalar gradient = raw_gradient * batch_scale;
                gradient += optimizer.l2 * layer.weights[i];
                gradient += optimizer.l1 * sign_or_zero(layer.weights[i]);
                if (!is_finite(gradient)) {
                    return "layer " + std::to_string(layer_index) + " effective grad_weight[" +
                           std::to_string(i) + "]=" + scalar_debug_string(gradient) +
                           " (" + non_finite_kind(gradient) + ")";
                }
            }
            for (std::size_t i = 0; i < layer.grad_biases.size(); ++i) {
                const Scalar raw_gradient = layer.grad_biases[i];
                if (!is_finite(raw_gradient)) {
                    return "layer " + std::to_string(layer_index) + " grad_bias[" +
                           std::to_string(i) + "]=" + scalar_debug_string(raw_gradient) +
                           " (" + non_finite_kind(raw_gradient) + ")";
                }
                const Scalar gradient = raw_gradient * batch_scale;
                if (!is_finite(gradient)) {
                    return "layer " + std::to_string(layer_index) + " effective grad_bias[" +
                           std::to_string(i) + "]=" + scalar_debug_string(gradient) +
                           " (" + non_finite_kind(gradient) + ")";
                }
            }
        }
        return "unknown gradient";
    }

    struct GradientApplicationResult {
        Scalar gradient_norm{static_cast<Scalar>(0)};
        GradientClippingDiagnostics clipping{};
        std::string non_finite_location{};
    };

    [[nodiscard]] GradientApplicationResult apply_gradients(std::size_t batch_size,
                                                            Scalar gradient_noise_stddev = Scalar{0},
                                                            std::mt19937_64* rng = nullptr) {
        GradientApplicationResult result;
        const Scalar gradient_norm = accumulated_gradient_norm(batch_size);
        result.gradient_norm = gradient_norm;
        if (!is_finite(gradient_norm)) {
            result.non_finite_location = first_non_finite_gradient_location(batch_size);
            zero_gradients();
            return result;
        }

        ++optimizer_step;
        const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
        result.clipping.batch_count = 1;
        Scalar global_clip_scale = Scalar{1};
        if (optimizer.gradient_clip_norm > Scalar{0}) {
            if (gradient_norm > optimizer.gradient_clip_norm) {
                global_clip_scale =
                    optimizer.gradient_clip_norm / (gradient_norm + std::numeric_limits<Scalar>::epsilon());
                result.clipping.global_clip_count = 1;
            }
        }
        result.clipping.minimum_clip_scale = std::min(result.clipping.minimum_clip_scale, global_clip_scale);

        for (Layer& layer : layers) {
            Scalar layer_clip_scale = Scalar{1};
            if (optimizer.layer_gradient_clip_norm > Scalar{0}) {
                const Scalar layer_norm = layer_gradient_norm(layer, batch_scale);
                if (layer_norm > optimizer.layer_gradient_clip_norm) {
                    layer_clip_scale =
                        optimizer.layer_gradient_clip_norm / (layer_norm + std::numeric_limits<Scalar>::epsilon());
                    ++result.clipping.layer_clip_count;
                }
            }
            ++result.clipping.layer_count;
            const Scalar norm_clip_scale = global_clip_scale * layer_clip_scale;
            result.clipping.minimum_clip_scale = std::min(result.clipping.minimum_clip_scale, norm_clip_scale);
            apply_vector_gradients(layer.weights,
                                   layer.grad_weights,
                                   layer.velocity_weights,
                                   layer.first_moment_weights,
                                   layer.second_moment_weights,
                                   batch_scale,
                                   norm_clip_scale,
                                   true,
                                   gradient_noise_stddev,
                                   rng,
                                   result.clipping);
            apply_vector_gradients(layer.biases,
                                   layer.grad_biases,
                                   layer.velocity_biases,
                                   layer.first_moment_biases,
                                   layer.second_moment_biases,
                                   batch_scale,
                                   norm_clip_scale,
                                   false,
                                   gradient_noise_stddev,
                                   rng,
                                   result.clipping);
        }

        finalize_gradient_clipping(result.clipping);
        enforce_max_norm();
        zero_gradients();
        return result;
    }

    void apply_vector_gradients(Vector& values,
                                const Vector& accumulated_gradients,
                                Vector& velocities,
                                Vector& first_moments,
                                Vector& second_moments,
                                Scalar batch_scale,
                                Scalar clip_scale,
                                bool regularize,
                                Scalar gradient_noise_stddev,
                                std::mt19937_64* rng,
                                GradientClippingDiagnostics& clipping) {
        std::normal_distribution<Scalar> noise(Scalar{0}, gradient_noise_stddev);
        for (std::size_t i = 0; i < values.size(); ++i) {
            Scalar gradient = accumulated_gradients[i] * batch_scale;
            if (regularize) {
                gradient += optimizer.l2 * values[i];
                gradient += optimizer.l1 * sign_or_zero(values[i]);
            }
            if (gradient_noise_stddev > Scalar{0} && rng != nullptr) {
                gradient += noise(*rng);
            }
            gradient *= clip_scale;
            ++clipping.gradient_value_count;
            if (optimizer.gradient_clip_value > Scalar{0}) {
                if (gradient > optimizer.gradient_clip_value) {
                    gradient = optimizer.gradient_clip_value;
                    ++clipping.gradient_value_clip_count;
                } else if (gradient < -optimizer.gradient_clip_value) {
                    gradient = -optimizer.gradient_clip_value;
                    ++clipping.gradient_value_clip_count;
                }
            }

            switch (optimizer.type) {
            case OptimizerType::sgd:
                if (optimizer.momentum > Scalar{0}) {
                    velocities[i] = optimizer.momentum * velocities[i] - optimizer.learning_rate * gradient;
                    values[i] += velocities[i];
                } else {
                    values[i] -= optimizer.learning_rate * gradient;
                }
                break;
            case OptimizerType::adam:
            case OptimizerType::adamw: {
                if (regularize && optimizer.type == OptimizerType::adamw &&
                    optimizer.decoupled_weight_decay > Scalar{0}) {
                    values[i] *= Scalar{1} - optimizer.learning_rate * optimizer.decoupled_weight_decay;
                }
                first_moments[i] = optimizer.beta1 * first_moments[i] + (Scalar{1} - optimizer.beta1) * gradient;
                second_moments[i] =
                    optimizer.beta2 * second_moments[i] + (Scalar{1} - optimizer.beta2) * gradient * gradient;

                const Scalar bias_correction1 =
                    Scalar{1} - static_cast<Scalar>(std::pow(optimizer.beta1, static_cast<Scalar>(optimizer_step)));
                const Scalar bias_correction2 =
                    Scalar{1} - static_cast<Scalar>(std::pow(optimizer.beta2, static_cast<Scalar>(optimizer_step)));
                const Scalar m_hat = first_moments[i] / bias_correction1;
                const Scalar v_hat = second_moments[i] / bias_correction2;
                values[i] -= optimizer.learning_rate * m_hat /
                             (static_cast<Scalar>(std::sqrt(v_hat)) + optimizer.epsilon);
                break;
            }
            }
        }
    }

    void enforce_max_norm() {
        if (optimizer.max_norm <= Scalar{0}) {
            return;
        }
        for (Layer& layer : layers) {
            for (std::size_t out = 0; out < layer.output_size; ++out) {
                const std::size_t offset = out * layer.input_size;
                Scalar norm_squared = Scalar{0};
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    const Scalar weight = layer.weights[offset + in];
                    norm_squared += weight * weight;
                }
                const Scalar norm = static_cast<Scalar>(std::sqrt(norm_squared));
                if (norm > optimizer.max_norm && norm > Scalar{0}) {
                    const Scalar scale = optimizer.max_norm / norm;
                    for (std::size_t in = 0; in < layer.input_size; ++in) {
                        layer.weights[offset + in] *= scale;
                    }
                }
            }
        }
    }

    [[nodiscard]] Scalar loss_for_indices(const Matrix& inputs,
                                          const Matrix& targets,
                                          const std::vector<std::size_t>& indices) const {
        Scalar total = Scalar{0};
        for (std::size_t index : indices) {
            total += sample_loss(forward(inputs[index]), targets[index]);
        }
        return total / static_cast<Scalar>(indices.size());
    }

    void normalize_sample(const Scalar* raw_input,
                          std::size_t raw_input_size,
                          const Scalar* raw_target,
                          std::size_t raw_target_size,
                          Scalar* normalized_input,
                          Scalar* normalized_target) const {
        validate_sample_values(raw_input, raw_input_size, raw_target, raw_target_size, loss.type);
        normalize_buffer_into(raw_input, raw_input_size, normalization.input_features, normalized_input);
        normalize_buffer_into(raw_target, raw_target_size, normalization.output_features, normalized_target);
    }

    void normalize_sample(const Vector& raw_input,
                          const Vector& raw_target,
                          Vector& normalized_input,
                          Vector& normalized_target) const {
        normalized_input.resize(raw_input.size());
        normalized_target.resize(raw_target.size());
        normalize_sample(raw_input.data(),
                         raw_input.size(),
                         raw_target.data(),
                         raw_target.size(),
                         normalized_input.data(),
                         normalized_target.data());
    }

    void read_dataset_sample(const Dataset& dataset,
                             std::size_t index,
                             Vector& raw_input,
                             Vector& raw_target,
                             Vector& normalized_input,
                             Vector& normalized_target,
                             const TrainingDebugOptions* debug = nullptr,
                             const TrainingDebugContext& debug_context = {}) const {
        dataset.sample(index, raw_input.data(), raw_input.size(), raw_target.data(), raw_target.size());
        if (debug != nullptr && debug->enabled) {
            debug_check_sample_values(raw_input.data(),
                                      raw_input.size(),
                                      raw_target.data(),
                                      raw_target.size(),
                                      debug_context);
        }
        normalize_sample(raw_input, raw_target, normalized_input, normalized_target);
    }

    void fill_dataset_batch(const Dataset& dataset,
                            const std::vector<std::size_t>& indices,
                            std::size_t begin,
                            std::size_t count,
                            MiniBatch& raw_batch,
                            MiniBatch& normalized_batch,
                            const TrainingDebugOptions* debug = nullptr,
                            const TrainingDebugContext& base_debug_context = {}) const {
        require(begin <= indices.size(), "batch begin index is out of range");
        require(count <= indices.size() - begin, "batch count is out of range");
        raw_batch.resize(count, input_size, output_size());
        normalized_batch.resize(count, input_size, output_size());

        for (std::size_t sample = 0; sample < count; ++sample) {
            dataset.sample(indices[begin + sample],
                           raw_batch.input_data(sample),
                           raw_batch.input_size,
                           raw_batch.target_data(sample),
                           raw_batch.output_size);
            if (debug != nullptr && debug->enabled) {
                TrainingDebugContext context = base_debug_context;
                context.sample_index = indices[begin + sample];
                context.has_sample_index = true;
                debug_check_sample_values(raw_batch.input_data(sample),
                                          raw_batch.input_size,
                                          raw_batch.target_data(sample),
                                          raw_batch.output_size,
                                          context);
            }
            normalize_sample(raw_batch.input_data(sample),
                             raw_batch.input_size,
                             raw_batch.target_data(sample),
                             raw_batch.output_size,
                             normalized_batch.input_data(sample),
                             normalized_batch.target_data(sample));
        }
    }

    [[nodiscard]] Scalar loss_for_dataset_indices(const Dataset& dataset,
                                                  const std::vector<std::size_t>& indices,
                                                  const TrainingDebugOptions* debug = nullptr,
                                                  const char* debug_phase = "loss",
                                                  std::size_t epoch = 0,
                                                  bool has_epoch = false) const {
        require(!indices.empty(), "dataset loss requires at least one sample");
        Vector raw_input(input_size);
        Vector raw_target(output_size());
        Vector normalized_input(input_size);
        Vector normalized_target(output_size());
        Scalar total = Scalar{0};
        for (std::size_t index : indices) {
            TrainingDebugContext context;
            context.phase = debug_phase;
            context.epoch = epoch;
            context.has_epoch = has_epoch;
            context.sample_index = index;
            context.has_sample_index = true;
            read_dataset_sample(dataset,
                                index,
                                raw_input,
                                raw_target,
                                normalized_input,
                                normalized_target,
                                debug,
                                context);
            total += sample_loss(forward(normalized_input), normalized_target);
        }
        return total / static_cast<Scalar>(indices.size());
    }

    void read_streaming_sample(StreamingDataset& dataset,
                               Vector& raw_input,
                               Vector& raw_target,
                               Vector& normalized_input,
                               Vector& normalized_target,
                               const TrainingDebugOptions* debug = nullptr,
                               const TrainingDebugContext& debug_context = {}) const {
        const bool has_sample =
            dataset.next(raw_input.data(), raw_input.size(), raw_target.data(), raw_target.size());
        require(has_sample, "streaming dataset ended before sample_count");
        if (debug != nullptr && debug->enabled) {
            debug_check_sample_values(raw_input.data(),
                                      raw_input.size(),
                                      raw_target.data(),
                                      raw_target.size(),
                                      debug_context);
        }
        normalize_sample(raw_input, raw_target, normalized_input, normalized_target);
    }

    [[nodiscard]] std::vector<LayerParameters> parameters() const {
        std::vector<LayerParameters> result;
        result.reserve(layers.size());
        for (const Layer& layer : layers) {
            result.push_back(LayerParameters{
                layer.input_size,
                layer.output_size,
                layer.activation,
                layer.weights,
                layer.biases,
                layer.dropout_probability,
            });
        }
        return result;
    }

    [[nodiscard]] std::vector<LayerTrainingState> training_states() const {
        std::vector<LayerTrainingState> result;
        result.reserve(layers.size());
        for (const Layer& layer : layers) {
            result.push_back(LayerTrainingState{
                layer.grad_weights,
                layer.grad_biases,
                layer.velocity_weights,
                layer.velocity_biases,
                layer.first_moment_weights,
                layer.first_moment_biases,
                layer.second_moment_weights,
                layer.second_moment_biases,
            });
        }
        return result;
    }

    void set_parameters(const std::vector<LayerParameters>& parameters) {
        require(parameters.size() == layers.size(), "parameter layer count does not match model topology");
        for (std::size_t i = 0; i < layers.size(); ++i) {
            Layer& layer = layers[i];
            const LayerParameters& source = parameters[i];
            require(source.input_size == layer.input_size, "parameter input_size mismatch");
            require(source.output_size == layer.output_size, "parameter output_size mismatch");
            require(source.activation == layer.activation, "parameter activation mismatch");
            require(source.weights.size() == layer.weights.size(), "parameter weights size mismatch");
            require(source.biases.size() == layer.biases.size(), "parameter biases size mismatch");
            validate_dropout_probability(source.dropout_probability, "parameter dropout_probability");
            if (i + 1 == layers.size()) {
                require(source.dropout_probability == Scalar{0}, "output layer dropout must be zero");
            }
            for (Scalar value : source.weights) {
                require_finite(value, "weight");
            }
            for (Scalar value : source.biases) {
                require_finite(value, "bias");
            }
            layer.weights = source.weights;
            layer.biases = source.biases;
            layer.dropout_probability = source.dropout_probability;
            reset_optimizer_state(layer);
        }
        optimizer_step = 0;
    }

    void set_training_states(const std::vector<LayerTrainingState>& states, std::uint64_t step) {
        require(states.size() == layers.size(), "checkpoint optimizer state layer count mismatch");
        const std::vector<LayerParameters> current_parameters = parameters();
        for (std::size_t i = 0; i < layers.size(); ++i) {
            validate_layer_training_state(states[i], current_parameters[i]);
            Layer& layer = layers[i];
            layer.grad_weights = states[i].grad_weights;
            layer.grad_biases = states[i].grad_biases;
            layer.velocity_weights = states[i].velocity_weights;
            layer.velocity_biases = states[i].velocity_biases;
            layer.first_moment_weights = states[i].first_moment_weights;
            layer.first_moment_biases = states[i].first_moment_biases;
            layer.second_moment_weights = states[i].second_moment_weights;
            layer.second_moment_biases = states[i].second_moment_biases;
        }
        optimizer_step = step;
    }

    static void reset_optimizer_state(Layer& layer) {
        std::fill(layer.velocity_weights.begin(), layer.velocity_weights.end(), Scalar{0});
        std::fill(layer.velocity_biases.begin(), layer.velocity_biases.end(), Scalar{0});
        std::fill(layer.first_moment_weights.begin(), layer.first_moment_weights.end(), Scalar{0});
        std::fill(layer.first_moment_biases.begin(), layer.first_moment_biases.end(), Scalar{0});
        std::fill(layer.second_moment_weights.begin(), layer.second_moment_weights.end(), Scalar{0});
        std::fill(layer.second_moment_biases.begin(), layer.second_moment_biases.end(), Scalar{0});
    }
};

TrainingModel::Builder& TrainingModel::Builder::input_size(std::size_t input_size) {
    input_size_ = input_size;
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::add_layer(std::size_t neurons,
                                                          Activation activation,
                                                          Scalar dropout_probability) {
    layers_.push_back(LayerSpec{neurons, activation, dropout_probability});
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::loss(Loss loss_value) {
    loss_ = LossConfig::from_loss(loss_value);
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::loss(LossConfig loss_value) {
    loss_ = std::move(loss_value);
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::optimizer(OptimizerConfig optimizer_value) {
    optimizer_ = optimizer_value;
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::normalization(NormalizationSpec normalization_value) {
    normalization_ = std::move(normalization_value);
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::initialization(InitializationConfig initialization_value) {
    initialization_ = initialization_value;
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::seed(std::uint64_t seed_value) {
    seed_ = seed_value;
    return *this;
}

TrainingModel TrainingModel::Builder::build() const {
    require(input_size_ > 0, "input_size must be positive");
    require(!layers_.empty(), "model must contain at least one layer");
    validate_optimizer(optimizer_);
    validate_loss_config(loss_, layers_.back().neurons);
    validate_normalization_spec(normalization_, input_size_, layers_.back().neurons);
    validate_normalization_for_loss(normalization_, loss_);
    validate_initialization_config(initialization_);

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        require(layers_[i].neurons > 0, "layer neurons must be positive");
        validate_dropout_probability(layers_[i].dropout_probability, "layer dropout_probability");
        if (i + 1 == layers_.size()) {
            require(layers_[i].dropout_probability == Scalar{0}, "output layer dropout must be zero");
        }
        if (layers_[i].activation == Activation::softmax) {
            require(i + 1 == layers_.size(), "softmax is only supported on the output layer");
            require(layers_[i].neurons >= 2, "softmax output layer must contain at least two neurons");
        }
    }

    const Activation output_activation = layers_.back().activation;
    if (loss_.type == Loss::binary_cross_entropy) {
        require(output_activation == Activation::sigmoid,
                "binary_cross_entropy requires a sigmoid output layer");
    }
    if (loss_.type == Loss::categorical_cross_entropy) {
        require(output_activation == Activation::softmax,
                "categorical_cross_entropy requires a softmax output layer");
    }

    auto impl = std::make_unique<Impl>();
    impl->input_size = input_size_;
    impl->loss = loss_;
    impl->optimizer = optimizer_;
    impl->normalization = normalization_;
    impl->seed = seed_;

    std::mt19937_64 rng(seed_);
    std::size_t previous_size = input_size_;
    impl->layers.reserve(layers_.size());

    for (const LayerSpec& spec : layers_) {
        Impl::Layer layer;
        layer.input_size = previous_size;
        layer.output_size = spec.neurons;
        layer.activation = spec.activation;
        layer.dropout_probability = spec.dropout_probability;

        const std::size_t weight_count = layer.input_size * layer.output_size;
        layer.weights.resize(weight_count);
        layer.biases.assign(layer.output_size, initialization_.bias);
        layer.grad_weights.assign(weight_count, Scalar{0});
        layer.grad_biases.assign(layer.output_size, Scalar{0});
        layer.velocity_weights.assign(weight_count, Scalar{0});
        layer.velocity_biases.assign(layer.output_size, Scalar{0});
        layer.first_moment_weights.assign(weight_count, Scalar{0});
        layer.first_moment_biases.assign(layer.output_size, Scalar{0});
        layer.second_moment_weights.assign(weight_count, Scalar{0});
        layer.second_moment_biases.assign(layer.output_size, Scalar{0});

        initialize_weights(layer.weights,
                           resolve_weight_initialization(initialization_.weights, spec.activation),
                           layer.input_size,
                           layer.output_size,
                           rng);

        previous_size = spec.neurons;
        impl->layers.push_back(std::move(layer));
    }

    return TrainingModel(std::move(impl));
}

TrainingModel::Builder TrainingModel::builder() {
    return Builder{};
}

TrainingModel::TrainingModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TrainingModel::TrainingModel(const TrainingModel& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

TrainingModel& TrainingModel::operator=(const TrainingModel& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

TrainingModel::TrainingModel(TrainingModel&& other) noexcept = default;

TrainingModel& TrainingModel::operator=(TrainingModel&& other) noexcept = default;

TrainingModel::~TrainingModel() = default;

std::size_t TrainingModel::input_size() const {
    return impl_->input_size;
}

std::size_t TrainingModel::output_size() const {
    return impl_->output_size();
}

std::size_t TrainingModel::layer_count() const {
    return impl_->layers.size();
}

Loss TrainingModel::loss() const {
    return impl_->loss.type;
}

LossConfig TrainingModel::loss_config() const {
    return impl_->loss;
}

OptimizerConfig TrainingModel::optimizer() const {
    return impl_->optimizer;
}

NormalizationSpec TrainingModel::normalization() const {
    return impl_->normalization;
}

std::size_t TrainingModel::completed_epochs() const {
    return impl_->completed_epochs;
}

Vector TrainingModel::predict(const Vector& input) const {
    require(input.size() == input_size(), "input has an invalid size");
    for (Scalar value : input) {
        require_finite(value, "input value");
    }
    return impl_->predict_raw(input);
}

Matrix TrainingModel::predict_batch(const Matrix& inputs) const {
    validate_matrix(inputs, input_size(), "inputs", true);
    Matrix result;
    result.reserve(inputs.size());
    for (const Vector& input : inputs) {
        result.push_back(impl_->predict_raw(input));
    }
    return result;
}

Matrix TrainingModel::predict_batch(std::initializer_list<Vector> inputs) const {
    return predict_batch(Matrix(inputs));
}

DenseMatrix TrainingModel::predict_batch(const DenseMatrix& inputs) const {
    validate_dense_matrix(inputs, input_size(), "inputs", true);
    DenseMatrix result(inputs.rows(), output_size());
    for (std::size_t row = 0; row < inputs.rows(); ++row) {
        const Vector output = impl_->predict_raw(inputs.row_data(row), inputs.columns());
        std::copy(output.begin(), output.end(), result.row_data(row));
    }
    return result;
}

Scalar TrainingModel::evaluate_loss(const Matrix& inputs, const Matrix& targets) const {
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);
    Scalar total = Scalar{0};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        total += impl_->sample_loss(impl_->predict_raw(inputs[i]), targets[i]);
    }
    return total / static_cast<Scalar>(inputs.size());
}

Scalar TrainingModel::evaluate_loss(std::initializer_list<Vector> inputs,
                                    std::initializer_list<Vector> targets) const {
    return evaluate_loss(Matrix(inputs), Matrix(targets));
}

Scalar TrainingModel::evaluate_loss(const DenseMatrix& inputs, const DenseMatrix& targets) const {
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);
    Scalar total = Scalar{0};
    for (std::size_t row = 0; row < inputs.rows(); ++row) {
        total += impl_->sample_loss(impl_->predict_raw(inputs.row_data(row), inputs.columns()),
                                    targets.row_data(row),
                                    targets.columns());
    }
    return total / static_cast<Scalar>(inputs.rows());
}

EvaluationResult TrainingModel::evaluate_metrics(const Matrix& inputs,
                                                 const Matrix& targets,
                                                 const EvaluationOptions& options) const {
    validate_matrix(inputs, input_size(), "inputs");
    validate_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

EvaluationResult TrainingModel::evaluate_metrics(std::initializer_list<Vector> inputs,
                                                 std::initializer_list<Vector> targets,
                                                 const EvaluationOptions& options) const {
    return evaluate_metrics(Matrix(inputs), Matrix(targets), options);
}

EvaluationResult TrainingModel::evaluate_metrics(const DenseMatrix& inputs,
                                                 const DenseMatrix& targets,
                                                 const EvaluationOptions& options) const {
    validate_dense_matrix(inputs, input_size(), "inputs");
    validate_dense_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

TrainingHistory TrainingModel::fit(const Matrix& inputs, const Matrix& targets, const TrainingOptions& options) {
    MatrixDataset dataset(inputs, targets);
    return fit(dataset, options);
}

TrainingHistory TrainingModel::fit(std::initializer_list<Vector> inputs,
                                   std::initializer_list<Vector> targets,
                                   const TrainingOptions& options) {
    return fit(Matrix(inputs), Matrix(targets), options);
}

TrainingHistory TrainingModel::fit(const DenseMatrix& inputs,
                                   const DenseMatrix& targets,
                                   const TrainingOptions& options) {
    DenseMatrixDataset dataset(inputs, targets);
    return fit(dataset, options);
}

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
    std::vector<LayerParameters> best_parameters;
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
        impl_->optimizer.learning_rate = scheduled_learning_rate;
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
                if (options.restore_best_weights && !best_parameters.empty()) {
                    impl_->set_parameters(best_parameters);
                }
                return finish_history();
            }
            impl_->zero_gradients();
            impl_->accumulate_batch_gradients(normalized_batch, &epoch_rng);
            const auto gradient_result =
                impl_->apply_gradients(batch_count, runtime_options.gradient_noise_stddev, &epoch_rng);
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
            training_loss = impl_->loss_for_dataset_indices(dataset,
                                                            training_indices,
                                                            &runtime_options.debug,
                                                            "training loss",
                                                            epoch,
                                                            true);
        } catch (const TrainingDebugFailure& failure) {
            history.stop_reason = failure.reason;
            history.stop_message = failure.what();
            if (options.restore_best_weights && !best_parameters.empty()) {
                impl_->set_parameters(best_parameters);
            }
            return finish_history();
        }
        history.training_loss.push_back(training_loss);

        bool has_validation_loss = false;
        Scalar validation_loss = Scalar{0};
        if (!split.validation_indices.empty()) {
            try {
                validation_loss = impl_->loss_for_dataset_indices(dataset,
                                                                  split.validation_indices,
                                                                  &runtime_options.debug,
                                                                  "validation loss",
                                                                  epoch,
                                                                  true);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && !best_parameters.empty()) {
                    impl_->set_parameters(best_parameters);
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
                test_loss = impl_->loss_for_dataset_indices(dataset,
                                                            split.test_indices,
                                                            &runtime_options.debug,
                                                            "test loss",
                                                            epoch,
                                                            true);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && !best_parameters.empty()) {
                    impl_->set_parameters(best_parameters);
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
                best_parameters = impl_->parameters();
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
        diagnostics.learning_rate = impl_->optimizer.learning_rate;
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

    if (options.restore_best_weights && !best_parameters.empty()) {
        impl_->set_parameters(best_parameters);
    }

    return finish_history();
}

TrainingHistory TrainingModel::fit(StreamingDataset& dataset, const TrainingOptions& options) {
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
    const DatasetSplit split = make_dataset_split(dataset.sample_count(), split_options);

    std::vector<unsigned char> partition(dataset.sample_count(), 0);
    for (std::size_t index : split.validation_indices) {
        partition[index] = 1;
    }
    for (std::size_t index : split.test_indices) {
        partition[index] = 2;
    }

    TrainingHistory history;
    history.training_loss.reserve(options.epochs);
    history.validation_loss.reserve(options.epochs);
    history.test_loss.reserve(options.epochs);
    history.epochs.reserve(options.epochs);
    history.run_config = impl_->make_training_run_config(run_optimizer,
                                                         runtime_options,
                                                         split,
                                                         dataset.sample_count(),
                                                         true);
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
    std::vector<LayerParameters> best_parameters;
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

    struct BufferedSample {
        Vector input;
        Vector target;
    };

    Vector raw_input(input_size());
    Vector raw_target(output_size());
    Vector normalized_input(input_size());
    Vector normalized_target(output_size());
    TrainingDebugScalarTracker debug_training_loss;
    TrainingDebugScalarTracker debug_validation_loss;
    TrainingDebugScalarTracker debug_test_loss;
    TrainingDebugScalarTracker debug_gradient_norm;

    auto stream_loss_for_partition = [&](unsigned char requested_partition) {
        dataset.reset();
        Scalar total = Scalar{0};
        std::size_t count = 0;
        for (std::size_t index = 0; index < dataset.sample_count(); ++index) {
            TrainingDebugContext context;
            context.phase = requested_partition == 0 ? "streaming training loss"
                          : requested_partition == 1 ? "streaming validation loss"
                                                     : "streaming test loss";
            context.sample_index = index;
            context.has_sample_index = true;
            impl_->read_streaming_sample(dataset,
                                         raw_input,
                                         raw_target,
                                         normalized_input,
                                         normalized_target,
                                         &runtime_options.debug,
                                         context);
            if (partition[index] == requested_partition) {
                total += impl_->sample_loss(impl_->forward(normalized_input), normalized_target);
                ++count;
            }
        }
        require(!dataset.next(raw_input.data(), raw_input.size(), raw_target.data(), raw_target.size()),
                "streaming dataset produced more samples than sample_count");
        require(count > 0, "streaming loss partition is empty");
        return total / static_cast<Scalar>(count);
    };

    for (std::size_t epoch = first_epoch; epoch < options.epochs; ++epoch) {
        const auto epoch_start = std::chrono::steady_clock::now();
        const Scalar scheduled_learning_rate = learning_rate_for_epoch(runtime_options.learning_rate_schedule,
                                                                       learning_rate_state,
                                                                       epoch,
                                                                       options.epochs);
        impl_->optimizer.learning_rate = scheduled_learning_rate;
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
        std::vector<BufferedSample> shuffle_buffer;
        const bool use_shuffle_buffer = options.shuffle && options.streaming_shuffle_buffer_size > 0;
        if (use_shuffle_buffer) {
            shuffle_buffer.reserve(options.streaming_shuffle_buffer_size);
        }

        impl_->zero_gradients();
        std::size_t batch_count = 0;

        auto apply_pending_batch = [&]() {
            const auto gradient_result =
                impl_->apply_gradients(batch_count, runtime_options.gradient_noise_stddev, &epoch_rng);
            if (!is_finite(gradient_result.gradient_norm)) {
                epoch_gradient_norm = gradient_result.gradient_norm;
                non_finite_gradient = true;
                non_finite_gradient_message =
                    "stopped because gradient norm became non-finite at epoch " +
                    std::to_string(epoch) + ": norm=" + scalar_debug_string(gradient_result.gradient_norm);
                if (!gradient_result.non_finite_location.empty()) {
                    non_finite_gradient_message += " first_non_finite=" + gradient_result.non_finite_location;
                }
            } else {
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
                                                "streaming_batch=true");
                    batch_count = 0;
                    return;
                }
                update_debug_tracker_if_finite(debug_gradient_norm, gradient_result.gradient_norm);
                merge_gradient_clipping(epoch_clipping, gradient_result.clipping);
                epoch_gradient_norm = std::max(epoch_gradient_norm, gradient_result.gradient_norm);
            }
            batch_count = 0;
        };

        auto train_normalized_sample = [&](const Vector& input_sample, const Vector& target_sample) {
            if (non_finite_gradient || gradient_explosion) {
                return;
            }
            impl_->accumulate_gradients(input_sample.data(),
                                        input_sample.size(),
                                        target_sample.data(),
                                        target_sample.size(),
                                        &epoch_rng);
            ++batch_count;
            if (batch_count == options.batch_size) {
                apply_pending_batch();
            }
        };

        auto submit_training_sample = [&](const Vector& input_sample, const Vector& target_sample) {
            if (non_finite_gradient || gradient_explosion) {
                return;
            }
            if (!use_shuffle_buffer) {
                train_normalized_sample(input_sample, target_sample);
                return;
            }

            if (shuffle_buffer.size() < options.streaming_shuffle_buffer_size) {
                shuffle_buffer.push_back(BufferedSample{input_sample, target_sample});
                return;
            }

            std::uniform_int_distribution<std::size_t> distribution(0, shuffle_buffer.size() - 1);
            const std::size_t selected = distribution(epoch_rng);
            train_normalized_sample(shuffle_buffer[selected].input, shuffle_buffer[selected].target);
            if (non_finite_gradient || gradient_explosion) {
                return;
            }
            shuffle_buffer[selected].input = input_sample;
            shuffle_buffer[selected].target = target_sample;
        };

        dataset.reset();
        for (std::size_t index = 0; index < dataset.sample_count(); ++index) {
            TrainingDebugContext context;
            context.phase = "streaming training";
            context.epoch = epoch;
            context.has_epoch = true;
            context.sample_index = index;
            context.has_sample_index = true;
            try {
                impl_->read_streaming_sample(dataset,
                                             raw_input,
                                             raw_target,
                                             normalized_input,
                                             normalized_target,
                                             &runtime_options.debug,
                                             context);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && !best_parameters.empty()) {
                    impl_->set_parameters(best_parameters);
                }
                return finish_history();
            }
            if (partition[index] == 0) {
                submit_training_sample(normalized_input, normalized_target);
                if (non_finite_gradient || gradient_explosion) {
                    break;
                }
            }
        }
        if (!non_finite_gradient && !gradient_explosion) {
            require(!dataset.next(raw_input.data(), raw_input.size(), raw_target.data(), raw_target.size()),
                    "streaming dataset produced more samples than sample_count");
        }

        if (!non_finite_gradient && !gradient_explosion && use_shuffle_buffer) {
            std::shuffle(shuffle_buffer.begin(), shuffle_buffer.end(), epoch_rng);
            for (const BufferedSample& sample : shuffle_buffer) {
                train_normalized_sample(sample.input, sample.target);
                if (non_finite_gradient || gradient_explosion) {
                    break;
                }
            }
        }
        if (!non_finite_gradient && !gradient_explosion && batch_count > 0) {
            apply_pending_batch();
        }
        finalize_gradient_clipping(epoch_clipping);

        Scalar training_loss = Scalar{0};
        try {
            training_loss = stream_loss_for_partition(0);
        } catch (const TrainingDebugFailure& failure) {
            history.stop_reason = failure.reason;
            history.stop_message = failure.what();
            if (options.restore_best_weights && !best_parameters.empty()) {
                impl_->set_parameters(best_parameters);
            }
            return finish_history();
        }
        history.training_loss.push_back(training_loss);

        bool has_validation_loss = false;
        Scalar validation_loss = Scalar{0};
        if (!split.validation_indices.empty()) {
            try {
                validation_loss = stream_loss_for_partition(1);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && !best_parameters.empty()) {
                    impl_->set_parameters(best_parameters);
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
                test_loss = stream_loss_for_partition(2);
            } catch (const TrainingDebugFailure& failure) {
                history.stop_reason = failure.reason;
                history.stop_message = failure.what();
                if (options.restore_best_weights && !best_parameters.empty()) {
                    impl_->set_parameters(best_parameters);
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
                best_parameters = impl_->parameters();
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
        diagnostics.learning_rate = impl_->optimizer.learning_rate;
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

    if (options.restore_best_weights && !best_parameters.empty()) {
        impl_->set_parameters(best_parameters);
    }

    return finish_history();
}

std::vector<LayerParameters> TrainingModel::parameters() const {
    return impl_->parameters();
}

void TrainingModel::set_parameters(const std::vector<LayerParameters>& parameters) {
    impl_->set_parameters(parameters);
}

CompiledModel TrainingModel::compile() const {
    return CompiledModel(to_inference_model());
}

InferenceModel TrainingModel::to_inference_model() const {
    return InferenceModel::from_parameters(impl_->parameters(), impl_->normalization);
}

void TrainingModel::save(const std::filesystem::path& path) const {
    if (impl_->loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be serialized");
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open model file for writing: " + path.string());
    }

    output << std::setprecision(17);
    output << "BRUTAL_MLP_TRAINING_V5\n";
    output << impl_->input_size << '\n';
    write_loss_config(output, impl_->loss);
    output << to_string(impl_->optimizer.type) << ' '
           << impl_->optimizer.learning_rate << ' '
           << impl_->optimizer.beta1 << ' '
           << impl_->optimizer.beta2 << ' '
           << impl_->optimizer.epsilon << ' '
           << impl_->optimizer.momentum << ' '
           << impl_->optimizer.l2 << ' '
           << impl_->optimizer.l1 << ' '
           << impl_->optimizer.decoupled_weight_decay << ' '
           << impl_->optimizer.max_norm << ' '
           << impl_->optimizer.gradient_clip_norm << ' '
           << impl_->optimizer.gradient_clip_value << ' '
           << impl_->optimizer.layer_gradient_clip_norm << '\n';
    output << impl_->seed << '\n';
    write_normalization(output, impl_->normalization);
    output << impl_->layers.size() << '\n';

    for (const Impl::Layer& layer : impl_->layers) {
        output << layer.input_size << ' ' << layer.output_size << ' ' << to_string(layer.activation) << ' '
               << layer.dropout_probability << '\n';
        output << layer.weights.size();
        for (Scalar weight : layer.weights) {
            output << ' ' << weight;
        }
        output << '\n';
        output << layer.biases.size();
        for (Scalar bias : layer.biases) {
            output << ' ' << bias;
        }
        output << '\n';
    }

    if (!output) {
        throw std::runtime_error("failed while writing model file: " + path.string());
    }
}

void TrainingModel::save_binary(const std::filesystem::path& path, const BinaryMetadata& metadata) const {
    if (impl_->loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be serialized");
    }

    const std::vector<LayerParameters> model_parameters = impl_->parameters();
    BinaryWriter payload;
    write_binary_metadata(payload, metadata);
    payload.write_size(impl_->input_size);
    payload.write_size(impl_->output_size());
    write_binary_loss_config(payload, impl_->loss);
    write_binary_optimizer(payload, impl_->optimizer);
    payload.write_u64(impl_->seed);
    payload.write_u64(impl_->optimizer_step);
    payload.write_bool(impl_->has_last_training_options);
    if (impl_->has_last_training_options) {
        write_binary_training_options(payload, impl_->last_training_options);
    }
    write_binary_normalization(payload, impl_->normalization);
    payload.write_size(model_parameters.size());
    for (const LayerParameters& layer : model_parameters) {
        write_binary_layer(payload, layer);
    }

    write_binary_file(path, build_binary_envelope(BinaryModelKind::training, payload.bytes));
}

void TrainingModel::save_checkpoint(const std::filesystem::path& path,
                                    const TrainingCheckpointInfo& info) const {
    if (impl_->loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be serialized");
    }

    TrainingCheckpointInfo checkpoint = info;
    if (checkpoint.completed_epochs == 0 && impl_->completed_epochs > 0) {
        checkpoint.completed_epochs = impl_->completed_epochs;
    }
    if (!impl_->has_last_training_options &&
        checkpoint.training_options.epochs == TrainingOptions{}.epochs &&
        checkpoint.training_options.batch_size == TrainingOptions{}.batch_size) {
        checkpoint.training_options = TrainingOptions{};
    } else if (impl_->has_last_training_options &&
               checkpoint.training_options.epochs == TrainingOptions{}.epochs &&
               checkpoint.training_options.batch_size == TrainingOptions{}.batch_size &&
               checkpoint.training_options.best_checkpoint_path.empty() &&
               checkpoint.training_options.latest_checkpoint_path.empty()) {
        checkpoint.training_options = impl_->last_training_options;
    }
    if (!checkpoint.learning_rate_state.initialized && impl_->learning_rate_state.initialized) {
        checkpoint.learning_rate_state = impl_->learning_rate_state;
    }

    const std::vector<LayerParameters> model_parameters = impl_->parameters();
    const std::vector<LayerTrainingState> optimizer_states = impl_->training_states();

    BinaryWriter payload;
    write_binary_metadata(payload, checkpoint.metadata);
    payload.write_size(impl_->input_size);
    payload.write_size(impl_->output_size());
    write_binary_loss_config(payload, impl_->loss);
    write_binary_optimizer(payload, impl_->optimizer);
    payload.write_u64(impl_->seed);
    payload.write_u64(impl_->optimizer_step);
    payload.write_size(checkpoint.completed_epochs);
    payload.write_bool(checkpoint.best_checkpoint);
    payload.write_bool(checkpoint.has_metric);
    if (checkpoint.has_metric) {
        payload.write_scalar(checkpoint.metric);
    }
    payload.write_bool(true);
    write_binary_learning_rate_scheduler_state(payload, checkpoint.learning_rate_state);
    payload.write_bool(true);
    write_binary_training_options(payload, checkpoint.training_options);
    write_binary_normalization(payload, impl_->normalization);
    payload.write_size(model_parameters.size());
    for (std::size_t i = 0; i < model_parameters.size(); ++i) {
        write_binary_layer(payload, model_parameters[i]);
        write_binary_layer_training_state(payload, optimizer_states[i]);
    }

    write_binary_file(path, build_binary_envelope(BinaryModelKind::checkpoint, payload.bytes));
}

TrainingModel TrainingModel::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open model file for reading: " + path.string());
    }

    const std::string magic = read_token(input, "magic");
    const bool has_optimizer_clipping = magic == "BRUTAL_MLP_TRAINING_V5";
    const bool has_optimizer_regularization = has_optimizer_clipping || magic == "BRUTAL_MLP_TRAINING_V4";
    const bool has_loss_config = has_optimizer_regularization || magic == "BRUTAL_MLP_TRAINING_V3";
    const bool has_normalization = has_loss_config || magic == "BRUTAL_MLP_TRAINING_V2";
    if (!has_normalization && magic != "BRUTAL_MLP_TRAINING_V1" && magic != "BRUTAL_MLP_V1") {
        throw std::runtime_error("unsupported model file format");
    }

    const std::size_t input_size = read_value<std::size_t>(input, "input_size");
    require(input_size > 0, "training input_size must be positive");
    const LossConfig loss = has_loss_config
                                ? read_loss_config(input)
                                : LossConfig::from_loss(loss_from_string(read_token(input, "loss")));

    OptimizerConfig optimizer;
    optimizer.type = optimizer_type_from_string(read_token(input, "optimizer"));
    optimizer.learning_rate = read_value<Scalar>(input, "learning_rate");
    optimizer.beta1 = read_value<Scalar>(input, "beta1");
    optimizer.beta2 = read_value<Scalar>(input, "beta2");
    optimizer.epsilon = read_value<Scalar>(input, "epsilon");
    optimizer.momentum = read_value<Scalar>(input, "momentum");
    optimizer.l2 = read_value<Scalar>(input, "l2");
    if (has_optimizer_regularization) {
        optimizer.l1 = read_value<Scalar>(input, "l1");
        optimizer.decoupled_weight_decay = read_value<Scalar>(input, "decoupled_weight_decay");
        optimizer.max_norm = read_value<Scalar>(input, "max_norm");
    }
    optimizer.gradient_clip_norm = read_value<Scalar>(input, "gradient_clip_norm");
    if (has_optimizer_clipping) {
        optimizer.gradient_clip_value = read_value<Scalar>(input, "gradient_clip_value");
        optimizer.layer_gradient_clip_norm = read_value<Scalar>(input, "layer_gradient_clip_norm");
    }

    const std::uint64_t seed = read_value<std::uint64_t>(input, "seed");
    NormalizationSpec normalization;
    if (has_normalization) {
        normalization = read_normalization(input);
    }
    const std::size_t layer_count = read_value<std::size_t>(input, "layer_count");
    validate_serialized_layer_count(layer_count, "training model");

    Builder builder = TrainingModel::builder()
                          .input_size(input_size)
                          .loss(loss)
                          .optimizer(optimizer)
                          .normalization(normalization)
                          .seed(seed);
    std::vector<LayerParameters> parameters;
    parameters.reserve(layer_count);

    std::size_t expected_input_size = input_size;
    for (std::size_t i = 0; i < layer_count; ++i) {
        LayerParameters layer;
        layer.input_size = read_value<std::size_t>(input, "layer_input_size");
        layer.output_size = read_value<std::size_t>(input, "layer_output_size");
        layer.activation = activation_from_string(read_token(input, "layer_activation"));
        if (has_optimizer_regularization) {
            layer.dropout_probability = read_value<Scalar>(input, "layer_dropout_probability");
        }
        require(layer.input_size > 0, "training layer input_size must be positive");
        require(layer.output_size > 0, "training layer output_size must be positive");
        require(layer.input_size == expected_input_size, "training file topology is inconsistent");
        const std::size_t expected_weight_count =
            checked_product(layer.input_size, layer.output_size, "training weight");
        builder.add_layer(layer.output_size, layer.activation, layer.dropout_probability);

        const std::size_t weight_count = read_value<std::size_t>(input, "weight_count");
        require_exact_serialized_count(weight_count, expected_weight_count, "training weight");
        layer.weights.resize(weight_count);
        for (Scalar& weight : layer.weights) {
            weight = read_value<Scalar>(input, "weight");
        }

        const std::size_t bias_count = read_value<std::size_t>(input, "bias_count");
        require_exact_serialized_count(bias_count, layer.output_size, "training bias");
        layer.biases.resize(bias_count);
        for (Scalar& bias : layer.biases) {
            bias = read_value<Scalar>(input, "bias");
        }
        expected_input_size = layer.output_size;
        parameters.push_back(std::move(layer));
    }

    require_text_stream_consumed(input, "training model");
    validate_layer_parameters(parameters);
    validate_loss_config(loss, parameters.back().output_size);
    validate_normalization_spec(normalization, input_size, parameters.back().output_size);
    validate_normalization_for_loss(normalization, loss);
    TrainingModel model = builder.build();
    model.set_parameters(parameters);
    return model;
}

TrainingModel TrainingModel::load_binary(const std::filesystem::path& path) {
    ParsedBinaryModel parsed = parse_binary_model_file(path);
    if (parsed.info.model_kind != BinaryModelKind::training) {
        throw std::runtime_error("binary model is not a training model");
    }

    Builder builder = TrainingModel::builder()
                          .input_size(parsed.info.input_size)
                          .loss(parsed.loss)
                          .optimizer(parsed.optimizer)
                          .normalization(parsed.normalization)
                          .seed(parsed.info.seed);
    for (const LayerParameters& layer : parsed.parameters) {
        builder.add_layer(layer.output_size, layer.activation, layer.dropout_probability);
    }

    TrainingModel model = builder.build();
    model.set_parameters(parsed.parameters);
    model.impl_->optimizer_step = parsed.optimizer_step;
    model.impl_->has_last_training_options = parsed.info.has_training_options;
    model.impl_->last_training_options = parsed.info.training_options;
    return model;
}

TrainingModel TrainingModel::load_checkpoint(const std::filesystem::path& path,
                                             TrainingCheckpointInfo* info) {
    ParsedBinaryModel parsed = parse_binary_model_file(path);
    if (parsed.info.model_kind != BinaryModelKind::checkpoint) {
        throw std::runtime_error("binary file is not a training checkpoint");
    }

    Builder builder = TrainingModel::builder()
                          .input_size(parsed.info.input_size)
                          .loss(parsed.loss)
                          .optimizer(parsed.optimizer)
                          .normalization(parsed.normalization)
                          .seed(parsed.info.seed);
    for (const LayerParameters& layer : parsed.parameters) {
        builder.add_layer(layer.output_size, layer.activation, layer.dropout_probability);
    }

    TrainingModel model = builder.build();
    model.set_parameters(parsed.parameters);
    model.impl_->set_training_states(parsed.training_states, parsed.optimizer_step);
    model.impl_->completed_epochs = parsed.info.completed_epochs;
    model.impl_->resume_epoch = parsed.info.completed_epochs;
    model.impl_->has_resume_epoch = true;
    model.impl_->learning_rate_state = parsed.info.learning_rate_state;
    model.impl_->has_last_training_options = parsed.info.has_training_options;
    model.impl_->last_training_options = parsed.info.training_options;

    if (info != nullptr) {
        info->completed_epochs = parsed.info.completed_epochs;
        info->best_checkpoint = parsed.info.checkpoint_best;
        info->has_metric = parsed.info.has_checkpoint_metric;
        info->metric = parsed.info.checkpoint_metric;
        info->learning_rate_state = parsed.info.learning_rate_state;
        info->training_options = parsed.info.training_options;
        info->metadata = parsed.info.metadata;
    }

    return model;
}

} // namespace brutal_mlp

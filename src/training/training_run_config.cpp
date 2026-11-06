// TrainingRunConfig JSON export and run summary refresh.

#include "training/training_run_config.hpp"

#include "core/checks.hpp"

#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace brutal_mlp::detail {

namespace {

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

} // namespace

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

} // namespace brutal_mlp::detail

namespace brutal_mlp {

using namespace detail;

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

} // namespace brutal_mlp

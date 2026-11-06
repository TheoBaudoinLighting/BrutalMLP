// Binary model envelope, sections and payload parsing.

#include "serialization/binary_model_format.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "core/time_utils.hpp"
#include "inference/layer_validation.hpp"
#include "normalization/normalization.hpp"
#include "serialization/binary_io.hpp"
#include "serialization/enum_codes.hpp"
#include "serialization/serialization_limits.hpp"
#include "training/layer_training_state.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brutal_mlp::detail {

namespace {

constexpr unsigned char kBinaryMagic[8] = {'B', 'M', 'L', 'P', 'B', 'I', 'N', '\0'};

constexpr std::uint32_t kBinaryVersion = 7;

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

[[nodiscard]] NormalizationSpec read_binary_normalization(BinaryReader& reader,
                                                          std::size_t input_size,
                                                          std::size_t output_size) {
    NormalizationSpec normalization;
    normalization.input_features = read_binary_normalization_features(reader, "input_normalization", input_size);
    normalization.output_features = read_binary_normalization_features(reader, "output_normalization", output_size);
    validate_normalization_spec(normalization, input_size, output_size);
    return normalization;
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

struct BinaryEnvelope {
    std::uint32_t version{0};
    BinaryScalarType scalar_type{BinaryScalarType::float32};
    BinaryModelKind kind{BinaryModelKind::training};
    std::vector<unsigned char> payload;
    std::uint32_t checksum{0};
};

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

} // namespace

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

void write_binary_loss_config(BinaryWriter& writer, const LossConfig& loss) {
    if (loss.type == Loss::custom) {
        throw std::runtime_error("custom loss cannot be serialized");
    }
    writer.write_u32(loss_code(loss.type));
    writer.write_scalar(loss.huber_delta);
    writer.write_scalar(loss.relative_epsilon);
    writer.write_scalars(loss.weights);
}

void write_binary_normalization(BinaryWriter& writer, const NormalizationSpec& normalization) {
    write_binary_normalization_features(writer, normalization.input_features);
    write_binary_normalization_features(writer, normalization.output_features);
}

void write_binary_layer(BinaryWriter& writer, const LayerParameters& layer) {
    writer.write_size(layer.input_size);
    writer.write_size(layer.output_size);
    writer.write_u32(activation_code(layer.activation));
    writer.write_scalar(layer.dropout_probability);
    writer.write_scalars(layer.weights);
    writer.write_scalars(layer.biases);
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

} // namespace brutal_mlp::detail

namespace brutal_mlp {

using namespace detail;

BinaryModelInfo inspect_binary_model(const std::filesystem::path& path) {
    return parse_binary_model_file(path).info;
}

} // namespace brutal_mlp

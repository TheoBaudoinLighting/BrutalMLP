// TrainingModel text/binary persistence and checkpoints.

#include "brutal_mlp/brutal_mlp.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "inference/layer_validation.hpp"
#include "normalization/normalization.hpp"
#include "serialization/binary_io.hpp"
#include "serialization/binary_model_format.hpp"
#include "serialization/serialization_limits.hpp"
#include "serialization/text_format.hpp"
#include "training/layer_training_state.hpp"
#include "training/training_model_impl.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brutal_mlp {

using namespace detail;

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

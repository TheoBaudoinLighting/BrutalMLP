// InferenceModel text/binary persistence.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"
#include "inference/inference_model_impl.hpp"
#include "inference/layer_validation.hpp"
#include "normalization/normalization.hpp"
#include "serialization/binary_io.hpp"
#include "serialization/binary_model_format.hpp"
#include "serialization/serialization_limits.hpp"
#include "serialization/text_format.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brutal_mlp {

using namespace detail;

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

} // namespace brutal_mlp

// Text model format readers and writers.

#include "serialization/text_format.hpp"

#include "normalization/normalization.hpp"
#include "serialization/serialization_limits.hpp"

#include <cstddef>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brutal_mlp::detail {

namespace {

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

} // namespace

[[nodiscard]] std::string read_token(std::istream& stream, const std::string& name) {
    std::string token;
    if (!(stream >> token)) {
        throw std::runtime_error("failed to read " + name);
    }
    return token;
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

void write_normalization(std::ostream& output, const NormalizationSpec& normalization) {
    output << "NORMALIZATION_V1\n";
    write_normalization_features(output, normalization.input_features);
    write_normalization_features(output, normalization.output_features);
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

} // namespace brutal_mlp::detail

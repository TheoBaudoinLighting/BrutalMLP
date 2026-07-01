#include "brutal_mlp/brutal_mlp.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace brutal_mlp {
namespace {

#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
constexpr Scalar kProbabilityEpsilon = Scalar{1e-12};
constexpr Scalar kTargetSumTolerance = Scalar{1e-8};
#else
constexpr Scalar kProbabilityEpsilon = Scalar{1e-7f};
constexpr Scalar kTargetSumTolerance = Scalar{1e-4f};
#endif

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
    require_finite(optimizer.gradient_clip_norm, "gradient_clip_norm");
    require(optimizer.gradient_clip_norm >= Scalar{0}, "gradient_clip_norm must be non-negative");
}

void validate_training_options(const TrainingOptions& options) {
    require(options.epochs > 0, "epochs must be positive");
    require(options.batch_size > 0, "batch_size must be positive");
    require_finite(options.validation_split, "validation_split");
    require(options.validation_split >= Scalar{0} && options.validation_split < Scalar{1},
            "validation_split must be in [0, 1)");
    require_finite(options.min_delta, "min_delta");
    require(options.min_delta >= Scalar{0}, "min_delta must be non-negative");
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
        for (Scalar value : matrix[row]) {
            require_finite(value, name + " value");
        }
    }
}

void validate_targets_for_loss(const Matrix& targets, Loss loss) {
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
        for (const Vector& target : targets) {
            for (Scalar value : target) {
                require(value >= Scalar{0} && value <= Scalar{1},
                        "binary_cross_entropy targets must be in [0, 1]");
            }
        }
        return;
    case Loss::categorical_cross_entropy:
        for (const Vector& target : targets) {
            Scalar sum = Scalar{0};
            for (Scalar value : target) {
                require(value >= Scalar{0}, "categorical_cross_entropy targets must be non-negative");
                sum += value;
            }
            require(std::abs(sum - Scalar{1}) <= kTargetSumTolerance,
                    "categorical_cross_entropy targets must sum to 1");
        }
        return;
    }

    throw std::invalid_argument("unknown loss");
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
        require(layer.weights.size() == layer.input_size * layer.output_size,
                "parameter weights size mismatch");
        require(layer.biases.size() == layer.output_size, "parameter biases size mismatch");
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
    loss.weights.resize(weight_count);
    for (Scalar& weight : loss.weights) {
        weight = read_value<Scalar>(input, "loss_weight");
    }
    return loss;
}

} // namespace

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

std::string to_string(OptimizerType optimizer) {
    switch (optimizer) {
    case OptimizerType::sgd:
        return "sgd";
    case OptimizerType::adam:
        return "adam";
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

OptimizerType optimizer_type_from_string(std::string_view value) {
    if (value == "sgd") {
        return OptimizerType::sgd;
    }
    if (value == "adam") {
        return OptimizerType::adam;
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

OptimizerConfig OptimizerConfig::adam(Scalar learning_rate) {
    OptimizerConfig config;
    config.type = OptimizerType::adam;
    config.learning_rate = learning_rate;
    return config;
}

OptimizerConfig OptimizerConfig::sgd(Scalar learning_rate, Scalar momentum) {
    OptimizerConfig config;
    config.type = OptimizerType::sgd;
    config.learning_rate = learning_rate;
    config.momentum = momentum;
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
};

InferenceModel::InferenceModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

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

    Scalar* scratch_cursor = scratch;
    const Scalar* previous = input;
    if (impl_->normalization.has_input_normalization()) {
        for (std::size_t i = 0; i < impl_->input_size; ++i) {
            scratch_cursor[i] = normalize_value(input[i], impl_->normalization.input_features[i]);
        }
        previous = scratch_cursor;
        scratch_cursor += impl_->input_scratch_size;
    }

    Scalar* scratch_a = scratch_cursor;
    Scalar* scratch_b = scratch ? scratch + impl_->scratch_stride : nullptr;
    if (scratch_b && impl_->input_scratch_size > 0) {
        scratch_b += impl_->input_scratch_size;
    }
    bool write_a = true;

    for (std::size_t layer_index = 0; layer_index < impl_->layers.size(); ++layer_index) {
        const Impl::Layer& layer = impl_->layers[layer_index];
        const bool output_layer = layer_index + 1 == impl_->layers.size();
        Scalar* target = output_layer ? output : (write_a ? scratch_a : scratch_b);

        for (std::size_t out = 0; out < layer.output_size; ++out) {
            Scalar value = impl_->biases[layer.bias_offset + out];
            const std::size_t weight_offset = layer.weight_offset + out * layer.input_size;
            for (std::size_t in = 0; in < layer.input_size; ++in) {
                value += impl_->weights[weight_offset + in] * previous[in];
            }
            target[out] = value;
        }

        apply_activation_in_place(target, layer.output_size, layer.activation);
        previous = target;
        write_a = !write_a;
    }

    if (impl_->normalization.has_output_normalization()) {
        for (std::size_t i = 0; i < impl_->output_size; ++i) {
            output[i] = denormalize_value(output[i], impl_->normalization.output_features[i]);
        }
    }

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

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const InferenceStatus status = predict_to(inputs + sample * input_stride,
                                                  impl_->input_size,
                                                  outputs + sample * output_stride,
                                                  impl_->output_size,
                                                  scratch,
                                                  provided_scratch_size);
        if (status != InferenceStatus::ok) {
            return status;
        }
    }

    return InferenceStatus::ok;
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
    NormalizationSpec normalization;
    if (has_normalization) {
        normalization = read_normalization(input);
    }
    const std::size_t layer_count = read_value<std::size_t>(input, "layer_count");
    std::vector<LayerParameters> parameters;
    parameters.reserve(layer_count);

    std::size_t expected_input_size = input_size;
    for (std::size_t i = 0; i < layer_count; ++i) {
        LayerParameters layer;
        layer.input_size = read_value<std::size_t>(input, "layer_input_size");
        layer.output_size = read_value<std::size_t>(input, "layer_output_size");
        layer.activation = activation_from_string(read_token(input, "layer_activation"));
        require(layer.input_size == expected_input_size, "inference file topology is inconsistent");

        const std::size_t weight_count = read_value<std::size_t>(input, "weight_count");
        layer.weights.resize(weight_count);
        for (Scalar& weight : layer.weights) {
            weight = read_value<Scalar>(input, "weight");
        }

        const std::size_t bias_count = read_value<std::size_t>(input, "bias_count");
        layer.biases.resize(bias_count);
        for (Scalar& bias : layer.biases) {
            bias = read_value<Scalar>(input, "bias");
        }

        expected_input_size = layer.output_size;
        parameters.push_back(std::move(layer));
    }

    return InferenceModel::from_parameters(parameters, normalization);
}

struct TrainingModel::Impl {
    struct Layer {
        std::size_t input_size{0};
        std::size_t output_size{0};
        Activation activation{Activation::linear};
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
    std::vector<Layer> layers;

    [[nodiscard]] std::size_t output_size() const {
        return layers.back().output_size;
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

    [[nodiscard]] Vector forward_cached(const Vector& input,
                                        std::vector<Vector>& activations,
                                        std::vector<Vector>& pre_activations) const {
        activations.clear();
        pre_activations.clear();
        activations.reserve(layers.size() + 1);
        pre_activations.reserve(layers.size());
        activations.push_back(input);

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

    [[nodiscard]] Scalar sample_loss(const Vector& prediction, const Vector& target) const {
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
                                                        target.data(),
                                                        prediction.size(),
                                                        loss.custom_loss.context);
            require_finite(value, "custom loss");
            return value;
        }
        }

        throw std::invalid_argument("unknown loss");
    }

    [[nodiscard]] Vector output_delta(const Vector& prediction,
                                      const Vector& target,
                                      const Vector& output_pre_activation) const {
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
                                      target.data(),
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

    void accumulate_gradients(const Vector& input, const Vector& target) {
        std::vector<Vector> activations;
        std::vector<Vector> pre_activations;
        const Vector prediction = forward_cached(input, activations, pre_activations);
        Vector delta = output_delta(prediction, target, pre_activations.back());

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
                previous_delta[in] =
                    value * activation_derivative(previous_layer.activation,
                                                  pre_activations[layer_offset - 1][in],
                                                  activations[layer_offset][in]);
            }
            delta = std::move(previous_delta);
        }
    }

    void zero_gradients() {
        for (Layer& layer : layers) {
            std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), Scalar{0});
            std::fill(layer.grad_biases.begin(), layer.grad_biases.end(), Scalar{0});
        }
    }

    void apply_gradients(std::size_t batch_size) {
        ++optimizer_step;
        const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
        Scalar norm_squared = Scalar{0};

        for (const Layer& layer : layers) {
            for (std::size_t i = 0; i < layer.weights.size(); ++i) {
                const Scalar gradient = layer.grad_weights[i] * batch_scale + optimizer.l2 * layer.weights[i];
                norm_squared += gradient * gradient;
            }
            for (Scalar gradient_value : layer.grad_biases) {
                const Scalar gradient = gradient_value * batch_scale;
                norm_squared += gradient * gradient;
            }
        }

        Scalar clip_scale = Scalar{1};
        if (optimizer.gradient_clip_norm > Scalar{0}) {
            const Scalar norm = static_cast<Scalar>(std::sqrt(norm_squared));
            if (norm > optimizer.gradient_clip_norm) {
                clip_scale = optimizer.gradient_clip_norm / (norm + std::numeric_limits<Scalar>::epsilon());
            }
        }

        for (Layer& layer : layers) {
            apply_vector_gradients(layer.weights,
                                   layer.grad_weights,
                                   layer.velocity_weights,
                                   layer.first_moment_weights,
                                   layer.second_moment_weights,
                                   batch_scale,
                                   clip_scale,
                                   true);
            apply_vector_gradients(layer.biases,
                                   layer.grad_biases,
                                   layer.velocity_biases,
                                   layer.first_moment_biases,
                                   layer.second_moment_biases,
                                   batch_scale,
                                   clip_scale,
                                   false);
        }

        zero_gradients();
    }

    void apply_vector_gradients(Vector& values,
                                const Vector& accumulated_gradients,
                                Vector& velocities,
                                Vector& first_moments,
                                Vector& second_moments,
                                Scalar batch_scale,
                                Scalar clip_scale,
                                bool regularize) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            Scalar gradient = accumulated_gradients[i] * batch_scale;
            if (regularize) {
                gradient += optimizer.l2 * values[i];
            }
            gradient *= clip_scale;

            switch (optimizer.type) {
            case OptimizerType::sgd:
                if (optimizer.momentum > Scalar{0}) {
                    velocities[i] = optimizer.momentum * velocities[i] - optimizer.learning_rate * gradient;
                    values[i] += velocities[i];
                } else {
                    values[i] -= optimizer.learning_rate * gradient;
                }
                break;
            case OptimizerType::adam: {
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

    [[nodiscard]] Scalar loss_for_indices(const Matrix& inputs,
                                          const Matrix& targets,
                                          const std::vector<std::size_t>& indices) const {
        Scalar total = Scalar{0};
        for (std::size_t index : indices) {
            total += sample_loss(forward(inputs[index]), targets[index]);
        }
        return total / static_cast<Scalar>(indices.size());
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
            for (Scalar value : source.weights) {
                require_finite(value, "weight");
            }
            for (Scalar value : source.biases) {
                require_finite(value, "bias");
            }
            layer.weights = source.weights;
            layer.biases = source.biases;
            reset_optimizer_state(layer);
        }
        optimizer_step = 0;
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

TrainingModel::Builder& TrainingModel::Builder::add_layer(std::size_t neurons, Activation activation) {
    layers_.push_back(LayerSpec{neurons, activation});
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

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        require(layers_[i].neurons > 0, "layer neurons must be positive");
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

        const std::size_t weight_count = layer.input_size * layer.output_size;
        layer.weights.resize(weight_count);
        layer.biases.assign(layer.output_size, Scalar{0});
        layer.grad_weights.assign(weight_count, Scalar{0});
        layer.grad_biases.assign(layer.output_size, Scalar{0});
        layer.velocity_weights.assign(weight_count, Scalar{0});
        layer.velocity_biases.assign(layer.output_size, Scalar{0});
        layer.first_moment_weights.assign(weight_count, Scalar{0});
        layer.first_moment_biases.assign(layer.output_size, Scalar{0});
        layer.second_moment_weights.assign(weight_count, Scalar{0});
        layer.second_moment_biases.assign(layer.output_size, Scalar{0});

        const Scalar limit = spec.activation == Activation::relu
                                 ? static_cast<Scalar>(std::sqrt(Scalar{6} /
                                                                 static_cast<Scalar>(layer.input_size)))
                                 : static_cast<Scalar>(std::sqrt(
                                       Scalar{6} / static_cast<Scalar>(layer.input_size + layer.output_size)));
        std::uniform_real_distribution<Scalar> distribution(-limit, limit);
        for (Scalar& weight : layer.weights) {
            weight = distribution(rng);
        }

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

Scalar TrainingModel::evaluate_loss(const Matrix& inputs, const Matrix& targets) const {
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);
    Scalar total = Scalar{0};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        total += impl_->sample_loss(impl_->predict_raw(inputs[i]), targets[i]);
    }
    return total / static_cast<Scalar>(inputs.size());
}

TrainingHistory TrainingModel::fit(const Matrix& inputs, const Matrix& targets, const TrainingOptions& options) {
    validate_training_options(options);
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);

    const Matrix normalized_inputs = normalize_matrix(inputs, impl_->normalization.input_features);
    const Matrix normalized_targets = normalize_matrix(targets, impl_->normalization.output_features);

    std::vector<std::size_t> indices = make_indices(inputs.size());
    std::mt19937_64 split_rng(non_zero_seed(options.seed, impl_->seed));
    if (options.validation_split > Scalar{0}) {
        std::shuffle(indices.begin(), indices.end(), split_rng);
    }

    const auto validation_count =
        static_cast<std::size_t>(std::floor(static_cast<Scalar>(inputs.size()) * options.validation_split));
    require(validation_count < inputs.size(), "validation_split leaves no training samples");

    std::vector<std::size_t> validation_indices(indices.end() - static_cast<std::ptrdiff_t>(validation_count),
                                                indices.end());
    std::vector<std::size_t> training_indices(indices.begin(),
                                              indices.end() - static_cast<std::ptrdiff_t>(validation_count));

    TrainingHistory history;
    history.training_loss.reserve(options.epochs);
    history.validation_loss.reserve(options.epochs);

    Scalar best_metric = std::numeric_limits<Scalar>::infinity();
    std::vector<LayerParameters> best_parameters;
    std::size_t stale_epochs = 0;

    for (std::size_t epoch = 0; epoch < options.epochs; ++epoch) {
        std::mt19937_64 epoch_rng(non_zero_seed(options.seed, impl_->seed) + epoch);
        if (options.shuffle) {
            std::shuffle(training_indices.begin(), training_indices.end(), epoch_rng);
        }

        for (std::size_t begin = 0; begin < training_indices.size(); begin += options.batch_size) {
            const std::size_t end = std::min(begin + options.batch_size, training_indices.size());
            impl_->zero_gradients();
            for (std::size_t position = begin; position < end; ++position) {
                const std::size_t row = training_indices[position];
                impl_->accumulate_gradients(normalized_inputs[row], normalized_targets[row]);
            }
            impl_->apply_gradients(end - begin);
        }

        const Scalar training_loss = impl_->loss_for_indices(normalized_inputs, normalized_targets, training_indices);
        history.training_loss.push_back(training_loss);

        Scalar monitored_metric = training_loss;
        if (!validation_indices.empty()) {
            const Scalar validation_loss = impl_->loss_for_indices(normalized_inputs, normalized_targets, validation_indices);
            history.validation_loss.push_back(validation_loss);
            monitored_metric = validation_loss;
        }

        if (monitored_metric + options.min_delta < best_metric) {
            best_metric = monitored_metric;
            stale_epochs = 0;
            if (options.restore_best_weights) {
                best_parameters = impl_->parameters();
            }
        } else {
            ++stale_epochs;
        }

        if (options.early_stopping_patience > 0 && stale_epochs >= options.early_stopping_patience) {
            break;
        }
    }

    if (options.restore_best_weights && !best_parameters.empty()) {
        impl_->set_parameters(best_parameters);
    }

    return history;
}

std::vector<LayerParameters> TrainingModel::parameters() const {
    return impl_->parameters();
}

void TrainingModel::set_parameters(const std::vector<LayerParameters>& parameters) {
    impl_->set_parameters(parameters);
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
    output << "BRUTAL_MLP_TRAINING_V3\n";
    output << impl_->input_size << '\n';
    write_loss_config(output, impl_->loss);
    output << to_string(impl_->optimizer.type) << ' '
           << impl_->optimizer.learning_rate << ' '
           << impl_->optimizer.beta1 << ' '
           << impl_->optimizer.beta2 << ' '
           << impl_->optimizer.epsilon << ' '
           << impl_->optimizer.momentum << ' '
           << impl_->optimizer.l2 << ' '
           << impl_->optimizer.gradient_clip_norm << '\n';
    output << impl_->seed << '\n';
    write_normalization(output, impl_->normalization);
    output << impl_->layers.size() << '\n';

    for (const Impl::Layer& layer : impl_->layers) {
        output << layer.input_size << ' ' << layer.output_size << ' ' << to_string(layer.activation) << '\n';
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

TrainingModel TrainingModel::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open model file for reading: " + path.string());
    }

    const std::string magic = read_token(input, "magic");
    const bool has_loss_config = magic == "BRUTAL_MLP_TRAINING_V3";
    const bool has_normalization = has_loss_config || magic == "BRUTAL_MLP_TRAINING_V2";
    if (!has_normalization && magic != "BRUTAL_MLP_TRAINING_V1" && magic != "BRUTAL_MLP_V1") {
        throw std::runtime_error("unsupported model file format");
    }

    const std::size_t input_size = read_value<std::size_t>(input, "input_size");
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
    optimizer.gradient_clip_norm = read_value<Scalar>(input, "gradient_clip_norm");

    const std::uint64_t seed = read_value<std::uint64_t>(input, "seed");
    NormalizationSpec normalization;
    if (has_normalization) {
        normalization = read_normalization(input);
    }
    const std::size_t layer_count = read_value<std::size_t>(input, "layer_count");

    Builder builder = TrainingModel::builder()
                          .input_size(input_size)
                          .loss(loss)
                          .optimizer(optimizer)
                          .normalization(normalization)
                          .seed(seed);
    std::vector<LayerParameters> parameters;
    parameters.reserve(layer_count);

    for (std::size_t i = 0; i < layer_count; ++i) {
        LayerParameters layer;
        layer.input_size = read_value<std::size_t>(input, "layer_input_size");
        layer.output_size = read_value<std::size_t>(input, "layer_output_size");
        layer.activation = activation_from_string(read_token(input, "layer_activation"));
        builder.add_layer(layer.output_size, layer.activation);

        const std::size_t weight_count = read_value<std::size_t>(input, "weight_count");
        layer.weights.resize(weight_count);
        for (Scalar& weight : layer.weights) {
            weight = read_value<Scalar>(input, "weight");
        }

        const std::size_t bias_count = read_value<std::size_t>(input, "bias_count");
        layer.biases.resize(bias_count);
        for (Scalar& bias : layer.biases) {
            bias = read_value<Scalar>(input, "bias");
        }
        parameters.push_back(std::move(layer));
    }

    TrainingModel model = builder.build();
    model.set_parameters(parameters);
    return model;
}

} // namespace brutal_mlp

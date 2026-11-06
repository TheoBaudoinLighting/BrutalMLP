// Feature normalization: validation, application and factories.

#include "normalization/normalization.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace brutal_mlp::detail {

namespace {

[[nodiscard]] bool feature_changes_value(const FeatureNormalization& feature) noexcept {
    return feature.mode != NormalizationMode::none || feature.clamp;
}

[[nodiscard]] bool has_feature_normalization(const std::vector<FeatureNormalization>& features) noexcept {
    return std::any_of(features.begin(), features.end(), feature_changes_value);
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

} // namespace

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

} // namespace brutal_mlp::detail

namespace brutal_mlp {

using namespace detail;

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

} // namespace brutal_mlp

#pragma once

// Internal header: Feature normalization: validation, application and factories.

#include "brutal_mlp/brutal_mlp.hpp"

#include "math/scalar_math.hpp"

#include <cstddef>
#include <vector>

namespace brutal_mlp::detail {

void validate_feature_normalization(const FeatureNormalization& feature);

void validate_normalization_spec(const NormalizationSpec& normalization,
                                 std::size_t input_size,
                                 std::size_t output_size);

void validate_normalization_for_loss(const NormalizationSpec& normalization, const LossConfig& loss);

[[nodiscard]] inline Scalar normalize_value(Scalar value, const FeatureNormalization& feature) noexcept {
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

[[nodiscard]] inline Scalar denormalize_value(Scalar value, const FeatureNormalization& feature) noexcept {
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

[[nodiscard]] Vector normalize_vector(const Vector& values, const std::vector<FeatureNormalization>& features);

void normalize_buffer_into(const Scalar* values,
                           std::size_t size,
                           const std::vector<FeatureNormalization>& features,
                           Scalar* result);

[[nodiscard]] Vector denormalize_vector(const Vector& values, const std::vector<FeatureNormalization>& features);

} // namespace brutal_mlp::detail

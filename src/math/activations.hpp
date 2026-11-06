#pragma once

// Internal header: Activation functions and their derivatives.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/scalar_type.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brutal_mlp::detail {

[[nodiscard]] inline Scalar clamp_probability(Scalar value) {
    if (value < kProbabilityEpsilon) {
        return kProbabilityEpsilon;
    }
    if (value > Scalar{1} - kProbabilityEpsilon) {
        return Scalar{1} - kProbabilityEpsilon;
    }
    return value;
}

[[nodiscard]] inline Scalar sigmoid(Scalar value) noexcept {
    if (value >= Scalar{0}) {
        const Scalar z = static_cast<Scalar>(std::exp(-value));
        return Scalar{1} / (Scalar{1} + z);
    }
    const Scalar z = static_cast<Scalar>(std::exp(value));
    return z / (Scalar{1} + z);
}

[[nodiscard]] Vector apply_activation(const Vector& values, Activation activation);

[[nodiscard]] inline Scalar activation_derivative(Activation activation, Scalar pre_activation, Scalar activation_value) {
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

[[nodiscard]] inline Scalar activation_derivative_from_pre_activation(Activation activation, Scalar pre_activation) {
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

[[nodiscard]] bool is_supported_activation(Activation activation) noexcept;

void apply_activation_in_place(Scalar* values, std::size_t size, Activation activation) noexcept;

} // namespace brutal_mlp::detail

// Activation functions and their derivatives.

#include "math/activations.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace brutal_mlp::detail {

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

} // namespace brutal_mlp::detail

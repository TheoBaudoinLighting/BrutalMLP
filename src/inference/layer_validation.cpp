// Validation of exported layer parameters.

#include "inference/layer_validation.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "math/activations.hpp"

#include <cstddef>
#include <vector>

namespace brutal_mlp::detail {

void validate_layer_parameters(const std::vector<LayerParameters>& parameters) {
    require(!parameters.empty(), "parameters must contain at least one layer");

    std::size_t expected_input_size = parameters.front().input_size;
    require(expected_input_size > 0, "input_size must be positive");

    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const LayerParameters& layer = parameters[i];
        require(layer.input_size == expected_input_size, "parameter input_size mismatch");
        require(layer.output_size > 0, "parameter output_size must be positive");
        require(is_supported_activation(layer.activation), "parameter activation is invalid");
        const std::size_t expected_weight_count =
            checked_product(layer.input_size, layer.output_size, "parameter weight");
        require(layer.weights.size() == expected_weight_count, "parameter weights size mismatch");
        require(layer.biases.size() == layer.output_size, "parameter biases size mismatch");
        validate_dropout_probability(layer.dropout_probability, "layer dropout_probability");
        if (i + 1 == parameters.size()) {
            require(layer.dropout_probability == Scalar{0}, "output layer dropout must be zero");
        }
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

} // namespace brutal_mlp::detail

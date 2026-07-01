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

constexpr Scalar kProbabilityEpsilon = 1e-12;

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
    if (value > 1.0 - kProbabilityEpsilon) {
        return 1.0 - kProbabilityEpsilon;
    }
    return value;
}

[[nodiscard]] Scalar sigmoid(Scalar value) {
    if (value >= 0.0) {
        const Scalar z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const Scalar z = std::exp(value);
    return z / (1.0 + z);
}

[[nodiscard]] Vector apply_activation(const Vector& values, Activation activation) {
    Vector result(values.size());

    switch (activation) {
    case Activation::linear:
        return values;
    case Activation::relu:
        std::transform(values.begin(), values.end(), result.begin(), [](Scalar v) {
            return std::max<Scalar>(0.0, v);
        });
        return result;
    case Activation::sigmoid:
        std::transform(values.begin(), values.end(), result.begin(), sigmoid);
        return result;
    case Activation::tanh:
        std::transform(values.begin(), values.end(), result.begin(), [](Scalar v) {
            return std::tanh(v);
        });
        return result;
    case Activation::softmax: {
        const Scalar max_value = *std::max_element(values.begin(), values.end());
        Scalar sum = 0.0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            result[i] = std::exp(values[i] - max_value);
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
        return 1.0;
    case Activation::relu:
        return pre_activation > 0.0 ? 1.0 : 0.0;
    case Activation::sigmoid:
        return activation_value * (1.0 - activation_value);
    case Activation::tanh:
        return 1.0 - activation_value * activation_value;
    case Activation::softmax:
        throw std::invalid_argument("softmax derivative requires the full Jacobian");
    }

    throw std::invalid_argument("unknown activation");
}

void validate_optimizer(const OptimizerConfig& optimizer) {
    require_finite(optimizer.learning_rate, "learning_rate");
    require(optimizer.learning_rate > 0.0, "learning_rate must be positive");
    require_finite(optimizer.beta1, "beta1");
    require_finite(optimizer.beta2, "beta2");
    require(optimizer.beta1 >= 0.0 && optimizer.beta1 < 1.0, "beta1 must be in [0, 1)");
    require(optimizer.beta2 >= 0.0 && optimizer.beta2 < 1.0, "beta2 must be in [0, 1)");
    require_finite(optimizer.epsilon, "epsilon");
    require(optimizer.epsilon > 0.0, "epsilon must be positive");
    require_finite(optimizer.momentum, "momentum");
    require(optimizer.momentum >= 0.0 && optimizer.momentum < 1.0, "momentum must be in [0, 1)");
    require_finite(optimizer.l2, "l2");
    require(optimizer.l2 >= 0.0, "l2 must be non-negative");
    require_finite(optimizer.gradient_clip_norm, "gradient_clip_norm");
    require(optimizer.gradient_clip_norm >= 0.0, "gradient_clip_norm must be non-negative");
}

void validate_training_options(const TrainingOptions& options) {
    require(options.epochs > 0, "epochs must be positive");
    require(options.batch_size > 0, "batch_size must be positive");
    require_finite(options.validation_split, "validation_split");
    require(options.validation_split >= 0.0 && options.validation_split < 1.0,
            "validation_split must be in [0, 1)");
    require_finite(options.min_delta, "min_delta");
    require(options.min_delta >= 0.0, "min_delta must be non-negative");
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
        return;
    case Loss::binary_cross_entropy:
        for (const Vector& target : targets) {
            for (Scalar value : target) {
                require(value >= 0.0 && value <= 1.0,
                        "binary_cross_entropy targets must be in [0, 1]");
            }
        }
        return;
    case Loss::categorical_cross_entropy:
        for (const Vector& target : targets) {
            Scalar sum = 0.0;
            for (Scalar value : target) {
                require(value >= 0.0, "categorical_cross_entropy targets must be non-negative");
                sum += value;
            }
            require(std::abs(sum - 1.0) <= 1e-6,
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
                            Loss loss) {
    validate_matrix(inputs, input_size, "inputs");
    validate_matrix(targets, output_size, "targets");
    require(inputs.size() == targets.size(), "inputs and targets must have the same number of rows");
    validate_targets_for_loss(targets, loss);
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
    case Loss::binary_cross_entropy:
        return "binary_cross_entropy";
    case Loss::categorical_cross_entropy:
        return "categorical_cross_entropy";
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
    if (value == "binary_cross_entropy") {
        return Loss::binary_cross_entropy;
    }
    if (value == "categorical_cross_entropy") {
        return Loss::categorical_cross_entropy;
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

struct Model::Impl {
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
    Loss loss{Loss::mean_squared_error};
    OptimizerConfig optimizer{OptimizerConfig::adam()};
    std::uint64_t seed{5489u};
    std::uint64_t optimizer_step{0};
    std::vector<Layer> layers;

    [[nodiscard]] std::size_t output_size() const {
        return layers.back().output_size;
    }

    [[nodiscard]] Vector forward(const Vector& input) const {
        Vector activations = input;
        for (const Layer& layer : layers) {
            Vector z(layer.output_size, 0.0);
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
            Vector z(layer.output_size, 0.0);
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
        Scalar total = 0.0;

        switch (loss) {
        case Loss::mean_squared_error:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar delta = prediction[i] - target[i];
                total += delta * delta;
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::binary_cross_entropy:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar p = clamp_probability(prediction[i]);
                total += -target[i] * std::log(p) - (1.0 - target[i]) * std::log(1.0 - p);
            }
            return total / static_cast<Scalar>(prediction.size());
        case Loss::categorical_cross_entropy:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                total += -target[i] * std::log(clamp_probability(prediction[i]));
            }
            return total;
        }

        throw std::invalid_argument("unknown loss");
    }

    [[nodiscard]] Vector output_delta(const Vector& prediction,
                                      const Vector& target,
                                      const Vector& output_pre_activation) const {
        const Layer& output_layer = layers.back();
        Vector delta(prediction.size(), 0.0);

        if (loss == Loss::categorical_cross_entropy && output_layer.activation == Activation::softmax) {
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                delta[i] = prediction[i] - target[i];
            }
            return delta;
        }

        if (loss == Loss::binary_cross_entropy && output_layer.activation == Activation::sigmoid) {
            const Scalar scale = 1.0 / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                delta[i] = (prediction[i] - target[i]) * scale;
            }
            return delta;
        }

        Vector activation_gradient(prediction.size(), 0.0);
        switch (loss) {
        case Loss::mean_squared_error: {
            const Scalar scale = 2.0 / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                activation_gradient[i] = (prediction[i] - target[i]) * scale;
            }
            break;
        }
        case Loss::binary_cross_entropy: {
            const Scalar scale = 1.0 / static_cast<Scalar>(prediction.size());
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                const Scalar p = clamp_probability(prediction[i]);
                activation_gradient[i] = ((1.0 - target[i]) / (1.0 - p) - target[i] / p) * scale;
            }
            break;
        }
        case Loss::categorical_cross_entropy:
            for (std::size_t i = 0; i < prediction.size(); ++i) {
                activation_gradient[i] = -target[i] / clamp_probability(prediction[i]);
            }
            break;
        }

        if (output_layer.activation == Activation::softmax) {
            const Scalar dot = std::inner_product(activation_gradient.begin(),
                                                  activation_gradient.end(),
                                                  prediction.begin(),
                                                  Scalar{0.0});
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
            Vector previous_delta(layer.input_size, 0.0);
            for (std::size_t in = 0; in < layer.input_size; ++in) {
                Scalar value = 0.0;
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
            std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), 0.0);
            std::fill(layer.grad_biases.begin(), layer.grad_biases.end(), 0.0);
        }
    }

    void apply_gradients(std::size_t batch_size) {
        ++optimizer_step;
        const Scalar batch_scale = 1.0 / static_cast<Scalar>(batch_size);
        Scalar norm_squared = 0.0;

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

        Scalar clip_scale = 1.0;
        if (optimizer.gradient_clip_norm > 0.0) {
            const Scalar norm = std::sqrt(norm_squared);
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
                if (optimizer.momentum > 0.0) {
                    velocities[i] = optimizer.momentum * velocities[i] - optimizer.learning_rate * gradient;
                    values[i] += velocities[i];
                } else {
                    values[i] -= optimizer.learning_rate * gradient;
                }
                break;
            case OptimizerType::adam: {
                first_moments[i] = optimizer.beta1 * first_moments[i] + (1.0 - optimizer.beta1) * gradient;
                second_moments[i] =
                    optimizer.beta2 * second_moments[i] + (1.0 - optimizer.beta2) * gradient * gradient;

                const Scalar bias_correction1 = 1.0 - std::pow(optimizer.beta1, static_cast<Scalar>(optimizer_step));
                const Scalar bias_correction2 = 1.0 - std::pow(optimizer.beta2, static_cast<Scalar>(optimizer_step));
                const Scalar m_hat = first_moments[i] / bias_correction1;
                const Scalar v_hat = second_moments[i] / bias_correction2;
                values[i] -= optimizer.learning_rate * m_hat / (std::sqrt(v_hat) + optimizer.epsilon);
                break;
            }
            }
        }
    }

    [[nodiscard]] Scalar loss_for_indices(const Matrix& inputs,
                                          const Matrix& targets,
                                          const std::vector<std::size_t>& indices) const {
        Scalar total = 0.0;
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
        std::fill(layer.velocity_weights.begin(), layer.velocity_weights.end(), 0.0);
        std::fill(layer.velocity_biases.begin(), layer.velocity_biases.end(), 0.0);
        std::fill(layer.first_moment_weights.begin(), layer.first_moment_weights.end(), 0.0);
        std::fill(layer.first_moment_biases.begin(), layer.first_moment_biases.end(), 0.0);
        std::fill(layer.second_moment_weights.begin(), layer.second_moment_weights.end(), 0.0);
        std::fill(layer.second_moment_biases.begin(), layer.second_moment_biases.end(), 0.0);
    }
};

Model::Builder& Model::Builder::input_size(std::size_t input_size) {
    input_size_ = input_size;
    return *this;
}

Model::Builder& Model::Builder::add_layer(std::size_t neurons, Activation activation) {
    layers_.push_back(LayerSpec{neurons, activation});
    return *this;
}

Model::Builder& Model::Builder::loss(Loss loss_value) {
    loss_ = loss_value;
    return *this;
}

Model::Builder& Model::Builder::optimizer(OptimizerConfig optimizer_value) {
    optimizer_ = optimizer_value;
    return *this;
}

Model::Builder& Model::Builder::seed(std::uint64_t seed_value) {
    seed_ = seed_value;
    return *this;
}

Model Model::Builder::build() const {
    require(input_size_ > 0, "input_size must be positive");
    require(!layers_.empty(), "model must contain at least one layer");
    validate_optimizer(optimizer_);

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        require(layers_[i].neurons > 0, "layer neurons must be positive");
        if (layers_[i].activation == Activation::softmax) {
            require(i + 1 == layers_.size(), "softmax is only supported on the output layer");
            require(layers_[i].neurons >= 2, "softmax output layer must contain at least two neurons");
        }
    }

    const Activation output_activation = layers_.back().activation;
    if (loss_ == Loss::binary_cross_entropy) {
        require(output_activation == Activation::sigmoid,
                "binary_cross_entropy requires a sigmoid output layer");
    }
    if (loss_ == Loss::categorical_cross_entropy) {
        require(output_activation == Activation::softmax,
                "categorical_cross_entropy requires a softmax output layer");
    }

    auto impl = std::make_unique<Impl>();
    impl->input_size = input_size_;
    impl->loss = loss_;
    impl->optimizer = optimizer_;
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
        layer.biases.assign(layer.output_size, 0.0);
        layer.grad_weights.assign(weight_count, 0.0);
        layer.grad_biases.assign(layer.output_size, 0.0);
        layer.velocity_weights.assign(weight_count, 0.0);
        layer.velocity_biases.assign(layer.output_size, 0.0);
        layer.first_moment_weights.assign(weight_count, 0.0);
        layer.first_moment_biases.assign(layer.output_size, 0.0);
        layer.second_moment_weights.assign(weight_count, 0.0);
        layer.second_moment_biases.assign(layer.output_size, 0.0);

        const Scalar limit = spec.activation == Activation::relu
                                 ? std::sqrt(6.0 / static_cast<Scalar>(layer.input_size))
                                 : std::sqrt(6.0 / static_cast<Scalar>(layer.input_size + layer.output_size));
        std::uniform_real_distribution<Scalar> distribution(-limit, limit);
        for (Scalar& weight : layer.weights) {
            weight = distribution(rng);
        }

        previous_size = spec.neurons;
        impl->layers.push_back(std::move(layer));
    }

    return Model(std::move(impl));
}

Model::Builder Model::builder() {
    return Builder{};
}

Model::Model(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Model::Model(const Model& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

Model& Model::operator=(const Model& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

Model::Model(Model&& other) noexcept = default;

Model& Model::operator=(Model&& other) noexcept = default;

Model::~Model() = default;

std::size_t Model::input_size() const {
    return impl_->input_size;
}

std::size_t Model::output_size() const {
    return impl_->output_size();
}

std::size_t Model::layer_count() const {
    return impl_->layers.size();
}

Loss Model::loss() const {
    return impl_->loss;
}

OptimizerConfig Model::optimizer() const {
    return impl_->optimizer;
}

Vector Model::predict(const Vector& input) const {
    require(input.size() == input_size(), "input has an invalid size");
    for (Scalar value : input) {
        require_finite(value, "input value");
    }
    return impl_->forward(input);
}

Matrix Model::predict_batch(const Matrix& inputs) const {
    validate_matrix(inputs, input_size(), "inputs", true);
    Matrix result;
    result.reserve(inputs.size());
    for (const Vector& input : inputs) {
        result.push_back(impl_->forward(input));
    }
    return result;
}

Scalar Model::evaluate_loss(const Matrix& inputs, const Matrix& targets) const {
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);
    Scalar total = 0.0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        total += impl_->sample_loss(impl_->forward(inputs[i]), targets[i]);
    }
    return total / static_cast<Scalar>(inputs.size());
}

TrainingHistory Model::fit(const Matrix& inputs, const Matrix& targets, const TrainingOptions& options) {
    validate_training_options(options);
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);

    std::vector<std::size_t> indices = make_indices(inputs.size());
    std::mt19937_64 split_rng(non_zero_seed(options.seed, impl_->seed));
    if (options.validation_split > 0.0) {
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
                impl_->accumulate_gradients(inputs[row], targets[row]);
            }
            impl_->apply_gradients(end - begin);
        }

        const Scalar training_loss = impl_->loss_for_indices(inputs, targets, training_indices);
        history.training_loss.push_back(training_loss);

        Scalar monitored_metric = training_loss;
        if (!validation_indices.empty()) {
            const Scalar validation_loss = impl_->loss_for_indices(inputs, targets, validation_indices);
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

std::vector<LayerParameters> Model::parameters() const {
    return impl_->parameters();
}

void Model::set_parameters(const std::vector<LayerParameters>& parameters) {
    impl_->set_parameters(parameters);
}

void Model::save(const std::filesystem::path& path) const {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open model file for writing: " + path.string());
    }

    output << std::setprecision(17);
    output << "BRUTAL_MLP_V1\n";
    output << impl_->input_size << '\n';
    output << to_string(impl_->loss) << '\n';
    output << to_string(impl_->optimizer.type) << ' '
           << impl_->optimizer.learning_rate << ' '
           << impl_->optimizer.beta1 << ' '
           << impl_->optimizer.beta2 << ' '
           << impl_->optimizer.epsilon << ' '
           << impl_->optimizer.momentum << ' '
           << impl_->optimizer.l2 << ' '
           << impl_->optimizer.gradient_clip_norm << '\n';
    output << impl_->seed << '\n';
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

Model Model::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open model file for reading: " + path.string());
    }

    const std::string magic = read_token(input, "magic");
    if (magic != "BRUTAL_MLP_V1") {
        throw std::runtime_error("unsupported model file format");
    }

    const std::size_t input_size = read_value<std::size_t>(input, "input_size");
    const Loss loss = loss_from_string(read_token(input, "loss"));

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
    const std::size_t layer_count = read_value<std::size_t>(input, "layer_count");

    Builder builder = Model::builder()
                          .input_size(input_size)
                          .loss(loss)
                          .optimizer(optimizer)
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

    Model model = builder.build();
    model.set_parameters(parameters);
    return model;
}

} // namespace brutal_mlp

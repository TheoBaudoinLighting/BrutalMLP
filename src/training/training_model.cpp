// TrainingModel builder, accessors, prediction and evaluation.

#include "brutal_mlp/brutal_mlp.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "data/data_validation.hpp"
#include "inference/inference_model_impl.hpp"
#include "math/activations.hpp"
#include "normalization/normalization.hpp"
#include "training/training_model_impl.hpp"
#include "training/weight_initialization.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace brutal_mlp {

using namespace detail;

TrainingModel::Builder& TrainingModel::Builder::input_size(std::size_t input_size) {
    input_size_ = input_size;
    return *this;
}

TrainingModel::Builder& TrainingModel::Builder::add_layer(std::size_t neurons,
                                                          Activation activation,
                                                          Scalar dropout_probability) {
    layers_.push_back(LayerSpec{neurons, activation, dropout_probability});
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

TrainingModel::Builder& TrainingModel::Builder::initialization(InitializationConfig initialization_value) {
    initialization_ = initialization_value;
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
    validate_initialization_config(initialization_);

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        require(layers_[i].neurons > 0, "layer neurons must be positive");
        validate_dropout_probability(layers_[i].dropout_probability, "layer dropout_probability");
        if (i + 1 == layers_.size()) {
            require(layers_[i].dropout_probability == Scalar{0}, "output layer dropout must be zero");
        }
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
        layer.dropout_probability = spec.dropout_probability;

        const std::size_t weight_count = layer.input_size * layer.output_size;
        layer.weights.resize(weight_count);
        layer.biases.assign(layer.output_size, initialization_.bias);
        layer.grad_weights.assign(weight_count, Scalar{0});
        layer.grad_biases.assign(layer.output_size, Scalar{0});
        layer.velocity_weights.assign(weight_count, Scalar{0});
        layer.velocity_biases.assign(layer.output_size, Scalar{0});
        layer.first_moment_weights.assign(weight_count, Scalar{0});
        layer.first_moment_biases.assign(layer.output_size, Scalar{0});
        layer.second_moment_weights.assign(weight_count, Scalar{0});
        layer.second_moment_biases.assign(layer.output_size, Scalar{0});

        initialize_weights(layer.weights,
                           resolve_weight_initialization(initialization_.weights, spec.activation),
                           layer.input_size,
                           layer.output_size,
                           rng);

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

std::size_t TrainingModel::completed_epochs() const {
    return impl_->completed_epochs;
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

Matrix TrainingModel::predict_batch(std::initializer_list<Vector> inputs) const {
    return predict_batch(Matrix(inputs));
}

DenseMatrix TrainingModel::predict_batch(const DenseMatrix& inputs) const {
    validate_dense_matrix(inputs, input_size(), "inputs", true);
    DenseMatrix result(inputs.rows(), output_size());
    for (std::size_t row = 0; row < inputs.rows(); ++row) {
        const Vector output = impl_->predict_raw(inputs.row_data(row), inputs.columns());
        std::copy(output.begin(), output.end(), result.row_data(row));
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

Scalar TrainingModel::evaluate_loss(std::initializer_list<Vector> inputs,
                                    std::initializer_list<Vector> targets) const {
    return evaluate_loss(Matrix(inputs), Matrix(targets));
}

Scalar TrainingModel::evaluate_loss(const DenseMatrix& inputs, const DenseMatrix& targets) const {
    validate_training_data(inputs, targets, input_size(), output_size(), impl_->loss);
    Scalar total = Scalar{0};
    for (std::size_t row = 0; row < inputs.rows(); ++row) {
        total += impl_->sample_loss(impl_->predict_raw(inputs.row_data(row), inputs.columns()),
                                    targets.row_data(row),
                                    targets.columns());
    }
    return total / static_cast<Scalar>(inputs.rows());
}

EvaluationResult TrainingModel::evaluate_metrics(const Matrix& inputs,
                                                 const Matrix& targets,
                                                 const EvaluationOptions& options) const {
    validate_matrix(inputs, input_size(), "inputs");
    validate_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

EvaluationResult TrainingModel::evaluate_metrics(std::initializer_list<Vector> inputs,
                                                 std::initializer_list<Vector> targets,
                                                 const EvaluationOptions& options) const {
    return evaluate_metrics(Matrix(inputs), Matrix(targets), options);
}

EvaluationResult TrainingModel::evaluate_metrics(const DenseMatrix& inputs,
                                                 const DenseMatrix& targets,
                                                 const EvaluationOptions& options) const {
    validate_dense_matrix(inputs, input_size(), "inputs");
    validate_dense_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

TrainingHistory TrainingModel::fit(const Matrix& inputs, const Matrix& targets, const TrainingOptions& options) {
    MatrixDataset dataset(inputs, targets);
    return fit(dataset, options);
}

TrainingHistory TrainingModel::fit(std::initializer_list<Vector> inputs,
                                   std::initializer_list<Vector> targets,
                                   const TrainingOptions& options) {
    return fit(Matrix(inputs), Matrix(targets), options);
}

TrainingHistory TrainingModel::fit(const DenseMatrix& inputs,
                                   const DenseMatrix& targets,
                                   const TrainingOptions& options) {
    DenseMatrixDataset dataset(inputs, targets);
    return fit(dataset, options);
}

std::vector<LayerParameters> TrainingModel::parameters() const {
    return impl_->parameters();
}

void TrainingModel::set_parameters(const std::vector<LayerParameters>& parameters) {
    impl_->set_parameters(parameters);
}

CompiledModel TrainingModel::compile() const {
    return CompiledModel(to_inference_model());
}

InferenceModel TrainingModel::to_inference_model() const {
    return InferenceModel::from_parameters(impl_->parameters(), impl_->normalization);
}

} // namespace brutal_mlp

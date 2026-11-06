// InferenceModel and CompiledModel runtime.

#include "inference/inference_model_impl.hpp"

#include "core/checks.hpp"
#include "core/parallel_workers.hpp"
#include "data/data_validation.hpp"
#include "inference/layer_validation.hpp"
#include "math/activations.hpp"
#include "math/dense_layer.hpp"
#include "normalization/normalization.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace brutal_mlp {

using namespace detail;

std::size_t InferenceModel::Impl::scratch_size() const noexcept {
    return input_scratch_size + (scratch_stride == 0 ? 0 : scratch_stride * 2);
}

void InferenceModel::Impl::predict_unchecked_to(const Scalar* input, Scalar* output, Scalar* scratch) const noexcept {
    Scalar* scratch_cursor = scratch;
    const Scalar* previous = input;
    if (normalization.has_input_normalization()) {
        for (std::size_t i = 0; i < input_size; ++i) {
            scratch_cursor[i] = normalize_value(input[i], normalization.input_features[i]);
        }
        previous = scratch_cursor;
        scratch_cursor += input_scratch_size;
    }

    Scalar* scratch_a = scratch_cursor;
    Scalar* scratch_b = scratch ? scratch + scratch_stride : nullptr;
    if (scratch_b && input_scratch_size > 0) {
        scratch_b += input_scratch_size;
    }
    bool write_a = true;

    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const Layer& layer = layers[layer_index];
        const bool output_layer = layer_index + 1 == layers.size();
        Scalar* target = output_layer ? output : (write_a ? scratch_a : scratch_b);

        dense_layer_to(previous,
                       weights.data() + layer.weight_offset,
                       biases.data() + layer.bias_offset,
                       layer.input_size,
                       layer.output_size,
                       target);
        apply_activation_in_place(target, layer.output_size, layer.activation);
        previous = target;
        write_a = !write_a;
    }

    if (normalization.has_output_normalization()) {
        for (std::size_t i = 0; i < output_size; ++i) {
            output[i] = denormalize_value(output[i], normalization.output_features[i]);
        }
    }
}

void InferenceModel::Impl::predict_batch_range_unchecked_to(const Scalar* inputs,
                                                            std::size_t sample_begin,
                                                            std::size_t sample_end,
                                                            std::size_t input_stride,
                                                            Scalar* outputs,
                                                            std::size_t output_stride,
                                                            Scalar* scratch) const noexcept {
    for (std::size_t sample = sample_begin; sample < sample_end; ++sample) {
        predict_unchecked_to(inputs + sample * input_stride,
                             outputs + sample * output_stride,
                             scratch);
    }
}

void InferenceModel::Impl::predict_batch_unchecked_to(const Scalar* inputs,
                                                      std::size_t sample_count,
                                                      std::size_t input_stride,
                                                      Scalar* outputs,
                                                      std::size_t output_stride,
                                                      Scalar* scratch) const noexcept {
    predict_batch_range_unchecked_to(inputs, 0, sample_count, input_stride, outputs, output_stride, scratch);
}

InferenceModel::InferenceModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CompiledModel::CompiledModel(InferenceModel model)
    : InferenceModel(std::move(model)) {}

CompiledModel CompiledModel::from_parameters(const std::vector<LayerParameters>& parameters,
                                             const NormalizationSpec& normalization) {
    return CompiledModel(InferenceModel::from_parameters(parameters, normalization));
}

CompiledModel CompiledModel::load(const std::filesystem::path& path) {
    return CompiledModel(InferenceModel::load(path));
}

CompiledModel CompiledModel::load_binary(const std::filesystem::path& path) {
    return CompiledModel(InferenceModel::load_binary(path));
}

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

std::size_t InferenceModel::batch_scratch_size(std::size_t sample_count,
                                               const ParallelOptions& parallelism) const noexcept {
    return empty() ? 0 : required_batch_scratch_size(impl_->scratch_size(), sample_count, parallelism);
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

    impl_->predict_unchecked_to(input, output, scratch);
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

    impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
    return InferenceStatus::ok;
}

InferenceStatus InferenceModel::predict_batch_to(const Scalar* inputs,
                                                 std::size_t sample_count,
                                                 std::size_t input_stride,
                                                 Scalar* outputs,
                                                 std::size_t output_stride,
                                                 Scalar* scratch,
                                                 std::size_t provided_scratch_size,
                                                 const ParallelOptions& parallelism) const noexcept {
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

    const std::size_t worker_count = resolve_parallel_worker_count(parallelism, sample_count);
    const std::size_t single_scratch_size = impl_->scratch_size();
    const std::size_t required_scratch_size = required_batch_scratch_size(single_scratch_size,
                                                                         sample_count,
                                                                         parallelism);
    if (required_scratch_size == std::numeric_limits<std::size_t>::max()) {
        return InferenceStatus::insufficient_scratch;
    }
    if (required_scratch_size > 0 && !scratch) {
        return InferenceStatus::null_scratch;
    }
    if (provided_scratch_size < required_scratch_size) {
        return InferenceStatus::insufficient_scratch;
    }

    if (worker_count <= 1) {
        impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
        return InferenceStatus::ok;
    }

    std::vector<std::thread> workers;
    try {
        workers.reserve(worker_count - 1);
        const std::size_t base_count = sample_count / worker_count;
        const std::size_t remainder = sample_count % worker_count;
        std::size_t begin = 0;
        for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
            const std::size_t count = base_count + (worker_index < remainder ? 1 : 0);
            const std::size_t end = begin + count;
            Scalar* worker_scratch = single_scratch_size == 0
                                         ? nullptr
                                         : scratch + worker_index * single_scratch_size;
            if (worker_index == 0) {
                begin = end;
                continue;
            }
            workers.emplace_back([impl = impl_.get(),
                                  inputs,
                                  begin,
                                  end,
                                  input_stride,
                                  outputs,
                                  output_stride,
                                  worker_scratch] {
                impl->predict_batch_range_unchecked_to(inputs,
                                                       begin,
                                                       end,
                                                       input_stride,
                                                       outputs,
                                                       output_stride,
                                                       worker_scratch);
            });
            begin = end;
        }

        const std::size_t first_end = base_count + (remainder > 0 ? 1 : 0);
        impl_->predict_batch_range_unchecked_to(inputs,
                                               0,
                                               first_end,
                                               input_stride,
                                               outputs,
                                               output_stride,
                                               scratch);
    } catch (...) {
        join_workers_noexcept(workers);
        impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
        return InferenceStatus::ok;
    }

    join_workers_noexcept(workers);
    return InferenceStatus::ok;
}

void InferenceModel::predict_unchecked_to(const Scalar* input,
                                          Scalar* output,
                                          Scalar* scratch) const noexcept {
    impl_->predict_unchecked_to(input, output, scratch);
}

void InferenceModel::predict_unchecked_to(const Scalar* input,
                                          InferenceWorkspace& workspace) const noexcept {
    predict_unchecked_to(input, workspace.output_data(), workspace.scratch_data());
}

void InferenceModel::predict_batch_unchecked_to(const Scalar* inputs,
                                                std::size_t sample_count,
                                                std::size_t input_stride,
                                                Scalar* outputs,
                                                std::size_t output_stride,
                                                Scalar* scratch) const noexcept {
    impl_->predict_batch_unchecked_to(inputs, sample_count, input_stride, outputs, output_stride, scratch);
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

Matrix InferenceModel::predict_batch(std::initializer_list<Vector> inputs) const {
    return predict_batch(Matrix(inputs));
}

DenseMatrix InferenceModel::predict_batch(const DenseMatrix& inputs) const {
    validate_dense_matrix(inputs, input_size(), "inputs", true);
    DenseMatrix result(inputs.rows(), output_size());
    if (inputs.empty()) {
        return result;
    }

    Vector scratch(scratch_size(), Scalar{0});
    const InferenceStatus status = predict_batch_to(inputs.data(),
                                                    inputs.rows(),
                                                    inputs.columns(),
                                                    result.data(),
                                                    result.columns(),
                                                    scratch.data(),
                                                    scratch.size());
    if (status != InferenceStatus::ok) {
        throw std::invalid_argument("inference failed: " + to_string(status));
    }
    return result;
}

EvaluationResult InferenceModel::evaluate_metrics(const Matrix& inputs,
                                                  const Matrix& targets,
                                                  const EvaluationOptions& options) const {
    validate_matrix(inputs, input_size(), "inputs");
    validate_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
}

EvaluationResult InferenceModel::evaluate_metrics(std::initializer_list<Vector> inputs,
                                                  std::initializer_list<Vector> targets,
                                                  const EvaluationOptions& options) const {
    return evaluate_metrics(Matrix(inputs), Matrix(targets), options);
}

EvaluationResult InferenceModel::evaluate_metrics(const DenseMatrix& inputs,
                                                  const DenseMatrix& targets,
                                                  const EvaluationOptions& options) const {
    validate_dense_matrix(inputs, input_size(), "inputs");
    validate_dense_matrix(targets, output_size(), "targets");
    return evaluate_predictions(predict_batch(inputs), targets, options);
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

} // namespace brutal_mlp

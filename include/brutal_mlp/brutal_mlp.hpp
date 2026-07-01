#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brutal_mlp {

using Scalar = double;
using Vector = std::vector<Scalar>;
using Matrix = std::vector<Vector>;

enum class Activation {
    linear,
    relu,
    sigmoid,
    tanh,
    softmax
};

enum class Loss {
    mean_squared_error,
    binary_cross_entropy,
    categorical_cross_entropy
};

enum class OptimizerType {
    sgd,
    adam
};

enum class InferenceStatus {
    ok,
    null_input,
    null_output,
    invalid_input_size,
    invalid_output_size,
    invalid_input_stride,
    invalid_output_stride,
    null_scratch,
    insufficient_scratch
};

[[nodiscard]] std::string to_string(Activation activation);
[[nodiscard]] std::string to_string(Loss loss);
[[nodiscard]] std::string to_string(OptimizerType optimizer);
[[nodiscard]] std::string to_string(InferenceStatus status);

[[nodiscard]] Activation activation_from_string(std::string_view value);
[[nodiscard]] Loss loss_from_string(std::string_view value);
[[nodiscard]] OptimizerType optimizer_type_from_string(std::string_view value);

struct OptimizerConfig {
    OptimizerType type{OptimizerType::adam};
    Scalar learning_rate{0.001};
    Scalar beta1{0.9};
    Scalar beta2{0.999};
    Scalar epsilon{1e-8};
    Scalar momentum{0.0};
    Scalar l2{0.0};
    Scalar gradient_clip_norm{0.0};

    [[nodiscard]] static OptimizerConfig adam(Scalar learning_rate = 0.001);
    [[nodiscard]] static OptimizerConfig sgd(Scalar learning_rate = 0.01, Scalar momentum = 0.0);
};

struct TrainingOptions {
    std::size_t epochs{100};
    std::size_t batch_size{32};
    bool shuffle{true};
    std::uint64_t seed{0};
    Scalar validation_split{0.0};
    std::size_t early_stopping_patience{0};
    Scalar min_delta{1e-8};
    bool restore_best_weights{true};
};

struct TrainingHistory {
    std::vector<Scalar> training_loss;
    std::vector<Scalar> validation_loss;
};

struct LayerSpec {
    std::size_t neurons{0};
    Activation activation{Activation::linear};
};

struct LayerParameters {
    std::size_t input_size{0};
    std::size_t output_size{0};
    Activation activation{Activation::linear};
    Vector weights;
    Vector biases;
};

class InferenceWorkspace;

class InferenceModel {
public:
    [[nodiscard]] static InferenceModel from_parameters(const std::vector<LayerParameters>& parameters);
    [[nodiscard]] static InferenceModel load(const std::filesystem::path& path);

    InferenceModel(const InferenceModel& other);
    InferenceModel& operator=(const InferenceModel& other);
    InferenceModel(InferenceModel&& other) noexcept;
    InferenceModel& operator=(InferenceModel&& other) noexcept;
    ~InferenceModel();

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] std::size_t output_size() const noexcept;
    [[nodiscard]] std::size_t layer_count() const noexcept;
    [[nodiscard]] std::size_t scratch_size() const noexcept;
    [[nodiscard]] std::size_t weight_count() const noexcept;
    [[nodiscard]] std::size_t bias_count() const noexcept;
    [[nodiscard]] const Scalar* weights_data() const noexcept;
    [[nodiscard]] const Scalar* biases_data() const noexcept;

    [[nodiscard]] InferenceStatus predict_to(const Scalar* input,
                                             std::size_t input_size,
                                             Scalar* output,
                                             std::size_t output_size,
                                             Scalar* scratch,
                                             std::size_t scratch_size) const noexcept;
    [[nodiscard]] InferenceStatus predict_to(const Scalar* input,
                                             std::size_t input_size,
                                             InferenceWorkspace& workspace) const noexcept;
    [[nodiscard]] InferenceStatus predict_batch_to(const Scalar* inputs,
                                                   std::size_t sample_count,
                                                   std::size_t input_stride,
                                                   Scalar* outputs,
                                                   std::size_t output_stride,
                                                   Scalar* scratch,
                                                   std::size_t scratch_size) const noexcept;

    [[nodiscard]] Vector predict(const Vector& input) const;
    [[nodiscard]] Matrix predict_batch(const Matrix& inputs) const;

    [[nodiscard]] std::vector<LayerParameters> parameters() const;
    void save(const std::filesystem::path& path) const;

private:
    struct Impl;

    explicit InferenceModel(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

class InferenceWorkspace {
public:
    InferenceWorkspace() = default;
    explicit InferenceWorkspace(const InferenceModel& model);

    void resize_for(const InferenceModel& model);
    void clear() noexcept;

    [[nodiscard]] std::size_t output_size() const noexcept;
    [[nodiscard]] std::size_t scratch_size() const noexcept;
    [[nodiscard]] Scalar* output_data() noexcept;
    [[nodiscard]] const Scalar* output_data() const noexcept;
    [[nodiscard]] Scalar* scratch_data() noexcept;
    [[nodiscard]] const Scalar* scratch_data() const noexcept;
    [[nodiscard]] const Vector& output() const noexcept;

private:
    Vector output_{};
    Vector scratch_{};
};

class TrainingModel {
public:
    class Builder {
    public:
        Builder& input_size(std::size_t input_size);
        Builder& add_layer(std::size_t neurons, Activation activation);
        Builder& loss(Loss loss);
        Builder& optimizer(OptimizerConfig optimizer);
        Builder& seed(std::uint64_t seed);

        [[nodiscard]] TrainingModel build() const;

    private:
        friend class TrainingModel;

        std::size_t input_size_{0};
        std::vector<LayerSpec> layers_{};
        Loss loss_{Loss::mean_squared_error};
        OptimizerConfig optimizer_{OptimizerConfig::adam()};
        std::uint64_t seed_{5489u};
    };

    [[nodiscard]] static Builder builder();
    [[nodiscard]] static TrainingModel load(const std::filesystem::path& path);

    TrainingModel(const TrainingModel& other);
    TrainingModel& operator=(const TrainingModel& other);
    TrainingModel(TrainingModel&& other) noexcept;
    TrainingModel& operator=(TrainingModel&& other) noexcept;
    ~TrainingModel();

    [[nodiscard]] std::size_t input_size() const;
    [[nodiscard]] std::size_t output_size() const;
    [[nodiscard]] std::size_t layer_count() const;
    [[nodiscard]] Loss loss() const;
    [[nodiscard]] OptimizerConfig optimizer() const;

    [[nodiscard]] Vector predict(const Vector& input) const;
    [[nodiscard]] Matrix predict_batch(const Matrix& inputs) const;
    [[nodiscard]] Scalar evaluate_loss(const Matrix& inputs, const Matrix& targets) const;

    TrainingHistory fit(const Matrix& inputs, const Matrix& targets, const TrainingOptions& options = {});

    [[nodiscard]] std::vector<LayerParameters> parameters() const;
    void set_parameters(const std::vector<LayerParameters>& parameters);
    [[nodiscard]] InferenceModel to_inference_model() const;

    void save(const std::filesystem::path& path) const;

private:
    struct Impl;

    explicit TrainingModel(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

using Model = TrainingModel;

} // namespace brutal_mlp

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brutal_mlp {

#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

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
    mean_absolute_error,
    huber,
    relative_mean_squared_error,
    log_cosh,
    weighted_mean_squared_error,
    binary_cross_entropy,
    categorical_cross_entropy,
    custom
};

enum class Metric {
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    custom
};

enum class Averaging {
    binary,
    macro,
    micro
};

enum class OptimizerType {
    sgd,
    adam
};

enum class NormalizationMode {
    none,
    standard_score,
    min_max
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
[[nodiscard]] std::string to_string(Metric metric);
[[nodiscard]] std::string to_string(Averaging averaging);
[[nodiscard]] std::string to_string(OptimizerType optimizer);
[[nodiscard]] std::string to_string(NormalizationMode mode);
[[nodiscard]] std::string to_string(InferenceStatus status);

[[nodiscard]] Activation activation_from_string(std::string_view value);
[[nodiscard]] Loss loss_from_string(std::string_view value);
[[nodiscard]] Metric metric_from_string(std::string_view value);
[[nodiscard]] Averaging averaging_from_string(std::string_view value);
[[nodiscard]] OptimizerType optimizer_type_from_string(std::string_view value);
[[nodiscard]] NormalizationMode normalization_mode_from_string(std::string_view value);

using CustomLossValueFunction = Scalar (*)(const Scalar* prediction,
                                           const Scalar* target,
                                           std::size_t size,
                                           void* context);
using CustomLossGradientFunction = void (*)(const Scalar* prediction,
                                            const Scalar* target,
                                            std::size_t size,
                                            Scalar* gradient,
                                            void* context);

struct CustomLoss {
    CustomLossValueFunction value{nullptr};
    CustomLossGradientFunction gradient{nullptr};
    void* context{nullptr};
};

struct LossConfig {
    Loss type{Loss::mean_squared_error};
    Scalar huber_delta{static_cast<Scalar>(1)};
    Scalar relative_epsilon{static_cast<Scalar>(1e-6)};
    Vector weights;
    CustomLoss custom_loss{};

    [[nodiscard]] static LossConfig from_loss(Loss loss);
    [[nodiscard]] static LossConfig mean_squared_error();
    [[nodiscard]] static LossConfig mean_absolute_error();
    [[nodiscard]] static LossConfig huber(Scalar delta = static_cast<Scalar>(1));
    [[nodiscard]] static LossConfig relative_mean_squared_error(Scalar epsilon = static_cast<Scalar>(1e-6));
    [[nodiscard]] static LossConfig log_cosh();
    [[nodiscard]] static LossConfig weighted_mean_squared_error(const Vector& weights);
    [[nodiscard]] static LossConfig binary_cross_entropy();
    [[nodiscard]] static LossConfig categorical_cross_entropy();
    [[nodiscard]] static LossConfig custom(CustomLossValueFunction value,
                                           CustomLossGradientFunction gradient,
                                           void* context = nullptr);
};

using CustomMetricFunction = Scalar (*)(const Matrix& predictions, const Matrix& targets, void* context);

struct CustomMetric {
    std::string name;
    CustomMetricFunction evaluate{nullptr};
    void* context{nullptr};
};

struct ConfusionMatrix {
    std::vector<std::vector<std::size_t>> counts;

    [[nodiscard]] std::size_t class_count() const noexcept;
    [[nodiscard]] std::size_t total() const noexcept;
};

struct MetricValue {
    Metric metric{Metric::custom};
    std::string name;
    Scalar value{static_cast<Scalar>(0)};
};

struct EvaluationOptions {
    std::vector<Metric> metrics;
    std::vector<CustomMetric> custom_metrics;
    Scalar classification_threshold{static_cast<Scalar>(0.5)};
    std::size_t positive_class{1};
    Averaging averaging{Averaging::macro};

    [[nodiscard]] static EvaluationOptions regression();
    [[nodiscard]] static EvaluationOptions binary_classification(Scalar threshold = static_cast<Scalar>(0.5),
                                                                 std::size_t positive_class = 1);
    [[nodiscard]] static EvaluationOptions multiclass_classification(Averaging averaging = Averaging::macro);
    [[nodiscard]] static EvaluationOptions all();

    EvaluationOptions& include(Metric metric);
    EvaluationOptions& add_custom_metric(std::string name, CustomMetricFunction evaluate, void* context = nullptr);
};

struct EvaluationResult {
    std::vector<MetricValue> values;
    ConfusionMatrix confusion_matrix;
    bool has_confusion_matrix{false};

    [[nodiscard]] bool has_metric(Metric metric) const noexcept;
    [[nodiscard]] bool has_custom_metric(std::string_view name) const noexcept;
    [[nodiscard]] Scalar metric(Metric metric) const;
    [[nodiscard]] Scalar custom_metric(std::string_view name) const;
};

[[nodiscard]] EvaluationResult evaluate_predictions(const Matrix& predictions,
                                                    const Matrix& targets,
                                                    const EvaluationOptions& options = {});

struct OptimizerConfig {
    OptimizerType type{OptimizerType::adam};
    Scalar learning_rate{static_cast<Scalar>(0.001)};
    Scalar beta1{static_cast<Scalar>(0.9)};
    Scalar beta2{static_cast<Scalar>(0.999)};
    Scalar epsilon{static_cast<Scalar>(1e-8)};
    Scalar momentum{static_cast<Scalar>(0)};
    Scalar l2{static_cast<Scalar>(0)};
    Scalar gradient_clip_norm{static_cast<Scalar>(0)};

    [[nodiscard]] static OptimizerConfig adam(Scalar learning_rate = static_cast<Scalar>(0.001));
    [[nodiscard]] static OptimizerConfig sgd(Scalar learning_rate = static_cast<Scalar>(0.01),
                                             Scalar momentum = static_cast<Scalar>(0));
};

struct TrainingOptions {
    std::size_t epochs{100};
    std::size_t batch_size{32};
    bool shuffle{true};
    std::uint64_t seed{0};
    Scalar validation_split{static_cast<Scalar>(0)};
    std::size_t early_stopping_patience{0};
    Scalar min_delta{static_cast<Scalar>(1e-8)};
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

struct FeatureNormalization {
    NormalizationMode mode{NormalizationMode::none};
    Scalar mean{static_cast<Scalar>(0)};
    Scalar stddev{static_cast<Scalar>(1)};
    Scalar minimum{static_cast<Scalar>(0)};
    Scalar maximum{static_cast<Scalar>(1)};
    Scalar normalized_min{static_cast<Scalar>(-1)};
    Scalar normalized_max{static_cast<Scalar>(1)};
    bool clamp{false};
    Scalar clamp_min{static_cast<Scalar>(-1)};
    Scalar clamp_max{static_cast<Scalar>(1)};

    [[nodiscard]] static FeatureNormalization none();
    [[nodiscard]] static FeatureNormalization standard_score(Scalar mean,
                                                             Scalar stddev,
                                                             bool clamp = false,
                                                             Scalar clamp_min = static_cast<Scalar>(-1),
                                                             Scalar clamp_max = static_cast<Scalar>(1));
    [[nodiscard]] static FeatureNormalization min_max(Scalar minimum,
                                                      Scalar maximum,
                                                      Scalar normalized_min = static_cast<Scalar>(-1),
                                                      Scalar normalized_max = static_cast<Scalar>(1),
                                                      bool clamp = true);
};

struct NormalizationSpec {
    std::vector<FeatureNormalization> input_features;
    std::vector<FeatureNormalization> output_features;

    [[nodiscard]] static NormalizationSpec none();
    [[nodiscard]] static NormalizationSpec standard_score(const Vector& input_means,
                                                          const Vector& input_stddevs,
                                                          const Vector& output_means = {},
                                                          const Vector& output_stddevs = {});
    [[nodiscard]] static NormalizationSpec min_max(const Vector& input_minimums,
                                                   const Vector& input_maximums,
                                                   const Vector& output_minimums = {},
                                                   const Vector& output_maximums = {},
                                                   Scalar normalized_min = static_cast<Scalar>(-1),
                                                   Scalar normalized_max = static_cast<Scalar>(1),
                                                   bool clamp = true);

    [[nodiscard]] bool has_input_normalization() const noexcept;
    [[nodiscard]] bool has_output_normalization() const noexcept;
};

class InferenceWorkspace;

class InferenceModel {
public:
    [[nodiscard]] static InferenceModel from_parameters(const std::vector<LayerParameters>& parameters,
                                                        const NormalizationSpec& normalization = {});
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
    [[nodiscard]] NormalizationSpec normalization() const;

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
    [[nodiscard]] EvaluationResult evaluate_metrics(const Matrix& inputs,
                                                    const Matrix& targets,
                                                    const EvaluationOptions& options = {}) const;

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
        Builder& loss(LossConfig loss);
        Builder& optimizer(OptimizerConfig optimizer);
        Builder& normalization(NormalizationSpec normalization);
        Builder& seed(std::uint64_t seed);

        [[nodiscard]] TrainingModel build() const;

    private:
        friend class TrainingModel;

        std::size_t input_size_{0};
        std::vector<LayerSpec> layers_{};
        LossConfig loss_{};
        OptimizerConfig optimizer_{OptimizerConfig::adam()};
        NormalizationSpec normalization_{};
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
    [[nodiscard]] LossConfig loss_config() const;
    [[nodiscard]] OptimizerConfig optimizer() const;
    [[nodiscard]] NormalizationSpec normalization() const;

    [[nodiscard]] Vector predict(const Vector& input) const;
    [[nodiscard]] Matrix predict_batch(const Matrix& inputs) const;
    [[nodiscard]] Scalar evaluate_loss(const Matrix& inputs, const Matrix& targets) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(const Matrix& inputs,
                                                    const Matrix& targets,
                                                    const EvaluationOptions& options = {}) const;

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

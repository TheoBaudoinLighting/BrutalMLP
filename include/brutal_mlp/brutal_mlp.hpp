#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Public API contracts are documented in docs/API_CONTRACTS.md.
namespace brutal_mlp {

#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif

using Vector = std::vector<Scalar>;
using Matrix = std::vector<Vector>;

class DenseMatrix {
public:
    DenseMatrix() = default;
    explicit DenseMatrix(std::size_t rows, std::size_t columns, Scalar value = Scalar{0});

    [[nodiscard]] static DenseMatrix from_matrix(const Matrix& matrix);
    [[nodiscard]] Matrix to_matrix() const;

    void resize(std::size_t rows, std::size_t columns, Scalar value = Scalar{0});
    void clear() noexcept;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t scalar_count() const noexcept;
    [[nodiscard]] const Vector& values() const noexcept;
    [[nodiscard]] Scalar* data() noexcept;
    [[nodiscard]] const Scalar* data() const noexcept;
    [[nodiscard]] Scalar* row_data(std::size_t row);
    [[nodiscard]] const Scalar* row_data(std::size_t row) const;
    [[nodiscard]] Scalar& operator()(std::size_t row, std::size_t column);
    [[nodiscard]] Scalar operator()(std::size_t row, std::size_t column) const;
    [[nodiscard]] Vector row(std::size_t row) const;
    void set_row(std::size_t row, const Vector& values);

private:
    std::size_t rows_{0};
    std::size_t columns_{0};
    Vector values_{};
};

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
    adam,
    adamw
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

enum class TrainingStopReason {
    completed_epochs,
    early_stopping,
    invalid_training_data,
    non_finite_loss,
    non_finite_gradient,
    non_finite_weights,
    loss_explosion,
    gradient_explosion
};

enum class TrainingMonitor {
    automatic,
    training_loss,
    validation_loss,
    test_loss
};

enum class TrainingMonitorMode {
    minimize,
    maximize
};

enum class LearningRateSchedule {
    constant,
    step_decay,
    exponential_decay,
    cosine_annealing,
    reduce_on_plateau
};

enum class WeightInitialization {
    automatic,
    he_normal,
    he_uniform,
    xavier_normal,
    xavier_uniform,
    lecun_normal,
    lecun_uniform
};

enum class ParallelExecution {
    serial,
    worker_threads
};

[[nodiscard]] std::string_view version_string() noexcept;
[[nodiscard]] std::string to_string(Activation activation);
[[nodiscard]] std::string to_string(Loss loss);
[[nodiscard]] std::string to_string(Metric metric);
[[nodiscard]] std::string to_string(Averaging averaging);
[[nodiscard]] std::string to_string(OptimizerType optimizer);
[[nodiscard]] std::string to_string(NormalizationMode mode);
[[nodiscard]] std::string to_string(InferenceStatus status);
[[nodiscard]] std::string to_string(TrainingStopReason reason);
[[nodiscard]] std::string to_string(TrainingMonitor monitor);
[[nodiscard]] std::string to_string(TrainingMonitorMode mode);
[[nodiscard]] std::string to_string(LearningRateSchedule schedule);
[[nodiscard]] std::string to_string(WeightInitialization initialization);
[[nodiscard]] std::string to_string(ParallelExecution execution);

[[nodiscard]] Activation activation_from_string(std::string_view value);
[[nodiscard]] Loss loss_from_string(std::string_view value);
[[nodiscard]] Metric metric_from_string(std::string_view value);
[[nodiscard]] Averaging averaging_from_string(std::string_view value);
[[nodiscard]] OptimizerType optimizer_type_from_string(std::string_view value);
[[nodiscard]] NormalizationMode normalization_mode_from_string(std::string_view value);
[[nodiscard]] ParallelExecution parallel_execution_from_string(std::string_view value);

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
[[nodiscard]] EvaluationResult evaluate_predictions(std::initializer_list<Vector> predictions,
                                                    std::initializer_list<Vector> targets,
                                                    const EvaluationOptions& options = {});
[[nodiscard]] EvaluationResult evaluate_predictions(const DenseMatrix& predictions,
                                                    const DenseMatrix& targets,
                                                    const EvaluationOptions& options = {});

struct OptimizerConfig {
    OptimizerType type{OptimizerType::adam};
    Scalar learning_rate{static_cast<Scalar>(0.001)};
    Scalar beta1{static_cast<Scalar>(0.9)};
    Scalar beta2{static_cast<Scalar>(0.999)};
    Scalar epsilon{static_cast<Scalar>(1e-8)};
    Scalar momentum{static_cast<Scalar>(0)};
    Scalar l2{static_cast<Scalar>(0)};
    Scalar l1{static_cast<Scalar>(0)};
    Scalar decoupled_weight_decay{static_cast<Scalar>(0)};
    Scalar max_norm{static_cast<Scalar>(0)};
    Scalar gradient_clip_norm{static_cast<Scalar>(0)};
    Scalar gradient_clip_value{static_cast<Scalar>(0)};
    Scalar layer_gradient_clip_norm{static_cast<Scalar>(0)};

    [[nodiscard]] static OptimizerConfig adam(Scalar learning_rate = static_cast<Scalar>(0.001));
    [[nodiscard]] static OptimizerConfig adamw(Scalar learning_rate = static_cast<Scalar>(0.001),
                                               Scalar weight_decay = static_cast<Scalar>(0.01));
    [[nodiscard]] static OptimizerConfig sgd(Scalar learning_rate = static_cast<Scalar>(0.01),
                                             Scalar momentum = static_cast<Scalar>(0));
};

struct LearningRateScheduleConfig {
    LearningRateSchedule type{LearningRateSchedule::constant};
    Scalar base_learning_rate{static_cast<Scalar>(0)};
    Scalar minimum_learning_rate{static_cast<Scalar>(0)};
    std::size_t warmup_epochs{0};
    Scalar warmup_start_learning_rate{static_cast<Scalar>(0)};
    std::size_t step_size{10};
    Scalar step_decay_factor{static_cast<Scalar>(0.5)};
    Scalar exponential_decay_rate{static_cast<Scalar>(0.95)};
    std::size_t cosine_epochs{0};
    TrainingMonitor reduce_on_plateau_monitor{TrainingMonitor::automatic};
    TrainingMonitorMode reduce_on_plateau_mode{TrainingMonitorMode::minimize};
    std::size_t reduce_on_plateau_patience{10};
    Scalar reduce_on_plateau_factor{static_cast<Scalar>(0.5)};
    Scalar reduce_on_plateau_min_delta{static_cast<Scalar>(1e-8)};
    std::size_t reduce_on_plateau_cooldown{0};

    [[nodiscard]] static LearningRateScheduleConfig constant();
    [[nodiscard]] static LearningRateScheduleConfig step_decay(std::size_t step_size,
                                                               Scalar decay_factor);
    [[nodiscard]] static LearningRateScheduleConfig exponential_decay(Scalar decay_rate);
    [[nodiscard]] static LearningRateScheduleConfig cosine_annealing(
        std::size_t cosine_epochs = 0,
        Scalar minimum_learning_rate = static_cast<Scalar>(0));
    [[nodiscard]] static LearningRateScheduleConfig reduce_on_plateau(
        std::size_t patience = 10,
        Scalar factor = static_cast<Scalar>(0.5));
};

struct InitializationConfig {
    WeightInitialization weights{WeightInitialization::automatic};
    Scalar bias{static_cast<Scalar>(0)};

    [[nodiscard]] static InitializationConfig automatic(Scalar bias = static_cast<Scalar>(0));
    [[nodiscard]] static InitializationConfig he_normal(Scalar bias = static_cast<Scalar>(0));
    [[nodiscard]] static InitializationConfig he_uniform(Scalar bias = static_cast<Scalar>(0));
    [[nodiscard]] static InitializationConfig xavier_normal(Scalar bias = static_cast<Scalar>(0));
    [[nodiscard]] static InitializationConfig xavier_uniform(Scalar bias = static_cast<Scalar>(0));
    [[nodiscard]] static InitializationConfig lecun_normal(Scalar bias = static_cast<Scalar>(0));
    [[nodiscard]] static InitializationConfig lecun_uniform(Scalar bias = static_cast<Scalar>(0));
};

struct ParallelOptions {
    ParallelExecution execution{ParallelExecution::serial};
    std::size_t thread_count{0};
    std::size_t minimum_parallel_samples{1};
};

struct TrainingDebugOptions {
    bool enabled{false};
    Scalar loss_explosion_factor{static_cast<Scalar>(1000)};
    Scalar loss_explosion_threshold{static_cast<Scalar>(1e20)};
    Scalar gradient_norm_explosion_factor{static_cast<Scalar>(1000)};
    Scalar gradient_norm_explosion_threshold{static_cast<Scalar>(1e10)};
};

struct TrainingOptions {
    std::size_t epochs{100};
    std::size_t batch_size{32};
    bool shuffle{true};
    std::uint64_t seed{0};
    Scalar validation_split{static_cast<Scalar>(0)};
    Scalar test_split{static_cast<Scalar>(0)};
    std::size_t streaming_shuffle_buffer_size{0};
    std::size_t early_stopping_patience{0};
    Scalar min_delta{static_cast<Scalar>(1e-8)};
    TrainingMonitor early_stopping_monitor{TrainingMonitor::automatic};
    TrainingMonitorMode early_stopping_mode{TrainingMonitorMode::minimize};
    std::size_t early_stopping_cooldown{0};
    LearningRateScheduleConfig learning_rate_schedule{};
    Scalar gradient_noise_stddev{static_cast<Scalar>(0)};
    ParallelOptions parallelism{};
    TrainingDebugOptions debug{};
    bool restore_best_weights{true};
    std::filesystem::path best_checkpoint_path{};
    std::filesystem::path latest_checkpoint_path{};
};

struct LearningRateSchedulerState {
    bool initialized{false};
    Scalar base_learning_rate{static_cast<Scalar>(0)};
    Scalar current_learning_rate{static_cast<Scalar>(0)};
    bool has_best_metric{false};
    Scalar best_metric{static_cast<Scalar>(0)};
    std::size_t stale_epochs{0};
    std::size_t cooldown_remaining{0};
};

struct WeightStatistics {
    Scalar minimum{static_cast<Scalar>(0)};
    Scalar maximum{static_cast<Scalar>(0)};
    Scalar mean{static_cast<Scalar>(0)};
    std::size_t count{0};
    std::size_t non_finite_count{0};
    bool finite{true};
};

struct GradientClippingDiagnostics {
    std::size_t batch_count{0};
    std::size_t global_clip_count{0};
    std::size_t layer_count{0};
    std::size_t layer_clip_count{0};
    std::size_t gradient_value_count{0};
    std::size_t gradient_value_clip_count{0};
    Scalar global_clip_rate{static_cast<Scalar>(0)};
    Scalar layer_clip_rate{static_cast<Scalar>(0)};
    Scalar value_clip_rate{static_cast<Scalar>(0)};
    Scalar minimum_clip_scale{static_cast<Scalar>(1)};
};

struct TrainingEpochDiagnostics {
    std::size_t epoch{0};
    Scalar training_loss{static_cast<Scalar>(0)};
    bool has_validation_loss{false};
    Scalar validation_loss{static_cast<Scalar>(0)};
    bool has_test_loss{false};
    Scalar test_loss{static_cast<Scalar>(0)};
    TrainingMonitor monitor{TrainingMonitor::automatic};
    Scalar monitored_metric{static_cast<Scalar>(0)};
    LearningRateSchedule learning_rate_schedule{LearningRateSchedule::constant};
    Scalar learning_rate{static_cast<Scalar>(0)};
    Scalar next_learning_rate{static_cast<Scalar>(0)};
    bool learning_rate_reduced{false};
    Scalar gradient_norm{static_cast<Scalar>(0)};
    GradientClippingDiagnostics clipping{};
    WeightStatistics weights{};
    bool finite{true};
    std::size_t non_finite_parameter_count{0};
    double epoch_seconds{0.0};
    bool best_checkpoint{false};
    bool improved{false};
    std::size_t stale_epochs{0};
    std::size_t cooldown_remaining{0};
};

enum class BinaryScalarType {
    float32,
    float64
};

enum class BinaryModelKind {
    training,
    inference,
    checkpoint
};

struct BinaryMetadataEntry {
    std::string key;
    std::string value;
};

struct BinaryMetadata {
    std::uint64_t created_unix_time{0};
    std::string description;
    std::vector<BinaryMetadataEntry> entries;
};

struct TrainingCheckpointInfo {
    std::size_t completed_epochs{0};
    bool best_checkpoint{false};
    bool has_metric{false};
    Scalar metric{static_cast<Scalar>(0)};
    LearningRateSchedulerState learning_rate_state{};
    TrainingOptions training_options{};
    BinaryMetadata metadata{};
};

struct BinaryModelInfo {
    std::uint32_t version{0};
    BinaryScalarType scalar_type{BinaryScalarType::float32};
    BinaryModelKind model_kind{BinaryModelKind::training};
    std::size_t input_size{0};
    std::size_t output_size{0};
    std::size_t layer_count{0};
    std::size_t weight_count{0};
    std::size_t bias_count{0};
    std::uint64_t seed{0};
    std::uint64_t optimizer_step{0};
    std::size_t completed_epochs{0};
    bool checkpoint_best{false};
    bool has_checkpoint_metric{false};
    Scalar checkpoint_metric{static_cast<Scalar>(0)};
    bool has_learning_rate_state{false};
    LearningRateSchedulerState learning_rate_state{};
    bool has_training_options{false};
    TrainingOptions training_options{};
    BinaryMetadata metadata{};
    std::uint32_t checksum{0};
};

[[nodiscard]] BinaryModelInfo inspect_binary_model(const std::filesystem::path& path);

using IndexedSampleFunction = void (*)(std::size_t index,
                                       Scalar* input,
                                       std::size_t input_size,
                                       Scalar* target,
                                       std::size_t target_size,
                                       void* context);
using StreamingSampleFunction = bool (*)(Scalar* input,
                                         std::size_t input_size,
                                         Scalar* target,
                                         std::size_t target_size,
                                         void* context);
using StreamingResetFunction = void (*)(void* context);

class Dataset {
public:
    virtual ~Dataset();

    [[nodiscard]] virtual std::size_t sample_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t input_size() const noexcept = 0;
    [[nodiscard]] virtual std::size_t output_size() const noexcept = 0;
    virtual void sample(std::size_t index,
                        Scalar* input,
                        std::size_t input_size,
                        Scalar* target,
                        std::size_t target_size) const = 0;
};

class MatrixDataset final : public Dataset {
public:
    MatrixDataset(const Matrix& inputs, const Matrix& targets);

    [[nodiscard]] std::size_t sample_count() const noexcept override;
    [[nodiscard]] std::size_t input_size() const noexcept override;
    [[nodiscard]] std::size_t output_size() const noexcept override;
    void sample(std::size_t index,
                Scalar* input,
                std::size_t input_size,
                Scalar* target,
                std::size_t target_size) const override;

private:
    const Matrix* inputs_{nullptr};
    const Matrix* targets_{nullptr};
    std::size_t input_size_{0};
    std::size_t output_size_{0};
};

class DenseMatrixDataset final : public Dataset {
public:
    DenseMatrixDataset(const DenseMatrix& inputs, const DenseMatrix& targets);

    [[nodiscard]] std::size_t sample_count() const noexcept override;
    [[nodiscard]] std::size_t input_size() const noexcept override;
    [[nodiscard]] std::size_t output_size() const noexcept override;
    void sample(std::size_t index,
                Scalar* input,
                std::size_t input_size,
                Scalar* target,
                std::size_t target_size) const override;

private:
    const DenseMatrix* inputs_{nullptr};
    const DenseMatrix* targets_{nullptr};
};

class GeneratedDataset final : public Dataset {
public:
    GeneratedDataset(std::size_t sample_count,
                     std::size_t input_size,
                     std::size_t output_size,
                     IndexedSampleFunction generate,
                     void* context = nullptr);

    [[nodiscard]] std::size_t sample_count() const noexcept override;
    [[nodiscard]] std::size_t input_size() const noexcept override;
    [[nodiscard]] std::size_t output_size() const noexcept override;
    void sample(std::size_t index,
                Scalar* input,
                std::size_t input_size,
                Scalar* target,
                std::size_t target_size) const override;

private:
    std::size_t sample_count_{0};
    std::size_t input_size_{0};
    std::size_t output_size_{0};
    IndexedSampleFunction generate_{nullptr};
    void* context_{nullptr};
};

struct DatasetSplitOptions {
    Scalar validation_split{static_cast<Scalar>(0)};
    Scalar test_split{static_cast<Scalar>(0)};
    bool shuffle{true};
    std::uint64_t seed{0};
};

struct DatasetSplit {
    std::vector<std::size_t> training_indices;
    std::vector<std::size_t> validation_indices;
    std::vector<std::size_t> test_indices;
};

[[nodiscard]] DatasetSplit make_dataset_split(std::size_t sample_count, const DatasetSplitOptions& options = {});

class DatasetView final : public Dataset {
public:
    DatasetView(const Dataset& source, std::vector<std::size_t> indices);

    [[nodiscard]] std::size_t sample_count() const noexcept override;
    [[nodiscard]] std::size_t input_size() const noexcept override;
    [[nodiscard]] std::size_t output_size() const noexcept override;
    [[nodiscard]] const std::vector<std::size_t>& indices() const noexcept;
    void sample(std::size_t index,
                Scalar* input,
                std::size_t input_size,
                Scalar* target,
                std::size_t target_size) const override;

private:
    const Dataset* source_{nullptr};
    std::vector<std::size_t> indices_;
};

struct MiniBatch {
    Vector inputs;
    Vector targets;
    std::size_t sample_count{0};
    std::size_t input_size{0};
    std::size_t output_size{0};

    void resize(std::size_t samples, std::size_t input_width, std::size_t output_width);
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    void clear() noexcept;
    [[nodiscard]] Scalar* input_data(std::size_t sample);
    [[nodiscard]] const Scalar* input_data(std::size_t sample) const;
    [[nodiscard]] Scalar* target_data(std::size_t sample);
    [[nodiscard]] const Scalar* target_data(std::size_t sample) const;
    [[nodiscard]] Vector input(std::size_t sample) const;
    [[nodiscard]] Vector target(std::size_t sample) const;
};

class BatchGenerator {
public:
    BatchGenerator(const Dataset& dataset,
                   std::size_t batch_size,
                   bool shuffle = false,
                   std::uint64_t seed = 0);
    BatchGenerator(const BatchGenerator& other);
    BatchGenerator& operator=(const BatchGenerator& other);
    BatchGenerator(BatchGenerator&& other) noexcept;
    BatchGenerator& operator=(BatchGenerator&& other) noexcept;
    ~BatchGenerator();

    void reset(std::uint64_t epoch = 0);
    [[nodiscard]] bool next(MiniBatch& batch);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class StreamingDataset {
public:
    virtual ~StreamingDataset();

    [[nodiscard]] virtual std::size_t sample_count() const noexcept = 0;
    [[nodiscard]] virtual std::size_t input_size() const noexcept = 0;
    [[nodiscard]] virtual std::size_t output_size() const noexcept = 0;
    virtual void reset() = 0;
    [[nodiscard]] virtual bool next(Scalar* input,
                                    std::size_t input_size,
                                    Scalar* target,
                                    std::size_t target_size) = 0;
};

class FunctionStreamingDataset final : public StreamingDataset {
public:
    FunctionStreamingDataset(std::size_t sample_count,
                             std::size_t input_size,
                             std::size_t output_size,
                             StreamingSampleFunction next,
                             StreamingResetFunction reset = nullptr,
                             void* context = nullptr);

    [[nodiscard]] std::size_t sample_count() const noexcept override;
    [[nodiscard]] std::size_t input_size() const noexcept override;
    [[nodiscard]] std::size_t output_size() const noexcept override;
    void reset() override;
    [[nodiscard]] bool next(Scalar* input,
                            std::size_t input_size,
                            Scalar* target,
                            std::size_t target_size) override;

private:
    std::size_t sample_count_{0};
    std::size_t input_size_{0};
    std::size_t output_size_{0};
    StreamingSampleFunction next_{nullptr};
    StreamingResetFunction reset_{nullptr};
    void* context_{nullptr};
};

struct CsvStreamingOptions {
    char delimiter{','};
    bool has_header{false};
};

class CsvStreamingDataset final : public StreamingDataset {
public:
    CsvStreamingDataset(const std::filesystem::path& path,
                        std::size_t input_size,
                        std::size_t output_size,
                        CsvStreamingOptions options = {});
    CsvStreamingDataset(const CsvStreamingDataset& other);
    CsvStreamingDataset& operator=(const CsvStreamingDataset& other);
    CsvStreamingDataset(CsvStreamingDataset&& other) noexcept;
    CsvStreamingDataset& operator=(CsvStreamingDataset&& other) noexcept;
    ~CsvStreamingDataset();

    [[nodiscard]] std::size_t sample_count() const noexcept override;
    [[nodiscard]] std::size_t input_size() const noexcept override;
    [[nodiscard]] std::size_t output_size() const noexcept override;
    void reset() override;
    [[nodiscard]] bool next(Scalar* input,
                            std::size_t input_size,
                            Scalar* target,
                            std::size_t target_size) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct LayerSpec {
    std::size_t neurons{0};
    Activation activation{Activation::linear};
    Scalar dropout_probability{static_cast<Scalar>(0)};
};

struct LayerParameters {
    std::size_t input_size{0};
    std::size_t output_size{0};
    Activation activation{Activation::linear};
    Vector weights;
    Vector biases;
    Scalar dropout_probability{static_cast<Scalar>(0)};
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

struct TrainingRunLayerConfig {
    std::size_t input_size{0};
    std::size_t output_size{0};
    Activation activation{Activation::linear};
    Scalar dropout_probability{static_cast<Scalar>(0)};
};

struct TrainingRunDatasetConfig {
    std::size_t sample_count{0};
    std::size_t input_size{0};
    std::size_t output_size{0};
    std::size_t training_sample_count{0};
    std::size_t validation_sample_count{0};
    std::size_t test_sample_count{0};
    Scalar validation_split{static_cast<Scalar>(0)};
    Scalar test_split{static_cast<Scalar>(0)};
    bool shuffle{true};
    std::uint64_t split_seed{0};
    bool streaming{false};
    std::size_t streaming_shuffle_buffer_size{0};
};

struct TrainingRunResultConfig {
    std::size_t completed_epochs{0};
    bool has_best_score{false};
    std::size_t best_epoch{0};
    Scalar best_score{static_cast<Scalar>(0)};
    TrainingMonitor monitor{TrainingMonitor::automatic};
    TrainingMonitorMode monitor_mode{TrainingMonitorMode::minimize};
    TrainingStopReason stop_reason{TrainingStopReason::completed_epochs};
    std::string stop_message;
    bool has_final_training_loss{false};
    Scalar final_training_loss{static_cast<Scalar>(0)};
    bool has_final_validation_loss{false};
    Scalar final_validation_loss{static_cast<Scalar>(0)};
    bool has_final_test_loss{false};
    Scalar final_test_loss{static_cast<Scalar>(0)};
};

struct TrainingRunConfig {
    std::string library_name{"brutal_mlp"};
    std::string library_version;
    std::string scalar_type;
    std::uint64_t created_unix_time{0};
    std::string created_utc;
    std::uint64_t model_seed{0};
    std::uint64_t training_seed{0};
    std::size_t input_size{0};
    std::size_t output_size{0};
    std::vector<TrainingRunLayerConfig> architecture;
    OptimizerConfig optimizer{};
    LearningRateScheduleConfig learning_rate_schedule{};
    LossConfig loss{};
    NormalizationSpec normalization{};
    TrainingOptions options{};
    TrainingRunDatasetConfig dataset{};
    TrainingRunResultConfig result{};

    [[nodiscard]] std::string to_json() const;
};

struct TrainingHistory {
    std::vector<Scalar> training_loss;
    std::vector<Scalar> validation_loss;
    std::vector<Scalar> test_loss;
    std::vector<TrainingEpochDiagnostics> epochs;
    TrainingRunConfig run_config;
    bool has_best_checkpoint{false};
    std::size_t best_epoch{0};
    Scalar best_metric{static_cast<Scalar>(0)};
    TrainingMonitor monitor{TrainingMonitor::automatic};
    TrainingMonitorMode monitor_mode{TrainingMonitorMode::minimize};
    TrainingStopReason stop_reason{TrainingStopReason::completed_epochs};
    std::string stop_message;
};

class InferenceWorkspace;

class InferenceModel {
public:
    [[nodiscard]] static InferenceModel from_parameters(const std::vector<LayerParameters>& parameters,
                                                        const NormalizationSpec& normalization = {});
    [[nodiscard]] static InferenceModel load(const std::filesystem::path& path);
    [[nodiscard]] static InferenceModel load_binary(const std::filesystem::path& path);

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
    [[nodiscard]] std::size_t batch_scratch_size(std::size_t sample_count,
                                                 const ParallelOptions& parallelism) const noexcept;
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
    [[nodiscard]] InferenceStatus predict_batch_to(const Scalar* inputs,
                                                   std::size_t sample_count,
                                                   std::size_t input_stride,
                                                   Scalar* outputs,
                                                   std::size_t output_stride,
                                                   Scalar* scratch,
                                                   std::size_t scratch_size,
                                                   const ParallelOptions& parallelism) const noexcept;
    void predict_unchecked_to(const Scalar* input,
                              Scalar* output,
                              Scalar* scratch) const noexcept;
    void predict_unchecked_to(const Scalar* input,
                              InferenceWorkspace& workspace) const noexcept;
    void predict_batch_unchecked_to(const Scalar* inputs,
                                    std::size_t sample_count,
                                    std::size_t input_stride,
                                    Scalar* outputs,
                                    std::size_t output_stride,
                                    Scalar* scratch) const noexcept;

    [[nodiscard]] Vector predict(const Vector& input) const;
    [[nodiscard]] Matrix predict_batch(const Matrix& inputs) const;
    [[nodiscard]] Matrix predict_batch(std::initializer_list<Vector> inputs) const;
    [[nodiscard]] DenseMatrix predict_batch(const DenseMatrix& inputs) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(const Matrix& inputs,
                                                    const Matrix& targets,
                                                    const EvaluationOptions& options = {}) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(std::initializer_list<Vector> inputs,
                                                    std::initializer_list<Vector> targets,
                                                    const EvaluationOptions& options = {}) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(const DenseMatrix& inputs,
                                                    const DenseMatrix& targets,
                                                    const EvaluationOptions& options = {}) const;

    [[nodiscard]] std::vector<LayerParameters> parameters() const;
    void save(const std::filesystem::path& path) const;
    void save_binary(const std::filesystem::path& path, const BinaryMetadata& metadata = {}) const;

private:
    struct Impl;

    explicit InferenceModel(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

class CompiledModel final : public InferenceModel {
public:
    [[nodiscard]] static CompiledModel from_parameters(const std::vector<LayerParameters>& parameters,
                                                       const NormalizationSpec& normalization = {});
    [[nodiscard]] static CompiledModel load(const std::filesystem::path& path);
    [[nodiscard]] static CompiledModel load_binary(const std::filesystem::path& path);

    explicit CompiledModel(InferenceModel model);
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
        Builder& add_layer(std::size_t neurons,
                           Activation activation,
                           Scalar dropout_probability = static_cast<Scalar>(0));
        Builder& loss(Loss loss);
        Builder& loss(LossConfig loss);
        Builder& optimizer(OptimizerConfig optimizer);
        Builder& normalization(NormalizationSpec normalization);
        Builder& initialization(InitializationConfig initialization);
        Builder& seed(std::uint64_t seed);

        [[nodiscard]] TrainingModel build() const;

    private:
        friend class TrainingModel;

        std::size_t input_size_{0};
        std::vector<LayerSpec> layers_{};
        LossConfig loss_{};
        OptimizerConfig optimizer_{OptimizerConfig::adam()};
        NormalizationSpec normalization_{};
        InitializationConfig initialization_{};
        std::uint64_t seed_{5489u};
    };

    [[nodiscard]] static Builder builder();
    [[nodiscard]] static TrainingModel load(const std::filesystem::path& path);
    [[nodiscard]] static TrainingModel load_binary(const std::filesystem::path& path);
    [[nodiscard]] static TrainingModel load_checkpoint(const std::filesystem::path& path,
                                                       TrainingCheckpointInfo* info = nullptr);

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
    [[nodiscard]] std::size_t completed_epochs() const;

    [[nodiscard]] Vector predict(const Vector& input) const;
    [[nodiscard]] Matrix predict_batch(const Matrix& inputs) const;
    [[nodiscard]] Matrix predict_batch(std::initializer_list<Vector> inputs) const;
    [[nodiscard]] DenseMatrix predict_batch(const DenseMatrix& inputs) const;
    [[nodiscard]] Scalar evaluate_loss(const Matrix& inputs, const Matrix& targets) const;
    [[nodiscard]] Scalar evaluate_loss(std::initializer_list<Vector> inputs,
                                       std::initializer_list<Vector> targets) const;
    [[nodiscard]] Scalar evaluate_loss(const DenseMatrix& inputs, const DenseMatrix& targets) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(const Matrix& inputs,
                                                    const Matrix& targets,
                                                    const EvaluationOptions& options = {}) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(std::initializer_list<Vector> inputs,
                                                    std::initializer_list<Vector> targets,
                                                    const EvaluationOptions& options = {}) const;
    [[nodiscard]] EvaluationResult evaluate_metrics(const DenseMatrix& inputs,
                                                    const DenseMatrix& targets,
                                                    const EvaluationOptions& options = {}) const;

    TrainingHistory fit(const Matrix& inputs, const Matrix& targets, const TrainingOptions& options = {});
    TrainingHistory fit(std::initializer_list<Vector> inputs,
                        std::initializer_list<Vector> targets,
                        const TrainingOptions& options = {});
    TrainingHistory fit(const DenseMatrix& inputs, const DenseMatrix& targets, const TrainingOptions& options = {});
    TrainingHistory fit(const Dataset& dataset, const TrainingOptions& options = {});
    TrainingHistory fit(StreamingDataset& dataset, const TrainingOptions& options = {});

    [[nodiscard]] std::vector<LayerParameters> parameters() const;
    void set_parameters(const std::vector<LayerParameters>& parameters);
    [[nodiscard]] CompiledModel compile() const;
    [[nodiscard]] InferenceModel to_inference_model() const;

    void save(const std::filesystem::path& path) const;
    void save_binary(const std::filesystem::path& path, const BinaryMetadata& metadata = {}) const;
    void save_checkpoint(const std::filesystem::path& path,
                         const TrainingCheckpointInfo& info = {}) const;

private:
    struct Impl;

    explicit TrainingModel(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

using Model = TrainingModel;
using MutableModel = TrainingModel;

} // namespace brutal_mlp

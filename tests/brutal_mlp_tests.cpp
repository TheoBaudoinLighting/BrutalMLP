#include "brutal_mlp/brutal_mlp.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace bm = brutal_mlp;

namespace {

constexpr double kTightTolerance = sizeof(bm::Scalar) == sizeof(float) ? 1e-5 : 1e-12;
constexpr bm::Scalar kGradientStep = sizeof(bm::Scalar) == sizeof(float) ? bm::Scalar{1e-2f} : bm::Scalar{1e-5};
constexpr bm::Scalar kGradientLearningRate =
    sizeof(bm::Scalar) == sizeof(float) ? bm::Scalar{1e-2f} : bm::Scalar{1e-4};
constexpr double kGradientAbsTolerance = sizeof(bm::Scalar) == sizeof(float) ? 6e-3 : 1e-6;
constexpr double kGradientRelTolerance = sizeof(bm::Scalar) == sizeof(float) ? 6e-2 : 1e-5;

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocation_count{0};
std::atomic<std::size_t> g_temp_path_counter{0};

std::filesystem::path temp_test_path(const char* filename) {
    const std::filesystem::path base(filename);
    const auto tick_count = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::size_t counter = g_temp_path_counter.fetch_add(1, std::memory_order_relaxed);
    const std::string unique_name = base.stem().string() + "_s" +
                                    std::to_string(sizeof(bm::Scalar) * 8) + "_" +
                                    std::to_string(tick_count) + "_" +
                                    std::to_string(counter) +
                                    base.extension().string();
    return std::filesystem::temp_directory_path() / unique_name;
}

void expect_vector_near(const bm::Vector& actual, const bm::Vector& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
    }
}

void expect_matrix_near(const bm::Matrix& actual, const bm::Matrix& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t row = 0; row < actual.size(); ++row) {
        SCOPED_TRACE("row " + std::to_string(row));
        expect_vector_near(actual[row], expected[row], tolerance);
    }
}

void expect_parameters_near(const std::vector<bm::LayerParameters>& actual,
                            const std::vector<bm::LayerParameters>& expected,
                            double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t layer = 0; layer < actual.size(); ++layer) {
        EXPECT_EQ(actual[layer].input_size, expected[layer].input_size);
        EXPECT_EQ(actual[layer].output_size, expected[layer].output_size);
        EXPECT_EQ(actual[layer].activation, expected[layer].activation);
        EXPECT_NEAR(actual[layer].dropout_probability, expected[layer].dropout_probability, tolerance);
        expect_vector_near(actual[layer].weights, expected[layer].weights, tolerance);
        expect_vector_near(actual[layer].biases, expected[layer].biases, tolerance);
    }
}

void expect_parameters_finite(const std::vector<bm::LayerParameters>& parameters) {
    for (std::size_t layer = 0; layer < parameters.size(); ++layer) {
        for (std::size_t i = 0; i < parameters[layer].weights.size(); ++i) {
            EXPECT_TRUE(std::isfinite(parameters[layer].weights[i])) << "weight layer " << layer << " index " << i;
        }
        for (std::size_t i = 0; i < parameters[layer].biases.size(); ++i) {
            EXPECT_TRUE(std::isfinite(parameters[layer].biases[i])) << "bias layer " << layer << " index " << i;
        }
    }
}

void expect_history_finite(const bm::TrainingHistory& history) {
    ASSERT_FALSE(history.training_loss.empty());
    for (std::size_t i = 0; i < history.training_loss.size(); ++i) {
        EXPECT_TRUE(std::isfinite(history.training_loss[i])) << "training_loss index " << i;
    }
    for (std::size_t i = 0; i < history.validation_loss.size(); ++i) {
        EXPECT_TRUE(std::isfinite(history.validation_loss[i])) << "validation_loss index " << i;
    }
    for (std::size_t i = 0; i < history.test_loss.size(); ++i) {
        EXPECT_TRUE(std::isfinite(history.test_loss[i])) << "test_loss index " << i;
    }
    for (const auto& epoch : history.epochs) {
        EXPECT_TRUE(epoch.finite) << "epoch " << epoch.epoch;
        EXPECT_EQ(epoch.non_finite_parameter_count, 0U) << "epoch " << epoch.epoch;
        EXPECT_TRUE(std::isfinite(epoch.training_loss)) << "epoch " << epoch.epoch;
        EXPECT_TRUE(std::isfinite(epoch.gradient_norm)) << "epoch " << epoch.epoch;
        EXPECT_TRUE(epoch.weights.finite) << "epoch " << epoch.epoch;
        EXPECT_EQ(epoch.weights.non_finite_count, 0U) << "epoch " << epoch.epoch;
    }
}

bm::Model make_known_linear_model() {
    auto model = bm::Model::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .seed(7)
                     .build();

    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {2.0, -1.0}, {0.5}},
    });
    return model;
}

bm::TrainingModel make_known_two_layer_model() {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(3, bm::Activation::relu)
                     .add_layer(2, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .seed(9)
                     .build();

    model.set_parameters({
        bm::LayerParameters{2, 3, bm::Activation::relu,
                            {1.0, -1.0,
                             -0.5, 2.0,
                             0.25, 0.5},
                            {0.0, 0.25, -0.75}},
        bm::LayerParameters{3, 2, bm::Activation::linear,
                            {1.0, 0.5, -1.0,
                             -2.0, 0.25, 0.5},
                            {0.1, -0.2}},
    });
    return model;
}

bm::TrainingModel make_known_two_output_model(bm::LossConfig loss = bm::LossConfig::mean_squared_error()) {
    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(2, bm::Activation::linear)
                     .loss(std::move(loss))
                     .seed(13)
                     .build();

    model.set_parameters({
        bm::LayerParameters{1, 2, bm::Activation::linear, {bm::Scalar{1}, bm::Scalar{2}}, {bm::Scalar{0}, bm::Scalar{0}}},
    });
    return model;
}

bm::Scalar custom_mse_value(const bm::Scalar* prediction,
                            const bm::Scalar* target,
                            std::size_t size,
                            void*) {
    bm::Scalar total{0};
    for (std::size_t i = 0; i < size; ++i) {
        const bm::Scalar delta = prediction[i] - target[i];
        total += delta * delta;
    }
    return total / static_cast<bm::Scalar>(size);
}

void custom_mse_gradient(const bm::Scalar* prediction,
                         const bm::Scalar* target,
                         std::size_t size,
                         bm::Scalar* gradient,
                         void*) {
    const bm::Scalar scale = bm::Scalar{2} / static_cast<bm::Scalar>(size);
    for (std::size_t i = 0; i < size; ++i) {
        gradient[i] = (prediction[i] - target[i]) * scale;
    }
}

bm::Scalar custom_max_absolute_error(const bm::Matrix& predictions, const bm::Matrix& targets, void*) {
    bm::Scalar result{0};
    for (std::size_t row = 0; row < predictions.size(); ++row) {
        for (std::size_t column = 0; column < predictions[row].size(); ++column) {
            const bm::Scalar absolute_error =
                static_cast<bm::Scalar>(std::abs(predictions[row][column] - targets[row][column]));
            result = std::max(result, absolute_error);
        }
    }
    return result;
}

void generated_linear_sample(std::size_t index,
                             bm::Scalar* input,
                             std::size_t input_size,
                             bm::Scalar* target,
                             std::size_t target_size,
                             void*) {
    ASSERT_EQ(input_size, 1U);
    ASSERT_EQ(target_size, 1U);
    input[0] = static_cast<bm::Scalar>(static_cast<double>(index) - 4.0);
    target[0] = bm::Scalar{2} * input[0] + bm::Scalar{1};
}

struct StreamingLinearContext {
    std::size_t index{0};
    std::size_t count{0};
};

void reset_streaming_linear(void* context) {
    static_cast<StreamingLinearContext*>(context)->index = 0;
}

bool next_streaming_linear(bm::Scalar* input,
                           std::size_t input_size,
                           bm::Scalar* target,
                           std::size_t target_size,
                           void* context) {
    EXPECT_EQ(input_size, 1U);
    EXPECT_EQ(target_size, 1U);
    auto* state = static_cast<StreamingLinearContext*>(context);
    if (state->index >= state->count) {
        return false;
    }
    input[0] = static_cast<bm::Scalar>(static_cast<double>(state->index) - 4.0);
    target[0] = bm::Scalar{2} * input[0] + bm::Scalar{1};
    ++state->index;
    return true;
}

bm::TrainingOptions quick_options(std::size_t epochs, std::size_t batch_size) {
    bm::TrainingOptions options;
    options.epochs = epochs;
    options.batch_size = batch_size;
    options.shuffle = true;
    options.seed = 1234;
    options.restore_best_weights = true;
    return options;
}

bm::TrainingOptions one_step_options(std::size_t batch_size) {
    bm::TrainingOptions options;
    options.epochs = 1;
    options.batch_size = batch_size;
    options.shuffle = false;
    options.restore_best_weights = false;
    return options;
}

bm::TrainingModel make_learning_rate_test_model(bm::Scalar learning_rate = bm::Scalar{0.1}) {
    return bm::TrainingModel::builder()
        .input_size(1)
        .add_layer(1, bm::Activation::linear)
        .loss(bm::Loss::mean_squared_error)
        .optimizer(bm::OptimizerConfig::sgd(learning_rate))
        .seed(81)
        .build();
}

double mean_of(const bm::Vector& values) {
    if (values.empty()) {
        throw std::invalid_argument("values must not be empty");
    }
    const double total = std::accumulate(values.begin(), values.end(), 0.0);
    return total / static_cast<double>(values.size());
}

double variance_of(const bm::Vector& values) {
    if (values.size() <= 1) {
        throw std::invalid_argument("values must contain at least two entries");
    }
    const double mean = mean_of(values);
    double total = 0.0;
    for (bm::Scalar value : values) {
        const double delta = static_cast<double>(value) - mean;
        total += delta * delta;
    }
    return total / static_cast<double>(values.size());
}

double max_abs_of(const bm::Vector& values) {
    double result = 0.0;
    for (bm::Scalar value : values) {
        result = std::max(result, std::abs(static_cast<double>(value)));
    }
    return result;
}

std::vector<bm::LayerParameters> analytic_gradients_from_sgd_step(const bm::TrainingModel& base_model,
                                                                  const std::vector<bm::LayerParameters>& parameters,
                                                                  const bm::Matrix& inputs,
                                                                  const bm::Matrix& targets,
                                                                  bm::Scalar learning_rate) {
    auto model = base_model;
    model.set_parameters(parameters);
    model.fit(inputs, targets, one_step_options(inputs.size()));
    const auto updated = model.parameters();

    auto gradients = parameters;
    for (std::size_t layer = 0; layer < parameters.size(); ++layer) {
        for (std::size_t i = 0; i < parameters[layer].weights.size(); ++i) {
            gradients[layer].weights[i] = (parameters[layer].weights[i] - updated[layer].weights[i]) / learning_rate;
        }
        for (std::size_t i = 0; i < parameters[layer].biases.size(); ++i) {
            gradients[layer].biases[i] = (parameters[layer].biases[i] - updated[layer].biases[i]) / learning_rate;
        }
    }
    return gradients;
}

bm::Scalar evaluate_with_parameters(const bm::TrainingModel& base_model,
                                    const std::vector<bm::LayerParameters>& parameters,
                                    const bm::Matrix& inputs,
                                    const bm::Matrix& targets) {
    auto model = base_model;
    model.set_parameters(parameters);
    return model.evaluate_loss(inputs, targets);
}

std::vector<bm::LayerParameters> numerical_gradients(const bm::TrainingModel& base_model,
                                                     const std::vector<bm::LayerParameters>& parameters,
                                                     const bm::Matrix& inputs,
                                                     const bm::Matrix& targets,
                                                     bm::Scalar epsilon) {
    auto gradients = parameters;
    for (std::size_t layer = 0; layer < parameters.size(); ++layer) {
        for (std::size_t i = 0; i < parameters[layer].weights.size(); ++i) {
            auto plus = parameters;
            auto minus = parameters;
            plus[layer].weights[i] += epsilon;
            minus[layer].weights[i] -= epsilon;
            const bm::Scalar loss_plus = evaluate_with_parameters(base_model, plus, inputs, targets);
            const bm::Scalar loss_minus = evaluate_with_parameters(base_model, minus, inputs, targets);
            gradients[layer].weights[i] = (loss_plus - loss_minus) / (bm::Scalar{2} * epsilon);
        }

        for (std::size_t i = 0; i < parameters[layer].biases.size(); ++i) {
            auto plus = parameters;
            auto minus = parameters;
            plus[layer].biases[i] += epsilon;
            minus[layer].biases[i] -= epsilon;
            const bm::Scalar loss_plus = evaluate_with_parameters(base_model, plus, inputs, targets);
            const bm::Scalar loss_minus = evaluate_with_parameters(base_model, minus, inputs, targets);
            gradients[layer].biases[i] = (loss_plus - loss_minus) / (bm::Scalar{2} * epsilon);
        }
    }
    return gradients;
}

void expect_gradient_near(bm::Scalar analytic,
                          bm::Scalar numerical,
                          const char* parameter_kind,
                          std::size_t layer,
                          std::size_t index) {
    const double a = static_cast<double>(analytic);
    const double n = static_cast<double>(numerical);
    const double scale = std::max({1.0, std::abs(a), std::abs(n)});
    EXPECT_LE(std::abs(a - n), kGradientAbsTolerance + kGradientRelTolerance * scale)
        << parameter_kind << " layer " << layer << " index " << index
        << " analytic=" << a << " numerical=" << n;
}

void expect_gradients_match(const std::vector<bm::LayerParameters>& analytic,
                            const std::vector<bm::LayerParameters>& numerical) {
    ASSERT_EQ(analytic.size(), numerical.size());
    for (std::size_t layer = 0; layer < analytic.size(); ++layer) {
        ASSERT_EQ(analytic[layer].weights.size(), numerical[layer].weights.size());
        ASSERT_EQ(analytic[layer].biases.size(), numerical[layer].biases.size());
        for (std::size_t i = 0; i < analytic[layer].weights.size(); ++i) {
            expect_gradient_near(analytic[layer].weights[i], numerical[layer].weights[i], "weight", layer, i);
        }
        for (std::size_t i = 0; i < analytic[layer].biases.size(); ++i) {
            expect_gradient_near(analytic[layer].biases[i], numerical[layer].biases[i], "bias", layer, i);
        }
    }
}

void expect_backprop_matches_finite_difference(const bm::TrainingModel& model,
                                               const std::vector<bm::LayerParameters>& parameters,
                                               const bm::Matrix& inputs,
                                               const bm::Matrix& targets,
                                               bm::Scalar learning_rate) {
    const auto analytic = analytic_gradients_from_sgd_step(model, parameters, inputs, targets, learning_rate);
    const auto numerical = numerical_gradients(model, parameters, inputs, targets, kGradientStep);
    expect_gradients_match(analytic, numerical);
}

void expect_output_loss_backprop_matches_finite_difference(const bm::LossConfig& loss) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::linear)
                     .loss(loss)
                     .optimizer(bm::OptimizerConfig::sgd(kGradientLearningRate, bm::Scalar{0}))
                     .seed(105)
                     .build();

    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 2, bm::Activation::linear,
                            {bm::Scalar{0.5}, bm::Scalar{-0.25},
                             bm::Scalar{-0.3}, bm::Scalar{0.4}},
                            {bm::Scalar{0.1}, bm::Scalar{-0.2}}},
    };
    const bm::Matrix inputs{{bm::Scalar{0.2}, bm::Scalar{-0.4}},
                            {bm::Scalar{-0.3}, bm::Scalar{0.7}}};
    const bm::Matrix targets{{bm::Scalar{-0.5}, bm::Scalar{0.8}},
                             {bm::Scalar{0.6}, bm::Scalar{-0.9}}};

    expect_backprop_matches_finite_difference(model, parameters, inputs, targets, kGradientLearningRate);
}

} // namespace

void* operator new(std::size_t size) {
    if (g_count_allocations.load(std::memory_order_relaxed)) {
        g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    if (g_count_allocations.load(std::memory_order_relaxed)) {
        g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

TEST(ApiShape, ModelIsLegacyAliasForTrainingModel) {
    static_assert(std::is_same<bm::Model, bm::TrainingModel>::value,
                  "Model should remain a source-compatible alias for TrainingModel");
    static_assert(std::is_same<bm::MutableModel, bm::TrainingModel>::value,
                  "MutableModel should name the flexible training model");
    static_assert(std::is_base_of<bm::InferenceModel, bm::CompiledModel>::value,
                  "CompiledModel should keep the frozen inference API surface");
#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
    static_assert(std::is_same<bm::Scalar, double>::value, "BRUTAL_MLP_USE_DOUBLE should select double");
#else
    static_assert(std::is_same<bm::Scalar, float>::value, "Default Scalar should be float");
#endif
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_to(nullptr, 0, nullptr, 0, nullptr, 0)),
                  "InferenceModel::predict_to must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_to(
                      nullptr, 0, std::declval<bm::InferenceWorkspace&>())),
                  "InferenceModel::predict_to(workspace) must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_batch_to(
                      nullptr, 0, 0, nullptr, 0, nullptr, 0)),
                  "InferenceModel::predict_batch_to must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_batch_to(
                      nullptr, 0, 0, nullptr, 0, nullptr, 0, std::declval<const bm::ParallelOptions&>())),
                  "InferenceModel::predict_batch_to(parallelism) must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().batch_scratch_size(
                      0, std::declval<const bm::ParallelOptions&>())),
                  "InferenceModel::batch_scratch_size must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_unchecked_to(nullptr, nullptr, nullptr)),
                  "InferenceModel::predict_unchecked_to must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_unchecked_to(
                      nullptr, std::declval<bm::InferenceWorkspace&>())),
                  "InferenceModel::predict_unchecked_to(workspace) must remain noexcept");
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_batch_unchecked_to(
                      nullptr, 0, 0, nullptr, 0, nullptr)),
                  "InferenceModel::predict_batch_unchecked_to must remain noexcept");
}

TEST(StringConversions, RoundTripSupportedEnums) {
    EXPECT_EQ(bm::activation_from_string(bm::to_string(bm::Activation::linear)), bm::Activation::linear);
    EXPECT_EQ(bm::activation_from_string(bm::to_string(bm::Activation::relu)), bm::Activation::relu);
    EXPECT_EQ(bm::activation_from_string(bm::to_string(bm::Activation::sigmoid)), bm::Activation::sigmoid);
    EXPECT_EQ(bm::activation_from_string(bm::to_string(bm::Activation::tanh)), bm::Activation::tanh);
    EXPECT_EQ(bm::activation_from_string(bm::to_string(bm::Activation::softmax)), bm::Activation::softmax);

    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::mean_squared_error)),
              bm::Loss::mean_squared_error);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::mean_absolute_error)),
              bm::Loss::mean_absolute_error);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::huber)), bm::Loss::huber);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::relative_mean_squared_error)),
              bm::Loss::relative_mean_squared_error);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::log_cosh)), bm::Loss::log_cosh);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::weighted_mean_squared_error)),
              bm::Loss::weighted_mean_squared_error);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::binary_cross_entropy)),
              bm::Loss::binary_cross_entropy);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::categorical_cross_entropy)),
              bm::Loss::categorical_cross_entropy);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::custom)), bm::Loss::custom);

    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::mean_squared_error)),
              bm::Metric::mean_squared_error);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::mean_absolute_error)),
              bm::Metric::mean_absolute_error);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::root_mean_squared_error)),
              bm::Metric::root_mean_squared_error);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::r2_score)), bm::Metric::r2_score);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::accuracy)), bm::Metric::accuracy);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::precision)), bm::Metric::precision);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::recall)), bm::Metric::recall);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::f1_score)), bm::Metric::f1_score);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::confusion_matrix)),
              bm::Metric::confusion_matrix);
    EXPECT_EQ(bm::metric_from_string(bm::to_string(bm::Metric::custom)), bm::Metric::custom);

    EXPECT_EQ(bm::averaging_from_string(bm::to_string(bm::Averaging::binary)), bm::Averaging::binary);
    EXPECT_EQ(bm::averaging_from_string(bm::to_string(bm::Averaging::macro)), bm::Averaging::macro);
    EXPECT_EQ(bm::averaging_from_string(bm::to_string(bm::Averaging::micro)), bm::Averaging::micro);

    EXPECT_EQ(bm::optimizer_type_from_string(bm::to_string(bm::OptimizerType::sgd)), bm::OptimizerType::sgd);
    EXPECT_EQ(bm::optimizer_type_from_string(bm::to_string(bm::OptimizerType::adam)), bm::OptimizerType::adam);
    EXPECT_EQ(bm::optimizer_type_from_string(bm::to_string(bm::OptimizerType::adamw)), bm::OptimizerType::adamw);

    EXPECT_EQ(bm::to_string(bm::InferenceStatus::ok), "ok");
    EXPECT_EQ(bm::to_string(bm::InferenceStatus::invalid_input_stride), "invalid_input_stride");
    EXPECT_EQ(bm::to_string(bm::InferenceStatus::invalid_output_stride), "invalid_output_stride");
    EXPECT_EQ(bm::to_string(bm::InferenceStatus::insufficient_scratch), "insufficient_scratch");

    EXPECT_EQ(bm::to_string(bm::TrainingMonitor::automatic), "automatic");
    EXPECT_EQ(bm::to_string(bm::TrainingMonitor::training_loss), "training_loss");
    EXPECT_EQ(bm::to_string(bm::TrainingMonitor::validation_loss), "validation_loss");
    EXPECT_EQ(bm::to_string(bm::TrainingMonitor::test_loss), "test_loss");
    EXPECT_EQ(bm::to_string(bm::TrainingMonitorMode::minimize), "minimize");
    EXPECT_EQ(bm::to_string(bm::TrainingMonitorMode::maximize), "maximize");
    EXPECT_EQ(bm::to_string(bm::LearningRateSchedule::constant), "constant");
    EXPECT_EQ(bm::to_string(bm::LearningRateSchedule::step_decay), "step_decay");
    EXPECT_EQ(bm::to_string(bm::LearningRateSchedule::exponential_decay), "exponential_decay");
    EXPECT_EQ(bm::to_string(bm::LearningRateSchedule::cosine_annealing), "cosine_annealing");
    EXPECT_EQ(bm::to_string(bm::LearningRateSchedule::reduce_on_plateau), "reduce_on_plateau");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::automatic), "automatic");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::he_normal), "he_normal");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::he_uniform), "he_uniform");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::xavier_normal), "xavier_normal");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::xavier_uniform), "xavier_uniform");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::lecun_normal), "lecun_normal");
    EXPECT_EQ(bm::to_string(bm::WeightInitialization::lecun_uniform), "lecun_uniform");

    EXPECT_EQ(bm::parallel_execution_from_string(bm::to_string(bm::ParallelExecution::serial)),
              bm::ParallelExecution::serial);
    EXPECT_EQ(bm::parallel_execution_from_string(bm::to_string(bm::ParallelExecution::worker_threads)),
              bm::ParallelExecution::worker_threads);
}

TEST(StringConversions, RejectUnknownValues) {
    EXPECT_THROW((void)bm::activation_from_string("swish"), std::invalid_argument);
    EXPECT_THROW((void)bm::loss_from_string("hinge"), std::invalid_argument);
    EXPECT_THROW((void)bm::metric_from_string("balanced_accuracy"), std::invalid_argument);
    EXPECT_THROW((void)bm::averaging_from_string("weighted"), std::invalid_argument);
    EXPECT_THROW((void)bm::optimizer_type_from_string("rmsprop"), std::invalid_argument);
    EXPECT_THROW((void)bm::parallel_execution_from_string("task_graph"), std::invalid_argument);
}

TEST(OptimizerConfig, FactoriesExposeExpectedDefaults) {
    const auto adam = bm::OptimizerConfig::adam(0.02);
    EXPECT_EQ(adam.type, bm::OptimizerType::adam);
    EXPECT_NEAR(adam.learning_rate, 0.02, kTightTolerance);
    EXPECT_NEAR(adam.beta1, 0.9, kTightTolerance);
    EXPECT_NEAR(adam.beta2, 0.999, kTightTolerance);
    EXPECT_EQ(adam.gradient_clip_value, bm::Scalar{0});
    EXPECT_EQ(adam.layer_gradient_clip_norm, bm::Scalar{0});

    const auto adamw = bm::OptimizerConfig::adamw(0.01, 0.03);
    EXPECT_EQ(adamw.type, bm::OptimizerType::adamw);
    EXPECT_NEAR(adamw.learning_rate, 0.01, kTightTolerance);
    EXPECT_NEAR(adamw.decoupled_weight_decay, 0.03, kTightTolerance);

    const auto sgd = bm::OptimizerConfig::sgd(0.3, 0.4);
    EXPECT_EQ(sgd.type, bm::OptimizerType::sgd);
    EXPECT_NEAR(sgd.learning_rate, 0.3, kTightTolerance);
    EXPECT_NEAR(sgd.momentum, 0.4, kTightTolerance);

    const auto step = bm::LearningRateScheduleConfig::step_decay(3, bm::Scalar{0.25});
    EXPECT_EQ(step.type, bm::LearningRateSchedule::step_decay);
    EXPECT_EQ(step.step_size, 3U);
    EXPECT_NEAR(step.step_decay_factor, 0.25, kTightTolerance);

    const auto exponential = bm::LearningRateScheduleConfig::exponential_decay(bm::Scalar{0.9});
    EXPECT_EQ(exponential.type, bm::LearningRateSchedule::exponential_decay);
    EXPECT_NEAR(exponential.exponential_decay_rate, 0.9, kTightTolerance);

    const auto cosine = bm::LearningRateScheduleConfig::cosine_annealing(12, bm::Scalar{0.001});
    EXPECT_EQ(cosine.type, bm::LearningRateSchedule::cosine_annealing);
    EXPECT_EQ(cosine.cosine_epochs, 12U);
    EXPECT_NEAR(cosine.minimum_learning_rate, 0.001, kTightTolerance);

    const auto plateau = bm::LearningRateScheduleConfig::reduce_on_plateau(4, bm::Scalar{0.2});
    EXPECT_EQ(plateau.type, bm::LearningRateSchedule::reduce_on_plateau);
    EXPECT_EQ(plateau.reduce_on_plateau_patience, 4U);
    EXPECT_NEAR(plateau.reduce_on_plateau_factor, 0.2, kTightTolerance);

    const auto he = bm::InitializationConfig::he_normal(bm::Scalar{0.25});
    EXPECT_EQ(he.weights, bm::WeightInitialization::he_normal);
    EXPECT_NEAR(he.bias, 0.25, kTightTolerance);

    const auto xavier = bm::InitializationConfig::xavier_uniform();
    EXPECT_EQ(xavier.weights, bm::WeightInitialization::xavier_uniform);
    EXPECT_NEAR(xavier.bias, 0.0, kTightTolerance);

    const auto lecun = bm::InitializationConfig::lecun_normal(bm::Scalar{-0.1});
    EXPECT_EQ(lecun.weights, bm::WeightInitialization::lecun_normal);
    EXPECT_NEAR(lecun.bias, -0.1, kTightTolerance);
}

TEST(BuilderValidation, RejectsInvalidTopology) {
    EXPECT_THROW((void)bm::Model::builder().build(), std::invalid_argument);
    EXPECT_THROW((void)bm::Model::builder().input_size(0).add_layer(1, bm::Activation::linear).build(),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::Model::builder().input_size(2).add_layer(0, bm::Activation::linear).build(),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::softmax)
                     .add_layer(1, bm::Activation::linear)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::sigmoid)
                     .loss(bm::Loss::categorical_cross_entropy)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::binary_cross_entropy)
                     .build(),
                 std::invalid_argument);
}

TEST(BuilderValidation, RejectsInvalidOptimizerConfiguration) {
    auto invalid = bm::OptimizerConfig::adam(-0.1);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::sgd(0.1, 1.0);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::adam(0.1);
    invalid.l2 = -0.01;
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::adam(0.1);
    invalid.l1 = -0.01;
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::adamw(0.1, -0.01);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::sgd(0.1);
    invalid.max_norm = -1;
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::adam(0.1);
    invalid.gradient_clip_value = -1;
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);

    invalid = bm::OptimizerConfig::adam(0.1);
    invalid.layer_gradient_clip_norm = -1;
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(invalid)
                     .build(),
                 std::invalid_argument);
}

TEST(BuilderValidation, RejectsInvalidInitializationConfiguration) {
    auto invalid = bm::InitializationConfig::he_uniform(std::numeric_limits<bm::Scalar>::quiet_NaN());
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .initialization(invalid)
                     .build(),
                 std::invalid_argument);
}

TEST(BuilderValidation, RejectsInvalidDropoutConfiguration) {
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(2, bm::Activation::relu, bm::Scalar{-0.1})
                     .add_layer(1, bm::Activation::linear)
                     .build(),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::Model::builder()
                     .input_size(1)
                     .add_layer(2, bm::Activation::relu)
                     .add_layer(1, bm::Activation::linear, bm::Scalar{0.5})
                     .build(),
                 std::invalid_argument);
}

TEST(Initialization, SameSeedProducesIdenticalWeightsAndConstantBiases) {
    const auto build = [](std::uint64_t seed) {
        return bm::Model::builder()
            .input_size(16)
            .add_layer(32, bm::Activation::relu)
            .initialization(bm::InitializationConfig::he_normal(bm::Scalar{0.25}))
            .seed(seed)
            .build();
    };

    const auto first = build(123);
    const auto second = build(123);
    const auto different = build(124);

    const auto first_parameters = first.parameters();
    const auto second_parameters = second.parameters();
    const auto different_parameters = different.parameters();

    ASSERT_EQ(first_parameters.size(), 1U);
    EXPECT_EQ(first_parameters[0].weights, second_parameters[0].weights);
    EXPECT_EQ(first_parameters[0].biases, second_parameters[0].biases);
    EXPECT_NE(first_parameters[0].weights, different_parameters[0].weights);
    for (bm::Scalar bias : first_parameters[0].biases) {
        EXPECT_NEAR(bias, 0.25, kTightTolerance);
    }
}

TEST(Initialization, HeNormalStatisticsMatchFanIn) {
    constexpr std::size_t fan_in = 64;
    constexpr std::size_t fan_out = 512;
    const auto model = bm::Model::builder()
        .input_size(fan_in)
        .add_layer(fan_out, bm::Activation::relu)
        .initialization(bm::InitializationConfig::he_normal())
        .seed(200)
        .build();

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    const double expected_variance = 2.0 / static_cast<double>(fan_in);
    EXPECT_NEAR(mean_of(parameters[0].weights), 0.0, 0.006);
    EXPECT_NEAR(variance_of(parameters[0].weights), expected_variance, expected_variance * 0.08);
}

TEST(Initialization, XavierUniformStatisticsMatchFanInAndFanOut) {
    constexpr std::size_t fan_in = 64;
    constexpr std::size_t fan_out = 512;
    const auto model = bm::Model::builder()
        .input_size(fan_in)
        .add_layer(fan_out, bm::Activation::tanh)
        .initialization(bm::InitializationConfig::xavier_uniform())
        .seed(201)
        .build();

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    const double expected_variance = 2.0 / static_cast<double>(fan_in + fan_out);
    const double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
    EXPECT_NEAR(mean_of(parameters[0].weights), 0.0, 0.004);
    EXPECT_NEAR(variance_of(parameters[0].weights), expected_variance, expected_variance * 0.08);
    EXPECT_LE(max_abs_of(parameters[0].weights), limit + kTightTolerance);
}

TEST(Initialization, LeCunUniformStatisticsMatchFanIn) {
    constexpr std::size_t fan_in = 128;
    constexpr std::size_t fan_out = 512;
    const auto model = bm::Model::builder()
        .input_size(fan_in)
        .add_layer(fan_out, bm::Activation::linear)
        .initialization(bm::InitializationConfig::lecun_uniform())
        .seed(202)
        .build();

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    const double expected_variance = 1.0 / static_cast<double>(fan_in);
    const double limit = std::sqrt(3.0 / static_cast<double>(fan_in));
    EXPECT_NEAR(mean_of(parameters[0].weights), 0.0, 0.004);
    EXPECT_NEAR(variance_of(parameters[0].weights), expected_variance, expected_variance * 0.08);
    EXPECT_LE(max_abs_of(parameters[0].weights), limit + kTightTolerance);
}

TEST(Initialization, AutomaticKeepsActivationSpecificUniformRanges) {
    constexpr std::size_t input_size = 16;
    constexpr std::size_t hidden_size = 32;
    constexpr std::size_t output_size = 8;
    const auto model = bm::Model::builder()
        .input_size(input_size)
        .add_layer(hidden_size, bm::Activation::relu)
        .add_layer(output_size, bm::Activation::linear)
        .initialization(bm::InitializationConfig::automatic())
        .seed(203)
        .build();

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 2U);
    const double he_limit = std::sqrt(6.0 / static_cast<double>(input_size));
    const double xavier_limit = std::sqrt(6.0 / static_cast<double>(hidden_size + output_size));
    EXPECT_LE(max_abs_of(parameters[0].weights), he_limit + kTightTolerance);
    EXPECT_LE(max_abs_of(parameters[1].weights), xavier_limit + kTightTolerance);
}

TEST(Prediction, ComputesKnownLinearLayer) {
    const auto model = make_known_linear_model();
    const auto prediction = model.predict({3.0, 1.0});
    expect_vector_near(prediction, {5.5}, kTightTolerance);
}

TEST(Prediction, BatchPredictionMatchesIndividualPrediction) {
    const auto model = make_known_linear_model();
    const bm::Matrix inputs{{3.0, 1.0}, {0.0, 0.0}, {-1.0, 2.0}};
    const bm::Matrix batch = model.predict_batch(inputs);

    ASSERT_EQ(batch.size(), inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        expect_vector_near(batch[i], model.predict(inputs[i]), kTightTolerance);
    }
}

TEST(Prediction, RejectsInvalidInputs) {
    const auto model = make_known_linear_model();
    EXPECT_THROW((void)model.predict({1.0}), std::invalid_argument);
    EXPECT_THROW((void)model.predict({1.0, std::numeric_limits<bm::Scalar>::quiet_NaN()}), std::invalid_argument);
    EXPECT_THROW((void)model.predict_batch({{1.0}}), std::invalid_argument);
}

TEST(Activations, SigmoidIsStableForLargeMagnitudeInputs) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::sigmoid)
                     .loss(bm::Loss::binary_cross_entropy)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::sigmoid, {1.0}, {0.0}},
    });

    EXPECT_NEAR(model.predict({1000.0})[0], 1.0, kTightTolerance);
    EXPECT_NEAR(model.predict({-1000.0})[0], 0.0, kTightTolerance);
}

TEST(Activations, SoftmaxProducesProbabilityDistribution) {
    auto model = bm::Model::builder()
                     .input_size(3)
                     .add_layer(3, bm::Activation::softmax)
                     .loss(bm::Loss::categorical_cross_entropy)
                     .build();
    model.set_parameters({
        bm::LayerParameters{3, 3, bm::Activation::softmax,
                            {1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0},
                            {0.0, 0.0, 0.0}},
    });

    const auto prediction = model.predict({1.0, 2.0, 3.0});
    const double sum = std::accumulate(prediction.begin(), prediction.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, kTightTolerance);
    EXPECT_GT(prediction[2], prediction[1]);
    EXPECT_GT(prediction[1], prediction[0]);
}

TEST(Losses, MeanSquaredErrorMatchesKnownValue) {
    const auto model = make_known_linear_model();
    EXPECT_NEAR(model.evaluate_loss({{3.0, 1.0}}, {{4.5}}), 1.0, kTightTolerance);
}

TEST(Losses, MeanAbsoluteErrorMatchesKnownValue) {
    auto model = make_known_linear_model();
    model = bm::TrainingModel::builder()
                .input_size(2)
                .add_layer(1, bm::Activation::linear)
                .loss(bm::LossConfig::mean_absolute_error())
                .seed(7)
                .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{2}, bm::Scalar{-1}}, {bm::Scalar{0.5}}},
    });

    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{3}, bm::Scalar{1}}}, {{bm::Scalar{4.5}}}), 1.0, kTightTolerance);
}

TEST(Losses, HuberMatchesKnownValue) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::huber(bm::Scalar{0.5}))
                     .seed(7)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{2}, bm::Scalar{-1}}, {bm::Scalar{0.5}}},
    });

    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{3}, bm::Scalar{1}}}, {{bm::Scalar{4.5}}}), 0.375, kTightTolerance);
}

TEST(Losses, RelativeMeanSquaredErrorMatchesKnownValue) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::relative_mean_squared_error(bm::Scalar{0.01}))
                     .seed(7)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{2}, bm::Scalar{-1}}, {bm::Scalar{0.5}}},
    });

    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{3}, bm::Scalar{1}}}, {{bm::Scalar{4.5}}}),
                1.0 / (4.5 * 4.5),
                kTightTolerance);
}

TEST(Losses, LogCoshMatchesKnownValue) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::log_cosh())
                     .seed(7)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{2}, bm::Scalar{-1}}, {bm::Scalar{0.5}}},
    });

    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{3}, bm::Scalar{1}}}, {{bm::Scalar{4.5}}}),
                std::log(std::cosh(1.0)),
                kTightTolerance);
}

TEST(Losses, WeightedMeanSquaredErrorMatchesKnownValue) {
    const auto model = make_known_two_output_model(
        bm::LossConfig::weighted_mean_squared_error({bm::Scalar{1}, bm::Scalar{3}}));

    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{3}}}, {{bm::Scalar{1}, bm::Scalar{10}}}),
                13.0,
                kTightTolerance);
}

TEST(Losses, CustomLossCanMatchMeanSquaredError) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::custom(custom_mse_value, custom_mse_gradient))
                     .seed(7)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{2}, bm::Scalar{-1}}, {bm::Scalar{0.5}}},
    });

    EXPECT_EQ(model.loss(), bm::Loss::custom);
    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{3}, bm::Scalar{1}}}, {{bm::Scalar{4.5}}}),
                1.0,
                kTightTolerance);
}

TEST(Losses, BinaryCrossEntropyMatchesKnownValue) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::sigmoid)
                     .loss(bm::Loss::binary_cross_entropy)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::sigmoid, {0.0}, {0.0}},
    });

    EXPECT_NEAR(model.evaluate_loss({{99.0}}, {{1.0}}), std::log(2.0), kTightTolerance);
}

TEST(Losses, CategoricalCrossEntropyMatchesKnownValue) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(3, bm::Activation::softmax)
                     .loss(bm::Loss::categorical_cross_entropy)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 3, bm::Activation::softmax, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    });

    EXPECT_NEAR(model.evaluate_loss({{0.0}}, {{0.0, 1.0, 0.0}}), std::log(3.0), kTightTolerance);
}

TEST(Losses, RejectsInvalidTargetsForLossContracts) {
    auto binary = bm::Model::builder()
                      .input_size(1)
                      .add_layer(1, bm::Activation::sigmoid)
                      .loss(bm::Loss::binary_cross_entropy)
                      .build();
    EXPECT_THROW((void)binary.evaluate_loss({{0.0}}, {{2.0}}), std::invalid_argument);

    auto categorical = bm::Model::builder()
                           .input_size(1)
                           .add_layer(3, bm::Activation::softmax)
                           .loss(bm::Loss::categorical_cross_entropy)
                           .build();
    EXPECT_THROW((void)categorical.evaluate_loss({{0.0}}, {{0.0, 0.5, 0.0}}), std::invalid_argument);
    EXPECT_THROW((void)categorical.evaluate_loss({{0.0}}, {{0.0, -1.0, 2.0}}), std::invalid_argument);
}

TEST(Losses, RejectsInvalidLossConfigurations) {
    EXPECT_THROW((void)bm::LossConfig::huber(bm::Scalar{0}), std::invalid_argument);
    EXPECT_THROW((void)bm::LossConfig::relative_mean_squared_error(bm::Scalar{0}), std::invalid_argument);
    EXPECT_THROW((void)bm::LossConfig::weighted_mean_squared_error({}), std::invalid_argument);
    EXPECT_THROW((void)bm::LossConfig::weighted_mean_squared_error({bm::Scalar{0}, bm::Scalar{0}}),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::LossConfig::weighted_mean_squared_error({bm::Scalar{1}, bm::Scalar{-1}}),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::LossConfig::custom(nullptr, custom_mse_gradient), std::invalid_argument);
    EXPECT_THROW((void)bm::LossConfig::custom(custom_mse_value, nullptr), std::invalid_argument);
    EXPECT_THROW((void)bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::weighted_mean_squared_error),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::weighted_mean_squared_error({bm::Scalar{1}, bm::Scalar{1}}))
                     .build(),
                 std::invalid_argument);
}

TEST(Metrics, RegressionMetricsMatchKnownValues) {
    const bm::Matrix predictions{{bm::Scalar{2}, bm::Scalar{4}}, {bm::Scalar{6}, bm::Scalar{8}}};
    const bm::Matrix targets{{bm::Scalar{1}, bm::Scalar{5}}, {bm::Scalar{5}, bm::Scalar{10}}};

    const auto result = bm::evaluate_predictions(predictions, targets);

    EXPECT_TRUE(result.has_metric(bm::Metric::mean_squared_error));
    EXPECT_FALSE(result.has_metric(bm::Metric::confusion_matrix));
    EXPECT_NEAR(result.metric(bm::Metric::mean_squared_error), 1.75, kTightTolerance);
    EXPECT_NEAR(result.metric(bm::Metric::mean_absolute_error), 1.25, kTightTolerance);
    EXPECT_NEAR(result.metric(bm::Metric::root_mean_squared_error), std::sqrt(1.75), kTightTolerance);
    EXPECT_NEAR(result.metric(bm::Metric::r2_score), 1.0 - 7.0 / 40.75, kTightTolerance);
}

TEST(Metrics, R2HandlesConstantTargets) {
    EXPECT_NEAR(bm::evaluate_predictions({{bm::Scalar{2}}, {bm::Scalar{2}}},
                                         {{bm::Scalar{2}}, {bm::Scalar{2}}})
                    .metric(bm::Metric::r2_score),
                1.0,
                kTightTolerance);
    EXPECT_NEAR(bm::evaluate_predictions({{bm::Scalar{3}}, {bm::Scalar{2}}},
                                         {{bm::Scalar{2}}, {bm::Scalar{2}}})
                    .metric(bm::Metric::r2_score),
                0.0,
                kTightTolerance);
}

TEST(Metrics, BinaryClassificationMetricsAndConfusionMatrixMatchKnownValues) {
    const bm::Matrix predictions{{bm::Scalar{0.9}},
                                 {bm::Scalar{0.8}},
                                 {bm::Scalar{0.4}},
                                 {bm::Scalar{0.7}},
                                 {bm::Scalar{0.2}}};
    const bm::Matrix targets{{bm::Scalar{1}},
                             {bm::Scalar{0}},
                             {bm::Scalar{0}},
                             {bm::Scalar{1}},
                             {bm::Scalar{1}}};

    const auto result = bm::evaluate_predictions(predictions, targets, bm::EvaluationOptions::binary_classification());

    ASSERT_TRUE(result.has_confusion_matrix);
    ASSERT_EQ(result.confusion_matrix.class_count(), 2U);
    EXPECT_EQ(result.confusion_matrix.total(), 5U);
    EXPECT_EQ(result.confusion_matrix.counts[0][0], 1U);
    EXPECT_EQ(result.confusion_matrix.counts[0][1], 1U);
    EXPECT_EQ(result.confusion_matrix.counts[1][0], 1U);
    EXPECT_EQ(result.confusion_matrix.counts[1][1], 2U);

    EXPECT_NEAR(result.metric(bm::Metric::accuracy), 0.6, kTightTolerance);
    EXPECT_NEAR(result.metric(bm::Metric::precision), 2.0 / 3.0, kTightTolerance);
    EXPECT_NEAR(result.metric(bm::Metric::recall), 2.0 / 3.0, kTightTolerance);
    EXPECT_NEAR(result.metric(bm::Metric::f1_score), 2.0 / 3.0, kTightTolerance);
}

TEST(Metrics, MulticlassMacroAndMicroMetricsMatchKnownValues) {
    const bm::Matrix predictions{{bm::Scalar{0.9}, bm::Scalar{0.1}, bm::Scalar{0.0}},
                                 {bm::Scalar{0.1}, bm::Scalar{0.2}, bm::Scalar{0.7}},
                                 {bm::Scalar{0.1}, bm::Scalar{0.3}, bm::Scalar{0.6}},
                                 {bm::Scalar{0.1}, bm::Scalar{0.8}, bm::Scalar{0.1}}};
    const bm::Matrix targets{{bm::Scalar{1}, bm::Scalar{0}, bm::Scalar{0}},
                             {bm::Scalar{0}, bm::Scalar{1}, bm::Scalar{0}},
                             {bm::Scalar{0}, bm::Scalar{0}, bm::Scalar{1}},
                             {bm::Scalar{0}, bm::Scalar{0}, bm::Scalar{1}}};

    const auto macro = bm::evaluate_predictions(
        predictions, targets, bm::EvaluationOptions::multiclass_classification(bm::Averaging::macro));
    const auto micro = bm::evaluate_predictions(
        predictions, targets, bm::EvaluationOptions::multiclass_classification(bm::Averaging::micro));

    ASSERT_TRUE(macro.has_confusion_matrix);
    ASSERT_EQ(macro.confusion_matrix.class_count(), 3U);
    EXPECT_EQ(macro.confusion_matrix.counts[0][0], 1U);
    EXPECT_EQ(macro.confusion_matrix.counts[1][2], 1U);
    EXPECT_EQ(macro.confusion_matrix.counts[2][1], 1U);
    EXPECT_EQ(macro.confusion_matrix.counts[2][2], 1U);

    EXPECT_NEAR(macro.metric(bm::Metric::accuracy), 0.5, kTightTolerance);
    EXPECT_NEAR(macro.metric(bm::Metric::precision), 0.5, kTightTolerance);
    EXPECT_NEAR(macro.metric(bm::Metric::recall), 0.5, kTightTolerance);
    EXPECT_NEAR(macro.metric(bm::Metric::f1_score), 0.5, kTightTolerance);

    EXPECT_NEAR(micro.metric(bm::Metric::precision), 0.5, kTightTolerance);
    EXPECT_NEAR(micro.metric(bm::Metric::recall), 0.5, kTightTolerance);
    EXPECT_NEAR(micro.metric(bm::Metric::f1_score), 0.5, kTightTolerance);
}

TEST(Metrics, CustomMetricIsEvaluatedSeparately) {
    bm::EvaluationOptions options;
    options.add_custom_metric("max_absolute_error", custom_max_absolute_error);

    const auto result = bm::evaluate_predictions({{bm::Scalar{2}}, {bm::Scalar{5}}},
                                                 {{bm::Scalar{1}}, {bm::Scalar{9}}},
                                                 options);

    EXPECT_TRUE(result.has_custom_metric("max_absolute_error"));
    EXPECT_FALSE(result.has_metric(bm::Metric::mean_squared_error));
    EXPECT_NEAR(result.custom_metric("max_absolute_error"), 4.0, kTightTolerance);
    EXPECT_THROW((void)result.metric(bm::Metric::custom), std::invalid_argument);
    EXPECT_THROW((void)result.custom_metric("missing"), std::out_of_range);
}

TEST(Metrics, ModelEvaluationUsesPredictionsAndNormalization) {
    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{10}, bm::Scalar{20}},
        {bm::Scalar{2}, bm::Scalar{4}},
        {bm::Scalar{100}},
        {bm::Scalar{10}});

    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .normalization(normalization)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{1}, bm::Scalar{2}}, {bm::Scalar{0.5}}},
    });

    const bm::Matrix inputs{{bm::Scalar{12}, bm::Scalar{28}}};
    const bm::Matrix targets{{bm::Scalar{165}}};
    const auto training_metrics = model.evaluate_metrics(inputs, targets);
    const auto inference_metrics = model.to_inference_model().evaluate_metrics(inputs, targets);

    EXPECT_NEAR(training_metrics.metric(bm::Metric::mean_squared_error), 100.0, 1e-2);
    EXPECT_NEAR(inference_metrics.metric(bm::Metric::mean_squared_error), 100.0, 1e-2);
}

TEST(Metrics, RejectsInvalidMetricInputsAndOptions) {
    EXPECT_THROW((void)bm::evaluate_predictions({}, {}), std::invalid_argument);
    EXPECT_THROW((void)bm::evaluate_predictions({{bm::Scalar{1}}}, {{bm::Scalar{1}, bm::Scalar{2}}}),
                 std::invalid_argument);

    bm::EvaluationOptions duplicate;
    duplicate.metrics = {bm::Metric::accuracy, bm::Metric::accuracy};
    EXPECT_THROW((void)bm::evaluate_predictions({{bm::Scalar{1}}}, {{bm::Scalar{1}}}, duplicate),
                 std::invalid_argument);

    bm::EvaluationOptions invalid_custom_metric;
    EXPECT_THROW((void)invalid_custom_metric.add_custom_metric("", custom_max_absolute_error), std::invalid_argument);
    EXPECT_THROW((void)invalid_custom_metric.add_custom_metric("bad", nullptr), std::invalid_argument);

    bm::EvaluationOptions invalid_metric;
    EXPECT_THROW((void)invalid_metric.include(bm::Metric::custom), std::invalid_argument);

    bm::EvaluationOptions invalid_threshold = bm::EvaluationOptions::binary_classification();
    invalid_threshold.classification_threshold = bm::Scalar{2};
    EXPECT_THROW((void)bm::evaluate_predictions({{bm::Scalar{1}}}, {{bm::Scalar{1}}}, invalid_threshold),
                 std::invalid_argument);

    bm::EvaluationOptions invalid_positive_class = bm::EvaluationOptions::binary_classification();
    invalid_positive_class.positive_class = 2;
    EXPECT_THROW((void)bm::evaluate_predictions({{bm::Scalar{1}}}, {{bm::Scalar{1}}}, invalid_positive_class),
                 std::invalid_argument);
}

TEST(DatasetPipeline, SplitViewAndBatchGeneratorWork) {
    const bm::Matrix inputs{{bm::Scalar{0}}, {bm::Scalar{1}}, {bm::Scalar{2}}, {bm::Scalar{3}}, {bm::Scalar{4}},
                            {bm::Scalar{5}}, {bm::Scalar{6}}, {bm::Scalar{7}}, {bm::Scalar{8}}, {bm::Scalar{9}}};
    const bm::Matrix targets{{bm::Scalar{1}}, {bm::Scalar{3}}, {bm::Scalar{5}}, {bm::Scalar{7}}, {bm::Scalar{9}},
                             {bm::Scalar{11}}, {bm::Scalar{13}}, {bm::Scalar{15}}, {bm::Scalar{17}}, {bm::Scalar{19}}};
    const bm::MatrixDataset dataset(inputs, targets);

    bm::DatasetSplitOptions split_options;
    split_options.validation_split = bm::Scalar{0.2};
    split_options.test_split = bm::Scalar{0.2};
    split_options.shuffle = false;
    const auto split = bm::make_dataset_split(dataset.sample_count(), split_options);

    EXPECT_EQ(split.training_indices, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(split.validation_indices, (std::vector<std::size_t>{6, 7}));
    EXPECT_EQ(split.test_indices, (std::vector<std::size_t>{8, 9}));

    const bm::DatasetView test_view(dataset, split.test_indices);
    bm::Vector input(test_view.input_size(), bm::Scalar{0});
    bm::Vector target(test_view.output_size(), bm::Scalar{0});
    test_view.sample(1, input.data(), input.size(), target.data(), target.size());
    expect_vector_near(input, {bm::Scalar{9}}, kTightTolerance);
    expect_vector_near(target, {bm::Scalar{19}}, kTightTolerance);

    bm::BatchGenerator generator(dataset, 4, false, 0);
    bm::MiniBatch batch;
    ASSERT_TRUE(generator.next(batch));
    ASSERT_EQ(batch.size(), 4U);
    EXPECT_EQ(batch.sample_count, 4U);
    EXPECT_EQ(batch.input_size, 1U);
    EXPECT_EQ(batch.output_size, 1U);
    EXPECT_EQ(batch.inputs.size(), batch.size() * batch.input_size);
    EXPECT_EQ(batch.targets.size(), batch.size() * batch.output_size);
    expect_vector_near(batch.input(0), {bm::Scalar{0}}, kTightTolerance);
    expect_vector_near(batch.target(3), {bm::Scalar{7}}, kTightTolerance);
    ASSERT_TRUE(generator.next(batch));
    ASSERT_EQ(batch.size(), 4U);
    ASSERT_TRUE(generator.next(batch));
    ASSERT_EQ(batch.size(), 2U);
    EXPECT_FALSE(generator.next(batch));
}

TEST(DatasetPipeline, MiniBatchStoresRowsContiguously) {
    const bm::Matrix inputs{{bm::Scalar{0}, bm::Scalar{10}},
                            {bm::Scalar{1}, bm::Scalar{11}},
                            {bm::Scalar{2}, bm::Scalar{12}}};
    const bm::Matrix targets{{bm::Scalar{100}, bm::Scalar{200}},
                             {bm::Scalar{101}, bm::Scalar{201}},
                             {bm::Scalar{102}, bm::Scalar{202}}};
    const bm::MatrixDataset dataset(inputs, targets);

    bm::BatchGenerator generator(dataset, 2, false, 0);
    bm::MiniBatch batch;
    ASSERT_TRUE(generator.next(batch));

    EXPECT_EQ(batch.inputs, (bm::Vector{bm::Scalar{0}, bm::Scalar{10}, bm::Scalar{1}, bm::Scalar{11}}));
    EXPECT_EQ(batch.targets, (bm::Vector{bm::Scalar{100}, bm::Scalar{200}, bm::Scalar{101}, bm::Scalar{201}}));
    EXPECT_EQ(batch.input_data(1)[0], bm::Scalar{1});
    EXPECT_EQ(batch.input_data(1)[1], bm::Scalar{11});
    EXPECT_EQ(batch.target_data(1)[0], bm::Scalar{101});
    EXPECT_EQ(batch.target_data(1)[1], bm::Scalar{201});
    expect_vector_near(batch.input(1), {bm::Scalar{1}, bm::Scalar{11}}, kTightTolerance);
    expect_vector_near(batch.target(1), {bm::Scalar{101}, bm::Scalar{201}}, kTightTolerance);

    batch.clear();
    EXPECT_TRUE(batch.empty());
    EXPECT_TRUE(batch.inputs.empty());
    EXPECT_TRUE(batch.targets.empty());
    EXPECT_THROW((void)batch.input(0), std::invalid_argument);
}

TEST(DatasetPipeline, GeneratedDatasetCanTrainWithoutMatrixStorage) {
    bm::GeneratedDataset dataset(16, 1, 1, generated_linear_sample);

    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.05}))
                     .seed(501)
                     .build();

    auto options = quick_options(500, 4);
    options.validation_split = bm::Scalar{0.25};
    options.test_split = bm::Scalar{0.25};
    options.shuffle = true;
    const auto history = model.fit(dataset, options);

    ASSERT_FALSE(history.training_loss.empty());
    EXPECT_EQ(history.validation_loss.size(), history.training_loss.size());
    EXPECT_EQ(history.test_loss.size(), history.training_loss.size());
    EXPECT_NEAR(model.predict({bm::Scalar{3}})[0], 7.0, 0.15);
}

TEST(DatasetPipeline, StreamingFunctionDatasetCanTrainWithBoundedShuffleBuffer) {
    StreamingLinearContext context;
    context.count = 16;
    bm::FunctionStreamingDataset dataset(context.count, 1, 1, next_streaming_linear, reset_streaming_linear, &context);

    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.05}))
                     .seed(502)
                     .build();

    auto options = quick_options(450, 4);
    options.validation_split = bm::Scalar{0.25};
    options.test_split = bm::Scalar{0.25};
    options.streaming_shuffle_buffer_size = 3;
    const auto history = model.fit(dataset, options);

    ASSERT_FALSE(history.training_loss.empty());
    EXPECT_EQ(history.validation_loss.size(), history.training_loss.size());
    EXPECT_EQ(history.test_loss.size(), history.training_loss.size());
    EXPECT_NEAR(model.predict({bm::Scalar{2}})[0], 5.0, 0.2);
}

TEST(DatasetPipeline, CsvStreamingDatasetReadsFromDiskAndTrains) {
    const auto path = temp_test_path("brutal_mlp_streaming_dataset.csv");
    {
        std::ofstream file(path);
        file << "x,y\n";
        for (int i = -4; i <= 11; ++i) {
            file << i << ',' << (2 * i + 1) << '\n';
        }
    }

    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.05}))
                     .seed(503)
                     .build();

    {
        bm::CsvStreamingOptions csv_options;
        csv_options.has_header = true;
        bm::CsvStreamingDataset dataset(path, 1, 1, csv_options);
        EXPECT_EQ(dataset.sample_count(), 16U);

        bm::Vector input(dataset.input_size(), bm::Scalar{0});
        bm::Vector target(dataset.output_size(), bm::Scalar{0});
        ASSERT_TRUE(dataset.next(input.data(), input.size(), target.data(), target.size()));
        expect_vector_near(input, {bm::Scalar{-4}}, kTightTolerance);
        expect_vector_near(target, {bm::Scalar{-7}}, kTightTolerance);

        auto options = quick_options(450, 4);
        options.validation_split = bm::Scalar{0.25};
        options.test_split = bm::Scalar{0.25};
        options.streaming_shuffle_buffer_size = 4;
        model.fit(dataset, options);
    }
    std::filesystem::remove(path);

    EXPECT_NEAR(model.predict({bm::Scalar{4}})[0], 9.0, 0.2);
}

TEST(DatasetPipeline, RejectsInvalidDatasetsAndSplits) {
    EXPECT_THROW((void)bm::GeneratedDataset(0, 1, 1, generated_linear_sample), std::invalid_argument);
    EXPECT_THROW((void)bm::GeneratedDataset(1, 0, 1, generated_linear_sample), std::invalid_argument);
    EXPECT_THROW((void)bm::GeneratedDataset(1, 1, 1, nullptr), std::invalid_argument);

    bm::DatasetSplitOptions invalid_split;
    invalid_split.validation_split = bm::Scalar{0.7};
    invalid_split.test_split = bm::Scalar{0.4};
    EXPECT_THROW((void)bm::make_dataset_split(10, invalid_split), std::invalid_argument);

    const bm::Matrix inputs{{bm::Scalar{0}}, {bm::Scalar{1}}};
    const bm::Matrix targets{{bm::Scalar{1}}, {bm::Scalar{3}}};
    const bm::MatrixDataset dataset(inputs, targets);
    EXPECT_THROW((void)bm::DatasetView(dataset, {0, 2}), std::invalid_argument);
    EXPECT_THROW((void)bm::BatchGenerator(dataset, 0), std::invalid_argument);

    auto model = bm::TrainingModel::builder().input_size(2).add_layer(1, bm::Activation::linear).build();
    EXPECT_THROW((void)model.fit(dataset, one_step_options(1)), std::invalid_argument);
}

TEST(Normalization, StringConversionsAndFactoryValidation) {
    EXPECT_EQ(bm::normalization_mode_from_string(bm::to_string(bm::NormalizationMode::none)),
              bm::NormalizationMode::none);
    EXPECT_EQ(bm::normalization_mode_from_string(bm::to_string(bm::NormalizationMode::standard_score)),
              bm::NormalizationMode::standard_score);
    EXPECT_EQ(bm::normalization_mode_from_string(bm::to_string(bm::NormalizationMode::min_max)),
              bm::NormalizationMode::min_max);
    EXPECT_THROW((void)bm::normalization_mode_from_string("median_mad"), std::invalid_argument);

    EXPECT_THROW((void)bm::FeatureNormalization::standard_score(bm::Scalar{0}, bm::Scalar{0}),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::FeatureNormalization::min_max(bm::Scalar{1}, bm::Scalar{1}),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::NormalizationSpec::standard_score({bm::Scalar{0}}, {}), std::invalid_argument);
    EXPECT_THROW((void)bm::NormalizationSpec::min_max({bm::Scalar{0}}, {}), std::invalid_argument);
}

TEST(Normalization, StandardScoreInputAndOutputAreAppliedToPrediction) {
    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{10}, bm::Scalar{20}},
        {bm::Scalar{2}, bm::Scalar{4}},
        {bm::Scalar{100}},
        {bm::Scalar{10}});

    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .normalization(normalization)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{1}, bm::Scalar{2}}, {bm::Scalar{0.5}}},
    });

    EXPECT_TRUE(model.normalization().has_input_normalization());
    EXPECT_TRUE(model.normalization().has_output_normalization());
    EXPECT_NEAR(model.predict({bm::Scalar{12}, bm::Scalar{28}})[0], 155.0, 1e-3);
    EXPECT_NEAR(model.evaluate_loss({{bm::Scalar{12}, bm::Scalar{28}}}, {{bm::Scalar{165}}}), 100.0, 1e-2);

    const auto inference = model.to_inference_model();
    EXPECT_EQ(inference.scratch_size(), 2U);

    bm::Vector output(inference.output_size(), bm::Scalar{0});
    bm::Vector scratch(inference.scratch_size(), bm::Scalar{0});
    bm::Vector input{bm::Scalar{12}, bm::Scalar{28}};
    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    const auto status = inference.predict_to(input.data(),
                                             input.size(),
                                             output.data(),
                                             output.size(),
                                             scratch.data(),
                                             scratch.size());
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(status, bm::InferenceStatus::ok);
    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
    EXPECT_NEAR(output[0], 155.0, 1e-3);
}

TEST(Normalization, MinMaxClampIsControlledPerFeature) {
    bm::NormalizationSpec clamped;
    clamped.input_features = {bm::FeatureNormalization::min_max(bm::Scalar{0}, bm::Scalar{10})};

    auto clamped_model = bm::TrainingModel::builder()
                             .input_size(1)
                             .add_layer(1, bm::Activation::linear)
                             .normalization(clamped)
                             .build();
    clamped_model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{1}}, {bm::Scalar{0}}},
    });

    EXPECT_NEAR(clamped_model.predict({bm::Scalar{15}})[0], 1.0, kTightTolerance);
    EXPECT_NEAR(clamped_model.predict({bm::Scalar{-5}})[0], -1.0, kTightTolerance);

    bm::NormalizationSpec unclamped;
    unclamped.input_features = {
        bm::FeatureNormalization::min_max(bm::Scalar{0}, bm::Scalar{10}, bm::Scalar{-1}, bm::Scalar{1}, false)};

    auto unclamped_model = bm::TrainingModel::builder()
                               .input_size(1)
                               .add_layer(1, bm::Activation::linear)
                               .normalization(unclamped)
                               .build();
    unclamped_model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{1}}, {bm::Scalar{0}}},
    });

    EXPECT_NEAR(unclamped_model.predict({bm::Scalar{15}})[0], 2.0, kTightTolerance);
}

TEST(Normalization, FitUsesNormalizedTargetsAndPredictDenormalizesOutputs) {
    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{0}},
        {bm::Scalar{2}},
        {bm::Scalar{10}},
        {bm::Scalar{5}});

    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.05}))
                     .normalization(normalization)
                     .seed(77)
                     .build();

    const bm::Matrix inputs{{bm::Scalar{-4}}, {bm::Scalar{-2}}, {bm::Scalar{0}}, {bm::Scalar{2}}, {bm::Scalar{4}}};
    const bm::Matrix targets{{bm::Scalar{-10}}, {bm::Scalar{0}}, {bm::Scalar{10}}, {bm::Scalar{20}}, {bm::Scalar{30}}};

    auto options = quick_options(600, inputs.size());
    options.shuffle = false;
    model.fit(inputs, targets, options);

    EXPECT_NEAR(model.predict({bm::Scalar{6}})[0], 40.0, 0.75);
}

TEST(Normalization, SerializationPreservesTrainingAndInferenceNormalization) {
    const auto training_path = temp_test_path("brutal_mlp_training_norm.model");
    const auto inference_path = temp_test_path("brutal_mlp_inference_norm.model");

    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{10}, bm::Scalar{20}},
        {bm::Scalar{2}, bm::Scalar{4}},
        {bm::Scalar{100}},
        {bm::Scalar{10}});

    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .normalization(normalization)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{1}, bm::Scalar{2}}, {bm::Scalar{0.5}}},
    });

    model.save(training_path);
    const auto loaded_training = bm::TrainingModel::load(training_path);
    std::filesystem::remove(training_path);
    EXPECT_TRUE(loaded_training.normalization().has_input_normalization());
    EXPECT_TRUE(loaded_training.normalization().has_output_normalization());
    EXPECT_NEAR(loaded_training.predict({bm::Scalar{12}, bm::Scalar{28}})[0], 155.0, 1e-3);

    const auto inference = model.to_inference_model();
    inference.save(inference_path);
    const auto loaded_inference = bm::InferenceModel::load(inference_path);
    std::filesystem::remove(inference_path);
    EXPECT_TRUE(loaded_inference.normalization().has_input_normalization());
    EXPECT_TRUE(loaded_inference.normalization().has_output_normalization());
    EXPECT_NEAR(loaded_inference.predict({bm::Scalar{12}, bm::Scalar{28}})[0], 155.0, 1e-3);
}

TEST(Normalization, RejectsShapeMismatchAndOutputNormalizationForClassificationLosses) {
    bm::NormalizationSpec bad_input_shape;
    bad_input_shape.input_features = {bm::FeatureNormalization::standard_score(bm::Scalar{0}, bm::Scalar{1})};
    EXPECT_THROW((void)bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .normalization(bad_input_shape)
                     .build(),
                 std::invalid_argument);

    bm::NormalizationSpec bad_output_loss;
    bad_output_loss.output_features = {bm::FeatureNormalization::standard_score(bm::Scalar{0}, bm::Scalar{1})};
    EXPECT_THROW((void)bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::sigmoid)
                     .loss(bm::Loss::binary_cross_entropy)
                     .normalization(bad_output_loss)
                     .build(),
                 std::invalid_argument);

    EXPECT_NO_THROW((void)bm::TrainingModel::builder()
                        .input_size(1)
                        .add_layer(1, bm::Activation::linear)
                        .loss(bm::LossConfig::huber(bm::Scalar{1}))
                        .normalization(bad_output_loss)
                        .build());
}

TEST(GradientCheck, MeanSquaredErrorTanhLinearNetwork) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::tanh)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::sgd(kGradientLearningRate, bm::Scalar{0}))
                     .seed(101)
                     .build();

    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 2, bm::Activation::tanh,
                            {bm::Scalar{0.35}, bm::Scalar{-0.21},
                             bm::Scalar{0.17}, bm::Scalar{0.44}},
                            {bm::Scalar{0.08}, bm::Scalar{-0.11}}},
        bm::LayerParameters{2, 1, bm::Activation::linear,
                            {bm::Scalar{0.31}, bm::Scalar{-0.27}},
                            {bm::Scalar{0.06}}},
    };
    const bm::Matrix inputs{{bm::Scalar{0.2}, bm::Scalar{-0.4}},
                            {bm::Scalar{-0.3}, bm::Scalar{0.7}}};
    const bm::Matrix targets{{bm::Scalar{0.15}}, {bm::Scalar{-0.25}}};

    expect_backprop_matches_finite_difference(model, parameters, inputs, targets, kGradientLearningRate);
}

TEST(GradientCheck, BinaryCrossEntropySigmoidNetwork) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::tanh)
                     .add_layer(1, bm::Activation::sigmoid)
                     .loss(bm::Loss::binary_cross_entropy)
                     .optimizer(bm::OptimizerConfig::sgd(kGradientLearningRate, bm::Scalar{0}))
                     .seed(102)
                     .build();

    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 2, bm::Activation::tanh,
                            {bm::Scalar{0.22}, bm::Scalar{-0.37},
                             bm::Scalar{0.41}, bm::Scalar{0.19}},
                            {bm::Scalar{-0.03}, bm::Scalar{0.07}}},
        bm::LayerParameters{2, 1, bm::Activation::sigmoid,
                            {bm::Scalar{0.28}, bm::Scalar{-0.33}},
                            {bm::Scalar{0.04}}},
    };
    const bm::Matrix inputs{{bm::Scalar{0.6}, bm::Scalar{-0.2}},
                            {bm::Scalar{-0.5}, bm::Scalar{0.3}}};
    const bm::Matrix targets{{bm::Scalar{1}}, {bm::Scalar{0}}};

    expect_backprop_matches_finite_difference(model, parameters, inputs, targets, kGradientLearningRate);
}

TEST(GradientCheck, CategoricalCrossEntropySoftmaxNetwork) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::tanh)
                     .add_layer(3, bm::Activation::softmax)
                     .loss(bm::Loss::categorical_cross_entropy)
                     .optimizer(bm::OptimizerConfig::sgd(kGradientLearningRate, bm::Scalar{0}))
                     .seed(103)
                     .build();

    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 2, bm::Activation::tanh,
                            {bm::Scalar{0.19}, bm::Scalar{-0.24},
                             bm::Scalar{0.33}, bm::Scalar{0.12}},
                            {bm::Scalar{0.02}, bm::Scalar{-0.05}}},
        bm::LayerParameters{2, 3, bm::Activation::softmax,
                            {bm::Scalar{0.21}, bm::Scalar{-0.16},
                             bm::Scalar{-0.11}, bm::Scalar{0.26},
                             bm::Scalar{0.08}, bm::Scalar{0.14}},
                            {bm::Scalar{0.03}, bm::Scalar{-0.02}, bm::Scalar{0.01}}},
    };
    const bm::Matrix inputs{{bm::Scalar{0.4}, bm::Scalar{-0.1}},
                            {bm::Scalar{-0.2}, bm::Scalar{0.5}}};
    const bm::Matrix targets{{bm::Scalar{1}, bm::Scalar{0}, bm::Scalar{0}},
                             {bm::Scalar{0}, bm::Scalar{0}, bm::Scalar{1}}};

    expect_backprop_matches_finite_difference(model, parameters, inputs, targets, kGradientLearningRate);
}

TEST(GradientCheck, ReluNetworkAwayFromKinks) {
    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::relu)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::sgd(kGradientLearningRate, bm::Scalar{0}))
                     .seed(104)
                     .build();

    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 2, bm::Activation::relu,
                            {bm::Scalar{0.8}, bm::Scalar{-0.3},
                             bm::Scalar{-1.2}, bm::Scalar{0.4}},
                            {bm::Scalar{1.0}, bm::Scalar{-1.1}}},
        bm::LayerParameters{2, 1, bm::Activation::linear,
                            {bm::Scalar{0.25}, bm::Scalar{-0.45}},
                            {bm::Scalar{0.12}}},
    };
    const bm::Matrix inputs{{bm::Scalar{0.2}, bm::Scalar{-0.4}},
                            {bm::Scalar{-0.6}, bm::Scalar{0.3}}};
    const bm::Matrix targets{{bm::Scalar{0.35}}, {bm::Scalar{-0.15}}};

    expect_backprop_matches_finite_difference(model, parameters, inputs, targets, kGradientLearningRate);
}

TEST(GradientCheck, MeanAbsoluteErrorAwayFromKinks) {
    expect_output_loss_backprop_matches_finite_difference(bm::LossConfig::mean_absolute_error());
}

TEST(GradientCheck, HuberAwayFromKinks) {
    expect_output_loss_backprop_matches_finite_difference(bm::LossConfig::huber(bm::Scalar{0.5}));
}

TEST(GradientCheck, RelativeMeanSquaredError) {
    expect_output_loss_backprop_matches_finite_difference(
        bm::LossConfig::relative_mean_squared_error(bm::Scalar{0.1}));
}

TEST(GradientCheck, LogCosh) {
    expect_output_loss_backprop_matches_finite_difference(bm::LossConfig::log_cosh());
}

TEST(GradientCheck, WeightedMeanSquaredError) {
    expect_output_loss_backprop_matches_finite_difference(
        bm::LossConfig::weighted_mean_squared_error({bm::Scalar{1}, bm::Scalar{3}}));
}

TEST(GradientCheck, CustomLoss) {
    expect_output_loss_backprop_matches_finite_difference(
        bm::LossConfig::custom(custom_mse_value, custom_mse_gradient));
}

TEST(Training, LinearRegressionConvergesWithAdam) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::adam(0.05))
                     .seed(11)
                     .build();

    const bm::Matrix inputs{{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}};
    const bm::Matrix targets{{-5.0}, {-3.0}, {-1.0}, {1.0}, {3.0}};
    const double before = model.evaluate_loss(inputs, targets);

    auto options = quick_options(500, inputs.size());
    options.shuffle = false;
    const auto history = model.fit(inputs, targets, options);
    const double after = model.evaluate_loss(inputs, targets);

    ASSERT_FALSE(history.training_loss.empty());
    EXPECT_LT(after, before * 0.001);
    EXPECT_NEAR(model.predict({1.5})[0], 2.0, 0.05);
}

TEST(Training, XorConvergesWithHiddenLayer) {
    auto optimizer = bm::OptimizerConfig::adam(0.08);
    optimizer.gradient_clip_norm = 5.0;

    auto model = bm::Model::builder()
                     .input_size(2)
                     .add_layer(8, bm::Activation::tanh)
                     .add_layer(1, bm::Activation::sigmoid)
                     .loss(bm::Loss::binary_cross_entropy)
                     .optimizer(optimizer)
                     .seed(42)
                     .build();

    const bm::Matrix inputs{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const bm::Matrix targets{{0.0}, {1.0}, {1.0}, {0.0}};

    auto options = quick_options(1500, inputs.size());
    options.shuffle = true;
    model.fit(inputs, targets, options);

    EXPECT_LT(model.evaluate_loss(inputs, targets), 0.08);
    EXPECT_LT(model.predict({0.0, 0.0})[0], 0.2);
    EXPECT_GT(model.predict({0.0, 1.0})[0], 0.8);
    EXPECT_GT(model.predict({1.0, 0.0})[0], 0.8);
    EXPECT_LT(model.predict({1.0, 1.0})[0], 0.2);
}

TEST(Training, MulticlassSoftmaxConverges) {
    auto model = bm::Model::builder()
                     .input_size(2)
                     .add_layer(3, bm::Activation::softmax)
                     .loss(bm::Loss::categorical_cross_entropy)
                     .optimizer(bm::OptimizerConfig::adam(0.08))
                     .seed(19)
                     .build();

    const bm::Matrix inputs{{2.0, 0.0}, {0.0, 2.0}, {-2.0, -2.0},
                            {3.0, 0.1}, {0.1, 3.0}, {-3.0, -2.0}};
    const bm::Matrix targets{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
                             {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

    auto options = quick_options(500, 3);
    model.fit(inputs, targets, options);

    EXPECT_LT(model.evaluate_loss(inputs, targets), 0.05);
    EXPECT_GT(model.predict({2.5, 0.0})[0], 0.9);
    EXPECT_GT(model.predict({0.0, 2.5})[1], 0.9);
    EXPECT_GT(model.predict({-2.5, -2.5})[2], 0.9);
}

TEST(Training, SgdWithMomentumReducesLoss) {
    auto optimizer = bm::OptimizerConfig::sgd(0.05, 0.8);
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(5)
                     .build();

    const bm::Matrix inputs{{-1.0}, {0.0}, {1.0}, {2.0}};
    const bm::Matrix targets{{-3.0}, {-1.0}, {1.0}, {3.0}};
    const double before = model.evaluate_loss(inputs, targets);
    model.fit(inputs, targets, quick_options(250, 2));
    EXPECT_LT(model.evaluate_loss(inputs, targets), before * 0.01);
}

TEST(Training, ValidationSplitAndEarlyStoppingPopulateHistory) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::adam(0.03))
                     .seed(17)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 50;
    options.batch_size = 2;
    options.validation_split = 0.25;
    options.early_stopping_patience = 3;
    options.seed = 99;

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}, {3.0}, {5.0}, {7.0}, {9.0}},
                                   options);
    EXPECT_FALSE(history.training_loss.empty());
    EXPECT_EQ(history.validation_loss.size(), history.training_loss.size());
    EXPECT_LE(history.training_loss.size(), options.epochs);
    EXPECT_EQ(history.monitor, bm::TrainingMonitor::validation_loss);
    EXPECT_EQ(history.monitor_mode, bm::TrainingMonitorMode::minimize);
}

TEST(Training, DiagnosticsReportEpochMetrics) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(2, bm::Activation::tanh)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.02}))
                     .seed(71)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 4;
    options.batch_size = 2;
    options.shuffle = false;
    options.validation_split = bm::Scalar{0.25};
    options.test_split = bm::Scalar{0.25};
    options.restore_best_weights = true;

    const auto history = model.fit({{-3.0}, {-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}, {3.0}, {4.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}, {3.0}, {5.0}, {7.0}, {9.0}},
                                   options);

    ASSERT_EQ(history.epochs.size(), history.training_loss.size());
    EXPECT_EQ(history.validation_loss.size(), history.training_loss.size());
    EXPECT_EQ(history.test_loss.size(), history.training_loss.size());
    EXPECT_EQ(history.stop_reason, bm::TrainingStopReason::completed_epochs);
    EXPECT_EQ(bm::to_string(history.stop_reason), "completed_epochs");
    EXPECT_FALSE(history.stop_message.empty());
    EXPECT_TRUE(history.has_best_checkpoint);
    EXPECT_LT(history.best_epoch, history.epochs.size());

    std::size_t best_checkpoint_count = 0;
    for (std::size_t i = 0; i < history.epochs.size(); ++i) {
        const auto& epoch = history.epochs[i];
        EXPECT_EQ(epoch.epoch, i);
        EXPECT_NEAR(epoch.training_loss, history.training_loss[i], kTightTolerance);
        ASSERT_TRUE(epoch.has_validation_loss);
        EXPECT_NEAR(epoch.validation_loss, history.validation_loss[i], kTightTolerance);
        ASSERT_TRUE(epoch.has_test_loss);
        EXPECT_NEAR(epoch.test_loss, history.test_loss[i], kTightTolerance);
        EXPECT_EQ(epoch.monitor, bm::TrainingMonitor::validation_loss);
        EXPECT_NEAR(epoch.monitored_metric, history.validation_loss[i], kTightTolerance);
        EXPECT_NEAR(epoch.learning_rate, 0.02, kTightTolerance);
        EXPECT_TRUE(std::isfinite(epoch.gradient_norm));
        EXPECT_GE(epoch.gradient_norm, bm::Scalar{0});
        EXPECT_GT(epoch.clipping.batch_count, 0U);
        EXPECT_GT(epoch.clipping.layer_count, 0U);
        EXPECT_GT(epoch.clipping.gradient_value_count, 0U);
        EXPECT_EQ(epoch.clipping.global_clip_count, 0U);
        EXPECT_EQ(epoch.clipping.layer_clip_count, 0U);
        EXPECT_EQ(epoch.clipping.gradient_value_clip_count, 0U);
        EXPECT_NEAR(epoch.clipping.global_clip_rate, 0.0, kTightTolerance);
        EXPECT_NEAR(epoch.clipping.layer_clip_rate, 0.0, kTightTolerance);
        EXPECT_NEAR(epoch.clipping.value_clip_rate, 0.0, kTightTolerance);
        EXPECT_NEAR(epoch.clipping.minimum_clip_scale, 1.0, kTightTolerance);
        EXPECT_TRUE(epoch.weights.finite);
        EXPECT_GT(epoch.weights.count, 0U);
        EXPECT_EQ(epoch.weights.non_finite_count, 0U);
        EXPECT_LE(epoch.weights.minimum, epoch.weights.maximum);
        EXPECT_TRUE(std::isfinite(epoch.weights.mean));
        EXPECT_TRUE(epoch.finite);
        EXPECT_EQ(epoch.non_finite_parameter_count, 0U);
        EXPECT_GE(epoch.epoch_seconds, 0.0);
        EXPECT_EQ(epoch.cooldown_remaining, 0U);
        EXPECT_EQ(epoch.improved, epoch.best_checkpoint);
        if (epoch.best_checkpoint) {
            ++best_checkpoint_count;
        }
    }
    EXPECT_GT(best_checkpoint_count, 0U);
}

TEST(Training, DiagnosticsReportEarlyStoppingReason) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.01}))
                     .seed(72)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 10;
    options.batch_size = 2;
    options.shuffle = false;
    options.early_stopping_patience = 1;
    options.min_delta = bm::Scalar{1000000};

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    EXPECT_EQ(history.stop_reason, bm::TrainingStopReason::early_stopping);
    EXPECT_EQ(history.training_loss.size(), 2U);
    EXPECT_EQ(history.epochs.size(), history.training_loss.size());
    EXPECT_EQ(history.monitor, bm::TrainingMonitor::training_loss);
    EXPECT_EQ(history.monitor_mode, bm::TrainingMonitorMode::minimize);
    EXPECT_TRUE(history.epochs.front().best_checkpoint);
    EXPECT_TRUE(history.epochs.front().improved);
    EXPECT_EQ(history.epochs.front().stale_epochs, 0U);
    EXPECT_NEAR(history.epochs.front().monitored_metric, history.training_loss.front(), kTightTolerance);
    EXPECT_FALSE(history.epochs.back().best_checkpoint);
    EXPECT_FALSE(history.epochs.back().improved);
    EXPECT_EQ(history.epochs.back().stale_epochs, 1U);
}

TEST(Training, EarlyStoppingCooldownDelaysPatience) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.01}))
                     .seed(74)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 10;
    options.batch_size = 2;
    options.shuffle = false;
    options.early_stopping_patience = 1;
    options.early_stopping_cooldown = 2;
    options.min_delta = bm::Scalar{1000000};

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.stop_reason, bm::TrainingStopReason::early_stopping);
    ASSERT_EQ(history.epochs.size(), 4U);
    EXPECT_TRUE(history.epochs[0].improved);
    EXPECT_EQ(history.epochs[0].cooldown_remaining, 2U);
    EXPECT_EQ(history.epochs[0].stale_epochs, 0U);
    EXPECT_FALSE(history.epochs[1].improved);
    EXPECT_EQ(history.epochs[1].cooldown_remaining, 1U);
    EXPECT_EQ(history.epochs[1].stale_epochs, 0U);
    EXPECT_FALSE(history.epochs[2].improved);
    EXPECT_EQ(history.epochs[2].cooldown_remaining, 0U);
    EXPECT_EQ(history.epochs[2].stale_epochs, 0U);
    EXPECT_FALSE(history.epochs[3].improved);
    EXPECT_EQ(history.epochs[3].cooldown_remaining, 0U);
    EXPECT_EQ(history.epochs[3].stale_epochs, 1U);
}

TEST(Training, EarlyStoppingUsesExplicitMonitorAndMode) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::sgd(bm::Scalar{1e-30}))
                     .seed(75)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 10;
    options.batch_size = 2;
    options.shuffle = false;
    options.early_stopping_patience = 1;
    options.early_stopping_monitor = bm::TrainingMonitor::training_loss;
    options.early_stopping_mode = bm::TrainingMonitorMode::maximize;
    options.min_delta = bm::Scalar{0};

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.stop_reason, bm::TrainingStopReason::early_stopping);
    ASSERT_EQ(history.epochs.size(), 2U);
    EXPECT_EQ(history.monitor, bm::TrainingMonitor::training_loss);
    EXPECT_EQ(history.monitor_mode, bm::TrainingMonitorMode::maximize);
    EXPECT_TRUE(history.epochs[0].improved);
    EXPECT_FALSE(history.epochs[1].improved);
    EXPECT_EQ(history.epochs[1].stale_epochs, 1U);
    EXPECT_NEAR(history.epochs[0].monitored_metric, history.training_loss[0], kTightTolerance);
    EXPECT_NEAR(history.epochs[1].monitored_metric, history.training_loss[1], kTightTolerance);
}

TEST(Training, RejectsUnavailableEarlyStoppingMonitor) {
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .seed(76)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 2;
    options.batch_size = 1;
    options.early_stopping_monitor = bm::TrainingMonitor::validation_loss;

    EXPECT_THROW((void)model.fit({{bm::Scalar{1}}}, {{bm::Scalar{2}}}, options), std::invalid_argument);

    options.early_stopping_monitor = bm::TrainingMonitor::test_loss;
    EXPECT_THROW((void)model.fit({{bm::Scalar{1}}}, {{bm::Scalar{2}}}, options), std::invalid_argument);
}

TEST(Training, LearningRateStepDecayReportsExpectedCurve) {
    auto model = make_learning_rate_test_model();
    auto options = quick_options(5, 2);
    options.shuffle = false;
    options.restore_best_weights = false;
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::step_decay(2, bm::Scalar{0.5});

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.epochs.size(), 5U);
    const bm::Vector expected{bm::Scalar{0.1}, bm::Scalar{0.1}, bm::Scalar{0.05},
                              bm::Scalar{0.05}, bm::Scalar{0.025}};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(history.epochs[i].learning_rate_schedule, bm::LearningRateSchedule::step_decay);
        EXPECT_NEAR(history.epochs[i].learning_rate, expected[i], kTightTolerance);
    }
    EXPECT_NEAR(history.epochs[1].next_learning_rate, 0.05, kTightTolerance);
}

TEST(Training, LearningRateWarmupAppliesBeforeConstantSchedule) {
    auto model = make_learning_rate_test_model();
    auto options = quick_options(4, 2);
    options.shuffle = false;
    options.restore_best_weights = false;
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::constant();
    options.learning_rate_schedule.warmup_epochs = 3;
    options.learning_rate_schedule.warmup_start_learning_rate = bm::Scalar{0.01};

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.epochs.size(), 4U);
    EXPECT_NEAR(history.epochs[0].learning_rate, 0.04, kTightTolerance);
    EXPECT_NEAR(history.epochs[1].learning_rate, 0.07, kTightTolerance);
    EXPECT_NEAR(history.epochs[2].learning_rate, 0.1, kTightTolerance);
    EXPECT_NEAR(history.epochs[3].learning_rate, 0.1, kTightTolerance);
}

TEST(Training, LearningRateExponentialDecayReportsExpectedCurve) {
    auto model = make_learning_rate_test_model();
    auto options = quick_options(4, 2);
    options.shuffle = false;
    options.restore_best_weights = false;
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::exponential_decay(bm::Scalar{0.5});

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.epochs.size(), 4U);
    EXPECT_NEAR(history.epochs[0].learning_rate, 0.1, kTightTolerance);
    EXPECT_NEAR(history.epochs[1].learning_rate, 0.05, kTightTolerance);
    EXPECT_NEAR(history.epochs[2].learning_rate, 0.025, kTightTolerance);
    EXPECT_NEAR(history.epochs[3].learning_rate, 0.0125, kTightTolerance);
}

TEST(Training, LearningRateCosineAnnealingReachesConfiguredFloor) {
    auto model = make_learning_rate_test_model();
    auto options = quick_options(5, 2);
    options.shuffle = false;
    options.restore_best_weights = false;
    options.learning_rate_schedule =
        bm::LearningRateScheduleConfig::cosine_annealing(4, bm::Scalar{0.01});

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.epochs.size(), 5U);
    EXPECT_NEAR(history.epochs[0].learning_rate, 0.1, kTightTolerance);
    EXPECT_NEAR(history.epochs[2].learning_rate, 0.055, 1e-5);
    EXPECT_NEAR(history.epochs[4].learning_rate, 0.01, kTightTolerance);
}

TEST(Training, LearningRateReduceOnPlateauUsesMetricHistory) {
    auto model = make_learning_rate_test_model();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{2}}, {bm::Scalar{0}}},
    });

    auto options = quick_options(4, 2);
    options.shuffle = false;
    options.restore_best_weights = false;
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::reduce_on_plateau(1, bm::Scalar{0.5});
    options.learning_rate_schedule.reduce_on_plateau_monitor = bm::TrainingMonitor::training_loss;

    const auto history = model.fit({{bm::Scalar{1}}, {bm::Scalar{2}}},
                                   {{bm::Scalar{2}}, {bm::Scalar{4}}},
                                   options);

    ASSERT_EQ(history.epochs.size(), 4U);
    EXPECT_NEAR(history.epochs[0].learning_rate, 0.1, kTightTolerance);
    EXPECT_FALSE(history.epochs[0].learning_rate_reduced);
    EXPECT_NEAR(history.epochs[0].next_learning_rate, 0.1, kTightTolerance);
    EXPECT_NEAR(history.epochs[1].learning_rate, 0.1, kTightTolerance);
    EXPECT_TRUE(history.epochs[1].learning_rate_reduced);
    EXPECT_NEAR(history.epochs[1].next_learning_rate, 0.05, kTightTolerance);
    EXPECT_NEAR(history.epochs[2].learning_rate, 0.05, kTightTolerance);
    EXPECT_TRUE(history.epochs[2].learning_rate_reduced);
    EXPECT_NEAR(history.epochs[2].next_learning_rate, 0.025, kTightTolerance);
    EXPECT_NEAR(history.epochs[3].learning_rate, 0.025, kTightTolerance);
}

TEST(Training, L1RegularizationShrinksWeightsWithZeroDataGradient) {
    auto optimizer = bm::OptimizerConfig::sgd(bm::Scalar{0.1});
    optimizer.l1 = bm::Scalar{0.5};
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(91)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{2}}, {bm::Scalar{0}}},
    });

    auto options = one_step_options(1);
    model.fit({{bm::Scalar{0}}}, {{bm::Scalar{0}}}, options);

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_NEAR(parameters[0].weights[0], 1.95, kTightTolerance);
}

TEST(Training, AdamWAppliesDecoupledWeightDecayWithoutDataGradient) {
    auto optimizer = bm::OptimizerConfig::adamw(bm::Scalar{0.1}, bm::Scalar{0.5});
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(92)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{2}}, {bm::Scalar{0}}},
    });

    auto options = one_step_options(1);
    model.fit({{bm::Scalar{0}}}, {{bm::Scalar{0}}}, options);

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_NEAR(parameters[0].weights[0], 1.9, 1e-6);
}

TEST(Training, MaxNormConstrainsIncomingWeightVectors) {
    auto optimizer = bm::OptimizerConfig::sgd(bm::Scalar{0.1});
    optimizer.max_norm = bm::Scalar{1};
    auto model = bm::Model::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(93)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear, {bm::Scalar{3}, bm::Scalar{4}}, {bm::Scalar{0}}},
    });

    auto options = one_step_options(1);
    model.fit({{bm::Scalar{0}, bm::Scalar{0}}}, {{bm::Scalar{0}}}, options);

    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    const double norm = std::sqrt(static_cast<double>(parameters[0].weights[0] * parameters[0].weights[0] +
                                                     parameters[0].weights[1] * parameters[0].weights[1]));
    EXPECT_NEAR(norm, 1.0, kTightTolerance);
    EXPECT_NEAR(parameters[0].weights[0], 0.6, kTightTolerance);
    EXPECT_NEAR(parameters[0].weights[1], 0.8, kTightTolerance);
}

TEST(Training, GlobalGradientClipNormReportsClipRate) {
    auto optimizer = bm::OptimizerConfig::sgd(bm::Scalar{1});
    optimizer.gradient_clip_norm = bm::Scalar{1};
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(96)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{0}}, {bm::Scalar{0}}},
    });

    const auto history = model.fit({{bm::Scalar{1}}}, {{bm::Scalar{10}}}, one_step_options(1));
    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_NEAR(parameters[0].weights[0], 0.70710678, 1e-5);
    EXPECT_NEAR(parameters[0].biases[0], 0.70710678, 1e-5);
    ASSERT_EQ(history.epochs.size(), 1U);
    EXPECT_EQ(history.epochs[0].clipping.batch_count, 1U);
    EXPECT_EQ(history.epochs[0].clipping.global_clip_count, 1U);
    EXPECT_NEAR(history.epochs[0].clipping.global_clip_rate, 1.0, kTightTolerance);
    EXPECT_GT(history.epochs[0].clipping.minimum_clip_scale, bm::Scalar{0});
    EXPECT_LT(history.epochs[0].clipping.minimum_clip_scale, bm::Scalar{1});
}

TEST(Training, GradientClipValueClampsIndividualGradientsAndReportsRate) {
    auto optimizer = bm::OptimizerConfig::sgd(bm::Scalar{0.1});
    optimizer.gradient_clip_value = bm::Scalar{0.5};
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(97)
                     .build();
    model.set_parameters({
        bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{0}}, {bm::Scalar{0}}},
    });

    const auto history = model.fit({{bm::Scalar{1}}}, {{bm::Scalar{10}}}, one_step_options(1));
    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_NEAR(parameters[0].weights[0], 0.05, kTightTolerance);
    EXPECT_NEAR(parameters[0].biases[0], 0.05, kTightTolerance);
    ASSERT_EQ(history.epochs.size(), 1U);
    EXPECT_EQ(history.epochs[0].clipping.gradient_value_count, 2U);
    EXPECT_EQ(history.epochs[0].clipping.gradient_value_clip_count, 2U);
    EXPECT_NEAR(history.epochs[0].clipping.value_clip_rate, 1.0, kTightTolerance);
    EXPECT_NEAR(history.epochs[0].clipping.global_clip_rate, 0.0, kTightTolerance);
    EXPECT_NEAR(history.epochs[0].clipping.layer_clip_rate, 0.0, kTightTolerance);
    EXPECT_NEAR(history.epochs[0].clipping.minimum_clip_scale, 1.0, kTightTolerance);
}

TEST(Training, LayerGradientClipNormClipsLayerAndReportsRate) {
    auto optimizer = bm::OptimizerConfig::sgd(bm::Scalar{1});
    optimizer.layer_gradient_clip_norm = bm::Scalar{1};
    auto model = bm::Model::builder()
                     .input_size(2)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(98)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 1, bm::Activation::linear,
                            {bm::Scalar{0}, bm::Scalar{0}},
                            {bm::Scalar{0}}},
    });

    const auto history = model.fit({{bm::Scalar{3}, bm::Scalar{4}}}, {{bm::Scalar{1}}}, one_step_options(1));
    const auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_NEAR(parameters[0].weights[0], 0.5883484, 1e-5);
    EXPECT_NEAR(parameters[0].weights[1], 0.7844645, 1e-5);
    EXPECT_NEAR(parameters[0].biases[0], 0.1961161, 1e-5);
    ASSERT_EQ(history.epochs.size(), 1U);
    EXPECT_EQ(history.epochs[0].clipping.layer_count, 1U);
    EXPECT_EQ(history.epochs[0].clipping.layer_clip_count, 1U);
    EXPECT_NEAR(history.epochs[0].clipping.layer_clip_rate, 1.0, kTightTolerance);
    EXPECT_EQ(history.epochs[0].clipping.gradient_value_count, 3U);
    EXPECT_EQ(history.epochs[0].clipping.gradient_value_clip_count, 0U);
    EXPECT_NEAR(history.epochs[0].clipping.global_clip_rate, 0.0, kTightTolerance);
    EXPECT_GT(history.epochs[0].clipping.minimum_clip_scale, bm::Scalar{0});
    EXPECT_LT(history.epochs[0].clipping.minimum_clip_scale, bm::Scalar{1});
}

TEST(Training, GradientNoiseIsSeededAndChangesUpdates) {
    const auto build = [] {
        auto model = bm::Model::builder()
                         .input_size(1)
                         .add_layer(1, bm::Activation::linear)
                         .loss(bm::Loss::mean_squared_error)
                         .optimizer(bm::OptimizerConfig::sgd(bm::Scalar{0.1}))
                         .seed(94)
                         .build();
        model.set_parameters({
            bm::LayerParameters{1, 1, bm::Activation::linear, {bm::Scalar{0}}, {bm::Scalar{0}}},
        });
        return model;
    };

    auto first = build();
    auto second = build();
    auto different = build();

    auto options = one_step_options(1);
    options.gradient_noise_stddev = bm::Scalar{0.1};
    options.seed = 100;
    first.fit({{bm::Scalar{0}}}, {{bm::Scalar{0}}}, options);
    second.fit({{bm::Scalar{0}}}, {{bm::Scalar{0}}}, options);
    options.seed = 101;
    different.fit({{bm::Scalar{0}}}, {{bm::Scalar{0}}}, options);

    EXPECT_EQ(first.parameters()[0].weights, second.parameters()[0].weights);
    EXPECT_EQ(first.parameters()[0].biases, second.parameters()[0].biases);
    EXPECT_NE(first.parameters()[0].weights, different.parameters()[0].weights);
}

TEST(Training, DropoutIsSeededAndStoredInLayerParameters) {
    const auto build = [] {
        auto model = bm::Model::builder()
                         .input_size(2)
                         .add_layer(4, bm::Activation::relu, bm::Scalar{0.5})
                         .add_layer(1, bm::Activation::linear)
                         .loss(bm::Loss::mean_squared_error)
                         .optimizer(bm::OptimizerConfig::sgd(bm::Scalar{0.05}))
                         .seed(95)
                         .build();
        model.set_parameters({
            bm::LayerParameters{2, 4, bm::Activation::relu,
                                {bm::Scalar{0.1}, bm::Scalar{0.2},
                                 bm::Scalar{-0.1}, bm::Scalar{0.3},
                                 bm::Scalar{0.4}, bm::Scalar{-0.2},
                                 bm::Scalar{0.5}, bm::Scalar{0.1}},
                                {bm::Scalar{0}, bm::Scalar{0}, bm::Scalar{0}, bm::Scalar{0}},
                                bm::Scalar{0.5}},
            bm::LayerParameters{4, 1, bm::Activation::linear,
                                {bm::Scalar{0.2}, bm::Scalar{-0.3}, bm::Scalar{0.4}, bm::Scalar{0.1}},
                                {bm::Scalar{0}}},
        });
        return model;
    };

    auto first = build();
    auto second = build();
    auto different = build();

    auto options = quick_options(2, 2);
    options.shuffle = false;
    options.restore_best_weights = false;
    options.seed = 200;
    const bm::Matrix inputs{{bm::Scalar{1}, bm::Scalar{2}}, {bm::Scalar{-1}, bm::Scalar{1}}};
    const bm::Matrix targets{{bm::Scalar{1}}, {bm::Scalar{-1}}};
    first.fit(inputs, targets, options);
    second.fit(inputs, targets, options);
    options.seed = 201;
    different.fit(inputs, targets, options);

    EXPECT_EQ(first.parameters()[0].dropout_probability, bm::Scalar{0.5});
    EXPECT_EQ(first.parameters()[0].weights, second.parameters()[0].weights);
    EXPECT_NE(first.parameters()[0].weights, different.parameters()[0].weights);
}

TEST(Training, DiagnosticsStopOnNonFiniteParameters) {
    auto optimizer = bm::OptimizerConfig::sgd(std::numeric_limits<bm::Scalar>::max(), bm::Scalar{0});
    auto model = bm::Model::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(73)
                     .build();

    auto options = one_step_options(1);
    options.epochs = 5;
    options.shuffle = false;
    options.restore_best_weights = false;

    const auto history = model.fit({{bm::Scalar{1}}}, {{bm::Scalar{100}}}, options);

    ASSERT_FALSE(history.epochs.empty());
    EXPECT_TRUE(history.stop_reason == bm::TrainingStopReason::non_finite_loss ||
                history.stop_reason == bm::TrainingStopReason::non_finite_weights);
    EXPECT_FALSE(history.epochs.back().finite);
    EXPECT_GT(history.epochs.back().non_finite_parameter_count, 0U);
    EXPECT_FALSE(history.epochs.back().weights.finite);
}

TEST(Training, RejectsInvalidTrainingDataAndOptions) {
    auto model = make_known_linear_model();

    EXPECT_THROW((void)model.fit({}, {}, quick_options(1, 1)), std::invalid_argument);
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}, {2.0}}, quick_options(1, 1)),
                 std::invalid_argument);
    EXPECT_THROW((void)model.fit({{1.0}}, {{1.0}}, quick_options(1, 1)), std::invalid_argument);
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0, 2.0}}, quick_options(1, 1)),
                 std::invalid_argument);

    auto options = quick_options(1, 1);
    options.epochs = 0;
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);

    options = quick_options(1, 1);
    options.validation_split = 1.0;
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);

    options = quick_options(1, 1);
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::step_decay(0, bm::Scalar{0.5});
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);

    options = quick_options(1, 1);
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::exponential_decay(bm::Scalar{1.5});
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);

    options = quick_options(1, 1);
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::reduce_on_plateau(1, bm::Scalar{1});
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);

    options = quick_options(1, 1);
    options.learning_rate_schedule.minimum_learning_rate = bm::Scalar{2};
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);

    options = quick_options(1, 1);
    options.gradient_noise_stddev = bm::Scalar{-0.1};
    EXPECT_THROW((void)model.fit({{1.0, 2.0}}, {{1.0}}, options), std::invalid_argument);
}

TEST(Parameters, RoundTripAndSetParametersWork) {
    auto model = make_known_linear_model();
    auto parameters = model.parameters();
    ASSERT_EQ(parameters.size(), 1U);
    EXPECT_EQ(parameters[0].input_size, 2U);
    EXPECT_EQ(parameters[0].output_size, 1U);
    EXPECT_EQ(parameters[0].activation, bm::Activation::linear);

    parameters[0].weights = {-1.0, 4.0};
    parameters[0].biases = {2.0};
    model.set_parameters(parameters);

    expect_vector_near(model.predict({3.0, 1.0}), {3.0}, kTightTolerance);
}

TEST(Parameters, RejectsShapeAndActivationMismatches) {
    auto model = make_known_linear_model();
    auto parameters = model.parameters();

    auto bad = parameters;
    bad[0].weights.pop_back();
    EXPECT_THROW(model.set_parameters(bad), std::invalid_argument);

    bad = parameters;
    bad[0].activation = bm::Activation::relu;
    EXPECT_THROW(model.set_parameters(bad), std::invalid_argument);

    bad = parameters;
    bad.push_back(parameters[0]);
    EXPECT_THROW(model.set_parameters(bad), std::invalid_argument);
}

TEST(Parameters, ReturnedParametersAreIndependentCopies) {
    auto model = make_known_linear_model();
    auto parameters = model.parameters();
    parameters[0].weights[0] = 100.0;

    expect_vector_near(model.predict({3.0, 1.0}), {5.5}, kTightTolerance);
}

TEST(CompiledModel, CompileProducesFrozenRuntimeWithContiguousStorage) {
    bm::MutableModel training = make_known_two_layer_model();
    const bm::CompiledModel compiled = training.compile();
    const bm::InferenceModel legacy_inference = training.to_inference_model();
    const bm::Vector input{bm::Scalar{2}, bm::Scalar{-1}};

    EXPECT_EQ(compiled.input_size(), training.input_size());
    EXPECT_EQ(compiled.output_size(), training.output_size());
    EXPECT_EQ(compiled.layer_count(), training.layer_count());
    EXPECT_EQ(compiled.weight_count(), legacy_inference.weight_count());
    EXPECT_EQ(compiled.bias_count(), legacy_inference.bias_count());
    EXPECT_NE(compiled.weights_data(), nullptr);
    EXPECT_NE(compiled.biases_data(), nullptr);
    EXPECT_NE(compiled.weights_data(), training.parameters().front().weights.data());
    expect_vector_near(compiled.predict(input), training.predict(input), kTightTolerance);

    bm::InferenceWorkspace workspace(compiled);
    ASSERT_EQ(compiled.predict_to(input.data(), input.size(), workspace), bm::InferenceStatus::ok);
    expect_vector_near(workspace.output(), training.predict(input), kTightTolerance);
}

TEST(CompiledModel, LoadersReturnCompiledRuntime) {
    const auto text_path = temp_test_path("brutal_mlp_compiled_roundtrip.model");
    const auto binary_path = temp_test_path("brutal_mlp_compiled_roundtrip.bmlp");
    const bm::CompiledModel compiled = make_known_two_layer_model().compile();
    const bm::Vector input{bm::Scalar{-0.5}, bm::Scalar{0.25}};

    compiled.save(text_path);
    compiled.save_binary(binary_path);
    const bm::CompiledModel loaded_text = bm::CompiledModel::load(text_path);
    const bm::CompiledModel loaded_binary = bm::CompiledModel::load_binary(binary_path);
    std::filesystem::remove(text_path);
    std::filesystem::remove(binary_path);

    expect_vector_near(loaded_text.predict(input), compiled.predict(input), kTightTolerance);
    expect_vector_near(loaded_binary.predict(input), compiled.predict(input), kTightTolerance);
    EXPECT_EQ(loaded_text.weight_count(), compiled.weight_count());
    EXPECT_EQ(loaded_binary.bias_count(), compiled.bias_count());
}

TEST(CompiledModel, FromParametersCompilesNormalizationAndTopology) {
    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{10}, bm::Scalar{-2}},
        {bm::Scalar{2}, bm::Scalar{4}},
        {bm::Scalar{5}},
        {bm::Scalar{10}});
    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 1, bm::Activation::linear,
                            {bm::Scalar{3}, bm::Scalar{-2}},
                            {bm::Scalar{1}}},
    };

    const bm::CompiledModel compiled = bm::CompiledModel::from_parameters(parameters, normalization);
    EXPECT_EQ(compiled.scratch_size(), 2U);
    EXPECT_TRUE(compiled.normalization().has_input_normalization());
    EXPECT_TRUE(compiled.normalization().has_output_normalization());
    expect_vector_near(compiled.predict({bm::Scalar{12}, bm::Scalar{6}}),
                       {bm::Scalar{5}},
                       kTightTolerance);
}

TEST(InferenceModel, FrozenModelMatchesTrainingPrediction) {
    const auto training = make_known_two_layer_model();
    const auto inference = training.to_inference_model();
    const bm::Vector input{2.0, -1.0};

    EXPECT_FALSE(inference.empty());
    EXPECT_EQ(inference.input_size(), training.input_size());
    EXPECT_EQ(inference.output_size(), training.output_size());
    EXPECT_EQ(inference.layer_count(), training.layer_count());
    EXPECT_EQ(inference.weight_count(), 12U);
    EXPECT_EQ(inference.bias_count(), 5U);
    EXPECT_NE(inference.weights_data(), nullptr);
    EXPECT_NE(inference.biases_data(), nullptr);
    EXPECT_EQ(inference.scratch_size(), 6U);

    expect_vector_near(inference.predict(input), training.predict(input), kTightTolerance);

    bm::Vector output(inference.output_size(), 0.0);
    bm::Vector scratch(inference.scratch_size(), 0.0);
    const auto status =
        inference.predict_to(input.data(), input.size(), output.data(), output.size(), scratch.data(), scratch.size());

    EXPECT_EQ(status, bm::InferenceStatus::ok);
    expect_vector_near(output, training.predict(input), kTightTolerance);
}

TEST(InferenceModel, PredictToDoesNotAllocateOnHotPath) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const bm::Vector input{2.0, -1.0};
    bm::Vector output(inference.output_size(), 0.0);
    bm::Vector scratch(inference.scratch_size(), 0.0);

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    const auto status =
        inference.predict_to(input.data(), input.size(), output.data(), output.size(), scratch.data(), scratch.size());
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(status, bm::InferenceStatus::ok);
    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
}

TEST(InferenceModel, WorkspacePredictDoesNotAllocateOnHotPath) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const bm::Vector input{2.0, -1.0};
    bm::InferenceWorkspace workspace(inference);

    EXPECT_EQ(workspace.output_size(), inference.output_size());
    EXPECT_EQ(workspace.scratch_size(), inference.scratch_size());

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    const auto status = inference.predict_to(input.data(), input.size(), workspace);
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(status, bm::InferenceStatus::ok);
    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
    expect_vector_near(workspace.output(), inference.predict(input), kTightTolerance);
}

TEST(InferenceModel, PredictUncheckedToMatchesSafePathAndDoesNotAllocate) {
    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{10}, bm::Scalar{-2}},
        {bm::Scalar{2}, bm::Scalar{4}},
        {bm::Scalar{5}, bm::Scalar{-3}},
        {bm::Scalar{10}, bm::Scalar{2}});
    const auto training = bm::TrainingModel::builder()
                              .input_size(2)
                              .add_layer(3, bm::Activation::tanh)
                              .add_layer(2, bm::Activation::linear)
                              .normalization(normalization)
                              .build();
    auto inference = training.to_inference_model();
    const bm::Vector input{bm::Scalar{12}, bm::Scalar{6}};
    bm::Vector safe_output(inference.output_size(), bm::Scalar{0});
    bm::Vector fast_output(inference.output_size(), bm::Scalar{0});
    bm::Vector safe_scratch(inference.scratch_size(), bm::Scalar{0});
    bm::Vector fast_scratch(inference.scratch_size(), bm::Scalar{0});

    ASSERT_EQ(inference.predict_to(input.data(),
                                   input.size(),
                                   safe_output.data(),
                                   safe_output.size(),
                                   safe_scratch.data(),
                                   safe_scratch.size()),
              bm::InferenceStatus::ok);

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    inference.predict_unchecked_to(input.data(), fast_output.data(), fast_scratch.data());
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
    expect_vector_near(fast_output, safe_output, kTightTolerance);
}

TEST(InferenceModel, WorkspacePredictUncheckedDoesNotAllocate) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const bm::Vector input{2.0, -1.0};
    bm::InferenceWorkspace safe_workspace(inference);
    bm::InferenceWorkspace fast_workspace(inference);

    ASSERT_EQ(inference.predict_to(input.data(), input.size(), safe_workspace), bm::InferenceStatus::ok);

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    inference.predict_unchecked_to(input.data(), fast_workspace);
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
    expect_vector_near(fast_workspace.output(), safe_workspace.output(), kTightTolerance);
}

TEST(InferenceModel, BatchPredictToMatchesScalarPredictionAndDoesNotAllocate) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const std::size_t samples = 3;
    const std::size_t input_stride = 4;
    const std::size_t output_stride = 3;

    bm::Vector inputs(samples * input_stride, -777.0);
    inputs[0] = 2.0;
    inputs[1] = -1.0;
    inputs[input_stride] = -1.0;
    inputs[input_stride + 1] = 0.5;
    inputs[2 * input_stride] = 0.25;
    inputs[2 * input_stride + 1] = 3.0;

    bm::Vector outputs(samples * output_stride, -999.0);
    bm::Vector scratch(inference.scratch_size(), 0.0);

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    const auto status = inference.predict_batch_to(inputs.data(),
                                                   samples,
                                                   input_stride,
                                                   outputs.data(),
                                                   output_stride,
                                                   scratch.data(),
                                                   scratch.size());
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(status, bm::InferenceStatus::ok);
    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);

    for (std::size_t sample = 0; sample < samples; ++sample) {
        bm::Vector input{inputs[sample * input_stride], inputs[sample * input_stride + 1]};
        bm::Vector actual{outputs[sample * output_stride], outputs[sample * output_stride + 1]};
        expect_vector_near(actual, inference.predict(input), kTightTolerance);
        EXPECT_EQ(outputs[sample * output_stride + 2], -999.0);
    }
}

TEST(InferenceModel, BatchPredictUncheckedMatchesSafePathAndDoesNotAllocate) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const std::size_t samples = 3;
    const std::size_t input_stride = 4;
    const std::size_t output_stride = 3;

    bm::Vector inputs(samples * input_stride, -777.0);
    inputs[0] = 2.0;
    inputs[1] = -1.0;
    inputs[input_stride] = -1.0;
    inputs[input_stride + 1] = 0.5;
    inputs[2 * input_stride] = 0.25;
    inputs[2 * input_stride + 1] = 3.0;

    bm::Vector safe_outputs(samples * output_stride, -999.0);
    bm::Vector fast_outputs(samples * output_stride, -999.0);
    bm::Vector safe_scratch(inference.scratch_size(), 0.0);
    bm::Vector fast_scratch(inference.scratch_size(), 0.0);

    ASSERT_EQ(inference.predict_batch_to(inputs.data(),
                                         samples,
                                         input_stride,
                                         safe_outputs.data(),
                                         output_stride,
                                         safe_scratch.data(),
                                         safe_scratch.size()),
              bm::InferenceStatus::ok);

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    inference.predict_batch_unchecked_to(inputs.data(),
                                         samples,
                                         input_stride,
                                         fast_outputs.data(),
                                         output_stride,
                                         fast_scratch.data());
    g_count_allocations.store(false, std::memory_order_relaxed);

    EXPECT_EQ(g_allocation_count.load(std::memory_order_relaxed), 0U);
    expect_vector_near(fast_outputs, safe_outputs, kTightTolerance);
}

TEST(InferenceModel, BatchPredictToCanUseWorkerThreadsWithPerWorkerScratch) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const std::size_t samples = 17;
    const std::size_t input_stride = 5;
    const std::size_t output_stride = 4;

    bm::Vector inputs(samples * input_stride, bm::Scalar{-777});
    for (std::size_t sample = 0; sample < samples; ++sample) {
        inputs[sample * input_stride] = static_cast<bm::Scalar>(static_cast<double>(sample) * 0.25 - 2.0);
        inputs[sample * input_stride + 1] =
            static_cast<bm::Scalar>(static_cast<double>(sample % 5) - 1.5);
    }

    bm::Vector serial_outputs(samples * output_stride, bm::Scalar{-999});
    bm::Vector parallel_outputs(samples * output_stride, bm::Scalar{-999});
    bm::Vector serial_scratch(inference.scratch_size(), bm::Scalar{0});
    ASSERT_EQ(inference.predict_batch_to(inputs.data(),
                                         samples,
                                         input_stride,
                                         serial_outputs.data(),
                                         output_stride,
                                         serial_scratch.data(),
                                         serial_scratch.size()),
              bm::InferenceStatus::ok);

    bm::ParallelOptions parallel;
    parallel.execution = bm::ParallelExecution::worker_threads;
    parallel.thread_count = 3;
    parallel.minimum_parallel_samples = 2;

    const std::size_t required_scratch = inference.batch_scratch_size(samples, parallel);
    EXPECT_EQ(required_scratch, inference.scratch_size() * parallel.thread_count);
    bm::Vector parallel_scratch(required_scratch, bm::Scalar{0});

    ASSERT_EQ(inference.predict_batch_to(inputs.data(),
                                         samples,
                                         input_stride,
                                         parallel_outputs.data(),
                                         output_stride,
                                         parallel_scratch.data(),
                                         parallel_scratch.size(),
                                         parallel),
              bm::InferenceStatus::ok);

    expect_vector_near(parallel_outputs, serial_outputs, kTightTolerance);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        EXPECT_EQ(parallel_outputs[sample * output_stride + 2], bm::Scalar{-999});
        EXPECT_EQ(parallel_outputs[sample * output_stride + 3], bm::Scalar{-999});
    }

    EXPECT_EQ(inference.predict_batch_to(inputs.data(),
                                         samples,
                                         input_stride,
                                         parallel_outputs.data(),
                                         output_stride,
                                         nullptr,
                                         required_scratch,
                                         parallel),
              bm::InferenceStatus::null_scratch);
    EXPECT_EQ(inference.predict_batch_to(inputs.data(),
                                         samples,
                                         input_stride,
                                         parallel_outputs.data(),
                                         output_stride,
                                         parallel_scratch.data(),
                                         required_scratch - 1,
                                         parallel),
              bm::InferenceStatus::insufficient_scratch);
}

TEST(InferenceModel, BatchParallelismFallsBackToSerialBelowThreshold) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    bm::ParallelOptions parallel;
    parallel.execution = bm::ParallelExecution::worker_threads;
    parallel.thread_count = 8;
    parallel.minimum_parallel_samples = 16;

    EXPECT_EQ(inference.batch_scratch_size(3, parallel), inference.scratch_size());

    const auto one_layer = make_known_linear_model().to_inference_model();
    bm::Vector inputs{bm::Scalar{3}, bm::Scalar{1}, bm::Scalar{2}, bm::Scalar{-4}};
    bm::Vector outputs(2, bm::Scalar{0});
    parallel.minimum_parallel_samples = 1;
    EXPECT_EQ(one_layer.batch_scratch_size(2, parallel), 0U);
    EXPECT_EQ(one_layer.predict_batch_to(inputs.data(),
                                         2,
                                         one_layer.input_size(),
                                         outputs.data(),
                                         one_layer.output_size(),
                                         nullptr,
                                         0,
                                         parallel),
              bm::InferenceStatus::ok);
    expect_vector_near(outputs, {bm::Scalar{5.5}, bm::Scalar{8.5}}, kTightTolerance);
}

TEST(InferenceModel, ConstModelCanBeSharedAcrossThreadsWithPerThreadWorkspace) {
    const bm::InferenceModel inference = make_known_two_layer_model().to_inference_model();
    const bm::Vector input{2.0, -1.0};
    const bm::Vector expected = inference.predict(input);
    const bm::Scalar* weights_before = inference.weights_data();
    const bm::Scalar* biases_before = inference.biases_data();
    constexpr std::size_t kThreadCount = 8;
    constexpr std::size_t kIterations = 256;

    std::atomic<std::size_t> failures{0};
    std::vector<bm::Vector> outputs(kThreadCount, bm::Vector(inference.output_size(), bm::Scalar{0}));
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (std::size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        threads.emplace_back([&, thread_index] {
            bm::InferenceWorkspace workspace(inference);
            for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
                if ((iteration % 2) == 0) {
                    const auto status = inference.predict_to(input.data(), input.size(), workspace);
                    if (status != bm::InferenceStatus::ok) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                } else {
                    inference.predict_unchecked_to(input.data(), workspace);
                }

                const bm::Vector& actual = workspace.output();
                for (std::size_t output_index = 0; output_index < actual.size(); ++output_index) {
                    const double delta = std::abs(static_cast<double>(actual[output_index] - expected[output_index]));
                    if (delta > kTightTolerance) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
            }
            outputs[thread_index] = workspace.output();
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0U);
    EXPECT_EQ(inference.weights_data(), weights_before);
    EXPECT_EQ(inference.biases_data(), biases_before);
    for (const bm::Vector& output : outputs) {
        expect_vector_near(output, expected, kTightTolerance);
    }
}

TEST(InferenceModel, PredictToReportsStatusInsteadOfThrowing) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    const bm::Vector input{2.0, -1.0};
    bm::Vector output(inference.output_size(), 0.0);
    bm::Vector scratch(inference.scratch_size(), 0.0);

    EXPECT_EQ(inference.predict_to(nullptr, input.size(), output.data(), output.size(), scratch.data(), scratch.size()),
              bm::InferenceStatus::null_input);
    EXPECT_EQ(inference.predict_to(input.data(), input.size(), nullptr, output.size(), scratch.data(), scratch.size()),
              bm::InferenceStatus::null_output);
    EXPECT_EQ(inference.predict_to(input.data(), 1, output.data(), output.size(), scratch.data(), scratch.size()),
              bm::InferenceStatus::invalid_input_size);
    EXPECT_EQ(inference.predict_to(input.data(), input.size(), output.data(), 1, scratch.data(), scratch.size()),
              bm::InferenceStatus::invalid_output_size);
    EXPECT_EQ(inference.predict_to(input.data(), input.size(), output.data(), output.size(), nullptr, scratch.size()),
              bm::InferenceStatus::null_scratch);
    EXPECT_EQ(inference.predict_to(input.data(), input.size(), output.data(), output.size(), scratch.data(), 1),
              bm::InferenceStatus::insufficient_scratch);
}

TEST(InferenceModel, BatchPredictToReportsStatusInsteadOfThrowing) {
    const auto inference = make_known_two_layer_model().to_inference_model();
    bm::Vector inputs{2.0, -1.0, 0.5, 3.0};
    bm::Vector outputs(4, 0.0);
    bm::Vector scratch(inference.scratch_size(), 0.0);

    EXPECT_EQ(inference.predict_batch_to(nullptr, 2, 2, outputs.data(), 2, scratch.data(), scratch.size()),
              bm::InferenceStatus::null_input);
    EXPECT_EQ(inference.predict_batch_to(inputs.data(), 2, 2, nullptr, 2, scratch.data(), scratch.size()),
              bm::InferenceStatus::null_output);
    EXPECT_EQ(inference.predict_batch_to(inputs.data(), 2, 1, outputs.data(), 2, scratch.data(), scratch.size()),
              bm::InferenceStatus::invalid_input_stride);
    EXPECT_EQ(inference.predict_batch_to(inputs.data(), 2, 2, outputs.data(), 1, scratch.data(), scratch.size()),
              bm::InferenceStatus::invalid_output_stride);
    EXPECT_EQ(inference.predict_batch_to(inputs.data(), 2, 2, outputs.data(), 2, nullptr, scratch.size()),
              bm::InferenceStatus::null_scratch);
    EXPECT_EQ(inference.predict_batch_to(inputs.data(), 2, 2, outputs.data(), 2, scratch.data(), 1),
              bm::InferenceStatus::insufficient_scratch);
    EXPECT_EQ(inference.predict_batch_to(nullptr, 0, 0, nullptr, 0, nullptr, 0),
              bm::InferenceStatus::ok);
}

TEST(InferenceModel, OneLayerModelDoesNotRequireScratch) {
    const auto inference = make_known_linear_model().to_inference_model();
    const bm::Vector input{3.0, 1.0};
    bm::Vector output(inference.output_size(), 0.0);

    EXPECT_EQ(inference.scratch_size(), 0U);
    EXPECT_EQ(inference.predict_to(input.data(), input.size(), output.data(), output.size(), nullptr, 0),
              bm::InferenceStatus::ok);
    expect_vector_near(output, {5.5}, kTightTolerance);

    output[0] = 0.0;
    inference.predict_unchecked_to(input.data(), output.data(), nullptr);
    expect_vector_near(output, {5.5}, kTightTolerance);
}

TEST(InferenceModel, SaveLoadRoundTripsFrozenModel) {
    const auto path = temp_test_path("brutal_mlp_inference_roundtrip.model");

    const auto inference = make_known_two_layer_model().to_inference_model();
    inference.save(path);
    const auto loaded = bm::InferenceModel::load(path);
    std::filesystem::remove(path);

    EXPECT_EQ(loaded.input_size(), inference.input_size());
    EXPECT_EQ(loaded.output_size(), inference.output_size());
    EXPECT_EQ(loaded.layer_count(), inference.layer_count());
    EXPECT_EQ(loaded.scratch_size(), inference.scratch_size());
    expect_vector_near(loaded.predict({2.0, -1.0}), inference.predict({2.0, -1.0}), kTightTolerance);
}

TEST(InferenceModel, RejectsInvalidFrozenParameters) {
    EXPECT_THROW((void)bm::InferenceModel::from_parameters({}), std::invalid_argument);
    EXPECT_THROW((void)bm::InferenceModel::from_parameters({
                     bm::LayerParameters{2, 1, bm::Activation::linear, {1.0}, {0.0}},
                 }),
                 std::invalid_argument);
    EXPECT_THROW((void)bm::InferenceModel::from_parameters({
                     bm::LayerParameters{2, 2, bm::Activation::softmax,
                                         {1.0, 0.0, 0.0, 1.0},
                                         {0.0, 0.0}},
                     bm::LayerParameters{2, 1, bm::Activation::linear, {1.0, 1.0}, {0.0}},
                 }),
                 std::invalid_argument);
}

TEST(CopyAndMove, CopyIsDeepAndIndependent) {
    auto original = make_known_linear_model();
    auto copy = original;

    auto copy_parameters = copy.parameters();
    copy_parameters[0].biases[0] = 10.0;
    copy.set_parameters(copy_parameters);

    expect_vector_near(original.predict({3.0, 1.0}), {5.5}, kTightTolerance);
    expect_vector_near(copy.predict({3.0, 1.0}), {15.0}, kTightTolerance);
}

TEST(Serialization, SaveLoadRoundTripsPredictionsAndMetadata) {
    const auto path = temp_test_path("brutal_mlp_roundtrip.model");

    auto model = make_known_linear_model();
    model.save(path);
    auto loaded = bm::Model::load(path);
    std::filesystem::remove(path);

    EXPECT_EQ(loaded.input_size(), model.input_size());
    EXPECT_EQ(loaded.output_size(), model.output_size());
    EXPECT_EQ(loaded.layer_count(), model.layer_count());
    EXPECT_EQ(loaded.loss(), model.loss());
    expect_vector_near(loaded.predict({3.0, 1.0}), model.predict({3.0, 1.0}), kTightTolerance);
}

TEST(Serialization, SaveLoadRoundTripsLossConfig) {
    const auto huber_path = temp_test_path("brutal_mlp_huber_roundtrip.model");
    auto huber = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::huber(bm::Scalar{0.25}))
                     .build();

    huber.save(huber_path);
    const auto loaded_huber = bm::TrainingModel::load(huber_path);
    std::filesystem::remove(huber_path);
    EXPECT_EQ(loaded_huber.loss(), bm::Loss::huber);
    EXPECT_NEAR(loaded_huber.loss_config().huber_delta, 0.25, kTightTolerance);

    const auto weighted_path = temp_test_path("brutal_mlp_weighted_roundtrip.model");
    auto weighted = make_known_two_output_model(
        bm::LossConfig::weighted_mean_squared_error({bm::Scalar{1}, bm::Scalar{3}}));

    weighted.save(weighted_path);
    const auto loaded_weighted = bm::TrainingModel::load(weighted_path);
    std::filesystem::remove(weighted_path);
    EXPECT_EQ(loaded_weighted.loss(), bm::Loss::weighted_mean_squared_error);
    expect_vector_near(loaded_weighted.loss_config().weights, {bm::Scalar{1}, bm::Scalar{3}}, kTightTolerance);
    EXPECT_NEAR(loaded_weighted.evaluate_loss({{bm::Scalar{3}}}, {{bm::Scalar{1}, bm::Scalar{10}}}),
                13.0,
                kTightTolerance);
}

TEST(Serialization, TextRoundTripsRegularizationAndDropout) {
    const auto path = temp_test_path("brutal_mlp_regularization_roundtrip.model");

    auto optimizer = bm::OptimizerConfig::adamw(bm::Scalar{0.003}, bm::Scalar{0.04});
    optimizer.l1 = bm::Scalar{0.001};
    optimizer.l2 = bm::Scalar{0.002};
    optimizer.max_norm = bm::Scalar{3};
    optimizer.gradient_clip_norm = bm::Scalar{2};
    optimizer.gradient_clip_value = bm::Scalar{0.5};
    optimizer.layer_gradient_clip_norm = bm::Scalar{1.5};

    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(2, bm::Activation::relu, bm::Scalar{0.25})
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(optimizer)
                     .seed(717)
                     .build();
    model.set_parameters({
        bm::LayerParameters{2, 2, bm::Activation::relu,
                            {bm::Scalar{0.5}, bm::Scalar{-0.25},
                             bm::Scalar{0.75}, bm::Scalar{0.125}},
                            {bm::Scalar{0.1}, bm::Scalar{-0.2}},
                            bm::Scalar{0.25}},
        bm::LayerParameters{2, 1, bm::Activation::linear,
                            {bm::Scalar{0.3}, bm::Scalar{-0.4}},
                            {bm::Scalar{0.05}}},
    });

    model.save(path);
    const auto loaded = bm::TrainingModel::load(path);
    std::filesystem::remove(path);

    EXPECT_EQ(loaded.optimizer().type, bm::OptimizerType::adamw);
    EXPECT_NEAR(loaded.optimizer().l1, 0.001, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().l2, 0.002, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().decoupled_weight_decay, 0.04, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().max_norm, 3.0, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().gradient_clip_norm, 2.0, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().gradient_clip_value, 0.5, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().layer_gradient_clip_norm, 1.5, kTightTolerance);
    const auto parameters = loaded.parameters();
    ASSERT_EQ(parameters.size(), 2U);
    EXPECT_NEAR(parameters[0].dropout_probability, 0.25, kTightTolerance);
    EXPECT_EQ(parameters[1].dropout_probability, bm::Scalar{0});
    expect_vector_near(loaded.predict({bm::Scalar{2}, bm::Scalar{-1}}),
                       model.predict({bm::Scalar{2}, bm::Scalar{-1}}),
                       kTightTolerance);
}

TEST(Serialization, BinaryTrainingRoundTripsAndReportsSelfDescribingInfo) {
    const auto path = temp_test_path("brutal_mlp_training_roundtrip.bmlp");
    const auto normalization = bm::NormalizationSpec::standard_score(
        {bm::Scalar{1}, bm::Scalar{2}},
        {bm::Scalar{2}, bm::Scalar{4}},
        {bm::Scalar{3}},
        {bm::Scalar{5}});

    auto optimizer = bm::OptimizerConfig::adamw(bm::Scalar{0.02}, bm::Scalar{0.015});
    optimizer.l1 = bm::Scalar{0.001};
    optimizer.l2 = bm::Scalar{0.002};
    optimizer.max_norm = bm::Scalar{4};
    optimizer.gradient_clip_norm = bm::Scalar{3};
    optimizer.gradient_clip_value = bm::Scalar{0.75};
    optimizer.layer_gradient_clip_norm = bm::Scalar{2};

    auto model = bm::TrainingModel::builder()
                     .input_size(2)
                     .add_layer(3, bm::Activation::tanh, bm::Scalar{0.2})
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::huber(bm::Scalar{0.75}))
                     .optimizer(optimizer)
                     .normalization(normalization)
                     .seed(777)
                     .build();

    bm::TrainingOptions training_options;
    training_options.epochs = 2;
    training_options.batch_size = 2;
    training_options.shuffle = false;
    training_options.seed = 44;
    training_options.validation_split = bm::Scalar{0};
    training_options.test_split = bm::Scalar{0};
    training_options.min_delta = bm::Scalar{0};
    training_options.early_stopping_monitor = bm::TrainingMonitor::training_loss;
    training_options.early_stopping_mode = bm::TrainingMonitorMode::maximize;
    training_options.early_stopping_cooldown = 2;
    training_options.learning_rate_schedule =
        bm::LearningRateScheduleConfig::step_decay(10, bm::Scalar{0.5});
    training_options.learning_rate_schedule.base_learning_rate = bm::Scalar{0.02};
    training_options.learning_rate_schedule.minimum_learning_rate = bm::Scalar{0.001};
    training_options.gradient_noise_stddev = bm::Scalar{0.0001};
    training_options.parallelism.execution = bm::ParallelExecution::worker_threads;
    training_options.parallelism.thread_count = 2;
    training_options.parallelism.minimum_parallel_samples = 8;
    training_options.restore_best_weights = false;
    model.fit({{bm::Scalar{1}, bm::Scalar{2}},
               {bm::Scalar{3}, bm::Scalar{6}},
               {bm::Scalar{-1}, bm::Scalar{-2}},
               {bm::Scalar{5}, bm::Scalar{10}}},
              {{bm::Scalar{3}}, {bm::Scalar{8}}, {bm::Scalar{-2}}, {bm::Scalar{12}}},
              training_options);

    bm::BinaryMetadata metadata;
    metadata.created_unix_time = 123456789;
    metadata.description = "training binary roundtrip";
    metadata.entries.push_back({"dataset", "unit"});
    metadata.entries.push_back({"purpose", "serialization"});
    model.save_binary(path, metadata);

    const auto info = bm::inspect_binary_model(path);
    EXPECT_EQ(info.version, 6U);
    EXPECT_EQ(info.scalar_type,
              sizeof(bm::Scalar) == sizeof(float) ? bm::BinaryScalarType::float32
                                                   : bm::BinaryScalarType::float64);
    EXPECT_EQ(info.model_kind, bm::BinaryModelKind::training);
    EXPECT_EQ(info.input_size, 2U);
    EXPECT_EQ(info.output_size, 1U);
    EXPECT_EQ(info.layer_count, 2U);
    EXPECT_EQ(info.weight_count, 9U);
    EXPECT_EQ(info.bias_count, 4U);
    EXPECT_EQ(info.seed, 777U);
    EXPECT_TRUE(info.has_training_options);
    EXPECT_EQ(info.training_options.epochs, training_options.epochs);
    EXPECT_EQ(info.training_options.batch_size, training_options.batch_size);
    EXPECT_FALSE(info.training_options.shuffle);
    EXPECT_EQ(info.training_options.seed, training_options.seed);
    EXPECT_EQ(info.training_options.early_stopping_monitor, training_options.early_stopping_monitor);
    EXPECT_EQ(info.training_options.early_stopping_mode, training_options.early_stopping_mode);
    EXPECT_EQ(info.training_options.early_stopping_cooldown, training_options.early_stopping_cooldown);
    EXPECT_EQ(info.training_options.learning_rate_schedule.type, bm::LearningRateSchedule::step_decay);
    EXPECT_NEAR(info.training_options.learning_rate_schedule.base_learning_rate, 0.02, kTightTolerance);
    EXPECT_NEAR(info.training_options.learning_rate_schedule.minimum_learning_rate, 0.001, kTightTolerance);
    EXPECT_EQ(info.training_options.learning_rate_schedule.step_size, 10U);
    EXPECT_NEAR(info.training_options.gradient_noise_stddev, 0.0001, kTightTolerance);
    EXPECT_EQ(info.training_options.parallelism.execution, bm::ParallelExecution::worker_threads);
    EXPECT_EQ(info.training_options.parallelism.thread_count, 2U);
    EXPECT_EQ(info.training_options.parallelism.minimum_parallel_samples, 8U);
    EXPECT_EQ(info.metadata.created_unix_time, metadata.created_unix_time);
    EXPECT_EQ(info.metadata.description, metadata.description);
    ASSERT_EQ(info.metadata.entries.size(), 2U);
    EXPECT_EQ(info.metadata.entries[0].key, "dataset");
    EXPECT_EQ(info.metadata.entries[0].value, "unit");
    EXPECT_NE(info.checksum, 0U);

    const auto loaded = bm::TrainingModel::load_binary(path);
    std::filesystem::remove(path);

    EXPECT_EQ(loaded.input_size(), model.input_size());
    EXPECT_EQ(loaded.output_size(), model.output_size());
    EXPECT_EQ(loaded.layer_count(), model.layer_count());
    EXPECT_EQ(loaded.loss(), bm::Loss::huber);
    EXPECT_NEAR(loaded.loss_config().huber_delta, 0.75, kTightTolerance);
    EXPECT_EQ(loaded.optimizer().type, bm::OptimizerType::adamw);
    EXPECT_NEAR(loaded.optimizer().learning_rate, 0.02, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().l1, 0.001, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().l2, 0.002, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().decoupled_weight_decay, 0.015, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().max_norm, 4.0, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().gradient_clip_norm, 3.0, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().gradient_clip_value, 0.75, kTightTolerance);
    EXPECT_NEAR(loaded.optimizer().layer_gradient_clip_norm, 2.0, kTightTolerance);
    const auto loaded_parameters = loaded.parameters();
    ASSERT_EQ(loaded_parameters.size(), 2U);
    EXPECT_NEAR(loaded_parameters[0].dropout_probability, 0.2, kTightTolerance);
    EXPECT_EQ(loaded_parameters[1].dropout_probability, bm::Scalar{0});
    ASSERT_EQ(loaded.normalization().input_features.size(), 2U);
    EXPECT_EQ(loaded.normalization().input_features[0].mode, bm::NormalizationMode::standard_score);
    expect_vector_near(loaded.predict({bm::Scalar{4}, bm::Scalar{8}}),
                       model.predict({bm::Scalar{4}, bm::Scalar{8}}),
                       kTightTolerance);
}

TEST(Serialization, BinaryInferenceRoundTripsFrozenModel) {
    const auto path = temp_test_path("brutal_mlp_inference_roundtrip.bmlp");
    const auto inference = make_known_two_layer_model().to_inference_model();

    bm::BinaryMetadata metadata;
    metadata.created_unix_time = 222222222;
    metadata.description = "inference binary roundtrip";
    inference.save_binary(path, metadata);

    const auto info = bm::inspect_binary_model(path);
    EXPECT_EQ(info.model_kind, bm::BinaryModelKind::inference);
    EXPECT_EQ(info.input_size, inference.input_size());
    EXPECT_EQ(info.output_size, inference.output_size());
    EXPECT_EQ(info.layer_count, inference.layer_count());
    EXPECT_FALSE(info.has_training_options);
    EXPECT_EQ(info.seed, 0U);
    EXPECT_EQ(info.metadata.description, metadata.description);

    const auto loaded = bm::InferenceModel::load_binary(path);
    std::filesystem::remove(path);

    expect_vector_near(loaded.predict({bm::Scalar{2}, bm::Scalar{-1}}),
                       inference.predict({bm::Scalar{2}, bm::Scalar{-1}}),
                       kTightTolerance);
    EXPECT_EQ(loaded.weight_count(), inference.weight_count());
    EXPECT_EQ(loaded.bias_count(), inference.bias_count());
}

TEST(Serialization, BinaryChecksumRejectsCorruption) {
    const auto path = temp_test_path("brutal_mlp_corrupt.bmlp");
    auto model = make_known_linear_model();
    model.save_binary(path);

    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(file);
        file.seekg(40, std::ios::beg);
        char byte = 0;
        file.get(byte);
        byte = static_cast<char>(byte ^ 0x5A);
        file.seekp(40, std::ios::beg);
        file.put(byte);
    }

    EXPECT_THROW((void)bm::inspect_binary_model(path), std::runtime_error);
    EXPECT_THROW((void)bm::TrainingModel::load_binary(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Serialization, TrainingCheckpointRestoresOptimizerStateAndResumes) {
    const auto path = temp_test_path("brutal_mlp_resume_checkpoint.bmlp");
    const bm::Matrix inputs{{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}, {3.0}};
    const bm::Matrix targets{{-5.0}, {-3.0}, {-1.0}, {1.0}, {3.0}, {5.0}};

    bm::TrainingOptions full_options;
    full_options.epochs = 6;
    full_options.batch_size = 2;
    full_options.shuffle = false;
    full_options.restore_best_weights = false;
    full_options.learning_rate_schedule = bm::LearningRateScheduleConfig::step_decay(2, bm::Scalar{0.5});
    full_options.learning_rate_schedule.base_learning_rate = bm::Scalar{0.01};
    full_options.parallelism.execution = bm::ParallelExecution::worker_threads;
    full_options.parallelism.thread_count = 2;
    full_options.parallelism.minimum_parallel_samples = 4;

    auto full = bm::TrainingModel::builder()
                    .input_size(1)
                    .add_layer(3, bm::Activation::tanh)
                    .add_layer(1, bm::Activation::linear)
                    .loss(bm::Loss::mean_squared_error)
                    .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.01}))
                    .seed(901)
                    .build();
    full.fit(inputs, targets, full_options);

    auto partial = bm::TrainingModel::builder()
                       .input_size(1)
                       .add_layer(3, bm::Activation::tanh)
                       .add_layer(1, bm::Activation::linear)
                       .loss(bm::Loss::mean_squared_error)
                       .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.01}))
                       .seed(901)
                       .build();
    auto partial_options = full_options;
    partial_options.epochs = 3;
    const auto partial_history = partial.fit(inputs, targets, partial_options);
    ASSERT_EQ(partial_history.training_loss.size(), 3U);
    EXPECT_EQ(partial.completed_epochs(), 3U);

    bm::TrainingCheckpointInfo checkpoint_info;
    checkpoint_info.completed_epochs = partial.completed_epochs();
    checkpoint_info.has_metric = true;
    checkpoint_info.metric = partial_history.training_loss.back();
    checkpoint_info.training_options = full_options;
    checkpoint_info.metadata.created_unix_time = 333333333;
    checkpoint_info.metadata.description = "resume checkpoint";
    partial.save_checkpoint(path, checkpoint_info);

    bm::TrainingCheckpointInfo loaded_info;
    auto resumed = bm::TrainingModel::load_checkpoint(path, &loaded_info);
    EXPECT_EQ(resumed.completed_epochs(), 3U);
    EXPECT_EQ(loaded_info.completed_epochs, 3U);
    EXPECT_TRUE(loaded_info.has_metric);
    EXPECT_EQ(loaded_info.training_options.epochs, full_options.epochs);
    EXPECT_EQ(loaded_info.training_options.learning_rate_schedule.type, bm::LearningRateSchedule::step_decay);
    EXPECT_EQ(loaded_info.training_options.parallelism.execution, bm::ParallelExecution::worker_threads);
    EXPECT_EQ(loaded_info.training_options.parallelism.thread_count, 2U);
    EXPECT_EQ(loaded_info.training_options.parallelism.minimum_parallel_samples, 4U);
    EXPECT_TRUE(loaded_info.learning_rate_state.initialized);
    EXPECT_NEAR(loaded_info.learning_rate_state.base_learning_rate, 0.01, kTightTolerance);
    EXPECT_EQ(loaded_info.metadata.description, checkpoint_info.metadata.description);

    const auto inspect = bm::inspect_binary_model(path);
    EXPECT_EQ(inspect.model_kind, bm::BinaryModelKind::checkpoint);
    EXPECT_EQ(inspect.completed_epochs, 3U);
    EXPECT_EQ(inspect.optimizer_step, 9U);
    EXPECT_TRUE(inspect.has_checkpoint_metric);
    EXPECT_TRUE(inspect.has_learning_rate_state);
    EXPECT_NEAR(inspect.learning_rate_state.current_learning_rate, 0.005, kTightTolerance);

    const auto resume_history = resumed.fit(inputs, targets, loaded_info.training_options);
    std::filesystem::remove(path);
    ASSERT_EQ(resume_history.training_loss.size(), 3U);
    EXPECT_EQ(resumed.completed_epochs(), full_options.epochs);

    expect_vector_near(resumed.predict({bm::Scalar{1.5}}),
                       full.predict({bm::Scalar{1.5}}),
                       kTightTolerance);
    expect_vector_near(resumed.predict({bm::Scalar{-0.5}}),
                       full.predict({bm::Scalar{-0.5}}),
                       kTightTolerance);
}

TEST(Serialization, TrainingOptionsWriteBestAndLatestCheckpoints) {
    const auto best_path = temp_test_path("brutal_mlp_best_checkpoint.bmlp");
    const auto latest_path = temp_test_path("brutal_mlp_latest_checkpoint.bmlp");
    std::filesystem::remove(best_path);
    std::filesystem::remove(latest_path);

    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::Loss::mean_squared_error)
                     .optimizer(bm::OptimizerConfig::sgd(bm::Scalar{0.02}))
                     .seed(902)
                     .build();

    bm::TrainingOptions options;
    options.epochs = 4;
    options.batch_size = 2;
    options.shuffle = false;
    options.restore_best_weights = false;
    options.early_stopping_monitor = bm::TrainingMonitor::training_loss;
    options.early_stopping_mode = bm::TrainingMonitorMode::minimize;
    options.early_stopping_cooldown = 1;
    options.learning_rate_schedule = bm::LearningRateScheduleConfig::step_decay(2, bm::Scalar{0.5});
    options.learning_rate_schedule.base_learning_rate = bm::Scalar{0.02};
    options.parallelism.execution = bm::ParallelExecution::worker_threads;
    options.parallelism.thread_count = 3;
    options.parallelism.minimum_parallel_samples = 5;
    options.best_checkpoint_path = best_path;
    options.latest_checkpoint_path = latest_path;

    const auto history = model.fit({{-2.0}, {-1.0}, {0.0}, {1.0}},
                                   {{-5.0}, {-3.0}, {-1.0}, {1.0}},
                                   options);

    ASSERT_EQ(history.training_loss.size(), options.epochs);
    ASSERT_TRUE(std::filesystem::exists(best_path));
    ASSERT_TRUE(std::filesystem::exists(latest_path));

    const auto best_info = bm::inspect_binary_model(best_path);
    const auto latest_info = bm::inspect_binary_model(latest_path);
    EXPECT_EQ(best_info.model_kind, bm::BinaryModelKind::checkpoint);
    EXPECT_EQ(latest_info.model_kind, bm::BinaryModelKind::checkpoint);
    EXPECT_TRUE(best_info.checkpoint_best);
    EXPECT_FALSE(latest_info.checkpoint_best);
    EXPECT_TRUE(best_info.has_checkpoint_metric);
    EXPECT_TRUE(latest_info.has_checkpoint_metric);
    EXPECT_EQ(latest_info.completed_epochs, options.epochs);
    EXPECT_EQ(latest_info.training_options.best_checkpoint_path, best_path);
    EXPECT_EQ(latest_info.training_options.latest_checkpoint_path, latest_path);
    EXPECT_EQ(latest_info.training_options.early_stopping_monitor, options.early_stopping_monitor);
    EXPECT_EQ(latest_info.training_options.early_stopping_mode, options.early_stopping_mode);
    EXPECT_EQ(latest_info.training_options.early_stopping_cooldown, options.early_stopping_cooldown);
    EXPECT_EQ(latest_info.training_options.learning_rate_schedule.type, bm::LearningRateSchedule::step_decay);
    EXPECT_EQ(latest_info.training_options.parallelism.execution, bm::ParallelExecution::worker_threads);
    EXPECT_EQ(latest_info.training_options.parallelism.thread_count, 3U);
    EXPECT_EQ(latest_info.training_options.parallelism.minimum_parallel_samples, 5U);
    EXPECT_TRUE(latest_info.has_learning_rate_state);
    EXPECT_TRUE(latest_info.learning_rate_state.initialized);
    EXPECT_NEAR(latest_info.learning_rate_state.base_learning_rate, 0.02, kTightTolerance);
    EXPECT_NEAR(latest_info.learning_rate_state.current_learning_rate, 0.01, kTightTolerance);

    bm::TrainingCheckpointInfo best_loaded_info;
    bm::TrainingCheckpointInfo latest_loaded_info;
    const auto best_model = bm::TrainingModel::load_checkpoint(best_path, &best_loaded_info);
    const auto latest_model = bm::TrainingModel::load_checkpoint(latest_path, &latest_loaded_info);
    EXPECT_TRUE(best_loaded_info.best_checkpoint);
    EXPECT_FALSE(latest_loaded_info.best_checkpoint);
    EXPECT_TRUE(latest_loaded_info.learning_rate_state.initialized);
    EXPECT_EQ(latest_model.completed_epochs(), options.epochs);
    EXPECT_GT(best_model.completed_epochs(), 0U);

    std::filesystem::remove(best_path);
    std::filesystem::remove(latest_path);
}

TEST(Serialization, RejectsCustomLossSave) {
    const auto path = temp_test_path("brutal_mlp_custom_loss.model");
    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::custom(custom_mse_value, custom_mse_gradient))
                     .build();

    EXPECT_THROW(model.save(path), std::runtime_error);
    EXPECT_THROW(model.save_binary(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Serialization, LoadRejectsMissingOrInvalidFiles) {
    EXPECT_THROW((void)bm::Model::load("does-not-exist.brutal_mlp"), std::runtime_error);

    const auto path = temp_test_path("brutal_mlp_invalid.model");
    {
        std::ofstream file(path);
        file << "NOPE\n";
    }
    EXPECT_THROW((void)bm::Model::load(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(NonRegression, SameSeedProducesSameInitialWeights) {
    const auto build = [] {
        return bm::TrainingModel::builder()
            .input_size(3)
            .add_layer(5, bm::Activation::relu)
            .add_layer(2, bm::Activation::tanh)
            .add_layer(1, bm::Activation::linear)
            .loss(bm::Loss::mean_squared_error)
            .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.01}))
            .initialization(bm::InitializationConfig::xavier_uniform(bm::Scalar{0.01}))
            .seed(2026)
            .build();
    };

    const auto first = build();
    const auto second = build();

    expect_parameters_near(first.parameters(), second.parameters(), 0.0);
    expect_vector_near(first.predict({bm::Scalar{1}, bm::Scalar{-2}, bm::Scalar{0.5}}),
                       second.predict({bm::Scalar{1}, bm::Scalar{-2}, bm::Scalar{0.5}}),
                       0.0);
}

TEST(NonRegression, SaveLoadPreservesTrainingAndInferencePredictions) {
    const auto training_path = temp_test_path("brutal_mlp_regression_training.model");
    const auto training_binary_path = temp_test_path("brutal_mlp_regression_training.bmlp");
    const auto inference_path = temp_test_path("brutal_mlp_regression_inference.model");
    const auto inference_binary_path = temp_test_path("brutal_mlp_regression_inference.bmlp");

    const auto training = make_known_two_layer_model();
    const auto inference = training.to_inference_model();
    const bm::Matrix probes{{bm::Scalar{2}, bm::Scalar{-1}},
                            {bm::Scalar{-0.5}, bm::Scalar{0.25}},
                            {bm::Scalar{1.25}, bm::Scalar{3}}};
    const bm::Matrix expected_training = training.predict_batch(probes);
    const bm::Matrix expected_inference = inference.predict_batch(probes);

    training.save(training_path);
    training.save_binary(training_binary_path);
    inference.save(inference_path);
    inference.save_binary(inference_binary_path);

    const auto loaded_training = bm::TrainingModel::load(training_path);
    const auto loaded_binary_training = bm::TrainingModel::load_binary(training_binary_path);
    const auto loaded_inference = bm::InferenceModel::load(inference_path);
    const auto loaded_binary_inference = bm::InferenceModel::load_binary(inference_binary_path);

    std::filesystem::remove(training_path);
    std::filesystem::remove(training_binary_path);
    std::filesystem::remove(inference_path);
    std::filesystem::remove(inference_binary_path);

    expect_matrix_near(loaded_training.predict_batch(probes), expected_training, kTightTolerance);
    expect_matrix_near(loaded_binary_training.predict_batch(probes), expected_training, kTightTolerance);
    expect_matrix_near(loaded_inference.predict_batch(probes), expected_inference, kTightTolerance);
    expect_matrix_near(loaded_binary_inference.predict_batch(probes), expected_inference, kTightTolerance);
}

TEST(NonRegression, SameTrainingRunIsStableWithinTolerance) {
    const auto build = [] {
        return bm::TrainingModel::builder()
            .input_size(2)
            .add_layer(4, bm::Activation::tanh)
            .add_layer(1, bm::Activation::linear)
            .loss(bm::Loss::mean_squared_error)
            .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.02}))
            .seed(303)
            .build();
    };

    const bm::Matrix inputs{{bm::Scalar{-2}, bm::Scalar{1}},
                            {bm::Scalar{-1}, bm::Scalar{0}},
                            {bm::Scalar{0}, bm::Scalar{1}},
                            {bm::Scalar{1}, bm::Scalar{2}},
                            {bm::Scalar{2}, bm::Scalar{3}},
                            {bm::Scalar{3}, bm::Scalar{4}}};
    const bm::Matrix targets{{bm::Scalar{-5}},
                             {bm::Scalar{-3}},
                             {bm::Scalar{-1}},
                             {bm::Scalar{1}},
                             {bm::Scalar{3}},
                             {bm::Scalar{5}}};

    auto options = quick_options(120, 2);
    options.seed = 404;
    options.restore_best_weights = false;

    auto first = build();
    auto second = build();
    const auto first_history = first.fit(inputs, targets, options);
    const auto second_history = second.fit(inputs, targets, options);

    ASSERT_EQ(first_history.training_loss.size(), second_history.training_loss.size());
    for (std::size_t i = 0; i < first_history.training_loss.size(); ++i) {
        EXPECT_NEAR(first_history.training_loss[i], second_history.training_loss[i], kTightTolerance)
            << "epoch " << i;
    }
    expect_parameters_near(first.parameters(), second.parameters(), kTightTolerance);
    expect_matrix_near(first.predict_batch(inputs), second.predict_batch(inputs), kTightTolerance);
}

TEST(NonRegression, TrainingStaysFiniteAndLossFallsOnSimpleDatasets) {
    {
        auto model = bm::TrainingModel::builder()
            .input_size(1)
            .add_layer(1, bm::Activation::linear)
            .loss(bm::Loss::mean_squared_error)
            .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.05}))
            .seed(505)
            .build();

        const bm::Matrix inputs{{bm::Scalar{-2}},
                                {bm::Scalar{-1}},
                                {bm::Scalar{0}},
                                {bm::Scalar{1}},
                                {bm::Scalar{2}}};
        const bm::Matrix targets{{bm::Scalar{-5}},
                                 {bm::Scalar{-3}},
                                 {bm::Scalar{-1}},
                                 {bm::Scalar{1}},
                                 {bm::Scalar{3}}};
        const bm::Scalar before = model.evaluate_loss(inputs, targets);
        auto options = quick_options(200, 5);
        options.shuffle = false;
        options.restore_best_weights = false;
        const auto history = model.fit(inputs, targets, options);
        const bm::Scalar after = model.evaluate_loss(inputs, targets);

        expect_history_finite(history);
        expect_parameters_finite(model.parameters());
        EXPECT_LT(history.training_loss.back(), history.training_loss.front());
        EXPECT_LT(after, before);
        EXPECT_TRUE(std::isfinite(model.predict({bm::Scalar{1.5}})[0]));
    }

    {
        auto model = bm::TrainingModel::builder()
            .input_size(2)
            .add_layer(1, bm::Activation::sigmoid)
            .loss(bm::Loss::binary_cross_entropy)
            .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.08}))
            .seed(606)
            .build();

        const bm::Matrix inputs{{bm::Scalar{0}, bm::Scalar{0}},
                                {bm::Scalar{0}, bm::Scalar{1}},
                                {bm::Scalar{1}, bm::Scalar{0}},
                                {bm::Scalar{1}, bm::Scalar{1}}};
        const bm::Matrix targets{{bm::Scalar{0}},
                                 {bm::Scalar{1}},
                                 {bm::Scalar{1}},
                                 {bm::Scalar{1}}};
        const bm::Scalar before = model.evaluate_loss(inputs, targets);
        auto options = quick_options(250, 4);
        options.shuffle = false;
        options.restore_best_weights = false;
        const auto history = model.fit(inputs, targets, options);
        const bm::Scalar after = model.evaluate_loss(inputs, targets);

        expect_history_finite(history);
        expect_parameters_finite(model.parameters());
        EXPECT_LT(history.training_loss.back(), history.training_loss.front());
        EXPECT_LT(after, before);
        EXPECT_TRUE(std::isfinite(model.predict({bm::Scalar{1}, bm::Scalar{0}})[0]));
    }
}

TEST(NonRegression, NumericGradientCheckPassesForSmallNetwork) {
    auto model = bm::TrainingModel::builder()
        .input_size(2)
        .add_layer(2, bm::Activation::tanh)
        .add_layer(1, bm::Activation::linear)
        .loss(bm::Loss::mean_squared_error)
        .optimizer(bm::OptimizerConfig::sgd(kGradientLearningRate, bm::Scalar{0}))
        .seed(707)
        .build();

    const std::vector<bm::LayerParameters> parameters{
        bm::LayerParameters{2, 2, bm::Activation::tanh,
                            {bm::Scalar{0.23}, bm::Scalar{-0.31},
                             bm::Scalar{0.41}, bm::Scalar{0.12}},
                            {bm::Scalar{0.07}, bm::Scalar{-0.04}}},
        bm::LayerParameters{2, 1, bm::Activation::linear,
                            {bm::Scalar{0.29}, bm::Scalar{-0.18}},
                            {bm::Scalar{0.03}}},
    };
    const bm::Matrix inputs{{bm::Scalar{0.25}, bm::Scalar{-0.5}},
                            {bm::Scalar{-0.2}, bm::Scalar{0.45}}};
    const bm::Matrix targets{{bm::Scalar{0.1}}, {bm::Scalar{-0.2}}};

    expect_backprop_matches_finite_difference(model, parameters, inputs, targets, kGradientLearningRate);
}

TEST(Determinism, SameSeedProducesSameInitialPredictions) {
    auto first = bm::Model::builder()
                     .input_size(3)
                     .add_layer(4, bm::Activation::relu)
                     .add_layer(1, bm::Activation::linear)
                     .seed(123)
                     .build();
    auto second = bm::Model::builder()
                      .input_size(3)
                      .add_layer(4, bm::Activation::relu)
                      .add_layer(1, bm::Activation::linear)
                      .seed(123)
                      .build();

    expect_vector_near(first.predict({1.0, 2.0, 3.0}), second.predict({1.0, 2.0, 3.0}), kTightTolerance);
}

TEST(Determinism, DifferentSeedsUsuallyProduceDifferentInitialPredictions) {
    auto first = bm::Model::builder()
                     .input_size(3)
                     .add_layer(4, bm::Activation::relu)
                     .add_layer(1, bm::Activation::linear)
                     .seed(123)
                     .build();
    auto second = bm::Model::builder()
                      .input_size(3)
                      .add_layer(4, bm::Activation::relu)
                      .add_layer(1, bm::Activation::linear)
                      .seed(124)
                      .build();

    EXPECT_NE(first.predict({1.0, 2.0, 3.0})[0], second.predict({1.0, 2.0, 3.0})[0]);
}

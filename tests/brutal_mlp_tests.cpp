#include "brutal_mlp/brutal_mlp.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
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

void expect_vector_near(const bm::Vector& actual, const bm::Vector& expected, double tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
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

    EXPECT_EQ(bm::to_string(bm::InferenceStatus::ok), "ok");
    EXPECT_EQ(bm::to_string(bm::InferenceStatus::invalid_input_stride), "invalid_input_stride");
    EXPECT_EQ(bm::to_string(bm::InferenceStatus::invalid_output_stride), "invalid_output_stride");
    EXPECT_EQ(bm::to_string(bm::InferenceStatus::insufficient_scratch), "insufficient_scratch");
}

TEST(StringConversions, RejectUnknownValues) {
    EXPECT_THROW((void)bm::activation_from_string("swish"), std::invalid_argument);
    EXPECT_THROW((void)bm::loss_from_string("hinge"), std::invalid_argument);
    EXPECT_THROW((void)bm::metric_from_string("balanced_accuracy"), std::invalid_argument);
    EXPECT_THROW((void)bm::averaging_from_string("weighted"), std::invalid_argument);
    EXPECT_THROW((void)bm::optimizer_type_from_string("rmsprop"), std::invalid_argument);
}

TEST(OptimizerConfig, FactoriesExposeExpectedDefaults) {
    const auto adam = bm::OptimizerConfig::adam(0.02);
    EXPECT_EQ(adam.type, bm::OptimizerType::adam);
    EXPECT_NEAR(adam.learning_rate, 0.02, kTightTolerance);
    EXPECT_NEAR(adam.beta1, 0.9, kTightTolerance);
    EXPECT_NEAR(adam.beta2, 0.999, kTightTolerance);

    const auto sgd = bm::OptimizerConfig::sgd(0.3, 0.4);
    EXPECT_EQ(sgd.type, bm::OptimizerType::sgd);
    EXPECT_NEAR(sgd.learning_rate, 0.3, kTightTolerance);
    EXPECT_NEAR(sgd.momentum, 0.4, kTightTolerance);
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
    const auto path = std::filesystem::temp_directory_path() / "brutal_mlp_streaming_dataset.csv";
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
    const auto training_path = std::filesystem::temp_directory_path() / "brutal_mlp_training_norm.model";
    const auto inference_path = std::filesystem::temp_directory_path() / "brutal_mlp_inference_norm.model";

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
}

TEST(InferenceModel, SaveLoadRoundTripsFrozenModel) {
    const auto path = std::filesystem::temp_directory_path() / "brutal_mlp_inference_roundtrip.model";

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
    const auto path = std::filesystem::temp_directory_path() / "brutal_mlp_roundtrip.model";

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
    const auto huber_path = std::filesystem::temp_directory_path() / "brutal_mlp_huber_roundtrip.model";
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

    const auto weighted_path = std::filesystem::temp_directory_path() / "brutal_mlp_weighted_roundtrip.model";
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

TEST(Serialization, RejectsCustomLossSave) {
    const auto path = std::filesystem::temp_directory_path() / "brutal_mlp_custom_loss.model";
    auto model = bm::TrainingModel::builder()
                     .input_size(1)
                     .add_layer(1, bm::Activation::linear)
                     .loss(bm::LossConfig::custom(custom_mse_value, custom_mse_gradient))
                     .build();

    EXPECT_THROW(model.save(path), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Serialization, LoadRejectsMissingOrInvalidFiles) {
    EXPECT_THROW((void)bm::Model::load("does-not-exist.brutal_mlp"), std::runtime_error);

    const auto path = std::filesystem::temp_directory_path() / "brutal_mlp_invalid.model";
    {
        std::ofstream file(path);
        file << "NOPE\n";
    }
    EXPECT_THROW((void)bm::Model::load(path), std::runtime_error);
    std::filesystem::remove(path);
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

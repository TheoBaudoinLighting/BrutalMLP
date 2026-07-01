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
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::binary_cross_entropy)),
              bm::Loss::binary_cross_entropy);
    EXPECT_EQ(bm::loss_from_string(bm::to_string(bm::Loss::categorical_cross_entropy)),
              bm::Loss::categorical_cross_entropy);

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

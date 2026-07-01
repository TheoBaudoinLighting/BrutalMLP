#include "brutal_mlp/brutal_mlp.hpp"

#include <gtest/gtest.h>

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
    static_assert(noexcept(std::declval<const bm::InferenceModel&>().predict_to(nullptr, 0, nullptr, 0, nullptr, 0)),
                  "InferenceModel::predict_to must remain noexcept");
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
    EXPECT_DOUBLE_EQ(adam.learning_rate, 0.02);
    EXPECT_DOUBLE_EQ(adam.beta1, 0.9);
    EXPECT_DOUBLE_EQ(adam.beta2, 0.999);

    const auto sgd = bm::OptimizerConfig::sgd(0.3, 0.4);
    EXPECT_EQ(sgd.type, bm::OptimizerType::sgd);
    EXPECT_DOUBLE_EQ(sgd.learning_rate, 0.3);
    EXPECT_DOUBLE_EQ(sgd.momentum, 0.4);
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
    expect_vector_near(prediction, {5.5}, 1e-12);
}

TEST(Prediction, BatchPredictionMatchesIndividualPrediction) {
    const auto model = make_known_linear_model();
    const bm::Matrix inputs{{3.0, 1.0}, {0.0, 0.0}, {-1.0, 2.0}};
    const bm::Matrix batch = model.predict_batch(inputs);

    ASSERT_EQ(batch.size(), inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        expect_vector_near(batch[i], model.predict(inputs[i]), 1e-12);
    }
}

TEST(Prediction, RejectsInvalidInputs) {
    const auto model = make_known_linear_model();
    EXPECT_THROW((void)model.predict({1.0}), std::invalid_argument);
    EXPECT_THROW((void)model.predict({1.0, std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
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

    EXPECT_NEAR(model.predict({1000.0})[0], 1.0, 1e-12);
    EXPECT_NEAR(model.predict({-1000.0})[0], 0.0, 1e-12);
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
    EXPECT_NEAR(sum, 1.0, 1e-12);
    EXPECT_GT(prediction[2], prediction[1]);
    EXPECT_GT(prediction[1], prediction[0]);
}

TEST(Losses, MeanSquaredErrorMatchesKnownValue) {
    const auto model = make_known_linear_model();
    EXPECT_NEAR(model.evaluate_loss({{3.0, 1.0}}, {{4.5}}), 1.0, 1e-12);
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

    EXPECT_NEAR(model.evaluate_loss({{99.0}}, {{1.0}}), std::log(2.0), 1e-12);
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

    EXPECT_NEAR(model.evaluate_loss({{0.0}}, {{0.0, 1.0, 0.0}}), std::log(3.0), 1e-12);
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

    expect_vector_near(model.predict({3.0, 1.0}), {3.0}, 1e-12);
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

    expect_vector_near(model.predict({3.0, 1.0}), {5.5}, 1e-12);
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

    expect_vector_near(inference.predict(input), training.predict(input), 1e-12);

    bm::Vector output(inference.output_size(), 0.0);
    bm::Vector scratch(inference.scratch_size(), 0.0);
    const auto status =
        inference.predict_to(input.data(), input.size(), output.data(), output.size(), scratch.data(), scratch.size());

    EXPECT_EQ(status, bm::InferenceStatus::ok);
    expect_vector_near(output, training.predict(input), 1e-12);
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

TEST(InferenceModel, OneLayerModelDoesNotRequireScratch) {
    const auto inference = make_known_linear_model().to_inference_model();
    const bm::Vector input{3.0, 1.0};
    bm::Vector output(inference.output_size(), 0.0);

    EXPECT_EQ(inference.scratch_size(), 0U);
    EXPECT_EQ(inference.predict_to(input.data(), input.size(), output.data(), output.size(), nullptr, 0),
              bm::InferenceStatus::ok);
    expect_vector_near(output, {5.5}, 1e-12);
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
    expect_vector_near(loaded.predict({2.0, -1.0}), inference.predict({2.0, -1.0}), 1e-12);
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

    expect_vector_near(original.predict({3.0, 1.0}), {5.5}, 1e-12);
    expect_vector_near(copy.predict({3.0, 1.0}), {15.0}, 1e-12);
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
    expect_vector_near(loaded.predict({3.0, 1.0}), model.predict({3.0, 1.0}), 1e-12);
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

    expect_vector_near(first.predict({1.0, 2.0, 3.0}), second.predict({1.0, 2.0, 3.0}), 1e-12);
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

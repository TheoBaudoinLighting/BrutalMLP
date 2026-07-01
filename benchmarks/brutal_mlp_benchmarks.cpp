#include "brutal_mlp/brutal_mlp.hpp"

#include <benchmark/benchmark.h>

#include <cmath>

namespace bm = brutal_mlp;

namespace {

bm::Matrix make_inputs(std::size_t rows, std::size_t columns) {
    bm::Matrix result(rows, bm::Vector(columns, 0.0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row][column] = std::sin(static_cast<double>(row + column) * 0.01);
        }
    }
    return result;
}

bm::Vector make_flat_inputs(std::size_t rows, std::size_t columns) {
    bm::Vector result(rows * columns, 0.0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            result[row * columns + column] = std::sin(static_cast<double>(row + column) * 0.01);
        }
    }
    return result;
}

bm::Matrix make_regression_targets(const bm::Matrix& inputs, std::size_t outputs) {
    bm::Matrix targets(inputs.size(), bm::Vector(outputs, 0.0));
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        for (std::size_t out = 0; out < outputs; ++out) {
            double value = 0.0;
            for (std::size_t in = 0; in < inputs[row].size(); ++in) {
                value += inputs[row][in] * static_cast<double>((in + out) % 7 + 1) * 0.01;
            }
            targets[row][out] = value;
        }
    }
    return targets;
}

bm::TrainingModel make_training_model(std::size_t inputs, std::size_t hidden, std::size_t outputs) {
    return bm::TrainingModel::builder()
        .input_size(inputs)
        .add_layer(hidden, bm::Activation::relu)
        .add_layer(hidden, bm::Activation::relu)
        .add_layer(outputs, bm::Activation::linear)
        .loss(bm::Loss::mean_squared_error)
        .optimizer(bm::OptimizerConfig::adam(0.001))
        .seed(42)
        .build();
}

bm::InferenceModel make_inference_model(std::size_t inputs, std::size_t hidden, std::size_t outputs) {
    return make_training_model(inputs, hidden, outputs).to_inference_model();
}

} // namespace

static void BM_PredictSingle(benchmark::State& state) {
    const auto input_size = static_cast<std::size_t>(state.range(0));
    const auto hidden_size = static_cast<std::size_t>(state.range(1));
    auto model = make_inference_model(input_size, hidden_size, 8);
    auto input = make_inputs(1, input_size).front();
    bm::InferenceWorkspace workspace(model);

    for (auto _ : state) {
        const auto status = model.predict_to(input.data(), input.size(), workspace);
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(workspace.output_data());
    }

    state.SetItemsProcessed(state.iterations());
}

static void BM_PredictBatch(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto model = make_inference_model(32, 64, 8);
    auto inputs = make_flat_inputs(batch_size, 32);
    bm::Vector outputs(batch_size * model.output_size(), 0.0);
    bm::Vector scratch(model.scratch_size(), 0.0);

    for (auto _ : state) {
        const auto status = model.predict_batch_to(inputs.data(),
                                                   batch_size,
                                                   model.input_size(),
                                                   outputs.data(),
                                                   model.output_size(),
                                                   scratch.data(),
                                                   scratch.size());
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(outputs.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
}

static void BM_FitEpoch(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    const auto inputs = make_inputs(128, 16);
    const auto targets = make_regression_targets(inputs, 4);

    for (auto _ : state) {
        state.PauseTiming();
        auto model = make_training_model(16, 32, 4);
        bm::TrainingOptions options;
        options.epochs = 1;
        options.batch_size = batch_size;
        options.shuffle = false;
        state.ResumeTiming();

        auto history = model.fit(inputs, targets, options);
        benchmark::DoNotOptimize(history);
    }

    state.SetItemsProcessed(state.iterations() * 128);
}

BENCHMARK(BM_PredictSingle)->Args({16, 32})->Args({64, 128});
BENCHMARK(BM_PredictBatch)->Arg(1)->Arg(32)->Arg(256);
BENCHMARK(BM_FitEpoch)->Arg(8)->Arg(32)->Arg(128);

BENCHMARK_MAIN();

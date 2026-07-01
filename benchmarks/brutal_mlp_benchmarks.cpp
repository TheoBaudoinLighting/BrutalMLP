#include "brutal_mlp/brutal_mlp.hpp"

#include <benchmark/benchmark.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <string>
#include <vector>

namespace {

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocation_count{0};
std::atomic<std::size_t> g_temp_path_counter{0};

} // namespace

void* operator new(std::size_t size) {
    if (g_count_allocations.load(std::memory_order_relaxed)) {
        g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    void* pointer = std::malloc(size == 0 ? 1 : size);
    if (!pointer) {
        throw std::bad_alloc();
    }
    return pointer;
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void* operator new[](std::size_t size) {
    if (g_count_allocations.load(std::memory_order_relaxed)) {
        g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
    void* pointer = std::malloc(size == 0 ? 1 : size);
    if (!pointer) {
        throw std::bad_alloc();
    }
    return pointer;
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace bm = brutal_mlp;

namespace {

struct ModelSpec {
    const char* name;
    std::size_t inputs;
    std::size_t hidden;
    std::size_t hidden_layers;
    std::size_t outputs;
    std::size_t training_samples;
    std::size_t training_batch;
};

constexpr ModelSpec kSmallModel{"small", 8, 16, 1, 4, 128, 16};
constexpr ModelSpec kMediumModel{"medium", 32, 64, 2, 8, 256, 32};
constexpr ModelSpec kLargeModel{"large", 128, 256, 3, 16, 128, 32};

template <typename Function>
std::size_t count_allocations(Function&& function) {
    g_allocation_count.store(0, std::memory_order_relaxed);
    g_count_allocations.store(true, std::memory_order_relaxed);
    function();
    g_count_allocations.store(false, std::memory_order_relaxed);
    return g_allocation_count.load(std::memory_order_relaxed);
}

bm::Matrix make_inputs(std::size_t rows, std::size_t columns) {
    bm::Matrix result(rows, bm::Vector(columns, bm::Scalar{0}));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const double angle = static_cast<double>((row + 1) * (column + 3)) * 0.013;
            result[row][column] = static_cast<bm::Scalar>(std::sin(angle));
        }
    }
    return result;
}

bm::Vector make_flat_inputs(std::size_t rows, std::size_t columns) {
    bm::Vector result(rows * columns, bm::Scalar{0});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const double angle = static_cast<double>((row + 1) * (column + 3)) * 0.013;
            result[row * columns + column] = static_cast<bm::Scalar>(std::sin(angle));
        }
    }
    return result;
}

bm::DenseMatrix make_dense_inputs(std::size_t rows, std::size_t columns) {
    bm::DenseMatrix result(rows, columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            const double angle = static_cast<double>((row + 1) * (column + 3)) * 0.013;
            result(row, column) = static_cast<bm::Scalar>(std::sin(angle));
        }
    }
    return result;
}

bm::Matrix make_regression_targets(const bm::Matrix& inputs, std::size_t outputs) {
    bm::Matrix targets(inputs.size(), bm::Vector(outputs, bm::Scalar{0}));
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        for (std::size_t out = 0; out < outputs; ++out) {
            double value = 0.0;
            for (std::size_t in = 0; in < inputs[row].size(); ++in) {
                const double coefficient = static_cast<double>((in + 3 * out) % 11) - 5.0;
                value += static_cast<double>(inputs[row][in]) * coefficient * 0.003;
            }
            targets[row][out] = static_cast<bm::Scalar>(value);
        }
    }
    return targets;
}

bm::TrainingModel make_training_model(const ModelSpec& spec) {
    auto builder = bm::TrainingModel::builder();
    builder.input_size(spec.inputs)
        .loss(bm::Loss::mean_squared_error)
        .optimizer(bm::OptimizerConfig::adam(bm::Scalar{0.001}))
        .seed(42);

    for (std::size_t layer = 0; layer < spec.hidden_layers; ++layer) {
        builder.add_layer(spec.hidden, bm::Activation::relu);
    }
    builder.add_layer(spec.outputs, bm::Activation::linear);
    return builder.build();
}

bm::TrainingModel make_trained_model(const ModelSpec& spec) {
    auto model = make_training_model(spec);
    auto inputs = make_inputs(spec.training_samples, spec.inputs);
    auto targets = make_regression_targets(inputs, spec.outputs);
    bm::TrainingOptions options;
    options.epochs = 2;
    options.batch_size = spec.training_batch;
    options.shuffle = false;
    options.restore_best_weights = false;
    model.fit(inputs, targets, options);
    return model;
}

bm::CompiledModel make_compiled_model(const ModelSpec& spec) {
    return make_training_model(spec).compile();
}

std::filesystem::path unique_benchmark_path(const char* stem, const char* extension) {
    const auto tick_count = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::size_t counter = g_temp_path_counter.fetch_add(1, std::memory_order_relaxed);
    const std::string filename = std::string(stem) + "_" +
                                 std::to_string(sizeof(bm::Scalar) * 8) + "_" +
                                 std::to_string(tick_count) + "_" +
                                 std::to_string(counter) + extension;
    return std::filesystem::temp_directory_path() / filename;
}

void set_model_counters(benchmark::State& state, const ModelSpec& spec, const bm::InferenceModel& model) {
    state.counters["scalar_bits"] = static_cast<double>(sizeof(bm::Scalar) * 8);
    state.counters["input_size"] = static_cast<double>(spec.inputs);
    state.counters["hidden_size"] = static_cast<double>(spec.hidden);
    state.counters["hidden_layers"] = static_cast<double>(spec.hidden_layers);
    state.counters["output_size"] = static_cast<double>(spec.outputs);
    state.counters["weight_scalars"] = static_cast<double>(model.weight_count());
    state.counters["bias_scalars"] = static_cast<double>(model.bias_count());
    state.counters["scratch_scalars"] = static_cast<double>(model.scratch_size());
}

void set_training_counters(benchmark::State& state, const ModelSpec& spec) {
    state.counters["scalar_bits"] = static_cast<double>(sizeof(bm::Scalar) * 8);
    state.counters["input_size"] = static_cast<double>(spec.inputs);
    state.counters["hidden_size"] = static_cast<double>(spec.hidden);
    state.counters["hidden_layers"] = static_cast<double>(spec.hidden_layers);
    state.counters["output_size"] = static_cast<double>(spec.outputs);
    state.counters["samples"] = static_cast<double>(spec.training_samples);
    state.counters["batch_size"] = static_cast<double>(spec.training_batch);
}

static void BM_SinglePredictUnchecked(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_compiled_model(*spec);
    auto input = make_inputs(1, spec->inputs).front();
    bm::InferenceWorkspace workspace(model);

    const std::size_t allocations = count_allocations([&] {
        model.predict_unchecked_to(input.data(), workspace);
        benchmark::DoNotOptimize(workspace.output_data());
    });

    for (auto _ : state) {
        model.predict_unchecked_to(input.data(), workspace);
        benchmark::DoNotOptimize(workspace.output_data());
    }

    state.SetItemsProcessed(state.iterations());
    set_model_counters(state, *spec, model);
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 0.0;
}

static void BM_SinglePredictAllocating(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_compiled_model(*spec);
    auto input = make_inputs(1, spec->inputs).front();

    const std::size_t allocations = count_allocations([&] {
        auto output = model.predict(input);
        benchmark::DoNotOptimize(output.data());
    });

    for (auto _ : state) {
        auto output = model.predict(input);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(state.iterations());
    set_model_counters(state, *spec, model);
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 1.0;
}

static void BM_BatchPredictUnchecked(benchmark::State& state, const ModelSpec* spec) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto model = make_compiled_model(*spec);
    auto inputs = make_flat_inputs(batch_size, spec->inputs);
    bm::Vector outputs(batch_size * model.output_size(), bm::Scalar{0});
    bm::Vector scratch(model.scratch_size(), bm::Scalar{0});

    const std::size_t allocations = count_allocations([&] {
        model.predict_batch_unchecked_to(inputs.data(),
                                         batch_size,
                                         model.input_size(),
                                         outputs.data(),
                                         model.output_size(),
                                         scratch.data());
        benchmark::DoNotOptimize(outputs.data());
    });

    for (auto _ : state) {
        model.predict_batch_unchecked_to(inputs.data(),
                                         batch_size,
                                         model.input_size(),
                                         outputs.data(),
                                         model.output_size(),
                                         scratch.data());
        benchmark::DoNotOptimize(outputs.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
    set_model_counters(state, *spec, model);
    state.counters["batch_size"] = static_cast<double>(batch_size);
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 0.0;
}

static void BM_BatchPredictSafe(benchmark::State& state, const ModelSpec* spec) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto model = make_compiled_model(*spec);
    auto inputs = make_flat_inputs(batch_size, spec->inputs);
    bm::Vector outputs(batch_size * model.output_size(), bm::Scalar{0});
    bm::Vector scratch(model.scratch_size(), bm::Scalar{0});

    const std::size_t allocations = count_allocations([&] {
        const auto status = model.predict_batch_to(inputs.data(),
                                                   batch_size,
                                                   model.input_size(),
                                                   outputs.data(),
                                                   model.output_size(),
                                                   scratch.data(),
                                                   scratch.size());
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(outputs.data());
    });

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
    set_model_counters(state, *spec, model);
    state.counters["batch_size"] = static_cast<double>(batch_size);
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 0.0;
}

static void BM_BatchPredictAllocatingMatrix(benchmark::State& state, const ModelSpec* spec) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto model = make_compiled_model(*spec);
    auto inputs = make_inputs(batch_size, spec->inputs);

    const std::size_t allocations = count_allocations([&] {
        auto outputs = model.predict_batch(inputs);
        benchmark::DoNotOptimize(outputs.size());
    });

    for (auto _ : state) {
        auto outputs = model.predict_batch(inputs);
        benchmark::DoNotOptimize(outputs.size());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
    set_model_counters(state, *spec, model);
    state.counters["batch_size"] = static_cast<double>(batch_size);
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 1.0;
}

static void BM_BatchPredictAllocatingDense(benchmark::State& state, const ModelSpec* spec) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    auto model = make_compiled_model(*spec);
    auto inputs = make_dense_inputs(batch_size, spec->inputs);

    const std::size_t allocations = count_allocations([&] {
        auto outputs = model.predict_batch(inputs);
        benchmark::DoNotOptimize(outputs.data());
        benchmark::DoNotOptimize(outputs.scalar_count());
    });

    for (auto _ : state) {
        auto outputs = model.predict_batch(inputs);
        benchmark::DoNotOptimize(outputs.data());
        benchmark::DoNotOptimize(outputs.scalar_count());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
    set_model_counters(state, *spec, model);
    state.counters["batch_size"] = static_cast<double>(batch_size);
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 1.0;
}

static void BM_BatchPredictParallel(benchmark::State& state, const ModelSpec* spec) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    const auto thread_count = static_cast<std::size_t>(state.range(1));
    auto model = make_compiled_model(*spec);
    auto inputs = make_flat_inputs(batch_size, spec->inputs);
    bm::Vector outputs(batch_size * model.output_size(), bm::Scalar{0});

    bm::ParallelOptions parallel;
    parallel.execution = bm::ParallelExecution::worker_threads;
    parallel.thread_count = thread_count;
    parallel.minimum_parallel_samples = 1;

    bm::Vector scratch(model.batch_scratch_size(batch_size, parallel), bm::Scalar{0});
    const std::size_t allocations = count_allocations([&] {
        const auto status = model.predict_batch_to(inputs.data(),
                                                   batch_size,
                                                   model.input_size(),
                                                   outputs.data(),
                                                   model.output_size(),
                                                   scratch.data(),
                                                   scratch.size(),
                                                   parallel);
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(outputs.data());
    });

    for (auto _ : state) {
        const auto status = model.predict_batch_to(inputs.data(),
                                                   batch_size,
                                                   model.input_size(),
                                                   outputs.data(),
                                                   model.output_size(),
                                                   scratch.data(),
                                                   scratch.size(),
                                                   parallel);
        benchmark::DoNotOptimize(status);
        benchmark::DoNotOptimize(outputs.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
    set_model_counters(state, *spec, model);
    state.counters["batch_size"] = static_cast<double>(batch_size);
    state.counters["thread_count"] = static_cast<double>(thread_count);
    state.counters["parallel_scratch_scalars"] = static_cast<double>(scratch.size());
    state.counters["allocations_per_call"] = static_cast<double>(allocations);
    state.counters["allocating_path"] = 1.0;
}

static void BM_TrainingEpoch(benchmark::State& state, const ModelSpec* spec) {
    const auto inputs = make_inputs(spec->training_samples, spec->inputs);
    const auto targets = make_regression_targets(inputs, spec->outputs);

    for (auto _ : state) {
        state.PauseTiming();
        auto model = make_training_model(*spec);
        bm::TrainingOptions options;
        options.epochs = 1;
        options.batch_size = spec->training_batch;
        options.shuffle = false;
        options.restore_best_weights = false;
        state.ResumeTiming();

        auto history = model.fit(inputs, targets, options);
        benchmark::DoNotOptimize(history.training_loss.data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(spec->training_samples));
    set_training_counters(state, *spec);
}

static void BM_SaveInferenceText(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_compiled_model(*spec);
    const auto path = unique_benchmark_path("brutal_mlp_inference_save", ".model");

    for (auto _ : state) {
        model.save(path);
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, model);
    state.counters["format_binary"] = 0.0;
}

static void BM_SaveInferenceBinary(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_compiled_model(*spec);
    const auto path = unique_benchmark_path("brutal_mlp_inference_save", ".bmlp");

    for (auto _ : state) {
        model.save_binary(path);
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, model);
    state.counters["format_binary"] = 1.0;
}

static void BM_LoadInferenceText(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_compiled_model(*spec);
    const auto path = unique_benchmark_path("brutal_mlp_inference_load", ".model");
    model.save(path);

    for (auto _ : state) {
        auto loaded = bm::CompiledModel::load(path);
        benchmark::DoNotOptimize(loaded.weight_count());
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, model);
    state.counters["format_binary"] = 0.0;
}

static void BM_LoadInferenceBinary(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_compiled_model(*spec);
    const auto path = unique_benchmark_path("brutal_mlp_inference_load", ".bmlp");
    model.save_binary(path);

    for (auto _ : state) {
        auto loaded = bm::CompiledModel::load_binary(path);
        benchmark::DoNotOptimize(loaded.weight_count());
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, model);
    state.counters["format_binary"] = 1.0;
}

static void BM_SaveTrainingText(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_trained_model(*spec);
    auto inference = model.to_inference_model();
    const auto path = unique_benchmark_path("brutal_mlp_training_save", ".model");

    for (auto _ : state) {
        model.save(path);
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, inference);
    state.counters["format_binary"] = 0.0;
}

static void BM_SaveTrainingBinary(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_trained_model(*spec);
    auto inference = model.to_inference_model();
    const auto path = unique_benchmark_path("brutal_mlp_training_save", ".bmlp");

    for (auto _ : state) {
        model.save_binary(path);
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, inference);
    state.counters["format_binary"] = 1.0;
}

static void BM_LoadTrainingText(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_trained_model(*spec);
    auto inference = model.to_inference_model();
    const auto path = unique_benchmark_path("brutal_mlp_training_load", ".model");
    model.save(path);

    for (auto _ : state) {
        auto loaded = bm::TrainingModel::load(path);
        benchmark::DoNotOptimize(loaded.completed_epochs());
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, inference);
    state.counters["format_binary"] = 0.0;
}

static void BM_LoadTrainingBinary(benchmark::State& state, const ModelSpec* spec) {
    auto model = make_trained_model(*spec);
    auto inference = model.to_inference_model();
    const auto path = unique_benchmark_path("brutal_mlp_training_load", ".bmlp");
    model.save_binary(path);

    for (auto _ : state) {
        auto loaded = bm::TrainingModel::load_binary(path);
        benchmark::DoNotOptimize(loaded.completed_epochs());
    }

    std::filesystem::remove(path);
    set_model_counters(state, *spec, inference);
    state.counters["format_binary"] = 1.0;
}

#define REGISTER_MODEL_BENCHMARKS(label, spec_pointer)                                                      \
    BENCHMARK_CAPTURE(BM_SinglePredictUnchecked, label, spec_pointer);                                      \
    BENCHMARK_CAPTURE(BM_SinglePredictAllocating, label, spec_pointer);                                     \
    BENCHMARK_CAPTURE(BM_BatchPredictUnchecked, label, spec_pointer)->Arg(1)->Arg(32)->Arg(256);            \
    BENCHMARK_CAPTURE(BM_BatchPredictSafe, label, spec_pointer)->Arg(1)->Arg(32)->Arg(256);                 \
    BENCHMARK_CAPTURE(BM_BatchPredictAllocatingMatrix, label, spec_pointer)->Arg(1)->Arg(32)->Arg(256);     \
    BENCHMARK_CAPTURE(BM_BatchPredictAllocatingDense, label, spec_pointer)->Arg(1)->Arg(32)->Arg(256);      \
    BENCHMARK_CAPTURE(BM_BatchPredictParallel, label, spec_pointer)                                        \
        ->Args({256, 2})                                                                                   \
        ->Args({256, 4})                                                                                   \
        ->UseRealTime();                                                                                   \
    BENCHMARK_CAPTURE(BM_TrainingEpoch, label, spec_pointer);                                               \
    BENCHMARK_CAPTURE(BM_SaveInferenceText, label, spec_pointer)->UseRealTime();                            \
    BENCHMARK_CAPTURE(BM_SaveInferenceBinary, label, spec_pointer)->UseRealTime();                          \
    BENCHMARK_CAPTURE(BM_LoadInferenceText, label, spec_pointer)->UseRealTime();                            \
    BENCHMARK_CAPTURE(BM_LoadInferenceBinary, label, spec_pointer)->UseRealTime();                          \
    BENCHMARK_CAPTURE(BM_SaveTrainingText, label, spec_pointer)->UseRealTime();                             \
    BENCHMARK_CAPTURE(BM_SaveTrainingBinary, label, spec_pointer)->UseRealTime();                           \
    BENCHMARK_CAPTURE(BM_LoadTrainingText, label, spec_pointer)->UseRealTime();                             \
    BENCHMARK_CAPTURE(BM_LoadTrainingBinary, label, spec_pointer)->UseRealTime()

REGISTER_MODEL_BENCHMARKS(small, &kSmallModel);
REGISTER_MODEL_BENCHMARKS(medium, &kMediumModel);
REGISTER_MODEL_BENCHMARKS(large, &kLargeModel);

} // namespace

BENCHMARK_MAIN();

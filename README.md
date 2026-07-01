# brutal_mlp

Small C++17 multilayer perceptron library with a stable CMake integration surface, a flexible training API, and a frozen allocation-free inference hot path.

The runtime library only uses the C++ standard library. GoogleTest and Google Benchmark are fetched only for `BRUTAL_MLP_BUILD_TESTS` and `BRUTAL_MLP_BUILD_BENCHMARKS`.

`brutal_mlp::Scalar` is `float` by default. Use double precision only for debug or numerical investigation builds:

```bash
cmake -S . -B build-double -DBRUTAL_MLP_USE_DOUBLE=ON
```

When using the CMake target `brutal_mlp::brutal_mlp`, the precision macro is propagated to consumers. Do not mix objects built with different `BRUTAL_MLP_USE_DOUBLE` settings.

## API split

- `brutal_mlp::TrainingModel`: owns training state, gradients, optimizer moments, validation, history, serialization, and debug-friendly APIs.
- `brutal_mlp::MutableModel`: alias for `TrainingModel` when you want to name the flexible build/train/modify side explicitly.
- `brutal_mlp::CompiledModel`: frozen model for production inference. `TrainingModel::compile()` precomputes layer offsets, compacts weights and biases into contiguous buffers, fixes scratch sizing, drops optimizer/gradient state, and exposes safe `predict_to(...) noexcept` APIs plus unchecked fast-path APIs so callers provide all runtime buffers.
- `brutal_mlp::InferenceModel`: compatibility name for the frozen runtime API. New code should prefer `CompiledModel`.
- `brutal_mlp::InferenceWorkspace`: reusable output + scratch storage for single-sample inference. Allocate one per thread, worker, renderer tile, or calling context, then reuse it.
- `brutal_mlp::ParallelOptions`: explicit CPU parallelism policy for coarse batch inference and future training phases. The default is serial execution.
- `brutal_mlp::Model`: legacy alias for `TrainingModel`.

## Normalization

Input and output normalization are part of the model, not an external convention. The same normalization spec is saved with training and inference files, and `TrainingModel::compile()` carries it into the frozen runtime model. `TrainingModel::to_inference_model()` remains available for existing integrations.

Supported per-feature transforms:

- none
- mean / standard deviation
- min / max into a configurable normalized range
- optional clamp in normalized space
- output denormalization after network inference

```cpp
auto normalization = brutal_mlp::NormalizationSpec::standard_score(
    {0.5f, 2.0f},   // input means
    {0.25f, 4.0f},  // input standard deviations
    {10.0f},        // output means
    {5.0f}          // output standard deviations
);

auto training = brutal_mlp::TrainingModel::builder()
    .input_size(2)
    .add_layer(16, brutal_mlp::Activation::relu)
    .add_layer(1, brutal_mlp::Activation::linear)
    .loss(brutal_mlp::Loss::mean_squared_error)
    .normalization(normalization)
    .build();
```

Output normalization is intended for regression losses and is rejected for classification losses.

## Losses

Built-in training losses:

- `mean_squared_error`
- `mean_absolute_error`
- `huber`
- `relative_mean_squared_error`
- `log_cosh`
- `weighted_mean_squared_error`
- `binary_cross_entropy`
- `categorical_cross_entropy`

Simple losses remain source-compatible:

```cpp
.loss(brutal_mlp::Loss::mean_squared_error)
```

Configured losses use `LossConfig`:

```cpp
auto training = brutal_mlp::TrainingModel::builder()
    .input_size(8)
    .add_layer(32, brutal_mlp::Activation::relu)
    .add_layer(3, brutal_mlp::Activation::linear)
    .loss(brutal_mlp::LossConfig::weighted_mean_squared_error({1.0f, 0.25f, 4.0f}))
    .build();
```

Useful factory examples:

```cpp
brutal_mlp::LossConfig::mean_absolute_error();
brutal_mlp::LossConfig::huber(1.0f);
brutal_mlp::LossConfig::relative_mean_squared_error(1e-4f);
brutal_mlp::LossConfig::log_cosh();
brutal_mlp::LossConfig::weighted_mean_squared_error({1.0f, 2.0f});
```

`relative_mean_squared_error` divides each squared error by `max(abs(target), epsilon)^2`. `weighted_mean_squared_error` requires one non-negative weight per output and divides the weighted squared error by the sum of weights.

Custom losses are supported on the training side with raw function pointers and an optional context pointer:

```cpp
brutal_mlp::Scalar value(const brutal_mlp::Scalar* prediction,
                         const brutal_mlp::Scalar* target,
                         std::size_t size,
                         void* context);

void gradient(const brutal_mlp::Scalar* prediction,
              const brutal_mlp::Scalar* target,
              std::size_t size,
              brutal_mlp::Scalar* output_gradient,
              void* context);

.loss(brutal_mlp::LossConfig::custom(value, gradient, context))
```

Custom loss callbacks are not serialized. Convert trained weights to `InferenceModel` for production, or use a built-in loss when the training model itself must be saved.

## Initialization

The builder supports explicit weight initialization and constant bias initialization:

```cpp
auto model = brutal_mlp::TrainingModel::builder()
    .input_size(16)
    .add_layer(64, brutal_mlp::Activation::relu)
    .add_layer(1, brutal_mlp::Activation::linear)
    .initialization(brutal_mlp::InitializationConfig::he_normal(0.01f))
    .seed(1234)
    .build();
```

Available weight strategies are `automatic`, `he_normal`, `he_uniform`, `xavier_normal`, `xavier_uniform`, `lecun_normal`, and `lecun_uniform`. The default `automatic` mode keeps ReLU layers on He uniform and other layers on Xavier uniform. The builder seed fully controls the initialized weights.

## Regularization

Regularization is configured on the training side. Inference models never carry dropout, gradients, optimizer state, or training-only buffers.

```cpp
auto optimizer = brutal_mlp::OptimizerConfig::adamw(1e-3f, 1e-2f);
optimizer.l1 = 1e-6f;
optimizer.l2 = 1e-5f;
optimizer.max_norm = 3.0f;

auto training = brutal_mlp::TrainingModel::builder()
    .input_size(16)
    .add_layer(64, brutal_mlp::Activation::relu, 0.10f) // hidden-layer dropout
    .add_layer(1, brutal_mlp::Activation::linear)
    .optimizer(optimizer)
    .seed(1234)
    .build();

brutal_mlp::TrainingOptions options;
options.gradient_noise_stddev = 1e-5f;
```

Supported options are L1, coupled L2, AdamW decoupled weight decay, hidden-layer inverted dropout, optional gradient noise, and max-norm constraints on each neuron's incoming weight vector. Output-layer dropout is rejected.

## Gradient Clipping

Clipping is configured on `OptimizerConfig` and reported per epoch:

```cpp
auto optimizer = brutal_mlp::OptimizerConfig::adamw(1e-3f, 1e-2f);
optimizer.gradient_clip_norm = 5.0f;       // global gradient norm
optimizer.layer_gradient_clip_norm = 2.0f; // per-layer gradient norm
optimizer.gradient_clip_value = 0.25f;     // final per-value clamp

auto history = training.fit(dataset, options);

auto global_rate = history.epochs.back().clipping.global_clip_rate;
auto layer_rate = history.epochs.back().clipping.layer_clip_rate;
auto value_rate = history.epochs.back().clipping.value_clip_rate;
```

`gradient_norm` remains the pre-clipping global norm. The clipping diagnostics include batch, layer, and gradient-value counts, clip counts, per-epoch rates, and the smallest norm clip scale applied during the epoch.

## Metrics

Losses are used for optimization. Metrics are evaluated separately on predictions and targets.

Built-in metrics:

- `mean_squared_error`
- `mean_absolute_error`
- `root_mean_squared_error`
- `r2_score`
- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `confusion_matrix`

Regression metrics are the default:

```cpp
auto report = training.evaluate_metrics(inputs, targets);

auto mse = report.metric(brutal_mlp::Metric::mean_squared_error);
auto rmse = report.metric(brutal_mlp::Metric::root_mean_squared_error);
auto r2 = report.metric(brutal_mlp::Metric::r2_score);
```

Classification metrics use a binary threshold for one-output models and argmax for multi-output models. Precision, recall, and F1 support binary, macro, and micro averaging:

```cpp
auto options = brutal_mlp::EvaluationOptions::binary_classification(0.5f);
auto report = training.evaluate_metrics(inputs, targets, options);

auto accuracy = report.metric(brutal_mlp::Metric::accuracy);
auto f1 = report.metric(brutal_mlp::Metric::f1_score);
const auto& confusion = report.confusion_matrix.counts; // rows = target class, columns = predicted class
```

Custom metrics receive the full prediction and target matrices:

```cpp
brutal_mlp::Scalar max_error(const brutal_mlp::Matrix& predictions,
                             const brutal_mlp::Matrix& targets,
                             void* context);

brutal_mlp::EvaluationOptions options;
options.add_custom_metric("max_error", max_error);

auto report = training.evaluate_metrics(inputs, targets, options);
auto max_error_value = report.custom_metric("max_error");
```

`evaluate_predictions(predictions, targets, options)` is also available when predictions are produced outside a model.

## Training Diagnostics

`fit(...)` returns a `TrainingHistory` with legacy loss vectors and structured diagnostics:

```cpp
auto history = training.fit(inputs, targets, options);

for (const auto& epoch : history.epochs) {
    auto loss = epoch.training_loss;
    auto validation = epoch.has_validation_loss ? epoch.validation_loss : 0.0f;
    auto gradient_norm = epoch.gradient_norm;
    auto global_clip_rate = epoch.clipping.global_clip_rate;
    auto weight_min = epoch.weights.minimum;
    auto weight_max = epoch.weights.maximum;
    auto seconds = epoch.epoch_seconds;
}

if (history.stop_reason != brutal_mlp::TrainingStopReason::completed_epochs) {
    auto reason = brutal_mlp::to_string(history.stop_reason);
}
```

Early stopping can monitor training, validation, or test loss. The default `automatic` monitor uses validation loss when a validation split exists, otherwise training loss.

```cpp
brutal_mlp::TrainingOptions options;
options.validation_split = 0.15f;
options.early_stopping_monitor = brutal_mlp::TrainingMonitor::validation_loss;
options.early_stopping_mode = brutal_mlp::TrainingMonitorMode::minimize;
options.early_stopping_patience = 10;
options.min_delta = 1e-4f;
options.early_stopping_cooldown = 2;
options.restore_best_weights = true;
options.best_checkpoint_path = "best.bmlp";
```

Learning rate schedules are configured through `TrainingOptions` and are reported in each epoch diagnostic:

```cpp
brutal_mlp::TrainingOptions options;
options.learning_rate_schedule =
    brutal_mlp::LearningRateScheduleConfig::cosine_annealing(200, 1e-5f);
options.learning_rate_schedule.warmup_epochs = 10;
options.learning_rate_schedule.warmup_start_learning_rate = 1e-6f;
```

Available schedules are constant, step decay, exponential decay, cosine annealing, and reduce-on-plateau. Warmup can prefix any schedule:

```cpp
auto schedule = brutal_mlp::LearningRateScheduleConfig::reduce_on_plateau(5, 0.5f);
schedule.reduce_on_plateau_monitor = brutal_mlp::TrainingMonitor::validation_loss;
schedule.reduce_on_plateau_min_delta = 1e-4f;
schedule.reduce_on_plateau_cooldown = 2;
schedule.minimum_learning_rate = 1e-6f;

options.learning_rate_schedule = schedule;
```

Each epoch records training loss, optional validation/test loss, the monitored metric, learning rate, gradient norm, clipping diagnostics, weight min/max/mean, non-finite parameter counts, elapsed time, cooldown/stale counters, and whether it produced the best checkpoint. Training stops with a structured reason for early stopping, non-finite loss, non-finite gradients, or non-finite weights.

## Datasets

`fit(inputs, targets)` remains available for small in-memory datasets. For larger or generated data, use the dataset pipeline.

Random-access datasets support deterministic train/validation/test splits and shuffled mini-batches without copying the full dataset:

```cpp
brutal_mlp::MatrixDataset dataset(inputs, targets);

brutal_mlp::TrainingOptions options;
options.epochs = 500;
options.batch_size = 64;
options.validation_split = 0.15f;
options.test_split = 0.15f;
options.shuffle = true;
options.seed = 42;

auto history = training.fit(dataset, options);
```

Split utilities are exposed directly:

```cpp
brutal_mlp::DatasetSplitOptions split_options;
split_options.validation_split = 0.2f;
split_options.test_split = 0.1f;
split_options.shuffle = true;
split_options.seed = 123;

auto split = brutal_mlp::make_dataset_split(dataset.sample_count(), split_options);
brutal_mlp::DatasetView validation(dataset, split.validation_indices);
```

Mini-batches can be generated independently from training:

```cpp
brutal_mlp::BatchGenerator batches(dataset, 128, true, 123);
brutal_mlp::MiniBatch batch;

while (batches.next(batch)) {
    // Row-major contiguous buffers:
    // batch.inputs.size() == batch.size() * batch.input_size
    // batch.targets.size() == batch.size() * batch.output_size
    const brutal_mlp::Scalar* first_input = batch.input_data(0);
    const brutal_mlp::Scalar* first_target = batch.target_data(0);
}
```

`MiniBatch::input(i)` and `MiniBatch::target(i)` are convenience copies. Hot paths should use `input_data(i)` and `target_data(i)`.

Generated datasets fill caller-owned sample buffers:

```cpp
void sample(std::size_t index,
            brutal_mlp::Scalar* input,
            std::size_t input_size,
            brutal_mlp::Scalar* target,
            std::size_t target_size,
            void* context);

brutal_mlp::GeneratedDataset generated(sample_count, input_size, output_size, sample, context);
training.fit(generated, options);
```

Streaming datasets are resettable streams. They keep only one sample plus an optional bounded shuffle buffer in memory:

```cpp
options.streaming_shuffle_buffer_size = 4096;

brutal_mlp::CsvStreamingOptions csv;
csv.has_header = true;

brutal_mlp::CsvStreamingDataset streamed("train.csv", input_size, output_size, csv);
training.fit(streamed, options);
```

`FunctionStreamingDataset` is available for procedural or application-owned streams. Its callback returns `false` when the stream is exhausted.

## Serialization

Text serialization remains available through `save(...)` and `load(...)` for debugging and review. Production artifacts should use the binary format:

```cpp
brutal_mlp::BinaryMetadata metadata;
metadata.description = "shipping renderer model";
metadata.entries.push_back({"dataset", "lighting-v4"});

training.save_binary("lighting-training.bmlp", metadata);
auto training_info = brutal_mlp::inspect_binary_model("lighting-training.bmlp");
auto restored_training = brutal_mlp::TrainingModel::load_binary("lighting-training.bmlp");

auto compiled = training.compile();
compiled.save_binary("lighting-compiled.bmlp", metadata);
auto restored_compiled = brutal_mlp::CompiledModel::load_binary("lighting-compiled.bmlp");
```

Binary files include a magic number, format version, scalar type, model kind, architecture, activations, dimensions, weights, biases, normalization, metadata, checksum, seed, and training configuration when available. Loading rejects unsupported versions, scalar mismatches, truncated files, and checksum failures.

Training checkpoints extend the binary format with optimizer state, completed epoch, checkpoint metric, and full training options:

```cpp
brutal_mlp::TrainingOptions options;
options.epochs = 1000;
options.batch_size = 64;
options.best_checkpoint_path = "best.bmlp";
options.latest_checkpoint_path = "latest.bmlp";

auto history = training.fit(dataset, options);

brutal_mlp::TrainingCheckpointInfo info;
auto resumed = brutal_mlp::TrainingModel::load_checkpoint("latest.bmlp", &info);
auto resumed_history = resumed.fit(dataset, info.training_options);

auto best = brutal_mlp::TrainingModel::load_checkpoint("best.bmlp");
```

`latest_checkpoint_path` is written after each finite epoch. `best_checkpoint_path` is updated only when the monitored metric improves. A loaded checkpoint resumes from `completed_epochs` on the next `fit(...)` call when the saved options are reused.

## Integration

```cmake
include(FetchContent)

FetchContent_Declare(
    brutal_mlp
    GIT_REPOSITORY https://github.com/TheoBaudoinLighting/BrutalMLP.git
    GIT_TAG main
)
FetchContent_MakeAvailable(brutal_mlp)

target_link_libraries(your_target PRIVATE brutal_mlp::brutal_mlp)
```

Installed package usage:

```cmake
find_package(brutal_mlp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE brutal_mlp::brutal_mlp)
```

## Example

```cpp
#include "brutal_mlp/brutal_mlp.hpp"

auto optimizer = brutal_mlp::OptimizerConfig::adam(0.01);

auto training = brutal_mlp::TrainingModel::builder()
    .input_size(2)
    .add_layer(16, brutal_mlp::Activation::relu)
    .add_layer(1, brutal_mlp::Activation::sigmoid)
    .loss(brutal_mlp::Loss::binary_cross_entropy)
    .optimizer(optimizer)
    .seed(42)
    .build();

brutal_mlp::Matrix x{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
brutal_mlp::Matrix y{{0.0}, {1.0}, {1.0}, {0.0}};

brutal_mlp::TrainingOptions options;
options.epochs = 1000;
options.batch_size = 4;

training.fit(x, y, options);

auto compiled = training.compile();

brutal_mlp::Vector input{1.0, 0.0};
brutal_mlp::InferenceWorkspace workspace(compiled);

auto status = compiled.predict_to(input.data(), input.size(), workspace);

if (status != brutal_mlp::InferenceStatus::ok) {
    // Handle invalid buffers or shape mismatch outside the hot path.
}
```

The safe inference API validates pointers, sizes, strides, and scratch capacity, then returns an `InferenceStatus`. The fast API assumes every contract is already satisfied and performs no validation:

```cpp
// Contract: input/output/scratch point to valid buffers sized for this model.
compiled.predict_unchecked_to(input.data(), workspace);
```

`CompiledModel` is immutable after construction or loading. It can be shared across threads for concurrent `const` inference calls as long as every thread uses its own output buffer and scratch storage. `InferenceWorkspace` is mutable caller-owned storage; allocate one per thread, worker, tile, or job, or use application-managed `thread_local` storage initialized before the tight loop. Do not share one workspace between simultaneous predictions.

For flat batch inference, use caller-owned contiguous memory:

```cpp
std::vector<brutal_mlp::Scalar> inputs(sample_count * compiled.input_size());
std::vector<brutal_mlp::Scalar> outputs(sample_count * compiled.output_size());
std::vector<brutal_mlp::Scalar> scratch(compiled.scratch_size());

auto status = compiled.predict_batch_to(inputs.data(),
                                        sample_count,
                                        compiled.input_size(),
                                        outputs.data(),
                                        compiled.output_size(),
                                        scratch.data(),
                                        scratch.size());

compiled.predict_batch_unchecked_to(inputs.data(),
                                     sample_count,
                                     compiled.input_size(),
                                     outputs.data(),
                                     compiled.output_size(),
                                     scratch.data());
```

Batch parallelism is opt-in and uses one scratch slice per worker:

```cpp
brutal_mlp::ParallelOptions parallel;
parallel.execution = brutal_mlp::ParallelExecution::worker_threads;
parallel.thread_count = 0;                 // 0 = hardware_concurrency()
parallel.minimum_parallel_samples = 64;    // stay serial for tiny batches

std::vector<brutal_mlp::Scalar> parallel_scratch(
    compiled.batch_scratch_size(sample_count, parallel));

auto parallel_status = compiled.predict_batch_to(inputs.data(),
                                                 sample_count,
                                                 compiled.input_size(),
                                                 outputs.data(),
                                                 compiled.output_size(),
                                                 parallel_scratch.data(),
                                                 parallel_scratch.size(),
                                                 parallel);
```

`CompiledModel::predict_to(...)`, the serial `predict_batch_to(...)`, `predict_unchecked_to(...)`, and `predict_batch_unchecked_to(...)` do not allocate, do not throw, and do not mutate the model. The parallel batch overload is also `noexcept` and caller-scratch-based, but it creates worker threads internally; for renderer hot loops, prefer one worker-owned scratch/workspace per application thread and call the serial fast path inside that worker. The convenience methods `predict(...)` and `predict_batch(...)` return `Vector`/`Matrix` and are intentionally outside the hot path.

`TrainingOptions::parallelism` stores the same policy and is written into binary models and checkpoints. The current optimizer update path remains serial and deterministic; the policy is part of the training config so CPU-parallel batch loading, normalization, or gradient accumulation can be enabled without changing the public configuration surface.

## Build

Prerequisites:

- CMake 3.20+
- A C++17 compiler
- Git, because tests and benchmarks use CMake `FetchContent`

Portable multi-config build, recommended for Visual Studio and Xcode:

```bash
cmake -S . -B build -DBRUTAL_MLP_BUILD_TESTS=ON -DBRUTAL_MLP_BUILD_BENCHMARKS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

Single-config build, recommended for Ninja and Makefiles:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBRUTAL_MLP_BUILD_TESTS=ON -DBRUTAL_MLP_BUILD_BENCHMARKS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Benchmarks are exposed by the `brutal_mlp_benchmarks` executable when `BRUTAL_MLP_BUILD_BENCHMARKS=ON`. They cover small, medium, and large models across single prediction, flat batch prediction, matrix convenience prediction, worker-thread batch prediction, one training epoch, and text/binary save/load. Useful filters:

```bash
# List every benchmark case.
build/benchmarks/Release/brutal_mlp_benchmarks --benchmark_list_tests

# Compare no-allocation and allocating inference paths.
build/benchmarks/Release/brutal_mlp_benchmarks --benchmark_filter="SinglePredict|BatchPredict.*(Unchecked|AllocatingMatrix)"

# Measure training epoch and serialization costs.
build/benchmarks/Release/brutal_mlp_benchmarks --benchmark_filter="TrainingEpoch|Save|Load"
```

Each benchmark reports `scalar_bits`, model dimensions, weight counts, and, for inference paths, `allocations_per_call` plus `allocating_path`. To compare float and double, build two benchmark trees and run the same filter:

```bash
cmake -S . -B build-bench-float -DCMAKE_BUILD_TYPE=Release -DBRUTAL_MLP_BUILD_TESTS=OFF -DBRUTAL_MLP_BUILD_BENCHMARKS=ON
cmake --build build-bench-float --parallel

cmake -S . -B build-bench-double -DCMAKE_BUILD_TYPE=Release -DBRUTAL_MLP_USE_DOUBLE=ON -DBRUTAL_MLP_BUILD_TESTS=OFF -DBRUTAL_MLP_BUILD_BENCHMARKS=ON
cmake --build build-bench-double --parallel

build-bench-float/benchmarks/brutal_mlp_benchmarks --benchmark_filter="SinglePredict|BatchPredictUnchecked|TrainingEpoch"
build-bench-double/benchmarks/brutal_mlp_benchmarks --benchmark_filter="SinglePredict|BatchPredictUnchecked|TrainingEpoch"
```

With multi-config generators such as Visual Studio or Xcode, use the same commands with `cmake --build ... --config Release`; the executable is under `benchmarks/Release/`.

When consumed from another CMake project through `FetchContent`, tests and examples default to `OFF`.

Install locally:

```bash
cmake --install build --config Release --prefix install
```

Then consume the installed package:

```cmake
find_package(brutal_mlp CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE brutal_mlp::brutal_mlp)
```

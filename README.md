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
- `brutal_mlp::InferenceModel`: frozen model for production inference. It stores weights and biases contiguously, has no optimizer or gradient state, and exposes `predict_to(...) noexcept` and `predict_batch_to(...) noexcept` so callers provide all runtime buffers.
- `brutal_mlp::InferenceWorkspace`: reusable output + scratch storage for single-sample inference. Allocate one per thread, worker, renderer tile, or calling context, then reuse it.
- `brutal_mlp::Model`: legacy alias for `TrainingModel`.

## Normalization

Input and output normalization are part of the model, not an external convention. The same normalization spec is saved with training and inference files, and `TrainingModel::to_inference_model()` carries it into the frozen runtime model.

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

auto inference = training.to_inference_model();

brutal_mlp::Vector input{1.0, 0.0};
brutal_mlp::InferenceWorkspace workspace(inference);

auto status = inference.predict_to(input.data(), input.size(), workspace);

if (status != brutal_mlp::InferenceStatus::ok) {
    // Handle invalid buffers or shape mismatch outside the hot path.
}
```

For flat batch inference, use caller-owned contiguous memory:

```cpp
std::vector<brutal_mlp::Scalar> inputs(sample_count * inference.input_size());
std::vector<brutal_mlp::Scalar> outputs(sample_count * inference.output_size());
std::vector<brutal_mlp::Scalar> scratch(inference.scratch_size());

auto status = inference.predict_batch_to(inputs.data(),
                                         sample_count,
                                         inference.input_size(),
                                         outputs.data(),
                                         inference.output_size(),
                                         scratch.data(),
                                         scratch.size());
```

`InferenceModel::predict_to(...)` and `InferenceModel::predict_batch_to(...)` do not allocate, do not throw, and do not mutate the model. The convenience methods `predict(...)` and `predict_batch(...)` return `Vector`/`Matrix` and are intentionally outside the hot path.

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

Benchmarks are exposed by the `brutal_mlp_benchmarks` executable when `BRUTAL_MLP_BUILD_BENCHMARKS=ON`.

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

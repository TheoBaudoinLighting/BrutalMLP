# brutal_mlp

Small C++17 multilayer perceptron library with a stable CMake integration surface, a flexible training API, and a frozen allocation-free inference hot path.

The runtime library only uses the C++ standard library. GoogleTest and Google Benchmark are fetched only for `BRUTAL_MLP_BUILD_TESTS` and `BRUTAL_MLP_BUILD_BENCHMARKS`.

## API split

- `brutal_mlp::TrainingModel`: owns training state, gradients, optimizer moments, validation, history, serialization, and debug-friendly APIs.
- `brutal_mlp::InferenceModel`: frozen model for production inference. It stores weights and biases contiguously, has no optimizer or gradient state, and exposes `predict_to(...) noexcept` and `predict_batch_to(...) noexcept` so callers provide all runtime buffers.
- `brutal_mlp::InferenceWorkspace`: reusable output + scratch storage for single-sample inference. Allocate one per thread, worker, renderer tile, or calling context, then reuse it.
- `brutal_mlp::Model`: legacy alias for `TrainingModel`.

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

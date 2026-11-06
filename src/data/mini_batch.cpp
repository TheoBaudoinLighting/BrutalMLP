// MiniBatch storage and BatchGenerator.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"
#include "core/random_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace brutal_mlp {

using namespace detail;

void MiniBatch::resize(std::size_t samples, std::size_t input_width, std::size_t output_width) {
    require(input_width > 0 || samples == 0, "mini-batch input width must be positive");
    require(output_width > 0 || samples == 0, "mini-batch output width must be positive");
    sample_count = samples;
    input_size = input_width;
    output_size = output_width;
    inputs.resize(sample_count * input_size);
    targets.resize(sample_count * output_size);
}

std::size_t MiniBatch::size() const noexcept {
    return sample_count;
}

bool MiniBatch::empty() const noexcept {
    return sample_count == 0;
}

void MiniBatch::clear() noexcept {
    sample_count = 0;
    input_size = 0;
    output_size = 0;
    inputs.clear();
    targets.clear();
}

Scalar* MiniBatch::input_data(std::size_t sample) {
    require(sample < sample_count, "mini-batch input sample index is out of range");
    return inputs.data() + sample * input_size;
}

const Scalar* MiniBatch::input_data(std::size_t sample) const {
    require(sample < sample_count, "mini-batch input sample index is out of range");
    return inputs.data() + sample * input_size;
}

Scalar* MiniBatch::target_data(std::size_t sample) {
    require(sample < sample_count, "mini-batch target sample index is out of range");
    return targets.data() + sample * output_size;
}

const Scalar* MiniBatch::target_data(std::size_t sample) const {
    require(sample < sample_count, "mini-batch target sample index is out of range");
    return targets.data() + sample * output_size;
}

Vector MiniBatch::input(std::size_t sample) const {
    const Scalar* row = input_data(sample);
    return Vector(row, row + input_size);
}

Vector MiniBatch::target(std::size_t sample) const {
    const Scalar* row = target_data(sample);
    return Vector(row, row + output_size);
}

struct BatchGenerator::Impl {
    const Dataset* dataset{nullptr};
    std::size_t batch_size{0};
    bool shuffle{false};
    std::uint64_t seed{0};
    std::size_t cursor{0};
    std::vector<std::size_t> indices;
};

BatchGenerator::BatchGenerator(const Dataset& dataset,
                               std::size_t batch_size,
                               bool shuffle,
                               std::uint64_t seed)
    : impl_(std::make_unique<Impl>()) {
    require(batch_size > 0, "batch_size must be positive");
    require(dataset.sample_count() > 0, "dataset must not be empty");
    impl_->dataset = &dataset;
    impl_->batch_size = batch_size;
    impl_->shuffle = shuffle;
    impl_->seed = seed;
    reset();
}

BatchGenerator::BatchGenerator(const BatchGenerator& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

BatchGenerator& BatchGenerator::operator=(const BatchGenerator& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

BatchGenerator::BatchGenerator(BatchGenerator&& other) noexcept = default;

BatchGenerator& BatchGenerator::operator=(BatchGenerator&& other) noexcept = default;

BatchGenerator::~BatchGenerator() = default;

void BatchGenerator::reset(std::uint64_t epoch) {
    require(impl_ && impl_->dataset, "batch generator is not initialized");
    impl_->indices = make_indices(impl_->dataset->sample_count());
    impl_->cursor = 0;
    if (impl_->shuffle) {
        std::mt19937_64 rng(non_zero_seed(impl_->seed, 5489u) + epoch);
        std::shuffle(impl_->indices.begin(), impl_->indices.end(), rng);
    }
}

bool BatchGenerator::next(MiniBatch& batch) {
    require(impl_ && impl_->dataset, "batch generator is not initialized");
    batch.clear();
    if (impl_->cursor >= impl_->indices.size()) {
        return false;
    }

    const std::size_t count = std::min(impl_->batch_size, impl_->indices.size() - impl_->cursor);
    batch.resize(count, impl_->dataset->input_size(), impl_->dataset->output_size());
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t row = impl_->indices[impl_->cursor + i];
        impl_->dataset->sample(row,
                               batch.input_data(i),
                               batch.input_size,
                               batch.target_data(i),
                               batch.output_size);
    }
    impl_->cursor += count;
    return true;
}

} // namespace brutal_mlp

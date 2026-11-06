// In-memory and generated datasets, dataset views and splits.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"
#include "core/random_utils.hpp"
#include "data/data_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace brutal_mlp {

using namespace detail;

Dataset::~Dataset() = default;

MatrixDataset::MatrixDataset(const Matrix& inputs, const Matrix& targets)
    : inputs_(&inputs),
      targets_(&targets),
      input_size_(inputs.empty() ? 0 : inputs.front().size()),
      output_size_(targets.empty() ? 0 : targets.front().size()) {
    validate_matrix(inputs, input_size_, "inputs");
    validate_matrix(targets, output_size_, "targets");
    require(inputs.size() == targets.size(), "inputs and targets must have the same number of rows");
    require(input_size_ > 0, "inputs must have at least one column");
    require(output_size_ > 0, "targets must have at least one column");
}

std::size_t MatrixDataset::sample_count() const noexcept {
    return inputs_ ? inputs_->size() : 0;
}

std::size_t MatrixDataset::input_size() const noexcept {
    return input_size_;
}

std::size_t MatrixDataset::output_size() const noexcept {
    return output_size_;
}

void MatrixDataset::sample(std::size_t index,
                           Scalar* input,
                           std::size_t provided_input_size,
                           Scalar* target,
                           std::size_t provided_target_size) const {
    require(inputs_ != nullptr && targets_ != nullptr, "matrix dataset is not initialized");
    require(index < sample_count(), "dataset sample index is out of range");
    require(input != nullptr, "dataset input buffer must not be null");
    require(target != nullptr, "dataset target buffer must not be null");
    require(provided_input_size == input_size_, "dataset input buffer size mismatch");
    require(provided_target_size == output_size_, "dataset target buffer size mismatch");

    const Vector& source_input = (*inputs_)[index];
    const Vector& source_target = (*targets_)[index];
    std::copy(source_input.begin(), source_input.end(), input);
    std::copy(source_target.begin(), source_target.end(), target);
}

DenseMatrixDataset::DenseMatrixDataset(const DenseMatrix& inputs, const DenseMatrix& targets)
    : inputs_(&inputs),
      targets_(&targets) {
    validate_dense_matrix(inputs, inputs.columns(), "inputs");
    validate_dense_matrix(targets, targets.columns(), "targets");
    require(inputs.rows() == targets.rows(), "inputs and targets must have the same number of rows");
    require(inputs.columns() > 0, "inputs must have at least one column");
    require(targets.columns() > 0, "targets must have at least one column");
}

std::size_t DenseMatrixDataset::sample_count() const noexcept {
    return inputs_ ? inputs_->rows() : 0;
}

std::size_t DenseMatrixDataset::input_size() const noexcept {
    return inputs_ ? inputs_->columns() : 0;
}

std::size_t DenseMatrixDataset::output_size() const noexcept {
    return targets_ ? targets_->columns() : 0;
}

void DenseMatrixDataset::sample(std::size_t index,
                                Scalar* input,
                                std::size_t provided_input_size,
                                Scalar* target,
                                std::size_t provided_target_size) const {
    require(inputs_ != nullptr && targets_ != nullptr, "dense matrix dataset is not initialized");
    require(index < sample_count(), "dataset sample index is out of range");
    require(input != nullptr, "dataset input buffer must not be null");
    require(target != nullptr, "dataset target buffer must not be null");
    require(provided_input_size == input_size(), "dataset input buffer size mismatch");
    require(provided_target_size == output_size(), "dataset target buffer size mismatch");

    const Scalar* source_input = inputs_->row_data(index);
    const Scalar* source_target = targets_->row_data(index);
    std::copy(source_input, source_input + input_size(), input);
    std::copy(source_target, source_target + output_size(), target);
}

GeneratedDataset::GeneratedDataset(std::size_t sample_count_value,
                                   std::size_t input_size_value,
                                   std::size_t output_size_value,
                                   IndexedSampleFunction generate,
                                   void* context)
    : sample_count_(sample_count_value),
      input_size_(input_size_value),
      output_size_(output_size_value),
      generate_(generate),
      context_(context) {
    require(sample_count_ > 0, "generated dataset sample_count must be positive");
    require(input_size_ > 0, "generated dataset input_size must be positive");
    require(output_size_ > 0, "generated dataset output_size must be positive");
    require(generate_ != nullptr, "generated dataset callback is required");
}

std::size_t GeneratedDataset::sample_count() const noexcept {
    return sample_count_;
}

std::size_t GeneratedDataset::input_size() const noexcept {
    return input_size_;
}

std::size_t GeneratedDataset::output_size() const noexcept {
    return output_size_;
}

void GeneratedDataset::sample(std::size_t index,
                              Scalar* input,
                              std::size_t provided_input_size,
                              Scalar* target,
                              std::size_t provided_target_size) const {
    require(index < sample_count_, "generated dataset sample index is out of range");
    require(input != nullptr, "generated dataset input buffer must not be null");
    require(target != nullptr, "generated dataset target buffer must not be null");
    require(provided_input_size == input_size_, "generated dataset input buffer size mismatch");
    require(provided_target_size == output_size_, "generated dataset target buffer size mismatch");
    generate_(index, input, input_size_, target, output_size_, context_);
}

DatasetSplit make_dataset_split(std::size_t sample_count, const DatasetSplitOptions& options) {
    require(sample_count > 0, "sample_count must be positive");
    validate_dataset_split_options(options);

    std::vector<std::size_t> indices = make_indices(sample_count);
    if (options.shuffle) {
        std::mt19937_64 rng(non_zero_seed(options.seed, 5489u));
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    const std::size_t validation_count =
        static_cast<std::size_t>(std::floor(static_cast<Scalar>(sample_count) * options.validation_split));
    const std::size_t test_count =
        static_cast<std::size_t>(std::floor(static_cast<Scalar>(sample_count) * options.test_split));
    require(validation_count + test_count < sample_count,
            "dataset split leaves no training samples");

    DatasetSplit split;
    const auto test_begin = indices.end() - static_cast<std::ptrdiff_t>(test_count);
    const auto validation_begin = test_begin - static_cast<std::ptrdiff_t>(validation_count);
    split.training_indices.assign(indices.begin(), validation_begin);
    split.validation_indices.assign(validation_begin, test_begin);
    split.test_indices.assign(test_begin, indices.end());
    return split;
}

DatasetView::DatasetView(const Dataset& source, std::vector<std::size_t> indices)
    : source_(&source),
      indices_(std::move(indices)) {
    require(source_ != nullptr, "dataset view source must not be null");
    for (std::size_t index : indices_) {
        require(index < source.sample_count(), "dataset view index is out of range");
    }
}

std::size_t DatasetView::sample_count() const noexcept {
    return indices_.size();
}

std::size_t DatasetView::input_size() const noexcept {
    return source_ ? source_->input_size() : 0;
}

std::size_t DatasetView::output_size() const noexcept {
    return source_ ? source_->output_size() : 0;
}

const std::vector<std::size_t>& DatasetView::indices() const noexcept {
    return indices_;
}

void DatasetView::sample(std::size_t index,
                         Scalar* input,
                         std::size_t provided_input_size,
                         Scalar* target,
                         std::size_t provided_target_size) const {
    require(source_ != nullptr, "dataset view source must not be null");
    require(index < indices_.size(), "dataset view sample index is out of range");
    source_->sample(indices_[index], input, provided_input_size, target, provided_target_size);
}

} // namespace brutal_mlp

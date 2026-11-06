#pragma once

// Internal header: Validation of training matrices, targets and dataset shapes.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <string>

namespace brutal_mlp::detail {

void validate_matrix(const Matrix& matrix,
                     std::size_t expected_columns,
                     const std::string& name,
                     bool allow_empty = false);

void validate_dense_matrix(const DenseMatrix& matrix,
                           std::size_t expected_columns,
                           const std::string& name,
                           bool allow_empty = false);

void validate_training_data(const Matrix& inputs,
                            const Matrix& targets,
                            std::size_t input_size,
                            std::size_t output_size,
                            const LossConfig& loss);

void validate_training_data(const DenseMatrix& inputs,
                            const DenseMatrix& targets,
                            std::size_t input_size,
                            std::size_t output_size,
                            const LossConfig& loss);

void validate_dataset_shape(std::size_t sample_count,
                            std::size_t dataset_input_size,
                            std::size_t dataset_output_size,
                            std::size_t expected_input_size,
                            std::size_t expected_output_size);

void validate_sample_values(const Vector& input, const Vector& target, Loss loss);

void validate_sample_values(const Scalar* input,
                            std::size_t input_size,
                            const Scalar* target,
                            std::size_t target_size,
                            Loss loss);

void validate_dataset_split_options(const DatasetSplitOptions& options);

} // namespace brutal_mlp::detail

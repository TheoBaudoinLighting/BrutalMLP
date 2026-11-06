// Validation of training matrices, targets and dataset shapes.

#include "data/data_validation.hpp"

#include "core/checks.hpp"
#include "core/scalar_type.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace brutal_mlp::detail {

namespace {

void validate_target_for_loss(const Scalar* target, std::size_t target_size, Loss loss) {
    require(target != nullptr || target_size == 0, "target buffer must not be null");
    switch (loss) {
    case Loss::mean_squared_error:
    case Loss::mean_absolute_error:
    case Loss::huber:
    case Loss::relative_mean_squared_error:
    case Loss::relative_l2:
    case Loss::log_cosh:
    case Loss::weighted_mean_squared_error:
    case Loss::custom:
        return;
    case Loss::binary_cross_entropy:
        for (std::size_t i = 0; i < target_size; ++i) {
            const Scalar value = target[i];
            require(value >= Scalar{0} && value <= Scalar{1},
                    "binary_cross_entropy targets must be in [0, 1]");
        }
        return;
    case Loss::categorical_cross_entropy: {
        Scalar sum = Scalar{0};
        for (std::size_t i = 0; i < target_size; ++i) {
            const Scalar value = target[i];
            require(value >= Scalar{0}, "categorical_cross_entropy targets must be non-negative");
            sum += value;
        }
        require(std::abs(sum - Scalar{1}) <= kTargetSumTolerance,
                "categorical_cross_entropy targets must sum to 1");
        return;
    }
    }

    throw std::invalid_argument("unknown loss");
}

void validate_target_for_loss(const Vector& target, Loss loss) {
    validate_target_for_loss(target.data(), target.size(), loss);
}

void validate_targets_for_loss(const Matrix& targets, Loss loss) {
    for (const Vector& target : targets) {
        validate_target_for_loss(target, loss);
    }
}

void validate_targets_for_loss(const DenseMatrix& targets, Loss loss) {
    for (std::size_t row = 0; row < targets.rows(); ++row) {
        validate_target_for_loss(targets.row_data(row), targets.columns(), loss);
    }
}

} // namespace

void validate_matrix(const Matrix& matrix,
                     std::size_t expected_columns,
                     const std::string& name,
                     bool allow_empty) {
    if (!allow_empty && matrix.empty()) {
        throw_invalid_argument(name + " must not be empty");
    }
    // Messages are only built for the failing element: this loop visits every value.
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (matrix[row].size() != expected_columns) {
            throw_invalid_argument(name + " row " + std::to_string(row) + " has an invalid column count");
        }
        for (std::size_t column = 0; column < matrix[row].size(); ++column) {
            if (!is_finite(matrix[row][column])) {
                throw_non_finite(name + " row " + std::to_string(row) + " column " + std::to_string(column));
            }
        }
    }
}

void validate_dense_matrix(const DenseMatrix& matrix,
                           std::size_t expected_columns,
                           const std::string& name,
                           bool allow_empty) {
    require(allow_empty || !matrix.empty(), name + " must not be empty");
    if (matrix.empty()) {
        require(matrix.columns() == expected_columns || matrix.columns() == 0,
                name + " has an invalid column count");
        return;
    }
    require(matrix.columns() == expected_columns, name + " has an invalid column count");
    require(matrix.scalar_count() == matrix.rows() * matrix.columns(), name + " storage size mismatch");
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t column = 0; column < matrix.columns(); ++column) {
            if (!is_finite(matrix(row, column))) {
                throw_non_finite(name + " row " + std::to_string(row) + " column " + std::to_string(column));
            }
        }
    }
}

void validate_training_data(const Matrix& inputs,
                            const Matrix& targets,
                            std::size_t input_size,
                            std::size_t output_size,
                            const LossConfig& loss) {
    validate_matrix(inputs, input_size, "inputs");
    validate_matrix(targets, output_size, "targets");
    require(inputs.size() == targets.size(), "inputs and targets must have the same number of rows");
    validate_targets_for_loss(targets, loss.type);
}

void validate_training_data(const DenseMatrix& inputs,
                            const DenseMatrix& targets,
                            std::size_t input_size,
                            std::size_t output_size,
                            const LossConfig& loss) {
    validate_dense_matrix(inputs, input_size, "inputs");
    validate_dense_matrix(targets, output_size, "targets");
    require(inputs.rows() == targets.rows(), "inputs and targets must have the same number of rows");
    validate_targets_for_loss(targets, loss.type);
}

void validate_dataset_shape(std::size_t sample_count,
                            std::size_t dataset_input_size,
                            std::size_t dataset_output_size,
                            std::size_t expected_input_size,
                            std::size_t expected_output_size) {
    require(sample_count > 0, "dataset must not be empty");
    require(dataset_input_size == expected_input_size, "dataset input_size mismatch");
    require(dataset_output_size == expected_output_size, "dataset output_size mismatch");
}

void validate_sample_values(const Vector& input, const Vector& target, Loss loss) {
    for (Scalar value : input) {
        require_finite(value, "dataset input value");
    }
    for (Scalar value : target) {
        require_finite(value, "dataset target value");
    }
    validate_target_for_loss(target, loss);
}

void validate_sample_values(const Scalar* input,
                            std::size_t input_size,
                            const Scalar* target,
                            std::size_t target_size,
                            Loss loss) {
    require(input != nullptr || input_size == 0, "dataset input buffer must not be null");
    require(target != nullptr || target_size == 0, "dataset target buffer must not be null");
    for (std::size_t i = 0; i < input_size; ++i) {
        require_finite(input[i], "dataset input value");
    }
    for (std::size_t i = 0; i < target_size; ++i) {
        require_finite(target[i], "dataset target value");
    }
    validate_target_for_loss(target, target_size, loss);
}

void validate_dataset_split_options(const DatasetSplitOptions& options) {
    require_finite(options.validation_split, "validation_split");
    require_finite(options.test_split, "test_split");
    require(options.validation_split >= Scalar{0} && options.validation_split < Scalar{1},
            "validation_split must be in [0, 1)");
    require(options.test_split >= Scalar{0} && options.test_split < Scalar{1}, "test_split must be in [0, 1)");
    require(options.validation_split + options.test_split < Scalar{1},
            "validation_split + test_split must be less than 1");
}

} // namespace brutal_mlp::detail

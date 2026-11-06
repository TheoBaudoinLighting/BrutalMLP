// DenseMatrix implementation.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"
#include "data/data_validation.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace brutal_mlp {

using namespace detail;

DenseMatrix::DenseMatrix(std::size_t row_count, std::size_t column_count, Scalar value) {
    resize(row_count, column_count, value);
}

DenseMatrix DenseMatrix::from_matrix(const Matrix& matrix) {
    if (matrix.empty()) {
        return {};
    }
    const std::size_t column_count = matrix.front().size();
    validate_matrix(matrix, column_count, "matrix");
    DenseMatrix dense(matrix.size(), column_count);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        dense.set_row(row, matrix[row]);
    }
    return dense;
}

Matrix DenseMatrix::to_matrix() const {
    Matrix matrix(rows_, Vector(columns_, Scalar{0}));
    for (std::size_t row = 0; row < rows_; ++row) {
        const Scalar* source = row_data(row);
        std::copy(source, source + columns_, matrix[row].begin());
    }
    return matrix;
}

void DenseMatrix::resize(std::size_t row_count, std::size_t column_count, Scalar value) {
    require(column_count > 0 || row_count == 0, "dense matrix columns must be positive");
    if (column_count != 0) {
        require(row_count <= std::numeric_limits<std::size_t>::max() / column_count,
                "dense matrix dimensions overflow");
    }
    rows_ = row_count;
    columns_ = column_count;
    values_.assign(rows_ * columns_, value);
}

void DenseMatrix::clear() noexcept {
    rows_ = 0;
    columns_ = 0;
    values_.clear();
}

bool DenseMatrix::empty() const noexcept {
    return rows_ == 0;
}

std::size_t DenseMatrix::rows() const noexcept {
    return rows_;
}

std::size_t DenseMatrix::columns() const noexcept {
    return columns_;
}

std::size_t DenseMatrix::size() const noexcept {
    return rows_;
}

std::size_t DenseMatrix::scalar_count() const noexcept {
    return values_.size();
}

const Vector& DenseMatrix::values() const noexcept {
    return values_;
}

Scalar* DenseMatrix::data() noexcept {
    return values_.data();
}

const Scalar* DenseMatrix::data() const noexcept {
    return values_.data();
}

Scalar* DenseMatrix::row_data(std::size_t row) {
    require(row < rows_, "dense matrix row index is out of range");
    return values_.data() + row * columns_;
}

const Scalar* DenseMatrix::row_data(std::size_t row) const {
    require(row < rows_, "dense matrix row index is out of range");
    return values_.data() + row * columns_;
}

Scalar& DenseMatrix::operator()(std::size_t row, std::size_t column) {
    require(row < rows_, "dense matrix row index is out of range");
    require(column < columns_, "dense matrix column index is out of range");
    return values_[row * columns_ + column];
}

Scalar DenseMatrix::operator()(std::size_t row, std::size_t column) const {
    require(row < rows_, "dense matrix row index is out of range");
    require(column < columns_, "dense matrix column index is out of range");
    return values_[row * columns_ + column];
}

Vector DenseMatrix::row(std::size_t row_index) const {
    const Scalar* source = row_data(row_index);
    return Vector(source, source + columns_);
}

void DenseMatrix::set_row(std::size_t row_index, const Vector& values) {
    require(values.size() == columns_, "dense matrix row has an invalid column count");
    const Scalar* source = values.data();
    Scalar* destination = row_data(row_index);
    std::copy(source, source + columns_, destination);
}

} // namespace brutal_mlp

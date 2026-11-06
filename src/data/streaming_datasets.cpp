// Callback and CSV streaming datasets.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brutal_mlp {

using namespace detail;

StreamingDataset::~StreamingDataset() = default;

FunctionStreamingDataset::FunctionStreamingDataset(std::size_t sample_count_value,
                                                   std::size_t input_size_value,
                                                   std::size_t output_size_value,
                                                   StreamingSampleFunction next,
                                                   StreamingResetFunction reset,
                                                   void* context)
    : sample_count_(sample_count_value),
      input_size_(input_size_value),
      output_size_(output_size_value),
      next_(next),
      reset_(reset),
      context_(context) {
    require(sample_count_ > 0, "streaming dataset sample_count must be positive");
    require(input_size_ > 0, "streaming dataset input_size must be positive");
    require(output_size_ > 0, "streaming dataset output_size must be positive");
    require(next_ != nullptr, "streaming dataset callback is required");
}

std::size_t FunctionStreamingDataset::sample_count() const noexcept {
    return sample_count_;
}

std::size_t FunctionStreamingDataset::input_size() const noexcept {
    return input_size_;
}

std::size_t FunctionStreamingDataset::output_size() const noexcept {
    return output_size_;
}

void FunctionStreamingDataset::reset() {
    if (reset_) {
        reset_(context_);
    }
}

bool FunctionStreamingDataset::next(Scalar* input,
                                    std::size_t provided_input_size,
                                    Scalar* target,
                                    std::size_t provided_target_size) {
    require(input != nullptr, "streaming dataset input buffer must not be null");
    require(target != nullptr, "streaming dataset target buffer must not be null");
    require(provided_input_size == input_size_, "streaming dataset input buffer size mismatch");
    require(provided_target_size == output_size_, "streaming dataset target buffer size mismatch");
    return next_(input, input_size_, target, output_size_, context_);
}

namespace {

[[nodiscard]] std::size_t count_csv_samples(const std::filesystem::path& path, const CsvStreamingOptions& options) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open csv dataset for reading: " + path.string());
    }

    std::string line;
    if (options.has_header) {
        std::getline(input, line);
    }

    std::size_t count = 0;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            ++count;
        }
    }
    return count;
}

void parse_csv_line(const std::string& line,
                    char delimiter,
                    Scalar* input,
                    std::size_t input_size,
                    Scalar* target,
                    std::size_t target_size) {
    std::stringstream stream(line);
    std::string token;
    std::size_t column = 0;
    const std::size_t total_columns = input_size + target_size;

    while (std::getline(stream, token, delimiter)) {
        require(column < total_columns, "csv row has too many columns");
        Scalar value{};
        try {
            value = static_cast<Scalar>(std::stod(token));
        } catch (const std::exception&) {
            throw std::runtime_error("failed to parse csv numeric value");
        }
        require_finite(value, "csv value");
        if (column < input_size) {
            input[column] = value;
        } else {
            target[column - input_size] = value;
        }
        ++column;
    }

    require(column == total_columns, "csv row has an invalid column count");
}

} // namespace

struct CsvStreamingDataset::Impl {
    std::filesystem::path path;
    std::size_t input_size{0};
    std::size_t output_size{0};
    CsvStreamingOptions options{};
    std::size_t sample_count{0};
    std::ifstream stream;
};

CsvStreamingDataset::CsvStreamingDataset(const std::filesystem::path& path,
                                         std::size_t input_size,
                                         std::size_t output_size,
                                         CsvStreamingOptions options)
    : impl_(std::make_unique<Impl>()) {
    require(input_size > 0, "csv dataset input_size must be positive");
    require(output_size > 0, "csv dataset output_size must be positive");
    impl_->path = path;
    impl_->input_size = input_size;
    impl_->output_size = output_size;
    impl_->options = options;
    impl_->sample_count = count_csv_samples(path, options);
    require(impl_->sample_count > 0, "csv dataset must not be empty");
    reset();
}

CsvStreamingDataset::CsvStreamingDataset(const CsvStreamingDataset& other)
    : impl_(std::make_unique<Impl>()) {
    impl_->path = other.impl_->path;
    impl_->input_size = other.impl_->input_size;
    impl_->output_size = other.impl_->output_size;
    impl_->options = other.impl_->options;
    impl_->sample_count = other.impl_->sample_count;
    reset();
}

CsvStreamingDataset& CsvStreamingDataset::operator=(const CsvStreamingDataset& other) {
    if (this != &other) {
        impl_ = std::make_unique<Impl>();
        impl_->path = other.impl_->path;
        impl_->input_size = other.impl_->input_size;
        impl_->output_size = other.impl_->output_size;
        impl_->options = other.impl_->options;
        impl_->sample_count = other.impl_->sample_count;
        reset();
    }
    return *this;
}

CsvStreamingDataset::CsvStreamingDataset(CsvStreamingDataset&& other) noexcept = default;

CsvStreamingDataset& CsvStreamingDataset::operator=(CsvStreamingDataset&& other) noexcept = default;

CsvStreamingDataset::~CsvStreamingDataset() = default;

std::size_t CsvStreamingDataset::sample_count() const noexcept {
    return impl_ ? impl_->sample_count : 0;
}

std::size_t CsvStreamingDataset::input_size() const noexcept {
    return impl_ ? impl_->input_size : 0;
}

std::size_t CsvStreamingDataset::output_size() const noexcept {
    return impl_ ? impl_->output_size : 0;
}

void CsvStreamingDataset::reset() {
    require(impl_ != nullptr, "csv dataset is not initialized");
    impl_->stream.close();
    impl_->stream.clear();
    impl_->stream.open(impl_->path);
    if (!impl_->stream) {
        throw std::runtime_error("failed to open csv dataset for reading: " + impl_->path.string());
    }
    if (impl_->options.has_header) {
        std::string header;
        std::getline(impl_->stream, header);
    }
}

bool CsvStreamingDataset::next(Scalar* input,
                               std::size_t provided_input_size,
                               Scalar* target,
                               std::size_t provided_target_size) {
    require(impl_ != nullptr, "csv dataset is not initialized");
    require(input != nullptr, "csv dataset input buffer must not be null");
    require(target != nullptr, "csv dataset target buffer must not be null");
    require(provided_input_size == impl_->input_size, "csv dataset input buffer size mismatch");
    require(provided_target_size == impl_->output_size, "csv dataset target buffer size mismatch");

    std::string line;
    while (std::getline(impl_->stream, line)) {
        if (line.empty()) {
            continue;
        }
        parse_csv_line(line, impl_->options.delimiter, input, impl_->input_size, target, impl_->output_size);
        return true;
    }
    return false;
}

} // namespace brutal_mlp

#pragma once

// Internal header: Text model format readers and writers.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"

#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace brutal_mlp::detail {

[[nodiscard]] std::string read_token(std::istream& stream, const std::string& name);

template <typename T>
T read_value(std::istream& stream, const std::string& name) {
    T value{};
    if (!(stream >> value)) {
        throw std::runtime_error("failed to read " + name);
    }
    return value;
}

template <>
inline Scalar read_value<Scalar>(std::istream& stream, const std::string& name) {
    Scalar value{};
    if (!(stream >> value)) {
        throw std::runtime_error("failed to read " + name);
    }
    require_finite(value, name);
    return value;
}

void require_text_stream_consumed(std::istream& stream, const std::string& format_name);

void write_normalization(std::ostream& output, const NormalizationSpec& normalization);

[[nodiscard]] NormalizationSpec read_normalization(std::istream& input);

void write_loss_config(std::ostream& output, const LossConfig& loss);

[[nodiscard]] LossConfig read_loss_config(std::istream& input);

} // namespace brutal_mlp::detail

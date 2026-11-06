// Upper bounds enforced when reading serialized models.

#include "serialization/serialization_limits.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace brutal_mlp::detail {

namespace {

constexpr std::size_t kMaxSerializedLayers = 1000000;

constexpr std::size_t kMaxSerializedScalars = 268435456;

} // namespace

void validate_serialized_layer_count(std::size_t layer_count, const std::string& name) {
    if (layer_count == 0) {
        throw std::runtime_error(name + " layer count must be positive");
    }
    if (layer_count > kMaxSerializedLayers) {
        throw std::runtime_error(name + " layer count is unreasonably large");
    }
}

void validate_serialized_scalar_count(std::size_t count, const std::string& name) {
    if (count > kMaxSerializedScalars) {
        throw std::runtime_error(name + " count is unreasonably large");
    }
}

void require_exact_serialized_count(std::size_t actual, std::size_t expected, const std::string& name) {
    if (actual != expected) {
        throw std::runtime_error(name + " count mismatch");
    }
}

} // namespace brutal_mlp::detail

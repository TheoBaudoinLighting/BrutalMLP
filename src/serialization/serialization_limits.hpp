#pragma once

// Internal header: Upper bounds enforced when reading serialized models.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <string>

namespace brutal_mlp::detail {

void validate_serialized_layer_count(std::size_t layer_count, const std::string& name);

void validate_serialized_scalar_count(std::size_t count, const std::string& name);

void require_exact_serialized_count(std::size_t actual, std::size_t expected, const std::string& name);

} // namespace brutal_mlp::detail

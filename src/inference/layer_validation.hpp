#pragma once

// Internal header: Validation of exported layer parameters.

#include "brutal_mlp/brutal_mlp.hpp"

#include <vector>

namespace brutal_mlp::detail {

void validate_layer_parameters(const std::vector<LayerParameters>& parameters);

} // namespace brutal_mlp::detail

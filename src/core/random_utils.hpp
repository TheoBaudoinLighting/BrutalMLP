#pragma once

// Internal header: Seed resolution and index sequence helpers.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace brutal_mlp::detail {

[[nodiscard]] std::vector<std::size_t> make_indices(std::size_t size);

[[nodiscard]] std::uint64_t non_zero_seed(std::uint64_t requested, std::uint64_t fallback);

} // namespace brutal_mlp::detail

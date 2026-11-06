#pragma once

// Internal header: Seed resolution and index sequence helpers.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace brutal_mlp::detail {

[[nodiscard]] std::vector<std::size_t> make_indices(std::size_t size);

[[nodiscard]] std::uint64_t non_zero_seed(std::uint64_t requested, std::uint64_t fallback);

// SplitMix64 finalizer: a bijective mix of the 64 input bits.
[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept;

// Counter-based sampling: a uniform index in [0, bound) that depends only on (seed, stream,
// counter). Any element of a sampling sequence can be computed directly, in any order and on
// any thread, without storing the sequence. `bound` must be positive.
[[nodiscard]] std::size_t counter_based_index(std::uint64_t seed,
                                              std::uint64_t stream,
                                              std::uint64_t counter,
                                              std::size_t bound) noexcept;

} // namespace brutal_mlp::detail

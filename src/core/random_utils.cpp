// Seed resolution and index sequence helpers.

#include "core/random_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace brutal_mlp::detail {

[[nodiscard]] std::vector<std::size_t> make_indices(std::size_t size) {
    std::vector<std::size_t> indices(size);
    std::iota(indices.begin(), indices.end(), std::size_t{0});
    return indices;
}

[[nodiscard]] std::uint64_t non_zero_seed(std::uint64_t requested, std::uint64_t fallback) {
    return requested == 0 ? fallback : requested;
}

} // namespace brutal_mlp::detail

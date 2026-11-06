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

std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

namespace {

// High 64 bits of the 128-bit product, without compiler-specific 128-bit types.
[[nodiscard]] std::uint64_t multiply_high(std::uint64_t a, std::uint64_t b) noexcept {
    const std::uint64_t a_low = a & 0xFFFFFFFFULL;
    const std::uint64_t a_high = a >> 32U;
    const std::uint64_t b_low = b & 0xFFFFFFFFULL;
    const std::uint64_t b_high = b >> 32U;
    const std::uint64_t low_low = a_low * b_low;
    const std::uint64_t high_low = a_high * b_low;
    const std::uint64_t low_high = a_low * b_high;
    const std::uint64_t high_high = a_high * b_high;
    const std::uint64_t middle = (low_low >> 32U) + (high_low & 0xFFFFFFFFULL) + low_high;
    return high_high + (high_low >> 32U) + (middle >> 32U);
}

} // namespace

std::size_t counter_based_index(std::uint64_t seed,
                                std::uint64_t stream,
                                std::uint64_t counter,
                                std::size_t bound) noexcept {
    const std::uint64_t random = mix64(mix64(mix64(seed) ^ stream) + counter);
    // Maps the 64 random bits to [0, bound) by fixed-point multiplication (Lemire's method,
    // without rejection: the bias is below bound / 2^64).
    return static_cast<std::size_t>(multiply_high(random, static_cast<std::uint64_t>(bound)));
}

} // namespace brutal_mlp::detail

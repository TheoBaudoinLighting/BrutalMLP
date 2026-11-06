#pragma once

// Internal header: Argument checks, finiteness checks and overflow-safe size arithmetic.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace brutal_mlp::detail {

[[nodiscard]] inline bool is_finite(Scalar value) {
    return std::isfinite(value);
}

// Error paths are out of line so that the checks below never build a message unless they fail.
[[noreturn]] void throw_invalid_argument(const char* message);
[[noreturn]] void throw_invalid_argument(const std::string& message);
[[noreturn]] void throw_non_finite(const char* name);
[[noreturn]] void throw_non_finite(const std::string& name);

// The `const char*` overloads keep literal messages allocation-free on the success path;
// the `std::string` overloads serve messages that are already built.
inline void require(bool condition, const char* message) {
    if (!condition) {
        throw_invalid_argument(message);
    }
}

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        throw_invalid_argument(message);
    }
}

inline void require_finite(Scalar value, const char* name) {
    if (!is_finite(value)) {
        throw_non_finite(name);
    }
}

inline void require_finite(Scalar value, const std::string& name) {
    if (!is_finite(value)) {
        throw_non_finite(name);
    }
}

[[nodiscard]] std::string scalar_debug_string(Scalar value);

[[nodiscard]] std::string non_finite_kind(Scalar value);

[[nodiscard]] std::size_t checked_product(std::size_t lhs, std::size_t rhs, const std::string& name);

[[nodiscard]] std::size_t checked_size(std::uint64_t value, const std::string& name);

} // namespace brutal_mlp::detail

// Argument checks, finiteness checks and overflow-safe size arithmetic.

#include "core/checks.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brutal_mlp::detail {

void throw_invalid_argument(const char* message) {
    throw std::invalid_argument(message);
}

void throw_invalid_argument(const std::string& message) {
    throw std::invalid_argument(message);
}

void throw_non_finite(const char* name) {
    throw std::invalid_argument(std::string(name) + " must be finite");
}

void throw_non_finite(const std::string& name) {
    throw std::invalid_argument(name + " must be finite");
}

[[nodiscard]] std::string scalar_debug_string(Scalar value) {
    std::ostringstream output;
    output << std::setprecision(9) << value;
    return output.str();
}

[[nodiscard]] std::string non_finite_kind(Scalar value) {
    if (std::isnan(value)) {
        return "NaN";
    }
    if (std::isinf(value)) {
        return value > Scalar{0} ? "+Inf" : "-Inf";
    }
    return "non-finite";
}

[[nodiscard]] std::size_t checked_product(std::size_t lhs, std::size_t rhs, const std::string& name) {
    require(lhs == 0 || rhs <= std::numeric_limits<std::size_t>::max() / lhs,
            name + " dimensions overflow");
    return lhs * rhs;
}

[[nodiscard]] std::size_t checked_size(std::uint64_t value, const std::string& name) {
    require(value <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
            name + " is too large for this platform");
    return static_cast<std::size_t>(value);
}

} // namespace brutal_mlp::detail

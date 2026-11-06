#pragma once

// Internal header: Small scalar math helpers.

#include "brutal_mlp/brutal_mlp.hpp"

#include <algorithm>

namespace brutal_mlp::detail {

[[nodiscard]] inline Scalar sign_or_zero(Scalar value) noexcept {
    if (value > Scalar{0}) {
        return Scalar{1};
    }
    if (value < Scalar{0}) {
        return -Scalar{1};
    }
    return Scalar{0};
}

[[nodiscard]] inline Scalar clamp_value(Scalar value, Scalar minimum, Scalar maximum) noexcept {
    return std::min(std::max(value, minimum), maximum);
}

} // namespace brutal_mlp::detail

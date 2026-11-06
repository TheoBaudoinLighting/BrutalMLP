#pragma once

// Internal header: Scalar-type dependent constants and naming.

#include "brutal_mlp/brutal_mlp.hpp"

#include <string>

namespace brutal_mlp::detail {

#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
inline constexpr Scalar kProbabilityEpsilon = Scalar{1e-12};
inline constexpr Scalar kTargetSumTolerance = Scalar{1e-8};
#else
inline constexpr Scalar kProbabilityEpsilon = Scalar{1e-7f};
inline constexpr Scalar kTargetSumTolerance = Scalar{1e-4f};
#endif

[[nodiscard]] std::string scalar_type_string();

} // namespace brutal_mlp::detail

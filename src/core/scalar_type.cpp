// Scalar-type dependent constants and naming.

#include "core/scalar_type.hpp"

#include <string>

namespace brutal_mlp::detail {

[[nodiscard]] std::string scalar_type_string() {
    return sizeof(Scalar) == sizeof(double) ? "float64" : "float32";
}

} // namespace brutal_mlp::detail

// Library version and compiled backend reporting.

#include "brutal_mlp/brutal_mlp.hpp"

#include <string_view>

#ifndef BRUTAL_MLP_VERSION_STRING
#define BRUTAL_MLP_VERSION_STRING "0.0.0"
#endif

namespace brutal_mlp {

std::string_view version_string() noexcept {
    return BRUTAL_MLP_VERSION_STRING;
}

InferenceBackend inference_backend() noexcept {
#if defined(BRUTAL_MLP_HAS_ISPC) && !(defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE)
    return InferenceBackend::ispc;
#else
    return InferenceBackend::scalar;
#endif
}

} // namespace brutal_mlp

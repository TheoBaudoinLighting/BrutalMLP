// Dense layer kernels (scalar and optional ISPC backend).

#include "math/dense_layer.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(BRUTAL_MLP_HAS_ISPC) && !(defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE)
#include "dense_layer_kernel_ispc.h"
#endif

namespace brutal_mlp::detail {

namespace {

void dense_layer_scalar_to(const Scalar* input,
                           const Scalar* weights,
                           const Scalar* biases,
                           std::size_t input_size,
                           std::size_t output_size,
                           Scalar* output) noexcept {
    for (std::size_t out = 0; out < output_size; ++out) {
        Scalar value = biases[out];
        const std::size_t weight_offset = out * input_size;
        for (std::size_t in = 0; in < input_size; ++in) {
            value += weights[weight_offset + in] * input[in];
        }
        output[out] = value;
    }
}

#if defined(BRUTAL_MLP_HAS_ISPC) && !(defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE)
[[nodiscard]] bool ispc_layer_size_supported(std::size_t input_size, std::size_t output_size) noexcept {
    // The ISPC kernel addresses weights with 32-bit `out * input_size + column` offsets,
    // so the whole weight block, not only each dimension, must fit in int32.
    constexpr std::size_t kMaxIspcIndex =
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (input_size > kMaxIspcIndex || output_size > kMaxIspcIndex) {
        return false;
    }
    return input_size == 0 || output_size <= kMaxIspcIndex / input_size;
}
#endif

} // namespace

void dense_layer_to(const Scalar* input,
                    const Scalar* weights,
                    const Scalar* biases,
                    std::size_t input_size,
                    std::size_t output_size,
                    Scalar* output) noexcept {
#if defined(BRUTAL_MLP_HAS_ISPC) && !(defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE)
    if (ispc_layer_size_supported(input_size, output_size)) {
        ::ispc::brutal_mlp_dense_layer_f32(const_cast<Scalar*>(input),
                                           const_cast<Scalar*>(weights),
                                           const_cast<Scalar*>(biases),
                                           static_cast<std::int32_t>(input_size),
                                           static_cast<std::int32_t>(output_size),
                                           output);
        return;
    }
#endif
    dense_layer_scalar_to(input, weights, biases, input_size, output_size, output);
}

} // namespace brutal_mlp::detail

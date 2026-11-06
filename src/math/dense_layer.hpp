#pragma once

// Internal header: Dense layer kernels (scalar and optional ISPC backend).

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>

namespace brutal_mlp::detail {

// One sample through a dense layer (inference path):
// output[o] = biases[o] + sum_i weights[o * input_size + i] * input[i].
// Dispatched to the ISPC kernel when it is compiled in, to the scalar loop otherwise.
void dense_layer_to(const Scalar* input,
                    const Scalar* weights,
                    const Scalar* biases,
                    std::size_t input_size,
                    std::size_t output_size,
                    Scalar* output) noexcept;

} // namespace brutal_mlp::detail

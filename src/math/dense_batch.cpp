// Mini-batch dense layer kernels used by training.

#include "math/dense_batch.hpp"

#include <algorithm>
#include <cstddef>

#if defined(_MSC_VER)
#define BRUTAL_MLP_RESTRICT __restrict
#else
#define BRUTAL_MLP_RESTRICT __restrict__
#endif

namespace brutal_mlp::detail {

namespace {

// Number of samples processed together: each weight loaded into registers is used for
// kSampleBlock samples before the next one is loaded.
constexpr std::size_t kSampleBlock = 4;

// Cache blocking. A weight (or gradient) matrix larger than this is processed as tiles of
// whole rows of at most kTileBytes, a size that stays in the L1/L2 cache. Each tile is applied
// to every sample of the block before the next tile is loaded, so a weight read from memory
// serves the whole block instead of only kSampleBlock samples. Tiles are visited in increasing
// row order, which keeps every value's summation order unchanged.
constexpr std::size_t kTileBytes = 32 * 1024;

[[nodiscard]] std::size_t rows_per_tile(std::size_t row_length) noexcept {
    const std::size_t row_bytes = std::max<std::size_t>(1, row_length) * sizeof(Scalar);
    return std::max<std::size_t>(1, kTileBytes / row_bytes);
}

} // namespace

void transpose_weights(const Scalar* weights,
                       std::size_t input_size,
                       std::size_t output_size,
                       Scalar* weights_transposed) noexcept {
    for (std::size_t out = 0; out < output_size; ++out) {
        const Scalar* row = weights + out * input_size;
        for (std::size_t in = 0; in < input_size; ++in) {
            weights_transposed[in * output_size + out] = row[in];
        }
    }
}

void dense_forward_batch(const Scalar* inputs,
                         std::size_t count,
                         std::size_t input_size,
                         const Scalar* weights_transposed,
                         const Scalar* biases,
                         std::size_t output_size,
                         Scalar* z) noexcept {
    for (std::size_t sample = 0; sample < count; ++sample) {
        Scalar* BRUTAL_MLP_RESTRICT row = z + sample * output_size;
        for (std::size_t out = 0; out < output_size; ++out) {
            row[out] = biases[out];
        }
    }

    // Row `in` of W transposed holds the weights of input `in` for every output, so the inner
    // loop updates all outputs of four samples with one contiguous weight row. Tiles group
    // consecutive inputs; z[s][o] still receives the inputs in increasing order.
    const std::size_t tile = rows_per_tile(output_size);
    for (std::size_t tile_begin = 0; tile_begin < input_size; tile_begin += tile) {
        const std::size_t tile_end = std::min(input_size, tile_begin + tile);
        std::size_t sample = 0;
        for (; sample + kSampleBlock <= count; sample += kSampleBlock) {
            const Scalar* x0 = inputs + (sample + 0) * input_size;
            const Scalar* x1 = inputs + (sample + 1) * input_size;
            const Scalar* x2 = inputs + (sample + 2) * input_size;
            const Scalar* x3 = inputs + (sample + 3) * input_size;
            Scalar* BRUTAL_MLP_RESTRICT z0 = z + (sample + 0) * output_size;
            Scalar* BRUTAL_MLP_RESTRICT z1 = z + (sample + 1) * output_size;
            Scalar* BRUTAL_MLP_RESTRICT z2 = z + (sample + 2) * output_size;
            Scalar* BRUTAL_MLP_RESTRICT z3 = z + (sample + 3) * output_size;
            for (std::size_t in = tile_begin; in < tile_end; ++in) {
                const Scalar* BRUTAL_MLP_RESTRICT w = weights_transposed + in * output_size;
                const Scalar a0 = x0[in];
                const Scalar a1 = x1[in];
                const Scalar a2 = x2[in];
                const Scalar a3 = x3[in];
                for (std::size_t out = 0; out < output_size; ++out) {
                    z0[out] += a0 * w[out];
                    z1[out] += a1 * w[out];
                    z2[out] += a2 * w[out];
                    z3[out] += a3 * w[out];
                }
            }
        }
        for (; sample < count; ++sample) {
            const Scalar* x = inputs + sample * input_size;
            Scalar* BRUTAL_MLP_RESTRICT row = z + sample * output_size;
            for (std::size_t in = tile_begin; in < tile_end; ++in) {
                const Scalar* BRUTAL_MLP_RESTRICT w = weights_transposed + in * output_size;
                const Scalar a = x[in];
                for (std::size_t out = 0; out < output_size; ++out) {
                    row[out] += a * w[out];
                }
            }
        }
    }
}

void dense_accumulate_gradients_batch(const Scalar* delta,
                                      const Scalar* inputs,
                                      std::size_t count,
                                      std::size_t input_size,
                                      std::size_t output_size,
                                      Scalar* grad_weights,
                                      Scalar* grad_biases) noexcept {
    for (std::size_t sample = 0; sample < count; ++sample) {
        const Scalar* d = delta + sample * output_size;
        for (std::size_t out = 0; out < output_size; ++out) {
            grad_biases[out] += d[out];
        }
    }

    // Tiles group consecutive gradient rows (outputs). Within a tile every sample is added in
    // increasing order, exactly as without tiling.
    const std::size_t tile = rows_per_tile(input_size);
    for (std::size_t tile_begin = 0; tile_begin < output_size; tile_begin += tile) {
        const std::size_t tile_end = std::min(output_size, tile_begin + tile);
        std::size_t sample = 0;
        for (; sample + kSampleBlock <= count; sample += kSampleBlock) {
            const Scalar* a0 = inputs + (sample + 0) * input_size;
            const Scalar* a1 = inputs + (sample + 1) * input_size;
            const Scalar* a2 = inputs + (sample + 2) * input_size;
            const Scalar* a3 = inputs + (sample + 3) * input_size;
            for (std::size_t out = tile_begin; out < tile_end; ++out) {
                const Scalar d0 = delta[(sample + 0) * output_size + out];
                const Scalar d1 = delta[(sample + 1) * output_size + out];
                const Scalar d2 = delta[(sample + 2) * output_size + out];
                const Scalar d3 = delta[(sample + 3) * output_size + out];
                Scalar* BRUTAL_MLP_RESTRICT row = grad_weights + out * input_size;
                for (std::size_t in = 0; in < input_size; ++in) {
                    // Four additions in sample order: the same rounding steps as four
                    // separate one-sample updates, with one load and one store of row[in].
                    Scalar value = row[in];
                    value += d0 * a0[in];
                    value += d1 * a1[in];
                    value += d2 * a2[in];
                    value += d3 * a3[in];
                    row[in] = value;
                }
            }
        }
        for (; sample < count; ++sample) {
            const Scalar* a = inputs + sample * input_size;
            for (std::size_t out = tile_begin; out < tile_end; ++out) {
                const Scalar d = delta[sample * output_size + out];
                Scalar* BRUTAL_MLP_RESTRICT row = grad_weights + out * input_size;
                for (std::size_t in = 0; in < input_size; ++in) {
                    row[in] += d * a[in];
                }
            }
        }
    }
}

void dense_input_gradient_batch(const Scalar* delta,
                                const Scalar* weights,
                                std::size_t count,
                                std::size_t input_size,
                                std::size_t output_size,
                                Scalar* input_gradient) noexcept {
    for (std::size_t sample = 0; sample < count; ++sample) {
        Scalar* BRUTAL_MLP_RESTRICT g = input_gradient + sample * input_size;
        for (std::size_t in = 0; in < input_size; ++in) {
            g[in] = Scalar{0};
        }
    }

    // Row `out` of W is contiguous over inputs: one weight row serves four samples. Tiles group
    // consecutive rows; input_gradient[s][i] still receives the outputs in increasing order.
    const std::size_t tile = rows_per_tile(input_size);
    for (std::size_t tile_begin = 0; tile_begin < output_size; tile_begin += tile) {
        const std::size_t tile_end = std::min(output_size, tile_begin + tile);
        std::size_t sample = 0;
        for (; sample + kSampleBlock <= count; sample += kSampleBlock) {
            Scalar* BRUTAL_MLP_RESTRICT g0 = input_gradient + (sample + 0) * input_size;
            Scalar* BRUTAL_MLP_RESTRICT g1 = input_gradient + (sample + 1) * input_size;
            Scalar* BRUTAL_MLP_RESTRICT g2 = input_gradient + (sample + 2) * input_size;
            Scalar* BRUTAL_MLP_RESTRICT g3 = input_gradient + (sample + 3) * input_size;
            for (std::size_t out = tile_begin; out < tile_end; ++out) {
                const Scalar* BRUTAL_MLP_RESTRICT w = weights + out * input_size;
                const Scalar d0 = delta[(sample + 0) * output_size + out];
                const Scalar d1 = delta[(sample + 1) * output_size + out];
                const Scalar d2 = delta[(sample + 2) * output_size + out];
                const Scalar d3 = delta[(sample + 3) * output_size + out];
                for (std::size_t in = 0; in < input_size; ++in) {
                    g0[in] += w[in] * d0;
                    g1[in] += w[in] * d1;
                    g2[in] += w[in] * d2;
                    g3[in] += w[in] * d3;
                }
            }
        }
        for (; sample < count; ++sample) {
            Scalar* BRUTAL_MLP_RESTRICT g = input_gradient + sample * input_size;
            for (std::size_t out = tile_begin; out < tile_end; ++out) {
                const Scalar* BRUTAL_MLP_RESTRICT w = weights + out * input_size;
                const Scalar d = delta[sample * output_size + out];
                for (std::size_t in = 0; in < input_size; ++in) {
                    g[in] += w[in] * d;
                }
            }
        }
    }
}

} // namespace brutal_mlp::detail

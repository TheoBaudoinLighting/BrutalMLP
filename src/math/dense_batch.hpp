#pragma once

// Internal header: Mini-batch dense layer kernels used by training.
//
// Matrices are row-major and contiguous: a batch of `count` samples of width `n` is
// `count * n` scalars, one row per sample. W is the layer weight matrix with one row
// per output (`output_size x input_size`), as stored by the model.
//
// Every kernel keeps, for each output value, the exact sequence of floating-point
// operations of the one-sample loops it replaces (same operands, same summation order).
// Batched training is therefore bit-identical to sample-by-sample training; the speed-up
// comes from reusing each loaded weight for several samples and from contiguous inner
// loops that the compiler can vectorize.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>

namespace brutal_mlp::detail {

// weights_transposed[i * output_size + o] = weights[o * input_size + i].
void transpose_weights(const Scalar* weights,
                       std::size_t input_size,
                       std::size_t output_size,
                       Scalar* weights_transposed) noexcept;

// Forward pass of a dense layer for a batch:
// z[s][o] = biases[o] + sum over i (in increasing i) of inputs[s][i] * W[o][i].
// Takes W transposed (`input_size x output_size`) so that the inner loop runs over
// contiguous outputs.
void dense_forward_batch(const Scalar* inputs,
                         std::size_t count,
                         std::size_t input_size,
                         const Scalar* weights_transposed,
                         const Scalar* biases,
                         std::size_t output_size,
                         Scalar* z) noexcept;

// Parameter gradients of a dense layer, accumulated sample after sample (s = 0, 1, ...):
// grad_weights[o][i] += delta[s][o] * inputs[s][i] and grad_biases[o] += delta[s][o].
void dense_accumulate_gradients_batch(const Scalar* delta,
                                      const Scalar* inputs,
                                      std::size_t count,
                                      std::size_t input_size,
                                      std::size_t output_size,
                                      Scalar* grad_weights,
                                      Scalar* grad_biases) noexcept;

// Gradient with respect to the layer input, before the previous activation's derivative:
// input_gradient[s][i] = sum over o (in increasing o) of W[o][i] * delta[s][o].
void dense_input_gradient_batch(const Scalar* delta,
                                const Scalar* weights,
                                std::size_t count,
                                std::size_t input_size,
                                std::size_t output_size,
                                Scalar* input_gradient) noexcept;

} // namespace brutal_mlp::detail

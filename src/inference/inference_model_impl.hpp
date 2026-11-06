#pragma once

// Internal header: InferenceModel::Impl: compact parameter storage and the inference hot path.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>
#include <vector>

namespace brutal_mlp {

struct InferenceModel::Impl {
    struct Layer {
        std::size_t input_size{0};
        std::size_t output_size{0};
        std::size_t weight_offset{0};
        std::size_t bias_offset{0};
        Activation activation{Activation::linear};
    };

    std::size_t input_size{0};
    std::size_t output_size{0};
    std::size_t scratch_stride{0};
    std::size_t input_scratch_size{0};
    NormalizationSpec normalization{};
    std::vector<Layer> layers;
    Vector weights;
    Vector biases;

    [[nodiscard]] std::size_t scratch_size() const noexcept;

    void predict_unchecked_to(const Scalar* input, Scalar* output, Scalar* scratch) const noexcept;

    void predict_batch_range_unchecked_to(const Scalar* inputs,
                                          std::size_t sample_begin,
                                          std::size_t sample_end,
                                          std::size_t input_stride,
                                          Scalar* outputs,
                                          std::size_t output_stride,
                                          Scalar* scratch) const noexcept;

    void predict_batch_unchecked_to(const Scalar* inputs,
                                    std::size_t sample_count,
                                    std::size_t input_stride,
                                    Scalar* outputs,
                                    std::size_t output_stride,
                                    Scalar* scratch) const noexcept;
};

} // namespace brutal_mlp

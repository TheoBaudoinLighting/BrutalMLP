#pragma once

// Internal header: Binary model envelope, sections and payload parsing.

#include "brutal_mlp/brutal_mlp.hpp"

#include "serialization/binary_io.hpp"
#include "training/layer_training_state.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace brutal_mlp::detail {

void write_binary_metadata(BinaryWriter& writer, BinaryMetadata metadata);

void write_binary_training_options(BinaryWriter& writer, const TrainingOptions& options);

void write_binary_learning_rate_scheduler_state(BinaryWriter& writer,
                                                const LearningRateSchedulerState& state);

void write_binary_optimizer(BinaryWriter& writer, const OptimizerConfig& optimizer);

void write_binary_loss_config(BinaryWriter& writer, const LossConfig& loss);

void write_binary_normalization(BinaryWriter& writer, const NormalizationSpec& normalization);

void write_binary_layer(BinaryWriter& writer, const LayerParameters& layer);

void write_binary_layer_training_state(BinaryWriter& writer, const LayerTrainingState& state);

[[nodiscard]] std::vector<unsigned char> build_binary_envelope(BinaryModelKind kind,
                                                               const std::vector<unsigned char>& payload);

struct ParsedBinaryModel {
    BinaryModelInfo info;
    LossConfig loss{};
    OptimizerConfig optimizer{OptimizerConfig::adam()};
    NormalizationSpec normalization{};
    std::uint64_t optimizer_step{0};
    std::vector<LayerParameters> parameters;
    std::vector<LayerTrainingState> training_states;
};

[[nodiscard]] ParsedBinaryModel parse_binary_model_file(const std::filesystem::path& path);

void write_common_binary_model_payload(BinaryWriter& writer,
                                       const BinaryMetadata& metadata,
                                       std::size_t input_size,
                                       std::size_t output_size,
                                       std::uint64_t seed,
                                       bool has_training_options,
                                       const TrainingOptions& training_options,
                                       const NormalizationSpec& normalization,
                                       const std::vector<LayerParameters>& parameters);

} // namespace brutal_mlp::detail

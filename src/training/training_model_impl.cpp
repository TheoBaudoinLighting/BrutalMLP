// TrainingModel::Impl: forward/backward passes, losses and optimizer updates.

#include "training/training_model_impl.hpp"

#include "config/config_validation.hpp"
#include "core/checks.hpp"
#include "core/random_utils.hpp"
#include "core/scalar_type.hpp"
#include "core/time_utils.hpp"
#include "data/data_validation.hpp"
#include "math/activations.hpp"
#include "core/parallel_workers.hpp"
#include "math/dense_batch.hpp"
#include "math/scalar_math.hpp"
#include "normalization/normalization.hpp"
#include "training/layer_training_state.hpp"
#include "training/training_diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brutal_mlp {

using namespace detail;

namespace {

// Largest block of samples pushed through the network at once. Larger mini-batches are split
// into blocks; 128 rows of the widest layers stay in the L1/L2 caches.
constexpr std::size_t kWorkspaceBlockSamples = 128;

// Parallel training splits a mini-batch into gradient chunks. The split depends only on the
// batch size and on the parameter count, never on the thread count, so results do not depend
// on how many threads run.
constexpr std::size_t kMinChunkSamples = 64;          // Enough samples to amortize a chunk buffer.
constexpr std::size_t kMaxGradientChunks = 32;        // Upper bound on parallel tasks per batch.
constexpr std::size_t kChunkGradientBudget = 1u << 23; // Scalars for all chunk buffers (32 MiB in float).
constexpr std::size_t kLossChunkSamples = 256;        // Samples per task of the loss evaluation.
constexpr std::size_t kLossChunksPerBlock = 16;       // Loss tasks per block read from the dataset.
constexpr std::size_t kReductionSlice = 4096;         // Parameters summed per reduction task.

[[nodiscard]] std::size_t gradient_chunk_count(std::size_t samples, std::size_t parameter_count) noexcept {
    const std::size_t by_samples = (samples + kMinChunkSamples - 1) / kMinChunkSamples;
    const std::size_t by_memory = std::max<std::size_t>(1, kChunkGradientBudget / std::max<std::size_t>(1, parameter_count));
    return std::max<std::size_t>(1, std::min({by_samples, kMaxGradientChunks, by_memory}));
}

// First sample of chunk `chunk` when `samples` are split into `chunks` nearly equal ranges.
[[nodiscard]] std::size_t chunk_begin(std::size_t chunk, std::size_t chunks, std::size_t samples) noexcept {
    return chunk * samples / chunks;
}

} // namespace

std::size_t TrainingModel::Impl::output_size() const {
    return layers.back().output_size;
}

TrainingRunConfig TrainingModel::Impl::make_training_run_config(const OptimizerConfig& run_optimizer,
                                                                const TrainingOptions& runtime_options,
                                                                const DatasetSplit& split,
                                                                std::size_t sample_count,
                                                                bool streaming) const {
    TrainingRunConfig config;
    config.library_version = std::string(version_string());
    config.scalar_type = scalar_type_string();
    config.created_unix_time = current_unix_time();
    config.created_utc = utc_timestamp(config.created_unix_time);
    config.model_seed = seed;
    config.training_seed = non_zero_seed(runtime_options.seed, seed);
    config.input_size = input_size;
    config.output_size = output_size();

    config.architecture.reserve(layers.size());
    for (const Layer& layer : layers) {
        TrainingRunLayerConfig layer_config;
        layer_config.input_size = layer.input_size;
        layer_config.output_size = layer.output_size;
        layer_config.activation = layer.activation;
        layer_config.dropout_probability = layer.dropout_probability;
        config.architecture.push_back(layer_config);
    }

    config.optimizer = run_optimizer;
    config.learning_rate_schedule = runtime_options.learning_rate_schedule;
    config.loss = loss;
    config.normalization = normalization;
    config.options = runtime_options;

    config.dataset.sample_count = sample_count;
    config.dataset.input_size = input_size;
    config.dataset.output_size = output_size();
    config.dataset.training_sample_count = split.training_indices.size();
    config.dataset.validation_sample_count = split.validation_indices.size();
    config.dataset.test_sample_count = split.test_indices.size();
    config.dataset.validation_split = runtime_options.validation_split;
    config.dataset.test_split = runtime_options.test_split;
    config.dataset.shuffle = runtime_options.shuffle;
    config.dataset.split_seed = config.training_seed;
    config.dataset.streaming = streaming;
    config.dataset.streaming_shuffle_buffer_size = runtime_options.streaming_shuffle_buffer_size;

    config.result.monitor = runtime_options.early_stopping_monitor;
    config.result.monitor_mode = runtime_options.early_stopping_mode;
    return config;
}

Vector TrainingModel::Impl::forward(const Vector& input) const {
    Vector activations = input;
    for (const Layer& layer : layers) {
        Vector z(layer.output_size, Scalar{0});
        for (std::size_t out = 0; out < layer.output_size; ++out) {
            Scalar value = layer.biases[out];
            const std::size_t offset = out * layer.input_size;
            for (std::size_t in = 0; in < layer.input_size; ++in) {
                value += layer.weights[offset + in] * activations[in];
            }
            z[out] = value;
        }
        activations = apply_activation(z, layer.activation);
    }
    return activations;
}

Vector TrainingModel::Impl::predict_raw(const Vector& input) const {
    return denormalize_vector(forward(normalize_vector(input, normalization.input_features)),
                              normalization.output_features);
}

Vector TrainingModel::Impl::predict_raw(const Scalar* input, std::size_t input_count) const {
    require(input != nullptr, "input buffer must not be null");
    require(input_count == input_size, "input has an invalid size");
    Vector normalized(input_count, Scalar{0});
    normalize_buffer_into(input, input_count, normalization.input_features, normalized.data());
    return denormalize_vector(forward(normalized), normalization.output_features);
}

TrainingModel::Impl::TrainingWorkspace TrainingModel::Impl::make_workspace(std::size_t batch_size) const {
    TrainingWorkspace workspace;
    workspace.capacity = std::max<std::size_t>(1, std::min(batch_size, kWorkspaceBlockSamples));
    const std::size_t rows = workspace.capacity;
    std::size_t widest = input_size;
    workspace.activations.resize(layers.size() + 1);
    workspace.pre_activations.resize(layers.size());
    workspace.dropout_masks.resize(layers.size());
    workspace.transposed_weights.resize(layers.size());
    workspace.activations[0].assign(rows * input_size, Scalar{0});
    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const Layer& layer = layers[layer_index];
        workspace.pre_activations[layer_index].assign(rows * layer.output_size, Scalar{0});
        workspace.activations[layer_index + 1].assign(rows * layer.output_size, Scalar{0});
        workspace.dropout_masks[layer_index].assign(rows * layer.output_size, Scalar{1});
        workspace.transposed_weights[layer_index].assign(layer.weights.size(), Scalar{0});
        widest = std::max(widest, layer.output_size);
    }
    // delta and previous_delta swap roles layer after layer, so both hold the widest layer.
    workspace.delta.assign(rows * widest, Scalar{0});
    workspace.previous_delta.assign(rows * widest, Scalar{0});
    workspace.activation_gradient.assign(output_size(), Scalar{0});
    workspace.raw_batch.resize(rows, input_size, output_size());
    workspace.normalized_batch.resize(rows, input_size, output_size());
    workspace.layer_sink.weights.assign(layers.size(), nullptr);
    workspace.layer_sink.biases.assign(layers.size(), nullptr);
    return workspace;
}

void TrainingModel::Impl::draw_dropout_masks(std::size_t count,
                                             std::mt19937_64& rng,
                                             std::vector<Vector>& masks) const {
    for (std::size_t sample = 0; sample < count; ++sample) {
        for (std::size_t layer_index = 0; layer_index + 1 < layers.size(); ++layer_index) {
            const Layer& layer = layers[layer_index];
            if (layer.dropout_probability <= Scalar{0}) {
                continue;
            }
            // Inverted dropout: kept units are scaled by 1 / keep so the expected value is unchanged.
            const Scalar keep_probability = Scalar{1} - layer.dropout_probability;
            const Scalar scale = Scalar{1} / keep_probability;
            std::bernoulli_distribution keep(static_cast<double>(keep_probability));
            Scalar* mask = masks[layer_index].data() + sample * layer.output_size;
            for (std::size_t unit = 0; unit < layer.output_size; ++unit) {
                mask[unit] = keep(rng) ? scale : Scalar{0};
            }
        }
    }
}

void TrainingModel::Impl::forward_batch(const Scalar* inputs,
                                        std::size_t count,
                                        TrainingWorkspace& workspace,
                                        bool apply_dropout) const {
    require(inputs != nullptr, "input buffer must not be null");
    require(count > 0 && count <= workspace.capacity, "forward block size is out of range");
    std::copy(inputs, inputs + count * input_size, workspace.activations[0].begin());

    // The forward kernel reads W transposed; refresh the copies only after the weights changed.
    if (!workspace.has_transposed_weights || workspace.transposed_version != parameter_version) {
        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            const Layer& layer = layers[layer_index];
            transpose_weights(layer.weights.data(),
                              layer.input_size,
                              layer.output_size,
                              workspace.transposed_weights[layer_index].data());
        }
        workspace.transposed_version = parameter_version;
        workspace.has_transposed_weights = true;
    }

    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const Layer& layer = layers[layer_index];
        Scalar* z = workspace.pre_activations[layer_index].data();
        Scalar* activation = workspace.activations[layer_index + 1].data();
        const std::size_t values = count * layer.output_size;

        // z = W * previous + b for every sample, then activation = f(z) row by row.
        dense_forward_batch(workspace.activations[layer_index].data(),
                            count,
                            layer.input_size,
                            workspace.transposed_weights[layer_index].data(),
                            layer.biases.data(),
                            layer.output_size,
                            z);
        std::copy(z, z + values, activation);
        for (std::size_t sample = 0; sample < count; ++sample) {
            apply_activation_in_place(activation + sample * layer.output_size, layer.output_size, layer.activation);
        }

        const bool output_layer = layer_index + 1 == layers.size();
        if (apply_dropout && !output_layer && layer.dropout_probability > Scalar{0}) {
            const Scalar* mask = workspace.dropout_masks[layer_index].data();
            for (std::size_t i = 0; i < values; ++i) {
                activation[i] = mask[i] == Scalar{0} ? Scalar{0} : activation[i] * mask[i];
            }
        }
    }
}

Scalar TrainingModel::Impl::sample_loss(const Scalar* prediction,
                                        std::size_t prediction_size,
                                        const Scalar* target,
                                        std::size_t target_count) const {
    require(target != nullptr, "target buffer must not be null");
    require(target_count == prediction_size, "target has an invalid size");
    Scalar total = Scalar{0};

    switch (loss.type) {
    case Loss::mean_squared_error:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar delta = prediction[i] - target[i];
            total += delta * delta;
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::mean_absolute_error:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            total += static_cast<Scalar>(std::abs(prediction[i] - target[i]));
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::huber:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar abs_delta = static_cast<Scalar>(std::abs(prediction[i] - target[i]));
            if (abs_delta <= loss.huber_delta) {
                total += Scalar{0.5} * abs_delta * abs_delta;
            } else {
                total += loss.huber_delta * (abs_delta - Scalar{0.5} * loss.huber_delta);
            }
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::relative_mean_squared_error:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar delta = prediction[i] - target[i];
            const Scalar denominator = std::max(static_cast<Scalar>(std::abs(target[i])), loss.relative_epsilon);
            total += (delta * delta) / (denominator * denominator);
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::relative_l2:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar error = prediction[i] - target[i];
            const Scalar denominator = prediction[i] * prediction[i] + loss.relative_epsilon;
            total += error * error / denominator;
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::log_cosh:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar abs_delta = static_cast<Scalar>(std::abs(prediction[i] - target[i]));
            total += abs_delta +
                     static_cast<Scalar>(std::log1p(std::exp(-Scalar{2} * abs_delta))) -
                     static_cast<Scalar>(std::log(Scalar{2}));
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::weighted_mean_squared_error: {
        const Scalar total_weight = std::accumulate(loss.weights.begin(), loss.weights.end(), Scalar{0});
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar delta = prediction[i] - target[i];
            total += loss.weights[i] * delta * delta;
        }
        return total / total_weight;
    }
    case Loss::binary_cross_entropy:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar p = clamp_probability(prediction[i]);
            total += static_cast<Scalar>(-target[i] * std::log(p) -
                                         (Scalar{1} - target[i]) * std::log(Scalar{1} - p));
        }
        return total / static_cast<Scalar>(prediction_size);
    case Loss::categorical_cross_entropy:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            total += static_cast<Scalar>(-target[i] * std::log(clamp_probability(prediction[i])));
        }
        return total;
    case Loss::custom: {
        const Scalar value = loss.custom_loss.value(prediction,
                                                    target,
                                                    prediction_size,
                                                    loss.custom_loss.context);
        require_finite(value, "custom loss");
        return value;
    }
    }

    throw std::invalid_argument("unknown loss");
}

Scalar TrainingModel::Impl::sample_loss(const Vector& prediction, const Scalar* target, std::size_t target_count) const {
    return sample_loss(prediction.data(), prediction.size(), target, target_count);
}

Scalar TrainingModel::Impl::sample_loss(const Vector& prediction, const Vector& target) const {
    return sample_loss(prediction, target.data(), target.size());
}

void TrainingModel::Impl::output_delta_into(const Scalar* prediction,
                                            std::size_t prediction_size,
                                            const Scalar* target,
                                            std::size_t target_count,
                                            const Scalar* output_pre_activation,
                                            Scalar* delta,
                                            Scalar* activation_gradient) const {
    require(target != nullptr, "target buffer must not be null");
    require(target_count == prediction_size, "target has an invalid size");
    const Layer& output_layer = layers.back();

    // Fused cases: the softmax/sigmoid Jacobian cancels against the cross-entropy gradient.
    if (loss.type == Loss::categorical_cross_entropy && output_layer.activation == Activation::softmax) {
        for (std::size_t i = 0; i < prediction_size; ++i) {
            delta[i] = prediction[i] - target[i];
        }
        return;
    }

    if (loss.type == Loss::binary_cross_entropy && output_layer.activation == Activation::sigmoid) {
        const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            delta[i] = (prediction[i] - target[i]) * scale;
        }
        return;
    }

    // General case: dLoss/d(output) first, then the chain rule through the output activation.
    std::fill(activation_gradient, activation_gradient + prediction_size, Scalar{0});
    switch (loss.type) {
    case Loss::mean_squared_error: {
        const Scalar scale = Scalar{2} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            activation_gradient[i] = (prediction[i] - target[i]) * scale;
        }
        break;
    }
    case Loss::mean_absolute_error: {
        const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar delta_value = prediction[i] - target[i];
            if (delta_value > Scalar{0}) {
                activation_gradient[i] = scale;
            } else if (delta_value < Scalar{0}) {
                activation_gradient[i] = -scale;
            }
        }
        break;
    }
    case Loss::huber: {
        const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar delta_value = prediction[i] - target[i];
            const Scalar abs_delta = static_cast<Scalar>(std::abs(delta_value));
            if (abs_delta <= loss.huber_delta) {
                activation_gradient[i] = delta_value * scale;
            } else {
                activation_gradient[i] = (delta_value > Scalar{0} ? loss.huber_delta : -loss.huber_delta) * scale;
            }
        }
        break;
    }
    case Loss::relative_mean_squared_error: {
        const Scalar scale = Scalar{2} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar denominator = std::max(static_cast<Scalar>(std::abs(target[i])), loss.relative_epsilon);
            activation_gradient[i] =
                (prediction[i] - target[i]) * scale / (denominator * denominator);
        }
        break;
    }
    case Loss::relative_l2: {
        // The denominator is a stop-gradient term: only the squared error is differentiated.
        const Scalar channel_scale = Scalar{1} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar error = prediction[i] - target[i];
            const Scalar denominator = prediction[i] * prediction[i] + loss.relative_epsilon;
            activation_gradient[i] = Scalar{2} * channel_scale * error / denominator;
        }
        break;
    }
    case Loss::log_cosh: {
        const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            activation_gradient[i] = static_cast<Scalar>(std::tanh(prediction[i] - target[i])) * scale;
        }
        break;
    }
    case Loss::weighted_mean_squared_error: {
        const Scalar total_weight = std::accumulate(loss.weights.begin(), loss.weights.end(), Scalar{0});
        for (std::size_t i = 0; i < prediction_size; ++i) {
            activation_gradient[i] =
                Scalar{2} * loss.weights[i] * (prediction[i] - target[i]) / total_weight;
        }
        break;
    }
    case Loss::binary_cross_entropy: {
        const Scalar scale = Scalar{1} / static_cast<Scalar>(prediction_size);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            const Scalar p = clamp_probability(prediction[i]);
            activation_gradient[i] = ((Scalar{1} - target[i]) / (Scalar{1} - p) - target[i] / p) * scale;
        }
        break;
    }
    case Loss::categorical_cross_entropy:
        for (std::size_t i = 0; i < prediction_size; ++i) {
            activation_gradient[i] = -target[i] / clamp_probability(prediction[i]);
        }
        break;
    case Loss::custom:
        loss.custom_loss.gradient(prediction,
                                  target,
                                  prediction_size,
                                  activation_gradient,
                                  loss.custom_loss.context);
        for (std::size_t i = 0; i < prediction_size; ++i) {
            require_finite(activation_gradient[i], "custom loss gradient");
        }
        break;
    }

    if (output_layer.activation == Activation::softmax) {
        const Scalar dot = std::inner_product(activation_gradient,
                                              activation_gradient + prediction_size,
                                              prediction,
                                              Scalar{0});
        for (std::size_t i = 0; i < prediction_size; ++i) {
            delta[i] = prediction[i] * (activation_gradient[i] - dot);
        }
        return;
    }

    for (std::size_t i = 0; i < prediction_size; ++i) {
        delta[i] = activation_gradient[i] *
                   activation_derivative(output_layer.activation, output_pre_activation[i], prediction[i]);
    }
}

void TrainingModel::Impl::accumulate_block_gradients(const Scalar* inputs,
                                                     const Scalar* targets,
                                                     std::size_t count,
                                                     TrainingWorkspace& workspace,
                                                     bool apply_dropout,
                                                     const GradientSink& sink) const {
    require(targets != nullptr || count == 0, "target buffer must not be null");
    const std::size_t outputs = output_size();
    forward_batch(inputs, count, workspace, apply_dropout);

    // dLoss/dz of the output layer, one row per sample.
    Scalar* delta = workspace.delta.data();
    Scalar* previous_delta = workspace.previous_delta.data();
    const Scalar* predictions = workspace.activations.back().data();
    const Scalar* output_z = workspace.pre_activations.back().data();
    for (std::size_t sample = 0; sample < count; ++sample) {
        output_delta_into(predictions + sample * outputs,
                          outputs,
                          targets + sample * outputs,
                          outputs,
                          output_z + sample * outputs,
                          delta + sample * outputs,
                          workspace.activation_gradient.data());
    }

    // Back-propagation, last layer first. `delta` holds dLoss/dz of the current layer for
    // every sample; `previous_delta` receives the layer before. The buffers then swap.
    for (std::size_t layer_offset = layers.size(); layer_offset-- > 0;) {
        const Layer& layer = layers[layer_offset];
        const Scalar* layer_inputs = workspace.activations[layer_offset].data();
        dense_accumulate_gradients_batch(delta,
                                         layer_inputs,
                                         count,
                                         layer.input_size,
                                         layer.output_size,
                                         sink.weights[layer_offset],
                                         sink.biases[layer_offset]);
        if (layer_offset == 0) {
            break;
        }

        // dLoss/d(previous activation) = W^T * delta, then times f'(z) of the previous layer
        // (and the dropout mask that scaled that activation during the forward pass).
        dense_input_gradient_batch(delta,
                                   layer.weights.data(),
                                   count,
                                   layer.input_size,
                                   layer.output_size,
                                   previous_delta);
        const Layer& previous_layer = layers[layer_offset - 1];
        const Scalar* previous_z = workspace.pre_activations[layer_offset - 1].data();
        const Scalar* previous_activation = workspace.activations[layer_offset].data();
        const Scalar* mask = workspace.dropout_masks[layer_offset - 1].data();
        const std::size_t values = count * layer.input_size;
        for (std::size_t i = 0; i < values; ++i) {
            Scalar value = previous_delta[i];
            if (apply_dropout) {
                value *= mask[i];
                previous_delta[i] =
                    value * activation_derivative_from_pre_activation(previous_layer.activation, previous_z[i]);
            } else {
                previous_delta[i] =
                    value * activation_derivative(previous_layer.activation, previous_z[i], previous_activation[i]);
            }
        }
        std::swap(delta, previous_delta);
    }
}

void TrainingModel::Impl::refresh_layer_gradient_sink(GradientSink& sink) {
    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        sink.weights[layer_index] = layers[layer_index].grad_weights.data();
        sink.biases[layer_index] = layers[layer_index].grad_biases.data();
    }
}

void TrainingModel::Impl::accumulate_batch_gradients(const Scalar* inputs,
                                                     const Scalar* targets,
                                                     std::size_t count,
                                                     TrainingWorkspace& workspace,
                                                     std::mt19937_64* rng) {
    require(inputs != nullptr || count == 0, "input buffer must not be null");
    require(targets != nullptr || count == 0, "target buffer must not be null");
    refresh_layer_gradient_sink(workspace.layer_sink);
    const std::size_t outputs = output_size();
    for (std::size_t begin = 0; begin < count; begin += workspace.capacity) {
        const std::size_t block = std::min(workspace.capacity, count - begin);
        if (rng != nullptr) {
            draw_dropout_masks(block, *rng, workspace.dropout_masks);
        }
        accumulate_block_gradients(inputs + begin * input_size,
                                   targets + begin * outputs,
                                   block,
                                   workspace,
                                   rng != nullptr,
                                   workspace.layer_sink);
    }
}

void TrainingModel::Impl::accumulate_batch_gradients(const MiniBatch& batch,
                                                     TrainingWorkspace& workspace,
                                                     std::mt19937_64* rng) {
    require(batch.input_size == input_size, "mini-batch input width mismatch");
    require(batch.output_size == output_size(), "mini-batch output width mismatch");
    accumulate_batch_gradients(batch.inputs.data(), batch.targets.data(), batch.size(), workspace, rng);
}

TrainingModel::Impl::ParallelTraining TrainingModel::Impl::make_parallel_training(const TrainingOptions& options,
                                                                                 std::size_t sample_count) const {
    ParallelTraining parallel;
    const std::size_t batch_size = options.batch_size;
    parallel.layer_offsets.reserve(layers.size());
    for (const Layer& layer : layers) {
        parallel.layer_offsets.push_back(parallel.parameter_count);
        parallel.parameter_count += layer.weights.size() + layer.biases.size();
    }
    const std::size_t chunks = gradient_chunk_count(batch_size, parallel.parameter_count);

    // Threads beyond the number of independent tasks would only add wake-up costs: a batch of
    // 32 samples, for instance, is a single gradient chunk.
    const std::size_t loss_tasks =
        std::min(kLossChunksPerBlock, (sample_count + kLossChunkSamples - 1) / kLossChunkSamples);
    const std::size_t useful_workers = std::max<std::size_t>(1, std::max(chunks, loss_tasks));
    const std::size_t requested_workers = resolve_parallel_worker_count(options.parallelism, batch_size);
    parallel.pool = std::make_unique<WorkerPool>(std::max<std::size_t>(1, std::min(requested_workers, useful_workers)));
    for (std::size_t worker = 0; worker < parallel.pool->worker_count(); ++worker) {
        parallel.workspaces.push_back(make_workspace(kWorkspaceBlockSamples));
    }

    parallel.chunk_gradients.assign(chunks, Vector(parallel.parameter_count, Scalar{0}));
    parallel.chunk_sinks.resize(chunks);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
        GradientSink& sink = parallel.chunk_sinks[chunk];
        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            Scalar* base = parallel.chunk_gradients[chunk].data() + parallel.layer_offsets[layer_index];
            sink.weights.push_back(base);
            sink.biases.push_back(base + layers[layer_index].weights.size());
        }
    }

    parallel.batch_masks.resize(layers.size());
    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        parallel.batch_masks[layer_index].assign(batch_size * layers[layer_index].output_size, Scalar{1});
    }
    parallel.chunk_losses.assign(kLossChunksPerBlock, Scalar{0});
    parallel.loss_raw_batch.resize(kLossChunkSamples * kLossChunksPerBlock, input_size, output_size());
    parallel.loss_normalized_batch.resize(kLossChunkSamples * kLossChunksPerBlock, input_size, output_size());
    return parallel;
}

void TrainingModel::Impl::accumulate_batch_gradients_parallel(const MiniBatch& batch,
                                                              ParallelTraining& parallel,
                                                              std::mt19937_64* rng) {
    require(batch.input_size == input_size, "mini-batch input width mismatch");
    require(batch.output_size == output_size(), "mini-batch output width mismatch");
    const std::size_t count = batch.size();
    if (count == 0) {
        return;
    }
    const std::size_t chunks = gradient_chunk_count(count, parallel.parameter_count);
    require(chunks <= parallel.chunk_gradients.size(), "mini-batch is larger than the parallel training setup");
    require(count * layers.front().output_size <= parallel.batch_masks.front().size(),
            "mini-batch is larger than the parallel training setup");

    // Masks are drawn on this thread, in the serial order, before any worker starts.
    const bool apply_dropout = rng != nullptr;
    if (apply_dropout) {
        draw_dropout_masks(count, *rng, parallel.batch_masks);
    }

    const std::size_t outputs = output_size();
    auto chunk_task = [&](std::size_t chunk, std::size_t worker) {
        TrainingWorkspace& workspace = parallel.workspaces[worker];
        Vector& gradients = parallel.chunk_gradients[chunk];
        std::fill(gradients.begin(), gradients.end(), Scalar{0});
        const std::size_t first = chunk_begin(chunk, chunks, count);
        const std::size_t last = chunk_begin(chunk + 1, chunks, count);
        for (std::size_t begin = first; begin < last; begin += workspace.capacity) {
            const std::size_t block = std::min(workspace.capacity, last - begin);
            if (apply_dropout) {
                for (std::size_t layer_index = 0; layer_index + 1 < layers.size(); ++layer_index) {
                    const std::size_t width = layers[layer_index].output_size;
                    const Scalar* source = parallel.batch_masks[layer_index].data() + begin * width;
                    std::copy(source, source + block * width, workspace.dropout_masks[layer_index].begin());
                }
            }
            accumulate_block_gradients(batch.inputs.data() + begin * input_size,
                                       batch.targets.data() + begin * outputs,
                                       block,
                                       workspace,
                                       apply_dropout,
                                       parallel.chunk_sinks[chunk]);
        }
    };
    parallel.pool->run(chunks, chunk_task);

    // Reduction: every parameter adds the chunk contributions in chunk order 0, 1, 2, ...
    // Slices of parameters are independent, so they can be summed by any worker.
    const std::size_t slices = (parallel.parameter_count + kReductionSlice - 1) / kReductionSlice;
    auto reduce_task = [&](std::size_t slice, std::size_t) {
        const std::size_t first = slice * kReductionSlice;
        const std::size_t last = std::min(parallel.parameter_count, first + kReductionSlice);
        for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            Layer& layer = layers[layer_index];
            const std::size_t layer_first = parallel.layer_offsets[layer_index];
            const std::size_t weight_count = layer.weights.size();
            const std::size_t layer_last = layer_first + weight_count + layer.biases.size();
            const std::size_t begin = std::max(first, layer_first);
            const std::size_t end = std::min(last, layer_last);
            for (std::size_t index = begin; index < end; ++index) {
                const std::size_t local = index - layer_first;
                Scalar& destination = local < weight_count ? layer.grad_weights[local]
                                                           : layer.grad_biases[local - weight_count];
                Scalar value = destination;
                for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
                    value += parallel.chunk_gradients[chunk][index];
                }
                destination = value;
            }
        }
    };
    parallel.pool->run(slices, reduce_task);
}

void TrainingModel::Impl::zero_gradients() {
    for (Layer& layer : layers) {
        std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), Scalar{0});
        std::fill(layer.grad_biases.begin(), layer.grad_biases.end(), Scalar{0});
    }
}

Scalar TrainingModel::Impl::accumulated_gradient_norm(std::size_t batch_size) const {
    const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
    Scalar norm_squared = Scalar{0};
    for (const Layer& layer : layers) {
        for (std::size_t i = 0; i < layer.weights.size(); ++i) {
            Scalar gradient = layer.grad_weights[i] * batch_scale;
            gradient += optimizer.l2 * layer.weights[i];
            gradient += optimizer.l1 * sign_or_zero(layer.weights[i]);
            norm_squared += gradient * gradient;
        }
        for (Scalar gradient_value : layer.grad_biases) {
            const Scalar gradient = gradient_value * batch_scale;
            norm_squared += gradient * gradient;
        }
    }

    return static_cast<Scalar>(std::sqrt(norm_squared));
}

Scalar TrainingModel::Impl::layer_gradient_norm(const Layer& layer, Scalar batch_scale) const {
    Scalar norm_squared = Scalar{0};
    for (std::size_t i = 0; i < layer.weights.size(); ++i) {
        Scalar gradient = layer.grad_weights[i] * batch_scale;
        gradient += optimizer.l2 * layer.weights[i];
        gradient += optimizer.l1 * sign_or_zero(layer.weights[i]);
        norm_squared += gradient * gradient;
    }
    for (Scalar gradient_value : layer.grad_biases) {
        const Scalar gradient = gradient_value * batch_scale;
        norm_squared += gradient * gradient;
    }
    return static_cast<Scalar>(std::sqrt(norm_squared));
}

WeightStatistics TrainingModel::Impl::weight_statistics() const {
    WeightStatistics statistics;
    bool has_finite_weight = false;
    Scalar total = Scalar{0};

    for (const Layer& layer : layers) {
        for (Scalar weight : layer.weights) {
            ++statistics.count;
            if (!is_finite(weight)) {
                statistics.finite = false;
                ++statistics.non_finite_count;
                continue;
            }
            if (!has_finite_weight) {
                statistics.minimum = weight;
                statistics.maximum = weight;
                has_finite_weight = true;
            } else {
                statistics.minimum = std::min(statistics.minimum, weight);
                statistics.maximum = std::max(statistics.maximum, weight);
            }
            total += weight;
        }
    }

    const std::size_t finite_count = statistics.count - statistics.non_finite_count;
    if (finite_count > 0) {
        statistics.mean = total / static_cast<Scalar>(finite_count);
    }
    return statistics;
}

std::size_t TrainingModel::Impl::non_finite_parameter_count() const {
    std::size_t count = 0;
    for (const Layer& layer : layers) {
        for (Scalar weight : layer.weights) {
            if (!is_finite(weight)) {
                ++count;
            }
        }
        for (Scalar bias : layer.biases) {
            if (!is_finite(bias)) {
                ++count;
            }
        }
    }
    return count;
}

std::string TrainingModel::Impl::first_non_finite_parameter_location() const {
    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const Layer& layer = layers[layer_index];
        for (std::size_t i = 0; i < layer.weights.size(); ++i) {
            const Scalar value = layer.weights[i];
            if (!is_finite(value)) {
                return "layer " + std::to_string(layer_index) + " weight[" +
                       std::to_string(i) + "]=" + scalar_debug_string(value) +
                       " (" + non_finite_kind(value) + ")";
            }
        }
        for (std::size_t i = 0; i < layer.biases.size(); ++i) {
            const Scalar value = layer.biases[i];
            if (!is_finite(value)) {
                return "layer " + std::to_string(layer_index) + " bias[" +
                       std::to_string(i) + "]=" + scalar_debug_string(value) +
                       " (" + non_finite_kind(value) + ")";
            }
        }
    }
    return "unknown parameter";
}

std::string TrainingModel::Impl::first_non_finite_gradient_location(std::size_t batch_size) const {
    const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
    for (std::size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        const Layer& layer = layers[layer_index];
        for (std::size_t i = 0; i < layer.grad_weights.size(); ++i) {
            const Scalar raw_gradient = layer.grad_weights[i];
            if (!is_finite(raw_gradient)) {
                return "layer " + std::to_string(layer_index) + " grad_weight[" +
                       std::to_string(i) + "]=" + scalar_debug_string(raw_gradient) +
                       " (" + non_finite_kind(raw_gradient) + ")";
            }
            Scalar gradient = raw_gradient * batch_scale;
            gradient += optimizer.l2 * layer.weights[i];
            gradient += optimizer.l1 * sign_or_zero(layer.weights[i]);
            if (!is_finite(gradient)) {
                return "layer " + std::to_string(layer_index) + " effective grad_weight[" +
                       std::to_string(i) + "]=" + scalar_debug_string(gradient) +
                       " (" + non_finite_kind(gradient) + ")";
            }
        }
        for (std::size_t i = 0; i < layer.grad_biases.size(); ++i) {
            const Scalar raw_gradient = layer.grad_biases[i];
            if (!is_finite(raw_gradient)) {
                return "layer " + std::to_string(layer_index) + " grad_bias[" +
                       std::to_string(i) + "]=" + scalar_debug_string(raw_gradient) +
                       " (" + non_finite_kind(raw_gradient) + ")";
            }
            const Scalar gradient = raw_gradient * batch_scale;
            if (!is_finite(gradient)) {
                return "layer " + std::to_string(layer_index) + " effective grad_bias[" +
                       std::to_string(i) + "]=" + scalar_debug_string(gradient) +
                       " (" + non_finite_kind(gradient) + ")";
            }
        }
    }
    return "unknown gradient";
}

TrainingModel::Impl::GradientApplicationResult TrainingModel::Impl::apply_gradients(std::size_t batch_size,
                                                                                    Scalar learning_rate,
                                                                                    Scalar gradient_noise_stddev,
                                                                                    std::mt19937_64* rng) {
    GradientApplicationResult result;
    const Scalar gradient_norm = accumulated_gradient_norm(batch_size);
    result.gradient_norm = gradient_norm;
    if (!is_finite(gradient_norm)) {
        result.non_finite_location = first_non_finite_gradient_location(batch_size);
        zero_gradients();
        return result;
    }

    ++optimizer_step;
    const Scalar batch_scale = Scalar{1} / static_cast<Scalar>(batch_size);
    result.clipping.batch_count = 1;
    Scalar global_clip_scale = Scalar{1};
    if (optimizer.gradient_clip_norm > Scalar{0}) {
        if (gradient_norm > optimizer.gradient_clip_norm) {
            global_clip_scale =
                optimizer.gradient_clip_norm / (gradient_norm + std::numeric_limits<Scalar>::epsilon());
            result.clipping.global_clip_count = 1;
        }
    }
    result.clipping.minimum_clip_scale = std::min(result.clipping.minimum_clip_scale, global_clip_scale);

    for (Layer& layer : layers) {
        Scalar layer_clip_scale = Scalar{1};
        if (optimizer.layer_gradient_clip_norm > Scalar{0}) {
            const Scalar layer_norm = layer_gradient_norm(layer, batch_scale);
            if (layer_norm > optimizer.layer_gradient_clip_norm) {
                layer_clip_scale =
                    optimizer.layer_gradient_clip_norm / (layer_norm + std::numeric_limits<Scalar>::epsilon());
                ++result.clipping.layer_clip_count;
            }
        }
        ++result.clipping.layer_count;
        const Scalar norm_clip_scale = global_clip_scale * layer_clip_scale;
        result.clipping.minimum_clip_scale = std::min(result.clipping.minimum_clip_scale, norm_clip_scale);
        apply_vector_gradients(layer.weights,
                               layer.grad_weights,
                               layer.velocity_weights,
                               layer.first_moment_weights,
                               layer.second_moment_weights,
                               batch_scale,
                               norm_clip_scale,
                               true,
                               learning_rate,
                               gradient_noise_stddev,
                               rng,
                               result.clipping);
        apply_vector_gradients(layer.biases,
                               layer.grad_biases,
                               layer.velocity_biases,
                               layer.first_moment_biases,
                               layer.second_moment_biases,
                               batch_scale,
                               norm_clip_scale,
                               false,
                               learning_rate,
                               gradient_noise_stddev,
                               rng,
                               result.clipping);
    }

    finalize_gradient_clipping(result.clipping);
    enforce_max_norm();
    ++parameter_version;
    zero_gradients();
    return result;
}

void TrainingModel::Impl::apply_vector_gradients(Vector& values,
                                                 const Vector& accumulated_gradients,
                                                 Vector& velocities,
                                                 Vector& first_moments,
                                                 Vector& second_moments,
                                                 Scalar batch_scale,
                                                 Scalar clip_scale,
                                                 bool regularize,
                                                 Scalar learning_rate,
                                                 Scalar gradient_noise_stddev,
                                                 std::mt19937_64* rng,
                                                 GradientClippingDiagnostics& clipping) {
    // std::normal_distribution requires a strictly positive standard deviation.
    const bool add_noise = gradient_noise_stddev > Scalar{0} && rng != nullptr;
    std::normal_distribution<Scalar> noise(Scalar{0}, add_noise ? gradient_noise_stddev : Scalar{1});

    // Adam bias corrections depend only on the optimizer step, not on the parameter.
    const Scalar bias_correction1 =
        Scalar{1} - static_cast<Scalar>(std::pow(optimizer.beta1, static_cast<Scalar>(optimizer_step)));
    const Scalar bias_correction2 =
        Scalar{1} - static_cast<Scalar>(std::pow(optimizer.beta2, static_cast<Scalar>(optimizer_step)));
    for (std::size_t i = 0; i < values.size(); ++i) {
        Scalar gradient = accumulated_gradients[i] * batch_scale;
        if (regularize) {
            gradient += optimizer.l2 * values[i];
            gradient += optimizer.l1 * sign_or_zero(values[i]);
        }
        if (add_noise) {
            gradient += noise(*rng);
        }
        gradient *= clip_scale;
        ++clipping.gradient_value_count;
        if (optimizer.gradient_clip_value > Scalar{0}) {
            if (gradient > optimizer.gradient_clip_value) {
                gradient = optimizer.gradient_clip_value;
                ++clipping.gradient_value_clip_count;
            } else if (gradient < -optimizer.gradient_clip_value) {
                gradient = -optimizer.gradient_clip_value;
                ++clipping.gradient_value_clip_count;
            }
        }

        switch (optimizer.type) {
        case OptimizerType::sgd:
            if (optimizer.momentum > Scalar{0}) {
                velocities[i] = optimizer.momentum * velocities[i] - learning_rate * gradient;
                values[i] += velocities[i];
            } else {
                values[i] -= learning_rate * gradient;
            }
            break;
        case OptimizerType::adam:
        case OptimizerType::adamw: {
            if (regularize && optimizer.type == OptimizerType::adamw &&
                optimizer.decoupled_weight_decay > Scalar{0}) {
                values[i] *= Scalar{1} - learning_rate * optimizer.decoupled_weight_decay;
            }
            first_moments[i] = optimizer.beta1 * first_moments[i] + (Scalar{1} - optimizer.beta1) * gradient;
            second_moments[i] =
                optimizer.beta2 * second_moments[i] + (Scalar{1} - optimizer.beta2) * gradient * gradient;

            const Scalar m_hat = first_moments[i] / bias_correction1;
            const Scalar v_hat = second_moments[i] / bias_correction2;
            values[i] -= learning_rate * m_hat /
                         (static_cast<Scalar>(std::sqrt(v_hat)) + optimizer.epsilon);
            break;
        }
        }
    }
}

void TrainingModel::Impl::enforce_max_norm() {
    if (optimizer.max_norm <= Scalar{0}) {
        return;
    }
    for (Layer& layer : layers) {
        for (std::size_t out = 0; out < layer.output_size; ++out) {
            const std::size_t offset = out * layer.input_size;
            Scalar norm_squared = Scalar{0};
            for (std::size_t in = 0; in < layer.input_size; ++in) {
                const Scalar weight = layer.weights[offset + in];
                norm_squared += weight * weight;
            }
            const Scalar norm = static_cast<Scalar>(std::sqrt(norm_squared));
            if (norm > optimizer.max_norm && norm > Scalar{0}) {
                const Scalar scale = optimizer.max_norm / norm;
                for (std::size_t in = 0; in < layer.input_size; ++in) {
                    layer.weights[offset + in] *= scale;
                }
            }
        }
    }
}

void TrainingModel::Impl::normalize_sample(const Scalar* raw_input,
                                           std::size_t raw_input_size,
                                           const Scalar* raw_target,
                                           std::size_t raw_target_size,
                                           Scalar* normalized_input,
                                           Scalar* normalized_target) const {
    validate_sample_values(raw_input, raw_input_size, raw_target, raw_target_size, loss.type);
    normalize_buffer_into(raw_input, raw_input_size, normalization.input_features, normalized_input);
    normalize_buffer_into(raw_target, raw_target_size, normalization.output_features, normalized_target);
}

void TrainingModel::Impl::normalize_sample(const Vector& raw_input,
                                           const Vector& raw_target,
                                           Vector& normalized_input,
                                           Vector& normalized_target) const {
    normalized_input.resize(raw_input.size());
    normalized_target.resize(raw_target.size());
    normalize_sample(raw_input.data(),
                     raw_input.size(),
                     raw_target.data(),
                     raw_target.size(),
                     normalized_input.data(),
                     normalized_target.data());
}

void TrainingModel::Impl::read_dataset_sample(const Dataset& dataset,
                                              std::size_t index,
                                              Vector& raw_input,
                                              Vector& raw_target,
                                              Vector& normalized_input,
                                              Vector& normalized_target,
                                              const TrainingDebugOptions* debug,
                                              const TrainingDebugContext& debug_context) const {
    dataset.sample(index, raw_input.data(), raw_input.size(), raw_target.data(), raw_target.size());
    if (debug != nullptr && debug->enabled) {
        debug_check_sample_values(raw_input.data(),
                                  raw_input.size(),
                                  raw_target.data(),
                                  raw_target.size(),
                                  debug_context);
    }
    normalize_sample(raw_input, raw_target, normalized_input, normalized_target);
}

void TrainingModel::Impl::fill_dataset_batch(const Dataset& dataset,
                                             const std::vector<std::size_t>& indices,
                                             std::size_t begin,
                                             std::size_t count,
                                             MiniBatch& raw_batch,
                                             MiniBatch& normalized_batch,
                                             const TrainingDebugOptions* debug,
                                             const TrainingDebugContext& base_debug_context) const {
    require(begin <= indices.size(), "batch begin index is out of range");
    require(count <= indices.size() - begin, "batch count is out of range");
    raw_batch.resize(count, input_size, output_size());
    normalized_batch.resize(count, input_size, output_size());
    fill_dataset_rows(dataset, indices, begin, count, 0, raw_batch, normalized_batch, debug, base_debug_context);
}

void TrainingModel::Impl::fill_dataset_rows(const Dataset& dataset,
                                            const std::vector<std::size_t>& indices,
                                            std::size_t begin,
                                            std::size_t count,
                                            std::size_t row,
                                            MiniBatch& raw_batch,
                                            MiniBatch& normalized_batch,
                                            const TrainingDebugOptions* debug,
                                            const TrainingDebugContext& base_debug_context) const {
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t sample = row + offset;
        dataset.sample(indices[begin + offset],
                       raw_batch.input_data(sample),
                       raw_batch.input_size,
                       raw_batch.target_data(sample),
                       raw_batch.output_size);
        if (debug != nullptr && debug->enabled) {
            TrainingDebugContext context = base_debug_context;
            context.sample_index = indices[begin + offset];
            context.has_sample_index = true;
            debug_check_sample_values(raw_batch.input_data(sample),
                                      raw_batch.input_size,
                                      raw_batch.target_data(sample),
                                      raw_batch.output_size,
                                      context);
        }
        normalize_sample(raw_batch.input_data(sample),
                         raw_batch.input_size,
                         raw_batch.target_data(sample),
                         raw_batch.output_size,
                         normalized_batch.input_data(sample),
                         normalized_batch.target_data(sample));
    }
}

void TrainingModel::Impl::fill_dataset_batch_parallel(const Dataset& dataset,
                                                      const std::vector<std::size_t>& indices,
                                                      std::size_t begin,
                                                      std::size_t count,
                                                      MiniBatch& raw_batch,
                                                      MiniBatch& normalized_batch,
                                                      ParallelTraining& parallel,
                                                      const TrainingDebugOptions* debug,
                                                      const TrainingDebugContext& base_debug_context) const {
    if (!dataset.supports_concurrent_sample()) {
        fill_dataset_batch(dataset, indices, begin, count, raw_batch, normalized_batch, debug, base_debug_context);
        return;
    }
    require(begin <= indices.size(), "batch begin index is out of range");
    require(count <= indices.size() - begin, "batch count is out of range");
    // Resized once on this thread; the workers then write disjoint row ranges.
    raw_batch.resize(count, input_size, output_size());
    normalized_batch.resize(count, input_size, output_size());
    const std::size_t chunks = (count + kLossChunkSamples - 1) / kLossChunkSamples;
    auto fill_task = [&](std::size_t chunk, std::size_t) {
        const std::size_t first = chunk * kLossChunkSamples;
        const std::size_t rows = std::min(kLossChunkSamples, count - first);
        fill_dataset_rows(dataset, indices, begin + first, rows, first, raw_batch, normalized_batch, debug,
                          base_debug_context);
    };
    parallel.pool->run(chunks, fill_task);
}

Scalar TrainingModel::Impl::loss_for_dataset_indices(const Dataset& dataset,
                                                     const std::vector<std::size_t>& indices,
                                                     TrainingWorkspace& workspace,
                                                     const TrainingDebugOptions* debug,
                                                     const char* debug_phase,
                                                     std::size_t epoch,
                                                     bool has_epoch) const {
    require(!indices.empty(), "dataset loss requires at least one sample");
    TrainingDebugContext context;
    context.phase = debug_phase;
    context.epoch = epoch;
    context.has_epoch = has_epoch;
    const std::size_t outputs = output_size();
    Scalar total = Scalar{0};
    for (std::size_t begin = 0; begin < indices.size(); begin += workspace.capacity) {
        const std::size_t block = std::min(workspace.capacity, indices.size() - begin);
        fill_dataset_batch(dataset,
                           indices,
                           begin,
                           block,
                           workspace.raw_batch,
                           workspace.normalized_batch,
                           debug,
                           context);
        forward_batch(workspace.normalized_batch.inputs.data(), block, workspace, false);
        const Scalar* predictions = workspace.activations.back().data();
        for (std::size_t sample = 0; sample < block; ++sample) {
            total += sample_loss(predictions + sample * outputs,
                                 outputs,
                                 workspace.normalized_batch.targets.data() + sample * outputs,
                                 outputs);
        }
    }
    return total / static_cast<Scalar>(indices.size());
}

Scalar TrainingModel::Impl::loss_for_dataset_indices_parallel(const Dataset& dataset,
                                                              const std::vector<std::size_t>& indices,
                                                              ParallelTraining& parallel,
                                                              const TrainingDebugOptions* debug,
                                                              const char* debug_phase,
                                                              std::size_t epoch,
                                                              bool has_epoch) const {
    require(!indices.empty(), "dataset loss requires at least one sample");
    TrainingDebugContext context;
    context.phase = debug_phase;
    context.epoch = epoch;
    context.has_epoch = has_epoch;
    const std::size_t outputs = output_size();
    const std::size_t block_samples = kLossChunkSamples * kLossChunksPerBlock;
    Scalar total = Scalar{0};
    for (std::size_t block_begin = 0; block_begin < indices.size(); block_begin += block_samples) {
        const std::size_t block = std::min(block_samples, indices.size() - block_begin);
        // Samples are read on this thread unless the dataset supports concurrent sample() calls,
        // in which case each task reads its own rows.
        const bool concurrent_reads = dataset.supports_concurrent_sample();
        parallel.loss_raw_batch.resize(block, input_size, outputs);
        parallel.loss_normalized_batch.resize(block, input_size, outputs);
        if (!concurrent_reads) {
            fill_dataset_rows(dataset,
                              indices,
                              block_begin,
                              block,
                              0,
                              parallel.loss_raw_batch,
                              parallel.loss_normalized_batch,
                              debug,
                              context);
        }
        const std::size_t chunks = (block + kLossChunkSamples - 1) / kLossChunkSamples;
        auto loss_task = [&](std::size_t chunk, std::size_t worker) {
            TrainingWorkspace& workspace = parallel.workspaces[worker];
            const std::size_t first = chunk * kLossChunkSamples;
            const std::size_t last = std::min(block, first + kLossChunkSamples);
            if (concurrent_reads) {
                fill_dataset_rows(dataset,
                                  indices,
                                  block_begin + first,
                                  last - first,
                                  first,
                                  parallel.loss_raw_batch,
                                  parallel.loss_normalized_batch,
                                  debug,
                                  context);
            }
            Scalar sum = Scalar{0};
            for (std::size_t begin = first; begin < last; begin += workspace.capacity) {
                const std::size_t count = std::min(workspace.capacity, last - begin);
                forward_batch(parallel.loss_normalized_batch.inputs.data() + begin * input_size, count, workspace, false);
                const Scalar* predictions = workspace.activations.back().data();
                for (std::size_t sample = 0; sample < count; ++sample) {
                    sum += sample_loss(predictions + sample * outputs,
                                       outputs,
                                       parallel.loss_normalized_batch.targets.data() + (begin + sample) * outputs,
                                       outputs);
                }
            }
            parallel.chunk_losses[chunk] = sum;
        };
        parallel.pool->run(chunks, loss_task);
        // Chunk sums are added in chunk order, whichever worker produced them.
        for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
            total += parallel.chunk_losses[chunk];
        }
    }
    return total / static_cast<Scalar>(indices.size());
}

void TrainingModel::Impl::read_streaming_sample(StreamingDataset& dataset,
                                                Vector& raw_input,
                                                Vector& raw_target,
                                                Vector& normalized_input,
                                                Vector& normalized_target,
                                                const TrainingDebugOptions* debug,
                                                const TrainingDebugContext& debug_context) const {
    const bool has_sample =
        dataset.next(raw_input.data(), raw_input.size(), raw_target.data(), raw_target.size());
    require(has_sample, "streaming dataset ended before sample_count");
    if (debug != nullptr && debug->enabled) {
        debug_check_sample_values(raw_input.data(),
                                  raw_input.size(),
                                  raw_target.data(),
                                  raw_target.size(),
                                  debug_context);
    }
    normalize_sample(raw_input, raw_target, normalized_input, normalized_target);
}

std::vector<LayerParameters> TrainingModel::Impl::parameters() const {
    std::vector<LayerParameters> result;
    result.reserve(layers.size());
    for (const Layer& layer : layers) {
        result.push_back(LayerParameters{
            layer.input_size,
            layer.output_size,
            layer.activation,
            layer.weights,
            layer.biases,
            layer.dropout_probability,
        });
    }
    return result;
}

std::vector<LayerTrainingState> TrainingModel::Impl::training_states() const {
    std::vector<LayerTrainingState> result;
    result.reserve(layers.size());
    for (const Layer& layer : layers) {
        result.push_back(LayerTrainingState{
            layer.grad_weights,
            layer.grad_biases,
            layer.velocity_weights,
            layer.velocity_biases,
            layer.first_moment_weights,
            layer.first_moment_biases,
            layer.second_moment_weights,
            layer.second_moment_biases,
        });
    }
    return result;
}

void TrainingModel::Impl::set_parameters(const std::vector<LayerParameters>& parameters) {
    require(parameters.size() == layers.size(), "parameter layer count does not match model topology");
    for (std::size_t i = 0; i < layers.size(); ++i) {
        Layer& layer = layers[i];
        const LayerParameters& source = parameters[i];
        require(source.input_size == layer.input_size, "parameter input_size mismatch");
        require(source.output_size == layer.output_size, "parameter output_size mismatch");
        require(source.activation == layer.activation, "parameter activation mismatch");
        require(source.weights.size() == layer.weights.size(), "parameter weights size mismatch");
        require(source.biases.size() == layer.biases.size(), "parameter biases size mismatch");
        validate_dropout_probability(source.dropout_probability, "parameter dropout_probability");
        if (i + 1 == layers.size()) {
            require(source.dropout_probability == Scalar{0}, "output layer dropout must be zero");
        }
        for (Scalar value : source.weights) {
            require_finite(value, "weight");
        }
        for (Scalar value : source.biases) {
            require_finite(value, "bias");
        }
        layer.weights = source.weights;
        layer.biases = source.biases;
        layer.dropout_probability = source.dropout_probability;
        reset_optimizer_state(layer);
    }
    optimizer_step = 0;
    ++parameter_version;
}

void TrainingModel::Impl::set_training_states(const std::vector<LayerTrainingState>& states, std::uint64_t step) {
    require(states.size() == layers.size(), "checkpoint optimizer state layer count mismatch");
    const std::vector<LayerParameters> current_parameters = parameters();
    for (std::size_t i = 0; i < layers.size(); ++i) {
        validate_layer_training_state(states[i], current_parameters[i]);
        Layer& layer = layers[i];
        layer.grad_weights = states[i].grad_weights;
        layer.grad_biases = states[i].grad_biases;
        layer.velocity_weights = states[i].velocity_weights;
        layer.velocity_biases = states[i].velocity_biases;
        layer.first_moment_weights = states[i].first_moment_weights;
        layer.first_moment_biases = states[i].first_moment_biases;
        layer.second_moment_weights = states[i].second_moment_weights;
        layer.second_moment_biases = states[i].second_moment_biases;
    }
    optimizer_step = step;
}

TrainingModel::Impl::TrainingSnapshot TrainingModel::Impl::snapshot() const {
    return TrainingSnapshot{parameters(), training_states(), optimizer_step};
}

void TrainingModel::Impl::restore_snapshot(const TrainingSnapshot& snapshot) {
    set_parameters(snapshot.parameters);
    set_training_states(snapshot.states, snapshot.optimizer_step);
}

void TrainingModel::Impl::reset_optimizer_state(Layer& layer) {
    std::fill(layer.velocity_weights.begin(), layer.velocity_weights.end(), Scalar{0});
    std::fill(layer.velocity_biases.begin(), layer.velocity_biases.end(), Scalar{0});
    std::fill(layer.first_moment_weights.begin(), layer.first_moment_weights.end(), Scalar{0});
    std::fill(layer.first_moment_biases.begin(), layer.first_moment_biases.end(), Scalar{0});
    std::fill(layer.second_moment_weights.begin(), layer.second_moment_weights.end(), Scalar{0});
    std::fill(layer.second_moment_biases.begin(), layer.second_moment_biases.end(), Scalar{0});
}

} // namespace brutal_mlp

#pragma once

// Internal header: TrainingModel::Impl: forward/backward passes, losses and optimizer updates.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/parallel_workers.hpp"
#include "training/layer_training_state.hpp"
#include "training/training_diagnostics.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace brutal_mlp {

struct TrainingModel::Impl {
    struct Layer {
        std::size_t input_size{0};
        std::size_t output_size{0};
        Activation activation{Activation::linear};
        Scalar dropout_probability{static_cast<Scalar>(0)};
        Vector weights;
        Vector biases;
        Vector grad_weights;
        Vector grad_biases;
        Vector velocity_weights;
        Vector velocity_biases;
        Vector first_moment_weights;
        Vector first_moment_biases;
        Vector second_moment_weights;
        Vector second_moment_biases;
    };

    std::size_t input_size{0};
    LossConfig loss{};
    OptimizerConfig optimizer{OptimizerConfig::adam()};
    NormalizationSpec normalization{};
    std::uint64_t seed{5489u};
    std::uint64_t optimizer_step{0};
    bool has_last_training_options{false};
    TrainingOptions last_training_options{};
    LearningRateSchedulerState learning_rate_state{};
    std::size_t completed_epochs{0};
    bool has_resume_epoch{false};
    std::size_t resume_epoch{0};
    // Incremented whenever weights change, so cached copies (transposed weights) know when to refresh.
    std::uint64_t parameter_version{0};
    std::vector<Layer> layers;

    [[nodiscard]] std::size_t output_size() const;

    [[nodiscard]] TrainingRunConfig make_training_run_config(const OptimizerConfig& run_optimizer,
                                                             const TrainingOptions& runtime_options,
                                                             const DatasetSplit& split,
                                                             std::size_t sample_count,
                                                             bool streaming) const;

    [[nodiscard]] Vector forward(const Vector& input) const;

    [[nodiscard]] Vector predict_raw(const Vector& input) const;

    [[nodiscard]] Vector predict_raw(const Scalar* input, std::size_t input_count) const;

    // Where gradient accumulation adds its results: one weight and one bias buffer per layer,
    // laid out like Layer::grad_weights / Layer::grad_biases.
    struct GradientSink {
        std::vector<Scalar*> weights;
        std::vector<Scalar*> biases;
    };

    // Buffers of the forward and backward pass of a block of up to `capacity` samples.
    // Every per-layer buffer is a row-major matrix with one row per sample. Allocated once per
    // fit by make_workspace() and reused, so training does not allocate per sample or batch.
    struct TrainingWorkspace {
        std::size_t capacity{0};                 // Maximum samples per block.
        std::vector<Vector> activations;         // [0] = inputs, [l + 1] = outputs of layer l.
        std::vector<Vector> pre_activations;     // [l] = z = W * input + b of layer l.
        std::vector<Vector> dropout_masks;       // [l] = 0 or 1 / keep_probability per unit of layer l.
        std::vector<Vector> transposed_weights;  // [l] = W of layer l transposed (input x output).
        std::uint64_t transposed_version{0};     // parameter_version the transposed copies match.
        bool has_transposed_weights{false};
        Vector delta;                            // dLoss/dz of the layer being back-propagated.
        Vector previous_delta;                   // dLoss/dz of the layer before it.
        Vector activation_gradient;              // dLoss/d(output) of one sample, scratch.
        MiniBatch raw_batch;                     // Samples read for loss evaluation.
        MiniBatch normalized_batch;
        GradientSink layer_sink;                 // Points at the model's own gradient buffers.
    };

    // `batch_size` is the largest block the caller will pass; the workspace processes blocks of
    // at most kWorkspaceBlockSamples samples and splits larger ones.
    [[nodiscard]] TrainingWorkspace make_workspace(std::size_t batch_size) const;

    // Draws the dropout masks of `count` samples into rows [0, count) of `masks` (one matrix per
    // layer), sample by sample and layer by layer: the random sequence of one-sample training.
    void draw_dropout_masks(std::size_t count, std::mt19937_64& rng, std::vector<Vector>& masks) const;

    // Forward pass of `count` contiguous input rows (count <= capacity): writes every layer's z
    // and activation into the workspace. With `apply_dropout`, multiplies hidden activations by
    // the masks already stored in workspace.dropout_masks.
    void forward_batch(const Scalar* inputs,
                       std::size_t count,
                       TrainingWorkspace& workspace,
                       bool apply_dropout) const;

    [[nodiscard]] Scalar sample_loss(const Scalar* prediction,
                                     std::size_t prediction_size,
                                     const Scalar* target,
                                     std::size_t target_count) const;

    [[nodiscard]] Scalar sample_loss(const Vector& prediction, const Scalar* target, std::size_t target_count) const;

    [[nodiscard]] Scalar sample_loss(const Vector& prediction, const Vector& target) const;

    // Writes dLoss/dz of the output layer for one sample into `delta`.
    void output_delta_into(const Scalar* prediction,
                           std::size_t prediction_size,
                           const Scalar* target,
                           std::size_t target_count,
                           const Scalar* output_pre_activation,
                           Scalar* delta,
                           Scalar* activation_gradient) const;

    // Adds to `sink` the parameter gradients of `count <= capacity` contiguous samples, in sample
    // order. Only reads the model, so concurrent calls with distinct workspaces and sinks are safe.
    void accumulate_block_gradients(const Scalar* inputs,
                                    const Scalar* targets,
                                    std::size_t count,
                                    TrainingWorkspace& workspace,
                                    bool apply_dropout,
                                    const GradientSink& sink) const;

    // Points `sink` at the model's gradient buffers (Layer::grad_weights / grad_biases).
    void refresh_layer_gradient_sink(GradientSink& sink);

    // Accumulates the parameter gradients of `count` contiguous samples (rows of `inputs` and
    // `targets`), in sample order, on the calling thread.
    void accumulate_batch_gradients(const Scalar* inputs,
                                    const Scalar* targets,
                                    std::size_t count,
                                    TrainingWorkspace& workspace,
                                    std::mt19937_64* rng = nullptr);

    void accumulate_batch_gradients(const MiniBatch& batch,
                                    TrainingWorkspace& workspace,
                                    std::mt19937_64* rng = nullptr);

    // State of deterministic multi-threaded training, built once per fit call when
    // TrainingOptions::parallelism selects worker threads.
    //
    // A mini-batch is split into chunks whose boundaries depend only on the batch size and the
    // model size, never on the thread count. Each chunk accumulates its gradients into its own
    // buffer; the buffers are then summed in chunk order. Any thread may compute any chunk, so the
    // result is the same for every thread count.
    struct ParallelTraining {
        std::unique_ptr<detail::WorkerPool> pool;
        std::vector<TrainingWorkspace> workspaces;  // One per worker.
        std::size_t parameter_count{0};             // Weights + biases of every layer.
        std::vector<std::size_t> layer_offsets;     // Start of layer l in a flat gradient buffer.
        std::vector<Vector> chunk_gradients;        // One flat [W0 b0 W1 b1 ...] buffer per chunk.
        std::vector<GradientSink> chunk_sinks;      // Views of chunk_gradients per layer.
        std::vector<Vector> batch_masks;            // Dropout masks of one whole mini-batch.
        std::vector<Scalar> chunk_losses;           // Per-chunk sums of the loss evaluation.
        MiniBatch loss_raw_batch;                   // Samples of one loss evaluation block.
        MiniBatch loss_normalized_batch;
    };

    // `sample_count` is the dataset size; it bounds the useful number of loss evaluation tasks.
    [[nodiscard]] ParallelTraining make_parallel_training(const TrainingOptions& options, std::size_t sample_count) const;

    // Same contract as fill_dataset_batch(). When the dataset supports concurrent sample() calls,
    // the samples are read by the worker pool; the batch content is identical either way.
    void fill_dataset_batch_parallel(const Dataset& dataset,
                                     const std::vector<std::size_t>& indices,
                                     std::size_t begin,
                                     std::size_t count,
                                     MiniBatch& raw_batch,
                                     MiniBatch& normalized_batch,
                                     ParallelTraining& parallel,
                                     const TrainingDebugOptions* debug,
                                     const detail::TrainingDebugContext& base_debug_context) const;

    // Same contract as accumulate_batch_gradients(MiniBatch&, ...), computed by the worker pool.
    void accumulate_batch_gradients_parallel(const MiniBatch& batch,
                                             ParallelTraining& parallel,
                                             std::mt19937_64* rng);

    void zero_gradients();

    [[nodiscard]] Scalar accumulated_gradient_norm(std::size_t batch_size) const;

    [[nodiscard]] Scalar layer_gradient_norm(const Layer& layer, Scalar batch_scale) const;

    [[nodiscard]] WeightStatistics weight_statistics() const;

    [[nodiscard]] std::size_t non_finite_parameter_count() const;

    [[nodiscard]] std::string first_non_finite_parameter_location() const;

    [[nodiscard]] std::string first_non_finite_gradient_location(std::size_t batch_size) const;

    struct GradientApplicationResult {
        Scalar gradient_norm{static_cast<Scalar>(0)};
        GradientClippingDiagnostics clipping{};
        std::string non_finite_location{};
    };

    [[nodiscard]] GradientApplicationResult apply_gradients(std::size_t batch_size,
                                                            Scalar learning_rate,
                                                            Scalar gradient_noise_stddev = Scalar{0},
                                                            std::mt19937_64* rng = nullptr);

    void apply_vector_gradients(Vector& values,
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
                                GradientClippingDiagnostics& clipping);

    void enforce_max_norm();

    void normalize_sample(const Scalar* raw_input,
                          std::size_t raw_input_size,
                          const Scalar* raw_target,
                          std::size_t raw_target_size,
                          Scalar* normalized_input,
                          Scalar* normalized_target) const;

    void normalize_sample(const Vector& raw_input,
                          const Vector& raw_target,
                          Vector& normalized_input,
                          Vector& normalized_target) const;

    void read_dataset_sample(const Dataset& dataset,
                             std::size_t index,
                             Vector& raw_input,
                             Vector& raw_target,
                             Vector& normalized_input,
                             Vector& normalized_target,
                             const TrainingDebugOptions* debug = nullptr,
                             const detail::TrainingDebugContext& debug_context = {}) const;

    // Reads samples indices[begin .. begin + count) into rows [row, row + count) of batches that
    // are already large enough. Distinct row ranges may be filled concurrently when the dataset
    // supports concurrent sample() calls.
    void fill_dataset_rows(const Dataset& dataset,
                           const std::vector<std::size_t>& indices,
                           std::size_t begin,
                           std::size_t count,
                           std::size_t row,
                           MiniBatch& raw_batch,
                           MiniBatch& normalized_batch,
                           const TrainingDebugOptions* debug,
                           const detail::TrainingDebugContext& base_debug_context) const;

    void fill_dataset_batch(const Dataset& dataset,
                            const std::vector<std::size_t>& indices,
                            std::size_t begin,
                            std::size_t count,
                            MiniBatch& raw_batch,
                            MiniBatch& normalized_batch,
                            const TrainingDebugOptions* debug = nullptr,
                            const detail::TrainingDebugContext& base_debug_context = {}) const;

    [[nodiscard]] Scalar loss_for_dataset_indices(const Dataset& dataset,
                                                  const std::vector<std::size_t>& indices,
                                                  TrainingWorkspace& workspace,
                                                  const TrainingDebugOptions* debug = nullptr,
                                                  const char* debug_phase = "loss",
                                                  std::size_t epoch = 0,
                                                  bool has_epoch = false) const;

    // Same contract as loss_for_dataset_indices(), computed by the worker pool. Samples are read
    // by the workers when the dataset supports concurrent sample() calls, on the calling thread
    // otherwise.
    [[nodiscard]] Scalar loss_for_dataset_indices_parallel(const Dataset& dataset,
                                                           const std::vector<std::size_t>& indices,
                                                           ParallelTraining& parallel,
                                                           const TrainingDebugOptions* debug = nullptr,
                                                           const char* debug_phase = "loss",
                                                           std::size_t epoch = 0,
                                                           bool has_epoch = false) const;

    void read_streaming_sample(StreamingDataset& dataset,
                               Vector& raw_input,
                               Vector& raw_target,
                               Vector& normalized_input,
                               Vector& normalized_target,
                               const TrainingDebugOptions* debug = nullptr,
                               const detail::TrainingDebugContext& debug_context = {}) const;

    [[nodiscard]] std::vector<LayerParameters> parameters() const;

    [[nodiscard]] std::vector<detail::LayerTrainingState> training_states() const;

    void set_parameters(const std::vector<LayerParameters>& parameters);

    void set_training_states(const std::vector<detail::LayerTrainingState>& states, std::uint64_t step);

    struct TrainingSnapshot {
        std::vector<LayerParameters> parameters;
        std::vector<detail::LayerTrainingState> states;
        std::uint64_t optimizer_step{0};

        [[nodiscard]] bool valid() const noexcept {
            return !parameters.empty();
        }
    };

    [[nodiscard]] TrainingSnapshot snapshot() const;

    void restore_snapshot(const TrainingSnapshot& snapshot);

    static void reset_optimizer_state(Layer& layer);
};

} // namespace brutal_mlp

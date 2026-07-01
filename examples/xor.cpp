#include "brutal_mlp/brutal_mlp.hpp"

#include <iostream>

int main() {
    auto optimizer = brutal_mlp::OptimizerConfig::adam(0.08);
    optimizer.gradient_clip_norm = 5.0;

    auto training = brutal_mlp::TrainingModel::builder()
                        .input_size(2)
                        .add_layer(8, brutal_mlp::Activation::tanh)
                        .add_layer(1, brutal_mlp::Activation::sigmoid)
                        .loss(brutal_mlp::Loss::binary_cross_entropy)
                        .optimizer(optimizer)
                        .seed(42)
                        .build();

    const brutal_mlp::Matrix inputs{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    const brutal_mlp::Matrix targets{{0.0}, {1.0}, {1.0}, {0.0}};

    brutal_mlp::TrainingOptions options;
    options.epochs = 1200;
    options.batch_size = 4;
    options.shuffle = true;
    options.seed = 123;

    training.fit(inputs, targets, options);

    const auto inference = training.to_inference_model();
    brutal_mlp::Vector output(inference.output_size(), 0.0);
    brutal_mlp::Vector scratch(inference.scratch_size(), 0.0);

    for (const auto& input : inputs) {
        const auto status = inference.predict_to(input.data(),
                                                 input.size(),
                                                 output.data(),
                                                 output.size(),
                                                 scratch.data(),
                                                 scratch.size());
        if (status != brutal_mlp::InferenceStatus::ok) {
            std::cerr << "inference failed: " << brutal_mlp::to_string(status) << '\n';
            return 1;
        }
        std::cout << input[0] << " xor " << input[1] << " = " << output[0] << '\n';
    }

    return 0;
}

// Weight initialization schemes.

#include "training/weight_initialization.hpp"

#include "core/checks.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace brutal_mlp::detail {

namespace {

struct WeightInitializationParameters {
    bool normal{false};
    Scalar scale{static_cast<Scalar>(0)};
};

[[nodiscard]] WeightInitializationParameters weight_initialization_parameters(
    WeightInitialization initialization,
    std::size_t fan_in,
    std::size_t fan_out) {
    require(fan_in > 0, "fan_in must be positive");
    require(fan_out > 0, "fan_out must be positive");

    const Scalar fan_in_scalar = static_cast<Scalar>(fan_in);
    const Scalar fan_sum = static_cast<Scalar>(fan_in + fan_out);
    switch (initialization) {
    case WeightInitialization::automatic:
        throw std::invalid_argument("automatic initialization must be resolved before use");
    case WeightInitialization::he_normal:
        return WeightInitializationParameters{true,
                                              static_cast<Scalar>(std::sqrt(Scalar{2} / fan_in_scalar))};
    case WeightInitialization::he_uniform:
        return WeightInitializationParameters{false,
                                              static_cast<Scalar>(std::sqrt(Scalar{6} / fan_in_scalar))};
    case WeightInitialization::xavier_normal:
        return WeightInitializationParameters{true,
                                              static_cast<Scalar>(std::sqrt(Scalar{2} / fan_sum))};
    case WeightInitialization::xavier_uniform:
        return WeightInitializationParameters{false,
                                              static_cast<Scalar>(std::sqrt(Scalar{6} / fan_sum))};
    case WeightInitialization::lecun_normal:
        return WeightInitializationParameters{true,
                                              static_cast<Scalar>(std::sqrt(Scalar{1} / fan_in_scalar))};
    case WeightInitialization::lecun_uniform:
        return WeightInitializationParameters{false,
                                              static_cast<Scalar>(std::sqrt(Scalar{3} / fan_in_scalar))};
    }

    throw std::invalid_argument("unknown weight initialization");
}

} // namespace

[[nodiscard]] WeightInitialization resolve_weight_initialization(WeightInitialization initialization,
                                                                 Activation activation) {
    if (initialization != WeightInitialization::automatic) {
        return initialization;
    }
    return activation == Activation::relu ? WeightInitialization::he_uniform
                                          : WeightInitialization::xavier_uniform;
}

void initialize_weights(Vector& weights,
                        WeightInitialization initialization,
                        std::size_t fan_in,
                        std::size_t fan_out,
                        std::mt19937_64& rng) {
    const WeightInitializationParameters parameters =
        weight_initialization_parameters(initialization, fan_in, fan_out);
    if (parameters.normal) {
        std::normal_distribution<Scalar> distribution(Scalar{0}, parameters.scale);
        for (Scalar& weight : weights) {
            weight = distribution(rng);
        }
        return;
    }

    std::uniform_real_distribution<Scalar> distribution(-parameters.scale, parameters.scale);
    for (Scalar& weight : weights) {
        weight = distribution(rng);
    }
}

} // namespace brutal_mlp::detail

// InferenceWorkspace storage.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>

namespace brutal_mlp {

InferenceWorkspace::InferenceWorkspace(const InferenceModel& model) {
    resize_for(model);
}

void InferenceWorkspace::resize_for(const InferenceModel& model) {
    output_.assign(model.output_size(), Scalar{0});
    scratch_.assign(model.scratch_size(), Scalar{0});
}

void InferenceWorkspace::clear() noexcept {
    output_.clear();
    scratch_.clear();
}

std::size_t InferenceWorkspace::output_size() const noexcept {
    return output_.size();
}

std::size_t InferenceWorkspace::scratch_size() const noexcept {
    return scratch_.size();
}

Scalar* InferenceWorkspace::output_data() noexcept {
    return output_.data();
}

const Scalar* InferenceWorkspace::output_data() const noexcept {
    return output_.data();
}

Scalar* InferenceWorkspace::scratch_data() noexcept {
    return scratch_.data();
}

const Scalar* InferenceWorkspace::scratch_data() const noexcept {
    return scratch_.data();
}

const Vector& InferenceWorkspace::output() const noexcept {
    return output_;
}

} // namespace brutal_mlp

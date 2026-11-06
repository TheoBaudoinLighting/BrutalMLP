#pragma once

// Internal header: TrainingRunConfig JSON export and run summary refresh.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstddef>

namespace brutal_mlp::detail {

void refresh_training_run_result(TrainingHistory& history, std::size_t completed_epochs);

} // namespace brutal_mlp::detail

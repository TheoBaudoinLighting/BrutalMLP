#pragma once

// Internal header: Wall-clock helpers used for metadata timestamps.

#include "brutal_mlp/brutal_mlp.hpp"

#include <cstdint>
#include <string>

namespace brutal_mlp::detail {

[[nodiscard]] std::string utc_timestamp(std::uint64_t unix_time);

[[nodiscard]] std::uint64_t current_unix_time() noexcept;

} // namespace brutal_mlp::detail

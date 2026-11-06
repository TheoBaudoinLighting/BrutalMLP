// Wall-clock helpers used for metadata timestamps.

#include "core/time_utils.hpp"

#include <cstdint>
#include <ctime>
#include <sstream>
#include <string>

namespace brutal_mlp::detail {

[[nodiscard]] std::string utc_timestamp(std::uint64_t unix_time) {
    const std::time_t raw_time = static_cast<std::time_t>(unix_time);
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &raw_time) != 0) {
        return {};
    }
#else
    if (gmtime_r(&raw_time, &utc) == nullptr) {
        return {};
    }
#endif

    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

[[nodiscard]] std::uint64_t current_unix_time() noexcept {
    const std::time_t now = std::time(nullptr);
    return now < static_cast<std::time_t>(0) ? 0 : static_cast<std::uint64_t>(now);
}

} // namespace brutal_mlp::detail

// Low-level binary writer/reader, CRC32 and file access.

#include "serialization/binary_io.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace brutal_mlp::detail {

[[nodiscard]] std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = static_cast<std::uint32_t>(-(static_cast<int>(crc & 1u)));
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

[[nodiscard]] std::vector<unsigned char> read_binary_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary model file for reading: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamoff size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to read binary model file size: " + path.string());
    }
    input.seekg(0, std::ios::beg);

    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!input && !bytes.empty()) {
        throw std::runtime_error("failed while reading binary model file: " + path.string());
    }
    return bytes;
}

void write_binary_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open binary model file for writing: " + path.string());
    }
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!output) {
        throw std::runtime_error("failed while writing binary model file: " + path.string());
    }
}

} // namespace brutal_mlp::detail

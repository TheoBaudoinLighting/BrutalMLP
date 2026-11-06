#pragma once

// Internal header: Low-level binary writer/reader, CRC32 and file access.

#include "brutal_mlp/brutal_mlp.hpp"

#include "core/checks.hpp"
#include "serialization/serialization_limits.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace brutal_mlp::detail {

[[nodiscard]] std::uint32_t crc32(const unsigned char* data, std::size_t size) noexcept;

struct BinaryWriter {
    std::vector<unsigned char> bytes;

    void write_u8(std::uint8_t value) {
        bytes.push_back(value);
    }

    void write_bool(bool value) {
        write_u8(value ? 1u : 0u);
    }

    void write_u32(std::uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xFFu));
        }
    }

    void write_u64(std::uint64_t value) {
        for (int shift = 0; shift < 64; shift += 8) {
            bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xFFu));
        }
    }

    void write_size(std::size_t value) {
        write_u64(static_cast<std::uint64_t>(value));
    }

    void write_scalar(Scalar value) {
        require_finite(value, "binary scalar");
#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
        static_assert(sizeof(Scalar) == sizeof(std::uint64_t), "double binary scalar size mismatch");
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        write_u64(bits);
#else
        static_assert(sizeof(Scalar) == sizeof(std::uint32_t), "float binary scalar size mismatch");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        write_u32(bits);
#endif
    }

    void write_string(const std::string& value) {
        write_size(value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }

    void write_scalars(const Vector& values) {
        write_size(values.size());
        for (Scalar value : values) {
            write_scalar(value);
        }
    }
};

struct BinaryReader {
    const std::vector<unsigned char>* bytes{nullptr};
    std::size_t cursor{0};

    explicit BinaryReader(const std::vector<unsigned char>& data)
        : bytes(&data) {}

    void require_available(std::size_t count, const std::string& name) const {
        require(bytes != nullptr, "binary reader is not initialized");
        if (count > bytes->size() || cursor > bytes->size() - count) {
            throw std::runtime_error("truncated binary model while reading " + name);
        }
    }

    [[nodiscard]] std::uint8_t read_u8(const std::string& name) {
        require_available(1, name);
        return (*bytes)[cursor++];
    }

    [[nodiscard]] bool read_bool(const std::string& name) {
        const std::uint8_t value = read_u8(name);
        if (value > 1u) {
            throw std::runtime_error("invalid binary boolean: " + name);
        }
        return value != 0u;
    }

    [[nodiscard]] std::uint32_t read_u32(const std::string& name) {
        require_available(4, name);
        std::uint32_t value = 0;
        for (int shift = 0; shift < 32; shift += 8) {
            value |= static_cast<std::uint32_t>((*bytes)[cursor++]) << shift;
        }
        return value;
    }

    [[nodiscard]] std::uint64_t read_u64(const std::string& name) {
        require_available(8, name);
        std::uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 8) {
            value |= static_cast<std::uint64_t>((*bytes)[cursor++]) << shift;
        }
        return value;
    }

    [[nodiscard]] std::size_t read_size(const std::string& name) {
        return checked_size(read_u64(name), name);
    }

    [[nodiscard]] Scalar read_scalar(const std::string& name) {
        Scalar value{};
#if defined(BRUTAL_MLP_USE_DOUBLE) && BRUTAL_MLP_USE_DOUBLE
        static_assert(sizeof(Scalar) == sizeof(std::uint64_t), "double binary scalar size mismatch");
        const std::uint64_t bits = read_u64(name);
        std::memcpy(&value, &bits, sizeof(value));
#else
        static_assert(sizeof(Scalar) == sizeof(std::uint32_t), "float binary scalar size mismatch");
        const std::uint32_t bits = read_u32(name);
        std::memcpy(&value, &bits, sizeof(value));
#endif
        require_finite(value, name);
        return value;
    }

    [[nodiscard]] std::string read_string(const std::string& name) {
        const std::size_t size = read_size(name + "_size");
        require_available(size, name);
        std::string value;
        value.assign(reinterpret_cast<const char*>(bytes->data() + cursor), size);
        cursor += size;
        return value;
    }

    [[nodiscard]] Vector read_scalars(const std::string& name) {
        const std::size_t count = read_size(name + "_count");
        validate_serialized_scalar_count(count, name);
        Vector values(count);
        for (Scalar& value : values) {
            value = read_scalar(name);
        }
        return values;
    }

    [[nodiscard]] Vector read_scalars_exact(const std::string& name, std::size_t expected_count) {
        const std::size_t count = read_size(name + "_count");
        require_exact_serialized_count(count, expected_count, name);
        Vector values(count);
        for (Scalar& value : values) {
            value = read_scalar(name);
        }
        return values;
    }

    [[nodiscard]] bool at_end() const noexcept {
        return bytes != nullptr && cursor == bytes->size();
    }
};

[[nodiscard]] std::vector<unsigned char> read_binary_file(const std::filesystem::path& path);

void write_binary_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes);

} // namespace brutal_mlp::detail

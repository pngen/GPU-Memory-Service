#pragma once
// Framed, checksummed, versioned TCP protocol for the reference multi-process
// coordinator / CUDA-worker deployment.  Every frame carries enough authority
// (worker boot, allocation generation, epoch) to reject stale traffic.  This
// transport is a reference deployment mechanism, not the core abstraction.

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

namespace gpu_memory_service::mp {

constexpr std::uint32_t kMagic = 0x47534d50u;    // "GSMP"
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxFrame = 1u << 20;    // bounded payload

enum class Msg : std::uint8_t {
  HELLO = 1,            // {role}
  REGISTER = 2,         // {name, worker_boot, pid}
  PUBLISH_CAPACITY = 3, // {free_bytes, total_bytes, device_name}
  ALLOCATE = 4,         // {alloc_id, alloc_gen, bytes, worker_boot}
  ALLOCATE_RESULT = 5,  // {alloc_id, alloc_gen, status, err}
  RELEASE = 6,          // {alloc_id, alloc_gen, worker_boot}
  RELEASE_RESULT = 7,   // {alloc_id, alloc_gen, status, err}
  FENCE_WORKER = 8,     // {worker_boot}
  QUERY = 9,            // {}
  QUERY_RESULT = 10,    // {committed, live, pressure}
  SHUTDOWN = 11         // {}
};

struct Frame {
  Msg type{Msg::HELLO};
  std::vector<std::uint8_t> payload;
};

// Serialize a message with magic/version/length/checksum.  Returns empty on
// internal error (overflow).
[[nodiscard]] std::vector<std::uint8_t> encode(Msg type, const std::vector<std::uint8_t>& payload);

// Decode one frame from a buffer, returning consumed byte count.  Rejects bad
// magic, unsupported version, malformed length, oversized frame, and checksum
// mismatch.  Returns false on failure.
[[nodiscard]] bool decode(const std::vector<std::uint8_t>& buf, std::size_t& consumed, Frame& out);

// Small payload codec helpers (deterministic little-endian).
[[nodiscard]] std::vector<std::uint8_t> pack_u8(std::uint8_t v);
[[nodiscard]] std::vector<std::uint8_t> pack_u32(std::uint32_t v);
[[nodiscard]] std::vector<std::uint8_t> pack_u64(std::uint64_t v);
[[nodiscard]] std::vector<std::uint8_t> pack_i32(std::int32_t v);
[[nodiscard]] std::vector<std::uint8_t> pack_str(const std::string& s);

// Append helpers (for building composite payloads).
void append_u8(std::vector<std::uint8_t>& b, std::uint8_t v);
void append_u32(std::vector<std::uint8_t>& b, std::uint32_t v);
void append_u64(std::vector<std::uint8_t>& b, std::uint64_t v);
void append_i32(std::vector<std::uint8_t>& b, std::int32_t v);
void append_str(std::vector<std::uint8_t>& b, const std::string& s);

// Read helpers with bounds checks.
bool take_u8(const std::vector<std::uint8_t>& b, std::size_t& p, std::uint8_t& v);
bool take_u32(const std::vector<std::uint8_t>& b, std::size_t& p, std::uint32_t& v);
bool take_u64(const std::vector<std::uint8_t>& b, std::size_t& p, std::uint64_t& v);
bool take_i32(const std::vector<std::uint8_t>& b, std::size_t& p, std::int32_t& v);
bool take_str(const std::vector<std::uint8_t>& b, std::size_t& p, std::string& v);

// CRC32 (IEEE, reflected).
[[nodiscard]] std::uint32_t crc32(const std::uint8_t* data, std::size_t len);

}  // namespace gpu_memory_service::mp
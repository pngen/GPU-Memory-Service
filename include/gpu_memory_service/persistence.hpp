#pragma once
// Versioned binary persistence for durable GPU-memory metadata/history.
// The format is deterministic, size-prefixed, checksummed, and validated on
// load against a large set of structural invariants.  Never persist raw CUDA
// pointers as durable authority.

#include <cstdint>
#include <string>
#include <vector>

#include "allocation.hpp"
#include "error.hpp"
#include "memory_domain.hpp"

namespace gpu_memory_service::persist {

constexpr std::uint32_t kMagic = 0x47534d53u;      // "GSMS"
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxRecords = 1u << 20;    // bound hostile counts

// ---------------------------------------------------------------------------
// PersistSnapshot — durable records as a POD-like aggregate.
// ---------------------------------------------------------------------------
struct PersistSnapshot {
  std::uint32_t magic{kMagic};
  std::uint32_t version{kVersion};
  MemoryDomain domain;
  std::vector<AllocationRecord> records;
  std::vector<AllocationRecord> retired;
  std::uint64_t next_alloc_id{1};
  std::uint64_t history_count{0};
};

// Deterministic encode into a byte buffer.  Always produces the same bytes for
// the same snapshot (no pointer values, no hash-order iteration).
[[nodiscard]] std::vector<std::uint8_t> encode(const PersistSnapshot& s);

// Decode with full structural validation.  Rejects bad magic, unsupported
// version, truncation, checksum mismatch, hostile counts, invalid enums,
// duplicate ids/generations, overlapping regions, generation regression,
// impossible lifecycle, physical bytes > governed capacity, invalid alignment,
// invalid region interval, NaN/Inf policy, and trailing garbage.
[[nodiscard]] Result<PersistSnapshot> decode(const std::vector<std::uint8_t>& bytes);

// File helpers (deterministic binary I/O).
[[nodiscard]] Result<void> write_file(const std::string& path, const PersistSnapshot& s);
[[nodiscard]] Result<PersistSnapshot> read_file(const std::string& path);

}  // namespace gpu_memory_service::persist

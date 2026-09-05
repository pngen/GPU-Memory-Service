#pragma once
// Memory-domain model: binds physical device identity, generation, capacity
// evidence with provenance, and the watermark/pressure policy for one governed
// device-memory domain.  Provenance prevents presenting derived allocator
// bookkeeping as physical driver telemetry.

#include <cmath>
#include <string>

#include "enum.hpp"
#include "types.hpp"

namespace gpu_memory_service {

// Forward declarations for summary types (defined in governed_memory.hpp).
struct MemorySummary;

// ---------------------------------------------------------------------------
// WatermarkConfig — generation-bound high/low/critical watermarks (fractions
// of governed capacity) with hysteresis.  Validated to avoid oscillatory
// pressure transitions.
// ---------------------------------------------------------------------------
struct WatermarkConfig {
  double low{0.70};
  double high{0.90};
  double critical{0.97};
  WatermarkGeneration generation{WatermarkGeneration::first()};

  [[nodiscard]] bool valid() const noexcept {
    return !std::isnan(low) && !std::isnan(high) && !std::isnan(critical) &&
           !std::isinf(low) && !std::isinf(high) && !std::isinf(critical) &&
           low >= 0.0 && low <= 1.0 &&
           high >= low && high <= 1.0 &&
           critical >= high && critical <= 1.0;
  }
};

// ---------------------------------------------------------------------------
// MemoryDomain — authoritative governing state for one device-memory domain.
// ---------------------------------------------------------------------------
struct MemoryDomain {
  GpuMemoryDomainId domain_id;
  GpuMemoryDomainGeneration domain_generation{GpuMemoryDomainGeneration::first()};
  DeviceId device_id;
  DeviceGeneration device_generation{DeviceGeneration::first()};

  std::uint64_t total_capacity{0};         // physical device capacity evidence
  std::uint64_t allocator_visible_capacity{0};  // capacity the arena exposes
  std::uint64_t reserved_headroom{0};      // system/reserved headroom
  std::uint64_t managed_capacity{0};       // capacity this domain governs

  std::uint64_t current_committed{0};      // bytes committed to allocations
  std::uint64_t current_physically_allocated{0};
  std::uint64_t current_resident{0};
  std::uint64_t current_reclaimable{0};
  std::uint64_t current_movable{0};
  std::uint64_t current_pinned{0};

  double external_fragmentation{0.0};
  PressureState pressure{PressureState::UNKNOWN};
  DeviceHealth health{DeviceHealth::UNKNOWN};
  WatermarkConfig watermarks;
  Provenance provenance{Provenance::UNKNOWN};
};

}  // namespace gpu_memory_service

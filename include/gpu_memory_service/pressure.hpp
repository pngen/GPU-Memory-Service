#pragma once
// Pressure model.  Pressure is based on evidence (governed utilization plus
// reclaimable/contiguous evidence) and uses generation-bound watermarks with
// hysteresis.  UNKNOWN must never silently become NORMAL; a caller that has no
// evidence must keep input as UNKNOWN and get UNKNOWN back.

#include "enum.hpp"
#include "memory_domain.hpp"

namespace gpu_memory_service {

// Deterministic pressure classification with hysteresis: pressure rises at the
// high watermark but does not fall back to NORMAL/LOW until utilization drops
// below the low watermark, preventing oscillation around a single threshold.
[[nodiscard]] PressureState classify_pressure(double utilization,
                                               const WatermarkConfig& wm,
                                               PressureState last) noexcept;

// Evidence bundle used by the runtime to derive a pressure state.
struct PressureEvidence {
  std::uint64_t managed_capacity{0};
  std::uint64_t committed{0};
  std::uint64_t free{0};
  std::uint64_t reclaimable{0};
  std::uint64_t largest_free{0};
  double fragmentation{0.0};
  WatermarkConfig watermarks;
};

[[nodiscard]] PressureState evaluate_pressure(const PressureEvidence& ev,
                                               PressureState last) noexcept;

}  // namespace gpu_memory_service

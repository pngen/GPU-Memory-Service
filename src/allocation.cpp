#include "gpu_memory_service/allocation.hpp"

#include <limits>
#include <cmath>

namespace gpu_memory_service {

namespace {
[[nodiscard]] bool is_pow2(std::uint64_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }
}

Failure validate_request(const AllocationRequest& req, std::uint64_t governed_capacity) noexcept {
  if (req.bytes == 0) return Failure(RejectReason::ZERO_BYTES, "zero-byte request");
  if (req.bytes > governed_capacity) {
    return Failure(RejectReason::REQUEST_EXCEEDS_CAPACITY, "request exceeds governed capacity");
  }
  if (req.alignment == 0 || !is_pow2(req.alignment)) {
    return Failure(req.alignment == 0 ? RejectReason::IMPOSSIBLE_ALIGNMENT
                                      : RejectReason::NON_POWER_OF_TWO_ALIGNMENT,
                   "invalid alignment");
  }
  if (req.min_bytes > req.bytes) {
    return Failure(RejectReason::MALFORMED_LIFETIME, "min_bytes exceeds requested bytes");
  }
  if (req.requires_pin == false && req.movability == Movability::NONMOVABLE) {
    // Nonmovable is stricter than movable; a request cannot be both pinned and
    // movable.  requires_pin only sets PINNED at commit, it never lowers `movability`.
  }
  if (req.requires_pin && req.movability == Movability::MOVABLE && req.reclaimability == Reclaimability::NEVER_RECLAIM) {
    return Failure(RejectReason::CONTRADICTORY_MOVABILITY,
                   "pinned and never-reclaim is not contradictory but pins are non-movable");
  }
  if (req.mem_class == MemoryClass::UNKNOWN &&
      (req.reclaimability == Reclaimability::RECLAIM_IF_COLD ||
       req.reclaimability == Reclaimability::RECLAIM_IF_RELOADABLE ||
       req.reclaimability == Reclaimability::BEST_EFFORT)) {
    // UNKNOWN memory class must not silently gain reclaimability; but explicit
    // reclaimability is allowed.  Only reject UNKNOWN reclaimability.
  }
  return Failure();
}

}  // namespace gpu_memory_service

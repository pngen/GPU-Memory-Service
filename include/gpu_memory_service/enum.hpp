#pragma once
// Flat, strongly-typed enums describing GPU-memory state.  These categories
// drive policy but are not authority: a memory class alone never grants
// reclaimability; a provenance class never upgrades a derived quantity into
// measured physical telemetry.

#include <cstdint>
#include <ostream>

namespace gpu_memory_service {

// ---------------------------------------------------------------------------
// Provenance — how a quantity was derived.  Never present derived allocator
// bookkeeping as physical driver telemetry.
// ---------------------------------------------------------------------------
enum class Provenance : std::uint8_t {
  UNKNOWN = 0,
  MEASURED,        // measured by this process at runtime
  REPORTED,        // reported by a worker/driver
  DERIVED,         // derived from allocator bookkeeping
  ESTIMATED,       // estimated
  SYNTHETIC,       // synthetic (test/seed) — never mistaken for real telemetry
  RECONSTRUCTED    // reconstructed after recovery
};
inline constexpr const char* to_string(Provenance p) {
  switch (p) {
    case Provenance::UNKNOWN: return "UNKNOWN";
    case Provenance::MEASURED: return "MEASURED";
    case Provenance::REPORTED: return "REPORTED";
    case Provenance::DERIVED: return "DERIVED";
    case Provenance::ESTIMATED: return "ESTIMATED";
    case Provenance::SYNTHETIC: return "SYNTHETIC";
    case Provenance::RECONSTRUCTED: return "RECONSTRUCTED";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, Provenance p) { return os << to_string(p); }

// ---------------------------------------------------------------------------
// MemoryClass — explicit purpose classification.  Informs policy, not authority.
// ---------------------------------------------------------------------------
enum class MemoryClass : std::uint8_t {
  UNKNOWN = 0,
  MODEL_WEIGHTS,
  ADAPTER_WEIGHTS,
  KV_CACHE,
  ACTIVATION,
  TEMP_WORKSPACE,
  COMMUNICATION_BUFFER,
  COLLECTIVE_BUFFER,
  CHECKPOINT_STAGING,
  STATE_CACHE,
  COMPILATION_ARTIFACT,
  USER_BUFFER,
  INTERNAL
};
inline constexpr const char* to_string(MemoryClass c) {
  switch (c) {
    case MemoryClass::UNKNOWN: return "UNKNOWN";
    case MemoryClass::MODEL_WEIGHTS: return "MODEL_WEIGHTS";
    case MemoryClass::ADAPTER_WEIGHTS: return "ADAPTER_WEIGHTS";
    case MemoryClass::KV_CACHE: return "KV_CACHE";
    case MemoryClass::ACTIVATION: return "ACTIVATION";
    case MemoryClass::TEMP_WORKSPACE: return "TEMP_WORKSPACE";
    case MemoryClass::COMMUNICATION_BUFFER: return "COMMUNICATION_BUFFER";
    case MemoryClass::COLLECTIVE_BUFFER: return "COLLECTIVE_BUFFER";
    case MemoryClass::CHECKPOINT_STAGING: return "CHECKPOINT_STAGING";
    case MemoryClass::STATE_CACHE: return "STATE_CACHE";
    case MemoryClass::COMPILATION_ARTIFACT: return "COMPILATION_ARTIFACT";
    case MemoryClass::USER_BUFFER: return "USER_BUFFER";
    case MemoryClass::INTERNAL: return "INTERNAL";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, MemoryClass c) { return os << to_string(c); }

// ---------------------------------------------------------------------------
// ReclaimabilityClass — explicit and conservative.  UNKNOWN is never reclaimed.
// ---------------------------------------------------------------------------
enum class Reclaimability : std::uint8_t {
  UNKNOWN = 0,
  NEVER_RECLAIM,
  RECLAIM_AFTER_COMPLETION,
  RECLAIM_IF_COLD,
  RECLAIM_IF_RELOADABLE,
  RECLAIM_IF_CHECKPOINTED,
  BEST_EFFORT
};
inline constexpr const char* to_string(Reclaimability r) {
  switch (r) {
    case Reclaimability::UNKNOWN: return "UNKNOWN";
    case Reclaimability::NEVER_RECLAIM: return "NEVER_RECLAIM";
    case Reclaimability::RECLAIM_AFTER_COMPLETION: return "RECLAIM_AFTER_COMPLETION";
    case Reclaimability::RECLAIM_IF_COLD: return "RECLAIM_IF_COLD";
    case Reclaimability::RECLAIM_IF_RELOADABLE: return "RECLAIM_IF_RELOADABLE";
    case Reclaimability::RECLAIM_IF_CHECKPOINTED: return "RECLAIM_IF_CHECKPOINTED";
    case Reclaimability::BEST_EFFORT: return "BEST_EFFORT";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, Reclaimability r) { return os << to_string(r); }

// ---------------------------------------------------------------------------
// Allocation lifecycle.
// ---------------------------------------------------------------------------
enum class LifecycleState : std::uint8_t {
  REQUESTED = 0,
  VALIDATING,
  RESERVED,
  ALLOCATING,
  ALLOCATED,
  RESIDENT,
  PINNED,
  RECLAIMABLE,
  EVICTION_PENDING,
  EVICTING,
  RELOCATION_PENDING,
  RELOCATING,
  RELEASE_PENDING,
  RELEASED,
  FAILED,
  STALE,
  REVALIDATION_REQUIRED,
  SUPERSEDED,
  RETIRED
};
inline constexpr const char* to_string(LifecycleState s) {
  switch (s) {
    case LifecycleState::REQUESTED: return "REQUESTED";
    case LifecycleState::VALIDATING: return "VALIDATING";
    case LifecycleState::RESERVED: return "RESERVED";
    case LifecycleState::ALLOCATING: return "ALLOCATING";
    case LifecycleState::ALLOCATED: return "ALLOCATED";
    case LifecycleState::RESIDENT: return "RESIDENT";
    case LifecycleState::PINNED: return "PINNED";
    case LifecycleState::RECLAIMABLE: return "RECLAIMABLE";
    case LifecycleState::EVICTION_PENDING: return "EVICTION_PENDING";
    case LifecycleState::EVICTING: return "EVICTING";
    case LifecycleState::RELOCATION_PENDING: return "RELOCATION_PENDING";
    case LifecycleState::RELOCATING: return "RELOCATING";
    case LifecycleState::RELEASE_PENDING: return "RELEASE_PENDING";
    case LifecycleState::RELEASED: return "RELEASED";
    case LifecycleState::FAILED: return "FAILED";
    case LifecycleState::STALE: return "STALE";
    case LifecycleState::REVALIDATION_REQUIRED: return "REVALIDATION_REQUIRED";
    case LifecycleState::SUPERSEDED: return "SUPERSEDED";
    case LifecycleState::RETIRED: return "RETIRED";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, LifecycleState s) { return os << to_string(s); }

// ---------------------------------------------------------------------------
// Residency — physical residency of allocation bytes.
// ---------------------------------------------------------------------------
enum class ResidencyState : std::uint8_t {
  UNKNOWN = 0,
  NONRESIDENT,
  STAGING,
  RESIDENT,
  ACTIVE,
  COLD,
  EVICTABLE,
  EVICTING,
  RELOADING,
  STALE,
  REVALIDATION_REQUIRED
};
inline constexpr const char* to_string(ResidencyState s) {
  switch (s) {
    case ResidencyState::UNKNOWN: return "UNKNOWN";
    case ResidencyState::NONRESIDENT: return "NONRESIDENT";
    case ResidencyState::STAGING: return "STAGING";
    case ResidencyState::RESIDENT: return "RESIDENT";
    case ResidencyState::ACTIVE: return "ACTIVE";
    case ResidencyState::COLD: return "COLD";
    case ResidencyState::EVICTABLE: return "EVICTABLE";
    case ResidencyState::EVICTING: return "EVICTING";
    case ResidencyState::RELOADING: return "RELOADING";
    case ResidencyState::STALE: return "STALE";
    case ResidencyState::REVALIDATION_REQUIRED: return "REVALIDATION_REQUIRED";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, ResidencyState s) { return os << to_string(s); }

// ---------------------------------------------------------------------------
// Pressure — based on evidence.  UNKNOWN must never silently become NORMAL.
// ---------------------------------------------------------------------------
enum class PressureState : std::uint8_t {
  UNKNOWN = 0,
  LOW,
  NORMAL,
  ELEVATED,
  HIGH,
  CRITICAL,
  RECLAIMING,
  RECOVERING,
  STALE,
  REVALIDATION_REQUIRED
};
inline constexpr const char* to_string(PressureState p) {
  switch (p) {
    case PressureState::UNKNOWN: return "UNKNOWN";
    case PressureState::LOW: return "LOW";
    case PressureState::NORMAL: return "NORMAL";
    case PressureState::ELEVATED: return "ELEVATED";
    case PressureState::HIGH: return "HIGH";
    case PressureState::CRITICAL: return "CRITICAL";
    case PressureState::RECLAIMING: return "RECLAIMING";
    case PressureState::RECOVERING: return "RECOVERING";
    case PressureState::STALE: return "STALE";
    case PressureState::REVALIDATION_REQUIRED: return "REVALIDATION_REQUIRED";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, PressureState p) { return os << to_string(p); }

// ---------------------------------------------------------------------------
// Device health and capability.
// ---------------------------------------------------------------------------
enum class DeviceHealth : std::uint8_t {
  UNKNOWN = 0,
  HEALTHY,
  DEGRADED,
  UNRESPONSIVE,
  LOST
};
inline constexpr const char* to_string(DeviceHealth h) {
  switch (h) {
    case DeviceHealth::UNKNOWN: return "UNKNOWN";
    case DeviceHealth::HEALTHY: return "HEALTHY";
    case DeviceHealth::DEGRADED: return "DEGRADED";
    case DeviceHealth::UNRESPONSIVE: return "UNRESPONSIVE";
    case DeviceHealth::LOST: return "LOST";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, DeviceHealth h) { return os << to_string(h); }

// ---------------------------------------------------------------------------
// Mobile-ness / pinned classification.
// ---------------------------------------------------------------------------
enum class Movability : std::uint8_t {
  MOVABLE = 0,
  PINNED,
  NONMOVABLE
};
inline constexpr const char* to_string(Movability m) {
  switch (m) {
    case Movability::MOVABLE: return "MOVABLE";
    case Movability::PINNED: return "PINNED";
    case Movability::NONMOVABLE: return "NONMOVABLE";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, Movability m) { return os << to_string(m); }

// ---------------------------------------------------------------------------
// Typed reason for an allocation rejection / contiguity failure / OOM.
// ---------------------------------------------------------------------------
enum class RejectReason : std::uint8_t {
  NONE = 0,
  INVALID_REQUEST,
  ZERO_BYTES,
  SIZE_OVERFLOW,
  IMPOSSIBLE_ALIGNMENT,
  NON_POWER_OF_TWO_ALIGNMENT,
  REQUEST_EXCEEDS_CAPACITY,
  INVALID_OWNER,
  STALE_RESERVATION,
  STALE_RESOURCE_CLAIM,
  MALFORMED_LIFETIME,
  CONTRADICTORY_MOVABILITY,
  UNSUPPORTED_MEMORY_CLASS,
  INSUFFICIENT_TOTAL_CAPACITY,
  INSUFFICIENT_CONTIGUOUS_CAPACITY,
  BLOCKED_BY_PINNED,
  BLOCKED_BY_NONMOVABLE,
  RESERVATION_CONFLICT,
  REVALIDATION_REQUIRED,
  PHYSICAL_OOM,
  GOVERNED_CAPACITY_EXCEEDED,
  CONTIGUOUS_CAPACITY_UNAVAILABLE,
  FRAGMENTATION,
  PINNED_BLOCKERS,
  RESERVATION_PROTECTED,
  RECLAIM_INSUFFICIENT,
  EVICTION_NOT_SAFE,
  DEVICE_UNAVAILABLE,
  STALE_AUTHORITY,
  STALE_GENERATION,
  WRONG_OWNER,
  CROSS_DEVICE,
  DOUBLE_FREE,
  FREE_OF_ACTIVE_OR_PINNED,
  ILLEGAL_TRANSITION,
  INTERNAL_ERROR,
  UNKNOWN
};
inline constexpr const char* to_string(RejectReason r) {
  switch (r) {
    case RejectReason::NONE: return "NONE";
    case RejectReason::INVALID_REQUEST: return "INVALID_REQUEST";
    case RejectReason::ZERO_BYTES: return "ZERO_BYTES";
    case RejectReason::SIZE_OVERFLOW: return "SIZE_OVERFLOW";
    case RejectReason::IMPOSSIBLE_ALIGNMENT: return "IMPOSSIBLE_ALIGNMENT";
    case RejectReason::NON_POWER_OF_TWO_ALIGNMENT: return "NON_POWER_OF_TWO_ALIGNMENT";
    case RejectReason::REQUEST_EXCEEDS_CAPACITY: return "REQUEST_EXCEEDS_CAPACITY";
    case RejectReason::INVALID_OWNER: return "INVALID_OWNER";
    case RejectReason::STALE_RESERVATION: return "STALE_RESERVATION";
    case RejectReason::STALE_RESOURCE_CLAIM: return "STALE_RESOURCE_CLAIM";
    case RejectReason::MALFORMED_LIFETIME: return "MALFORMED_LIFETIME";
    case RejectReason::CONTRADICTORY_MOVABILITY: return "CONTRADICTORY_MOVABILITY";
    case RejectReason::UNSUPPORTED_MEMORY_CLASS: return "UNSUPPORTED_MEMORY_CLASS";
    case RejectReason::INSUFFICIENT_TOTAL_CAPACITY: return "INSUFFICIENT_TOTAL_CAPACITY";
    case RejectReason::INSUFFICIENT_CONTIGUOUS_CAPACITY: return "INSUFFICIENT_CONTIGUOUS_CAPACITY";
    case RejectReason::BLOCKED_BY_PINNED: return "BLOCKED_BY_PINNED";
    case RejectReason::BLOCKED_BY_NONMOVABLE: return "BLOCKED_BY_NONMOVABLE";
    case RejectReason::RESERVATION_CONFLICT: return "RESERVATION_CONFLICT";
    case RejectReason::REVALIDATION_REQUIRED: return "REVALIDATION_REQUIRED";
    case RejectReason::PHYSICAL_OOM: return "PHYSICAL_OOM";
    case RejectReason::GOVERNED_CAPACITY_EXCEEDED: return "GOVERNED_CAPACITY_EXCEEDED";
    case RejectReason::CONTIGUOUS_CAPACITY_UNAVAILABLE: return "CONTIGUOUS_CAPACITY_UNAVAILABLE";
    case RejectReason::FRAGMENTATION: return "FRAGMENTATION";
    case RejectReason::PINNED_BLOCKERS: return "PINNED_BLOCKERS";
    case RejectReason::RESERVATION_PROTECTED: return "RESERVATION_PROTECTED";
    case RejectReason::RECLAIM_INSUFFICIENT: return "RECLAIM_INSUFFICIENT";
    case RejectReason::EVICTION_NOT_SAFE: return "EVICTION_NOT_SAFE";
    case RejectReason::DEVICE_UNAVAILABLE: return "DEVICE_UNAVAILABLE";
    case RejectReason::STALE_AUTHORITY: return "STALE_AUTHORITY";
    case RejectReason::STALE_GENERATION: return "STALE_GENERATION";
    case RejectReason::WRONG_OWNER: return "WRONG_OWNER";
    case RejectReason::CROSS_DEVICE: return "CROSS_DEVICE";
    case RejectReason::DOUBLE_FREE: return "DOUBLE_FREE";
    case RejectReason::FREE_OF_ACTIVE_OR_PINNED: return "FREE_OF_ACTIVE_OR_PINNED";
    case RejectReason::ILLEGAL_TRANSITION: return "ILLEGAL_TRANSITION";
    case RejectReason::INTERNAL_ERROR: return "INTERNAL_ERROR";
    case RejectReason::UNKNOWN: return "UNKNOWN";
  }
  return "UNKNOWN";
}
inline std::ostream& operator<<(std::ostream& os, RejectReason r) { return os << to_string(r); }

}  // namespace gpu_memory_service
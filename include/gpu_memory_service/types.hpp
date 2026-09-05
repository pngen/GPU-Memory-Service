#pragma once
// Strongly typed identities and generations for GPU Memory Service.
// These prevent the accidental collapse of semantically distinct quantities
// into a single generic integer, and are the backbone of the stale-authority
// rejection doctrine: a stale AllocationGeneration must never free current
// memory, a stale WorkerBootId must never publish an allocation completion,
// and a stale DeviceGeneration must never satisfy current capacity.
#include <cstdint>
#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <utility>

namespace gpu_memory_service {

// ---------------------------------------------------------------------------
// Identity tags — one distinct static tag type per identity domain.  The
// TypedId/TypedGeneration templates are distinct types for distinct tags, so
// the compiler rejects accidental cross-domain assignment at compile time.
// ---------------------------------------------------------------------------
#define GMS_ID_TAG(name) struct name##Tag { };
#define GMS_GEN_TAG(name) struct name##GenTag { };

GMS_ID_TAG(GpuMemoryDomainId)
GMS_ID_TAG(DeviceId)
GMS_ID_TAG(MemoryRegionId)
GMS_ID_TAG(AllocationId)
GMS_ID_TAG(AllocationRequestId)
GMS_ID_TAG(AllocationLeaseId)
GMS_ID_TAG(ResidencyId)
GMS_ID_TAG(ReclaimPlanId)
GMS_ID_TAG(EvictionPlanId)
GMS_ID_TAG(CompactionPlanId)
GMS_ID_TAG(RelocationId)
GMS_ID_TAG(StagingPlanId)
GMS_ID_TAG(WorkloadId)
GMS_ID_TAG(ExecutionId)
GMS_ID_TAG(WorkerId)
GMS_ID_TAG(SourceId)

GMS_GEN_TAG(GpuMemoryDomainGeneration)
GMS_GEN_TAG(DeviceGeneration)
GMS_GEN_TAG(MemoryRegionGeneration)
GMS_GEN_TAG(AllocationGeneration)
GMS_GEN_TAG(AllocationRequestGeneration)
GMS_GEN_TAG(AllocationLeaseGeneration)
GMS_GEN_TAG(ResidencyGeneration)
GMS_GEN_TAG(ReclaimPlanGeneration)
GMS_GEN_TAG(EvictionPlanGeneration)
GMS_GEN_TAG(CompactionPlanGeneration)
GMS_GEN_TAG(RelocationGeneration)
GMS_GEN_TAG(StagingPlanGeneration)
GMS_GEN_TAG(PressureGeneration)
GMS_GEN_TAG(WatermarkGeneration)
GMS_GEN_TAG(CapacityGeneration)
GMS_GEN_TAG(ReservationGeneration)
GMS_GEN_TAG(ResourceClaimGeneration)
GMS_GEN_TAG(PlacementGeneration)
GMS_GEN_TAG(WorkloadGeneration)
GMS_GEN_TAG(ExecutionGeneration)
GMS_GEN_TAG(WorkerBootId)
GMS_GEN_TAG(SourceBootId)
GMS_GEN_TAG(CoordinatorEpoch)
GMS_GEN_TAG(AuthorityGeneration)
GMS_GEN_TAG(RecoveryGeneration)
GMS_GEN_TAG(RevalidationGeneration)
GMS_GEN_TAG(CheckpointGeneration)
GMS_GEN_TAG(ModelResidencyGeneration)
GMS_GEN_TAG(AdapterGeneration)
GMS_GEN_TAG(StateGeneration)
GMS_GEN_TAG(PolicyGeneration)

#undef GMS_ID_TAG
#undef GMS_GEN_TAG

// ---------------------------------------------------------------------------
// TypedId — a distinct identity value.  Default-constructed ids are empty
// (value 0) and are not valid for use as an authoritative identity.
// ---------------------------------------------------------------------------
template <typename Tag, typename Value = std::uint64_t>
class TypedId {
 public:
  using value_type = Value;

  constexpr TypedId() noexcept = default;
  constexpr explicit TypedId(Value v) noexcept : value_(v) {}
  static constexpr TypedId invalid() noexcept { return TypedId(); }

  [[nodiscard]] constexpr Value value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool is_valid() const noexcept { return value_ != Value{0}; }
  constexpr explicit operator bool() const noexcept { return is_valid(); }

  constexpr bool operator==(const TypedId&) const noexcept = default;
  constexpr bool operator<(const TypedId& o) const noexcept { return value_ < o.value_; }

 private:
  Value value_{0};
};

template <typename Tag, typename Value>
std::ostream& operator<<(std::ostream& os, const TypedId<Tag, Value>& id) {
  return os << id.value();
}

// ---------------------------------------------------------------------------
// TypedGeneration — a monotonic generation.  Generation 0 is "no generation"
// and is not authoritative.  Generations only ever advance; a stale generation
// must never be treated as current.
// ---------------------------------------------------------------------------
template <typename Tag, typename Value = std::uint64_t>
class TypedGeneration {
 public:
  using value_type = Value;

  constexpr TypedGeneration() noexcept = default;
  constexpr explicit TypedGeneration(Value v) noexcept : value_(v) {}
  static constexpr TypedGeneration none() noexcept { return TypedGeneration(); }
  static constexpr TypedGeneration first() noexcept { return TypedGeneration(Value{1}); }

  [[nodiscard]] constexpr Value value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool is_set() const noexcept { return value_ != Value{0}; }
  [[nodiscard]] constexpr bool is_first() const noexcept { return value_ == Value{1}; }
  constexpr explicit operator bool() const noexcept { return is_set(); }

  // Returns the next generation; guards against overflow by returning none.
  [[nodiscard]] constexpr TypedGeneration next() const noexcept {
    if (value_ == static_cast<Value>(-1)) return TypedGeneration();
    return TypedGeneration(static_cast<Value>(value_ + 1));
  }
  [[nodiscard]] constexpr bool is_current(const TypedGeneration& o) const noexcept {
    return is_set() && value_ == o.value_;
  }

  constexpr bool operator==(const TypedGeneration&) const noexcept = default;
  constexpr bool operator<(const TypedGeneration& o) const noexcept { return value_ < o.value_; }
  constexpr bool operator<=(const TypedGeneration& o) const noexcept { return value_ <= o.value_; }

 private:
  Value value_{0};
};

template <typename Tag, typename Value>
std::ostream& operator<<(std::ostream& os, const TypedGeneration<Tag, Value>& g) {
  return os << g.value();
}

// ---------------------------------------------------------------------------
// Concrete identity / generation type aliases.  Each is a distinct C++ type,
// so the compiler rejects cross-domain conversion at compile time.
// ---------------------------------------------------------------------------
using GpuMemoryDomainId = TypedId<GpuMemoryDomainIdTag>;
using DeviceId = TypedId<DeviceIdTag>;
using MemoryRegionId = TypedId<MemoryRegionIdTag>;
using AllocationId = TypedId<AllocationIdTag>;
using AllocationRequestId = TypedId<AllocationRequestIdTag>;
using AllocationLeaseId = TypedId<AllocationLeaseIdTag>;
using ResidencyId = TypedId<ResidencyIdTag>;
using ReclaimPlanId = TypedId<ReclaimPlanIdTag>;
using EvictionPlanId = TypedId<EvictionPlanIdTag>;
using CompactionPlanId = TypedId<CompactionPlanIdTag>;
using RelocationId = TypedId<RelocationIdTag>;
using StagingPlanId = TypedId<StagingPlanIdTag>;
using WorkloadId = TypedId<WorkloadIdTag>;
using ExecutionId = TypedId<ExecutionIdTag>;
using WorkerId = TypedId<WorkerIdTag>;
using SourceId = TypedId<SourceIdTag>;

using GpuMemoryDomainGeneration = TypedGeneration<GpuMemoryDomainGenerationGenTag>;
using DeviceGeneration = TypedGeneration<DeviceGenerationGenTag>;
using MemoryRegionGeneration = TypedGeneration<MemoryRegionGenerationGenTag>;
using AllocationGeneration = TypedGeneration<AllocationGenerationGenTag>;
using AllocationRequestGeneration = TypedGeneration<AllocationRequestGenerationGenTag>;
using AllocationLeaseGeneration = TypedGeneration<AllocationLeaseGenerationGenTag>;
using ResidencyGeneration = TypedGeneration<ResidencyGenerationGenTag>;
using ReclaimPlanGeneration = TypedGeneration<ReclaimPlanGenerationGenTag>;
using EvictionPlanGeneration = TypedGeneration<EvictionPlanGenerationGenTag>;
using CompactionPlanGeneration = TypedGeneration<CompactionPlanGenerationGenTag>;
using RelocationGeneration = TypedGeneration<RelocationGenerationGenTag>;
using StagingPlanGeneration = TypedGeneration<StagingPlanGenerationGenTag>;
using PressureGeneration = TypedGeneration<PressureGenerationGenTag>;
using WatermarkGeneration = TypedGeneration<WatermarkGenerationGenTag>;
using CapacityGeneration = TypedGeneration<CapacityGenerationGenTag>;
using ReservationGeneration = TypedGeneration<ReservationGenerationGenTag>;
using ResourceClaimGeneration = TypedGeneration<ResourceClaimGenerationGenTag>;
using PlacementGeneration = TypedGeneration<PlacementGenerationGenTag>;
using WorkloadGeneration = TypedGeneration<WorkloadGenerationGenTag>;
using ExecutionGeneration = TypedGeneration<ExecutionGenerationGenTag>;
using WorkerBootId = TypedGeneration<WorkerBootIdGenTag>;
using SourceBootId = TypedGeneration<SourceBootIdGenTag>;
using CoordinatorEpoch = TypedGeneration<CoordinatorEpochGenTag>;
using AuthorityGeneration = TypedGeneration<AuthorityGenerationGenTag>;
using RecoveryGeneration = TypedGeneration<RecoveryGenerationGenTag>;
using RevalidationGeneration = TypedGeneration<RevalidationGenerationGenTag>;
using CheckpointGeneration = TypedGeneration<CheckpointGenerationGenTag>;
using ModelResidencyGeneration = TypedGeneration<ModelResidencyGenerationGenTag>;
using AdapterGeneration = TypedGeneration<AdapterGenerationGenTag>;
using StateGeneration = TypedGeneration<StateGenerationGenTag>;
using PolicyGeneration = TypedGeneration<PolicyGenerationGenTag>;

}  // namespace gpu_memory_service
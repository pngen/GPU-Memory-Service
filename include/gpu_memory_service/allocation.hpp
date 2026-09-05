#pragma once
// Allocation request model, validation, and authoritative allocation record.
// A request is not an allocation; an allocation handle is not ownership unless
// its generation is current; a free request is not a completed free.

#include <cstdint>
#include <string>

#include "enum.hpp"
#include "error.hpp"
#include "types.hpp"

namespace gpu_memory_service {

// ---------------------------------------------------------------------------
// AllocationRequest — validated allocation intent (not yet an allocation).
// ---------------------------------------------------------------------------
struct AllocationRequest {
  std::uint64_t bytes{0};
  std::uint64_t min_bytes{0};              // elastic lower bound (0 = not elastic)
  std::uint64_t alignment{1};
  bool contiguous{true};
  MemoryClass mem_class{MemoryClass::UNKNOWN};
  Reclaimability reclaimability{Reclaimability::UNKNOWN};
  Movability movability{Movability::MOVABLE};
  bool requires_pin{false};

  std::string owner;
  WorkerId worker;
  WorkerBootId worker_boot;
  SourceBootId source_boot;
  WorkloadId workload;
  ExecutionId execution;

  ReservationGeneration reservation_gen{ReservationGeneration::none()};
  ResourceClaimGeneration resource_claim_gen{ResourceClaimGeneration::none()};
  PlacementGeneration placement_gen{PlacementGeneration::none()};
  PolicyGeneration policy_gen{PolicyGeneration::none()};

  std::string backing_ref;                // reconstructible backing reference
  Provenance provenance{Provenance::DERIVED};
};

// ---------------------------------------------------------------------------
// AllocationRecord — authoritative, generation-bound allocation state once a
// physical allocation is granted and committed.
// ---------------------------------------------------------------------------
struct AllocationRecord {
  AllocationId id;
  AllocationGeneration generation{AllocationGeneration::none()};
  std::uint64_t requested_bytes{0};
  std::uint64_t region_size{0};           // arena footprint (alignment-rounded)
  std::uint64_t offset{0};
  std::uint64_t alignment{1};
  MemoryRegionId region_id;
  MemoryRegionGeneration region_generation{MemoryRegionGeneration::none()};

  MemoryClass mem_class{MemoryClass::UNKNOWN};
  Reclaimability reclaimability{Reclaimability::UNKNOWN};
  Movability movability{Movability::MOVABLE};
  std::uint32_t pin_count{0};
  ResidencyState residency{ResidencyState::UNKNOWN};
  LifecycleState lifecycle{LifecycleState::REQUESTED};

  std::string owner;
  WorkerId worker;
  WorkerBootId worker_boot;
  SourceBootId source_boot;
  WorkloadId workload;
  ExecutionId execution;

  RelocationGeneration relocation_generation{RelocationGeneration::none()};
  ReservationGeneration reservation_gen{ReservationGeneration::none()};
  ResourceClaimGeneration resource_claim_gen{ResourceClaimGeneration::none()};
  PlacementGeneration placement_gen{PlacementGeneration::none()};
  PolicyGeneration policy_gen{PolicyGeneration::none()};

  std::string backing_ref;
  Provenance provenance{Provenance::DERIVED};
  bool recoverable{false};                // safe to reclaim while durably backed
};

// ---------------------------------------------------------------------------
// Request validation.  Returns ok() or a typed RejectReason.  Validation is
// pragmatic about boundaries but strict about correctness.
// ---------------------------------------------------------------------------
[[nodiscard]] Failure validate_request(const AllocationRequest& req,
                                       std::uint64_t governed_capacity) noexcept;

}  // namespace gpu_memory_service

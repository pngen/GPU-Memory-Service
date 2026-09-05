#pragma once
// MemoryGovernor — the transport-independent authoritative runtime for one
// governed GPU-memory domain.  It owns the authoritative byte layout
// (RegionAllocator), generation-bound allocation records, residency/movability,
// pressure, reclaim selection, compaction planning and relocation, and safe
// release.  The physical bytes come from an IBackend (host or CUDA); the
// governor never treats the backend as durable authority.

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "allocation.hpp"
#include "backend.hpp"
#include "enum.hpp"
#include "error.hpp"
#include "memory_domain.hpp"
#include "persistence.hpp"
#include "pressure.hpp"
#include "region_allocator.hpp"
#include "types.hpp"

namespace gpu_memory_service {

// ---------------------------------------------------------------------------
// MemorySummary — typed snapshot of governed memory facts and evidence class.
// ---------------------------------------------------------------------------
struct MemorySummary {
  std::uint64_t managed_capacity{0};
  std::uint64_t free{0};
  std::uint64_t committed{0};
  std::uint64_t resident{0};
  std::uint64_t reclaimable{0};
  std::uint64_t movable{0};
  std::uint64_t pinned{0};
  std::uint64_t largest_free{0};
  std::size_t free_regions{0};
  std::size_t live_allocations{0};
  double external_fragmentation{0.0};
  Provenance provenance{Provenance::UNKNOWN};
  PressureState pressure{PressureState::UNKNOWN};
  GpuMemoryDomainGeneration domain_generation{GpuMemoryDomainGeneration::none()};
};

// ---------------------------------------------------------------------------
// MemoryGovernor
// ---------------------------------------------------------------------------
class MemoryGovernor {
 public:
  // AllocationHandle — current authoritative location/generation for a request.
  struct AllocationHandle {
    AllocationId id;
    AllocationGeneration generation;
    std::uint64_t offset{0};
    std::uint64_t size{0};
    std::uint64_t alignment{1};
  };

  // One relocation step in a compaction plan (generation-bound).
  struct CompactionStep {
    AllocationId id;
    AllocationGeneration generation;     // generation before the move
    std::uint64_t old_offset{0};
    std::uint64_t old_size{0};
    std::uint64_t new_offset{0};
    std::uint64_t new_size{0};
    Movability movability{Movability::MOVABLE};
  };

  struct CompactionPlan {
    std::vector<CompactionStep> steps;
    std::uint64_t bytes_moved{0};
    std::uint64_t largest_free_before{0};
    std::uint64_t largest_free_after{0};
    std::string rationale;
  };

  // Construct with a domain descriptor and a physical backend.  The managed
  // capacity is taken from the domain.  initialize() allocates the arena.
  MemoryGovernor(MemoryDomain domain, std::unique_ptr<IBackend> backend);
  ~MemoryGovernor();
  MemoryGovernor(const MemoryGovernor&) = delete;
  MemoryGovernor& operator=(const MemoryGovernor&) = delete;

  // Allocate the physical arena for the managed capacity.
  [[nodiscard]] Result<void> initialize();

  // Request a governed allocation.  Validates, reserves the authoritative
  // region, and (because the arena is physically present) commits it.  Returns
  // the current handle.
  [[nodiscard]] Result<AllocationHandle> request(const AllocationRequest& req);

  // Release an allocation.  The generation must be current; a stale or double
  // free is rejected; a pinned allocation is rejected.  Optionally carries a
  // worker boot authority that must match the allocator authority.
  [[nodiscard]] Result<AllocationRecord> release(AllocationId id, AllocationGeneration gen);
  [[nodiscard]] Result<AllocationRecord> release_for_worker(AllocationId id,
                                                             AllocationGeneration gen,
                                                             WorkerBootId worker_boot,
                                                             SourceBootId source_boot);

  [[nodiscard]] Result<void> pin(AllocationId id, AllocationGeneration gen);
  [[nodiscard]] Result<void> unpin(AllocationId id, AllocationGeneration gen);
  [[nodiscard]] Result<AllocationRecord> get(AllocationId id, AllocationGeneration gen) const;

  [[nodiscard]] MemorySummary summary() const;
  [[nodiscard]] PressureState pressure() const;
  [[nodiscard]] std::vector<AllocationRecord> allocations() const;
  // Deterministic reclaim-candidate selection (hard exclusions first).
  [[nodiscard]] std::vector<AllocationId> reclaim_candidates(std::size_t max) const;

  // Deterministic compaction planning.  Never mutates state.
  [[nodiscard]] CompactionPlan plan_compaction() const;

  // Apply a validated, already-physically-performed relocation.  Advances the
  // allocation and region generations; the old location becomes stale.  Exactly
  // one current location survives.
  [[nodiscard]] Result<AllocationHandle> apply_relocation(AllocationId id,
                                                          AllocationGeneration gen,
                                                          std::uint64_t new_offset,
                                                          std::uint64_t new_size);

  // How many bytes the relocation physically moves (== allocation region size).
  [[nodiscard]] std::uint64_t region_size(AllocationId id) const;

  [[nodiscard]] const MemoryDomain& domain() const noexcept { return domain_; }
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] void* arena_base() const noexcept { return arena_base_; }
  [[nodiscard]] std::size_t record_count() const noexcept { return records_.size(); }

  // Human-readable structured explanation.
  [[nodiscard]] std::string explain() const;

  // Persistence hooks.  save() serializes a durable snapshot (never raw CUDA
  // pointers).  restore() conservatively re-derives governing state after a
  // coordinator/process restart: recovered process-owned dynamic allocations
  // become STALE / REVALIDATION_REQUIRED rather than silently current.
  [[nodiscard]] Result<void> save(const std::string& path) const;
  [[nodiscard]] persist::PersistSnapshot snapshot() const;
  [[nodiscard]] Result<void> restore(const persist::PersistSnapshot& s);
  [[nodiscard]] Result<void> load(const std::string& path);

 private:
  AllocationId next_id_locked();
  std::uint64_t reclaimable_bytes_locked() const;
  [[nodiscard]] Result<AllocationHandle> request_locked(const AllocationRequest& req);
  [[nodiscard]] Result<AllocationRecord> release_locked(AllocationId id,
                                                        AllocationGeneration gen,
                                                        WorkerBootId worker_boot,
                                                        SourceBootId source_boot);
  [[nodiscard]] Result<void> pin_locked(AllocationId id, AllocationGeneration gen);
  [[nodiscard]] Result<void> unpin_locked(AllocationId id, AllocationGeneration gen);
  [[nodiscard]] PressureState pressure_locked() const;
  [[nodiscard]] MemorySummary summary_locked() const;

  mutable std::mutex mutex_;
  mutable MemoryDomain domain_;
  std::unique_ptr<IBackend> backend_;
  RegionAllocator arena_;
  std::map<AllocationId, AllocationRecord> records_;
  std::map<AllocationId, AllocationRecord> retired_;
  void* arena_base_{nullptr};
  bool initialized_{false};
  std::uint64_t next_alloc_id_{1};
  std::uint64_t history_count_{0};
};

}  // namespace gpu_memory_service
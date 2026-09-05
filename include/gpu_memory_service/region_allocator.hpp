#pragma once
// Exact region / suballocation model over a managed GPU-memory arena.
// Invariants (checked by verify()): live regions never overlap; live+free ==
// managed bytes; each live region aligned; free regions canonical/merged.
// Allocation is O(log n) via a size-indexed free-slot set (no O(n^2) scans).

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "enum.hpp"
#include "error.hpp"
#include "types.hpp"

namespace gpu_memory_service {

enum class AllocationStrategy : std::uint8_t {
  FIRST_FIT = 0,
  BEST_FIT
};

struct RegionInfo {
  MemoryRegionId region_id;
  MemoryRegionGeneration region_generation;
  std::uint64_t offset{0};
  std::uint64_t size{0};
  std::uint64_t alignment{1};
  bool live{false};
  AllocationId alloc_id;
  AllocationGeneration alloc_generation;
  Movability movability{Movability::MOVABLE};
  std::uint32_t pin_count{0};
  MemoryClass mem_class{MemoryClass::UNKNOWN};
  Reclaimability reclaimability{Reclaimability::UNKNOWN};
  ResidencyState residency{ResidencyState::UNKNOWN};
  LifecycleState lifecycle{LifecycleState::REQUESTED};
  Provenance provenance{Provenance::UNKNOWN};
  WorkerBootId worker_boot;

  friend bool operator==(const RegionInfo&, const RegionInfo&) = default;
};

struct AllocRequest {
  std::uint64_t bytes{0};
  std::uint64_t alignment{1};
  Movability movability{Movability::MOVABLE};
  MemoryClass mem_class{MemoryClass::UNKNOWN};
  Reclaimability reclaimability{Reclaimability::UNKNOWN};
  AllocationId alloc_id;
  AllocationGeneration alloc_generation;
  WorkerBootId worker_boot;
  Provenance provenance{Provenance::DERIVED};
};

class RegionAllocator {
 public:
  explicit RegionAllocator(std::uint64_t capacity);
  RegionAllocator(const RegionAllocator&) = delete;
  RegionAllocator& operator=(const RegionAllocator&) = delete;
  RegionAllocator(RegionAllocator&&) = default;
  RegionAllocator& operator=(RegionAllocator&&) = default;

  Result<RegionInfo> allocate(const AllocRequest& req);
  Result<RegionInfo> free(AllocationId id, AllocationGeneration gen);
  Result<RegionInfo> relocate(AllocationId id, AllocationGeneration gen,
                              std::uint64_t new_offset, std::uint64_t new_size,
                              std::uint64_t new_alignment);
  Result<RegionInfo> pin(AllocationId id, AllocationGeneration gen);
  Result<RegionInfo> unpin(AllocationId id, AllocationGeneration gen);
  Result<RegionInfo> get(AllocationId id, AllocationGeneration gen) const;

  [[nodiscard]] std::uint64_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::uint64_t live_bytes() const noexcept { return live_bytes_; }
  [[nodiscard]] std::uint64_t free_bytes() const noexcept { return capacity_ - live_bytes_; }
  [[nodiscard]] std::size_t free_region_count() const;
  [[nodiscard]] std::uint64_t largest_free_region() const;
  [[nodiscard]] std::size_t live_region_count() const noexcept { return live_count_; }
  [[nodiscard]] std::uint64_t movable_bytes() const;
  [[nodiscard]] std::uint64_t pinned_bytes() const;
  [[nodiscard]] double external_fragmentation() const;
  [[nodiscard]] std::size_t region_count() const noexcept { return regions_.size(); }
  void set_strategy(AllocationStrategy s) noexcept { strategy_ = s; }
  [[nodiscard]] AllocationStrategy strategy() const noexcept { return strategy_; }

  [[nodiscard]] std::string verify() const;
  [[nodiscard]] std::string dump() const;
  [[nodiscard]] std::vector<RegionInfo> snapshot_live() const;
  [[nodiscard]] std::vector<RegionInfo> snapshot_free() const;

 private:
  struct FreeSlot {
    std::uint64_t size{0};
    std::uint64_t offset{0};
    friend bool operator<(const FreeSlot& a, const FreeSlot& b) noexcept {
      if (a.size != b.size) return a.size < b.size;
      return a.offset < b.offset;
    }
  };
  static std::uint64_t align_up(std::uint64_t v, std::uint64_t align) noexcept;
  void ensure_summary() const noexcept;
  Result<RegionInfo> canonicalize_free(std::uint64_t off, std::uint64_t size);
  void index_free(std::uint64_t off, std::uint64_t size) { free_slots_.insert({size, off}); }
  void unindex_free(std::uint64_t off, std::uint64_t size) {
    auto range = free_slots_.equal_range(FreeSlot{size, off});
    for (auto it = range.first; it != range.second; ++it) {
      if (it->offset == off) { free_slots_.erase(it); break; }
    }
  }

  std::uint64_t capacity_{0};
  std::uint64_t live_bytes_{0};
  std::size_t live_count_{0};
  std::uint64_t next_region_id_{1};
  AllocationStrategy strategy_{AllocationStrategy::BEST_FIT};
  std::map<std::uint64_t, RegionInfo> regions_;
  std::map<AllocationId, std::uint64_t> alloc_offset_;
  std::set<FreeSlot> free_slots_;
  mutable bool summary_dirty_{true};
  mutable std::size_t free_count_{0};
  mutable std::uint64_t largest_free_{0};
};

}  // namespace gpu_memory_service

#include "gpu_memory_service/region_allocator.hpp"

#include <algorithm>
#include <sstream>

namespace gpu_memory_service {

namespace {
[[nodiscard]] bool is_pow2(std::uint64_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }
}  // namespace

std::uint64_t RegionAllocator::align_up(std::uint64_t v, std::uint64_t align) noexcept {
  if (align == 0) return v;
  const std::uint64_t mask = align - 1;
  const std::uint64_t r = v + mask;
  if (r < v) return 0;
  return r & ~mask;
}

RegionAllocator::RegionAllocator(std::uint64_t capacity) : capacity_(capacity) {
  RegionInfo free;
  free.live = false;
  free.offset = 0;
  free.size = capacity;
  free.region_id = MemoryRegionId(next_region_id_);
  free.region_generation = MemoryRegionGeneration(1);
  free.provenance = Provenance::DERIVED;
  regions_.emplace(0, free);
  if (capacity > 0) index_free(0, capacity);
}

void RegionAllocator::ensure_summary() const noexcept {
  if (!summary_dirty_) return;
  free_count_ = free_slots_.size();
  largest_free_ = free_slots_.empty() ? 0 : free_slots_.rbegin()->size;
  summary_dirty_ = false;
}

Result<RegionInfo> RegionAllocator::allocate(const AllocRequest& req) {
  if (req.bytes == 0) return Failure(RejectReason::ZERO_BYTES, "allocation of zero bytes");
  if (req.alignment == 0 || !is_pow2(req.alignment)) {
    return Failure(req.alignment == 0 ? RejectReason::IMPOSSIBLE_ALIGNMENT
                                      : RejectReason::NON_POWER_OF_TWO_ALIGNMENT, "invalid alignment");
  }
  if (req.bytes > capacity_) return Failure(RejectReason::INSUFFICIENT_TOTAL_CAPACITY, "request exceeds capacity");
  std::uint64_t aligned = align_up(req.bytes, req.alignment);
  if (aligned == 0 || aligned < req.bytes) return Failure(RejectReason::SIZE_OVERFLOW, "aligned size overflow");
  if (aligned > capacity_) return Failure(RejectReason::INSUFFICIENT_TOTAL_CAPACITY, "aligned request exceeds capacity");

  // Best-fit by size via the size-indexed free-slot set (O(log n)).
  for (auto it = free_slots_.lower_bound(FreeSlot{aligned, 0}); it != free_slots_.end(); ++it) {
    const std::uint64_t foff = it->offset;
    const std::uint64_t fsize = it->size;
    const std::uint64_t front = align_up(foff, req.alignment) - foff;
    if (front > fsize || aligned > fsize - front) continue;

    const std::uint64_t alloc_off = foff + front;
    const std::uint64_t tail = fsize - front - aligned;
    auto rit = regions_.find(foff);
    const RegionInfo frec = rit->second;
    regions_.erase(rit);
    free_slots_.erase(it);

    if (front > 0) {
      RegionInfo f; f.live = false; f.offset = foff; f.size = front;
      f.region_id = MemoryRegionId(next_region_id_++); f.region_generation = MemoryRegionGeneration(1);
      f.provenance = frec.provenance;
      regions_.emplace(foff, f); index_free(foff, front);
    }
    RegionInfo live;
    live.live = true; live.offset = alloc_off; live.size = aligned; live.alignment = req.alignment;
    live.region_id = MemoryRegionId(next_region_id_++); live.region_generation = MemoryRegionGeneration(1);
    live.alloc_id = req.alloc_id; live.alloc_generation = req.alloc_generation;
    live.movability = req.movability; live.mem_class = req.mem_class; live.reclaimability = req.reclaimability;
    live.residency = ResidencyState::RESIDENT; live.lifecycle = LifecycleState::ALLOCATED;
    live.provenance = req.provenance; live.worker_boot = req.worker_boot;
    regions_.emplace(alloc_off, live);
    if (tail > 0) {
      RegionInfo f; f.live = false; f.offset = alloc_off + aligned; f.size = tail;
      f.region_id = MemoryRegionId(next_region_id_++); f.region_generation = MemoryRegionGeneration(1);
      f.provenance = frec.provenance;
      regions_.emplace(alloc_off + aligned, f); index_free(alloc_off + aligned, tail);
    }
    live_bytes_ += aligned;
    live_count_ += 1;
    alloc_offset_[req.alloc_id] = alloc_off;
    summary_dirty_ = true;
    return live;
  }

  if (free_bytes() < aligned) return Failure(RejectReason::INSUFFICIENT_TOTAL_CAPACITY, "total managed free bytes insufficient");
  return Failure(RejectReason::INSUFFICIENT_CONTIGUOUS_CAPACITY, "no contiguous free region of sufficient size");
}

Result<RegionInfo> RegionAllocator::canonicalize_free(std::uint64_t off, std::uint64_t size) {
  if (size == 0) return Failure(RejectReason::ZERO_BYTES, "zero-size free interval");
  if (off > capacity_ || size > capacity_ - off) return Failure(RejectReason::SIZE_OVERFLOW, "free interval out of bounds");
  std::uint64_t lo = off;
  std::uint64_t hi = off + size;

  auto it = regions_.lower_bound(off);
  if (it != regions_.begin()) {
    auto pit = std::prev(it);
    if (!pit->second.live && pit->first + pit->second.size == off) {
      unindex_free(pit->first, pit->second.size);
      lo = pit->first;
      regions_.erase(pit);
    }
  }
  auto nit = regions_.lower_bound(off);
  if (nit != regions_.end() && !nit->second.live && nit->first == hi) {
    unindex_free(nit->first, nit->second.size);
    hi = nit->first + nit->second.size;
    regions_.erase(nit);
  }

  RegionInfo f; f.live = false; f.offset = lo; f.size = hi - lo;
  f.region_id = MemoryRegionId(next_region_id_++); f.region_generation = MemoryRegionGeneration(1);
  f.provenance = Provenance::DERIVED;
  regions_.emplace(lo, f);
  index_free(lo, hi - lo);
  summary_dirty_ = true;
  return f;
}

Result<RegionInfo> RegionAllocator::free(AllocationId id, AllocationGeneration gen) {
  const auto it = alloc_offset_.find(id);
  if (it == alloc_offset_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown or already-freed allocation");
  const std::uint64_t off = it->second;
  const auto rit = regions_.find(off);
  if (rit == regions_.end() || !rit->second.live) return Failure(RejectReason::STALE_GENERATION, "allocation region is not live");
  if (rit->second.alloc_generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale allocation generation on free");
  const std::uint64_t size = rit->second.size;
  const RegionInfo freed = rit->second;
  regions_.erase(rit);
  alloc_offset_.erase(it);
  live_bytes_ -= size;
  live_count_ -= 1;
  return canonicalize_free(off, size);
}

Result<RegionInfo> RegionAllocator::relocate(AllocationId id, AllocationGeneration gen,
                                            std::uint64_t new_offset, std::uint64_t new_size,
                                            std::uint64_t new_alignment) {
  const auto it = alloc_offset_.find(id);
  if (it == alloc_offset_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  const std::uint64_t old_off = it->second;
  const auto rit = regions_.find(old_off);
  if (rit == regions_.end() || !rit->second.live) return Failure(RejectReason::STALE_GENERATION, "allocation region is not live");
  if (rit->second.alloc_generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on relocate");
  if (new_alignment == 0 || !is_pow2(new_alignment)) return Failure(RejectReason::IMPOSSIBLE_ALIGNMENT, "invalid relocation alignment");
  if (new_size == 0 || new_offset % new_alignment != 0) return Failure(RejectReason::INVALID_REQUEST, "invalid relocation target");
  if (new_offset > capacity_ || new_size > capacity_ - new_offset) return Failure(RejectReason::SIZE_OVERFLOW, "relocation target out of bounds");
  const std::uint64_t new_hi = new_offset + new_size;
  const std::uint64_t old_end = old_off + rit->second.size;
  if (new_offset < old_end && old_off < new_hi) return Failure(RejectReason::INVALID_REQUEST, "relocation target overlaps current region");

  // Validate no live region overlaps the target BEFORE any mutation.
  {
    auto c = regions_.lower_bound(new_offset);
    if (c != regions_.begin()) {
      auto pc = std::prev(c);
      if (pc->second.live && pc->first + pc->second.size > new_offset) return Failure(RejectReason::INVALID_REQUEST, "relocation target overlaps a live region");
    }
    for (; c != regions_.end() && c->first < new_hi; ++c) {
      if (c->second.live) return Failure(RejectReason::INVALID_REQUEST, "relocation target overlaps a live region");
    }
  }

  const RegionInfo old = rit->second;
  const std::uint64_t old_size = old.size;
  regions_.erase(rit);
  alloc_offset_.erase(it);

  // Carve the target interval out of free space, keeping slots consistent.
  {
    std::map<std::uint64_t, RegionInfo> next;
    for (const auto& kv : regions_) {
      const std::uint64_t s = kv.first;
      const std::uint64_t e = s + kv.second.size;
      if (kv.second.live) { next[s] = kv.second; continue; }
      const std::uint64_t ov_s = std::max(s, new_offset);
      const std::uint64_t ov_e = std::min(e, new_hi);
      if (ov_s >= ov_e) { next[s] = kv.second; continue; }
      unindex_free(s, e - s);
      if (s < ov_s) { RegionInfo f = kv.second; f.offset = s; f.size = ov_s - s; next[s] = f; index_free(s, ov_s - s); }
      if (ov_e < e) { RegionInfo f = kv.second; f.offset = ov_e; f.size = e - ov_e; next[ov_e] = f; index_free(ov_e, e - ov_e); }
    }
    regions_ = std::move(next);
  }

  const auto cf = canonicalize_free(old_off, old_size);
  if (!cf.ok()) return cf.error();

  RegionInfo moved = old;
  moved.offset = new_offset;
  moved.size = new_size;
  moved.alignment = new_alignment;
  moved.alloc_generation = gen.next();
  moved.region_generation = old.region_generation.next();
  moved.lifecycle = LifecycleState::RESIDENT;
  regions_.emplace(new_offset, moved);
  alloc_offset_[id] = new_offset;

  if (new_size >= old_size) live_bytes_ += (new_size - old_size);
  else live_bytes_ -= (old_size - new_size);
  summary_dirty_ = true;
  return moved;
}

Result<RegionInfo> RegionAllocator::pin(AllocationId id, AllocationGeneration gen) {
  const auto it = alloc_offset_.find(id);
  if (it == alloc_offset_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  auto& r = regions_.at(it->second);
  if (!r.live) return Failure(RejectReason::STALE_GENERATION, "allocation not live");
  if (r.alloc_generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on pin");
  r.movability = Movability::PINNED; ++r.pin_count;
  return r;
}

Result<RegionInfo> RegionAllocator::unpin(AllocationId id, AllocationGeneration gen) {
  const auto it = alloc_offset_.find(id);
  if (it == alloc_offset_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  auto& r = regions_.at(it->second);
  if (!r.live) return Failure(RejectReason::STALE_GENERATION, "allocation not live");
  if (r.alloc_generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on unpin");
  if (r.pin_count == 0) return Failure(RejectReason::INVALID_REQUEST, "unpin of non-pinned allocation");
  --r.pin_count;
  if (r.pin_count == 0) r.movability = Movability::MOVABLE;
  return r;
}

Result<RegionInfo> RegionAllocator::get(AllocationId id, AllocationGeneration gen) const {
  const auto it = alloc_offset_.find(id);
  if (it == alloc_offset_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  const auto rit = regions_.find(it->second);
  if (rit == regions_.end() || !rit->second.live) return Failure(RejectReason::STALE_GENERATION, "allocation not live");
  if (rit->second.alloc_generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on query");
  return rit->second;
}

std::size_t RegionAllocator::free_region_count() const { ensure_summary(); return free_count_; }
std::uint64_t RegionAllocator::largest_free_region() const { ensure_summary(); return largest_free_; }
double RegionAllocator::external_fragmentation() const {
  ensure_summary();
  if (free_bytes() == 0) return 0.0;
  return 1.0 - (static_cast<double>(largest_free_) / static_cast<double>(free_bytes()));
}
std::uint64_t RegionAllocator::movable_bytes() const {
  std::uint64_t s = 0; for (const auto& kv : regions_) if (kv.second.live && kv.second.movability == Movability::MOVABLE) s += kv.second.size;
  return s;
}
std::uint64_t RegionAllocator::pinned_bytes() const {
  std::uint64_t s = 0; for (const auto& kv : regions_) if (kv.second.live && kv.second.movability == Movability::PINNED) s += kv.second.size;
  return s;
}

std::vector<RegionInfo> RegionAllocator::snapshot_live() const {
  std::vector<RegionInfo> v; for (const auto& kv : regions_) if (kv.second.live) v.push_back(kv.second); return v;
}
std::vector<RegionInfo> RegionAllocator::snapshot_free() const {
  std::vector<RegionInfo> v; for (const auto& kv : regions_) if (!kv.second.live) v.push_back(kv.second); return v;
}

std::string RegionAllocator::verify() const {
  std::ostringstream os;
  std::uint64_t expected = 0;
  std::uint64_t live_sum = 0;
  bool prev_free = false, have_prev = false;
  std::uint64_t prev_end = 0;
  std::size_t free_count_regions = 0;
  for (auto it = regions_.begin(); it != regions_.end(); ++it) {
    const RegionInfo& r = it->second;
    if (it->first != expected) { os << "gap/overlap at " << it->first << " expected " << expected << "\n"; return os.str(); }
    if (have_prev) {
      if (prev_end > it->first) { os << "overlap at " << it->first << "\n"; return os.str(); }
      if (prev_free && !r.live && prev_end == it->first) { os << "adjacent free not merged at " << it->first << "\n"; return os.str(); }
    }
    if (r.live) {
      live_sum += r.size;
      const std::uint64_t al = r.alignment;
      if (al == 0 || (al & (al - 1)) != 0) { os << "bad alignment at " << it->first << "\n"; return os.str(); }
      if (it->first % al != 0) { os << "misaligned at " << it->first << "\n"; return os.str(); }
    } else { ++free_count_regions; }
    prev_end = it->first + r.size;
    prev_free = !r.live;
    have_prev = true;
    expected = prev_end;
  }
  if (expected != capacity_) { os << "arena does not cover capacity\n"; return os.str(); }
  if (live_sum != live_bytes_) { os << "live byte sum mismatch\n"; return os.str(); }
  if (free_count_regions != free_slots_.size()) { os << "free-slot index drift\n"; return os.str(); }
  return "";
}

std::string RegionAllocator::dump() const {
  std::ostringstream os;
  os << "arena capacity=" << capacity_ << " live=" << live_bytes_ << " free=" << free_bytes() << " live_count=" << live_count_ << "\n";
  for (const auto& kv : regions_) {
    const RegionInfo& r = kv.second;
    os << "  [" << r.offset << ", " << (r.offset + r.size) << ") " << (r.live ? "LIVE" : "free")
       << " id=" << r.alloc_id << " gen=" << r.alloc_generation;
    if (r.live) os << " mov=" << r.movability << " pin=" << r.pin_count << " class=" << r.mem_class;
    os << "\n";
  }
  return os.str();
}

}  // namespace gpu_memory_service
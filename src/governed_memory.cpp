#include "gpu_memory_service/governed_memory.hpp"

#include <algorithm>
#include <sstream>

#include "gpu_memory_service/util.hpp"

namespace gpu_memory_service {

namespace {
std::uint64_t sub(std::uint64_t a, std::uint64_t b) noexcept { return a >= b ? a - b : 0; }
[[nodiscard]] bool reclaimability_requires_backing(Reclaimability r) noexcept {
  return r == Reclaimability::RECLAIM_IF_RELOADABLE || r == Reclaimability::RECLAIM_IF_CHECKPOINTED;
}
[[nodiscard]] int reclaimability_priority(Reclaimability r) noexcept {
  switch (r) {
    case Reclaimability::BEST_EFFORT: return 0;
    case Reclaimability::RECLAIM_IF_CHECKPOINTED: return 1;
    case Reclaimability::RECLAIM_IF_RELOADABLE: return 2;
    case Reclaimability::RECLAIM_IF_COLD: return 3;
    case Reclaimability::RECLAIM_AFTER_COMPLETION: return 4;
    case Reclaimability::NEVER_RECLAIM: return 5;
    case Reclaimability::UNKNOWN: return 6;
  }
  return 6;
}
}  // namespace

MemoryGovernor::MemoryGovernor(MemoryDomain domain, std::unique_ptr<IBackend> backend)
    : domain_(domain),
      backend_(std::move(backend)),
      arena_(domain.managed_capacity) {}

MemoryGovernor::~MemoryGovernor() {
  if (backend_ && arena_base_ && initialized_) {
    backend_->free_arena(arena_base_);
    arena_base_ = nullptr;
  }
}

Result<void> MemoryGovernor::initialize() {
  std::lock_guard<std::mutex> lk(mutex_);
  if (initialized_) return Result<void>();
  if (!backend_) return Failure(RejectReason::INTERNAL_ERROR, "no backend");
  if (domain_.managed_capacity == 0) return Failure(RejectReason::INVALID_REQUEST, "managed capacity is zero");
  std::string err;
  void* base = nullptr;
  if (!backend_->allocate_arena(domain_.managed_capacity, 256, &base, &err)) {
    return Failure(RejectReason::PHYSICAL_OOM, err);
  }
  arena_base_ = base;
  initialized_ = true;
  domain_.health = DeviceHealth::HEALTHY;
  return Result<void>();
}

AllocationId MemoryGovernor::next_id_locked() { return AllocationId(next_alloc_id_++); }

Result<MemoryGovernor::AllocationHandle> MemoryGovernor::request(const AllocationRequest& req) {
  std::lock_guard<std::mutex> lk(mutex_);
  return request_locked(req);
}

Result<MemoryGovernor::AllocationHandle> MemoryGovernor::request_locked(const AllocationRequest& req) {
  const Failure vf = validate_request(req, domain_.managed_capacity);
  if (!vf.ok()) return vf;
  if (!initialized_) return Failure(RejectReason::INTERNAL_ERROR, "governor not initialized");

  const std::uint64_t region_size = detail::align_up(req.bytes, req.alignment);
  if (region_size == 0 || region_size < req.bytes) {
    return Failure(RejectReason::SIZE_OVERFLOW, "aligned allocation size overflow");
  }

  const AllocationId id = next_id_locked();
  const AllocationGeneration gen = AllocationGeneration::first();

  AllocRequest ar;
  ar.bytes = req.bytes;
  ar.alignment = req.alignment;
  ar.movability = req.requires_pin ? Movability::PINNED : req.movability;
  ar.mem_class = req.mem_class;
  ar.reclaimability = req.reclaimability;
  ar.alloc_id = id;
  ar.alloc_generation = gen;
  ar.worker_boot = req.worker_boot;
  ar.provenance = req.provenance;

  auto r = arena_.allocate(ar);
  if (!r.ok()) return r.error();

  AllocationRecord rec;
  rec.id = id;
  rec.generation = gen;
  rec.requested_bytes = req.bytes;
  rec.region_size = r->size;
  rec.offset = r->offset;
  rec.alignment = req.alignment;
  rec.region_id = r->region_id;
  rec.region_generation = r->region_generation;
  rec.mem_class = req.mem_class;
  rec.reclaimability = req.reclaimability;
  rec.movability = ar.movability;
  rec.pin_count = req.requires_pin ? 1u : 0u;
  rec.residency = ResidencyState::RESIDENT;
  rec.lifecycle = req.requires_pin ? LifecycleState::PINNED : LifecycleState::RESIDENT;
  rec.owner = req.owner;
  rec.worker = req.worker;
  rec.worker_boot = req.worker_boot;
  rec.source_boot = req.source_boot;
  rec.workload = req.workload;
  rec.execution = req.execution;
  rec.relocation_generation = RelocationGeneration::none();
  rec.reservation_gen = req.reservation_gen;
  rec.resource_claim_gen = req.resource_claim_gen;
  rec.placement_gen = req.placement_gen;
  rec.policy_gen = req.policy_gen;
  rec.backing_ref = req.backing_ref;
  rec.provenance = req.provenance;
  rec.recoverable = !req.backing_ref.empty();

  records_[id] = rec;
  domain_.current_committed += r->size;
  domain_.current_resident += r->size;
  if (rec.movability == Movability::MOVABLE) domain_.current_movable += r->size;
  else if (rec.movability == Movability::PINNED) domain_.current_pinned += r->size;

  return AllocationHandle{id, gen, r->offset, r->size, req.alignment};
}

Result<AllocationRecord> MemoryGovernor::release(AllocationId id, AllocationGeneration gen) {
  std::lock_guard<std::mutex> lk(mutex_);
  return release_locked(id, gen, WorkerBootId(), SourceBootId());
}

Result<AllocationRecord> MemoryGovernor::release_for_worker(AllocationId id, AllocationGeneration gen,
                                                            WorkerBootId worker_boot, SourceBootId source_boot) {
  std::lock_guard<std::mutex> lk(mutex_);
  return release_locked(id, gen, worker_boot, source_boot);
}

Result<AllocationRecord> MemoryGovernor::release_locked(AllocationId id, AllocationGeneration gen,
                                                       WorkerBootId worker_boot, SourceBootId source_boot) {
  auto it = records_.find(id);
  if (it == records_.end()) {
    auto rit = retired_.find(id);
    if (rit != retired_.end()) {
      if (rit->second.generation == gen) return Failure(RejectReason::DOUBLE_FREE, "double free of released allocation");
      return Failure(RejectReason::STALE_GENERATION, "stale generation on released allocation");
    }
    return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  }
  const AllocationRecord rec = it->second;
  if (rec.generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale allocation generation on release");
  if (worker_boot.is_set() && worker_boot != rec.worker_boot) return Failure(RejectReason::STALE_AUTHORITY, "stale worker boot authority");
  if (source_boot.is_set() && source_boot != rec.source_boot) return Failure(RejectReason::STALE_AUTHORITY, "stale source boot authority");
  if (rec.pin_count > 0) return Failure(RejectReason::FREE_OF_ACTIVE_OR_PINNED, "cannot release a pinned allocation");
  if (rec.residency == ResidencyState::ACTIVE) return Failure(RejectReason::FREE_OF_ACTIVE_OR_PINNED, "cannot release an active allocation");

  auto fr = arena_.free(id, gen);
  if (!fr.ok()) return fr.error();

  AllocationRecord rel = rec;
  rel.lifecycle = LifecycleState::RELEASED;
  rel.residency = ResidencyState::NONRESIDENT;
  retired_[id] = rel;
  records_.erase(it);

  domain_.current_committed = sub(domain_.current_committed, rel.region_size);
  domain_.current_resident = sub(domain_.current_resident, rel.region_size);
  if (rel.movability == Movability::MOVABLE) domain_.current_movable = sub(domain_.current_movable, rel.region_size);
  else if (rel.movability == Movability::PINNED) domain_.current_pinned = sub(domain_.current_pinned, rel.region_size);
  ++history_count_;
  return rel;
}

Result<void> MemoryGovernor::pin(AllocationId id, AllocationGeneration gen) {
  std::lock_guard<std::mutex> lk(mutex_);
  return pin_locked(id, gen);
}

Result<void> MemoryGovernor::pin_locked(AllocationId id, AllocationGeneration gen) {
  auto it = records_.find(id);
  if (it == records_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  if (it->second.generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on pin");
  auto r = arena_.pin(id, gen);
  if (!r.ok()) return r.error();
  auto& rec = it->second;
  rec.movability = Movability::PINNED;
  rec.pin_count = r->pin_count;
  rec.lifecycle = LifecycleState::PINNED;
  domain_.current_movable = sub(domain_.current_movable, rec.region_size);
  domain_.current_pinned = sub(domain_.current_pinned, 0);
  domain_.current_pinned += rec.region_size;
  return Result<void>();
}

Result<void> MemoryGovernor::unpin(AllocationId id, AllocationGeneration gen) {
  std::lock_guard<std::mutex> lk(mutex_);
  return unpin_locked(id, gen);
}

Result<void> MemoryGovernor::unpin_locked(AllocationId id, AllocationGeneration gen) {
  auto it = records_.find(id);
  if (it == records_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  if (it->second.generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on unpin");
  auto r = arena_.unpin(id, gen);
  if (!r.ok()) return r.error();
  auto& rec = it->second;
  rec.movability = r->movability;
  rec.pin_count = r->pin_count;
  rec.lifecycle = (r->pin_count == 0) ? LifecycleState::RESIDENT : LifecycleState::PINNED;
  if (rec.pin_count == 0) {
    domain_.current_pinned = sub(domain_.current_pinned, rec.region_size);
    domain_.current_movable += rec.region_size;
  }
  return Result<void>();
}

Result<AllocationRecord> MemoryGovernor::get(AllocationId id, AllocationGeneration gen) const {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = records_.find(id);
  if (it == records_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  if (it->second.generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale allocation generation on query");
  auto r = arena_.get(id, gen);
  if (!r.ok()) return r.error();
  return it->second;
}

std::uint64_t MemoryGovernor::reclaimable_bytes_locked() const {
  std::uint64_t total = 0;
  for (const auto& kv : records_) {
    const auto& rec = kv.second;
    const bool pinned = rec.pin_count > 0 || rec.movability == Movability::PINNED || rec.movability == Movability::NONMOVABLE;
    const bool never = rec.reclaimability == Reclaimability::NEVER_RECLAIM || rec.reclaimability == Reclaimability::UNKNOWN;
    const bool active = rec.residency == ResidencyState::ACTIVE;
    if (!pinned && !never && !active) total += rec.region_size;
  }
  return total;
}

PressureState MemoryGovernor::pressure_locked() const {
  PressureEvidence ev;
  ev.managed_capacity = domain_.managed_capacity;
  ev.committed = arena_.live_bytes();
  ev.free = arena_.free_bytes();
  ev.reclaimable = reclaimable_bytes_locked();
  ev.largest_free = arena_.largest_free_region();
  ev.fragmentation = arena_.external_fragmentation();
  ev.watermarks = domain_.watermarks;
  const PressureState p = evaluate_pressure(ev, domain_.pressure);
  domain_.pressure = p;
  return p;
}

MemorySummary MemoryGovernor::summary_locked() const {
  MemorySummary s;
  s.managed_capacity = domain_.managed_capacity;
  s.free = arena_.free_bytes();
  s.committed = arena_.live_bytes();
  s.resident = arena_.live_bytes();
  s.reclaimable = reclaimable_bytes_locked();
  s.movable = arena_.movable_bytes();
  s.pinned = arena_.pinned_bytes();
  s.largest_free = arena_.largest_free_region();
  s.free_regions = arena_.free_region_count();
  s.live_allocations = arena_.live_region_count();
  s.external_fragmentation = arena_.external_fragmentation();
  s.provenance = domain_.provenance;
  s.pressure = pressure_locked();
  s.domain_generation = domain_.domain_generation;
  return s;
}

MemorySummary MemoryGovernor::summary() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return summary_locked();
}

PressureState MemoryGovernor::pressure() const {
  std::lock_guard<std::mutex> lk(mutex_);
  return pressure_locked();
}

std::vector<AllocationRecord> MemoryGovernor::allocations() const {
  std::lock_guard<std::mutex> lk(mutex_);
  std::vector<AllocationRecord> v;
  v.reserve(records_.size());
  for (const auto& kv : records_) v.push_back(kv.second);
  return v;
}

std::vector<AllocationId> MemoryGovernor::reclaim_candidates(std::size_t max) const {
  std::lock_guard<std::mutex> lk(mutex_);
  struct Cand { AllocationId id; int pri; std::uint64_t size; };
  std::vector<Cand> c;
  for (const auto& kv : records_) {
    const auto& rec = kv.second;
    const bool pinned = rec.pin_count > 0 || rec.movability == Movability::PINNED || rec.movability == Movability::NONMOVABLE;
    const bool never = rec.reclaimability == Reclaimability::NEVER_RECLAIM || rec.reclaimability == Reclaimability::UNKNOWN;
    const bool active = rec.residency == ResidencyState::ACTIVE;
    const bool needs_back = reclaimability_requires_backing(rec.reclaimability);
    if (pinned || never || active) continue;
    if (needs_back && !rec.recoverable && rec.backing_ref.empty()) continue;
    c.push_back({rec.id, reclaimability_priority(rec.reclaimability), rec.region_size});
  }
  std::stable_sort(c.begin(), c.end(), [](const Cand& a, const Cand& b) {
    if (a.pri != b.pri) return a.pri < b.pri;
    if (a.size != b.size) return a.size > b.size;
    return a.id.value() < b.id.value();
  });
  std::vector<AllocationId> out;
  for (std::size_t i = 0; i < c.size() && i < max; ++i) out.push_back(c[i].id);
  return out;
}

MemoryGovernor::CompactionPlan MemoryGovernor::plan_compaction() const {
  std::lock_guard<std::mutex> lk(mutex_);
  CompactionPlan plan;
  const std::vector<RegionInfo> live = arena_.snapshot_live();
  const std::vector<RegionInfo> free = arena_.snapshot_free();
  plan.largest_free_before = arena_.largest_free_region();
  plan.largest_free_after = plan.largest_free_before;

  // For each movable, non-pinned, non-active allocation, find a target free
  // region (not its immediate left/right neighbor) and estimate best gain.
  double best_gain = -1.0;
  for (const auto& m : live) {
    if (!m.live) continue;
    if (m.movability != Movability::MOVABLE || m.pin_count > 0) continue;
    if (m.residency == ResidencyState::ACTIVE) continue;
    if (m.reclaimability == Reclaimability::NEVER_RECLAIM) continue;

    // immediate free neighbors
    std::uint64_t lft = 0, rgt = 0;
    for (const auto& f : free) {
      if (f.offset + f.size == m.offset) lft = f.size;
      if (f.offset == m.offset + m.size) rgt = f.size;
    }

    // find a target free region distinct from the two neighbors, big enough
    for (const auto& f : free) {
      if ((f.offset + f.size == m.offset) || (f.offset == m.offset + m.size)) continue;
      if (f.size < m.size) continue;
      const std::uint64_t nf = detail::align_up(f.offset, m.alignment);
      const std::uint64_t front = nf >= f.offset ? nf - f.offset : 0;
      if (front > f.size || m.size > f.size - front) continue;

      // Compute the true post-move largest free region:
      //   - the target T is split into a front pad and a tail
      //   - the moved allocation's old span merges with its left/right frees
      std::uint64_t after = plan.largest_free_before;
      std::uint64_t front_pad_free = front;
      std::uint64_t tail_free = f.size - front - m.size;
      std::uint64_t consolidated = lft + m.size + rgt;
      after = std::max(after, consolidated);
      if (front_pad_free > 0) after = std::max(after, front_pad_free);
      if (tail_free > 0) after = std::max(after, tail_free);

      const double gain = static_cast<double>(after) - static_cast<double>(plan.largest_free_before);
      if (gain > best_gain) {
        best_gain = gain;
        plan.steps.clear();
        plan.steps.push_back({m.alloc_id, m.alloc_generation, m.offset, m.size,
                              nf, m.size, Movability::MOVABLE});
        plan.bytes_moved = m.size;
        plan.largest_free_after = after;
      }
    }
  }
  if (plan.steps.empty()) {
    plan.rationale = "no safe compaction found (all movable allocations are pinned, active, or lack a free target)";
  } else {
    plan.rationale = "deterministic single-move compaction: " + std::to_string(plan.bytes_moved) + " bytes moved";
  }
  return plan;
}

std::uint64_t MemoryGovernor::region_size(AllocationId id) const {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = records_.find(id);
  if (it != records_.end()) return it->second.region_size;
  auto rit = retired_.find(id);
  if (rit != retired_.end()) return rit->second.region_size;
  return 0;
}

Result<MemoryGovernor::AllocationHandle> MemoryGovernor::apply_relocation(
    AllocationId id, AllocationGeneration gen, std::uint64_t new_offset, std::uint64_t new_size) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto it = records_.find(id);
  if (it == records_.end()) return Failure(RejectReason::STALE_GENERATION, "unknown allocation");
  if (it->second.generation != gen) return Failure(RejectReason::STALE_GENERATION, "stale generation on relocate");
  if (it->second.pin_count > 0) return Failure(RejectReason::BLOCKED_BY_PINNED, "cannot relocate a pinned allocation");
  auto r = arena_.relocate(id, gen, new_offset, new_size, it->second.alignment);
  if (!r.ok()) return r.error();
  auto& rec = it->second;
  rec.offset = r->offset;
  rec.region_size = r->size;
  rec.region_generation = r->region_generation;
  rec.generation = r->alloc_generation;
  rec.relocation_generation = rec.relocation_generation.is_set() ? rec.relocation_generation.next()
                                                                 : RelocationGeneration::first();
  rec.lifecycle = LifecycleState::RESIDENT;
  return AllocationHandle{id, rec.generation, r->offset, r->size, rec.alignment};
}

std::string MemoryGovernor::explain() const {
  std::lock_guard<std::mutex> lk(mutex_);
  const MemorySummary s = summary_locked();
  std::ostringstream os;
  os << "domain=" << domain_.domain_id << " gen=" << domain_.domain_generation
     << " device=" << domain_.device_id
     << " managed=" << s.managed_capacity
     << " committed=" << s.committed
     << " free=" << s.free
     << " largest_free=" << s.largest_free
     << " free_regions=" << s.free_regions
     << " live=" << s.live_allocations
     << " reclaimable=" << s.reclaimable
     << " movable=" << s.movable
     << " pinned=" << s.pinned
     << " frag=" << s.external_fragmentation
     << " pressure=" << s.pressure << "\n";
  return os.str();
}

persist::PersistSnapshot MemoryGovernor::snapshot() const {
  std::lock_guard<std::mutex> lk(mutex_);
  persist::PersistSnapshot s;
  s.domain = domain_;
  s.records.reserve(records_.size());
  for (const auto& kv : records_) s.records.push_back(kv.second);
  s.retired.reserve(retired_.size());
  for (const auto& kv : retired_) s.retired.push_back(kv.second);
  s.next_alloc_id = next_alloc_id_;
  s.history_count = history_count_;
  return s;
}

Result<void> MemoryGovernor::restore(const persist::PersistSnapshot& s) {
  std::lock_guard<std::mutex> lk(mutex_);
  // Conservative recovery: old process-owned physical allocations are NOT
  // silently current.  Mark every surviving record STALE / REVALIDATION_REQUIRED.
  domain_ = s.domain;
  records_.clear();
  retired_.clear();
  next_alloc_id_ = s.next_alloc_id;
  history_count_ = s.history_count;
  for (const auto& rec : s.records) {
    AllocationRecord r = rec;
    r.lifecycle = LifecycleState::REVALIDATION_REQUIRED;
    r.residency = ResidencyState::REVALIDATION_REQUIRED;
    retired_[r.id] = r;
  }
  for (const auto& rec : s.retired) {
    retired_[rec.id] = rec;
  }
  // Rebuild arena empty (fresh process owns no old physical allocations).
  arena_ = RegionAllocator(domain_.managed_capacity);
  domain_.current_committed = 0;
  domain_.current_resident = 0;
  domain_.current_movable = 0;
  domain_.current_pinned = 0;
  initialized_ = false;
  domain_.health = DeviceHealth::UNKNOWN;
  domain_.pressure = PressureState::REVALIDATION_REQUIRED;
  return Result<void>();
}

Result<void> MemoryGovernor::save(const std::string& path) const {
  return persist::write_file(path, snapshot());
}

Result<void> MemoryGovernor::load(const std::string& path) {
  auto snap = persist::read_file(path);
  if (!snap.ok()) return snap.error();
  return restore(*snap);
}

}  // namespace gpu_memory_service
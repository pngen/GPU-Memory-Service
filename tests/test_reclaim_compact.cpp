#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <cstdio>

using namespace gpu_memory_service;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

static MemoryDomain make_domain(std::uint64_t cap) {
  MemoryDomain d; d.domain_id = GpuMemoryDomainId(4); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED;
  d.watermarks.low = 0.3; d.watermarks.high = 0.7; d.watermarks.critical = 0.95;
  return d;
}

int main() {
  MemoryGovernor gov(make_domain(1u << 20), std::make_unique<HostBackend>());
  check(gov.initialize().ok(), "init");

  auto req = [&](std::uint64_t bytes, Reclaimability rec, Movability mov, bool pin) {
    AllocationRequest r; r.bytes = bytes; r.alignment = 4096; r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = rec; r.movability = mov; r.requires_pin = pin; r.owner = "t";
    return gov.request(r);
  };

  // Six blocks of 128 KiB; pin K1; free K2 and K5 to create an island K6
  // flanked by two free regions and with a third free (K2 hole) as target.
  auto K1 = req(128 * 1024, Reclaimability::BEST_EFFORT, Movability::PINNED, false); check(K1.ok(), "K1");
  auto K2 = req(128 * 1024, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE, false); check(K2.ok(), "K2");
  auto K3 = req(128 * 1024, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE, false); check(K3.ok(), "K3");
  auto K4 = req(128 * 1024, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE, false); check(K4.ok(), "K4");
  auto K5 = req(128 * 1024, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE, false); check(K5.ok(), "K5");
  auto K6 = req(128 * 1024, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE, false); check(K6.ok(), "K6");

  check(gov.pin(K1->id, K1->generation).ok(), "pin K1");
  check(gov.release(K2->id, K2->generation).ok(), "free K2");
  check(gov.release(K5->id, K5->generation).ok(), "free K5");

  // Reclaim candidates exclude K1 (pinned); include K3, K4, K6.
  auto cands = gov.reclaim_candidates(100);
  bool hasK1 = false, hasK3 = false, hasK6 = false;
  for (auto id : cands) { if (id == K1->id) hasK1 = true; if (id == K3->id) hasK3 = true; if (id == K6->id) hasK6 = true; }
  check(!hasK1, "pinned excluded from reclaim");
  check(hasK3 && hasK6, "movable reclaimable included");

  // Release of pinned K1 is rejected.
  check(!gov.release(K1->id, K1->generation).ok(), "release pinned rejected");

  // Deterministic compaction plan.
  const std::uint64_t lf_before = gov.summary().largest_free;
  auto plan1 = gov.plan_compaction();
  auto plan2 = gov.plan_compaction();
  check(plan1.steps.size() >= 1, "compaction plan non-empty");
  check(plan1.steps.size() == plan2.steps.size() && plan1.steps[0].id == plan2.steps[0].id, "plan deterministic");
  check(plan1.largest_free_after > lf_before, "compaction improves largest free");

  if (!plan1.steps.empty()) {
    const auto& st = plan1.steps[0];
    check(st.movability == Movability::MOVABLE, "plan moves movable only");
    check(st.id != K1->id, "plan excludes pinned K1");
    auto rel = gov.apply_relocation(st.id, st.generation, st.new_offset, st.new_size);
    check(rel.ok(), "apply relocation ok");
    check(rel->generation != st.generation, "relocation advances generation");
    check(!gov.get(st.id, st.generation).ok(), "old handle stale after relocate");
    check(gov.get(st.id, rel->generation).ok(), "new handle resolves");
    check(gov.region_size(st.id) == st.new_size, "region size preserved");
  }

  const auto after = gov.summary();
  check(after.largest_free > lf_before, "largest free increased");
  check(after.committed == 5 * 128 * 1024 - 128 * 1024, "committed correct");

  if (failures == 0) { std::printf("reclaim_compact test: PASS\n"); return 0; }
  std::printf("reclaim_compact test: %d FAILURES\n", failures);
  return 1;
}
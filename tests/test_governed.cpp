#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <cstdio>

using namespace gpu_memory_service;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

static MemoryDomain make_domain(std::uint64_t cap) {
  MemoryDomain d;
  d.domain_id = GpuMemoryDomainId(7);
  d.domain_generation = GpuMemoryDomainGeneration(3);
  d.device_id = DeviceId(0);
  d.device_generation = DeviceGeneration(2);
  d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED;
  d.watermarks.low = 0.5; d.watermarks.high = 0.8; d.watermarks.critical = 0.95;
  return d;
}

int main() {
  MemoryGovernor gov(make_domain(1u << 20), std::make_unique<HostBackend>());
  check(gov.initialize().ok(), "initialize ok");

  AllocationRequest r;
  r.bytes = 256 * 1024;
  r.alignment = 1024;
  r.mem_class = MemoryClass::USER_BUFFER;
  r.reclaimability = Reclaimability::RECLAIM_IF_COLD;
  r.movability = Movability::MOVABLE;
  r.owner = "client-a";

  auto h = gov.request(r);
  check(h.ok(), "request ok");
  check(h->size == 256 * 1024, "size exact");
  check(h->offset % 1024 == 0, "aligned");
  check(h->generation == AllocationGeneration::first(), "gen first");

  // stale query rejects
  auto stale = gov.get(h->id, AllocationGeneration(999));
  check(!stale.ok(), "stale get rejects");
  auto cur = gov.get(h->id, h->generation);
  check(cur.ok(), "current get ok");

  // release then duplicate free rejects
  auto rel = gov.release(h->id, h->generation);
  check(rel.ok(), "release ok");
  auto dup = gov.release(h->id, h->generation);
  check(!dup.ok(), "duplicate release rejects");
  check(dup.error().reason == RejectReason::DOUBLE_FREE, "duplicate is double free");
  auto staleRel = gov.release(h->id, AllocationGeneration(5));
  check(!staleRel.ok(), "stale release rejects");

  auto s = gov.summary();
  check(s.committed == 0, "committed back to zero");
  check(s.free == s.managed_capacity, "free back to managed");

  // OOM by request > capacity
  AllocationRequest big;
  big.bytes = (1u << 20) + 1;
  big.alignment = 64;
  auto b = gov.request(big);
  check(!b.ok(), "over-capacity request rejects");

  // zero bytes rejects
  AllocationRequest z; z.bytes = 0; z.alignment = 64;
  check(!gov.request(z).ok(), "zero-byte rejects");

  if (failures == 0) { std::printf("governed test: PASS\n"); return 0; }
  std::printf("governed test: %d FAILURES\n", failures);
  return 1;
}

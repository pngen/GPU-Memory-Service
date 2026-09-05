#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <cstdio>

using namespace gpu_memory_service;

int main() {
  MemoryDomain d;
  d.domain_id = GpuMemoryDomainId(1);
  d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0);
  d.device_generation = DeviceGeneration(1);
  d.managed_capacity = 4u << 20;
  d.provenance = Provenance::MEASURED;

  MemoryGovernor gov(d, std::make_unique<HostBackend>());
  if (!gov.initialize().ok()) { std::printf("init failed\n"); return 1; }

  AllocationRequest r;
  r.bytes = 2u << 20;
  r.alignment = 4096;
  r.mem_class = MemoryClass::USER_BUFFER;
  r.reclaimability = Reclaimability::RECLAIM_IF_COLD;
  r.owner = "consumer";
  auto h = gov.request(r);
  if (!h.ok()) { std::printf("request failed: %s\n", to_string(h.error().reason)); return 1; }

  auto s = gov.summary();
  std::printf("requested=%llu committed=%llu free=%llu largest=%llu pressure=%s\n",
              static_cast<unsigned long long>(h->size),
              static_cast<unsigned long long>(s.committed),
              static_cast<unsigned long long>(s.free),
              static_cast<unsigned long long>(s.largest_free),
              to_string(s.pressure));

  if (s.committed != (2u << 20)) { std::printf("committed mismatch\n"); return 1; }

  auto rel = gov.release(h->id, h->generation);
  if (!rel.ok()) { std::printf("release failed\n"); return 1; }
  const auto s2 = gov.summary();
  if (s2.committed != 0 || s2.free != s2.managed_capacity) { std::printf("accounting mismatch after release\n"); return 1; }
  std::printf("consumer: PASS\n");
  return 0;
}

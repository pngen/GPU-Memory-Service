#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <cstdio>
using namespace gpu_memory_service;
int main() {
  MemoryDomain d; d.domain_id = GpuMemoryDomainId(7); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = 8u << 20;
  d.provenance = Provenance::MEASURED;
  MemoryGovernor gov(d, std::make_unique<HostBackend>());
  (void)gov.initialize().ok();
  AllocationRequest r; r.bytes = 2u << 20; r.alignment = 4096; r.mem_class = MemoryClass::USER_BUFFER;
  r.reclaimability = Reclaimability::RECLAIM_IF_COLD; r.owner = "example";
  auto h = gov.request(r);
  std::printf("alloc ok=%d size=%u\n", h.ok() ? 1 : 0, static_cast<unsigned>(h->size));
  auto s = gov.summary();
  std::printf("committed=%u free=%u pressure=%s\n", static_cast<unsigned>(s.committed),
              static_cast<unsigned>(s.free), to_string(s.pressure));
  if (h.ok()) { auto rel = gov.release(h->id, h->generation); std::printf("release ok=%d\n", rel.ok() ? 1 : 0); }
  return 0;
}
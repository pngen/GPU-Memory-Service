#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <cstdio>
using namespace gpu_memory_service;
int main() {
  const char* path = "gms_recovery_example.bin";
  { MemoryDomain d; d.domain_id = GpuMemoryDomainId(9); d.domain_generation = GpuMemoryDomainGeneration(1);
    d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = 4u << 20;
    d.provenance = Provenance::MEASURED;
    MemoryGovernor gov(d, std::make_unique<HostBackend>()); (void)gov.initialize().ok();
    AllocationRequest r; r.bytes = 1u << 20; r.alignment = 4096; r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = Reclaimability::RECLAIM_IF_COLD; r.owner = "recovery";
    auto h = gov.request(r);
    std::printf("saved, committed=%u\n", static_cast<unsigned>(gov.summary().committed));
    (void)gov.save(path).ok();
  }
  { MemoryDomain d; d.domain_id = GpuMemoryDomainId(9); d.domain_generation = GpuMemoryDomainGeneration(1);
    d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = 4u << 20;
    d.provenance = Provenance::MEASURED;
    MemoryGovernor gov(d, std::make_unique<HostBackend>());
    (void)gov.load(path).ok();
    std::printf("recovered committed=%u (conservative, physical bytes are not silently current)\n",
                static_cast<unsigned>(gov.summary().committed));
  }
  std::remove(path);
  return 0;
}
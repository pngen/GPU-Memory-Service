#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
using namespace gpu_memory_service;

int main() {
  const std::uint64_t cap = 1ull << 26;
  MemoryDomain d; d.domain_id = GpuMemoryDomainId(1); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED;
  MemoryGovernor gov(d, std::make_unique<HostBackend>());
  (void)gov.initialize().ok();

  auto bench = [&](std::uint64_t n, std::uint64_t bytes, std::uint64_t align) {
    std::vector<MemoryGovernor::AllocationHandle> v;
    v.reserve(n);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < n; ++i) {
      AllocationRequest r; r.bytes = bytes; r.alignment = align; r.mem_class = MemoryClass::USER_BUFFER;
      r.reclaimability = Reclaimability::RECLAIM_IF_COLD; r.owner = "bench";
      auto h = gov.request(r);
      if (!h.ok()) { std::printf("  unexpected failure at %llu: %s\n", static_cast<unsigned long long>(i), to_string(h.error().reason)); return; }
      v.push_back(*h);
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (auto& h : v) { auto rx = gov.release(h.id, h.generation); (void)rx; }
    const auto t2 = std::chrono::steady_clock::now();
    const double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double free_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double alloc_rate = alloc_ms > 0 ? static_cast<double>(n) / (alloc_ms / 1000.0) : 0.0;
    const double free_rate = free_ms > 0 ? static_cast<double>(n) / (free_ms / 1000.0) : 0.0;
    std::printf("n=%llu bytes=%llu align=%llu  alloc=%.2f ms (%.0f ops/s)  free=%.2f ms (%.0f ops/s)\n",
                static_cast<unsigned long long>(n), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(align), alloc_ms, alloc_rate, free_ms, free_rate);
  };

  std::printf("--- allocation/free benchmark (governed, host backend) ---\n");
  bench(100, 64 * 1024, 256);
  bench(1000, 16 * 1024, 256);
  bench(10000, 4096, 256);
  bench(100000, 512, 64);
  return 0;
}
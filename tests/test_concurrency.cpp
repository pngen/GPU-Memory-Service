#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using namespace gpu_memory_service;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

static MemoryDomain make_domain(std::uint64_t cap) {
  MemoryDomain d; d.domain_id = GpuMemoryDomainId(11); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED; return d;
}

int main() {
  const std::uint64_t cap = 8u << 20;
  MemoryGovernor gov(make_domain(cap), std::make_unique<HostBackend>());
  check(gov.initialize().ok(), "init");

  const int nthreads = 8;
  const int iters = 300;
  std::atomic<bool> start{false};
  std::atomic<std::uint64_t> errors{0};
  std::vector<std::thread> threads;

  // Writers: allocate and free their own small allocations.
  for (int t = 0; t < nthreads; ++t) {
    threads.emplace_back([&, t]() {
      while (!start.load(std::memory_order_acquire)) {}
      std::vector<MemoryGovernor::AllocationHandle> mine;
      for (int i = 0; i < iters; ++i) {
        AllocationRequest r; r.bytes = 4096 + (t % 3) * 1024; r.alignment = 4096;
        r.mem_class = MemoryClass::USER_BUFFER; r.reclaimability = Reclaimability::RECLAIM_IF_COLD;
        auto h = gov.request(r);
        if (h.ok()) mine.push_back(*h);
        if (mine.size() > 4) {
          auto id = mine.back(); mine.pop_back();
          auto rel = gov.release(id.id, id.generation);
          if (!rel.ok()) errors.fetch_add(1);
        }
      }
      for (auto& a : mine) { auto rel = gov.release(a.id, a.generation); if (!rel.ok()) errors.fetch_add(1); }
    });
  }

  // Reader thread doing heavily concurrent summary queries.
  std::atomic<std::uint64_t> checksum{0};
  threads.emplace_back([&]() {
    while (!start.load(std::memory_order_acquire)) {}
    for (int i = 0; i < iters * 4; ++i) {
      auto s = gov.summary();
      auto cs = gov.reclaim_candidates(4);
      auto pl = gov.plan_compaction();
      auto ex = gov.explain();
      checksum.fetch_add(s.committed + s.free + cs.size() + pl.steps.size() + ex.size());
    }
  });

  start.store(true, std::memory_order_release);
  for (auto& th : threads) th.join();

  check(errors.load() == 0, "no release errors under contention");

  // Drain all remaining live allocations.
  for (const auto& rec : gov.allocations()) {
    auto rr = gov.release(rec.id, rec.generation);
    (void)rr;
  }
  auto s = gov.summary();
  check(s.committed == 0, "committed drained to zero");
  check(s.free == s.managed_capacity, "free back to capacity");

  if (failures == 0) { std::printf("concurrency test: PASS\n"); return 0; }
  std::printf("concurrency test: %d FAILURES\n", failures);
  return 1;
}
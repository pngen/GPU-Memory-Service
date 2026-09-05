#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace gpu_memory_service;

static std::uint64_t parse_cap(const char* s) {
  std::string t = s;
  std::uint64_t mult = 1;
  if (!t.empty() && t.back() == 'M') { mult = 1024ull * 1024; t.pop_back(); }
  else if (!t.empty() && t.back() == 'G') { mult = 1024ull * 1024 * 1024; t.pop_back(); }
  else if (!t.empty() && t.back() == 'K') { mult = 1024ull; t.pop_back(); }
  return std::strtoull(t.c_str(), nullptr, 10) * mult;
}

int main(int argc, char** argv) {
  std::uint64_t cap = 16u << 20;
  bool summary = false, dump = false;
  std::string alloc_spec;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--cap" && i + 1 < argc) cap = parse_cap(argv[++i]);
    else if (a == "--summary") summary = true;
    else if (a == "--dump") dump = true;
    else if (a == "--alloc" && i + 1 < argc) alloc_spec = argv[++i];
  }

  MemoryDomain d; d.domain_id = GpuMemoryDomainId(1); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED;
  MemoryGovernor gov(d, std::make_unique<HostBackend>());
  if (!gov.initialize().ok()) { std::printf("init failed\n"); return 1; }

  std::vector<MemoryGovernor::AllocationHandle> handles;
  if (!alloc_spec.empty()) {
    const std::size_t comma = alloc_spec.find(',');
    const std::uint64_t bytes = parse_cap(alloc_spec.substr(0, comma).c_str());
    const std::uint64_t align = comma != std::string::npos ? std::strtoull(alloc_spec.c_str() + comma + 1, nullptr, 10) : 4096;
    AllocationRequest r; r.bytes = bytes; r.alignment = align; r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = Reclaimability::RECLAIM_IF_COLD; r.owner = "cli";
    auto h = gov.request(r);
    if (!h.ok()) { std::printf("request failed: %s\n", to_string(h.error().reason)); return 1; }
    handles.push_back(*h);
    std::printf("allocated id=%llu gen=%llu off=%llu size=%llu align=%llu\n",
                static_cast<unsigned long long>(h->id.value()),
                static_cast<unsigned long long>(h->generation.value()),
                static_cast<unsigned long long>(h->offset),
                static_cast<unsigned long long>(h->size),
                static_cast<unsigned long long>(h->alignment));
  }

  if (summary) {
    auto s = gov.summary();
    std::printf("summary: managed=%llu free=%llu committed=%llu largest=%llu free_regions=%zu live=%zu frag=%.3f pressure=%s\n",
                static_cast<unsigned long long>(s.managed_capacity),
                static_cast<unsigned long long>(s.free),
                static_cast<unsigned long long>(s.committed),
                static_cast<unsigned long long>(s.largest_free),
                s.free_regions, s.live_allocations, s.external_fragmentation, to_string(s.pressure));
  }
  if (dump) {
    for (const auto& a : gov.allocations()) {
      std::printf("  alloc id=%llu gen=%llu off=%llu size=%llu pin=%u mov=%s class=%s owner=%s\n",
                  static_cast<unsigned long long>(a.id.value()),
                  static_cast<unsigned long long>(a.generation.value()),
                  static_cast<unsigned long long>(a.offset),
                  static_cast<unsigned long long>(a.region_size),
                  a.pin_count, to_string(a.movability), to_string(a.mem_class), a.owner.c_str());
    }
  }
  for (auto& h : handles) { auto rx = gov.release(h.id, h.generation); (void)rx; }
  return 0;
}
#include "gpu_memory_service/region_allocator.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

using namespace gpu_memory_service;

// Deterministic xorshift64* PRNG.
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  std::uint64_t next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 0x2545F4914F6CDD1DULL; }
  std::uint64_t below(std::uint64_t n) { return n == 0 ? 0 : next() % n; }
};

int main(int argc, char** argv) {
  std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 0x1234567890ABCDEFULL;
  Rng rng(seed);
  const std::uint64_t capacity = 1u << 20;  // 1 MiB
  RegionAllocator ra(capacity);
  std::vector<std::uint8_t> occ(capacity, 0);   // slow occupancy reference
  std::uint64_t live_ref = 0;
  std::set<std::uint64_t> live_ids;
  std::uint64_t next_id = 1;

  const std::uint64_t aligns[] = {1, 8, 64, 256, 1024, 4096};
  const std::uint64_t n_aligns = sizeof(aligns) / sizeof(aligns[0]);
  auto fail = [&](int step, const char* msg) {
    std::printf("PROPERTY FAIL at step %d: %s\nseed=%llu\n", step, msg,
                static_cast<unsigned long long>(seed));
    return 1;
  };

  for (int step = 0; step < 4000; ++step) {
    if (rng.below(100) < 72) {
      // allocate
      const std::uint64_t bytes = rng.below(48 * 1024) + 1;
      const std::uint64_t align = aligns[rng.below(n_aligns)];
      const std::uint64_t id = next_id++;
      AllocRequest req;
      req.bytes = bytes; req.alignment = align; req.alloc_id = AllocationId(id);
      req.alloc_generation = AllocationGeneration(1);
      req.movability = (rng.below(10) < 2) ? Movability::MOVABLE : Movability::MOVABLE;
      auto r = ra.allocate(req);
      if (r.ok()) {
        for (std::uint64_t i = 0; i < r->size; ++i) {
          if (occ[r->offset + i] != 0) return fail(step, "allocate into occupied byte");
          occ[r->offset + i] = 1;
        }
        live_ref += r->size;
        live_ids.insert(id);
      }
    } else {
      // free a random live id (or stale-gen free)
      if (!live_ids.empty()) {
        auto it = live_ids.begin();
        std::advance(it, rng.below(static_cast<std::uint64_t>(live_ids.size())));
        const std::uint64_t id = *it;
        if (rng.below(100) < 20) {
          // stale-generation free must not mutate
          auto before = ra.free_bytes();
          auto sf = ra.free(AllocationId(id), AllocationGeneration(999));
          if (sf.ok()) return fail(step, "stale free unexpectedly succeeded");
          if (ra.free_bytes() != before) return fail(step, "stale free mutated state");
        } else {
          auto g = ra.get(AllocationId(id), AllocationGeneration(1));
          std::uint64_t off = 0, sz = 0;
          if (g.ok()) { off = g->offset; sz = g->size; }
          auto fr = ra.free(AllocationId(id), AllocationGeneration(1));
          if (fr.ok()) {
            for (std::uint64_t i = 0; i < sz; ++i) occ[off + i] = 0;
            live_ref -= sz;
            live_ids.erase(it);
          }
        }
      }
    }

    // invariants every step
    const std::string v = ra.verify();
    if (!v.empty()) return fail(step, v.c_str());
    if (ra.live_bytes() != live_ref) return fail(step, "live_bytes != reference");
    if (ra.free_bytes() != capacity - live_ref) return fail(step, "free_bytes != capacity - live");
    if (ra.largest_free_region() > ra.free_bytes()) return fail(step, "largest_free > free_bytes");
    if (ra.live_region_count() != live_ids.size()) return fail(step, "live_count != reference");
    // no live region overlaps reference (already guaranteed by occ marking)
    for (const auto& lr : ra.snapshot_live()) {
      std::uint64_t cnt = 0;
      for (std::uint64_t i = 0; i < lr.size; ++i) cnt += occ[lr.offset + i];
      if (cnt != lr.size) return fail(step, "live region not fully occupied");
    }
  }
  std::printf("property test: PASS (seed=%llu)\n", static_cast<unsigned long long>(seed));
  return 0;
}

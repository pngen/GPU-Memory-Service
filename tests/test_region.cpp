#include "gpu_memory_service/region_allocator.hpp"
#include <cstdio>
#include <cstdint>

using namespace gpu_memory_service;

static int failures = 0;
static void check(bool cond, const char* msg) {
  if (!cond) { std::printf("FAIL: %s\n", msg); ++failures; }
}

int main() {
  RegionAllocator ra(1024 * 1024);
  check(ra.capacity() == 1024 * 1024, "capacity");
  check(ra.free_bytes() == 1024 * 1024, "initial free");
  check(ra.live_region_count() == 0, "initial live count");
  check(ra.largest_free_region() == 1024 * 1024, "initial largest free");
  check(ra.verify().empty(), "initial verify");

  auto mk = [](std::uint64_t id, std::uint64_t bytes, std::uint64_t align) {
    AllocRequest r;
    r.bytes = bytes; r.alignment = align;
    r.alloc_id = AllocationId(id);
    r.alloc_generation = AllocationGeneration(1);
    r.movability = Movability::MOVABLE;
    r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = Reclaimability::RECLAIM_IF_COLD;
    return r;
  };

  auto a = ra.allocate(mk(1, 128 * 1024, 256));
  check(a.ok(), "alloc1 ok");
  auto b = ra.allocate(mk(2, 64 * 1024, 1024));
  check(b.ok(), "alloc2 ok");
  auto c = ra.allocate(mk(3, 256 * 1024, 4096));
  check(c.ok(), "alloc3 ok");
  check(ra.live_region_count() == 3, "live count 3");
  check(ra.verify().empty(), "verify after 3 allocs");

  check(a->offset % 256 == 0, "a aligned");
  check(b->offset % 1024 == 0, "b aligned");
  check(c->offset % 4096 == 0, "c aligned");
  check(a->size == 128 * 1024 && b->size == 64 * 1024 && c->size == 256 * 1024, "sizes");

  auto fb = ra.free(AllocationId(2), AllocationGeneration(1));
  check(fb.ok(), "free b ok");
  check(ra.live_region_count() == 2, "live count 2 after free");
  check(ra.verify().empty(), "verify after free b");

  auto df = ra.free(AllocationId(2), AllocationGeneration(1));
  check(!df.ok(), "double free rejected");
  check(df.error().reason == RejectReason::STALE_GENERATION, "double free reason");

  auto sf = ra.free(AllocationId(1), AllocationGeneration(99));
  check(!sf.ok(), "stale free rejected");

  auto q = ra.get(AllocationId(1), AllocationGeneration(77));
  check(!q.ok(), "stale query rejected");
  auto q0 = ra.get(AllocationId(1), AllocationGeneration(1));
  check(q0.ok(), "current query ok");

  std::uint64_t live = ra.live_bytes();
  check(live == 128 * 1024 + 256 * 1024, "live bytes exact");
  check(ra.free_bytes() == 1024 * 1024 - live, "free bytes exact");

  if (failures == 0) { std::printf("region test: PASS\n"); return 0; }
  std::printf("region test: %d FAILURES\n", failures);
  return 1;
}

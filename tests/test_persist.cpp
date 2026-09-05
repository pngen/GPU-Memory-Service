#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/host_backend.hpp"
#include "gpu_memory_service/persistence.hpp"
#include <cstdio>
#include <fstream>
#include <cstring>

using namespace gpu_memory_service;
using namespace gpu_memory_service::persist;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

static MemoryDomain make_domain(std::uint64_t cap) {
  MemoryDomain d; d.domain_id = GpuMemoryDomainId(9); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED; return d;
}

int main() {
  const std::string path = "gms_persist_test.bin";
  {
    MemoryGovernor gov(make_domain(1u << 20), std::make_unique<HostBackend>());
    check(gov.initialize().ok(), "init");
    AllocationRequest a; a.bytes = 128 * 1024; a.alignment = 4096; a.mem_class = MemoryClass::USER_BUFFER;
    a.reclaimability = Reclaimability::RECLAIM_IF_COLD; a.owner = "alice";
    auto h1 = gov.request(a); check(h1.ok(), "req1");
    AllocationRequest b; b.bytes = 64 * 1024; b.alignment = 1024; b.mem_class = MemoryClass::KV_CACHE;
    b.reclaimability = Reclaimability::RECLAIM_IF_CHECKPOINTED; b.backing_ref = "kv-bucket/123"; b.owner = "bob";
    auto h2 = gov.request(b); check(h2.ok(), "req2");
    check(gov.save(path).ok(), "save ok");
  }

  // Round-trip via encode/decode on the raw bytes.
  {
    PersistSnapshot snap;
    snap.domain = make_domain(1u << 20);
    AllocationRecord r; r.id = AllocationId(3); r.generation = AllocationGeneration(1);
    r.requested_bytes = 4096; r.region_size = 4096; r.offset = 0; r.alignment = 4096;
    r.mem_class = MemoryClass::USER_BUFFER; r.reclaimability = Reclaimability::RECLAIM_IF_COLD;
    r.movability = Movability::MOVABLE; r.residency = ResidencyState::RESIDENT; r.lifecycle = LifecycleState::RESIDENT;
    r.provenance = Provenance::MEASURED;
    snap.records.push_back(r);
    const auto bytes = encode(snap);
    auto dec = decode(bytes);
    check(dec.ok(), "decode ok");
    check(dec->records.size() == 1 && dec->records[0].id == AllocationId(3), "round-trip id");

    // checksum mismatch
    auto bad = bytes; bad[15] ^= 0xFF;
    check(!decode(bad).ok(), "checksum mismatch rejects");

    // magic
    auto bm = bytes; bm[0] = 0x00; bm[1] = 0x00; bm[2] = 0x00; bm[3] = 0x00;
    check(!decode(bm).ok(), "bad magic rejects");

    // version
    auto bv = bytes; bv[4] = 0xFF; bv[5] = 0xFF; bv[6] = 0xFF; bv[7] = 0xFF;
    check(!decode(bv).ok(), "bad version rejects");

    // truncation
    auto tr = std::vector<std::uint8_t>(bytes.begin(), bytes.end() - 8);
    check(!decode(tr).ok(), "truncation rejects");

    // trailing garbage
    auto tg = bytes; tg.push_back(0x42);
    check(!decode(tg).ok(), "trailing garbage rejects");

    // duplicate id
    PersistSnapshot dup = snap; dup.records.push_back(r);
    check(!decode(encode(dup)).ok(), "duplicate id rejects");

    // invalid enum (lifecycle out of range)
    PersistSnapshot se = snap; se.records[0].lifecycle = static_cast<LifecycleState>(200);
    check(!decode(encode(se)).ok(), "invalid enum rejects");

    // invalid alignment
    PersistSnapshot se2 = snap; se2.records[0].alignment = 3;
    check(!decode(encode(se2)).ok(), "invalid alignment rejects");

    // physical bytes > governed capacity
    PersistSnapshot se3 = snap; se3.domain.managed_capacity = 1024; se3.records[0].region_size = 4096;
    check(!decode(encode(se3)).ok(), "capacity exceed rejects");
  }

  // Conservative recovery: fresh governor loads but does not restore bytes as current.
  {
    MemoryGovernor gov(make_domain(1u << 20), std::make_unique<HostBackend>());
    auto lr = gov.load(path);
    check(lr.ok(), "load ok");
    auto s = gov.summary();
    check(s.committed == 0, "conservative recovery: committed 0");
    check(s.free == s.managed_capacity, "conservative recovery: arena empty");
    auto all = gov.allocations();
    check(all.empty(), "no live allocations after recovery");
  }

  std::remove(path.c_str());
  if (failures == 0) { std::printf("persist test: PASS\n"); return 0; }
  std::printf("persist test: %d FAILURES\n", failures);
  return 1;
}

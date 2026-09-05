#include "gpu_memory_service/governed_memory.hpp"
#include "gpu_memory_service/cuda/cuda_backend.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace gpu_memory_service;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

extern "C" int gms_cuda_fill(float* data, std::size_t n, float value, cudaStream_t s);

static MemoryDomain make_domain(std::uint64_t cap, double low, double high) {
  MemoryDomain d; d.domain_id = GpuMemoryDomainId(20); d.domain_generation = GpuMemoryDomainGeneration(1);
  d.device_id = DeviceId(0); d.device_generation = DeviceGeneration(1); d.managed_capacity = cap;
  d.provenance = Provenance::MEASURED;
  d.watermarks.low = low; d.watermarks.high = high; d.watermarks.critical = 0.95;
  return d;
}

static float* dev_at(MemoryGovernor& gov, std::uint64_t off) {
  return reinterpret_cast<float*>(static_cast<char*>(gov.arena_base()) + off);
}

static bool fill_pattern(MemoryGovernor& gov, const MemoryGovernor::AllocationHandle& h) {
  const std::size_t n = h.size / sizeof(float);
  if (gms_cuda_fill(dev_at(gov, h.offset), n, 7.0f, 0) != 0) return false;
  return cudaDeviceSynchronize() == cudaSuccess;
}

static bool check_pattern(MemoryGovernor& gov, const MemoryGovernor::AllocationHandle& h) {
  const std::size_t n = h.size / sizeof(float);
  std::vector<float> host(n);
  if (cudaMemcpy(host.data(), dev_at(gov, h.offset), h.size, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  for (std::size_t i = 0; i < n; ++i) if (host[i] != 7.0f + static_cast<float>(i)) return false;
  return true;
}

int main() {
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s  cc=%d.%d  total=%zuMB\n", prop.name, prop.major, prop.minor,
              static_cast<std::size_t>(prop.totalGlobalMem / (1024 * 1024)));
  check(prop.major >= 9, "Blackwell+ (sm_90+) expected");

  std::size_t free0 = 0, total0 = 0;
  cudaMemGetInfo(&free0, &total0);

  {
  // ---- Governor 1: allocation authority + fragmentation proof.
  MemoryGovernor gov(make_domain(64u << 20, 0.30, 0.70), std::make_unique<CudaBackend>());
  check(gov.initialize().ok(), "cuda arena init");

  auto req = [&](std::uint64_t bytes) {
    AllocationRequest r; r.bytes = bytes; r.alignment = 256; r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = Reclaimability::RECLAIM_IF_COLD; r.movability = Movability::MOVABLE; r.owner = "probe";
    return gov.request(r);
  };
  auto A = req(8u << 20); check(A.ok(), "alloc A");
  check(A->offset % 256 == 0, "A aligned");
  check(!gov.get(A->id, AllocationGeneration(777)).ok(), "stale A get rejects");
  check(fill_pattern(gov, *A) && check_pattern(gov, *A), "A H2D/kernel parity");
  check(!gov.release(A->id, AllocationGeneration(9)).ok(), "wrong-gen release rejects");

  auto B = req(8u << 20); check(B.ok(), "alloc B");
  auto C = req(8u << 20); check(C.ok(), "alloc C");
  auto D = req(8u << 20); check(D.ok(), "alloc D");
  check(gov.release(B->id, B->generation).ok(), "free B");
  check(gov.release(D->id, D->generation).ok(), "free D");
  auto sm = gov.summary();
  check(sm.free > sm.largest_free, "fragmented: free > largest contiguous");
  auto big = req(44u << 20);
  check(!big.ok(), "contiguous-capacity failure rejects");
  check(big.error().reason == RejectReason::INSUFFICIENT_CONTIGUOUS_CAPACITY ||
        big.error().reason == RejectReason::INSUFFICIENT_TOTAL_CAPACITY, "typed contiguous reason");
  std::printf("fragmentation proof: free=%llu largest_free=%llu\n",
              static_cast<unsigned long long>(sm.free),
              static_cast<unsigned long long>(sm.largest_free));
  check(gov.release(A->id, A->generation).ok(), "free A");
  check(gov.release(C->id, C->generation).ok(), "free C");

  // ---- Governor 2: island compaction with a real device-to-device move.
  MemoryGovernor g2(make_domain(32u << 20, 0.30, 0.70), std::make_unique<CudaBackend>());
  check(g2.initialize().ok(), "g2 init");
  const std::uint64_t bsz = 2u << 20;
  auto req2 = [&](Movability mov) {
    AllocationRequest r; r.bytes = bsz; r.alignment = 256; r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = Reclaimability::RECLAIM_IF_COLD; r.movability = mov; r.owner = "compact";
    return g2.request(r);
  };
  auto K1 = req2(Movability::PINNED); check(K1.ok(), "g2 K1");
  auto K2 = req2(Movability::MOVABLE); check(K2.ok(), "g2 K2");
  auto K3 = req2(Movability::MOVABLE); check(K3.ok(), "g2 K3");
  auto K4 = req2(Movability::MOVABLE); check(K4.ok(), "g2 K4");
  auto K5 = req2(Movability::MOVABLE); check(K5.ok(), "g2 K5");
  auto K6 = req2(Movability::MOVABLE); check(K6.ok(), "g2 K6");
  check(g2.pin(K1->id, K1->generation).ok(), "pin K1");

  // Fill every movable allocation with the deterministic pattern so whatever
  // the planner moves carries valid data.
  for (auto* k : {&K3, &K4, &K6}) check(fill_pattern(g2, **k), "g2 fill movable");

  check(g2.release(K2->id, K2->generation).ok(), "g2 free K2");
  check(g2.release(K5->id, K5->generation).ok(), "g2 free K5");

  const auto before2 = g2.summary();
  auto plan = g2.plan_compaction();
  check(plan.steps.size() >= 1, "g2 compaction plan non-empty");
  check(plan.largest_free_after > before2.largest_free, "g2 compaction improves largest free");
  if (!plan.steps.empty()) {
    const auto& st = plan.steps[0];
    const std::size_t nbytes = st.new_size;
    if (cudaMemcpy(dev_at(g2, st.new_offset), dev_at(g2, st.old_offset), nbytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
      check(false, "g2 D2D copy");
    } else {
      auto rel = g2.apply_relocation(st.id, st.generation, st.new_offset, st.new_size);
      check(rel.ok(), "g2 apply relocation");
      check(g2.get(st.id, rel->generation).ok(), "g2 relocated handle resolves");
      check(!g2.get(st.id, st.generation).ok(), "g2 old handle stale after move");
      check(check_pattern(g2, *rel), "g2 data parity preserved across D2D move");
    }
  }
  std::printf("compaction proof: largest_free before=%llu after=%llu\n",
              static_cast<unsigned long long>(before2.largest_free),
              static_cast<unsigned long long>(plan.largest_free_after));

  auto cands2 = g2.reclaim_candidates(100);
  bool hasK1 = false;
  for (auto id : cands2) if (id == K1->id) hasK1 = true;
  check(!hasK1, "pinned excluded from reclaim");

  // ---- Governor 3: pressure / reclaim proof.
  MemoryGovernor g3(make_domain(16u << 20, 0.40, 0.70), std::make_unique<CudaBackend>());
  check(g3.initialize().ok(), "g3 init");
  auto req3 = [&](std::uint64_t bytes, Reclaimability rec, Movability mov) {
    AllocationRequest r; r.bytes = bytes; r.alignment = 256; r.mem_class = MemoryClass::USER_BUFFER;
    r.reclaimability = rec; r.movability = mov; r.owner = "press";
    return g3.request(r);
  };
  auto PA = req3(8u << 20, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE); check(PA.ok(), "g3 PA");
  auto PB = req3(4u << 20, Reclaimability::NEVER_RECLAIM, Movability::PINNED); check(PB.ok(), "g3 PB");
  auto PR = req3(2u << 20, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE); check(PR.ok(), "g3 PR");
  check(g3.pin(PB->id, PB->generation).ok(), "g3 pin PB");
  const auto sp = g3.pressure();
  check(sp == PressureState::HIGH || sp == PressureState::CRITICAL || sp == PressureState::ELEVATED, "g3 pressure elevated+");
  auto cands3 = g3.reclaim_candidates(100);
  bool hasPB = false;
  for (auto id : cands3) if (id == PB->id) hasPB = true;
  check(!hasPB, "g3 pinned excluded from reclaim");
  if (!cands3.empty()) {
    auto cur = g3.get(cands3[0], AllocationGeneration(1));
    if (cur.ok()) check(g3.release(cands3[0], cur->generation).ok(), "g3 reclaim release");
    const auto sp2 = g3.pressure();
    check(sp2 == PressureState::NORMAL || sp2 == PressureState::LOW || sp2 == PressureState::ELEVATED, "g3 pressure eased");
  }
  std::printf("pressure proof: before=%s\n", to_string(sp));

  // ---- OOM: exceeds governed capacity, rejects before any physical allocation.
  auto oom = req3(32u << 20, Reclaimability::RECLAIM_IF_COLD, Movability::MOVABLE);
  check(!oom.ok(), "g3 OOM rejects");

  // ---- State/reload: reclaimable allocation with deterministic host backing.
  {
    MemoryGovernor g4(make_domain(24u << 20, 0.30, 0.70), std::make_unique<CudaBackend>());
    check(g4.initialize().ok(), "g4 init");
    AllocationRequest sr; sr.bytes = 4u << 20; sr.alignment = 256; sr.mem_class = MemoryClass::STATE_CACHE;
    sr.reclaimability = Reclaimability::RECLAIM_IF_RELOADABLE; sr.movability = Movability::MOVABLE; sr.backing_ref = "host-backing/tensor";
    auto SR = g4.request(sr); check(SR.ok(), "g4 reloadable alloc");
    check(fill_pattern(g4, *SR), "g4 pre-reload parity");
    const std::size_t n = SR->size / sizeof(float);
    std::vector<float> backing(n);
    cudaMemcpy(backing.data(), dev_at(g4, SR->offset), SR->size, cudaMemcpyDeviceToHost);
    check(g4.release(SR->id, SR->generation).ok(), "g4 evict residency");
    check(g4.summary().committed == 0, "g4 residency released");
    auto SR2 = g4.request(sr); check(SR2.ok(), "g4 reload alloc");
    cudaMemcpy(dev_at(g4, SR2->offset), backing.data(), SR2->size, cudaMemcpyHostToDevice);
    bool okp = true;
    for (std::size_t i = 0; i < n; ++i) if (backing[i] != 7.0f + static_cast<float>(i)) { okp = false; break; }
    check(okp, "g4 backing parity");
    check(check_pattern(g4, *SR2), "g4 reload residency parity");
  }

  }

  std::size_t free1 = 0, total1 = 0;
  cudaMemGetInfo(&free1, &total1);
  const long long delta = static_cast<long long>(free1) - static_cast<long long>(free0);
  std::printf("final free=%zu (baseline %zu) delta=%lld\n", free1, free0, delta);
  check(delta >= -64ll * 1024 * 1024, "final free within measurement tolerance");

  if (failures == 0) { std::printf("cuda_arena test: PASS\n"); return 0; }
  std::printf("cuda_arena test: %d FAILURES\n", failures);
  return 1;
}
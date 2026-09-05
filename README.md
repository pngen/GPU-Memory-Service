# GPU Memory Service

**GPU Memory Service is an open-source, vendor-neutral C++20 runtime for governing GPU memory allocation, residency, eviction, reclamation, compaction, staging, pressure, and authoritative ownership across heterogeneous accelerator infrastructure.**

**It answers one systems question:**

**Which GPU-memory bytes are allocated, resident, reclaimable, movable, staged, pinned, or free now; who owns them; which generation is authoritative; and what action may safely change that memory state without corrupting active work?**

GPU memory is not a free-byte counter. A device may report enough nominal
free VRAM while still being unable to satisfy real work because of
fragmentation, non-contiguous free regions, pinned/non-movable allocations,
stale ownership, model residency, KV/cache residency, active kernels, in-flight
transfers, checkpoint/state staging, cross-workload pressure, allocator
metadata, pending frees, delayed completion, worker death, stale process
authority, device reset, allocation-generation changes, memory-class
restrictions, migration constraints, or unsafe reclamation. GPU Memory Service
makes that state explicit and authoritative by tracking, for every governed
device domain, the identity, generation, ownership, residency, movability,
reclaimability, contiguity, and physical lifetime of every byte.

## Systems boundary

GPU Memory Service owns authoritative GPU-memory state and safe memory-state
mutation. It is deliberately narrow:

- **Resource Broker** owns generic scarce-resource arbitration.
- **Reservation Fabric** owns future commitments.
- **Capacity Fabric** owns broader current/future capacity modeling.
- **Fragmentation Governor** owns infrastructure-wide fragmentation
  diagnosis/remediation intent.
- **Fabric Scheduler** owns workload placement.
- **Model Residency** owns model-level residency policy.
- **State Index / Distributed Cache Directory / Checkpoint Store** and related
  state runtimes own reusable-state discovery/persistence.
- **Execution Fabric** owns execution-attempt authority.
- **Preemption Fabric** owns safe interruption.
- **GPU Memory Service** owns authoritative device-memory regions,
  allocation/residency state, safe local reclaim/eviction/compaction, and
  physical memory mutation.

The defining thesis:

**GPU memory is not a free-byte counter. It is an authoritative layout of
generation-bound allocations whose residency, ownership, movability,
reclaimability, contiguity, and physical lifetime determine whether new
accelerator work can safely fit now.**

## Core doctrine

- Free bytes are not necessarily usable bytes.
- Allocated bytes are not necessarily resident bytes.
- Resident bytes are not necessarily active bytes.
- A request is not an allocation.
- An allocation handle is not ownership unless its generation is current.
- A free request is not a completed free.
- A process disappearing does not prove a graceful release occurred.
- A stale allocation generation must not free current memory.
- A successful `cudaMalloc` proves physical allocation, not durable authority.
- Recovery is conservative.

## Strongly typed identities and generations

Every identity and generation is a distinct C++ type (via the
`TypedId`/`TypedGeneration` templates) so the compiler rejects cross-domain
conversion. A stale `AllocationGeneration` must not free or relocate current
memory, a stale `WorkerBootId` must not publish an allocation completion, and a
stale `DeviceGeneration` must not satisfy current capacity.

## Design

The core runtime is the `MemoryGovernor`. It owns a `MemoryDomain` (physical
device identity, generation, capacity and provenance), a `RegionAllocator`
(exact authoritative byte layout with no-overlap and exact-accounting
invariants), generation-bound `AllocationRecord`s, pressure/watermark policy,
deterministic reclaim selection, safe compaction planning and relocation, and
conservative persistence/recovery. A pluggable `IBackend` supplies the real
physical bytes: a host backend for deterministic GPU-free testing and a CUDA
backend for the real RTX 5090 proof. The transport (framed TCP coordinator and
CUDA workers) is a reference deployment mechanism; the public API is usable
without it.

The region/suballocation model is **runtime-managed layout over one or more real
CUDA allocations**, not fabricated driver-internal fragmentation telemetry.
Compaction is **controlled reference allocation**: it moves movable
suballocations with a real device-to-device copy, advances generations, and
rejects stale handles; it does not claim transparent relocation of arbitrary
third-party CUDA allocations.

## Building

Requires CMake + a C++20 compiler (MSVC `/W4 /WX` enforced), and optionally the
CUDA toolkit (sm_120 / RTX 5090) for the real physical proof.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## Install and consume

```sh
cmake --install build --prefix <prefix>
```

Downstream:

```cmake
find_package(GPUMemoryService CONFIG REQUIRED)
target_link_libraries(app PRIVATE GPUMemoryService::GPUMemoryService)
```

## Proofs

The suite exercises, on the real physical RTX 5090 (sm_120): allocation
authority under generation (stale free / duplicate free reject), arena
suballocation, fragmentation (total free > largest contiguous region), physical
compaction with a real D2D move and data parity, pinning exclusion, pressure and
reclaim with watermarks/hysteresis, state/reload from host backing, and bounded
OOM. It also includes seeded property testing against an occupancy reference,
genuine multithreaded concurrency, and persistence/corruption rejection.

## License

Apache License 2.0. Copyright 2026 Summon Software Labs. No telemetry transmission.

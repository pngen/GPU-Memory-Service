#pragma once
// Physical-memory backend abstraction.  GPU Memory Service governs an
// authoritative byte layout; a backend supplies the real physical bytes.  For
// deterministic host testing this may be host memory; for the RTX 5090 proof a
// CUDA backend allocates a real device arena.  The backend never becomes
// durable authority — the layout is authoritative, the base pointer is
// process-local implementation state.

#include <cstdint>
#include <string>

namespace gpu_memory_service {

class IBackend {
 public:
  virtual ~IBackend() = default;

  // Allocate a bounded physical arena and return an opaque base handle.
  // On failure returns false and sets err.  The base handle is process-local
  // and must never be persisted as durable authority.
  virtual bool allocate_arena(std::uint64_t bytes, std::uint64_t alignment,
                              void** base, std::string* err) = 0;

  virtual void free_arena(void* base) noexcept = 0;
  virtual bool is_cuda() const noexcept = 0;
};

}  // namespace gpu_memory_service

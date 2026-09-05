#pragma once
// Host-memory IBackend for deterministic, GPU-free testing.  Provides real
// host bytes (so parity assertions can be made) but is clearly not a CUDA
// device arena.  Used for unit/property/concurrency suites and the downstream
// find_package consumer.

#include "backend.hpp"

namespace gpu_memory_service {

class HostBackend final : public IBackend {
 public:
  bool allocate_arena(std::uint64_t bytes, std::uint64_t alignment, void** base, std::string* err) override;
  void free_arena(void* base) noexcept override;
  bool is_cuda() const noexcept override { return false; }
};

}  // namespace gpu_memory_service

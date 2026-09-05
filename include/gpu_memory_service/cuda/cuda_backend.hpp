#pragma once
// CUDA physical backend: allocates a real device arena via cudaMalloc.
// The base pointer is process-local implementation state and is never
// persisted as durable authority.

#include "../backend.hpp"

namespace gpu_memory_service {

class CudaBackend final : public IBackend {
 public:
  bool allocate_arena(std::uint64_t bytes, std::uint64_t alignment, void** base, std::string* err) override;
  void free_arena(void* base) noexcept override;
  bool is_cuda() const noexcept override { return true; }
};

}  // namespace gpu_memory_service

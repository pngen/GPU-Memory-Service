#include "gpu_memory_service/host_backend.hpp"

#include <cstdlib>

namespace gpu_memory_service {

bool HostBackend::allocate_arena(std::uint64_t bytes, std::uint64_t alignment, void** base, std::string* err) {
  (void)err;
  if (bytes == 0) return false;
  void* p = _aligned_malloc(static_cast<std::size_t>(bytes), static_cast<std::size_t>(alignment));
  if (!p) return false;
  *base = p;
  return true;
}

void HostBackend::free_arena(void* base) noexcept {
  if (base) _aligned_free(base);
}

}  // namespace gpu_memory_service

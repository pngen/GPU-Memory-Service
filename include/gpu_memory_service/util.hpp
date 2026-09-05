#pragma once
// Checked arithmetic and small numeric helpers used throughout the runtime.
// Correctness depends on these never overflowing silently.

#include <cstdint>
#include <limits>

namespace gpu_memory_service::detail {

[[nodiscard]] inline bool is_pow2(std::uint64_t v) noexcept {
  return v != 0 && (v & (v - 1)) == 0;
}

[[nodiscard]] inline bool checked_add(std::uint64_t a, std::uint64_t b,
                                      std::uint64_t& out) noexcept {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) return false;
  out = a + b;
  return true;
}

// Align v up to a power-of-two alignment.  Returns 0 on overflow.
[[nodiscard]] inline std::uint64_t align_up(std::uint64_t v, std::uint64_t align) noexcept {
  if (!is_pow2(align)) return v;
  const std::uint64_t mask = align - 1;
  const std::uint64_t r = v + mask;
  if (r < v) return 0;
  return r & ~mask;
}

}  // namespace gpu_memory_service::detail

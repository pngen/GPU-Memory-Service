#include "gpu_memory_service/cuda/cuda_backend.hpp"

#include <cuda_runtime.h>

namespace gpu_memory_service {

bool CudaBackend::allocate_arena(std::uint64_t bytes, std::uint64_t alignment, void** base, std::string* err) {
  (void)alignment;  // cudaMalloc guarantees 256-byte alignment; s_m_80+ aligns to 256.
  if (bytes == 0) { if (err) *err = "zero-byte arena"; return false; }
  void* p = nullptr;
  cudaError_t e = cudaMalloc(&p, static_cast<std::size_t>(bytes));
  if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
  *base = p;
  return true;
}

void CudaBackend::free_arena(void* base) noexcept {
  if (base) cudaFree(base);
}

}  // namespace gpu_memory_service

// Kernel used by the real arena/suballocation proof: elementwise write+read.
__global__ void gms_fill_kernel(float* data, std::size_t n, float value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) data[i] = value + static_cast<float>(i);
}

__global__ void gms_scale_kernel(float* data, std::size_t n, float value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= value;
}

// Host-callable wrappers.
extern "C" int gms_cuda_fill(float* data, std::size_t n, float value, cudaStream_t s) {
  const int threads = 256;
  const std::size_t blocks = (n + threads - 1) / threads;
  gms_fill_kernel<<<static_cast<std::uint32_t>(blocks), threads, 0, s>>>(data, n, value);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int gms_cuda_scale(float* data, std::size_t n, float value, cudaStream_t s) {
  const int threads = 256;
  const std::size_t blocks = (n + threads - 1) / threads;
  gms_scale_kernel<<<static_cast<std::uint32_t>(blocks), threads, 0, s>>>(data, n, value);
  return static_cast<int>(cudaGetLastError());
}

#include "src/net.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <map>

using namespace gpu_memory_service;
using namespace gpu_memory_service::mp;
using namespace gpu_memory_service::net;

// Fresh WorkerBootId per process: PID high bits + a per-process timer.  A stale
// boot must never publish an allocation completion.
static std::uint64_t fresh_boot() {
  std::uint64_t pid = static_cast<std::uint64_t>(GetCurrentProcessId());
  std::uint64_t t = static_cast<std::uint64_t>(GetTickCount64());
  return (pid << 32) | (t & 0xFFFFFFFFull);
}

struct Alloc { std::uint64_t gen; std::uint64_t bytes; void* ptr; };

int main(int argc, char** argv) {
  if (argc < 3) { std::printf("usage: worker <host> <port> <name>\n"); return 1; }
  init();
  const int port = std::atoi(argv[2]);
  const std::string name = argc > 3 ? argv[3] : "worker";
  const std::uint64_t boot = fresh_boot();

  { void* warm = nullptr; cudaMalloc(&warm, 1024); if (warm) cudaFree(warm); }  // warm up context
  SOCKET s = connect_loopback(port, nullptr);
  if (s == INVALID_SOCKET) { std::printf("worker: connect failed\n"); return 1; }

  send_frame(s, Msg::HELLO, pack_u8(2));  // role WORKER
  { std::vector<std::uint8_t> p; append_str(p, name); append_u64(p, boot); append_u64(p, GetCurrentProcessId()); send_frame(s, Msg::REGISTER, p); }
  { std::size_t freeb = 0, totalb = 0; cudaMemGetInfo(&freeb, &totalb);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, 0);
    std::vector<std::uint8_t> p; append_u64(p, static_cast<std::uint64_t>(freeb)); append_u64(p, static_cast<std::uint64_t>(totalb)); append_str(p, prop.name); send_frame(s, Msg::PUBLISH_CAPACITY, p);
    std::printf("WORKER_PUBLISH boot=%llu free=%zu total=%zu dev=%s\n", static_cast<unsigned long long>(boot), freeb, totalb, prop.name);
    std::fflush(stdout);
  }

  std::map<std::uint64_t, Alloc> allocs;
  for (;;) {
    Frame f; if (!recv_frame(s, f)) { std::printf("WORKER_DISCONNECT\n"); return 0; }
    if (f.type == Msg::ALLOCATE) {
      std::size_t p = 0; std::uint64_t id=0, gen=0, bytes=0;
      take_u64(f.payload, p, id); take_u64(f.payload, p, gen); take_u64(f.payload, p, bytes);
      int status = 0; std::string e;
      void* ptr = nullptr;
      cudaError_t err = cudaMalloc(&ptr, static_cast<std::size_t>(bytes));
      if (err != cudaSuccess) { status = -1; e = cudaGetErrorString(err); }
      else allocs[id] = Alloc{gen, bytes, ptr};
      std::vector<std::uint8_t> o; append_u64(o, id); append_u64(o, gen); append_u32(o, static_cast<std::uint32_t>(status)); append_str(o, e); send_frame(s, Msg::ALLOCATE_RESULT, o);
    } else if (f.type == Msg::RELEASE) {
      std::size_t p = 0; std::uint64_t id=0, gen=0, bootv=0;
      take_u64(f.payload, p, id); take_u64(f.payload, p, gen); take_u64(f.payload, p, bootv);
      int status = 0; std::string e;
      auto it = allocs.find(id);
      if (it == allocs.end() || it->second.gen != gen) { status = -1; e = "stale or unknown"; }
      else { cudaFree(it->second.ptr); allocs.erase(it); }
      std::vector<std::uint8_t> o; append_u64(o, id); append_u64(o, gen); append_u32(o, static_cast<std::uint32_t>(status)); append_str(o, e); send_frame(s, Msg::RELEASE_RESULT, o);
    } else if (f.type == Msg::SHUTDOWN) {
      for (auto& kv : allocs) cudaFree(kv.second.ptr);
      return 0;
    }
  }
}
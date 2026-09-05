#include "src/net.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

using namespace gpu_memory_service;
using namespace gpu_memory_service::mp;
using namespace gpu_memory_service::net;

static CoordinatorEpoch g_epoch = CoordinatorEpoch(static_cast<std::uint64_t>(GetCurrentProcessId()));

struct AllocState { std::uint64_t id; std::uint64_t gen; std::uint64_t boot; std::uint64_t bytes; bool current; };
static std::map<std::uint64_t, AllocState> g_alloc;
static SOCKET g_worker = INVALID_SOCKET;
static std::uint64_t g_worker_boot = 0;
static std::mutex g_mu;         // guards g_alloc + g_worker + g_worker_boot
static std::mutex g_worker_mu;  // serializes forward+reply on the worker socket

static void handle_conn(SOCKET c) {
  Frame hello;
  if (!recv_frame(c, hello)) { shutdown_sock(c); return; }
  std::size_t p = 0; std::uint8_t role = 0;
  if (!take_u8(hello.payload, p, role)) { shutdown_sock(c); return; }

  if (role == 2) {  // WORKER
    Frame reg; if (!recv_frame(c, reg)) { shutdown_sock(c); return; }
    std::size_t rp = 0; std::string name; std::uint64_t boot = 0, pidv = 0;
    if (!take_str(reg.payload, rp, name)) { shutdown_sock(c); return; }
    if (!take_u64(reg.payload, rp, boot)) { shutdown_sock(c); return; }
    if (!take_u64(reg.payload, rp, pidv)) { shutdown_sock(c); return; }
    Frame cap; if (!recv_frame(c, cap)) { shutdown_sock(c); return; }
    { std::lock_guard<std::mutex> lk(g_mu); g_worker = c; g_worker_boot = boot; }
    std::printf("WORKER_REGISTERED name=%s boot=%llu pid=%llu\n", name.c_str(), (unsigned long long)boot, (unsigned long long)pidv);
    std::fflush(stdout);
    return;
  }

  // CONTROLLER
  for (;;) {
    Frame f; if (!recv_frame(c, f)) { break; }
    if (f.type == Msg::ALLOCATE) {
      std::size_t ap = 0; std::uint64_t idv = 0, genv = 0, bytes = 0;
      take_u64(f.payload, ap, idv); take_u64(f.payload, ap, genv); take_u64(f.payload, ap, bytes);
      std::lock_guard<std::mutex> wl(g_worker_mu);
      std::lock_guard<std::mutex> lk(g_mu);
      if (!g_worker_boot) { send_frame(c, Msg::ALLOCATE_RESULT, pack_i32(-2)); continue; }
      { std::vector<std::uint8_t> ar; append_u64(ar, idv); append_u64(ar, genv); append_u64(ar, bytes); send_frame(g_worker, Msg::ALLOCATE, ar); }
      Frame res; if (!recv_frame(g_worker, res) || res.type != Msg::ALLOCATE_RESULT) {
        std::printf("WORKER_DEAD\n"); std::fflush(stdout);
        for (auto& kv : g_alloc) if (kv.second.boot == g_worker_boot) kv.second.current = false;
        g_worker = INVALID_SOCKET; g_worker_boot = 0;
        { std::vector<std::uint8_t> out; append_u64(out,idv); append_u64(out,genv); append_u32(out,0xFFFFFFF6u); append_str(out,"worker dead"); send_frame(c, Msg::ALLOCATE_RESULT, out); }
        continue;
      }
      std::size_t rp=0; std::uint64_t a=0,b=0; std::uint32_t status=0; std::string e;
      take_u64(res.payload,rp,a); take_u64(res.payload,rp,b); take_u32(res.payload,rp,status); take_str(res.payload,rp,e);
      g_alloc[idv] = AllocState{idv, genv, g_worker_boot, bytes, status == 0};
      std::vector<std::uint8_t> out; append_u64(out,idv); append_u64(out,genv); append_u32(out,status); append_str(out,e); send_frame(c, Msg::ALLOCATE_RESULT, out);
    } else if (f.type == Msg::RELEASE) {
      std::size_t ap = 0; std::uint64_t idv=0, genv=0, bootv=0;
      take_u64(f.payload, ap, idv); take_u64(f.payload, ap, genv); take_u64(f.payload, ap, bootv);
      std::lock_guard<std::mutex> wl(g_worker_mu);
      std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_alloc.find(idv);
      int status = 0; std::string e;
      if (it == g_alloc.end() || it->second.gen != genv || it->second.boot != g_worker_boot) { status = -1; e = "stale or unknown"; }
      else {
        std::vector<std::uint8_t> rr; append_u64(rr,idv); append_u64(rr,genv); append_u64(rr,bootv); send_frame(g_worker, Msg::RELEASE, rr);
        Frame res; if (!recv_frame(g_worker, res) || res.type != Msg::RELEASE_RESULT) { status = -10; e = "worker dead"; }
        else { std::size_t rrp=0; std::uint64_t x=0,y=0; std::uint32_t st=0; std::string ee; take_u64(res.payload,rrp,x); take_u64(res.payload,rrp,y); take_u32(res.payload,rrp,st); take_str(res.payload,rrp,ee); status=st; e=ee; }
      }
      std::vector<std::uint8_t> out; append_u64(out,idv); append_u64(out,genv); append_u32(out,status); append_str(out,e); send_frame(c, Msg::RELEASE_RESULT, out);
    } else if (f.type == Msg::QUERY) {
      std::uint64_t committed = 0, current = 0;
      { std::lock_guard<std::mutex> lk(g_mu); for (auto& kv : g_alloc) if (kv.second.current) { committed += kv.second.bytes; ++current; } }
      std::vector<std::uint8_t> out; append_u64(out,committed); append_u64(out,current); send_frame(c, Msg::QUERY_RESULT, out);
    } else if (f.type == Msg::SHUTDOWN) {
      { std::lock_guard<std::mutex> lk(g_mu); if (g_worker != INVALID_SOCKET) shutdown_sock(g_worker); }
      shutdown_sock(c);
      std::printf("COORDINATOR_SHUTDOWN\n"); std::fflush(stdout);
      return;
    }
  }
  shutdown_sock(c);
}

int main(int argc, char** argv) {
  init();
  int port = argc > 1 ? std::atoi(argv[1]) : 0;
  int bound = 0; std::string err;
  SOCKET ls = listen_loopback(port, &bound, &err);
  if (ls == INVALID_SOCKET) { std::printf("coordinator: listen failed %s\n", err.c_str()); return 1; }
  std::printf("COORDINATOR_READY port=%d epoch=%llu\n", bound, (unsigned long long)g_epoch.value());
  std::fflush(stdout);
  std::vector<std::thread> threads;
  for (;;) {
    sockaddr_in cli{}; int clen = sizeof(cli);
    SOCKET c = accept(ls, reinterpret_cast<sockaddr*>(&cli), &clen);
    if (c == INVALID_SOCKET) continue;
    threads.emplace_back(handle_conn, c);
  }
}
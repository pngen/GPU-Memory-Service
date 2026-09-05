#include "src/net.hpp"
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gpu_memory_service;
using namespace gpu_memory_service::mp;
using namespace gpu_memory_service::net;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

/* Real spawn that returns the process handle and keeps it running. */
static HANDLE spawn_real(const std::string& cmdline) {
  STARTUPINFOA si{}; si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  std::vector<char> buf(cmdline.begin(), cmdline.end()); buf.push_back('\0');
  if (!CreateProcessA(nullptr, buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return nullptr;
  CloseHandle(pi.hThread);
  return pi.hProcess;
}

int main(int argc, char** argv) {
  if (argc < 4) { std::printf("usage: test_worker_death <coordinator_exe> <worker_exe> <port>\n"); return 2; }
  const std::string coord_exe = argv[1];
  const std::string worker_exe = argv[2];
  const int port = std::atoi(argv[3]);
  init();

  const HANDLE coord = spawn_real("\"" + coord_exe + "\" " + std::to_string(port));
  check(coord != nullptr, "coordinator spawned");
  Sleep(1500);

  const HANDLE workerA = spawn_real("\"" + worker_exe + "\" 127.0.0.1 " + std::to_string(port) + " A");
  check(workerA != nullptr, "worker A spawned");
  Sleep(1500);

  SOCKET ctl = connect_loopback(port, nullptr);
  check(ctl != INVALID_SOCKET, "controller connected");
  send_frame(ctl, Msg::HELLO, pack_u8(1));  // CONTROLLER
  { auto qh = [&](std::uint64_t& c, std::uint64_t& n) {
      send_frame(ctl, Msg::QUERY, pack_u8(0));
      Frame r; if (!recv_frame(ctl, r) || r.type != Msg::QUERY_RESULT) return false;
      std::size_t pp=0; take_u64(r.payload,pp,c); take_u64(r.payload,pp,n); return true;
    };
    std::uint64_t c=0,n=0; check(qh(c,n), "controller handshake QUERY");
  }

  auto allocate = [&](std::uint64_t id, std::uint64_t gen, std::uint64_t bytes, int& status) {
    std::vector<std::uint8_t> p; append_u64(p, id); append_u64(p, gen); append_u64(p, bytes);
    send_frame(ctl, Msg::ALLOCATE, p);
    Frame r; if (!recv_frame(ctl, r) || r.type != Msg::ALLOCATE_RESULT) { status = -999; return false; }
    std::size_t pp=0; std::uint64_t a=0,b=0; std::uint32_t st=0; std::string e; take_u64(r.payload,pp,a); take_u64(r.payload,pp,b); take_u32(r.payload,pp,st); take_str(r.payload,pp,e);
    status = static_cast<int>(st); return true;
  };
  auto release = [&](std::uint64_t id, std::uint64_t gen, int& status) {
    std::vector<std::uint8_t> p; append_u64(p, id); append_u64(p, gen); append_u64(p, 0);
    send_frame(ctl, Msg::RELEASE, p);
    Frame r; if (!recv_frame(ctl, r) || r.type != Msg::RELEASE_RESULT) { status = -999; return false; }
    std::size_t pp=0; std::uint64_t a=0,b=0; std::uint32_t st=0; std::string e; take_u64(r.payload,pp,a); take_u64(r.payload,pp,b); take_u32(r.payload,pp,st); take_str(r.payload,pp,e);
    status = static_cast<int>(st); return true;
  };
  auto query = [&](std::uint64_t& committed, std::uint64_t& current) {
    send_frame(ctl, Msg::QUERY, pack_u8(0));
    Frame r; if (!recv_frame(ctl, r) || r.type != Msg::QUERY_RESULT) { return false; }
    std::size_t pp=0; take_u64(r.payload,pp,committed); take_u64(r.payload,pp,current); return true;
  };

  int st = 0;
  check(allocate(1, 1, 1u << 20, st), "allocate1 frame"); check(st == 0, "allocate1 under worker A");
  std::uint64_t committed=0, current=0;
  query(committed, current);
  check(committed >= (1u << 20), "committed reflects worker A allocation");

  // Kill worker A as a real OS process; coordinator must fence.
  check(TerminateProcess(workerA, 0) != 0, "terminate worker A");
  Sleep(300);

  check(allocate(2, 1, 1u << 20, st), "allocate2 frame after death");
  check(st == -10, "allocate after worker death rejects (fenced)");
  query(committed, current);
  check(current == 0, "worker-death fenced all allocations");

  // Worker A' restarts with a fresh boot and republishes.
  const HANDLE workerB = spawn_real("\"" + worker_exe + "\" 127.0.0.1 " + std::to_string(port) + " B");
  check(workerB != nullptr, "worker B spawned");
  Sleep(800);

  check(allocate(3, 1, 1u << 20, st), "allocate3 frame"); check(st == 0, "allocate3 succeeds under worker B");
  check(release(3, 1, st) && st == 0, "release under worker B succeeds");

  send_frame(ctl, Msg::SHUTDOWN, pack_u8(0));
  Sleep(300);

  if (workerB) TerminateProcess(workerB, 0);
  TerminateProcess(coord, 0);
  CloseHandle(coord); if (workerA) CloseHandle(workerA); if (workerB) CloseHandle(workerB);
  shutdown_sock(ctl);

  if (failures == 0) { std::printf("worker_death test: PASS\n"); return 0; }
  std::printf("worker_death test: %d FAILURES\n", failures);
  return 1;
}
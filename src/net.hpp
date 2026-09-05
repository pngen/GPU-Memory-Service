#pragma once
// Minimal Winsock TCP helpers for the reference multi-process deployment.
// Handles partial send/recv correctness (loops until complete) and framed
// reads with a bounded length and checksum verification.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <vector>

#include "gpu_memory_service/protocol.hpp"

namespace gpu_memory_service::net {

inline bool init() {
  WSADATA d;
  return WSAStartup(MAKEWORD(2, 2), &d) == 0;
}

inline SOCKET listen_loopback(int port, int* bound_port, std::string* err) {
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) { if (err) *err = "socket"; return INVALID_SOCKET; }
  BOOL reuse = TRUE; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));
  BOOL nodelay = TRUE; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(static_cast<u_short>(port));
  if (bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { if (err) *err = "bind"; closesocket(s); return INVALID_SOCKET; }
  if (listen(s, 8) != 0) { if (err) *err = "listen"; closesocket(s); return INVALID_SOCKET; }
  if (bound_port) {
    sockaddr_in out{}; int len = sizeof(out); getsockname(s, reinterpret_cast<sockaddr*>(&out), &len);
    *bound_port = ntohs(out.sin_port);
  }
  return s;
}

inline SOCKET connect_loopback(int port, std::string* err) {
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) { if (err) *err = "socket"; return INVALID_SOCKET; }
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); a.sin_port = htons(static_cast<u_short>(port));
  if (connect(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) { if (err) *err = "connect"; closesocket(s); return INVALID_SOCKET; }
  BOOL nodelay = TRUE; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&nodelay), sizeof(nodelay));
  return s;
}

inline bool send_exact(SOCKET s, const void* data, std::size_t n) {
  const char* p = static_cast<const char*>(data);
  std::size_t done = 0;
  while (done < n) {
    int w = send(s, p + done, static_cast<int>(n - done), 0);
    if (w <= 0) return false;
    done += static_cast<std::size_t>(w);
  }
  return true;
}

inline bool recv_exact(SOCKET s, void* data, std::size_t n) {
  char* p = static_cast<char*>(data);
  std::size_t done = 0;
  while (done < n) {
    int r = recv(s, p + done, static_cast<int>(n - done), 0);
    if (r <= 0) return false;
    done += static_cast<std::size_t>(r);
  }
  return true;
}

inline bool send_frame(SOCKET s, mp::Msg type, const std::vector<std::uint8_t>& payload) {
  const auto b = mp::encode(type, payload);
  return send_exact(s, b.data(), b.size());
}

// Read one frame, assembling and verifying it.  Returns false on disconnect or
// a malformed/oversized/checksum-failed frame.
inline bool recv_frame(SOCKET s, mp::Frame& out) {
  std::uint8_t hdr[13];
  if (!recv_exact(s, hdr, 13)) return false;
  std::size_t hp = 0; std::vector<std::uint8_t> hb(hdr, hdr + 13);
  std::uint32_t magic = 0; if (!mp::take_u32(hb, hp, magic)) return false;
  if (magic != mp::kMagic) return false;
  std::uint32_t ver = 0; if (!mp::take_u32(hb, hp, ver)) return false;
  if (ver != mp::kVersion) return false;
  std::uint8_t t = 0; if (!mp::take_u8(hb, hp, t)) return false;
  std::uint32_t len = 0; if (!mp::take_u32(hb, hp, len)) return false;
  if (len > mp::kMaxFrame) return false;
  std::vector<std::uint8_t> buf = hb;
  buf.resize(13 + len);
  if (!recv_exact(s, buf.data() + 13, len)) return false;
  std::uint8_t crcb[4];
  if (!recv_exact(s, crcb, 4)) return false;
  buf.insert(buf.end(), crcb, crcb + 4);
  std::size_t consumed = 0;
  return mp::decode(buf, consumed, out);
}

inline void shutdown_sock(SOCKET s) {
  if (s != INVALID_SOCKET) { shutdown(s, SD_BOTH); closesocket(s); }
}

}  // namespace gpu_memory_service::net
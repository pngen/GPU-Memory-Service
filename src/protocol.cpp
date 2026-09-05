#include "gpu_memory_service/protocol.hpp"

namespace gpu_memory_service::mp {

std::uint32_t crc32(const std::uint8_t* data, std::size_t len) {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return crc ^ 0xFFFFFFFFu;
}

void append_u8(std::vector<std::uint8_t>& b, std::uint8_t v) { b.push_back(v); }
void append_u32(std::vector<std::uint8_t>& b, std::uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF)); }
void append_u64(std::vector<std::uint8_t>& b, std::uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF)); }
void append_i32(std::vector<std::uint8_t>& b, std::int32_t v) { append_u32(b, static_cast<std::uint32_t>(v)); }
void append_str(std::vector<std::uint8_t>& b, const std::string& s) {
  append_u32(b, static_cast<std::uint32_t>(s.size()));
  for (char c : s) b.push_back(static_cast<std::uint8_t>(c));
}

std::vector<std::uint8_t> pack_u8(std::uint8_t v) { std::vector<std::uint8_t> b; append_u8(b, v); return b; }
std::vector<std::uint8_t> pack_u32(std::uint32_t v) { std::vector<std::uint8_t> b; append_u32(b, v); return b; }
std::vector<std::uint8_t> pack_u64(std::uint64_t v) { std::vector<std::uint8_t> b; append_u64(b, v); return b; }
std::vector<std::uint8_t> pack_i32(std::int32_t v) { std::vector<std::uint8_t> b; append_i32(b, v); return b; }
std::vector<std::uint8_t> pack_str(const std::string& s) { std::vector<std::uint8_t> b; append_str(b, s); return b; }

bool take_u8(const std::vector<std::uint8_t>& b, std::size_t& p, std::uint8_t& v) { if (p + 1 > b.size()) return false; v = b[p++]; return true; }
bool take_u32(const std::vector<std::uint8_t>& b, std::size_t& p, std::uint32_t& v) { if (p + 4 > b.size()) return false; v = 0; for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(b[p++]) << (8 * i); return true; }
bool take_u64(const std::vector<std::uint8_t>& b, std::size_t& p, std::uint64_t& v) { if (p + 8 > b.size()) return false; v = 0; for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[p++]) << (8 * i); return true; }
bool take_i32(const std::vector<std::uint8_t>& b, std::size_t& p, std::int32_t& v) { std::uint32_t t = 0; if (!take_u32(b, p, t)) return false; v = static_cast<std::int32_t>(t); return true; }
bool take_str(const std::vector<std::uint8_t>& b, std::size_t& p, std::string& v) {
  std::uint32_t n = 0; if (!take_u32(b, p, n)) return false;
  if (n > kMaxFrame) return false;
  if (p + n > b.size()) return false;
  v.assign(reinterpret_cast<const char*>(&b[p]), n); p += n; return true;
}

std::vector<std::uint8_t> encode(Msg type, const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> b;
  append_u32(b, kMagic);
  append_u32(b, kVersion);
  append_u8(b, static_cast<std::uint8_t>(type));
  append_u32(b, static_cast<std::uint32_t>(payload.size()));
  for (std::uint8_t x : payload) b.push_back(x);
  append_u32(b, crc32(b.data() + 4, b.size() - 4));  // crc over bytes after magic (version..payload)
  return b;
}

bool decode(const std::vector<std::uint8_t>& buf, std::size_t& consumed, Frame& out) {
  consumed = 0;
  if (buf.size() < 4 + 4 + 1 + 4 + 4) return false;
  std::size_t p = 0;
  std::uint32_t magic = 0; if (!take_u32(buf, p, magic)) return false;
  if (magic != kMagic) return false;
  std::uint32_t ver = 0; if (!take_u32(buf, p, ver)) return false;
  if (ver != kVersion) return false;
  std::uint8_t t = 0; if (!take_u8(buf, p, t)) return false;
  std::uint32_t len = 0; if (!take_u32(buf, p, len)) return false;
  if (len > kMaxFrame) return false;
  if (p + len + 4 > buf.size()) return false;
  const std::size_t payload_start = p;   // 13
  p += len;                              // now at the crc field
  std::uint32_t crcval = 0; if (!take_u32(buf, p, crcval)) return false;
  const std::uint32_t expect_crc = crc32(buf.data() + 4, (payload_start + len) - 4);
  if (crcval != expect_crc) return false;
  out.type = static_cast<Msg>(t);
  out.payload.assign(buf.begin() + static_cast<std::ptrdiff_t>(payload_start),
                     buf.begin() + static_cast<std::ptrdiff_t>(payload_start + len));
  consumed = p;
  return true;
}

}  // namespace gpu_memory_service::mp
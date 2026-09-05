#include "gpu_memory_service/protocol.hpp"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace gpu_memory_service::mp;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

// Regression: a payload that ends with an empty string must decode without
// aborting.  take_str reads the 4-byte length (0) and advances p to the very
// end of the payload; the old '&b[p]' implementation then indexed
// vector::operator[] at p == size(), which MSVC's Debug STL bounds-checks and
// aborts (exit code 3).  Release does not bounds-check, so it silently worked.
int main() {
  // 1) Exact worker ALLOCATE_RESULT decode: {id, gen, status, ""}.
  {
    std::vector<std::uint8_t> p;
    append_u64(p, 1); append_u64(p, 2); append_u32(p, 0); append_str(p, "");
    std::size_t q = 0;
    std::uint64_t a = 0, b = 0; std::uint32_t st = 999; std::string e = "x";
    bool ok = take_u64(p, q, a) && take_u64(p, q, b) && take_u32(p, q, st) && take_str(p, q, e);
    check(ok, "decode payload with trailing empty err string");
    check(a == 1 && b == 2, "id/gen preserved");
    check(st == 0, "status preserved");
    check(e.empty(), "empty err parsed as empty");
    check(q == p.size(), "offset consumed exactly payload size");
  }
  // 2) Round-trip an ALLOCATE_RESULT frame whose error string is empty.
  {
    std::vector<std::uint8_t> payload;
    append_u64(payload, 7); append_u64(payload, 3); append_u32(payload, 0); append_str(payload, "");
    auto enc = encode(Msg::ALLOCATE_RESULT, payload);
    Frame f; std::size_t used = 0;
    bool ok = decode(enc, used, f);
    check(ok, "decode ALLOCATE_RESULT frame");
    check(f.type == Msg::ALLOCATE_RESULT, "message type preserved");
    std::size_t q = 0; std::uint64_t a = 0, b = 0; std::uint32_t st = 999; std::string e;
    ok = take_u64(f.payload, q, a) && take_u64(f.payload, q, b) && take_u32(f.payload, q, st) && take_str(f.payload, q, e);
    check(ok, "re-parse ALLOCATE_RESULT payload");
    check(a == 7 && b == 3 && st == 0 && e.empty(), "ALLOCATE_RESULT fields preserved");
  }
  // 3) Non-empty error string must still round-trip (prior working path).
  {
    std::vector<std::uint8_t> payload;
    append_u64(payload, 1); append_u64(payload, 1); append_u32(payload, 0xFFFFFFF6u); append_str(payload, "worker dead");
    auto enc = encode(Msg::ALLOCATE_RESULT, payload);
    Frame f; std::size_t used = 0;
    bool ok = decode(enc, used, f);
    check(ok, "decode frame with non-empty err");
    std::size_t q = 0; std::uint64_t a = 0, b = 0; std::uint32_t st = 999; std::string e;
    ok = take_u64(f.payload, q, a) && take_u64(f.payload, q, b) && take_u32(f.payload, q, st) && take_str(f.payload, q, e);
    check(ok, "re-parse non-empty err");
    check(a == 1 && b == 1 && st == 0xFFFFFFF6u && e == "worker dead", "non-empty err preserved");
  }
  // 4) take_str directly at end-of-payload (exact crash condition: p==size()).
  {
    std::vector<std::uint8_t> p;
    append_u32(p, 0);  // 4-byte length field = 0, empty string at tail
    std::size_t q = 0; std::string e = "unset";
    bool ok = take_str(p, q, e);
    check(ok, "take_str empty at tail succeeds");
    check(e.empty(), "take_str empty at tail produced empty string");
    check(q == p.size(), "take_str empty consumed past length");
  }
  if (failures == 0) { std::printf("protocol test: PASS\n"); return 0; }
  std::printf("protocol test: %d FAILURES\n", failures);
  return 1;
}

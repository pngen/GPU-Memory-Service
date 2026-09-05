#include "gpu_memory_service/persistence.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <bit>
#include <fstream>
#include <limits>
#include <set>

namespace gpu_memory_service::persist {

namespace {

// ---- CRC32 (IEEE, reflected) --------------------------------------------
std::uint32_t crc32(const std::uint8_t* data, std::size_t len) noexcept {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (int k = 0; k < 8; ++k) {
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

// ---- Writer --------------------------------------------------------------
class Writer {
 public:
  void u8(std::uint8_t v) { buf_.push_back(v); }
  void u32(std::uint32_t v) { for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF)); }
  void u64(std::uint64_t v) { for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF)); }
  void str(const std::string& s) { u32(static_cast<std::uint32_t>(s.size())); for (char c : s) buf_.push_back(static_cast<std::uint8_t>(c)); }
  void bytes(const std::uint8_t* p, std::size_t n) { for (std::size_t i = 0; i < n; ++i) buf_.push_back(p[i]); }
  [[nodiscard]] const std::vector<std::uint8_t>& data() const noexcept { return buf_; }
  std::uint32_t crc() const noexcept { return crc32(buf_.data(), buf_.size()); }
 private:
  std::vector<std::uint8_t> buf_;
};

// ---- Reader --------------------------------------------------------------
class Reader {
 public:
  explicit Reader(const std::vector<std::uint8_t>& b) : b_(b) {}
  bool u8(std::uint8_t* v) { if (pos_ + 1 > b_.size()) return false; *v = b_[pos_++]; return true; }
  bool u32(std::uint32_t* v) { if (pos_ + 4 > b_.size()) return false; std::uint32_t r = 0; for (int i = 0; i < 4; ++i) r |= static_cast<std::uint32_t>(b_[pos_++]) << (8 * i); *v = r; return true; }
  bool u64(std::uint64_t* v) { if (pos_ + 8 > b_.size()) return false; std::uint64_t r = 0; for (int i = 0; i < 8; ++i) r |= static_cast<std::uint64_t>(b_[pos_++]) << (8 * i); *v = r; return true; }
  bool str(std::string* s) { std::uint32_t n = 0; if (!u32(&n)) return false; if (n > 1u << 20) return false; if (pos_ + n > b_.size()) return false; s->assign(reinterpret_cast<const char*>(&b_[pos_]), n); pos_ += n; return true; }
  [[nodiscard]] std::size_t pos() const noexcept { return pos_; }
  [[nodiscard]] std::size_t remaining() const noexcept { return b_.size() - pos_; }
 private:
  const std::vector<std::uint8_t>& b_;
  std::size_t pos_{0};
};

// ---- Enum validity ranges -----------------------------------------------
template <typename E>
[[nodiscard]] bool enum_ok(E e, std::uint8_t max) noexcept {
  return static_cast<std::uint8_t>(e) <= max;
}

void write_record(Writer& w, const AllocationRecord& r) {
  w.u64(r.id.value()); w.u64(r.generation.value());
  w.u64(r.requested_bytes); w.u64(r.region_size); w.u64(r.offset); w.u64(r.alignment);
  w.u64(r.region_id.value()); w.u64(r.region_generation.value());
  w.u8(static_cast<std::uint8_t>(r.mem_class));
  w.u8(static_cast<std::uint8_t>(r.reclaimability));
  w.u8(static_cast<std::uint8_t>(r.movability));
  w.u32(r.pin_count);
  w.u8(static_cast<std::uint8_t>(r.residency));
  w.u8(static_cast<std::uint8_t>(r.lifecycle));
  w.str(r.owner);
  w.u64(r.worker.value()); w.u64(r.worker_boot.value()); w.u64(r.source_boot.value());
  w.u64(r.workload.value()); w.u64(r.execution.value());
  w.u64(r.relocation_generation.value());
  w.u64(r.reservation_gen.value()); w.u64(r.resource_claim_gen.value());
  w.u64(r.placement_gen.value()); w.u64(r.policy_gen.value());
  w.str(r.backing_ref);
  w.u8(static_cast<std::uint8_t>(r.provenance));
  w.u8(r.recoverable ? 1 : 0);
}

bool read_record(Reader& r, AllocationRecord* out) {
  std::uint64_t t = 0;
  if (!r.u64(&t)) return false; out->id = AllocationId(t);
  if (!r.u64(&t)) return false; out->generation = AllocationGeneration(t);
  if (!r.u64(&out->requested_bytes)) return false;
  if (!r.u64(&out->region_size)) return false;
  if (!r.u64(&out->offset)) return false;
  if (!r.u64(&out->alignment)) return false;
  if (!r.u64(&t)) return false; out->region_id = MemoryRegionId(t);
  if (!r.u64(&t)) return false; out->region_generation = MemoryRegionGeneration(t);
  std::uint8_t e = 0;
  if (!r.u8(&e)) return false; out->mem_class = static_cast<MemoryClass>(e);
  if (!r.u8(&e)) return false; out->reclaimability = static_cast<Reclaimability>(e);
  if (!r.u8(&e)) return false; out->movability = static_cast<Movability>(e);
  if (!r.u32(&out->pin_count)) return false;
  if (!r.u8(&e)) return false; out->residency = static_cast<ResidencyState>(e);
  if (!r.u8(&e)) return false; out->lifecycle = static_cast<LifecycleState>(e);
  if (!r.str(&out->owner)) return false;
  if (!r.u64(&t)) return false; out->worker = WorkerId(t);
  if (!r.u64(&t)) return false; out->worker_boot = WorkerBootId(t);
  if (!r.u64(&t)) return false; out->source_boot = SourceBootId(t);
  if (!r.u64(&t)) return false; out->workload = WorkloadId(t);
  if (!r.u64(&t)) return false; out->execution = ExecutionId(t);
  if (!r.u64(&t)) return false; out->relocation_generation = RelocationGeneration(t);
  if (!r.u64(&t)) return false; out->reservation_gen = ReservationGeneration(t);
  if (!r.u64(&t)) return false; out->resource_claim_gen = ResourceClaimGeneration(t);
  if (!r.u64(&t)) return false; out->placement_gen = PlacementGeneration(t);
  if (!r.u64(&t)) return false; out->policy_gen = PolicyGeneration(t);
  if (!r.str(&out->backing_ref)) return false;
  if (!r.u8(&e)) return false; out->provenance = static_cast<Provenance>(e);
  std::uint8_t rec = 0;
  if (!r.u8(&rec)) return false; out->recoverable = rec == 1;
  return true;
}

void write_domain(Writer& w, const MemoryDomain& d) {
  w.u64(d.domain_id.value()); w.u64(d.domain_generation.value());
  w.u64(d.device_id.value()); w.u64(d.device_generation.value());
  w.u64(d.total_capacity); w.u64(d.allocator_visible_capacity); w.u64(d.reserved_headroom); w.u64(d.managed_capacity);
  w.u64(d.current_committed); w.u64(d.current_physically_allocated); w.u64(d.current_resident);
  w.u64(d.current_reclaimable); w.u64(d.current_movable); w.u64(d.current_pinned);
  double frag = d.external_fragmentation; w.u64(static_cast<std::uint64_t>(std::bit_cast<std::uint64_t>(frag)));
  w.u8(static_cast<std::uint8_t>(d.pressure)); w.u8(static_cast<std::uint8_t>(d.health)); w.u8(static_cast<std::uint8_t>(d.provenance));
  w.u64(d.watermarks.generation.value());
  std::uint64_t l = static_cast<std::uint64_t>(std::bit_cast<std::uint64_t>(d.watermarks.low));
  std::uint64_t h = static_cast<std::uint64_t>(std::bit_cast<std::uint64_t>(d.watermarks.high));
  std::uint64_t c = static_cast<std::uint64_t>(std::bit_cast<std::uint64_t>(d.watermarks.critical));
  w.u64(l); w.u64(h); w.u64(c);
}

bool read_domain(Reader& r, MemoryDomain* d) {
  std::uint64_t t = 0;
  if (!r.u64(&t)) return false; d->domain_id = GpuMemoryDomainId(t);
  if (!r.u64(&t)) return false; d->domain_generation = GpuMemoryDomainGeneration(t);
  if (!r.u64(&t)) return false; d->device_id = DeviceId(t);
  if (!r.u64(&t)) return false; d->device_generation = DeviceGeneration(t);
  if (!r.u64(&d->total_capacity)) return false;
  if (!r.u64(&d->allocator_visible_capacity)) return false;
  if (!r.u64(&d->reserved_headroom)) return false;
  if (!r.u64(&d->managed_capacity)) return false;
  if (!r.u64(&d->current_committed)) return false;
  if (!r.u64(&d->current_physically_allocated)) return false;
  if (!r.u64(&d->current_resident)) return false;
  if (!r.u64(&d->current_reclaimable)) return false;
  if (!r.u64(&d->current_movable)) return false;
  if (!r.u64(&d->current_pinned)) return false;
  std::uint64_t fragbits = 0; if (!r.u64(&fragbits)) return false; d->external_fragmentation = std::bit_cast<double>(fragbits);
  std::uint8_t e = 0;
  if (!r.u8(&e)) return false; d->pressure = static_cast<PressureState>(e);
  if (!r.u8(&e)) return false; d->health = static_cast<DeviceHealth>(e);
  if (!r.u8(&e)) return false; d->provenance = static_cast<Provenance>(e);
  if (!r.u64(&t)) return false; d->watermarks.generation = WatermarkGeneration(t);
  std::uint64_t l = 0, h = 0, c = 0; if (!r.u64(&l)) return false; if (!r.u64(&h)) return false; if (!r.u64(&c)) return false;
  d->watermarks.low = std::bit_cast<double>(l); d->watermarks.high = std::bit_cast<double>(h); d->watermarks.critical = std::bit_cast<double>(c);
  return true;
}

[[nodiscard]] bool validate_record(const AllocationRecord& r, std::uint64_t managed) {
  if (r.requested_bytes == 0) return false;
  if (r.alignment == 0 || (r.alignment & (r.alignment - 1)) != 0) return false;
  if (r.region_size < r.requested_bytes) return false;
  if (r.offset > managed || r.region_size > managed - r.offset) return false;
  if (r.offset % r.alignment != 0) return false;
  if (!enum_ok(r.mem_class, static_cast<std::uint8_t>(MemoryClass::INTERNAL))) return false;
  if (!enum_ok(r.reclaimability, static_cast<std::uint8_t>(Reclaimability::BEST_EFFORT))) return false;
  if (!enum_ok(r.movability, static_cast<std::uint8_t>(Movability::NONMOVABLE))) return false;
  if (!enum_ok(r.residency, static_cast<std::uint8_t>(ResidencyState::REVALIDATION_REQUIRED))) return false;
  if (!enum_ok(r.lifecycle, static_cast<std::uint8_t>(LifecycleState::RETIRED))) return false;
  if (!enum_ok(r.provenance, static_cast<std::uint8_t>(Provenance::RECONSTRUCTED))) return false;
  if (r.generation.value() == 0) return false;
  return true;
}

}  // namespace

std::vector<std::uint8_t> encode(const PersistSnapshot& s) {
  Writer w;
  w.u32(kMagic); w.u32(kVersion);
  write_domain(w, s.domain);
  w.u32(static_cast<std::uint32_t>(s.records.size()));
  for (const auto& r : s.records) write_record(w, r);
  w.u32(static_cast<std::uint32_t>(s.retired.size()));
  for (const auto& r : s.retired) write_record(w, r);
  w.u64(s.next_alloc_id); w.u64(s.history_count);
  w.u32(w.crc());
  return w.data();
}

Result<PersistSnapshot> decode(const std::vector<std::uint8_t>& b) {
  Reader r(b);
  std::uint32_t magic = 0, ver = 0;
  if (!r.u32(&magic)) return Failure(RejectReason::INTERNAL_ERROR, "truncated magic");
  if (magic != kMagic) return Failure(RejectReason::INTERNAL_ERROR, "bad magic");
  if (!r.u32(&ver)) return Failure(RejectReason::INTERNAL_ERROR, "truncated version");
  if (ver != kVersion) return Failure(RejectReason::INTERNAL_ERROR, "unsupported version");

  PersistSnapshot s;
  if (!read_domain(r, &s.domain)) return Failure(RejectReason::INTERNAL_ERROR, "malformed domain");
  if (!s.domain.watermarks.valid()) return Failure(RejectReason::INTERNAL_ERROR, "invalid watermarks (NaN/Inf/bad order)");

  std::uint32_t rocount = 0, trcount = 0;
  if (!r.u32(&rocount)) return Failure(RejectReason::INTERNAL_ERROR, "truncated record count");
  if (rocount > kMaxRecords) return Failure(RejectReason::INTERNAL_ERROR, "hostile record count");
  s.records.reserve(rocount);
  for (std::uint32_t i = 0; i < rocount; ++i) {
    AllocationRecord rec;
    if (!read_record(r, &rec)) return Failure(RejectReason::INTERNAL_ERROR, "malformed record");
    s.records.push_back(rec);
  }
  if (!r.u32(&trcount)) return Failure(RejectReason::INTERNAL_ERROR, "truncated retired count");
  if (trcount > kMaxRecords) return Failure(RejectReason::INTERNAL_ERROR, "hostile retired count");
  s.retired.reserve(trcount);
  for (std::uint32_t i = 0; i < trcount; ++i) {
    AllocationRecord rec;
    if (!read_record(r, &rec)) return Failure(RejectReason::INTERNAL_ERROR, "malformed retired record");
    s.retired.push_back(rec);
  }
  if (!r.u64(&s.next_alloc_id)) return Failure(RejectReason::INTERNAL_ERROR, "truncated next alloc id");
  if (!r.u64(&s.history_count)) return Failure(RejectReason::INTERNAL_ERROR, "truncated history");

  // Checksum + trailing-garbage.
  std::uint32_t stored = 0;
  if (!r.u32(&stored)) return Failure(RejectReason::INTERNAL_ERROR, "truncated checksum");
  if (r.remaining() != 0) return Failure(RejectReason::INTERNAL_ERROR, "trailing garbage");
  if (stored != crc32(b.data(), b.size() - 4)) return Failure(RejectReason::INTERNAL_ERROR, "checksum mismatch");

  const std::uint64_t managed = s.domain.managed_capacity;
  std::set<AllocationId> seen;
  std::uint64_t sum_live = 0;
  for (const auto& rec : s.records) {
    if (!validate_record(rec, managed)) return Failure(RejectReason::INTERNAL_ERROR, "invalid record");
    if (!seen.insert(rec.id).second) return Failure(RejectReason::INTERNAL_ERROR, "duplicate allocation id");
    sum_live += rec.region_size;
  }
  for (const auto& rec : s.retired) {
    if (!validate_record(rec, managed)) return Failure(RejectReason::INTERNAL_ERROR, "invalid retired record");
    if (!seen.insert(rec.id).second) return Failure(RejectReason::INTERNAL_ERROR, "duplicate allocation id");
  }
  if (sum_live > managed) return Failure(RejectReason::INTERNAL_ERROR, "physical bytes exceed governed capacity");

  // Overlap detection for live records.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> iv;
  for (const auto& rec : s.records) iv.emplace_back(rec.offset, rec.offset + rec.region_size);
  std::sort(iv.begin(), iv.end());
  for (std::size_t i = 1; i < iv.size(); ++i) {
    if (iv[i].first < iv[i - 1].second) return Failure(RejectReason::INTERNAL_ERROR, "overlapping persisted regions");
  }

  return s;
}

Result<void> write_file(const std::string& path, const PersistSnapshot& s) {
  const auto bytes = encode(s);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return Failure(RejectReason::INTERNAL_ERROR, "cannot open file for write");
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!f) return Failure(RejectReason::INTERNAL_ERROR, "write failed");
  return Result<void>();
}

Result<PersistSnapshot> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return Failure(RejectReason::INTERNAL_ERROR, "cannot open file for read");
  const auto size = f.tellg();
  if (size <= 0) return Failure(RejectReason::INTERNAL_ERROR, "empty file");
  std::vector<std::uint8_t> b(static_cast<std::size_t>(size));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(b.data()), size);
  if (!f) return Failure(RejectReason::INTERNAL_ERROR, "read failed");
  return decode(b);
}

}  // namespace gpu_memory_service::persist
#pragma once
// Structured result / failure types.  Every fallible operation returns a typed
// explanation rather than a bare bool, so callers can distinguish
// GOVERNED_CAPACITY_EXCEEDED from CONTIGUOUS_CAPACITY_UNAVAILABLE from
// PHYSICAL_OOM without parsing strings.

#include <optional>
#include <string>
#include <utility>

#include "enum.hpp"

namespace gpu_memory_service {

// ---------------------------------------------------------------------------
// Failure — a typed reason plus a deterministic human-readable explanation.
// ---------------------------------------------------------------------------
struct Failure {
  RejectReason reason{RejectReason::NONE};
  std::string message;

  Failure() = default;
  explicit Failure(RejectReason r, std::string msg = {})
      : reason(r), message(std::move(msg)) {}

  [[nodiscard]] bool ok() const noexcept { return reason == RejectReason::NONE; }
};

// ---------------------------------------------------------------------------
// Result<T> — either a value or a typed failure.
// ---------------------------------------------------------------------------
template <typename T>
class Result {
 public:
  Result() = default;
  Result(T v) : value_(std::move(v)) {}                                  // NOLINT(google-explicit-constructor)
  Result(Failure f) : failure_(std::move(f)) {}                           // NOLINT(google-explicit-constructor)
  Result(RejectReason r, std::string m = {}) : failure_(r, std::move(m)) {} // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }
  [[nodiscard]] bool ok() const noexcept { return value_.has_value(); }
  [[nodiscard]] explicit operator bool() const noexcept { return value_.has_value(); }

  [[nodiscard]] const T& value() const& { return *value_; }
  [[nodiscard]] T& value() & { return *value_; }
  [[nodiscard]] T&& value() && { return std::move(*value_); }

  [[nodiscard]] const Failure& error() const& { return failure_; }
  [[nodiscard]] Failure& error() & { return failure_; }

  [[nodiscard]] const T& operator*() const& { return *value_; }
  [[nodiscard]] T& operator*() & { return *value_; }
  [[nodiscard]] const T* operator->() const& { return &*value_; }
  [[nodiscard]] T* operator->() & { return &*value_; }

 private:
  std::optional<T> value_;
  Failure failure_;
};

// ---------------------------------------------------------------------------
// Result<void> specialization — no value; ok() means no failure.
// ---------------------------------------------------------------------------
template <>
class Result<void> {
 public:
  Result() = default;
  Result(Failure f) : failure_(std::move(f)) {}  // NOLINT(google-explicit-constructor)
  Result(RejectReason r, std::string m = {}) : failure_(r, std::move(m)) {}  // NOLINT(google-explicit-constructor)

  [[nodiscard]] bool ok() const noexcept { return failure_.ok(); }
  [[nodiscard]] bool has_value() const noexcept { return failure_.ok(); }
  [[nodiscard]] explicit operator bool() const noexcept { return failure_.ok(); }
  [[nodiscard]] const Failure& error() const& { return failure_; }

 private:
  Failure failure_;
};

}  // namespace gpu_memory_service
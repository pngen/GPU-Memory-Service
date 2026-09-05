#include "gpu_memory_service/pressure.hpp"

namespace gpu_memory_service {

PressureState classify_pressure(double utilization, const WatermarkConfig& wm,
                                PressureState last) noexcept {
  // While in (or above) HIGH, stay HIGH until utilization falls below the LOW
  // watermark (hysteresis).  This prevents oscillation around a threshold.
  if (last == PressureState::HIGH || last == PressureState::CRITICAL ||
      last == PressureState::RECLAIMING) {
    if (utilization < wm.low) {
      return utilization < wm.low * 0.5 ? PressureState::LOW : PressureState::NORMAL;
    }
    if (utilization >= wm.critical) return PressureState::CRITICAL;
    if (utilization >= wm.high) return PressureState::HIGH;
    return PressureState::HIGH;  // held by hysteresis
  }

  if (utilization >= wm.critical) return PressureState::CRITICAL;
  if (utilization >= wm.high) return PressureState::HIGH;
  if (utilization >= wm.low) return PressureState::NORMAL;
  return utilization < wm.low * 0.5 ? PressureState::LOW : PressureState::NORMAL;
}

PressureState evaluate_pressure(const PressureEvidence& ev, PressureState last) noexcept {
  // UNKNOWN can never silently become NORMAL: with no managed capacity or with
  // an UNKNOWN prior state, we stay UNKNOWN unless real evidence exists.
  if (ev.managed_capacity == 0) return PressureState::UNKNOWN;
  const double utilization = static_cast<double>(ev.committed) /
                             static_cast<double>(ev.managed_capacity);
  return classify_pressure(utilization, ev.watermarks, last);
}

}  // namespace gpu_memory_service

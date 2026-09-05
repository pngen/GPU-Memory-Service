#include "gpu_memory_service/pressure.hpp"
#include <cmath>
#include <limits>
#include <cstdio>

using namespace gpu_memory_service;

static int failures = 0;
static void check(bool c, const char* m) { if (!c) { std::printf("FAIL: %s\n", m); ++failures; } }

int main() {
  WatermarkConfig wm;
  check(wm.valid(), "default watermarks valid");

  // Watermark validation rejects bad order / NaN / Inf.
  WatermarkConfig bad = wm; bad.high = 0.4; bad.low = 0.9;
  check(!bad.valid(), "reject high < low");
  WatermarkConfig nan = wm; nan.critical = std::nan("");
  check(!nan.valid(), "reject NaN critical");
  WatermarkConfig inf = wm; inf.low = std::numeric_limits<double>::infinity();
  check(!inf.valid(), "reject Inf low");

  // Baseline at NORMAL -> LOW below low*0.5, NORMAL otherwise.
  check(classify_pressure(0.25, wm, PressureState::NORMAL) == PressureState::LOW, "low util -> LOW");
  check(classify_pressure(0.60, wm, PressureState::NORMAL) == PressureState::NORMAL, "mid util -> NORMAL");
  check(classify_pressure(0.95, wm, PressureState::NORMAL) == PressureState::HIGH, "above high -> HIGH");
  check(classify_pressure(0.98, wm, PressureState::NORMAL) == PressureState::CRITICAL, "above critical -> CRITICAL");

  // Hysteresis: once HIGH, stay HIGH until drop below LOW.
  check(classify_pressure(0.80, wm, PressureState::HIGH) == PressureState::HIGH, "hysteresis stays HIGH at high");
  check(classify_pressure(0.82, wm, PressureState::HIGH) == PressureState::HIGH, "hysteresis stays HIGH");
  check(classify_pressure(0.40, wm, PressureState::HIGH) == PressureState::NORMAL, "below LOW drops to NORMAL");
  check(classify_pressure(0.10, wm, PressureState::HIGH) == PressureState::LOW, "well below LOW -> LOW");

  // UNKNOWN never silently becomes NORMAL.
  PressureEvidence ev;
  ev.managed_capacity = 0;
  check(evaluate_pressure(ev, PressureState::UNKNOWN) == PressureState::UNKNOWN, "no evidence stays UNKNOWN");

  if (failures == 0) { std::printf("pressure test: PASS\n"); return 0; }
  std::printf("pressure test: %d FAILURES\n", failures);
  return 1;
}
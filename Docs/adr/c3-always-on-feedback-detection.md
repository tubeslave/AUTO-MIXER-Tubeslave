# ADR: C3 — always-on feedback detection service

**Date:** 2026-07-19
**Status:** Accepted (operator directive "продолжай", 2026-07-19)
**Authors:** Kimi (implementation), review requested from Codex
**Refs:** `.ai/briefs/c3-always-on-feedback-detection.md`, `.ai/proposals/c3-always-on-feedback-detection-kimi.md`, review finding C3 (2026-07-19), P0 ADR `Docs/adr/p0-secret-hygiene-feedback-failsafe.md`

## 1. Context

`server.py` declared `self.feedback_detector = None` (line 977) and never instantiated it: continuous feedback suppression existed only inside `AutoSoundcheckEngine._monitor_loop`, i.e. only during soundcheck runs. During normal server operation — the mode the console is actually mixed with — there was no continuous feedback detection, contradicting the project canon "feedback first: absolute priority".

## 2. Options considered

A. Detection + operator alerting only, no console writes (chosen).
B. Detection + automatic bounded fader reduction in v1.
C. Reuse the M110 one-shot recheck (`_m110_feedback_clear_recheck`) as a periodic job.
D. Move the monitor loop into a WS handlers module.

## 3. Decision

Option A implemented as a self-contained `backend/feedback_live_monitor.py`:

- `FeedbackLiveMonitor` daemon thread (engine-style; detector is CPU-bound numpy) runs `FeedbackDetector.process_audio` over every captured channel above −50 dB peak, using the server's single shared `AudioCapture` (thread-safe for external readers; detector serialized by the service's own lock).
- Alerts are rate-limited (per channel/50 Hz bucket/action: 5 s cooldown; global 30/min flood guard) and broadcast to operator clients as additive `feedback_alert` messages (`recommended_action`, `auto_action: "none"`, `read_only: true`) via the canonical `loop.call_soon_threadsafe` + `asyncio.create_task(self.broadcast(...))` pattern.
- Fail-loud error handling mirrors the P0 engine patch: `logger.exception` with consecutive counter, one `logger.critical` at 10 consecutive failures, loop stays alive; idle 1 s cadence when capture is absent, 0.1 s when active.
- Server wiring: `"feedback_live"` added to the yaml merge tuple; `_start_feedback_live_monitor()` called in `start()` before `_shutdown_event.wait()`; `_stop_feedback_live_monitor()` in `cleanup_all_controllers` **before** audio capture teardown; config in `config/automixer.yaml` `feedback_live:` with detector defaults mirroring engine tuning.

## 4. Why this won

- Closes the safety gap immediately with zero console-write risk: every gated write path in server.py requires an operator-bound context and fails closed otherwise, so a background service has no safe write path today; alerting is the maximal safe action.
- No new audio streams (convention: AudioCapture is the single service), additive WS message (no frontend change needed), smallest safe diff in server.py (7 touches, ~50 lines).

## 5. Rejected alternatives

- **B (auto fader in v1):** the "feedback first" canon grants the *right* to reduce faders, but implementing it requires either an operator-free emergency write scope or weakening M-hardened gates — a policy decision for the Council, tracked as follow-up (candidate: standing engine-owned executor with `RuntimeState.EMERGENCY_FEEDBACK`).
- **C (reuse recheck):** synchronous one-shot verifier, wrong lifecycle for continuous monitoring.
- **D (handlers):** the monitor is not a WS route; lifecycle ownership belongs with engine/capture ownership in the server.

## 6. Implementation plan

Done in this change: new module, server wiring, yaml section, tests, this ADR. Follow-ups: Council decision on auto fader action; optional frontend rendering of `feedback_alert`; optional dedupe with the engine's event stream during soundcheck (double alerting is acceptable: engine suppresses, service alerts).

## 7. Test plan

Executed: `tests/test_feedback_live_monitor.py` — 14 tests (synthetic growing pure tone per the proven M110 fixture → alert with correct channel/frequency; silence and steady noise → no alerts; exclusion, rate limits, lifecycle, sink-failure resilience, config parsing) — **14 passed**. Regression: `tests/test_server.py`, `test_feedback_detector.py`, `test_feedback_tonality_criteria.py`, `test_auto_soundcheck_engine*.py`, `test_meter_stream.py`, `test_ws_transport.py` — **550 passed**. `py_compile` clean for `server.py` and the new module.

## 8. Risks and rollback

- CPU load equals the engine monitor's proven load (48 ch × 10 Hz × 2048-FFT); idle path near-zero.
- Double alerting with the engine during soundcheck: acceptable, documented above.
- Rollback: revert the two code commits; the service is self-contained and config-gated (`feedback_live.enabled: false` disables it without a revert).

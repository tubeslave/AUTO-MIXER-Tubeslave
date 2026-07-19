# Proposal: C3 — always-on feedback detection service (Kimi)

## 1. Short thesis

Close the protection gap with a small, self-contained `backend/feedback_live_monitor.py` service that runs the proven `FeedbackDetector` against the existing single `AudioCapture` whenever capture is active, and broadcasts `feedback_alert` messages to operator clients. **v1 performs no console writes** — detection + operator alerting only; auto fader action is left as an explicit follow-up decision.

## 2. What I understood

- `server.py:977` declares `self.feedback_detector = None` and never fills it; live suppression exists only inside `AutoSoundcheckEngine._monitor_loop` (soundcheck runs).
- Integration precedent already exists in server.py: `_m110_feedback_clear_recheck` (23549) builds `FeedbackDetector(config=...)` and calls `process_audio(channel, frame)` on 2048-sample frames @ 48 kHz.
- `AudioCapture` is thread-safe for external readers (`_data_lock`, copies on read); the detector itself is not, so the service serializes detector access with its own lock (engine does the same via `_monitor_iteration_lock`).
- Broadcast from a non-asyncio thread must go through `loop.call_soon_threadsafe(lambda: asyncio.create_task(self.broadcast(...)))` (canonical pattern, server.py:12265–12275).
- All gated console-write paths (`_apply_manual_console_write*`) require an operator-bound websocket context and fail closed otherwise; the engine `_execute_action` path requires a running engine. A background service therefore has **no safe write path** today — any auto-fader action would either fail closed or require weakening M-hardened gates.
- `self.config` merges only selected yaml keys (`_merge_automixer_yaml`, 11837); adding `feedback_live` there is a one-tuple change.

## 3. Proposed solution

1. New module `backend/feedback_live_monitor.py`: `FeedbackLiveMonitorConfig` (with `from_dict`) + `FeedbackLiveMonitor`. Daemon thread (engine-style, not asyncio — detector is CPU-bound numpy), `run_once()` exposed for deterministic tests. Per-iteration: skip channels below `signal_threshold_db` (-50 dB, engine's SIGNAL_THRESHOLD_DB), `process_audio` under lock, emit alert.
2. Alert rate limiting: per (channel, 50 Hz bucket, action) cooldown `alert_min_interval_sec=5.0` + global `max_alerts_per_minute=30` flood guard. Alert payload: `type=feedback_alert`, channel, frequency_hz, magnitude_db, recommended_action, confidence, `auto_action="none"`, `read_only=true`, timestamp.
3. Fail-loud: `logger.exception` per error with consecutive counter; one `logger.critical` at 10 consecutive (same pattern as the P0 engine patch). Loop stays alive; capture-absent idle at `idle_interval_sec=1.0`.
4. Server wiring (6 small touches): import; `self.feedback_live_monitor` attr; `"feedback_live"` in yaml merge tuple; `_start/_stop_feedback_live_monitor` methods (alert sink captures the running loop at `start()`); start in `start()` before `_shutdown_event.wait()`; stop in `cleanup_all_controllers` **before** audio capture teardown.
5. `config/automixer.yaml`: new `feedback_live:` section, detector defaults mirroring engine tuning (persistence 12, confidence 0.75, height −12, prominence 10), `enabled: true`, `auto_apply: false` (reserved).

## 4. Likely files to touch

- new `backend/feedback_live_monitor.py`, new `tests/test_feedback_live_monitor.py`
- `backend/server.py` (6 small spots, ~45 lines total), `config/automixer.yaml` (1 section)
- `Docs/adr/c3-always-on-feedback-detection.md`, `.ai/` records

## 5. Alternatives considered

- **Auto fader reduction in v1** — rejected for now: no operator-free safe write path exists (all gates fail closed by design); adding one is a policy decision for the Council, not an implementation detail. The alert path delivers the core value immediately and safely.
- **Asyncio task instead of thread** — rejected: FFT work is blocking numpy; engine precedent is `threading.Thread`.
- **Reuse `_m110_feedback_clear_recheck`** — rejected: it is a synchronous one-shot verifier around corrections, not a continuous monitor; different lifecycle.
- **Put the loop in handlers/** — rejected: it is not a WS route; server lifecycle ownership is the correct place, consistent with engine/capture ownership.

## 6. Risks

- CPU: 48 ch × 10 Hz × 2048-FFT is the same load the engine monitor already runs during soundcheck; idle path is near-zero.
- Alert floods: two-level rate limiting; payload is small; broadcast only to connected clients.
- Double alerting during soundcheck (engine + this service): acceptable — engine suppresses, service alerts; dedupe could be added later via shared event keys.
- Thread racing capture teardown: monitor stops before capture in `cleanup_all_controllers`; reads are copies under capture's lock.

## 7. Test plan

- Synthetic growing pure tone (12×2048 frames, bin 42, amplitude 0.02→0.5 — the proven M110 pattern) advances through a cursor-based FakeCapture → alert emitted with correct channel/frequency.
- Silence and steady pink noise → no alerts. Excluded channels → skipped.
- Rate limits: stubbed detector always-event → second alert inside cooldown suppressed; global cap enforced.
- Lifecycle: disabled config → no thread; start/stop idempotent; sink exception does not kill the loop.
- Regression: `tests/test_server.py`, `tests/test_feedback_detector.py`, `tests/test_auto_soundcheck_engine*.py`, `tests/test_meter_stream.py`.

## 8. Where the other agent may disagree

- Codex may insist v1 must auto-reduce faders per the "feedback first" canon. My position: the canon grants the *right*, but the only fail-closed write paths require operator context; I'd rather ship detection now and add auto-action via a dedicated ADR that designs an explicit emergency write scope (possibly reusing `RuntimeState.EMERGENCY_FEEDBACK` through a standing engine-owned executor).
- Codex may prefer monitoring only channels with routed signal (from mixer state) instead of peak-threshold detection; peak detection is simpler, hardware-independent, and matches the engine.

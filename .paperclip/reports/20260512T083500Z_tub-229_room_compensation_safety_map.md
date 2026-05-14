# TUB-229 Room Compensation Safety Map

Date: 2026-05-12
Agent role: OSC Console Safety Engineer
Scope: repository analysis only, no source/runtime changes

## Executive Summary

- There is no dedicated room-compensation live apply path in the current repository.
- The only explicit room-compensation symbol found in the live/backend Auto-EQ surface is `room_analysis_stub()` in `backend/auto_eq.py`, and it returns `implemented=False`.
- Generic Auto-EQ apply paths can still write EQ to a real mixer, but they do so by calling mixer methods directly, outside the canonical `LiveApplyService` gate.
- Today the approved gated live write contract is intentionally narrow: `set_fader` and `set_gain` only, dry-run by default, explicit confirmation required for real send, mixer readback required for real success, and panic-stop semantics limited to manual live-apply scope.
- For TUB-229, the immediate conclusion is policy/contract oriented: no room-compensation live apply path is approved today, and any future room-compensation write path must not be modeled on the current direct Auto-EQ apply pattern.

## Technical Map

### Entry Points

- Main live backend entry: `backend/server.py`
- Frontend -> backend runtime entry: `frontend/src/services/websocket.js`
- Headless soundcheck entry: `start_soundcheck.py` -> `backend/auto_soundcheck_engine.py`
- Offline-first production mix entry: `python -m automixer.production_mix_v1`
- Virtual console test surface: `virtual_mixer/virtual_mixer.py`

### Audio Input Path

- Canonical capture service: `backend/audio_capture.py`
- Capture model:
  - unified `AudioCapture` service
  - per-channel ring buffers
  - subscriber fan-out
  - device auto-detect prefers SoundGrid, then Dante, then system default
  - fallback test generators when `sounddevice` is unavailable
- Server ownership:
  - `backend/server.py` owns `self.audio_capture`
  - live analyzers and automation are expected to reuse this service

### Analyzer Surfaces

- Live/backend analyzer and controller surfaces:
  - `backend/auto_eq.py`
  - `backend/feedback_detector.py`
  - `backend/arrangement_automation/*`
  - `backend/auto_fader.py`
  - `backend/auto_soundcheck_engine.py`
  - `backend/bleed_service.py`
- Offline/research analyzer surfaces:
  - `automixer/production_mix_v1/*`
  - `ai_mixing_pipeline/*`
- Room-compensation-specific finding:
  - `backend/auto_eq.py:2662` defines `room_analysis_stub()`
  - no real room-RTA or room-compensation apply controller was found in the active live runtime

### Decision Logic

- Canonical safer live decision/apply example:
  - `backend/arrangement_automation/live_input_trim_controller.py`
  - uses bounds, deadband, cooldown, role-aware trim limits, bleed-aware decisions, optional meter trust checks, `LiveApplyService`, and readback tracking
- Overlapping live decision surfaces:
  - `backend/arrangement_automation/controller.py`
  - `backend/auto_fader.py`
  - `backend/auto_soundcheck_engine.py`
- Offline/research decision surfaces:
  - `automixer/production_mix_v1/*`
  - `ai_mixing_pipeline/*`

### OSC Output Path

- Primary live console transport:
  - `backend/wing_client.py`
- Supporting transport wrapper:
  - `backend/osc/enhanced_osc_client.py`
- Legacy/helper communication surface:
  - `backend/osc_manager.py`
- Transport behavior found:
  - Wing handshake: `WING?` to UDP 2222, then OSC on 2223
  - `/xremote` subscription and renewal in `WingClient`
  - per-address OSC throttle in both `WingClient` and `OSCManager`
  - cached state and manual readback confirmation in `WingClient`

## Safety Contract Findings

### Canonical Live Gate

- File: `backend/live_apply.py`
- Current contract:
  - supports only `set_fader` and `set_gain`
  - dry-run by default
  - real write requires explicit confirmation
  - real write requires readback confirmation for success
  - fail-closed on unsupported operations, missing client, send failure, and unconfirmed readback
  - panic-stop semantics are scoped to `manual_live_apply_only`

### Canonical Maintenance Gate

- File: `backend/maintenance_gate.py`
- Used for snapshot/reset/restore style mutations
- Requires:
  - allowed operator source
  - explicit maintenance confirmation
  - rollback snapshot path for mixer-mutating maintenance operations
  - connected transport

### Safer Existing Live Apply Pattern

- File: `backend/arrangement_automation/live_input_trim_controller.py`
- Guards found:
  - per-channel cooldown
  - deadband no-send behavior
  - max step limits
  - role-based trim clamp
  - optional meter-required gate
  - `LiveApplyService` integration
  - readback status captured on result

### Direct or Partially Direct Apply Surfaces

- `backend/auto_eq.py`
  - `apply_to_mixer()` directly calls EQ methods on the mixer client
  - `reset_eq()` directly sends raw OSC addresses
  - multi-channel apply does the same
- `backend/server.py`
  - `apply_eq_correction()` calls `auto_eq_controller.apply_to_mixer()` directly
  - EQ reset helpers still contain direct mixer commands
- `backend/auto_soundcheck_engine.py`
  - direct EQ band writes
  - direct fader writes
  - direct gain writes
  - direct compressor writes
  - feedback reaction path can directly notch EQ or reduce fader
- `backend/arrangement_automation/controller.py`
  - routes through `LiveApplyService` for fader writes, but remains documented as overlapping and non-canonical in current ownership docs

## TUB-229 Specific Finding

- There is no existing approved room-compensation apply path to harden.
- There is also no evidence that room compensation is currently implemented as a live runtime feature.
- The nearest real-write surface is generic Auto-EQ live apply, and that surface bypasses the canonical gate.
- Therefore TUB-229 should be interpreted as a safety-contract definition task for a future capability, not a bugfix in a current room-compensation executor.

## TUB-229 Safety Contract Draft (Room Compensation Apply)

- Scope boundary
  - `backend/system_measurement.py` must remain analysis-only by default.
  - No console mutation from this module is allowed until a dedicated EQ gate exists and is explicitly enabled.
  - `SystemMeasurementController.apply_corrections()` should be treated as draft-only until policy review and gate integration are complete.

- Approved operation envelope (future)
  - Target surfaces: approved bus/channel roles only (Master/Group/Matrix context).
  - Approved operations:
    - `set_eq_on` enable.
    - `set_eq_low_shelf`, `set_eq_high_shelf`, `set_eq_band` via the approved gate.
  - Prohibited until first rollout:
    - raw `send("/ch/...")` and raw `send("/bus/...")` writes for apply intent.
    - free-form `freq`/`q` mutation outside bounded policy.
    - any operation without explicit dry-run/confirmation/readback semantics.

- Mandatory guardrail checks before real send
  - `dry_run` defaults to `true`.
  - `confirm_live_apply` required for real writes.
  - strict source allowlist in request metadata.
  - per-band caps enforced before queueing:
    - gain clamp: `[-6.0, +3.0]` (starter cap).
    - Q clamp: `[0.5, 4.0]`.
    - frequency clamp: clamp to mixer-valid domain.
  - per-band cooldown minimum (e.g., 800ms) and global write-rate guard (e.g., 1 command per 150ms) before sending.
  - max changed bands per apply and max per-channel aggregate gain delta enforced.
  - readback required:
    - failure to verify should return explicit blocked/pending status.
    - confirmed mismatch treated as blocked or high-risk warning by policy.

- Audit schema (structured event requirements)
  - must include: request_id, source, operation, target surface, channel/bus, desired params, dry_run, confirm flag, blocked_reason, rate-limit hits, send result, readback result, and operator id.
  - event type should align with `live_apply_audit`/`maintenance_gate_audit` shape for operational consistency.

- Emergency controls
  - room-compensation path must subscribe to global panic stop and be able to clear pending mutations immediately.
  - while panic-stop is active, all write attempts must be rejected.

## Logging Map

- General runtime logging is still mostly stdlib logging via `logging.getLogger(__name__)`
- `backend/logging_config.py` provides optional structlog-backed structured logging with file output support
- `backend/live_apply.py` emits `live_apply_audit` events through an audit sink
- `backend/maintenance_gate.py` emits `maintenance_gate_audit` events through an audit sink
- `backend/arrangement_automation/live_input_trim_controller.py` carries `audit_id`, send status, and readback status through decision payloads
- Current state:
  - structured logging primitives exist
  - structured audit is stronger in gated paths than in direct EQ apply paths

## Test Map

- Safety and gating coverage present:
  - `tests/test_live_apply_contract.py`
  - `tests/test_maintenance_gate.py`
  - `tests/test_snapshot_handlers.py`
  - `tests/test_soundcheck_handlers.py`
  - `tests/test_auto_soundcheck_engine_safety.py`
  - `tests/test_direct_write_inventory.py`
  - `tests/test_live_input_trim_replay.py`
  - `tests/test_arrangement_automation.py`
- Audio and infrastructure coverage present:
  - `tests/test_audio_capture.py`
  - `tests/test_infrastructure/test_audio_capture.py`
  - `tests/test_osc_manager.py`
  - `tests/test_wing_client.py`
- Auto-EQ coverage present, but mostly for offline/reporting logic:
  - `tests/test_autoeq_engine.py`
- Gap relevant to TUB-229:
  - no dedicated tests were found for a room-compensation live apply contract
  - no tests were found proving that backend live Auto-EQ apply is routed through an approved EQ gate

## Dead / Duplicate / Unclear Surface

- Duplicate auto-fader surfaces:
  - `backend/auto_fader.py`
  - `backend/auto_fader_v2/`
  - `backend/auto_fader_hybrid.py`
- Duplicate voice-control surfaces:
  - `backend/voice_control.py`
  - `backend/voice_control_v2.py`
  - `backend/voice_control_vosk.py`
  - `backend/voice_control_sherpa.py`
- Duplicate/legacy snapshot-reset utilities:
  - `backend/load_snap.py`
  - `backend/load_snap_v2.py`
  - `backend/load_snap_final.py`
  - `backend/find_and_load_snap.py`
  - `backend/scan_and_load_snap.py`
  - `backend/reset_all_channels.py`
  - `backend/reset_modules_trim_faders.py`
  - `backend/reset_mixer.sh`
- Research/unclear ownership surfaces:
  - `ai_mixing_pipeline/*`
  - `mix_agent/*`
  - parts of `backend/auto_fader_v2/*`

## Top 10 Safe Next Tasks

1. Document explicitly that room compensation is analysis-only today and has no approved live apply path.
2. Add a `room compensation` row to `Docs/runtime_ownership_matrix.md` with `can_send_osc=no` and `recommendation=document` until an approved gate exists.
3. Define an EQ-specific live-apply contract before any room-compensation sender is added.
4. Require dry-run, explicit confirmation, bounded per-band gain, bounded Q, bounded frequency movement, and readback semantics in that EQ contract.
5. Add a direct-write inventory test that fails if any future room-compensation code calls `set_eq_band`, `set_eq_low_shelf`, `set_eq_high_shelf`, or raw `send()` outside an approved gate.
6. Add structured audit events for EQ apply attempts matching the current `LiveApplyService` audit shape.
7. Decide whether EQ real-write confirmation will rely on per-parameter readback, staged snapshot compare, or a narrower approved subset.
8. Quarantine backend live Auto-EQ apply paths behind an explicit maintenance or analysis-only guard until an approved EQ gate exists.
9. Unify or archive duplicate live-control surfaces that can still reach console writes, especially legacy snapshot/reset scripts and duplicate auto-fader paths.
10. Build a panic-stop coverage matrix for all real-write paths and document which ones are actually stopped today versus only locally paused.

## Disposition

- Source code was left unchanged.
- No live console commands were sent.
- TUB-229 now has an explicit guarded contract draft for room compensation.
- Next gating item is explicit product sign-off for final numeric limits and readback failure policy.

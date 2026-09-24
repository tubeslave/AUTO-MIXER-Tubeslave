# ADR: service-owned live AudioCapture bridge lifecycle v1

Status: Accepted

## Context

The canonical live path already had a `LiveAudioCaptureBridge` that converts coherent 48-channel / 48 kHz `AudioCapture` snapshots into `live_runtime` feature frames, derives Main evidence only from an explicit post-console Main return, and hands the first proof snapshot to service-owned `PATCH_VERIFY`.

The remaining lifecycle seam was ownership. `AutoSoundcheckEngine` still creates the validated `AudioCapture` transport asynchronously, while callers had no canonical production owner that attached and detached `LiveAudioCaptureBridge`. Leaving that ownership outside `LiveSoundcheckService` would preserve a second composition path and make stop ordering ambiguous.

## Decision

`LiveSoundcheckService` now owns the optional capture bridge lifecycle through an explicit `LiveCaptureBridgeConfig` carried by `LiveStartRequest`.

The configuration contains only canonical composition evidence:

- exact `MainTapPatchContract`;
- explicit channel roles and optional names;
- feature window and analysis cadence.

The service derives `PostConsoleMainTapEvidenceProvider` from `patch_contract.tap`. A caller therefore cannot pair one reserved Main channel set with a different routing proof contract.

`AutoSoundcheckEngine` remains an ADAPT dependency only for temporary mixer/audio lifecycle plumbing. Its `audio_capture` object may appear after `start_async()` returns, so the service attempts bridge attachment both after `start_async()` and on each legacy state callback. The operation is idempotent under a lifecycle lock.

On stop, the service detaches/stops `LiveAudioCaptureBridge` before stopping the legacy engine and its audio transport. A stop-state callback cannot recreate the bridge because teardown sets a lifecycle-stopping guard first.

Bridge construction/start failure is fail-closed: the live FSM enters `HOLD`, the failure is audited, and no Director/Critic frame can pass the startup gate.

No WING routing, fader, EQ, dynamics or other console mutation is introduced by this lifecycle composition.

## Migration classification

### KEEP_CORE

- `backend/audio_capture.py`: validated callback/device/ring-buffer transport surface.
- WING transport/readback primitives already used by `live_runtime`.

### ADAPT

- `backend/auto_soundcheck_engine.py`: temporary owner of discovery/connection and `AudioCapture` construction only.
- `backend/live_runtime/service.py`: canonical lifecycle/composition owner.
- `backend/live_runtime/capture_bridge.py`: canonical realtime feature ingress.

### ARCHIVE after proof

Legacy soundcheck FSM, instrument presets, AutoFOH/AutoFader/AutoEQ decision policies and evaluation/rollback orchestration remain candidates only after runtime/import references are severed and replacement tests plus HIL evidence pass.

### DELETE_AFTER_PROOF

No new candidates in this change.

## Safety invariants

1. Bridge startup is explicit. Sessions without `LiveCaptureBridgeConfig` retain compatibility behavior and remain gated in `DISCOVER` until another valid PATCH_VERIFY path is supplied.
2. Main evidence is derived from the exact reserved channels declared by the patch contract, never by summing input stems.
3. Feature/Director work stays off the realtime audio callback.
4. Bridge startup failure forces `HOLD` rather than falling back to legacy decision policy.
5. Bridge stop precedes legacy engine/audio stop.
6. BENCH_TEST semantics are unchanged. This change performs no hardware writes and does not infer BENCH_TEST from hardware.

## Validation target

Focused CI must cover:

- delayed asynchronous `AudioCapture` readiness;
- capture already available during `start_async()`;
- exact Main provider/patch contract composition;
- bridge-before-engine teardown ordering with no restart during stop callback;
- bridge startup failure -> `HOLD`;
- unconfigured compatibility sessions remaining bridge-free.

Physical WING HIL remains required before enabling autonomous Main mutation or severing the legacy runtime authorities listed in the renovation plan.

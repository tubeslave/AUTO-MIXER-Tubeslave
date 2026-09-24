# ADR: Canonical LIVE session lifecycle without AutoSoundcheckEngine

Status: accepted for migration proof

## Context

The LIVE/SOUNDCHECK renovation makes `backend/live_runtime` the canonical runtime. Previous migration steps moved physical WING/dLive transport ownership into `LiveMixerSession` and physical USB capture ownership into `LiveAudioCaptureSession`, but the configured live path still constructed `AutoSoundcheckEngine` and invoked `start_async()`. That kept the old heuristic `run()` state machine in the runtime dependency graph even when its hardware ownership had already been bypassed through migration seams.

The repository renovation plan requires validated transport/routing/readback/metrics/DSP primitives to be preserved while heuristic decision authority moves to `live_runtime` Directors/Critics. Therefore the canonical configured path must not require the old engine merely to provide start/stop/status callbacks.

## Decision

For an explicitly configured LIVE capture path (`LiveStartRequest.capture_bridge is not None`), `LiveSoundcheckService` now creates a policy-free `LiveSessionLifecycle` handle instead of constructing `AutoSoundcheckEngine`.

Canonical startup ownership is:

1. `LiveSessionLifecycle.begin_start()`
2. `LiveMixerSession.start()`
3. `LiveAudioCaptureSession.start()`
4. `LiveAudioCaptureBridge.start()`
5. startup/PATCH_VERIFY gates owned by `LiveSoundcheckService`
6. `LiveSessionLifecycle.mark_running()`

Canonical stop ownership is:

1. stop `LiveAudioCaptureBridge`
2. stop `LiveAudioCaptureSession`
3. stop `LiveMixerSession`
4. mark the lifecycle handle stopped (or error if owner teardown failed)

`AutoSoundcheckEngine.start_async()`, its legacy `run()` decision FSM, legacy discovery/connect ownership and legacy audio ownership are not executed in this configured path.

The returned lifecycle handle intentionally implements `stop()` by delegating to the owning `LiveSoundcheckService`. This preserves the temporary server/handler cleanup alias while preventing that alias from becoming an independent hardware owner.

The unconfigured compatibility path (`capture_bridge is None`) remains temporarily available and frozen. It may still construct the legacy engine while references are being severed. No new soundcheck decision features are to be added there.

## Mode contract

This lifecycle cutover does not weaken the mode contract. `BENCH_TEST` remains available only when explicitly declared for a development/test session, and only the canonical `live_runtime` control plane may exercise its permitted WING writes. A real soundcheck/show must use production protections. The legacy engine receives no restored write authority from this ADR.

## Migration classification

### KEEP_CORE

- mixer discovery probes that are transport facts rather than policy
- `WingClient` / `DLiveClient` transport primitives
- routing and readback primitives
- `backend/audio_capture.py`
- `backend/audio_device_scanner.py`
- validated metrics and DSP primitives

### ADAPT

- `LiveSoundcheckService`
- `LiveSessionLifecycle`
- `LiveMixerSession`
- `LiveAudioCaptureSession`
- `LiveAudioCaptureBridge`
- startup/PATCH_VERIFY composition
- `live_runtime` Directors/Critics and verified control plane

### ARCHIVE after proof

- `AutoSoundcheckEngine.start_async()/run()/stop()` orchestration for canonical live use
- legacy soundcheck heuristic FSM and instrument presets
- AutoFOH/AutoFader/AutoEQ decision/evaluation/rollback policy paths
- `LegacyExternalMixerSeam` and `LegacyExternalAudioCaptureSeam` once all compatibility/proof references are severed

Historical implementations may be retained in archive form but must have no runtime imports.

### DELETE_AFTER_PROOF

No new delete candidate is approved by this ADR. Removal requires severed runtime/import references plus replacement tests and HIL evidence.

## Evidence required

Automated composition tests must prove that a configured canonical session:

- does not construct `AutoSoundcheckEngine`;
- performs no second mixer discovery/connect;
- performs no second audio-capture start;
- starts owners in `mixer -> audio -> bridge` order;
- stops consumers/owners in `bridge -> audio -> mixer` order;
- closes each physical owner exactly once;
- does not fall back to the legacy engine when a canonical owner fails;
- preserves a frozen unconfigured compatibility path until repository references are severed.

Before archival/removal of the legacy orchestration, HIL evidence on the target WING/live rig is still required.

## Consequences

The configured LIVE/SOUNDCHECK runtime no longer executes the legacy heuristic engine as a lifecycle side effect. Hardware ownership, startup gates and decision authority now have a single canonical composition boundary under `backend/live_runtime`.

The next cleanup step is to remove the eager `AutoSoundcheckEngine` import from the canonical `service_core` import graph, placing the frozen compatibility constructor behind an explicit lazy compatibility adapter. That will allow importing and running the canonical configured path without importing legacy `auto_*` decision code at all.

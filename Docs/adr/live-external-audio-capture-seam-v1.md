# ADR: External AudioCapture ownership seam for legacy soundcheck

Status: **accepted for migration only**

## Context

The LIVE/SOUNDCHECK renovation now has a canonical physical audio owner in
`backend/live_runtime/audio_capture_session.py`.  The remaining legacy
`AutoSoundcheckEngine` is still needed temporarily for mixer discovery,
connection plumbing and compatibility callbacks, but its `_start_audio()` and
`stop()` paths still assume that the legacy engine owns `AudioCapture`.

Opening a second WING/Dante/USB stream would violate the new runtime ownership
contract and can produce device contention, incoherent snapshots or a legacy
teardown that closes the stream while `LiveAudioCaptureBridge` is still using
it.

## Decision

Add `backend/live_runtime/legacy_audio_capture_seam.py` as an explicit,
temporary **ADAPT** seam.

`LegacyExternalAudioCaptureSeam` gives a legacy engine non-owning access to the
already-open canonical capture.  While bound:

1. the existing physical capture must already be owned by
   `LiveAudioCaptureSession`;
2. an already-populated legacy `audio_capture` is rejected fail-closed;
3. legacy `_start_audio()` is bypassed, so it cannot scan/create/open a second
   stream;
4. `engine.audio_capture` is a delegating proxy, so analysis/read APIs continue
   to work;
5. `start()` and `stop()` invoked through the legacy reference are suppressed,
   so the legacy lifecycle cannot take ownership of the physical stream;
6. after legacy teardown the seam is detached, then the canonical session may
   close the physical stream.

The intended composition order is:

`LiveAudioCaptureSession.start -> bind seam -> legacy start/mixer connection -> LiveAudioCaptureBridge attach -> bridge stop -> legacy stop -> seam detach -> LiveAudioCaptureSession.stop`.

This ADR does **not** authorize WING writes and does not change BENCH_TEST or
production safety semantics.

## Why a proxy instead of changing legacy policy

The legacy engine is frozen for feature development.  Its audio ownership is
an infrastructure migration concern, not a new decision feature.  The proxy
allows the ownership invariant to be proven without extending any legacy
`auto_*` heuristic policy.

The seam itself is not a desired permanent abstraction.  It exists only until
`LiveSoundcheckService` no longer depends on `AutoSoundcheckEngine` for runtime
plumbing.

## Migration classification

- **KEEP_CORE**: `backend/audio_capture.py`, `backend/audio_device_scanner.py`,
  validated device/stream primitives.
- **ADAPT**: `LiveAudioCaptureSession`; `LegacyExternalAudioCaptureSeam`;
  temporary legacy discovery/connection callbacks needed by the service.
- **ARCHIVE** after runtime references are severed and replacement tests/HIL
  pass: legacy `_start_audio()` ownership logic, legacy soundcheck FSM,
  instrument presets, AutoFOH/AutoFader/AutoEQ decision/evaluation/rollback
  authorities.
- **DELETE_AFTER_PROOF**: none promoted by this change.

No legacy module is deleted by this ADR.

## Evidence required before the next migration step

Software tests must prove that:

- legacy startup does not construct or start a second physical capture;
- capture reads still delegate to the canonical capture;
- legacy teardown cannot stop the canonical stream;
- conflicting ownership and displaced proxies fail closed;
- the seam can restore the frozen legacy method after teardown.

The focused `Stem Offline Test` must include
`tests/test_live_legacy_audio_capture_seam.py`.

## Next step

Wire the seam and `LiveAudioCaptureSession` into `LiveSoundcheckService` as one
bounded lifecycle change, with ordering tests around service start/stop.  Only
after that proof should the new LIVE path stop relying on legacy `_start_audio`
for physical stream ownership.  Hardware HIL remains required before any
legacy control/write authority is archived or removed from runtime.

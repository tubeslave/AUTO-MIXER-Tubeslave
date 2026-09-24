# Live AudioCapture session ownership v1

Date: 2026-09-24
Status: accepted for R3 migration

## Decision

`backend/live_runtime/audio_capture_session.py` is the canonical ownership boundary
for live audio-device selection, `AudioCapture` construction, stream start and
teardown.

This pass extracts transport/lifecycle behavior only. It does **not** move channel
classification, preset EQ/compression, AutoFOH evaluation, legacy soundcheck
decisions or mixer writes into `live_runtime`.

`backend/audio_capture.py` and `backend/audio_device_scanner.py` remain
**KEEP_CORE**. Their validated device/callback/ring-buffer primitives are reused
behind the new session boundary.

`backend/auto_soundcheck_engine.py::_start_audio()` remains **ADAPT** temporarily.
It contains the older duplicate ownership path and will be reduced to a migration
seam only after `LiveSoundcheckService` is wired to create the canonical session
and replacement tests pass. No new musical behavior is added to it.

## Live contract

The new session is intentionally fail-closed:

1. The requested channel count is an exact stream contract. A device with fewer
   inputs is rejected instead of silently shrinking the stream.
2. The configured sample rate is preserved rather than silently adopting a
   device default. This keeps the current WING USB bridge contract at 48 kHz.
3. A real-device start that falls back to `AudioSourceType.SILENCE` is rejected.
   Silence/test generators must never impersonate a successful production input.
4. Start is idempotent and owns at most one capture stream.
5. Stop releases the owned capture exactly once.
6. Device selection and lifecycle are auditable and contain no console writes.

These constraints match the current `LiveAudioCaptureBridge`, whose USB MVP
requires exactly 48 channels at 48 kHz. Dante/AoIP can later instantiate the same
session with its own explicit stream-width contract rather than weakening the USB
gate.

## Migration classification

- **KEEP_CORE**: `backend/audio_capture.py`,
  `backend/audio_device_scanner.py`.
- **ADAPT**: device selection and capture lifecycle formerly embedded in
  `AutoSoundcheckEngine`; `LiveSoundcheckService` is the next ownership target.
- **ARCHIVE**: no module is newly archived by this extraction. Legacy instrument
  presets, classifiers and AutoFOH decision/evaluation paths remain archive
  candidates only after runtime references are severed and HIL passes.
- **DELETE_AFTER_PROOF**: none.

## Evidence

Focused software tests cover exact 48-channel / 48 kHz construction, protocol to
capture-source mapping, single-stream/idempotent ownership, insufficient-channel
rejection, absence-of-device rejection, real-capture SILENCE fallback rejection
and exactly-once teardown.

No WING writes are performed by this component. BENCH_TEST does not alter the
audio-device ownership contract.

## Next migration step

Wire `LiveSoundcheckService` to own `LiveAudioCaptureSession` whenever the
canonical capture bridge is configured. Then add a narrow legacy migration seam
so `AutoSoundcheckEngine` consumes the externally owned capture without creating
or stopping another stream. After service/handler tests prove start/stop ordering,
the duplicate `_start_audio()` ownership path can be severed from the new LIVE
runtime while remaining available to legacy callers until HIL cutover.

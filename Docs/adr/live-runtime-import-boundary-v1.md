# Live runtime import boundary v1

Date: 2026-09-24
Status: accepted

## Decision

The canonical `backend/live_runtime` import path must remain free of legacy musical decision authorities. Importing and constructing `live_runtime.service.LiveSoundcheckService` must not import the frozen AutoSoundcheck / AutoEQ / AutoFader / AutoCompressor / AutoPanner / AutoFOH evaluation families.

This is a runtime dependency rule, not merely a source-layout preference. The configured LIVE/SOUNDCHECK path already runs without `AutoSoundcheckEngine`; this ADR adds executable evidence that loading the canonical service does not pull those legacy decision modules into `sys.modules` through an accidental transitive import.

`backend/autofoh_safety.py` is deliberately excluded from the deny-list for now because the repository renovation plan still classifies it as KEEP_CORE until live-runtime safety reaches parity. That exception must not be broadened to legacy musical policy.

## Migration classification

### KEEP_CORE

- WING/dLive transport, discovery probes and readback primitives.
- `backend/audio_capture.py` and audio-device discovery primitives.
- `backend/autofoh_safety.py` temporarily, only as the existing deterministic safety primitive.

### ADAPT

- `backend/live_runtime/service.py`, `service_core.py`, mixer/audio sessions, capture bridge, PATCH_VERIFY, Directors/Critics and control plane.
- The legacy compatibility constructor in `service_core.py` remains a lazy fallback only for unconfigured sessions. It is not part of the configured canonical execution path and must not be imported eagerly.

### ARCHIVE after proof

- `AutoSoundcheckEngine` orchestration and its legacy FSM/presets.
- AutoEQ/AutoFader/AutoCompressor/AutoPanner and AutoFOH evaluation decision authorities after their remaining runtime references are severed and replacement HIL evidence passes.
- Legacy mixer/audio migration seams once no compatibility caller depends on them.

### DELETE_AFTER_PROOF

No new candidate is promoted by this step.

## Executable proof

`tests/test_live_runtime_import_boundary.py` launches a fresh Python interpreter so pytest's own import state cannot hide accidental dependencies. It asserts that:

1. importing `live_runtime.service` does not load any listed legacy decision root;
2. constructing `LiveSoundcheckService()` remains equally clean.

The focused `Stem Offline Test` workflow executes this guard on every push to `live-soundcheck-renovation`.

## Consequence

Future cleanup can treat this test as a dependency-graph gate: a legacy module cannot silently re-enter the canonical runtime while archival work proceeds. This does not by itself authorize deleting any legacy module. Import/reference severing plus replacement tests and WING HIL remain mandatory before archive/delete transitions.

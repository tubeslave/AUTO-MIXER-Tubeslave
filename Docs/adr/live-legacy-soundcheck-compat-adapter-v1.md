# Live legacy soundcheck compatibility adapter v1

Date: 2026-09-24
Status: active migration decision

## Decision

The canonical backend/live_runtime import graph must not directly import legacy
musical decision authorities. The remaining unconfigured soundcheck
compatibility path is quarantined behind
backend/live_runtime/legacy_soundcheck_adapter.py.

That adapter is the only live_runtime source allowed to import
AutoSoundcheckEngine, and it does so lazily inside the default factory.
Importing or constructing LiveSoundcheckService therefore does not import
the legacy soundcheck engine or its AutoEQ/AutoFader/AutoCompressor/AutoPanner
decision stack.

Configured LIVE/SOUNDCHECK sessions remain fully canonical and do not call the
adapter. The adapter exists only for frozen unconfigured compatibility sessions
while remaining server/handler references and WING HIL gates are severed.

## Source boundary

Focused CI now enforces two independent proofs:

1. a fresh Python process imports and constructs the canonical service and
   asserts that forbidden legacy decision modules are absent from sys.modules;
   the compatibility adapter itself must also remain unloaded;
2. an AST source scan checks every backend/live_runtime Python file and rejects
   direct imports of the forbidden legacy decision roots. The only allowlisted
   source holder is legacy_soundcheck_adapter.py, which must import exactly
   auto_soundcheck_engine and no other forbidden decision root.

This makes re-coupling visible even when a new direct import is lazy and would
not be caught by import-time execution alone.

## Migration classification

### KEEP_CORE

- WING/dLive transport and discovery probes;
- routing/readback primitives;
- audio capture and audio device discovery;
- validated metrics/DSP primitives;
- autofoh_safety only until canonical live safety reaches parity.

### ADAPT

- LiveSoundcheckService, mixer/audio sessions, capture bridge, PATCH_VERIFY,
  Directors/Critics and the canonical control plane;
- legacy_soundcheck_adapter.py only as a temporary compatibility quarantine,
  not as a source of new features.

### ARCHIVE

- AutoSoundcheckEngine orchestration/FSM/presets after remaining runtime
  compatibility references are severed and replacement HIL passes;
- the compatibility adapter itself after no runtime caller requires the
  unconfigured legacy path;
- legacy AutoEQ/AutoFader/AutoCompressor/AutoPanner/AutoFOH decision and
  evaluation implementations after their own replacement proof gates.

### DELETE_AFTER_PROOF

No new candidates are promoted by this change. Nothing is deleted.

## Mode contract

This boundary does not change operating-mode semantics. BENCH_TEST remains an
explicit development/test mode for the canonical control plane. A declared
soundcheck or concert/show must use production protections. The frozen legacy
compatibility path receives no new decision features or broader write
authority.

## Next proof gate

Audit the remaining callers of unconfigured soundcheck sessions in
backend/server.py and handlers. Convert one caller at a time to an explicit
canonical LiveStartRequest.capture_bridge. Only after runtime references are
zero and WING HIL proves the replacement lifecycle may AutoSoundcheckEngine
and this adapter move out of runtime imports.

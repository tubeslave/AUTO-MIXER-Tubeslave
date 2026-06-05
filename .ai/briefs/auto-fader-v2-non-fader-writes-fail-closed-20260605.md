# AutoFaderV2 Non-Fader Writes Fail Closed - 2026-06-05

## Task

Close direct non-fader AutoFaderV2 write paths in `backend/auto_fader_v2/controller.py` while preserving adaptive gate and masking EQ decisions as blocked evidence.

## Context

- AutoFaderV2 runs inside the realtime auto-fader controller.
- The current proven live-write scope is supervised fader/gain only.
- Adaptive noise gate contained direct `set_gate_on` calls and a fader-mute fallback.
- Spectral masking and cross-adaptive EQ contained direct `set_eq_band` calls.
- Gate and EQ writes need source identity, freshness, readback, rollback and physical proof before promotion.

## Constraints

- Do not widen live writes.
- Do not call live OSC/MIDI.
- Do not change the main fader-control path in this slice.
- Preserve would-be gate and EQ decisions as structured blocked evidence.
- Keep `mutates_mixer=False` for blocked non-fader decisions.

## Expected Output

- AutoFaderV2 adaptive gate returns blocked evidence instead of calling gate or fader fallback methods.
- AutoFaderV2 spectral/cross-adaptive masking EQ returns blocked evidence instead of calling mixer EQ methods.
- `get_status()` exposes last blocked non-fader write telemetry.
- Registry, write-surface inventory, capability matrix, docs, tests and ADR reflect the fail-closed boundary.

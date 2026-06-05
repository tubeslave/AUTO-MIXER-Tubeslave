# Auto Compressor Apply Fail Closed - 2026-06-05

## Task

Close direct compressor enable/apply write paths in `backend/auto_compressor.py` and `backend/auto_compressor_cf.py` while preserving compressor parameter analysis as blocked evidence for future operator review.

## Context

- Automatic compression is a named project-goal capability.
- Current live-write proof only covers supervised fader/gain.
- Compressor writes require fresh dynamics readback, makeup-gain/headroom proof, rollback values, and hardware validation.
- Server deployment paths already quarantine compressor start/apply surfaces, but direct controller methods still called mixer-client `set_compressor_on` and `set_compressor`.

## Constraints

- Do not widen live writes.
- Do not call live OSC/MIDI.
- Preserve threshold, ratio, attack, release, knee, makeup-gain and CF/LUFS calculations.
- Return structured blocked evidence with `mutates_mixer=False`.

## Expected Output

- AutoCompressor and AutoCFCompressor direct apply paths return blocked evidence only.
- No direct compressor mixer calls remain in `backend/auto_compressor.py` or `backend/auto_compressor_cf.py`.
- Registry, write-surface inventory, capability matrix, docs, tests, and ADR reflect the fail-closed boundary.

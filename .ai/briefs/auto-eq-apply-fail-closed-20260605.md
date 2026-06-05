# Auto EQ Apply Fail Closed - 2026-06-05

## Task

Close direct AutoEQ apply/reset write paths in `backend/auto_eq.py` while preserving single-channel and multi-channel EQ analysis as blocked evidence for operator review.

## Context

- Automatic equalization is a named project-goal capability.
- Current proven live-write scope is supervised fader/gain only.
- EQ writes require filter audit, phase/feedback safety, readback, rollback values and hardware validation.
- Server deployment handlers already quarantine public EQ apply/reset paths, but the underlying AutoEQ controllers still contained direct mixer-client EQ writes.

## Constraints

- Do not widen live writes.
- Do not call live OSC/MIDI.
- Preserve EQ correction calculation, conflict resolution and preview.
- Return structured blocked evidence with `mutates_mixer=False`.

## Expected Output

- `AutoEQController.apply_to_mixer()`, `AutoEQController.reset_eq()`, and `MultiChannelAutoEQController.apply_channel_correction()` return blocked evidence.
- No direct mixer EQ writes remain in `backend/auto_eq.py`.
- Registry, write-surface inventory, capability matrix, docs, tests and ADR reflect the fail-closed boundary.

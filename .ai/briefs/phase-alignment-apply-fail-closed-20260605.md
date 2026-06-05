# Phase Alignment Apply Fail Closed - 2026-06-05

## Task

Close direct phase/delay/polarity write paths in `backend/phase_alignment.py` and the server's private phase helpers while preserving phase analysis and would-be correction evidence for operator review.

## Context

- The project goal requires automatic phase correction.
- Current live-write policy allows only supervised fader/gain apply.
- Delay and polarity can alter mono compatibility and stage feel, so live writes require fresh phase context, readback, rollback values, and physical proof.
- `apply_phase_corrections` and `reset_phase_delay` are already deployment-quarantined in `server.py`, but the underlying controller and private helpers still contained direct mixer-client writes.

## Constraints

- Do not widen live writes.
- Do not call live OSC/MIDI.
- Preserve GCC-PHAT/phase analysis and local correction calculations.
- Return structured blocked evidence instead of mutating the mixer.

## Expected Output

- Phase apply/reset methods return blocked evidence with `mutates_mixer=False`.
- No direct `mixer_client.set_channel_delay`, `mixer_client.set_channel_phase_invert`, or raw `mixer_client.send` remains in `backend/phase_alignment.py`.
- Registry, write-surface inventory, capability matrix, docs, and tests reflect the fail-closed boundary.

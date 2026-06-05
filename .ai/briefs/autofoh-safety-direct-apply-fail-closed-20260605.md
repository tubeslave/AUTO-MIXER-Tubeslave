# AutoFOH Safety Direct Apply Fail Closed - 2026-06-05

## Task

Close AutoFOH Safety direct apply paths outside the current channel fader/gain boundary while preserving bounded typed action evidence for future supervised promotion.

## Context

- `backend/autofoh_safety.py` validates, bounds, rate-limits and translates typed correction actions.
- The current proven live-write scope is Product Layer supervised fader/gain only.
- AutoFOH typed actions could directly call bus/DCA/main fader, EQ, HPF, compressor, gate, pan, send, phase/delay and feedback-notch mixer methods.
- Those broad write surfaces require readback, rollback, false-decision replay, filter/dynamics proof, source identity and hardware validation before promotion.

## Constraints

- Do not widen live writes.
- Preserve bounded target calculation as evidence.
- Keep channel fader/gain direct behavior unchanged in this slice.
- Do not call live OSC/MIDI for non-fader/gain AutoFOH typed actions.
- Record blocked evidence with `mutates_mixer=False`.

## Expected Output

- AutoFOH non-channel-fader/gain typed actions return blocked evidence instead of mutating mixer clients.
- Fake mixer tests prove non-fader/gain AutoFOH actions do not call mixer methods.
- Write-surface inventory, capability matrix, processing inventory and ADR document the boundary.

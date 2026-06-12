# WING MVP Rows Structured Gate Brief

## Task

Update the WING MVP blocker rows and generated progress report expectations so
supervised fader/gain proof remains blocked until the structured handoff gate
matches the expected gate.

## Constraints

- Status/progress wording only.
- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.

## Expected Output

- Supervised fader and gain rows require `supervised_prestart_gate` to match
  `expected_supervised_prestart_gate`.
- `MVP_PROGRESS_LATEST.md` regenerated from the blocker table carries the same
  requirement.
- Rails tests protect the wording.

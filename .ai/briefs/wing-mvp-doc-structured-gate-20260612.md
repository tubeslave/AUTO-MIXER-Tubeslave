# WING MVP Doc Structured Gate Brief

## Task

Align `Docs/MVP_WING_SUPERVISED.md` with the structured handoff gate now used
by the WING read-only runbook, blocker rows, and no-WING release summaries.

## Constraints

- Documentation and rails-test alignment only.
- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.

## Expected Output

- MVP stop rules block supervised proof when `supervised_prestart_gate` does not
  match `expected_supervised_prestart_gate`.
- MVP current action requires `supervised_prestart_gate.grants_live_authority:
  false`.
- Rails test protects the canonical MVP doc wording.

# WING Handoff Structured Prestart Gate Brief

## Task

Make the WING read-only preflight handoff packet carry a structured supervised
prestart gate contract, not only a human-readable final operator-sequence
string.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.
- Preserve the existing final-step operator-sequence gate.

## Expected Output

- Handoff packet JSON includes a `supervised_prestart_gate` object.
- Markdown handoff output renders that contract.
- No-WING release validation rejects packets whose structured gate is missing or
  inconsistent with the final operator step.

# Project Goal Supervised AutoMix Structured Gate Brief

## Task

Align the high-level project goal capability matrix with the current WING
structured handoff gate used before supervised physical proof.

## Constraints

- Goal/capability metadata only.
- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.

## Expected Output

- `mode_supervised_automix.next_step` requires `supervised_prestart_gate` to
  match `expected_supervised_prestart_gate`.
- Regression test protects the goal-level wording.

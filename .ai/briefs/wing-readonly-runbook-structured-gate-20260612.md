# WING Read-Only Runbook Structured Gate Brief

## Task

Update the WING read-only preflight runbook so its handoff validation checklist
includes the structured `supervised_prestart_gate` contract now emitted and
summarized by no-WING release tooling.

## Constraints

- Documentation and rails-test alignment only.
- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.

## Expected Output

- Runbook requires `supervised_prestart_gate` to match
  `expected_supervised_prestart_gate`.
- Runbook preserves `grants_live_authority: false`.
- Regression test keeps the structured gate visible in the read-only runbook.

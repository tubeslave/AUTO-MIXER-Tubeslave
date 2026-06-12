# WING Read-Only Handoff Structured Prestart Gate Brief

## Task

Make the backend-port prestart gate in the WING read-only handoff packet a
structured actual/expected contract, matching the pattern already used for the
final supervised prestart gate.

## Constraints

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- Do not stop or reconfigure local listener processes.
- Do not close MVP blockers from offline handoff evidence.

## Expected Output

- Handoff packets include `backend_port_prestart_gate` and
  `expected_backend_port_prestart_gate`.
- no-WING validation fails closed if either structured gate drifts.
- no-WING summaries preserve both gate objects for operator review.
- The read-only runbook names the structured first-gate contract.

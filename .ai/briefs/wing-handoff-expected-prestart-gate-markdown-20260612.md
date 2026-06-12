# WING Handoff Expected Prestart Gate Markdown Brief

## Task

Make the WING read-only preflight handoff artifact self-contained for manual
review by exposing the expected supervised prestart gate beside the actual
gate in JSON and Markdown.

## Constraints

- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.
- Keep the first WING MVP blocker open until real artifacts are reviewed.

## Expected Output

- Handoff JSON includes `expected_supervised_prestart_gate`.
- Handoff Markdown renders the expected gate and whether actual matches it.
- no-WING release validation rejects a packet with a declared expected gate
  that differs from the computed contract.

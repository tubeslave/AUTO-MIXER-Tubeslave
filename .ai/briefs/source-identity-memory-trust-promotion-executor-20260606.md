# Brief: source-identity-memory-trust-promotion-executor-20260606

## Context

Source identity memory-trust promotion now has dry-run apply specs for
`source_identity_downstream_gate`. The next safe step is an operator-gated
executor that can apply those specs to an explicit trust store without touching
live mixer transport or the Product Layer task queue.

## Objective

Add a non-live promotion executor that validates dry-run specs and applies trust
state changes only when `apply=true`, `operator_ack=true`, and a target trust
store is supplied.

## Safety Constraints

- Do not touch WING, dLive, OSC, MIDI, TCP, sockets, or mixer clients.
- Do not import executor records into the Product Layer task queue.
- Do not mutate classifier models.
- Product Layer preview must call the executor with `apply=false`.
- Applying may mutate only the explicit caller-provided source identity trust
  store.
- Direct downstream-priority mutation remains false; any priority influence must
  be a later, reviewed consequence of trusted source identity.

## Expected Evidence

- A backend executor validates dry-run envelopes, records, summary counts, target
  paths, and idempotency keys.
- `apply=false` reports readiness without mutation.
- `apply=true` without operator acknowledgement or without a target store fails
  closed.
- `apply=true` with operator acknowledgement mutates only the explicit local
  trust store in tests.
- Product Layer preview exposes the executor with zero apply results.
- Registry and capability matrix register the executor as non-live
  memory-learning infrastructure.

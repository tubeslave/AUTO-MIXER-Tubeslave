# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Execution Plan Result Consumer

## Context

The queue-import memory-trust promotion chain can now verify promotion contracts,
record explicit operator promotion acceptance, and expose a read-only promotion
execution-plan request. The next boundary is consuming that execution-plan result
without treating it as apply authority.

## Goal

Add a transport-free result consumer that verifies promotion execution-plan
candidates against the accepted queue-import promotion records and emits only an
executor dry-run request template.

## Constraints

- Do not run dry-run inside the consumer.
- Do not apply sequence policy promotion.
- Do not mutate queues, proposal state, policy stores, models, runtime memory,
  append-only memory, idempotency, or mixer transport.
- Keep live authority and full autonomous readiness false.
- Wire Product Layer access as a verifier plus a separate dry-run handler.

## Test Plan

- Add focused unit tests for not-ready, valid request emission, forged plan,
  forged prior consumer, registry, and capability matrix.
- Extend the Product Layer queue-import flow through the new verifier and dry-run
  handler.
- Run focused pytest, py_compile, memory lint, and diff whitespace checks.

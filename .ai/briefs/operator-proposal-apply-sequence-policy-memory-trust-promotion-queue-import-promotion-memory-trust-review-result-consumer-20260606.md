# Brief: Promotion Memory Trust Review Result Consumer

## Task

Add a read-only promotion queue-import consumer that verifies memory
trust-review results against promotion memory-index provenance before exposing
an acceptance-ledger request template.

## Definition of Done

- A backend builder verifies promotion memory-index provenance and generic
  memory trust-review evidence before any acceptance-ledger handoff.
- Product State dispatch exposes the verifier without invoking the acceptance
  ledger, mutating memory, policy, runtime state, models, queues, proposals,
  trust stores, idempotency, or live mixer state.
- Processing registry and project goal matrix list the method as non-live
  decision evidence.
- Focused tests and ProductState coverage prove the acceptance-ledger request
  handoff and rejection paths.
- Memory lint, focused pytest, full pytest, and diff check pass.

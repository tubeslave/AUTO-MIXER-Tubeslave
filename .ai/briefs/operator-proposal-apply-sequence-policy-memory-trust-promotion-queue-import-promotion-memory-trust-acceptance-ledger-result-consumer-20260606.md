# Brief: Promotion Memory Trust Acceptance Ledger Result Consumer

## Task

Add a read-only promotion queue-import consumer that verifies operator
acceptance-ledger results against promotion memory-trust-review provenance
before exposing a promotion-plan request template.

## Definition of Done

- A backend builder verifies promotion memory-trust-review provenance and
  generic acceptance-ledger evidence before any promotion-plan handoff.
- Product State dispatch exposes the verifier without invoking promotion plan,
  mutating memory, policy, runtime state, models, queues, proposals, trust
  stores, idempotency, or live mixer state.
- Processing registry and project goal matrix list the method as non-live
  decision evidence.
- Focused tests and ProductState coverage prove the promotion-plan request
  handoff and rejection paths.
- Memory lint, focused pytest, full pytest, and diff check pass.

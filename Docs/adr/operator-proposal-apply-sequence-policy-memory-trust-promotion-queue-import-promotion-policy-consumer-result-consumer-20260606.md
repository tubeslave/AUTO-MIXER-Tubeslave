# ADR: Promotion Policy-Consumer Result Consumer

## Context

Sequence-policy memory trust promotion can now verify sequence-policy gate
results and request a policy consumer. The next step is verifying the policy
consumer result before any ordering preview is requested.

## Options considered

1. Let policy-consumer result consumption skip directly to ordering preview.
2. Add a separate promotion policy-consumer result consumer.
3. Treat policy-consumer records as already ordered proposal evidence.

## Decision

Add a separate promotion policy-consumer result consumer that accepts only valid
policy-consumer records and emits a read-only ordering-preview request template.

## Why this won

The separate consumer keeps policy evidence and ordering preview as distinct
reviewable boundaries. It advances the evidence chain while preserving
fail-closed live-write and mutation rules.

## Rejected alternatives

- Direct ordering-preview invocation was rejected because it skips
  policy-consumer result validation.
- Treating policy-consumer records as already ordered was rejected because
  ordering preview and queue-order mutation remain separate read-only steps.

## Implementation plan

- Register the result-consumer module as `decision_evidence`, `shadow_only`.
- Add Product Layer verifier dispatch.
- Add a Product Layer ordering-preview handler that still requires operator
  proposals.
- Update capability matrix and durable project memory.
- Add focused unit tests and extend the Product Layer handler flow.

## Test plan

- Focused tests for transport-free behavior, not-ready state, blocked consumer,
  request emission, forged policy-consumer rejection, forged prior consumer
  rejection, and registry/matrix coverage.
- Product Layer flow through verified policy-consumer result consumer and
  read-only ordering preview.
- py_compile, focused pytest, memory lint, diff whitespace checks, promotion
  chain, and backend suite when practical.

## Risks and rollback

Risk is confusing an ordering-preview request template with proposal ordering,
queue import, policy mutation, or live authority. Rollback is to remove the
consumer registration, Product Layer dispatch entries, matrix references, tests,
and docs. The consumer itself introduces no mixer, queue, policy, model,
runtime memory, append-only memory, proposal-order, or idempotency writes.

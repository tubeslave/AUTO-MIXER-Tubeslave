# ADR: Promotion Sequence-Policy Gate Result Consumer

## Context

Sequence-policy memory trust promotion can now verify rollback-verification
results and request a sequence-policy gate. The next step is verifying the gate
result before any policy consumer is requested.

## Options considered

1. Let rollback-verification result consumption skip directly to policy
   consumption.
2. Add a separate promotion sequence-policy gate result consumer.
3. Treat sequence-policy gate records as already consumed evidence.

## Decision

Add a separate promotion sequence-policy gate result consumer that accepts only
valid gate records and emits a read-only policy-consumer request template.

## Why this won

The separate consumer keeps rollback verification, sequence-policy gate
eligibility, and policy consumption as distinct reviewable boundaries. It
advances the evidence chain while preserving fail-closed live-write and
mutation rules.

## Rejected alternatives

- Direct policy-consumer request from rollback-verification evidence was
  rejected because it skips gate-result validation.
- Treating gate records as already consumed was rejected because proposal
  matching and ordering remain separate read-only steps.

## Implementation plan

- Register the result-consumer module as `decision_evidence`, `shadow_only`.
- Add Product Layer verifier dispatch.
- Update capability matrix and durable project memory.
- Add focused unit tests and extend the Product Layer handler flow.

## Test plan

- Focused tests for transport-free behavior, not-ready state, blocked gate,
  request emission, forged gate rejection, forged prior consumer rejection, and
  registry/matrix coverage.
- Product Layer flow through verified sequence-policy gate result consumer.
- py_compile, focused pytest, memory lint, diff whitespace checks, promotion
  chain, and backend suite when practical.

## Risks and rollback

Risk is confusing a policy-consumer request template with proposal ordering,
queue import, policy mutation, or live authority. Rollback is to remove the
consumer registration, Product Layer dispatch entry, matrix references, tests,
and docs. The consumer itself introduces no mixer, queue, policy, model,
runtime memory, append-only memory, proposal-order, or idempotency writes.

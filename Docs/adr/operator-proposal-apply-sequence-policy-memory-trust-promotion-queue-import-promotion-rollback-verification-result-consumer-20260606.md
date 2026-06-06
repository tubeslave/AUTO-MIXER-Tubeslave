# ADR: Promotion Rollback-Verification Result Consumer

## Context

Sequence-policy memory trust promotion can now verify apply-audit records and
request rollback verification. The next step is verifying rollback-verification
results before sequence-policy gate records are requested.

## Options Considered

1. Let apply-audit result consumption skip directly to sequence-policy gate.
2. Add a separate promotion rollback-verification result consumer.
3. Treat verified rollback records as policy-gate records.

## Decision

Add a separate promotion rollback-verification result consumer that accepts only
valid verified rollback records and emits a read-only sequence-policy gate
request template.

## Why This Won

The separate consumer keeps apply audit, rollback verification, and
sequence-policy gate construction as distinct reviewable boundaries. It
advances autonomous learning while preserving fail-closed policy and live-write
authority boundaries.

## Rejected Alternatives

- Direct sequence-policy gate request from apply-audit evidence was rejected
  because it skips rollback-verification validation.
- Treating rollback verification as policy-gate output was rejected because the
  gate remains a separate read-only policy eligibility step.

## Implementation Plan

- Add the result-consumer module and focused tests.
- Wire Product Layer verifier and sequence-policy gate handler.
- Register the method as `decision_evidence`, `shadow_only`.
- Update capability matrix, durable memory, and this ADR.

## Test Plan

- Focused tests for transport-free behavior, not-ready state, blocked rollback,
  request emission, forged rollback rejection, forged prior consumer rejection,
  and registry/matrix coverage.
- Product Layer flow through verified rollback, verifier, and read-only
  sequence-policy gate.
- py_compile, focused pytest, memory lint, diff whitespace checks, promotion
  chain, and backend suite when practical.

## Risks And Rollback

Risk is confusing verified rollback evidence with sequence-policy mutation
authority. Rollback is to remove the consumer, Product Layer dispatch entries,
registry/matrix references, and docs. The consumer itself introduces no mixer,
queue, policy, model, runtime memory, append-only memory, or idempotency writes.

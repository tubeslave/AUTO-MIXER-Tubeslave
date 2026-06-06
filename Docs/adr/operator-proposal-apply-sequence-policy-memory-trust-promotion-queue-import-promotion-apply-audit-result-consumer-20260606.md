# ADR: Promotion Apply-Audit Result Consumer

## Context

Sequence-policy memory trust promotion can now verify applied executor output
and produce apply-audit records. The next step is verifying those audit records
before rollback verification is requested.

## Options Considered

1. Let executor-result consumption skip directly to rollback verification.
2. Add a separate promotion apply-audit result consumer.
3. Treat apply-audit records as rollback-verified evidence.

## Decision

Add a separate promotion apply-audit result consumer that accepts only valid
apply-audit records and emits a read-only rollback-verification request
template.

## Why This Won

The separate consumer keeps applied executor output, audit records, and
operator rollback verification as distinct reviewable boundaries. It advances
the autonomous learning path without turning audit availability into policy
gate authority.

## Rejected Alternatives

- Direct rollback-verification request from executor-result evidence was
  rejected because it skips apply-audit validation.
- Treating apply-audit output as rollback-verified was rejected because no
  operator-reviewed verification records exist yet.

## Implementation Plan

- Add the result-consumer module and focused tests.
- Wire Product Layer verifier and rollback-verification handler.
- Register the method as `decision_evidence`, `shadow_only`.
- Update capability matrix, durable memory, and this ADR.

## Test Plan

- Focused tests for transport-free behavior, not-ready state, blocked audit,
  request emission, forged audit rejection, forged prior consumer rejection, and
  registry/matrix coverage.
- Product Layer flow through apply audit, verifier, and rollback-verification
  not-ready state without operator verification records.
- py_compile, focused pytest, memory lint, diff whitespace checks, promotion
  chain, and backend suite when practical.

## Risks And Rollback

Risk is confusing audit records with rollback-verified policy evidence.
Rollback is to remove the consumer, Product Layer dispatch entries,
registry/matrix references, and docs. The consumer itself introduces no mixer,
queue, policy, model, runtime memory, append-only memory, or idempotency writes.

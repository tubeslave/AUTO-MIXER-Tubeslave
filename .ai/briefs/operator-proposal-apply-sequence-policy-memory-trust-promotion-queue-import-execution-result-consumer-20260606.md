# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-execution-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through batch preflight,
generic sequence planning, generic dry-run, generic execution gate, and generic
executor request templates. The next provenance gap is between a generic
executor result and the generic execution-audit handler.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_execution_result` payload against the
sequence-policy queue-import execution-gate result consumer before emitting an
execution-audit request template.

## Constraints

- Do not call audit, executor, apply, or live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal state, policy stores, runtime memory, or models.
- Keep live authority and full autonomous readiness false.
- Require delegated apply evidence through `apply_operator_proposal`, verified
  readback, rollback availability, and fader/gain-only write evidence.

## Validation

- Focused module tests for applied, blocked, and forged executor evidence.
- Product Layer chain test through verified executor result into audit without
  an extra write.
- Registry and capability-matrix assertions.

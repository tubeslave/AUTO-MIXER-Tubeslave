# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-execution-audit-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through executor results and
generic execution-audit request templates. The next provenance gap is between a
generic execution-audit payload and the generic audit-policy gate that creates
policy evidence.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_execution_audit` payload against the
sequence-policy queue-import execution-result consumer before emitting an
audit-policy-gate request template.

## Constraints

- Do not call audit-policy gate, audit, executor, apply, or live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal state, policy stores, runtime memory, or models.
- Keep live authority and full autonomous readiness false.
- Require passed audit rows, fader/gain-only evidence, verified readback, proof,
  rollback availability, and the `apply_operator_proposal` entrypoint.

## Validation

- Focused module tests for passed, blocked, and forged audit evidence.
- Product Layer chain test through verified audit into audit-policy gate without
  an extra write.
- Registry and capability-matrix assertions.

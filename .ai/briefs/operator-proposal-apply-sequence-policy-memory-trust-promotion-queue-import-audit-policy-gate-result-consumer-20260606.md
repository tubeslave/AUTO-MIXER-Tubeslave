# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-audit-policy-gate-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through execution audit and
generic audit-policy-gate request templates. The next provenance gap is between
the generic audit-policy gate result and the generic policy-memory draft
handler.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_audit_policy_gate` payload against the
sequence-policy queue-import execution-audit result consumer before emitting a
policy-memory draft request template.

## Constraints

- Do not call the memory draft handler, memory writer, policy gate, executor,
  apply handler, or live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate memory, queue, proposal state, policy stores, runtime memory, or
  models.
- Keep live authority and full autonomous readiness false.
- Require `policy_evidence_ready` rows, fader/gain-only evidence, verified
  readback/proof, rollback availability, supervised sequence-policy eligibility,
  and the `apply_operator_proposal` entrypoint.

## Validation

- Focused module tests for ready, blocked, and forged policy-gate evidence.
- Product Layer chain test through policy gate into memory draft without an
  extra write.
- Registry and capability-matrix assertions.

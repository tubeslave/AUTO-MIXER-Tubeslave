# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-execution-gate-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through batch preflight,
generic sequence planning, generic dry-run, and generic execution-gate request
templates. The next provenance gap is between the generic execution-gate payload
and the generic sequence executor.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_execution_gate` payload against the
sequence-policy queue-import dry-run result consumer before emitting a sequence
executor request template.

## Constraints

- Do not call the sequence executor, apply handler, or live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal state, policy stores, runtime memory, or models.
- Keep live authority and full autonomous readiness false.
- Default executor request flags to `execute=false`, `operator_ack=false`, and
  `approved=false`.

## Validation

- Focused module tests for ready, blocked, and forged execution-gate evidence.
- Product Layer chain test through verified gate into a blocked executor request
  without write.
- Registry and capability-matrix assertions.

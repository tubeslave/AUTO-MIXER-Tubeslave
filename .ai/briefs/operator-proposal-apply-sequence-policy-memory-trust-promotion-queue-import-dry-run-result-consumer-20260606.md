# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-dry-run-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through batch preflight,
generic sequence planning, and generic dry-run request templates. The next
provenance gap is between the generic dry-run payload and the execution gate.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_dry_run` payload against the sequence-policy
queue-import sequence-plan result consumer before emitting an execution-gate
request template.

## Constraints

- Do not call execution gate, executor, apply, or live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal state, policy stores, runtime memory, or models.
- Keep live authority and full autonomous readiness false.

## Validation

- Focused module tests for would-apply, blocked, and forged dry-run evidence.
- Product Layer chain test through verified dry-run into generic execution gate.
- Registry and capability-matrix assertions.

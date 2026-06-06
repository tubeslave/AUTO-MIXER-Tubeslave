# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-sequence-plan-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through approved batch
preflight evidence before generic sequence planning. The next provenance gap is
between the generic `operator_proposal_apply_sequence_plan` payload and the
dry-run request that follows it.

## Goal

Add a transport-free consumer that verifies a generic sequence-plan result
against the sequence-policy queue-import preflight-result consumer before
emitting a dry-run request template.

## Constraints

- Do not call dry-run, gate, executor, apply, or live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal state, policy stores, runtime memory, or models.
- Keep live authority and full autonomous readiness false.

## Validation

- Focused module tests for ready, blocked, and forged sequence-plan evidence.
- Product Layer chain test through verified sequence plan into generic dry-run.
- Registry and capability-matrix assertions.

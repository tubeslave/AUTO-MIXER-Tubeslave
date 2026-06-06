# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-preflight-result-consumer-20260606

## Context

Sequence-policy queue imports can now emit a read-only batch-preflight request
template. After an operator runs batch preflight, the project needs a separate
read-only verifier that proves the returned batch evidence still matches the
sequence-policy import order before generic sequence planning is requested.

## Goal

Add a transport-free Product Layer verifier for
`queue_import_preflight_plan + operator_proposal_apply_preflight_batch` that
checks provenance, order, safety flags, idempotency keys, and readiness.

## Constraints

- Do not call batch preflight, sequence planning, dry-run, gate, executor, or
  live mixer transport.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal state, policy stores, runtime memory, or models.
- Keep live authority and full autonomous readiness false.

## Validation

- Focused module tests for ready, blocked, and forged evidence.
- Product Layer chain test through imported queue, preflight plan, batch
  preflight, result consumer, and generic sequence plan.
- Registry and capability-matrix assertions.

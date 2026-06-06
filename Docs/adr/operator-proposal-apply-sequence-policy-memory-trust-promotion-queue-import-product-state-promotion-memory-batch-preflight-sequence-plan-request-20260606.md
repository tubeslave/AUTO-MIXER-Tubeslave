# ADR: ProductState Promotion-Memory Batch Preflight Sequence-Plan Request

## Context

The ProductState promotion-memory route can generate a batch-preflight request
with `approved=false` after read-only preflight-plan verification. The existing
batch-preflight result consumer can verify approved batch-preflight evidence
and emit a sequence-plan request template without creating a sequence plan.

## Options considered

1. Stop at the batch-preflight request boundary.
2. Run approved batch preflight and verify it into a sequence-plan request.
3. Run sequence planning in the same slice.

## Decision

Add memory-prefixed ProductState aliases for the batch-preflight-result
verifier handoff and extend ProductState coverage through the sequence-plan
request template.

## Why this won

It advances the autonomous chain with explicit preflight approval while keeping
sequence-plan creation separate. The verifier emits a template only and keeps
idempotency, queue/proposal mutation, direct execution, and live authority
false.

## Rejected alternatives

Stopping at the prior boundary leaves the approved batch-preflight evidence
path unproven. Running sequence planning was rejected because it is the next
authority boundary and should be reviewed separately.

## Implementation plan

- Add `promotion_memory_preflight_plan_result_consumer` and
  `queue_import_promotion_memory_preflight_plan_result_consumer` aliases to the
  ProductState batch-preflight-result verifier handoff.
- Extend the ProductState promotion-memory route test through approved batch
  preflight and sequence-plan request templating.
- Update durable memory and the promotion queue-import memory boundary report.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is confusing approved preflight evidence with sequence execution
authority. Rollback is to remove the aliases, test extension, and
documentation lines; no live mixer path is introduced.

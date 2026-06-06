# ADR: ProductState Promotion-Memory Preflight Plan Batch Request

## Context

The ProductState promotion-memory route can import a proposal into the operator
queue under explicit acknowledgement and verify that applied queue-import
evidence into a read-only preflight-plan request. The existing preflight-plan
result consumer verifies preflight-plan evidence and emits only a
batch-preflight request template.

## Options considered

1. Stop at the preflight-plan request boundary.
2. Run the preflight plan and verify it into a batch-preflight request.
3. Run batch preflight in the same slice.

## Decision

Add memory-prefixed ProductState aliases for the preflight-plan-result verifier
handoff and extend ProductState coverage through the batch-preflight request
template.

## Why this won

It advances the autonomous apply-sequence chain while keeping batch preflight
approval explicit. The generated request keeps `approved=false`, idempotency
unreserved, queue/proposal mutation false, and live authority false.

## Rejected alternatives

Stopping at the prior boundary leaves the preflight-plan verifier unproven in
ProductState. Running batch preflight was rejected because it needs a separate
operator approval step.

## Implementation plan

- Add `promotion_memory_queue_importer_result_consumer` and
  `queue_import_promotion_memory_queue_importer_result_consumer` aliases to the
  ProductState preflight-plan-result verifier handoff.
- Extend the ProductState promotion-memory route test through read-only
  preflight-plan execution and batch-preflight request templating.
- Update durable memory and the promotion queue-import memory boundary report.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is treating a request template as approval to run batch
preflight. Rollback is to remove the aliases, test extension, and documentation
lines; no live mixer path is introduced.

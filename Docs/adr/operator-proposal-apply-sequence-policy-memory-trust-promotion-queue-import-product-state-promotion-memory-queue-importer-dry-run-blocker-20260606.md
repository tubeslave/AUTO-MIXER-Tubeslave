# ADR: ProductState Promotion-Memory Queue Importer Dry-Run Blocker

## Context

The ProductState promotion-memory route can emit a queue-importer request
template with `apply=false` and `operator_ack=false`. The existing queue
importer can produce dry-run specs without mutating the operator queue, and the
existing promotion queue-importer result consumer blocks preflight planning
until an applied queue-importer result exists.

## Options considered

1. Stop at the queue-importer request boundary.
2. Run the queue importer as dry-run evidence and verify the applied-result
   blocker.
3. Apply the queue importer in the same ProductState memory slice.

## Decision

Add memory-prefixed ProductState aliases for the queue-importer-result verifier
handoff and extend ProductState coverage through the dry-run importer blocker.

## Why this won

The change proves the next safety gate without mutating queues or proposals.
It keeps the applied queue-import step separate, requires no new verifier
module, and relies on the existing consumer to block preflight-plan request
creation until applied queue-import evidence exists.

## Rejected alternatives

Stopping at the previous boundary leaves the dry-run blocker unproven in
ProductState. Applying the queue importer was rejected because it mutates an
explicit operator queue and needs a separate supervised slice.

## Implementation plan

- Add `promotion_memory_queue_ordering_plan_result_consumer` and
  `queue_import_promotion_memory_queue_ordering_plan_result_consumer` aliases
  to the queue-importer-result ProductState verifier handoff.
- Extend the ProductState promotion-memory route test through dry-run queue
  importer execution and queue-importer-result blocker verification.
- Update durable memory and the promotion queue-import memory boundary report.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is confusing dry-run specs with queue import results. Rollback is
to remove the aliases, ProductState test extension, and documentation lines;
no persisted data migration or live mixer path is introduced.

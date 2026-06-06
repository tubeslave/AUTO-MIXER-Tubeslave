# ADR: ProductState Promotion-Memory Queue Importer Request Handoff

## Context

The ProductState promotion-memory route can run a policy consumer, an ordering
preview, and an ordering-preview result consumer that emits a
queue-ordering-plan request. The existing queue-ordering-plan result consumer
already validates plan rows against ordering-preview provenance and emits only
a queue-importer request template with `apply=false` and `operator_ack=false`.

## Options considered

1. Stop at the queue-ordering-plan request boundary.
2. Add memory-prefixed ProductState aliases and verify the queue-ordering-plan
   result into a queue-importer request template.
3. Execute the queue importer in the same ProductState memory slice.

## Decision

Add memory-prefixed ProductState aliases for the queue-ordering-plan-result
verifier handoff and extend ProductState coverage through the queue-ordering
plan and queue-importer request template.

## Why this won

The change moves the route toward supervised queue import while keeping the
authority boundary explicit. The verifier builder remains responsible for
provenance and safety checks, and the emitted template keeps queue import
disabled until a separate operator-authorized importer call.

## Rejected alternatives

Stopping at the previous boundary leaves the memory route short of the next
available verifier. Executing the importer in this slice was rejected because
even dry-run importer evidence is a separate downstream action and should be
reviewed independently.

## Implementation plan

- Add `promotion_memory_ordering_preview_result_consumer` and
  `queue_import_promotion_memory_ordering_preview_result_consumer` aliases to
  the queue-ordering-plan-result ProductState verifier handoff.
- Extend the ProductState promotion-memory route test through queue-ordering
  plan execution and queue-importer request templating.
- Update durable memory and the promotion queue-import memory boundary report.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is confusing request-template availability with queue import
authority. Rollback is to remove the aliases, ProductState test extension, and
documentation lines; no persisted data migration or live mixer path is
introduced.

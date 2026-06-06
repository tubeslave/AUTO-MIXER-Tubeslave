# ADR: ProductState Promotion-Memory Ordering Preview Handoff

## Context

The ProductState promotion-memory queue-import route can verify rollback
evidence, expose a sequence-policy-gate request, run that gate, and expose a
policy-consumer request. The existing promotion policy-consumer result and
ordering-preview result consumers already validate downstream provenance and
emit the next request template without mutating queues or live mixer state.

## Options considered

1. Use only generic payload names for the ProductState handoffs.
2. Add memory-prefixed ProductState aliases and reuse existing verifier
   builders.
3. Add new promotion-memory-specific verifier modules.

## Decision

Add memory-prefixed ProductState aliases for the policy-consumer-result and
ordering-preview-result verifier handoffs. Extend ProductState coverage through
the policy consumer, ordering preview, and queue-ordering-plan request
template.

## Why this won

The change preserves explicit promotion-memory provenance in handler payloads
without adding a new method, dependency, or authority surface. The existing
verifier builders remain responsible for safety decisions and keep proposal
ordering, queue import, idempotency, runtime memory, policy/model stores,
direct sequence execution, live mixer authority, and full autonomous readiness
false.

## Rejected alternatives

Generic-only payload names were rejected because they obscure the
promotion-memory provenance. New verifier modules were rejected because they
would duplicate safety logic already covered by the existing consumers.

## Implementation plan

- Add `promotion_memory_sequence_policy_gate_result_consumer` and
  `queue_import_promotion_memory_sequence_policy_gate_result_consumer` aliases
  to the policy-consumer-result ProductState verifier handoff.
- Add `promotion_memory_policy_consumer_result_consumer` and
  `queue_import_promotion_memory_policy_consumer_result_consumer` aliases to
  the ordering-preview-result ProductState verifier handoff.
- Extend the ProductState promotion-memory route test through policy consumer,
  ordering preview, and queue-ordering-plan request templating.
- Update durable memory and the promotion queue-import memory boundary report.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is alias drift or accidentally implying queue-ordering authority.
Rollback is to remove the aliases, ProductState test extension, and
documentation lines; no persisted data migration or live mixer path is
introduced.

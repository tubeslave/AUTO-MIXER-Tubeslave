# Brief: ProductState Promotion-Memory Batch Preflight Sequence-Plan Request

## Task

Extend the ProductState promotion-memory queue-import route so an explicitly
approved batch preflight can be verified into a sequence-plan request template.

## Definition of Done

- ProductState accepts memory-prefixed prior-consumer payload names for the
  promotion batch-preflight-result verifier handoff.
- ProductState coverage proves the promotion-memory branch can run batch
  preflight only after setting `approved=true`.
- ProductState coverage proves the batch-preflight result consumer emits only a
  sequence-plan request template.
- Sequence planning is not run in this slice.
- Queue mutation, proposal status/order mutation, idempotency reservation,
  direct sequence execution, runtime/policy/model mutation, and live authority
  remain false at the consumer boundary.
- Memory/report artifacts describe the boundary as read-only evidence.
- Focused pytest, memory lint, py_compile, and diff check pass.

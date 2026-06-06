# Brief: ProductState Promotion-Memory Preflight Plan Batch Request

## Task

Extend the ProductState promotion-memory queue-import route so a read-only
queue-import preflight plan can be verified into a batch-preflight request
template.

## Definition of Done

- ProductState accepts memory-prefixed prior-consumer payload names for the
  promotion preflight-plan-result verifier handoff.
- ProductState coverage proves the promotion-memory branch can run the
  queue-import preflight plan after applied queue-import evidence exists.
- ProductState coverage proves the preflight-plan result consumer emits only a
  batch-preflight request template with `approved=false`.
- Batch preflight is not run in this slice; explicit `approved=true` remains a
  separate operator action.
- Queue mutation, proposal status/order mutation, idempotency reservation,
  direct sequence execution, runtime/policy/model mutation, and live authority
  remain false at the consumer boundary.
- Memory/report artifacts describe the boundary as read-only evidence.
- Focused pytest, memory lint, py_compile, and diff check pass.

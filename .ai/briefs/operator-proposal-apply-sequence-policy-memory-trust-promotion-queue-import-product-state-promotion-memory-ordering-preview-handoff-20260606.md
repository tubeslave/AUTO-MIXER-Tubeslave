# Brief: ProductState Promotion-Memory Ordering Preview Handoff

## Task

Extend the ProductState promotion-memory queue-import route so the
policy-consumer result verifier accepts memory-prefixed sequence-policy-gate
consumer evidence, then the ordering-preview result verifier accepts
memory-prefixed policy-consumer-result evidence. The route should reach only a
queue-ordering-plan request template.

## Definition of Done

- ProductState accepts memory-prefixed prior-consumer payload names for the
  promotion policy-consumer-result and ordering-preview-result verifier
  handoffs.
- The existing verifier builders remain the source of safety decisions.
- ProductState coverage proves the promotion-memory branch can run the policy
  consumer and ordering preview from verified promotion-memory evidence.
- ProductState coverage proves the ordering-preview result consumer emits only
  a queue-ordering-plan request template.
- Proposal order, queue state, runtime memory, policy/model stores,
  idempotency, direct sequence execution, and live mixer authority remain
  unchanged.
- Memory/report artifacts describe the boundary as read-only evidence.
- Focused pytest, memory lint, py_compile, and diff check pass.

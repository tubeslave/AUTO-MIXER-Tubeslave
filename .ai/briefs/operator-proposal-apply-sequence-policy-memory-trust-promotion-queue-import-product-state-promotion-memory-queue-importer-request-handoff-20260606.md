# Brief: ProductState Promotion-Memory Queue Importer Request Handoff

## Task

Extend the ProductState promotion-memory queue-import route so the
queue-ordering-plan result verifier accepts memory-prefixed ordering-preview
result evidence and reaches only a queue-importer request template.

## Definition of Done

- ProductState accepts memory-prefixed prior-consumer payload names for the
  promotion queue-ordering-plan-result verifier handoff.
- The existing queue-ordering-plan result verifier remains the source of
  safety decisions.
- ProductState coverage proves the promotion-memory branch can run the
  queue-ordering plan from verified ordering-preview evidence.
- ProductState coverage proves the queue-ordering-plan result consumer emits
  only a queue-importer request template with `apply=false` and
  `operator_ack=false`.
- Queue contents, proposal order, proposal status, runtime memory,
  policy/model stores, idempotency, direct sequence execution, and live mixer
  authority remain unchanged.
- Memory/report artifacts describe the boundary as read-only evidence.
- Focused pytest, memory lint, py_compile, and diff check pass.

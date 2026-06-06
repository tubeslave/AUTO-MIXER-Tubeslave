# Brief: ProductState Promotion-Memory Queue Importer Dry-Run Blocker

## Task

Extend the ProductState promotion-memory queue-import route so the
queue-importer result verifier accepts memory-prefixed queue-ordering-plan
result evidence and proves that a dry-run queue importer cannot expose a
preflight-plan request.

## Definition of Done

- ProductState accepts memory-prefixed prior-consumer payload names for the
  promotion queue-importer-result verifier handoff.
- The existing queue-importer result verifier remains the source of safety
  decisions.
- ProductState coverage proves the promotion-memory branch can run the queue
  importer with `apply=false` and `operator_ack=false`.
- ProductState coverage proves the queue-importer result consumer blocks
  preflight-plan request exposure until an applied queue-importer result exists.
- Queue contents, proposal order/status, runtime memory, policy/model stores,
  idempotency, direct sequence execution, and live mixer authority remain
  unchanged.
- Memory/report artifacts describe the boundary as dry-run evidence only.
- Focused pytest, memory lint, py_compile, and diff check pass.

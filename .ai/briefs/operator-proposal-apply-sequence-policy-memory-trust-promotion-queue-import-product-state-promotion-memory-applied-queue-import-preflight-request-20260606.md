# Brief: ProductState Promotion-Memory Applied Queue Import Preflight Request

## Task

Extend the ProductState promotion-memory queue-import route so an explicitly
acknowledged queue-importer apply can be verified into a preflight-plan request
template while the consumer itself remains read-only.

## Definition of Done

- ProductState coverage proves the promotion-memory branch can run the queue
  importer with `apply=true` and `operator_ack=true`.
- The applied queue importer may mutate only the server operator proposal
  queue; it must not touch live mixer transport or grant live authority.
- The queue-importer result consumer verifies the applied importer result and
  emits only a read-only preflight-plan request template.
- The consumer keeps proposal status/order mutation, idempotency reservation,
  direct sequence execution, policy/model/runtime memory mutation, and live
  authority false.
- Memory/report artifacts clearly separate source queue mutation from
  consumer-side mutation.
- Focused pytest, memory lint, py_compile, and diff check pass.

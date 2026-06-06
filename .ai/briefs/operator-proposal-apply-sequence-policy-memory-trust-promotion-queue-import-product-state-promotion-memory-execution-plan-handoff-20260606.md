# Brief: ProductState Promotion-Memory Policy Consumer Request Handoff

## Task

Extend the ProductState promotion-memory queue-import route so the new
promotion memory-trust acceptance-ledger result consumer can continue through
the existing promotion-contract, promotion-acceptance-ledger, and
promotion-execution-plan result consumers with memory-prefixed payload aliases,
then reach the executor dry-run, preview-safe executor result boundary, and
supervised apply-audit, rollback-verification, and policy-consumer request
boundary.

## Definition of Done

- ProductState accepts memory-prefixed prior-consumer payload names for the
  downstream promotion acceptance-ledger, execution-plan, dry-run-request, and
  executor-result/apply-audit/rollback-verification verifier handoffs.
- The existing verifier builders remain the source of safety decisions.
- ProductState coverage proves the promotion-memory branch reaches a promotion
  executor dry-run request template and a preview executor result that remains
  blocked from apply-audit without applying, mutating memory or policy,
  reserving idempotency, or touching live mixer state.
- ProductState coverage also proves that an explicit `apply=true`,
  `operator_ack=true`, caller-provided target-store executor result may expose
  an apply-audit result and rollback-verification result that only emits a
  sequence-policy-gate request without touching live mixer transport.
- ProductState coverage proves the sequence-policy gate result emits only a
  policy-consumer request while proposal order, queue, runtime memory,
  idempotency, and live mixer authority remain unchanged.
- Memory/report artifacts describe the handoff as read-only evidence.
- Focused pytest, memory lint, py_compile, and diff check pass.

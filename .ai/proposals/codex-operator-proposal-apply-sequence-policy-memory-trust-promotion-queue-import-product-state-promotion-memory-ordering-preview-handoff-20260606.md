# Proposal: ProductState Promotion-Memory Ordering Preview Handoff

## Short thesis

Keep the promotion-memory route readable at the ProductState boundary while
reusing the existing policy-consumer and ordering-preview verifiers.

## What I understood

The promotion-memory branch already reaches a policy-consumer request template.
The next autonomous step is to consume that evidence into read-only proposal
records, preview their order, and expose only the next queue-ordering-plan
request. This moves the sequence toward future supervised queue import without
granting queue, idempotency, or live-write authority.

## Proposed solution

Add narrow memory-prefixed payload aliases in
`backend/handlers/product_state_handlers.py` for the policy-consumer-result and
ordering-preview-result verifier handoffs. Extend the existing ProductState
promotion-memory regression to run the policy consumer, verify its result,
run the ordering preview, and verify that result into a queue-ordering-plan
request template.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Create a ProductState-only wrapper for the memory branch. Rejected because the
existing verifier modules already enforce the provenance and safety contract;
ProductState only needs transport aliases and integration coverage.

## Risks

The risk is confusing ordering-preview evidence with permission to reorder
Product Layer proposal payloads or import queue records. The test must keep
proposal-order mutation, queue mutation, idempotency reservation, direct
sequence execution, live authority, and full autonomous readiness false.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if focused checks remain stable.

## Where the other agent may disagree

Kimi may argue that the aliases are cosmetic. The counterpoint is operational:
explicit memory-prefixed payload names keep provenance readable for future
agents while preserving the existing verifier boundary.

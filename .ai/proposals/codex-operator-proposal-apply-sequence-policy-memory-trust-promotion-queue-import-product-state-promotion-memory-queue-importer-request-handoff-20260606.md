# Proposal: ProductState Promotion-Memory Queue Importer Request Handoff

## Short thesis

Carry the promotion-memory route one step closer to supervised queue import by
emitting only the queue-importer request template from verified ordering-plan
evidence.

## What I understood

The promotion-memory ProductState branch now reaches a queue-ordering-plan
request. The next safe step is to run that plan as read-only evidence and
verify it into a queue-importer request. This should not import proposals,
reorder Product Layer payloads, reserve idempotency, or grant live authority.

## Proposed solution

Add narrow memory-prefixed aliases in
`backend/handlers/product_state_handlers.py` for the queue-ordering-plan-result
verifier handoff. Extend the existing ProductState promotion-memory regression
to run the queue-ordering plan and verify its result into a queue-importer
request template that keeps `apply=false` and `operator_ack=false`.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Execute the queue importer as a dry run in the ProductState memory branch.
Rejected for this slice because the safer next boundary is proving the request
template itself; importer execution can remain a separate explicit task.

## Risks

The risk is treating the request template as a queue import. The test must
prove that the ProductState branch leaves queue mutation, proposal-order
mutation, proposal-state mutation, idempotency reservation, direct sequence
execution, live authority, and full autonomous readiness false.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if focused checks remain stable.

## Where the other agent may disagree

Kimi may argue that the branch should execute the dry-run importer immediately.
Keeping this slice at the request boundary makes the authority handoff explicit
and easier to review before any future queue-importer execution path.

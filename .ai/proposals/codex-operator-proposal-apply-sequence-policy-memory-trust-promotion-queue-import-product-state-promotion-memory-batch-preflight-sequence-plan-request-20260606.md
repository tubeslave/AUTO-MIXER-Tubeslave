# Proposal: ProductState Promotion-Memory Batch Preflight Sequence-Plan Request

## Short thesis

Use explicit batch-preflight approval to produce verified preflight evidence
and expose only the sequence-plan request template.

## What I understood

The ProductState promotion-memory route now reaches a batch-preflight request
with `approved=false`. The next safe step is to set `approved=true`, run batch
preflight as evidence, and verify that evidence into a sequence-plan request
without creating the sequence plan.

## Proposed solution

Add memory-prefixed payload aliases to the ProductState batch-preflight-result
verifier handoff. Extend the ProductState promotion-memory regression to run
`preflight_operator_proposal_apply_batch` from the generated request only after
setting `approved=true`, then feed the result to the existing promotion
batch-preflight result consumer and assert the sequence-plan request template.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Run the sequence planner immediately. Rejected because the request boundary
should remain explicit before the next downstream action creates a plan.

## Risks

The risk is treating approved batch preflight as sequence execution authority.
The verifier must keep sequence-plan creation, idempotency reservation, queue
or proposal mutation, direct execution, and live authority false.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if focused checks remain stable.

## Where the other agent may disagree

Kimi may prefer collapsing request and plan creation in one step. The safer
increment is to prove the approved preflight evidence boundary first.

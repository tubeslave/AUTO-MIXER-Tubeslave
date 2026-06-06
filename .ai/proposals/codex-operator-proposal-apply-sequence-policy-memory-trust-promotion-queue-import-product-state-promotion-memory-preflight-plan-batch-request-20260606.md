# Proposal: ProductState Promotion-Memory Preflight Plan Batch Request

## Short thesis

Use the applied queue-import source evidence to produce a read-only preflight
plan and expose only the next batch-preflight request template.

## What I understood

The ProductState promotion-memory branch now imports one proposal into the
operator queue under explicit acknowledgement and verifies that source evidence
into a preflight-plan request. The next step is to run the read-only preflight
plan and verify it into a batch-preflight request without running batch
preflight or reserving idempotency.

## Proposed solution

Add memory-prefixed payload aliases to the ProductState preflight-plan-result
verifier handoff. Extend the existing ProductState promotion-memory regression
to run `plan_operator_proposal_apply_sequence_policy_queue_import_preflight`
from the generated request, then verify that result into a
`preflight_operator_proposal_apply_batch` request with `approved=false`.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Run batch preflight immediately. Rejected because batch preflight requires an
explicit approval flag and should remain the next separate action.

## Risks

The risk is confusing batch-preflight request availability with approval to run
batch preflight. The request must keep `approved=false`, require the operator
to set `approved=true`, and preserve idempotency, queue/proposal mutation, and
live authority false.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if focused checks remain stable.

## Where the other agent may disagree

Kimi may argue that stopping at another request template is incremental. The
counterpoint is safety: every authority transition in this chain needs an
explicit, test-proven boundary before the next action can be reviewed.

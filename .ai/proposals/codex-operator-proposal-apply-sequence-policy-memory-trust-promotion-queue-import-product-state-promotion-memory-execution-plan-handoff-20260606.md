# Proposal: ProductState Promotion-Memory Policy Consumer Request Handoff

## Short thesis

Keep the promotion-memory branch explicit in ProductState by adding narrow
payload aliases and proving it can reuse the existing downstream read-only
promotion consumers.

## What I understood

The backend already has promotion acceptance-ledger, execution-plan,
executor-dry-run, and executor-result consumers. The newest promotion-memory
trust acceptance step can feed them, but the ProductState route should accept
memory-prefixed payload names so future agents do not have to collapse the
evidence back to generic field names.

## Proposed solution

Add aliases only in `backend/handlers/product_state_handlers.py` for downstream
promotion result consumers. Extend the ProductState regression to start from
the promotion-memory contract handoff, record explicit operator promotion
acceptance, verify the acceptance ledger, build the promotion execution plan,
verify the dry-run request, execute the dry-run template as a preview, and prove
the executor-result consumer remains blocked from apply-audit because no
applied executor result exists. Then execute a separate supervised apply with
`apply=true`, `operator_ack=true`, and a caller-provided target store to prove
the existing apply-audit and rollback-verification request boundaries without
touching live mixer state, then verify the sequence-policy gate result and emit
only the policy-consumer request template.

## Likely files to touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/reports/AUTOMIXER_PROMOTION_QUEUE_IMPORT_MEMORY_BOUNDARY_20260606.md`
- `Docs/adr/`

## Alternatives considered

Add a new promotion-memory acceptance-ledger wrapper. Rejected because the
existing promotion acceptance-ledger result consumer already validates the
contract provenance and emits the correct execution-plan request template.

## Risks

The risk is treating alias support, executor preview, or caller-provided
target-store mutation as live authority. The route must keep mixer writes,
runtime memory mutation, queue/proposal mutation, idempotency reservation, and
live authority false, and only allow target-store mutation after explicit
`apply=true` plus `operator_ack=true`; rollback verification remains
operator-reviewed evidence for a sequence-policy gate, and the gate result
remains only a request for policy-consumer review.

## Test plan

Run focused ProductState and adjacent promotion consumer tests, py_compile,
memory lint, diff check, and full pytest if shared coverage remains stable.

## Where the other agent may disagree

Kimi may argue that field aliases are unnecessary. The benefit is preserving
human-readable promotion-memory provenance in handler calls and proving the
preview boundary without adding a new verifier or widening authority.

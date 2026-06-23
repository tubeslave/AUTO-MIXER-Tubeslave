# Brief: WING MVP Proof-Plan Review-Decision Invalid-Record Diagnostics

## Task

Expose invalid WING read-only review decision records in the supervised
`next_wing_proof_plan` summary and no-WING proof-plan projection.

## Context

Standalone decision-review reports and the no-WING review-decision ledger
summary now show invalid operator records with `template_status` and
`violations`. The supervised proof-plan summary still carried only count and id
signals, so stale pending-template review records could be hidden when operators
read the next proof-plan section.

## Safety Boundary

This is an offline reporting and fail-closed consistency change. It does not
contact WING, mutate console state, approve readiness closure, or grant live
authority.

## Expected Evidence

- `next_wing_proof_plan.summary` carries normalized invalid review-decision
  records.
- no-WING validation compares proof-plan invalid records against the ledger.
- no-WING `summary.json` and `summary.md` expose the full invalid-record list.
- Targeted and standard validation pass.

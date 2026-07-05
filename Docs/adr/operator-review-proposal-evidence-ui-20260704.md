# ADR: Operator Review Proposal Evidence UI

## Context

`build_decision_queue` now reports whether pending proposals include the
operator review fields required for supervised soundcheck: reason, risk,
requested change, old value, new value and rollback. The frontend task adapter
still only rendered `requested_change.safety_evidence`, so old/new/rollback
fields could be present in the backend contract but absent from the visible
task evidence panel.

M005 and M006 remain external validation gates. This UI evidence work must not
grant live apply authority or imply console readiness.

## Options considered

1. Add dedicated `Task` fields for old value, new value and rollback.
2. Derive read-only safety evidence rows in `adaptTasks`.
3. Leave the modal details to infer old/new/rollback from fallback task preview
   text.

## Decision

Derive operator review evidence rows in `adaptTasks` from
`requested_change` and `summary.operator_review_contract.proposal_field_coverage`.
Put those rows before generic safety evidence so the visible task panel keeps
old value, new value, delta, rollback and review field completeness inside the
six-row evidence limit.

## Why this won

The adapter already owns backend-to-UI DTO shaping, and `Task.safetyEvidence`
is already rendered in the task queue and supervised apply modal. This keeps
the change small, read-only and testable without adding another UI contract.

## Rejected alternatives

- New `Task` fields would require wider component changes and more display
  decisions for the same read-only data.
- Fallback preview text is not backend evidence and can hide missing contract
  fields from the operator.

## Implementation plan

1. Build a proposal coverage map from `decision_queue.summary`.
2. Add derived evidence rows for old value, new value, delta, rollback and
   review field completeness.
3. Keep existing safety evidence rows after the derived review rows and dedupe
   by label.
4. Add frontend adapter tests for complete and incomplete review coverage.

## Test plan

- `cd frontend && node --test src/services/__tests__/operatorAdaptersContract.test.mjs`
- `cd frontend && npm test`
- `cd frontend && npm run build`
- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py::test_decision_queue_reports_operator_review_field_coverage tests/test_operator_product_state.py::test_decision_queue_marks_operator_review_incomplete_fields -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

The main risk is mistaking visible review evidence for live apply readiness.
The rows are display-only and do not call apply handlers or mutate console
state. Rollback is to remove the adapter helper, its test and this ADR.

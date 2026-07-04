# ADR: Operator Review Proposal Coverage

## Context

The supervised soundcheck MVP requires operators to see what is being proposed,
why it is proposed, the risk, old/new value and rollback path. M005 and M006
remain external validation gates, so unattended work cannot use this visibility
work to grant apply authority.

## Options considered

1. Enforce complete proposal context in `OperatorProposalQueue.create`.
2. Add a read-only coverage summary to `build_decision_queue`.
3. Leave the UI to infer missing fields from raw proposal payloads.

## Decision

Add `summary.operator_review_contract` to `build_decision_queue`. It checks
pending proposals for `reason`, `risk`, `requested_change`, `old_value`,
`new_value` and `rollback` context, and reports complete/incomplete coverage.

## Why this won

The summary makes proposal quality auditable without changing apply gates,
proposal creation, queue mutation or live authority.

## Rejected alternatives

- Enforcing completeness at creation could break existing analysis-only and
  transitional proposals before the UI can display the gaps.
- UI-only inference would be harder to test and would not give backend evidence
  for runtime audits.

## Implementation plan

1. Add read-only helper functions in `backend/operator_product_state.py`.
2. Include coverage counts and rows in `decision_queue.summary`.
3. Add tests for complete and incomplete pending proposals.
4. Record the scope as operator visibility only in truth surfaces.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -k "operator_review_field_coverage or operator_review_incomplete_fields or decision_queue_summarizes_apply_preflight_candidates or operator_proposal_queue_accepts_assist_suggestions" -q`
- `python3 -m py_compile backend/operator_product_state.py`

## Risks and rollback

The risk is treating coverage completeness as permission to apply. The summary
therefore carries `live_write=false` and only reports missing fields. Rollback
is to remove the summary helper, tests and truth-surface entries; no live state
is touched.

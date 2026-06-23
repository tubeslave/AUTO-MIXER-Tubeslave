# ADR: WING MVP Review Decision Run-Id Verdict Field

## 1. Context

WING read-only preflight artifact review validates that every `events.jsonl`
row belongs to the same `run_id` as `summary.json`. The review report exposes
`event_log_run_id_matches_summary`, and the checklist now asks the operator to
confirm run-id binding, but accepted decision records did not persist the
machine verdict itself.

## 2. Options considered

- Keep the verdict only in the artifact review report.
- Treat the checklist id as sufficient durable evidence.
- Add the verdict as a required decision-record field.

## 3. Decision

Add `event_log_run_id_matches_summary` to the shared required fields for WING
read-only preflight decision records, and require it to be `true` for accepted
or rejected records to be valid.

## 4. Why this won

The decision ledger is the durable operator acceptance surface. Storing the
machine verdict there keeps the artifact identity proof attached to the
operator decision, while preserving the existing read-only and fail-closed
safety boundary.

## 5. Rejected alternatives

Keeping the verdict only in the review report was rejected because decision
records can be copied or reviewed separately. Checklist-only confirmation was
rejected because it does not expose the machine-computed value.

## 6. Implementation plan

- Extend `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS`.
- Add `event_log_run_id_matches_summary: true` to generated decision templates.
- Reject records missing the true verdict in the decision ledger.
- Update runbook examples and focused tests.

## 7. Test plan

Run focused review-decision, decision-tool, review-tool, handoff, no-WING, WING
rail, and Project Director tests, then standard pytest and no-WING quick
runtime.

## 8. Risks and rollback

Older decision JSON without the verdict field will be rejected and must be
replaced. Rollback is removing the required field and its validation/tests.
This ADR does not contact WING, mutate console state, grant live authority, or
close readiness.

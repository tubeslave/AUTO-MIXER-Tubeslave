# ADR: WING MVP Review Checklist Run-Id Binding

## 1. Context

WING read-only preflight artifacts now validate that every `events.jsonl` row
has the same `run_id` as `summary.json`. The review report exposes
`event_log_run_id_matches_summary`, but the required operator checklist did not
separately confirm this row-level identity check.

## 2. Options considered

- Keep the existing event-log sequence checklist item only.
- Add an optional note to the review report.
- Add a required checklist id for event-log run-id binding.

## 3. Decision

Add `confirm_event_log_run_id_binding` to the shared WING read-only preflight
manual review checklist.

## 4. Why this won

The decision ledger already validates checklist ids exactly. Adding the new id
turns row-level event-log identity into durable operator-reviewed evidence,
without adding live authority or changing hardware behavior.

## 5. Rejected alternatives

The existing sequence checklist was rejected as insufficient because read-only
sequence does not prove that the event log belongs to the same run. Optional
notes were rejected because they are not machine-validated.

## 6. Implementation plan

- Extend `WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS`.
- Add the matching manual review label in the artifact review tool.
- Update runbook/template examples and exact-list tests.
- Add a negative ledger test for old checklist records.

## 7. Test plan

Run focused review-decision, review-tool, handoff, no-WING, and WING rail
tests, then standard pytest and no-WING quick runtime.

## 8. Risks and rollback

Older accepted-decision JSON records without the new checklist id will be
rejected and must be replaced. Rollback is removing the checklist id and its
focused assertions. This ADR does not contact WING, mutate console state, grant
live authority, or close readiness.

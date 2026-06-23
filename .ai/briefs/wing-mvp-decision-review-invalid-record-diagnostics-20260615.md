# Brief: WING MVP Decision Review Invalid-Record Diagnostics

## Task

Expose invalid WING read-only review decision records directly in the standalone
decision-review report.

## Context

The review-decision ledger already rejects accepted records that keep the
generated pending `template_status` marker. The standalone decision-review
report showed `invalid_record_ids`, but the exact `template_status` and
`violations` stayed nested inside the full ledger payload. During a physical
WING review, the operator needs the report markdown to show why the decision
record remains blocked.

## Safety Boundary

This is an offline diagnostic/reporting change. It does not contact WING,
mutate console state, update readiness rows, or grant live authority.

## Expected Evidence

- Decision-review report JSON includes `invalid_records`.
- Decision-review markdown renders invalid record details.
- A stale pending `template_status` is visible with
  `review_decision_template_status_unedited`.

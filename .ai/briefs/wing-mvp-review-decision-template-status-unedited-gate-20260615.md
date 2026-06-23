# Brief: WING MVP Review Decision Template-Status Unedited Gate

## Task

Make the WING read-only review-decision ledger fail closed when an accepted
operator decision still carries the generated pending template status marker.

## Context

The review decision template includes
`template_status: operator_must_replace_placeholders_before_ledger_acceptance`.
The template is intentionally not an accepted decision. If an operator changes
`decision`, `reviewed_by`, and `reviewed_at` but leaves the stale template
status marker, the ledger should treat the record as unedited pending template
state instead of accepting it.

## Safety Boundary

This is an offline ledger validation change. It does not contact WING, mutate
console state, update readiness rows, or grant live authority.

## Expected Evidence

- A record carrying the pending template status gets
  `review_decision_template_status_unedited`.
- That record is invalid and cannot count as accepted.
- The runbook tells the operator not to leave the generated template marker in
  an accepted decision.

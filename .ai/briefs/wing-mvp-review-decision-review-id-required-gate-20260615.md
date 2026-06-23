# Brief: WING MVP Review Decision Review-ID Required Gate

## Task

Make the WING read-only review-decision ledger fail closed when an operator
decision record omits `review_id`.

## Context

The shared decision-template contract and runbook minimum accepted-decision
shape require `review_id`, but the ledger used an internal fallback identifier
when the field was missing. That could let a manually edited accepted decision
advance without the required operator-facing decision identity.

## Safety Boundary

This is an offline ledger validation change. It does not contact WING, mutate
console state, update readiness rows, or grant live authority.

## Expected Evidence

- Missing `review_id` produces `review_decision_review_id_missing`.
- A record missing `review_id` is invalid and cannot count as accepted.
- The runbook tells the operator not to remove `review_id`.

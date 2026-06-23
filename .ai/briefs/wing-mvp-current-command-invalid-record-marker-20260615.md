# Brief: WING MVP Current-Command Invalid-Record Marker

## Task

Make Project Director Current First Command coverage require the supervised
proof-plan invalid review-decision record field.

## Context

The supervised `next_wing_proof_plan` contract now includes
`readonly_preflight_review_decision_invalid_records`, and no-WING validation
checks that it matches the review-decision ledger. The Current First Command
still listed older required proof-plan summary fields without naming that new
invalid-record field, so Project Director could treat stale operator guidance
as baseline-complete.

## Safety Boundary

This is a direction-guard and documentation change. It does not contact WING,
mutate console state, approve readiness closure, or grant live authority.

## Expected Evidence

- Current First Command names
  `readonly_preflight_review_decision_invalid_records`.
- Project Director baseline coverage includes the marker.
- Project Director tests require the marker in the selected action, prompt, and
  README.

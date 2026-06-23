# Brief: WING MVP Handoff Invalid-Record Marker

## Task

Make the WING read-only handoff packet and no-WING handoff summary explicitly
show whether required supervised proof-plan fields include invalid review
decision record diagnostics.

## Context

The proof-plan contract now includes
`readonly_preflight_review_decision_invalid_records`. The handoff packet already
carried the full required field list, but operators had to inspect a long JSON
array to confirm the invalid-record field was present.

## Safety Boundary

This is an offline handoff/reporting gate. It does not contact WING, mutate
console state, approve readiness closure, or grant live authority.

## Expected Evidence

- Handoff packet JSON includes
  `required_supervised_proof_plan_summary_fields_include_invalid_records: true`.
- Handoff markdown renders that flag.
- no-WING validation rejects stale or missing flag values.
- Current First Command baseline coverage requires the flag.

# Brief: WING MVP Supervised Prestart Expected Marker List Summary Field

## Task

Expose the expected structured supervised prestart marker list in the no-WING
handoff summary.

## Context

The no-WING handoff summary already exposes the actual
`supervised_prestart_gate_required_markers`, marker counts, and match verdicts.
Operators can see whether the structured gate marker list matches the contract,
but the expected contract list itself was not available as a flat summary field
next to the actual list.

## Scope

- Add top-level no-WING field:
  `expected_supervised_prestart_gate_required_markers`.
- Render it in JSON and Markdown summaries.
- Add Project Director baseline marker and tests.

## Out Of Scope

- Contacting WING hardware.
- Granting live authority.
- Closing readiness from offline evidence.

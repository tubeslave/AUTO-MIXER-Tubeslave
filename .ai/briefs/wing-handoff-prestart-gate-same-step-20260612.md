# WING Handoff Prestart Gate Same-Step Validation

## Context

The no-WING release validator requires handoff packet operator sequences to
include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`. The current check searched the full operator
sequence text, so stale transition wording could pass if the prestart phrase was
present in a separate step.

## Task

Require all supervised prestart gate markers to appear in the same operator
sequence step and expose that exact step in release summary output.

## Constraints

- Offline validator/report only.
- No backend startup.
- No WING transport usage.
- No live authority change.

## Test Plan

- Focused handoff packet validator tests.
- Summary embedding regression.
- Affected no-WING/WING tests.
- Full pytest suite before commit if feasible.

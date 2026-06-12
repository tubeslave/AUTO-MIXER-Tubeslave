# WING Handoff Prestart Gate Validator

## Context

The read-only handoff packet generator now tells operators to move to
`WING_SUPERVISED_FADER_GAIN_PROOF` only after both `accepted_decision_ready` and
a valid accepted-candidate prestart trace. The no-WING release validator still
accepted handoff packets whose operator sequence omitted that prestart gate.

## Task

Make no-WING release validation fail closed when a WING read-only handoff packet
does not carry the supervised proof transition marker requiring a valid
accepted-candidate prestart trace.

## Constraints

- Validator/test/report only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- Keep packet compatibility apart from stricter validation.

## Test Plan

- Focused handoff packet validator tests.
- Affected no-WING/WING proof tests.
- `git diff --check`.
- Full pytest suite before commit if feasible.

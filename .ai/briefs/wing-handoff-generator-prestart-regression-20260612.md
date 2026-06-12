# WING Handoff Generator Prestart Regression

## Context

The no-WING release validator rejects stale handoff packets without the
supervised prestart gate, but the handoff packet generator's own unit test only
checked the beginning of the operator sequence. A generator regression could be
caught later by release validation, but the generator contract should fail
locally too.

## Task

Add direct regression coverage that the generated handoff packet and rendered
Markdown include `accepted_decision_ready`, valid accepted-candidate prestart
trace, and `WING_SUPERVISED_FADER_GAIN_PROOF`.

## Constraints

- Test/report trace only.
- No backend startup.
- No WING transport usage.
- No live authority change.

## Test Plan

- Focused handoff packet tests.
- Affected handoff/no-WING tests.
- `git diff --check`.
- Full pytest suite before commit if feasible.

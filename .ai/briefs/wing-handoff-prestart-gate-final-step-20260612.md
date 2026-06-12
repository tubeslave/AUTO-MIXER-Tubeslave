# WING Handoff Prestart Gate Final-Step Validation

## Context

The handoff packet validator now requires all supervised proof transition
markers to appear in the same operator sequence step. A packet could still place
the correct gated transition earlier and then append a weaker later instruction
to move to `WING_SUPERVISED_FADER_GAIN_PROOF`.

## Task

Require the accepted-candidate prestart gate step to be the final operator
sequence item and expose its index/finality in release summaries.

## Constraints

- Offline validator/report only.
- No backend startup.
- No WING transport usage.
- No live authority change.

## Test Plan

- Focused handoff packet validator tests.
- Affected no-WING/WING tests.
- `git diff --check`.
- Full pytest suite before commit if feasible.

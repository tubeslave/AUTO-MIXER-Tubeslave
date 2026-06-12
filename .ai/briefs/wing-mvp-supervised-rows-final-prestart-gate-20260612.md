# WING MVP Supervised Rows Final Prestart Gate

## Context

The first WING blocker and no-WING handoff validator now require a final
operator-sequence step with valid accepted-candidate prestart trace before
supervised proof. The downstream supervised fader/gain blocker rows still
described their next action as clean review plus accepted decision JSON only.

## Task

Align the supervised fader and gain-down MVP blocker rows with the final handoff
prestart gate before `WING_SUPERVISED_FADER_GAIN_PROOF`.

## Constraints

- Status source/report/test only.
- Do not close any blocker.
- No backend startup.
- No WING transport usage.
- No live authority change.

## Test Plan

- Focused MVP rails tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Affected Project Director/MVP tests.
- Full pytest suite before commit if feasible.

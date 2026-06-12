# WING Supervised Runbook Prestart Trace Gate

## Context

The MVP blocker table and progress report now expose the accepted-candidate
prestart trace gate. The supervised fader/gain proof runbook still described
the handoff mostly as clean preflight plus accepted decision, and the offline
handoff packet sequence still said to move to supervised proof after
`accepted_decision_ready`.

## Task

Make the operator-facing supervised proof runbook and read-only handoff packet
state that a valid accepted-candidate prestart trace is required before
`WING_SUPERVISED_FADER_GAIN_PROOF`.

## Constraints

- Documentation and handoff guidance only.
- No WING transport usage.
- No backend startup.
- No new live write kinds.
- Keep supervised proof limited to fader and negative gain/trim deltas.

## Test Plan

- Focused runbook/handoff tests.
- Affected no-WING and MVP rails tests.
- `git diff --check`.
- Full pytest suite before commit if feasible.

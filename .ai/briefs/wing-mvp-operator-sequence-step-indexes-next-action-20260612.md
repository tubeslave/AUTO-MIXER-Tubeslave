# Brief: WING MVP Operator Sequence Step Indexes Next Action

## Context

The first open WING MVP blocker is still `WING read-only connection proof`.
Existing handoff/runbook rails require the backend prestart gate, blocker-anchor
digest review step, and final supervised prestart gate before any supervised
fader/gain proof.

## Task

Make the no-WING handoff review contract explicitly name the operator sequence
indexes for the digest review and final supervised gate steps:

- `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8`
- `operator_sequence_supervised_prestart_gate_step_index: 9`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not widen live writes beyond the existing supervised fader/gain rail.
- Keep this as a docs/tests/autoloop contract change.

## Validation

- Focused Project Director and WING MVP rail tests.
- MVP progress regeneration.
- Direction guard and Project Director checks.
- Full pytest before commit.

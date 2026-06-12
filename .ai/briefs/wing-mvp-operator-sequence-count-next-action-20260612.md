# Brief: WING MVP Operator Sequence Count Next Action

## Context

The current first open WING MVP blocker remains `WING read-only connection
proof`. The no-WING handoff validator already checks that the operator sequence
contains exactly ten steps before supervised fader/gain proof planning.

## Task

Make the selected action and operator-facing WING rails explicitly require:

- `operator_sequence_count: 10`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not widen live writes.
- Keep the change to WING read-only/supervised handoff docs and tests.

## Validation

- Regenerate MVP progress from the blocker table.
- Run focused WING rail tests.
- Run direction guard, Project Director, diff check, memory lint, and full
  pytest.

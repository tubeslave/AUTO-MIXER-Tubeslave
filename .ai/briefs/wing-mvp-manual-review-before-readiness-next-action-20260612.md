# Brief: WING MVP Manual Review Before Readiness Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff packet already carries
`manual_review_required_before_readiness_update: true`.

## Task

Make the selected action and operator-facing rails explicitly require:

- `manual_review_required_before_readiness_update: true`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not stop or reconfigure local listener processes.
- Do not close readiness from offline/no-WING evidence.
- Keep this as a docs/tests/autoloop contract change around existing no-WING
  handoff evidence.

## Validation

- Regenerate MVP progress.
- Run focused WING rail tests.
- Run direction guard, Project Director, diff check, memory lint, and full
  pytest.

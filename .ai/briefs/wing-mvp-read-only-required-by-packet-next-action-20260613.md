# Brief: WING MVP Read Only Required By Packet Next Action

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff packet already carries `read_only: true`.

## Task

Make the selected action and operator-facing rails explicitly require:

- `read_only: true`

## Boundaries

- Do not contact WING.
- Do not grant live authority.
- Do not treat a no-WING handoff packet as anything other than read-only proof
  preparation.
- Do not stop or reconfigure local listener processes.
- Keep this as a docs/tests/autoloop contract change around existing no-WING
  handoff evidence.

## Validation

- Regenerate MVP progress.
- Run focused WING rail tests.
- Run direction guard, Project Director, diff check, memory lint, and full
  pytest.
